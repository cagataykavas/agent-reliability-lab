from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from .models import (
    AgentTrace,
    CaseResult,
    EvaluationCase,
    EvaluationReport,
    FailureType,
    ScoreBreakdown,
    ToolCall,
)

WEIGHTS = {
    "tool_selection": 0.24,
    "argument_correctness": 0.20,
    "tool_order": 0.10,
    "evidence_coverage": 0.16,
    "answer_coverage": 0.10,
    "budget_compliance": 0.10,
    "escalation_correctness": 0.10,
}


def evaluate_trace(
    case: EvaluationCase,
    trace: AgentTrace,
    *,
    pass_threshold: float = 0.8,
) -> CaseResult:
    if case.case_id != trace.case_id:
        raise ValueError(f"case/trace ID mismatch: {case.case_id!r} != {trace.case_id!r}")
    if not 0.0 <= pass_threshold <= 1.0:
        raise ValueError("pass_threshold must be between 0 and 1")

    failures: list[FailureType] = []
    diagnostics: list[str] = []
    expected_names = [item.name for item in case.expected_tools]
    actual_names = [item.name for item in trace.tool_calls]

    tool_selection = _multiset_f1(expected_names, actual_names)
    missing = Counter(expected_names) - Counter(actual_names)
    unexpected = Counter(actual_names) - Counter(expected_names)
    if missing:
        failures.append(FailureType.MISSING_TOOL)
        diagnostics.append(f"Missing expected tools: {_format_counter(missing)}")
    disallowed = [name for name in actual_names if case.allowed_tools and name not in case.allowed_tools]
    if unexpected or disallowed:
        failures.append(FailureType.UNEXPECTED_TOOL)
        names = sorted(set(unexpected) | set(disallowed))
        diagnostics.append(f"Unexpected or disallowed tools: {', '.join(names)}")

    argument_correctness = _argument_score(case.expected_tools, trace.tool_calls)
    if argument_correctness < 1.0 and case.expected_tools:
        failures.append(FailureType.WRONG_ARGUMENTS)
        diagnostics.append("One or more matched tool calls used incomplete or incorrect arguments")

    tool_order = _order_score(expected_names, actual_names)
    if tool_order < 1.0 and len(expected_names) > 1:
        failures.append(FailureType.WRONG_TOOL_ORDER)
        diagnostics.append("Expected tool-call order was not preserved")

    evidence_coverage = _set_recall(case.required_evidence, trace.evidence_ids)
    missing_evidence = case.required_evidence - trace.evidence_ids
    if missing_evidence:
        failures.append(FailureType.MISSING_EVIDENCE)
        diagnostics.append(f"Missing evidence IDs: {', '.join(sorted(missing_evidence))}")

    answer_coverage = _answer_coverage(case.reference_answer_contains, trace.final_answer)
    if answer_coverage < 1.0:
        failures.append(FailureType.UNSAFE_COMPLETION)
        diagnostics.append("Final answer omitted one or more required reference facts")

    budget_compliance = 1.0
    budget_messages: list[str] = []
    if case.max_latency_ms is not None and trace.total_latency_ms > case.max_latency_ms:
        budget_compliance = 0.0
        budget_messages.append(
            f"latency {trace.total_latency_ms:.1f}ms > {case.max_latency_ms:.1f}ms"
        )
    if case.max_cost_usd is not None and trace.total_cost_usd > case.max_cost_usd:
        budget_compliance = 0.0
        budget_messages.append(f"cost ${trace.total_cost_usd:.4f} > ${case.max_cost_usd:.4f}")
    if budget_messages:
        failures.append(FailureType.BUDGET_EXCEEDED)
        diagnostics.append("Budget exceeded: " + "; ".join(budget_messages))

    escalation_correctness = float(trace.escalated == case.should_escalate)
    if trace.escalated and not case.should_escalate:
        failures.append(FailureType.UNNECESSARY_ESCALATION)
        diagnostics.append("Trace escalated a case that should be handled automatically")
    elif case.should_escalate and not trace.escalated:
        failures.append(FailureType.MISSING_ESCALATION)
        diagnostics.append("Trace failed to escalate a review-required case")

    if trace.error:
        failures.append(FailureType.UNHANDLED_ERROR)
        diagnostics.append(f"Unhandled trace error: {trace.error}")

    breakdown = ScoreBreakdown(
        tool_selection=tool_selection,
        argument_correctness=argument_correctness,
        tool_order=tool_order,
        evidence_coverage=evidence_coverage,
        answer_coverage=answer_coverage,
        budget_compliance=budget_compliance,
        escalation_correctness=escalation_correctness,
    )
    score = sum(getattr(breakdown, name) * weight for name, weight in WEIGHTS.items())
    if trace.error:
        score *= 0.5
    score = round(score, 6)
    return CaseResult(
        case_id=case.case_id,
        score=score,
        passed=score >= pass_threshold and not trace.error,
        breakdown=breakdown,
        failures=tuple(dict.fromkeys(failures)),
        diagnostics=tuple(diagnostics),
    )


def evaluate_suite(
    cases: Iterable[EvaluationCase],
    traces: Iterable[AgentTrace],
    *,
    pass_threshold: float = 0.8,
) -> EvaluationReport:
    case_map = _unique_by_id(cases, "case")
    trace_map = _unique_by_id(traces, "trace")
    unknown = trace_map.keys() - case_map.keys()
    if unknown:
        raise ValueError(f"traces reference unknown cases: {', '.join(sorted(unknown))}")

    results: list[CaseResult] = []
    for case_id, case in case_map.items():
        trace = trace_map.get(case_id)
        if trace is None:
            trace = AgentTrace(case_id=case_id, final_answer="", error="trace missing")
        results.append(evaluate_trace(case, trace, pass_threshold=pass_threshold))
    return EvaluationReport(results=tuple(results), pass_threshold=pass_threshold)


def _unique_by_id(values: Iterable[Any], label: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in values:
        if value.case_id in output:
            raise ValueError(f"duplicate {label} ID: {value.case_id}")
        output[value.case_id] = value
    return output


def _multiset_f1(expected: list[str], actual: list[str]) -> float:
    if not expected and not actual:
        return 1.0
    overlap = sum((Counter(expected) & Counter(actual)).values())
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else float(not actual)
    return _f1(precision, recall)


def _argument_score(expected: tuple[ToolCall, ...], actual: tuple[ToolCall, ...]) -> float:
    if not expected:
        return float(not actual)
    remaining = list(actual)
    scores: list[float] = []
    for wanted in expected:
        match_index = next((i for i, item in enumerate(remaining) if item.name == wanted.name), None)
        if match_index is None:
            scores.append(0.0)
            continue
        observed = remaining.pop(match_index)
        scores.append(_mapping_similarity(wanted.arguments, observed.arguments))
    return sum(scores) / len(scores)


def _mapping_similarity(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    correct = sum(key in actual and actual[key] == value for key, value in expected.items())
    return correct / len(expected)


def _order_score(expected: list[str], actual: list[str]) -> float:
    if len(expected) < 2:
        return 1.0
    filtered = [name for name in actual if name in set(expected)]
    positions = []
    cursor = 0
    for name in expected:
        try:
            index = filtered.index(name, cursor)
        except ValueError:
            return 0.0
        positions.append(index)
        cursor = index + 1
    return float(positions == sorted(positions))


def _set_recall(expected: frozenset[str], actual: frozenset[str]) -> float:
    return len(expected & actual) / len(expected) if expected else 1.0


def _answer_coverage(required: tuple[str, ...], answer: str) -> float:
    if not required:
        return 1.0
    normalized = answer.casefold()
    return sum(item.casefold() in normalized for item in required) / len(required)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _format_counter(values: Counter[str]) -> str:
    return ", ".join(f"{name}×{count}" if count > 1 else name for name, count in sorted(values.items()))
