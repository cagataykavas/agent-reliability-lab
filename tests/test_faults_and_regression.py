import pytest

from agent_reliability.faults import FaultType, inject_fault, inject_suite
from agent_reliability.models import (
    AgentTrace,
    CaseResult,
    EvaluationCase,
    EvaluationReport,
    ScoreBreakdown,
    ToolCall,
)
from agent_reliability.regression import RegressionPolicy, compare_reports
from agent_reliability.scoring import evaluate_suite


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="case",
        prompt="lookup",
        expected_tools=(ToolCall("lookup", {"id": 7}),),
        required_evidence=frozenset({"record:7"}),
        allowed_tools=frozenset({"lookup"}),
        max_latency_ms=500,
        max_cost_usd=0.01,
    )


def _trace() -> AgentTrace:
    return AgentTrace(
        case_id="case",
        final_answer="done",
        tool_calls=(ToolCall("lookup", {"id": 7}),),
        evidence_ids=frozenset({"record:7"}),
        total_latency_ms=100,
        total_cost_usd=0.001,
    )


def _report(scores: dict[str, float]) -> EvaluationReport:
    breakdown = ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    return EvaluationReport(
        results=tuple(
            CaseResult(
                case_id=case_id,
                score=score,
                passed=score >= 0.8,
                breakdown=breakdown,
                failures=(),
                diagnostics=(),
            )
            for case_id, score in sorted(scores.items())
        ),
        pass_threshold=0.8,
    )


@pytest.mark.parametrize(
    ("fault", "changed_field"),
    [
        (FaultType.DROP_EVIDENCE, "evidence_ids"),
        (FaultType.CORRUPT_ARGUMENT, "tool_calls"),
        (FaultType.INSERT_UNEXPECTED_TOOL, "tool_calls"),
        (FaultType.EXCEED_LATENCY, "total_latency_ms"),
        (FaultType.EXCEED_COST, "total_cost_usd"),
        (FaultType.FLIP_ESCALATION, "escalated"),
        (FaultType.UNHANDLED_ERROR, "error"),
    ],
)
def test_each_fault_changes_the_trace(fault: FaultType, changed_field: str) -> None:
    source = _trace()
    injected = inject_fault(source, fault, magnitude=1000)
    assert injected != source
    assert getattr(injected, changed_field) != getattr(source, changed_field)
    assert source == _trace(), "source trace must stay immutable"


def test_suite_injection_is_deterministic_and_selective() -> None:
    traces = [_trace(), AgentTrace(case_id="second", final_answer="ok")]
    first = inject_suite(traces, FaultType.UNHANDLED_ERROR, every=2)
    second = inject_suite(traces, FaultType.UNHANDLED_ERROR, every=2)
    assert first == second
    assert first[0].error
    assert first[1] == traces[1]


def test_regression_gate_rejects_worse_candidate() -> None:
    case = _case()
    baseline = evaluate_suite([case], [_trace()])
    faulty = inject_fault(_trace(), FaultType.DROP_EVIDENCE)
    candidate = evaluate_suite([case], [faulty])
    comparison = compare_reports(baseline, candidate)
    assert not comparison.passed
    assert comparison.regressed_cases == ("case",)
    assert comparison.mean_score_delta < 0
    assert comparison.reasons


def test_regression_policy_can_allow_small_known_drop() -> None:
    case = _case()
    baseline = evaluate_suite([case], [_trace()])
    faulty = inject_fault(_trace(), FaultType.DROP_EVIDENCE)
    candidate = evaluate_suite([case], [faulty])
    comparison = compare_reports(
        baseline,
        candidate,
        policy=RegressionPolicy(
            max_mean_score_drop=0.2,
            max_pass_rate_drop=1.0,
            max_case_regressions=1,
        ),
    )
    assert comparison.passed


def test_missing_baseline_case_always_fails_gate() -> None:
    baseline = evaluate_suite([_case()], [_trace()])
    candidate = evaluate_suite([], [])
    comparison = compare_reports(baseline, candidate)
    assert not comparison.passed
    assert comparison.missing_cases == ("case",)


def test_paired_evidence_reports_exact_directional_significance() -> None:
    baseline = _report({f"case-{index}": 0.9 for index in range(6)})
    candidate = _report({f"case-{index}": 0.7 for index in range(6)})
    comparison = compare_reports(
        baseline,
        candidate,
        policy=RegressionPolicy(
            max_mean_score_drop=1.0,
            max_pass_rate_drop=1.0,
            max_case_regressions=6,
            directional_significance_level=0.05,
            min_directional_changes=5,
        ),
    )

    evidence = comparison.paired_evidence
    assert evidence.directional_changes == 6
    assert evidence.regressed_cases == 6
    assert evidence.improved_cases == 0
    assert evidence.regression_share == 1.0
    assert evidence.one_sided_p_value == 0.015625
    assert evidence.regression_share_interval is not None
    assert evidence.significant_regression is True
    assert comparison.passed is False
    assert "paired case directions" in comparison.reasons[0]


def test_paired_evidence_does_not_overstate_balanced_or_tied_changes() -> None:
    baseline = _report({f"case-{index}": 0.8 for index in range(7)})
    candidate = _report(
        {
            "case-0": 0.7,
            "case-1": 0.7,
            "case-2": 0.7,
            "case-3": 0.9,
            "case-4": 0.9,
            "case-5": 0.9,
            "case-6": 0.8,
        }
    )
    comparison = compare_reports(
        baseline,
        candidate,
        policy=RegressionPolicy(
            max_mean_score_drop=1.0,
            max_pass_rate_drop=1.0,
            max_case_regressions=7,
            directional_significance_level=0.05,
        ),
    )

    evidence = comparison.paired_evidence
    assert evidence.directional_changes == 6
    assert evidence.tied_cases == 1
    assert evidence.regression_share == 0.5
    assert evidence.one_sided_p_value == 0.65625
    assert evidence.significant_regression is False
    assert comparison.passed is True


def test_paired_evidence_marks_an_unchanged_suite_as_insufficient() -> None:
    report = _report({"a": 0.9, "b": 0.8})
    evidence = compare_reports(report, report).paired_evidence

    assert evidence.directional_changes == 0
    assert evidence.regression_share is None
    assert evidence.regression_share_interval is None
    assert evidence.one_sided_p_value == 1.0
    assert evidence.significant_regression is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_mean_score_drop": -0.1},
        {"max_pass_rate_drop": 1.1},
        {"max_case_regressions": -1},
        {"case_score_tolerance": -1},
        {"directional_significance_level": 0},
        {"directional_significance_level": 1},
        {"directional_confidence_level": 0},
        {"min_directional_changes": 0},
    ],
)
def test_regression_policy_rejects_invalid_statistical_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        RegressionPolicy(**kwargs)
