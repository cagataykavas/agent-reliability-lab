from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FailureType(StrEnum):
    MISSING_TOOL = "missing_tool"
    UNEXPECTED_TOOL = "unexpected_tool"
    WRONG_ARGUMENTS = "wrong_arguments"
    WRONG_TOOL_ORDER = "wrong_tool_order"
    MISSING_EVIDENCE = "missing_evidence"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNHANDLED_ERROR = "unhandled_error"
    UNSAFE_COMPLETION = "unsafe_completion"
    UNNECESSARY_ESCALATION = "unnecessary_escalation"
    MISSING_ESCALATION = "missing_escalation"


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    succeeded: bool = True
    latency_ms: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ToolCall:
        return cls(
            name=str(values["name"]),
            arguments=dict(values.get("arguments", {})),
            succeeded=bool(values.get("succeeded", True)),
            latency_ms=float(values.get("latency_ms", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    prompt: str
    expected_tools: tuple[ToolCall, ...] = ()
    required_evidence: frozenset[str] = frozenset()
    allowed_tools: frozenset[str] = frozenset()
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None
    should_escalate: bool = False
    reference_answer_contains: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> EvaluationCase:
        expected = tuple(ToolCall.from_dict(item) for item in values.get("expected_tools", []))
        allowed = values.get("allowed_tools") or [item.name for item in expected]
        return cls(
            case_id=str(values["case_id"]),
            prompt=str(values["prompt"]),
            expected_tools=expected,
            required_evidence=frozenset(map(str, values.get("required_evidence", []))),
            allowed_tools=frozenset(map(str, allowed)),
            max_latency_ms=_optional_float(values.get("max_latency_ms")),
            max_cost_usd=_optional_float(values.get("max_cost_usd")),
            should_escalate=bool(values.get("should_escalate", False)),
            reference_answer_contains=tuple(map(str, values.get("reference_answer_contains", []))),
        )


@dataclass(frozen=True, slots=True)
class AgentTrace:
    case_id: str
    final_answer: str
    tool_calls: tuple[ToolCall, ...] = ()
    evidence_ids: frozenset[str] = frozenset()
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    escalated: bool = False
    error: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AgentTrace:
        return cls(
            case_id=str(values["case_id"]),
            final_answer=str(values.get("final_answer", "")),
            tool_calls=tuple(ToolCall.from_dict(item) for item in values.get("tool_calls", [])),
            evidence_ids=frozenset(map(str, values.get("evidence_ids", []))),
            total_latency_ms=float(values.get("total_latency_ms", 0.0)),
            total_cost_usd=float(values.get("total_cost_usd", 0.0)),
            escalated=bool(values.get("escalated", False)),
            error=values.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "tool_calls": [asdict(item) for item in self.tool_calls],
            "evidence_ids": sorted(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    tool_selection: float
    argument_correctness: float
    tool_order: float
    evidence_coverage: float
    answer_coverage: float
    budget_compliance: float
    escalation_correctness: float


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    score: float
    passed: bool
    breakdown: ScoreBreakdown
    failures: tuple[FailureType, ...]
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["failures"] = [str(item) for item in self.failures]
        return values


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    results: tuple[CaseResult, ...]
    pass_threshold: float

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(item.passed for item in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases else 0.0

    @property
    def mean_score(self) -> float:
        return sum(item.score for item in self.results) / self.total_cases if self.total_cases else 0.0

    def to_dict(self) -> dict[str, Any]:
        failure_counts: dict[str, int] = {}
        for result in self.results:
            for failure in result.failures:
                failure_counts[str(failure)] = failure_counts.get(str(failure), 0) + 1
        return {
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "pass_rate": round(self.pass_rate, 6),
                "mean_score": round(self.mean_score, 6),
                "pass_threshold": self.pass_threshold,
                "failure_counts": failure_counts,
            },
            "results": [item.to_dict() for item in self.results],
        }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
