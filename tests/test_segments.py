from __future__ import annotations

import json

import pytest

from agent_reliability import SegmentPolicy, audit_segments
from agent_reliability.models import (
    CaseResult,
    EvaluationReport,
    FailureType,
    ScoreBreakdown,
)
from agent_reliability.segments import SegmentFindingCode


def _result(
    case_id: str,
    score: float,
    *,
    passed: bool | None = None,
    failures: tuple[FailureType, ...] = (),
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        score=score,
        passed=score >= 0.8 if passed is None else passed,
        breakdown=ScoreBreakdown(1, 1, 1, 1, 1, 1, 1),
        failures=failures,
        diagnostics=(),
    )


def _report(*results: CaseResult) -> EvaluationReport:
    return EvaluationReport(results=results, pass_threshold=0.8)


def test_small_critical_slice_blocks_healthy_aggregate() -> None:
    report = _report(
        *[_result(f"routine-{index}", 1.0) for index in range(9)],
        _result("refund-write", 0.7, passed=False),
    )
    assignments = {
        **{f"routine-{index}": {"routine"} for index in range(9)},
        "refund-write": {"high-risk"},
    }

    audit = audit_segments(
        report,
        assignments,
        {
            "routine": SegmentPolicy(min_cases=5, min_pass_rate=0.9),
            "high-risk": SegmentPolicy(min_cases=1, min_pass_rate=1.0),
        },
    )

    assert report.pass_rate == 0.9
    assert audit.passed is False
    critical = next(item for item in audit.segments if item.name == "high-risk")
    assert critical.failed_case_ids == ("refund-write",)
    assert critical.findings[0].code is SegmentFindingCode.PASS_RATE_BELOW_MINIMUM


def test_cases_can_belong_to_overlapping_segments() -> None:
    report = _report(
        _result("read", 1.0),
        _result("write", 0.9),
    )

    audit = audit_segments(
        report,
        {"read": {"support"}, "write": {"support", "write-tools"}},
        {},
        default_policy=SegmentPolicy(min_mean_score=0.8),
    )

    assert audit.passed is True
    assert [(item.name, item.total_cases) for item in audit.segments] == [
        ("support", 2),
        ("write-tools", 1),
    ]


def test_explicit_policy_overrides_default() -> None:
    report = _report(_result("case", 0.9))

    audit = audit_segments(
        report,
        {"case": {"high-risk"}},
        {"high-risk": SegmentPolicy(min_mean_score=0.95)},
        default_policy=SegmentPolicy(min_mean_score=0.8),
    )

    assert audit.passed is False
    assert audit.segments[0].findings[0].code is SegmentFindingCode.MEAN_SCORE_BELOW_MINIMUM


def test_blocked_failure_rejects_even_when_case_score_passes() -> None:
    report = _report(
        _result(
            "unsafe-auto-complete",
            0.9,
            passed=True,
            failures=(FailureType.MISSING_ESCALATION,),
        )
    )

    audit = audit_segments(
        report,
        {"unsafe-auto-complete": {"high-risk"}},
        {
            "high-risk": SegmentPolicy(
                min_mean_score=0.8,
                blocked_failures=frozenset({"missing_escalation"}),
            )
        },
    )

    finding = audit.segments[0].findings[0]
    assert audit.passed is False
    assert finding.code is SegmentFindingCode.BLOCKED_FAILURE_PRESENT
    assert finding.case_ids == ("unsafe-auto-complete",)


def test_configured_segment_without_evidence_fails_closed() -> None:
    audit = audit_segments(
        _report(_result("routine", 1.0)),
        {"routine": {"routine"}},
        {
            "routine": SegmentPolicy(),
            "payments": SegmentPolicy(min_cases=2),
        },
    )

    payments = next(item for item in audit.segments if item.name == "payments")
    assert payments.passed is False
    assert [item.code for item in payments.findings] == [
        SegmentFindingCode.MISSING_REQUIRED_SEGMENT,
        SegmentFindingCode.INSUFFICIENT_CASES,
        SegmentFindingCode.PASS_RATE_BELOW_MINIMUM,
        SegmentFindingCode.MEAN_SCORE_BELOW_MINIMUM,
    ]


def test_failure_counts_and_json_evidence_are_deterministic() -> None:
    report = _report(
        _result("b", 0.7, failures=(FailureType.MISSING_TOOL,)),
        _result(
            "a",
            0.6,
            failures=(FailureType.MISSING_EVIDENCE, FailureType.MISSING_TOOL),
        ),
    )

    audit = audit_segments(
        report,
        {"a": {"retrieval"}, "b": {"retrieval"}},
        {"retrieval": SegmentPolicy(min_pass_rate=0, min_mean_score=0)},
    )
    values = audit.as_dict()

    assert values["segments"][0]["failure_counts"] == {
        "missing_evidence": 1,
        "missing_tool": 2,
    }
    assert values["segments"][0]["failed_case_ids"] == ["a", "b"]
    json.dumps(values, sort_keys=True, allow_nan=False)


def test_every_case_requires_an_assignment() -> None:
    with pytest.raises(ValueError, match="cases have no segment assignment: b"):
        audit_segments(
            _report(_result("a", 1), _result("b", 1)),
            {"a": {"routine"}},
            {"routine": SegmentPolicy()},
        )


def test_unknown_assignment_is_rejected() -> None:
    with pytest.raises(ValueError, match="assignments reference unknown case"):
        audit_segments(
            _report(_result("a", 1)),
            {"a": {"routine"}, "ghost": {"routine"}},
            {"routine": SegmentPolicy()},
        )


def test_observed_segment_requires_a_policy() -> None:
    with pytest.raises(ValueError, match="no policy configured.*unreviewed"):
        audit_segments(
            _report(_result("a", 1)),
            {"a": {"unreviewed"}},
            {},
        )


def test_default_policy_requires_a_validated_policy_object() -> None:
    with pytest.raises(TypeError, match="default_policy must be a SegmentPolicy"):
        audit_segments(
            _report(_result("a", 1)),
            {"a": {"routine"}},
            {},
            default_policy={"min_pass_rate": 1},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("name", ["HighRisk", " high-risk", "high risk", "", "a" * 65])
def test_invalid_segment_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="segment names must start"):
        audit_segments(
            _report(_result("a", 1)),
            {"a": {name}},
            {},
            default_policy=SegmentPolicy(),
        )


def test_string_assignment_is_not_treated_as_characters() -> None:
    with pytest.raises(TypeError, match="must be a collection, not a string"):
        audit_segments(
            _report(_result("a", 1)),
            {"a": "risk"},  # type: ignore[dict-item]
            {},
            default_policy=SegmentPolicy(),
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_cases": 0}, "min_cases must be at least one"),
        ({"min_pass_rate": -0.1}, "finite and between zero and one"),
        ({"min_mean_score": float("nan")}, "finite and between zero and one"),
        ({"blocked_failures": frozenset({"invented"})}, "unknown blocked failure type"),
    ],
)
def test_invalid_policy_fails_closed(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        SegmentPolicy(**kwargs)  # type: ignore[arg-type]


def test_duplicate_or_invalid_report_results_are_rejected() -> None:
    duplicate = _result("same", 1)
    with pytest.raises(ValueError, match="duplicate case ID"):
        audit_segments(
            _report(duplicate, duplicate),
            {"same": {"routine"}},
            {"routine": SegmentPolicy()},
        )

    with pytest.raises(ValueError, match="invalid score"):
        audit_segments(
            _report(_result("bad", float("nan"))),
            {"bad": {"routine"}},
            {"routine": SegmentPolicy()},
        )
