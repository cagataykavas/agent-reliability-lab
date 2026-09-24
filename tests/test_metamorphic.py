from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agent_reliability.metamorphic import (
    BehaviorDimension,
    FindingCode,
    MetamorphicPairSpec,
    MetamorphicPolicy,
    MetamorphicRelation,
    audit_metamorphic_behavior,
)
from agent_reliability.models import AgentTrace, ToolCall


def trace(
    case_id: str,
    *,
    arguments: dict[str, object] | None = None,
    succeeded: bool = True,
    evidence: frozenset[str] = frozenset({"ev-1"}),
    escalated: bool = False,
    error: str | None = None,
) -> AgentTrace:
    return AgentTrace(
        case_id=case_id,
        final_answer=f"wording for {case_id}",
        tool_calls=(ToolCall("retrieve", arguments or {"account": "acct-1"}, succeeded=succeeded),),
        evidence_ids=evidence,
        total_latency_ms=50.0,
        total_cost_usd=0.01,
        escalated=escalated,
        error=error,
    )


def invariant() -> MetamorphicPairSpec:
    return MetamorphicPairSpec(
        pair_id="whitespace-noise",
        baseline_case_id="base",
        transformed_case_id="variant",
        relation=MetamorphicRelation.INVARIANT,
    )


def responsive(
    required: frozenset[BehaviorDimension],
    allowed: frozenset[BehaviorDimension] | None = None,
) -> MetamorphicPairSpec:
    return MetamorphicPairSpec(
        pair_id="policy-change",
        baseline_case_id="base",
        transformed_case_id="variant",
        relation=MetamorphicRelation.RESPONSIVE,
        required_changes=required,
        allowed_changes=required if allowed is None else allowed,
    )


def finding_codes(report: object) -> list[FindingCode]:
    return [item.code for item in report.results[0].findings]  # type: ignore[attr-defined]


def test_invariant_ignores_answer_wording_latency_and_cost() -> None:
    baseline = trace("base")
    variant = replace(
        trace("variant"),
        final_answer="Semantically equivalent answer with new wording.",
        total_latency_ms=900.0,
        total_cost_usd=0.8,
    )

    report = audit_metamorphic_behavior([invariant()], [baseline, variant])

    assert report.passed
    assert report.results[0].changed_dimensions == ()
    assert report.to_dict()["summary"]["decision"] == "accept"


@pytest.mark.parametrize(
    ("variant", "dimension"),
    [
        (trace("variant", arguments={"account": "acct-2"}), BehaviorDimension.TOOL_ARGUMENTS),
        (
            trace("variant", evidence=frozenset({"ev-2"})),
            BehaviorDimension.EVIDENCE,
        ),
        (trace("variant", succeeded=False), BehaviorDimension.TOOL_OUTCOMES),
        (trace("variant", escalated=True), BehaviorDimension.ESCALATION),
        (trace("variant", error="timeout"), BehaviorDimension.ERROR_STATE),
    ],
)
def test_invariant_rejects_behavior_drift(
    variant: AgentTrace, dimension: BehaviorDimension
) -> None:
    report = audit_metamorphic_behavior([invariant()], [trace("base"), variant])

    assert not report.passed
    assert report.results[0].changed_dimensions == (dimension,)
    assert finding_codes(report) == [FindingCode.UNEXPECTED_BEHAVIOR_CHANGE]


def test_tool_sequence_change_is_not_double_counted_as_argument_or_outcome_drift() -> None:
    variant = replace(
        trace("variant"),
        tool_calls=(ToolCall("escalate", {"queue": "risk"}, succeeded=False),),
    )

    report = audit_metamorphic_behavior([invariant()], [trace("base"), variant])

    assert report.results[0].changed_dimensions == (BehaviorDimension.TOOL_SEQUENCE,)


def test_responsive_pair_accepts_required_allowlisted_change() -> None:
    spec = responsive(frozenset({BehaviorDimension.ESCALATION}))

    report = audit_metamorphic_behavior([spec], [trace("base"), trace("variant", escalated=True)])

    assert report.passed


def test_responsive_pair_rejects_missing_required_change() -> None:
    spec = responsive(frozenset({BehaviorDimension.ESCALATION}))

    report = audit_metamorphic_behavior([spec], [trace("base"), trace("variant")])

    assert finding_codes(report) == [FindingCode.REQUIRED_CHANGE_MISSING]


def test_responsive_pair_rejects_change_outside_allowlist() -> None:
    spec = responsive(frozenset({BehaviorDimension.ESCALATION}))

    report = audit_metamorphic_behavior(
        [spec],
        [
            trace("base"),
            trace("variant", escalated=True, evidence=frozenset({"ev-2"})),
        ],
    )

    result = report.results[0]
    assert not result.passed
    assert result.changed_dimensions == (
        BehaviorDimension.EVIDENCE,
        BehaviorDimension.ESCALATION,
    )
    assert finding_codes(report) == [FindingCode.UNEXPECTED_BEHAVIOR_CHANGE]
    assert result.findings[0].dimension is BehaviorDimension.EVIDENCE


def test_responsive_pair_can_require_multiple_changes() -> None:
    changes = frozenset({BehaviorDimension.ESCALATION, BehaviorDimension.ERROR_STATE})
    report = audit_metamorphic_behavior(
        [responsive(changes)],
        [trace("base"), trace("variant", escalated=True, error="requires review")],
    )

    assert report.passed


@pytest.mark.parametrize(
    ("available", "code"),
    [
        ([trace("variant")], FindingCode.MISSING_BASELINE_TRACE),
        ([trace("base")], FindingCode.MISSING_TRANSFORMED_TRACE),
    ],
)
def test_missing_trace_fails_closed(available: list[AgentTrace], code: FindingCode) -> None:
    report = audit_metamorphic_behavior([invariant()], available)

    assert not report.passed
    assert finding_codes(report) == [code]
    assert len(report.results[0].evidence_sha256) == 64


@pytest.mark.parametrize(
    "bad_arguments",
    [
        {1: "non-string-key"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": 2**53},
        {"value": (1, 2)},
        {"value": b"secret"},
    ],
)
def test_malformed_tool_arguments_fail_closed_without_leaking_values(
    bad_arguments: dict[object, object],
) -> None:
    malformed = trace("variant")
    malformed = replace(
        malformed,
        tool_calls=(ToolCall("retrieve", bad_arguments),),  # type: ignore[arg-type]
    )

    report = audit_metamorphic_behavior([invariant()], [trace("base"), malformed])
    encoded = json.dumps(report.to_dict(), sort_keys=True)

    assert finding_codes(report) == [FindingCode.MALFORMED_TRACE]
    assert "secret" not in encoded


def test_cyclic_arguments_fail_closed() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    malformed = replace(trace("variant"), tool_calls=(ToolCall("retrieve", cyclic),))

    report = audit_metamorphic_behavior([invariant()], [trace("base"), malformed])

    assert finding_codes(report) == [FindingCode.MALFORMED_TRACE]


def test_argument_and_trace_budgets_fail_closed() -> None:
    oversized = trace("variant", arguments={"payload": "x" * 32})
    policy = MetamorphicPolicy(max_argument_bytes=16, max_calls_per_trace=1)

    report = audit_metamorphic_behavior([invariant()], [trace("base"), oversized], policy)

    assert finding_codes(report) == [FindingCode.MALFORMED_TRACE]


def test_report_order_and_hash_are_deterministic() -> None:
    first = invariant()
    second = replace(
        invariant(), pair_id="a-first", baseline_case_id="base-2", transformed_case_id="variant-2"
    )
    traces = [trace("base"), trace("variant"), trace("base-2"), trace("variant-2")]

    report_a = audit_metamorphic_behavior([first, second], traces)
    report_b = audit_metamorphic_behavior([second, first], reversed(traces))

    assert report_a.to_dict() == report_b.to_dict()
    assert [item.pair_id for item in report_a.results] == ["a-first", "whitespace-noise"]
    json.dumps(report_a.to_dict(), allow_nan=False)


def test_duplicate_collection_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate pair_id"):
        audit_metamorphic_behavior([invariant(), invariant()], [trace("base"), trace("variant")])
    with pytest.raises(ValueError, match="duplicate trace case_id"):
        audit_metamorphic_behavior([invariant()], [trace("base"), trace("base")])


def test_empty_or_over_budget_pair_collections_are_rejected() -> None:
    with pytest.raises(ValueError, match="pair count"):
        audit_metamorphic_behavior([], [])
    with pytest.raises(ValueError, match="pair count"):
        audit_metamorphic_behavior(
            [invariant(), replace(invariant(), pair_id="second")],
            [trace("base"), trace("variant")],
            MetamorphicPolicy(max_pairs=1),
        )


def test_invalid_pair_contracts_are_rejected() -> None:
    with pytest.raises(ValueError, match="must differ"):
        replace(invariant(), transformed_case_id="base")
    with pytest.raises(ValueError, match="cannot declare"):
        replace(invariant(), allowed_changes=frozenset({BehaviorDimension.EVIDENCE}))
    with pytest.raises(ValueError, match="at least one"):
        replace(invariant(), relation=MetamorphicRelation.RESPONSIVE)
    with pytest.raises(ValueError, match="subset"):
        responsive(
            frozenset({BehaviorDimension.ESCALATION}),
            frozenset({BehaviorDimension.EVIDENCE}),
        )


def test_invalid_policy_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_pairs"):
        MetamorphicPolicy(min_pairs=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot exceed"):
        MetamorphicPolicy(min_pairs=2, max_pairs=1)
