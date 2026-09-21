from __future__ import annotations

import json

import pytest

from agent_reliability.lineage import (
    LineagePolicy,
    LineageTrace,
    audit_evidence_lineage,
    main,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def valid_payload() -> dict:
    return {
        "trace_id": "refund-review",
        "calls": [
            {
                "call_id": "order",
                "tool_name": "get_order",
                "depends_on": [],
                "succeeded": True,
                "evidence": [{"evidence_id": "order:O-77", "content_sha256": HASH_A}],
            },
            {
                "call_id": "policy",
                "tool_name": "get_policy",
                "depends_on": ["order"],
                "succeeded": True,
                "evidence": [{"evidence_id": "policy:refund:TR", "content_sha256": HASH_B}],
            },
        ],
        "cited_evidence_ids": ["order:O-77", "policy:refund:TR"],
    }


def finding_codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


def test_valid_lineage_passes_with_deterministic_sources() -> None:
    report = audit_evidence_lineage(
        LineageTrace.from_dict(valid_payload()),
        LineagePolicy(
            required_evidence_ids=frozenset({"order:O-77", "policy:refund:TR"}),
            require_all_produced_evidence_cited=True,
        ),
    )

    assert report.passed is True
    assert report.max_observed_dependency_depth == 1
    assert report.evidence_sources == {
        "order:O-77": "order",
        "policy:refund:TR": "policy",
    }
    assert report.to_dict()["findings"] == []


def test_non_causal_unknown_and_failed_dependencies_are_reported() -> None:
    payload = valid_payload()
    payload["calls"] = [
        {
            "call_id": "policy",
            "tool_name": "get_policy",
            "depends_on": ["order", "missing"],
            "succeeded": True,
            "evidence": [],
        },
        {
            "call_id": "order",
            "tool_name": "get_order",
            "depends_on": [],
            "succeeded": False,
            "evidence": [],
        },
        {
            "call_id": "decision",
            "tool_name": "decide",
            "depends_on": ["order"],
            "succeeded": True,
            "evidence": [],
        },
    ]
    payload["cited_evidence_ids"] = []

    report = audit_evidence_lineage(LineageTrace.from_dict(payload))

    assert finding_codes(report) == [
        "non_causal_dependency",
        "unknown_dependency",
        "failed_dependency",
    ]


def test_evidence_integrity_and_citation_failures_are_aggregated() -> None:
    payload = valid_payload()
    payload["calls"][0]["succeeded"] = False
    payload["calls"][0]["evidence"][0]["content_sha256"] = "NOT-A-HASH"
    payload["calls"][1]["evidence"][0]["evidence_id"] = "order:O-77"
    payload["cited_evidence_ids"] = ["ghost", "ghost"]

    report = audit_evidence_lineage(
        LineageTrace.from_dict(payload),
        LineagePolicy(
            required_evidence_ids=frozenset({"required"}),
            require_all_produced_evidence_cited=True,
        ),
    )

    assert finding_codes(report) == [
        "invalid_content_hash",
        "evidence_from_failed_call",
        "failed_dependency",
        "duplicate_evidence_id",
        "duplicate_citation",
        "unknown_citation",
        "required_evidence_not_cited",
        "uncited_evidence",
    ]


def test_duplicate_call_dependency_and_depth_budgets_fail() -> None:
    payload = valid_payload()
    payload["calls"].append(
        {
            "call_id": "policy",
            "tool_name": "approve",
            "depends_on": ["order", "order"],
            "succeeded": True,
            "evidence": [],
        }
    )

    report = audit_evidence_lineage(
        LineageTrace.from_dict(payload),
        LineagePolicy(max_calls=2, max_dependency_depth=0),
    )

    assert finding_codes(report) == [
        "call_budget_exceeded",
        "duplicate_call_id",
        "dependency_depth_exceeded",
        "duplicate_dependency",
        "dependency_depth_exceeded",
    ]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda payload: payload.update(trace_id=""), "trace_id must be a non-empty string"),
        (lambda payload: payload.update(calls="not-an-array"), "calls must be an array"),
        (
            lambda payload: payload["calls"][0].update(succeeded="true"),
            "succeeded must be a boolean",
        ),
        (
            lambda payload: payload["calls"][0].update(depends_on=[""]),
            "dependency call ID must be a non-empty string",
        ),
    ],
)
def test_malformed_artifacts_fail_closed(mutation, error) -> None:
    payload = valid_payload()
    mutation(payload)

    with pytest.raises((TypeError, ValueError), match=error):
        LineageTrace.from_dict(payload)


@pytest.mark.parametrize(
    "values",
    [
        {"max_calls": 0},
        {"max_calls": True},
        {"max_dependency_depth": -1},
        {"max_dependency_depth": 1.5},
        {"require_all_produced_evidence_cited": 1},
        {"required_evidence_ids": ["evidence"]},
    ],
)
def test_invalid_policies_fail_closed(values) -> None:
    with pytest.raises((TypeError, ValueError)):
        LineagePolicy(**values)


def test_cli_writes_json_and_returns_policy_exit_code(tmp_path, capsys) -> None:
    payload = valid_payload()
    payload["cited_evidence_ids"] = ["ghost"]
    trace_path = tmp_path / "trace.json"
    output_path = tmp_path / "report.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main([str(trace_path), "--output", str(output_path)])
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["passed"] is False
    assert report["findings"][0]["code"] == "unknown_citation"
    assert json.loads(capsys.readouterr().out) == report


def test_cli_returns_two_for_malformed_json(tmp_path, capsys) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{", encoding="utf-8")

    assert main([str(trace_path)]) == 2
    assert json.loads(capsys.readouterr().out)["passed"] is False
