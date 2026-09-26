from __future__ import annotations

import json
from copy import deepcopy

import pytest

from agent_reliability.delegation import (
    DelegationPolicy,
    audit_delegation,
    load_artifact,
    main,
)


def agent(
    agent_id: str,
    parent: str | None,
    delegation: str | None,
    *,
    started: str = "2026-09-26T08:00:00Z",
    ended: str | None = "2026-09-26T08:01:00Z",
    status: str = "succeeded",
    prompt: int = 100,
    completion: int = 50,
    tools: int = 1,
):
    return {
        "agent_id": agent_id,
        "parent_agent_id": parent,
        "delegation_id": delegation,
        "started_at": started,
        "ended_at": ended,
        "status": status,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "tool_calls": tools,
    }


def artifact():
    return {
        "schema_version": "agent-delegation/v1",
        "run_id": "run-sensitive-123",
        "policy_id": "policy-v1",
        "agents": [
            agent("root-sensitive", None, None, ended="2026-09-26T08:03:00Z"),
            agent(
                "research-sensitive",
                "root-sensitive",
                "delegate-1",
                started="2026-09-26T08:00:10Z",
            ),
            agent(
                "review-sensitive",
                "root-sensitive",
                "delegate-2",
                started="2026-09-26T08:00:20Z",
            ),
        ],
    }


def codes(report):
    return {finding.code for finding in report.findings}


def test_valid_graph_is_accepted_with_content_free_deterministic_evidence():
    report = audit_delegation(artifact())
    repeated = audit_delegation(deepcopy(artifact()))

    assert report.accepted is True
    assert report.as_dict() == repeated.as_dict()
    assert report.metrics == {
        "agent_count": 3,
        "root_count": 1,
        "max_depth": 1,
        "max_fanout": 2,
        "total_prompt_tokens": 300,
        "total_completion_tokens": 150,
        "total_tokens": 450,
        "total_tool_calls": 3,
        "run_seconds": 180.0,
    }
    encoded = json.dumps(report.as_dict())
    assert "sensitive" not in encoded
    assert len(report.artifact_sha256) == 64


def test_canonical_artifact_digest_ignores_object_key_order():
    value = artifact()
    reordered = {key: value[key] for key in reversed(value)}
    reordered["agents"] = [{key: item[key] for key in reversed(item)} for item in value["agents"]]

    assert audit_delegation(value).artifact_sha256 == audit_delegation(reordered).artifact_sha256


def test_policy_digest_binds_thresholds_and_policy_identity():
    baseline = audit_delegation(artifact(), DelegationPolicy(max_agents=64))
    changed_threshold = audit_delegation(artifact(), DelegationPolicy(max_agents=63))
    changed_identity = artifact()
    changed_identity["policy_id"] = "policy-v2"

    assert baseline.policy_sha256 != changed_threshold.policy_sha256
    assert baseline.policy_sha256 != audit_delegation(changed_identity).policy_sha256


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda value: value["agents"].append(deepcopy(value["agents"][1])), "DUPLICATE_AGENT_ID"),
        (lambda value: value["agents"][1].update(parent_agent_id="missing"), "UNKNOWN_PARENT"),
        (
            lambda value: value["agents"][1].update(parent_agent_id="research-sensitive"),
            "SELF_PARENT",
        ),
        (lambda value: value["agents"][1].update(delegation_id=None), "MISSING_DELEGATION_ID"),
        (
            lambda value: value["agents"][2].update(delegation_id="delegate-1"),
            "DUPLICATE_DELEGATION_ID",
        ),
        (
            lambda value: value["agents"][0].update(delegation_id="root-delegation"),
            "ROOT_HAS_DELEGATION_ID",
        ),
        (lambda value: value["agents"][1].update(parent_agent_id=None), "ROOT_COUNT"),
    ],
)
def test_rejects_graph_integrity_failures(mutate, expected):
    value = artifact()
    mutate(value)
    report = audit_delegation(value)
    assert report.accepted is False
    assert expected in codes(report)


def test_detects_multi_node_cycle_without_recursion_failure():
    value = artifact()
    value["agents"][0].update(parent_agent_id="review-sensitive", delegation_id="delegate-root")
    value["agents"][2]["parent_agent_id"] = "root-sensitive"
    report = audit_delegation(value)
    assert "DELEGATION_CYCLE" in codes(report)


def test_deep_chain_is_iterative_and_findings_are_bounded():
    value = artifact()
    value["agents"] = [
        agent("agent-0", None, None, ended="2026-09-26T09:00:00Z"),
        *[
            agent(
                f"agent-{index}",
                f"agent-{index - 1}",
                f"delegate-{index}",
                started="2026-09-26T08:00:01Z",
            )
            for index in range(1, 2_000)
        ],
    ]
    report = audit_delegation(
        value,
        DelegationPolicy(
            max_agents=3_000,
            max_depth=10,
            max_total_tokens=1_000_000,
            max_total_tool_calls=3_000,
        ),
    )

    assert report.accepted is False
    assert report.metrics["max_depth"] == 1_999
    assert len(report.findings) == 256
    assert report.findings[-1].code == "FINDINGS_TRUNCATED"


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (DelegationPolicy(max_agents=2), "AGENT_LIMIT"),
        (DelegationPolicy(max_depth=0), "DEPTH_LIMIT"),
        (DelegationPolicy(max_children_per_agent=1), "FANOUT_LIMIT"),
        (DelegationPolicy(max_total_tokens=449), "TOTAL_TOKEN_LIMIT"),
        (DelegationPolicy(max_tokens_per_agent=149), "AGENT_TOKEN_LIMIT"),
        (DelegationPolicy(max_total_tool_calls=2), "TOTAL_TOOL_CALL_LIMIT"),
        (DelegationPolicy(max_tool_calls_per_agent=0), "AGENT_TOOL_CALL_LIMIT"),
        (DelegationPolicy(max_run_seconds=179), "RUN_DURATION_LIMIT"),
    ],
)
def test_enforces_resource_and_amplification_budgets(policy, expected):
    report = audit_delegation(artifact(), policy)
    assert expected in codes(report)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"started_at": "2026-09-26T07:59:59Z"}, "CHILD_BEFORE_PARENT"),
        ({"started_at": "2026-09-26T08:04:00Z"}, "CHILD_AFTER_PARENT_END"),
        ({"ended_at": "2026-09-26T08:04:00Z"}, "CHILD_OUTLIVES_PARENT"),
        (
            {"started_at": "2026-09-26T08:02:00Z", "ended_at": "2026-09-26T08:01:00Z"},
            "NEGATIVE_DURATION",
        ),
    ],
)
def test_rejects_impossible_causal_timing(changes, expected):
    value = artifact()
    value["agents"][1].update(changes)
    assert expected in codes(audit_delegation(value))


def test_terminal_and_success_policies_are_independent():
    running = artifact()
    running["agents"][1].update(status="running", ended_at=None)
    assert {"NON_TERMINAL_AGENT", "AGENT_NOT_SUCCESSFUL"} <= codes(audit_delegation(running))

    failed = artifact()
    failed["agents"][1]["status"] = "failed"
    relaxed = audit_delegation(
        failed, DelegationPolicy(require_terminal=True, require_success=False)
    )
    assert relaxed.accepted is True


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"[]",
        b'{"a":1,"a":2}',
        b'{"value":NaN}',
        b"\xff",
        b"{",
    ],
)
def test_strict_loader_rejects_ambiguous_or_malformed_json(payload):
    with pytest.raises(ValueError):
        load_artifact(payload)


def test_strict_loader_enforces_byte_budget():
    payload = json.dumps(artifact()).encode()
    with pytest.raises(ValueError):
        load_artifact(payload, max_bytes=len(payload) - 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version="v2"),
        lambda value: value.update(extra=True),
        lambda value: value.update(run_id="bad id"),
        lambda value: value.update(agents=[]),
        lambda value: value["agents"][0].update(extra=True),
        lambda value: value["agents"][0].update(prompt_tokens=True),
        lambda value: value["agents"][0].update(started_at="2026-09-26T08:00:00"),
        lambda value: value["agents"][0].update(status="unknown"),
        lambda value: value["agents"][0].update(status="running"),
    ],
)
def test_rejects_malformed_artifact_contract(mutate):
    value = artifact()
    mutate(value)
    with pytest.raises(ValueError):
        audit_delegation(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_agents": 0},
        {"max_depth": -1},
        {"max_total_tokens": True},
        {"max_run_seconds": float("inf")},
        {"require_terminal": "yes"},
    ],
)
def test_rejects_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        audit_delegation(artifact(), DelegationPolicy(**kwargs))


def test_cli_has_distinct_success_rejection_and_malformed_exit_codes(tmp_path, capsys):
    source = tmp_path / "artifact.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(artifact()), encoding="utf-8")
    assert main([str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["accepted"] is True

    rejected = artifact()
    rejected["agents"][1]["status"] = "failed"
    source.write_text(json.dumps(rejected), encoding="utf-8")
    assert main([str(source)]) == 2
    assert json.loads(capsys.readouterr().out)["accepted"] is False

    source.write_text('{"duplicate":1,"duplicate":2}', encoding="utf-8")
    assert main([str(source)]) == 64
    malformed = json.loads(capsys.readouterr().out)
    assert malformed["error_code"] == "MALFORMED_ARTIFACT"
    assert "duplicate" not in json.dumps(malformed)
