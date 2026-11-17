from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
from pathlib import Path

import pytest

from agent_reliability.progress import (
    ProgressAuditError,
    ProgressPolicy,
    audit_progress_trace,
    load_progress_trace,
)
from agent_reliability.progress_cli import main


def _step(sequence: int, kind: str, action: str, state: str) -> dict[str, object]:
    return {
        "sequence": sequence,
        "kind": kind,
        "action_sha256": action * 64,
        "state_sha256": state * 64,
    }


def valid_trace() -> dict[str, object]:
    return {
        "schema_version": "agent-progress/v1",
        "run_id": "run-42",
        "steps": [
            _step(1, "model", "a", "1"),
            _step(2, "tool", "b", "2"),
            _step(3, "model", "c", "3"),
            _step(4, "final", "d", "4"),
        ],
    }


def test_progressing_terminal_trace_is_accepted() -> None:
    report = audit_progress_trace(valid_trace())

    assert report.accepted
    assert report.findings == ()
    assert report.terminal_kind == "final"
    assert report.run_sha256 != "run-42"


def test_consecutive_no_progress_is_rejected() -> None:
    trace = valid_trace()
    trace["steps"] = [
        _step(1, "model", "a", "1"),
        _step(2, "tool", "b", "1"),
        _step(3, "tool", "c", "1"),
        _step(4, "tool", "d", "1"),
        _step(5, "final", "e", "2"),
    ]

    report = audit_progress_trace(trace)

    assert "NO_PROGRESS_STREAK_EXCEEDED" in report.findings
    assert report.max_no_progress_steps == 3


def test_revisited_cycle_and_action_state_pair_are_rejected() -> None:
    trace = valid_trace()
    trace["steps"] = [
        _step(1, "model", "a", "1"),
        _step(2, "tool", "b", "2"),
        _step(3, "model", "a", "1"),
        _step(4, "tool", "b", "2"),
        _step(5, "model", "a", "1"),
        _step(6, "final", "f", "3"),
    ]
    policy = ProgressPolicy(max_action_state_repeats=2, max_state_visits=2)

    report = audit_progress_trace(trace, policy)

    assert "ACTION_STATE_REPEAT_EXCEEDED" in report.findings
    assert "STATE_VISIT_BUDGET_EXCEEDED" in report.findings


def test_missing_or_early_terminal_is_rejected() -> None:
    unterminated = valid_trace()
    unterminated["steps"][-1]["kind"] = "model"
    early = valid_trace()
    early["steps"][1]["kind"] = "final"

    assert audit_progress_trace(unterminated).findings == ("MISSING_TERMINAL",)
    assert audit_progress_trace(early).findings == ("EVENT_AFTER_TERMINAL",)


def test_terminal_event_does_not_consume_loop_budget() -> None:
    trace = valid_trace()
    trace["steps"] = [
        _step(1, "model", "a", "1"),
        _step(2, "final", "b", "1"),
    ]

    report = audit_progress_trace(
        trace,
        ProgressPolicy(
            max_no_progress_steps=0,
            max_action_state_repeats=1,
            max_state_visits=1,
        ),
    )

    assert report.accepted
    assert report.max_no_progress_steps == 0
    assert report.max_state_visits == 1


def test_step_budget_and_policy_types_fail_closed() -> None:
    report = audit_progress_trace(valid_trace(), ProgressPolicy(max_steps=3))
    assert report.findings == ("STEP_BUDGET_EXCEEDED",)

    with pytest.raises(ProgressAuditError, match="integers"):
        ProgressPolicy(max_steps=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda trace: trace.update(extra=True),
        lambda trace: trace["steps"][0].update(sequence=2),
        lambda trace: trace["steps"][0].update(kind="unknown"),
        lambda trace: trace["steps"][0].update(action_sha256="not-a-digest"),
    ],
)
def test_malformed_trace_is_rejected(mutation) -> None:
    trace = valid_trace()
    mutation(trace)

    with pytest.raises(ProgressAuditError):
        audit_progress_trace(trace)


def test_evidence_is_deterministic_and_binds_policy() -> None:
    trace = valid_trace()
    first = audit_progress_trace(trace)
    repeated = audit_progress_trace(copy.deepcopy(trace))
    changed_policy = audit_progress_trace(trace, ProgressPolicy(max_steps=64))

    assert first == repeated
    assert first.policy_sha256 != changed_policy.policy_sha256
    assert first.evidence_sha256 != changed_policy.evidence_sha256


def test_strict_file_loader_rejects_duplicate_nonfinite_and_symlink() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"run_id":"a","run_id":"b"}', encoding="utf-8")
        nonfinite = root / "nonfinite.json"
        nonfinite.write_text('{"value":NaN}', encoding="utf-8")
        target = root / "target.json"
        target.write_text(json.dumps(valid_trace()), encoding="utf-8")
        link = root / "link.json"
        link.symlink_to(target)

        with pytest.raises(ProgressAuditError, match="duplicate"):
            load_progress_trace(duplicate)
        with pytest.raises(ProgressAuditError, match="non-finite"):
            load_progress_trace(nonfinite)
        with pytest.raises(ProgressAuditError, match="non-symlink"):
            load_progress_trace(link)


def test_cli_distinguishes_accepted_rejected_and_malformed(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted.json"
    rejected = tmp_path / "rejected.json"
    malformed = tmp_path / "malformed.json"
    output = tmp_path / "report.json"
    accepted.write_text(json.dumps(valid_trace()), encoding="utf-8")
    rejected_trace = valid_trace()
    rejected_trace["steps"][-1]["kind"] = "model"
    rejected.write_text(json.dumps(rejected_trace), encoding="utf-8")
    malformed.write_text("not-json", encoding="utf-8")

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        assert main([str(accepted), "--output", str(output)]) == 0
        assert main([str(rejected)]) == 2
        assert main([str(malformed)]) == 3
    assert json.loads(output.read_text(encoding="utf-8"))["accepted"] is True
