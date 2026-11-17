from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "agent-progress/v1"
MAX_TRACE_BYTES = 2 * 1024 * 1024
MAX_TRACE_STEPS = 10_000
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ProgressAuditError(ValueError):
    """Raised when progress evidence cannot be interpreted unambiguously."""


class StepKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"
    HANDOFF = "handoff"
    FINAL = "final"
    ERROR = "error"

    @property
    def terminal(self) -> bool:
        return self in {StepKind.FINAL, StepKind.ERROR}


@dataclass(frozen=True, slots=True)
class ProgressPolicy:
    max_steps: int = 128
    max_no_progress_steps: int = 2
    max_action_state_repeats: int = 2
    max_state_visits: int = 3
    require_terminal: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_steps,
            self.max_no_progress_steps,
            self.max_action_state_repeats,
            self.max_state_visits,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise ProgressAuditError("policy budgets must be integers")
        if not 1 <= self.max_steps <= MAX_TRACE_STEPS:
            raise ProgressAuditError("max_steps is outside the permitted range")
        if not 0 <= self.max_no_progress_steps < self.max_steps:
            raise ProgressAuditError("max_no_progress_steps is outside the permitted range")
        if not 1 <= self.max_action_state_repeats <= self.max_steps:
            raise ProgressAuditError("max_action_state_repeats is outside the permitted range")
        if not 1 <= self.max_state_visits <= self.max_steps:
            raise ProgressAuditError("max_state_visits is outside the permitted range")
        if not isinstance(self.require_terminal, bool):
            raise ProgressAuditError("require_terminal must be boolean")


@dataclass(frozen=True, slots=True)
class ProgressAuditReport:
    schema_version: str
    accepted: bool
    run_sha256: str
    trace_sha256: str
    policy_sha256: str
    steps: int
    terminal_kind: str | None
    max_no_progress_steps: int
    max_action_state_repeats: int
    max_state_visits: int
    findings: tuple[str, ...]
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Step:
    sequence: int
    kind: StepKind
    action_sha256: str
    state_sha256: str


def audit_progress_trace(
    trace: dict[str, Any], policy: ProgressPolicy | None = None
) -> ProgressAuditReport:
    policy = policy or ProgressPolicy()
    if not isinstance(trace, dict):
        raise ProgressAuditError("trace root must be an object")
    if set(trace) != {"schema_version", "run_id", "steps"}:
        raise ProgressAuditError("trace root fields do not match the schema")
    if trace["schema_version"] != SCHEMA_VERSION:
        raise ProgressAuditError("unsupported progress trace schema")
    run_id = trace["run_id"]
    if not isinstance(run_id, str) or not 1 <= len(run_id) <= 200:
        raise ProgressAuditError("run_id must contain 1 to 200 characters")
    raw_steps = trace["steps"]
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_TRACE_STEPS:
        raise ProgressAuditError("steps must be a non-empty bounded array")
    steps = tuple(_parse_step(item, index) for index, item in enumerate(raw_steps, 1))

    findings: list[str] = []
    if len(steps) > policy.max_steps:
        _add_finding(findings, "STEP_BUDGET_EXCEEDED")

    state_visits: dict[str, int] = {}
    action_state_visits: dict[tuple[str, str], int] = {}
    previous_state: str | None = None
    no_progress_streak = 0
    max_no_progress = 0
    max_pair_repeats = 0
    max_visits = 0
    terminal_kind: str | None = None

    for index, step in enumerate(steps):
        if step.kind.terminal:
            terminal_kind = step.kind.value
            if index != len(steps) - 1:
                _add_finding(findings, "EVENT_AFTER_TERMINAL")
            continue

        state_visits[step.state_sha256] = state_visits.get(step.state_sha256, 0) + 1
        max_visits = max(max_visits, state_visits[step.state_sha256])
        pair = (step.action_sha256, step.state_sha256)
        action_state_visits[pair] = action_state_visits.get(pair, 0) + 1
        max_pair_repeats = max(max_pair_repeats, action_state_visits[pair])

        if step.state_sha256 == previous_state:
            no_progress_streak += 1
        else:
            no_progress_streak = 0
        max_no_progress = max(max_no_progress, no_progress_streak)
        previous_state = step.state_sha256

    if max_no_progress > policy.max_no_progress_steps:
        _add_finding(findings, "NO_PROGRESS_STREAK_EXCEEDED")
    if max_pair_repeats > policy.max_action_state_repeats:
        _add_finding(findings, "ACTION_STATE_REPEAT_EXCEEDED")
    if max_visits > policy.max_state_visits:
        _add_finding(findings, "STATE_VISIT_BUDGET_EXCEEDED")
    if policy.require_terminal and not steps[-1].kind.terminal:
        _add_finding(findings, "MISSING_TERMINAL")

    trace_sha256 = _digest(trace)
    policy_sha256 = _digest(asdict(policy))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "accepted": not findings,
        "run_sha256": _text_digest(run_id),
        "trace_sha256": trace_sha256,
        "policy_sha256": policy_sha256,
        "steps": len(steps),
        "terminal_kind": terminal_kind,
        "max_no_progress_steps": max_no_progress,
        "max_action_state_repeats": max_pair_repeats,
        "max_state_visits": max_visits,
        "findings": tuple(findings),
    }
    return ProgressAuditReport(**payload, evidence_sha256=_digest(payload))


def load_progress_trace(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        before = source.lstat()
    except OSError as exc:
        raise ProgressAuditError("trace cannot be inspected") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProgressAuditError("trace must be a regular non-symlink file")
    if not 0 < before.st_size <= MAX_TRACE_BYTES:
        raise ProgressAuditError("trace size is outside the permitted range")

    descriptor = -1
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ProgressAuditError("trace descriptor is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_TRACE_BYTES + 1)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ProgressAuditError("trace cannot be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after)
    }
    if len(identities) != 1 or len(raw) != opened.st_size or len(raw) > MAX_TRACE_BYTES:
        raise ProgressAuditError("trace changed while it was being read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProgressAuditError("trace must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProgressAuditError("trace root must be an object")
    return value


def _parse_step(value: Any, expected_sequence: int) -> _Step:
    if not isinstance(value, dict) or set(value) != {
        "sequence",
        "kind",
        "action_sha256",
        "state_sha256",
    }:
        raise ProgressAuditError(f"step {expected_sequence} fields do not match the schema")
    sequence = value["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != expected_sequence:
        raise ProgressAuditError("step sequence must be contiguous and one-based")
    try:
        kind = StepKind(value["kind"])
    except (TypeError, ValueError) as exc:
        raise ProgressAuditError(f"step {expected_sequence} kind is invalid") from exc
    for field in ("action_sha256", "state_sha256"):
        digest = value[field]
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ProgressAuditError(f"step {expected_sequence} {field} is invalid")
    return _Step(sequence, kind, value["action_sha256"], value["state_sha256"])


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProgressAuditError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProgressAuditError(f"non-finite JSON constant is forbidden: {value}")


def _add_finding(findings: list[str], finding: str) -> None:
    if finding not in findings:
        findings.append(finding)


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ProgressAuditError("evidence must be canonical finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()
