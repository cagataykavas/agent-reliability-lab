"""Fail-closed resource and topology audit for multi-agent delegation traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

SCHEMA_VERSION = "agent-delegation/v1"
REPORT_VERSION = "agent-delegation-report/v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_AGENT_FIELDS = frozenset(
    {
        "agent_id",
        "parent_agent_id",
        "delegation_id",
        "started_at",
        "ended_at",
        "status",
        "prompt_tokens",
        "completion_tokens",
        "tool_calls",
    }
)
_DOCUMENT_FIELDS = frozenset({"schema_version", "run_id", "policy_id", "agents"})
_STATUSES = frozenset({"succeeded", "failed", "cancelled", "running"})
_MAX_REPORTED_FINDINGS = 256
_MAX_REPORTED_CYCLE_MEMBERS = 32


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        _invalid(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _bounded_number(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        _invalid(f"{name} must be finite and in [{minimum}, {maximum}]")
    return number


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        _invalid(f"{name} must be a bounded identifier")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _timestamp(name: str, value: object, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or len(value) > 64:
        _invalid(f"{name} must be a bounded RFC 3339 timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _invalid(f"{name} must include a timezone offset")
    return parsed.astimezone(UTC)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _invalid("JSON objects must not contain duplicate fields")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _invalid(f"non-finite JSON constant is not allowed: {value}")


@dataclass(frozen=True)
class DelegationPolicy:
    max_agents: int = 64
    max_depth: int = 4
    max_children_per_agent: int = 8
    max_total_tokens: int = 100_000
    max_tokens_per_agent: int = 20_000
    max_total_tool_calls: int = 128
    max_tool_calls_per_agent: int = 32
    max_run_seconds: float = 900.0
    require_terminal: bool = True
    require_success: bool = True

    def validate(self) -> None:
        _bounded_int("max_agents", self.max_agents, 1, 100_000)
        _bounded_int("max_depth", self.max_depth, 0, 1_000)
        _bounded_int("max_children_per_agent", self.max_children_per_agent, 0, 100_000)
        _bounded_int("max_total_tokens", self.max_total_tokens, 0, 10**12)
        _bounded_int("max_tokens_per_agent", self.max_tokens_per_agent, 0, 10**12)
        _bounded_int("max_total_tool_calls", self.max_total_tool_calls, 0, 10**9)
        _bounded_int("max_tool_calls_per_agent", self.max_tool_calls_per_agent, 0, 10**9)
        _bounded_number("max_run_seconds", self.max_run_seconds, 0.001, 31_536_000.0)
        if not isinstance(self.require_terminal, bool) or not isinstance(
            self.require_success, bool
        ):
            _invalid("require_terminal and require_success must be booleans")


@dataclass(frozen=True)
class Finding:
    code: str
    evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "evidence": self.evidence}


@dataclass(frozen=True)
class DelegationReport:
    accepted: bool
    artifact_sha256: str
    run_sha256: str
    policy_sha256: str
    metrics: dict[str, int | float]
    findings: tuple[Finding, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_VERSION,
            "accepted": self.accepted,
            "artifact_sha256": self.artifact_sha256,
            "run_sha256": self.run_sha256,
            "policy_sha256": self.policy_sha256,
            "metrics": self.metrics,
            "findings": [finding.as_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class _Agent:
    agent_id: str
    parent_agent_id: str | None
    delegation_id: str | None
    started_at: datetime
    ended_at: datetime | None
    status: str
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def load_artifact(payload: bytes, *, max_bytes: int = 1_048_576) -> dict[str, Any]:
    """Parse bounded JSON while rejecting duplicate fields and non-finite values."""
    limit = _bounded_int("max_bytes", max_bytes, 1, 16_777_216)
    if not payload or len(payload) > limit:
        _invalid("artifact byte size is outside the configured budget")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        _invalid("artifact root must be an object")
    return value


def _parse_agent(value: object) -> _Agent:
    if not isinstance(value, dict) or set(value) != _AGENT_FIELDS:
        _invalid("each agent must contain exactly the documented fields")
    agent_id = _identifier("agent_id", value["agent_id"])
    parent = value["parent_agent_id"]
    delegation = value["delegation_id"]
    if parent is not None:
        parent = _identifier("parent_agent_id", parent)
    if delegation is not None:
        delegation = _identifier("delegation_id", delegation)
    status = value["status"]
    if status not in _STATUSES:
        _invalid("status must be succeeded, failed, cancelled, or running")
    ended_at = _timestamp("ended_at", value["ended_at"], optional=True)
    if status == "running" and ended_at is not None:
        _invalid("running agents must not have ended_at")
    if status != "running" and ended_at is None:
        _invalid("terminal agents require ended_at")
    return _Agent(
        agent_id=agent_id,
        parent_agent_id=parent,
        delegation_id=delegation,
        started_at=_timestamp("started_at", value["started_at"]),  # type: ignore[arg-type]
        ended_at=ended_at,
        status=status,
        prompt_tokens=_bounded_int("prompt_tokens", value["prompt_tokens"], 0, 10**12),
        completion_tokens=_bounded_int("completion_tokens", value["completion_tokens"], 0, 10**12),
        tool_calls=_bounded_int("tool_calls", value["tool_calls"], 0, 10**9),
    )


def audit_delegation(
    document: dict[str, Any], policy: DelegationPolicy | None = None
) -> DelegationReport:
    """Audit one completed delegation artifact without exposing its identifiers."""
    active_policy = policy or DelegationPolicy()
    active_policy.validate()
    if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
        _invalid("artifact must contain exactly the documented fields")
    if document["schema_version"] != SCHEMA_VERSION:
        _invalid("unsupported artifact schema_version")
    run_id = _identifier("run_id", document["run_id"])
    policy_id = _identifier("policy_id", document["policy_id"])
    values = document["agents"]
    if not isinstance(values, list) or not values:
        _invalid("agents must be a non-empty array")
    if len(values) > 100_000:
        _invalid("agents exceeds the parser safety budget")
    agents = [_parse_agent(value) for value in values]

    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    artifact_sha256 = _digest(canonical)
    findings: list[Finding] = []

    by_id: dict[str, _Agent] = {}
    duplicates: set[str] = set()
    for agent in agents:
        if agent.agent_id in by_id:
            duplicates.add(agent.agent_id)
        else:
            by_id[agent.agent_id] = agent
    for agent_id in sorted(duplicates):
        findings.append(Finding("DUPLICATE_AGENT_ID", {"agent_sha256": _digest(agent_id)}))

    roots = [agent for agent in agents if agent.parent_agent_id is None]
    if len(roots) != 1:
        findings.append(Finding("ROOT_COUNT", {"observed": len(roots), "required": 1}))
    for root in roots:
        if root.delegation_id is not None:
            findings.append(
                Finding("ROOT_HAS_DELEGATION_ID", {"agent_sha256": _digest(root.agent_id)})
            )

    delegation_owners: dict[str, str] = {}
    children: dict[str, list[_Agent]] = {agent_id: [] for agent_id in by_id}
    for agent in agents:
        if agent.parent_agent_id is None:
            continue
        if agent.delegation_id is None:
            findings.append(
                Finding("MISSING_DELEGATION_ID", {"agent_sha256": _digest(agent.agent_id)})
            )
        elif agent.delegation_id in delegation_owners:
            findings.append(
                Finding(
                    "DUPLICATE_DELEGATION_ID",
                    {"delegation_sha256": _digest(agent.delegation_id)},
                )
            )
        else:
            delegation_owners[agent.delegation_id] = agent.agent_id
        if agent.parent_agent_id == agent.agent_id:
            findings.append(Finding("SELF_PARENT", {"agent_sha256": _digest(agent.agent_id)}))
        elif agent.parent_agent_id not in by_id:
            findings.append(
                Finding(
                    "UNKNOWN_PARENT",
                    {
                        "agent_sha256": _digest(agent.agent_id),
                        "parent_sha256": _digest(agent.parent_agent_id),
                    },
                )
            )
        else:
            children[agent.parent_agent_id].append(agent)

    cycle_members: set[str] = set()
    depths: dict[str, int] = {}

    for agent_id in sorted(by_id):
        if agent_id in depths:
            continue
        chain: list[str] = []
        positions: dict[str, int] = {}
        current = agent_id
        base_depth: int | None = None
        while current not in depths:
            if current in positions:
                cycle_members.update(chain[positions[current] :])
                break
            positions[current] = len(chain)
            chain.append(current)
            parent = by_id[current].parent_agent_id
            if parent is None:
                base_depth = -1
                break
            if parent not in by_id:
                break
            current = parent
        else:
            base_depth = depths[current]
        if base_depth is not None:
            for item in reversed(chain):
                base_depth += 1
                depths[item] = base_depth
    if cycle_members:
        cycle_digests = sorted(_digest(item) for item in cycle_members)
        findings.append(
            Finding(
                "DELEGATION_CYCLE",
                {
                    "agent_count": len(cycle_members),
                    "agent_sha256": cycle_digests[:_MAX_REPORTED_CYCLE_MEMBERS],
                    "truncated": len(cycle_digests) > _MAX_REPORTED_CYCLE_MEMBERS,
                },
            )
        )

    for agent in agents:
        if agent.ended_at is not None and agent.ended_at < agent.started_at:
            findings.append(Finding("NEGATIVE_DURATION", {"agent_sha256": _digest(agent.agent_id)}))
        if active_policy.require_terminal and agent.status == "running":
            findings.append(
                Finding("NON_TERMINAL_AGENT", {"agent_sha256": _digest(agent.agent_id)})
            )
        if active_policy.require_success and agent.status != "succeeded":
            findings.append(
                Finding(
                    "AGENT_NOT_SUCCESSFUL",
                    {"agent_sha256": _digest(agent.agent_id), "status": agent.status},
                )
            )
        if agent.tokens > active_policy.max_tokens_per_agent:
            findings.append(
                Finding(
                    "AGENT_TOKEN_LIMIT",
                    {
                        "agent_sha256": _digest(agent.agent_id),
                        "observed": agent.tokens,
                        "limit": active_policy.max_tokens_per_agent,
                    },
                )
            )
        if agent.tool_calls > active_policy.max_tool_calls_per_agent:
            findings.append(
                Finding(
                    "AGENT_TOOL_CALL_LIMIT",
                    {
                        "agent_sha256": _digest(agent.agent_id),
                        "observed": agent.tool_calls,
                        "limit": active_policy.max_tool_calls_per_agent,
                    },
                )
            )
        agent_depth = depths.get(agent.agent_id)
        if agent_depth is not None and agent_depth > active_policy.max_depth:
            findings.append(
                Finding(
                    "DEPTH_LIMIT",
                    {
                        "agent_sha256": _digest(agent.agent_id),
                        "observed": agent_depth,
                        "limit": active_policy.max_depth,
                    },
                )
            )
        fanout = len(children.get(agent.agent_id, []))
        if fanout > active_policy.max_children_per_agent:
            findings.append(
                Finding(
                    "FANOUT_LIMIT",
                    {
                        "agent_sha256": _digest(agent.agent_id),
                        "observed": fanout,
                        "limit": active_policy.max_children_per_agent,
                    },
                )
            )
        if agent.parent_agent_id in by_id and agent.parent_agent_id != agent.agent_id:
            parent = by_id[agent.parent_agent_id]
            if agent.started_at < parent.started_at:
                findings.append(
                    Finding("CHILD_BEFORE_PARENT", {"agent_sha256": _digest(agent.agent_id)})
                )
            if parent.ended_at is not None and agent.started_at > parent.ended_at:
                findings.append(
                    Finding("CHILD_AFTER_PARENT_END", {"agent_sha256": _digest(agent.agent_id)})
                )
            if (
                parent.ended_at is not None
                and agent.ended_at is not None
                and agent.ended_at > parent.ended_at
            ):
                findings.append(
                    Finding("CHILD_OUTLIVES_PARENT", {"agent_sha256": _digest(agent.agent_id)})
                )

    total_tokens = sum(agent.tokens for agent in agents)
    total_tool_calls = sum(agent.tool_calls for agent in agents)
    if len(agents) > active_policy.max_agents:
        findings.append(
            Finding("AGENT_LIMIT", {"observed": len(agents), "limit": active_policy.max_agents})
        )
    if total_tokens > active_policy.max_total_tokens:
        findings.append(
            Finding(
                "TOTAL_TOKEN_LIMIT",
                {"observed": total_tokens, "limit": active_policy.max_total_tokens},
            )
        )
    if total_tool_calls > active_policy.max_total_tool_calls:
        findings.append(
            Finding(
                "TOTAL_TOOL_CALL_LIMIT",
                {"observed": total_tool_calls, "limit": active_policy.max_total_tool_calls},
            )
        )

    run_seconds = 0.0
    if len(roots) == 1 and roots[0].ended_at is not None:
        run_seconds = max(0.0, (roots[0].ended_at - roots[0].started_at).total_seconds())
        if run_seconds > active_policy.max_run_seconds:
            findings.append(
                Finding(
                    "RUN_DURATION_LIMIT",
                    {"observed": run_seconds, "limit": active_policy.max_run_seconds},
                )
            )

    findings.sort(key=lambda item: (item.code, json.dumps(item.evidence, sort_keys=True)))
    finding_count = len(findings)
    if finding_count > _MAX_REPORTED_FINDINGS:
        findings = findings[: _MAX_REPORTED_FINDINGS - 1]
        findings.append(
            Finding(
                "FINDINGS_TRUNCATED",
                {"observed": finding_count, "reported": _MAX_REPORTED_FINDINGS},
            )
        )
    metrics: dict[str, int | float] = {
        "agent_count": len(agents),
        "root_count": len(roots),
        "max_depth": max(depths.values(), default=0),
        "max_fanout": max((len(value) for value in children.values()), default=0),
        "total_prompt_tokens": sum(agent.prompt_tokens for agent in agents),
        "total_completion_tokens": sum(agent.completion_tokens for agent in agents),
        "total_tokens": total_tokens,
        "total_tool_calls": total_tool_calls,
        "run_seconds": run_seconds,
    }
    return DelegationReport(
        accepted=not findings,
        artifact_sha256=artifact_sha256,
        run_sha256=_digest(run_id),
        policy_sha256=_digest(
            json.dumps(
                {"policy_id": policy_id, **asdict(active_policy)},
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        metrics=metrics,
        findings=tuple(findings),
    )


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=1_048_576)
    args = parser.parse_args(argv)
    try:
        if args.artifact.stat().st_size > args.max_bytes:
            _invalid("artifact byte size is outside the configured budget")
        document = load_artifact(args.artifact.read_bytes(), max_bytes=args.max_bytes)
        result = audit_delegation(document)
        payload = json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":"))
        exit_code = 0 if result.accepted else 2
    except (OSError, ValueError):
        payload = json.dumps(
            {
                "schema_version": REPORT_VERSION,
                "accepted": False,
                "error_code": "MALFORMED_ARTIFACT",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        exit_code = 64
    if args.output is None:
        print(payload)
    else:
        _write_atomic(args.output, payload)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
