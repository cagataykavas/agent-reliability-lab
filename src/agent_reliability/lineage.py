from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    evidence_id: str
    content_sha256: str

    @classmethod
    def from_dict(cls, values: object) -> EvidenceArtifact:
        item = _mapping(values, "evidence artifact")
        return cls(
            evidence_id=_non_empty_string(item.get("evidence_id"), "evidence_id"),
            content_sha256=_non_empty_string(item.get("content_sha256"), "content_sha256"),
        )


@dataclass(frozen=True, slots=True)
class LineageCall:
    call_id: str
    tool_name: str
    depends_on: tuple[str, ...]
    succeeded: bool
    evidence: tuple[EvidenceArtifact, ...]

    @classmethod
    def from_dict(cls, values: object) -> LineageCall:
        item = _mapping(values, "lineage call")
        succeeded = item.get("succeeded")
        if not isinstance(succeeded, bool):
            raise TypeError("succeeded must be a boolean")
        dependencies = _sequence(item.get("depends_on", []), "depends_on")
        evidence = _sequence(item.get("evidence", []), "evidence")
        return cls(
            call_id=_non_empty_string(item.get("call_id"), "call_id"),
            tool_name=_non_empty_string(item.get("tool_name"), "tool_name"),
            depends_on=tuple(
                _non_empty_string(value, "dependency call ID") for value in dependencies
            ),
            succeeded=succeeded,
            evidence=tuple(EvidenceArtifact.from_dict(value) for value in evidence),
        )


@dataclass(frozen=True, slots=True)
class LineageTrace:
    trace_id: str
    calls: tuple[LineageCall, ...]
    cited_evidence_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, values: object) -> LineageTrace:
        item = _mapping(values, "lineage trace")
        calls = _sequence(item.get("calls"), "calls")
        citations = _sequence(item.get("cited_evidence_ids", []), "cited_evidence_ids")
        return cls(
            trace_id=_non_empty_string(item.get("trace_id"), "trace_id"),
            calls=tuple(LineageCall.from_dict(value) for value in calls),
            cited_evidence_ids=tuple(
                _non_empty_string(value, "cited evidence ID") for value in citations
            ),
        )


@dataclass(frozen=True, slots=True)
class LineagePolicy:
    required_evidence_ids: frozenset[str] = frozenset()
    max_calls: int = 32
    max_dependency_depth: int = 8
    require_all_produced_evidence_cited: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.required_evidence_ids, frozenset):
            raise TypeError("required_evidence_ids must be a frozenset")
        if isinstance(self.max_calls, bool) or not isinstance(self.max_calls, int):
            raise TypeError("max_calls must be an integer")
        if self.max_calls < 1:
            raise ValueError("max_calls must be positive")
        if isinstance(self.max_dependency_depth, bool) or not isinstance(
            self.max_dependency_depth, int
        ):
            raise TypeError("max_dependency_depth must be an integer")
        if self.max_dependency_depth < 0:
            raise ValueError("max_dependency_depth must be non-negative")
        if not isinstance(self.require_all_produced_evidence_cited, bool):
            raise TypeError("require_all_produced_evidence_cited must be a boolean")
        for evidence_id in self.required_evidence_ids:
            _non_empty_string(evidence_id, "required evidence ID")


@dataclass(frozen=True, slots=True)
class LineageFinding:
    code: str
    detail: str
    call_id: str | None = None
    evidence_id: str | None = None
    dependency_id: str | None = None


@dataclass(frozen=True, slots=True)
class LineageReport:
    trace_id: str
    passed: bool
    call_count: int
    evidence_count: int
    cited_evidence_count: int
    max_observed_dependency_depth: int
    evidence_sources: dict[str, str]
    findings: tuple[LineageFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "passed": self.passed,
            "summary": {
                "call_count": self.call_count,
                "evidence_count": self.evidence_count,
                "cited_evidence_count": self.cited_evidence_count,
                "max_observed_dependency_depth": self.max_observed_dependency_depth,
            },
            "evidence_sources": dict(sorted(self.evidence_sources.items())),
            "findings": [asdict(finding) for finding in self.findings],
        }


def audit_evidence_lineage(
    trace: LineageTrace,
    policy: LineagePolicy | None = None,
) -> LineageReport:
    active_policy = policy or LineagePolicy()
    findings: list[LineageFinding] = []

    if len(trace.calls) > active_policy.max_calls:
        findings.append(
            LineageFinding(
                code="call_budget_exceeded",
                detail=(
                    f"trace contains {len(trace.calls)} calls; maximum is {active_policy.max_calls}"
                ),
            )
        )

    call_positions: dict[str, int] = {}
    for position, call in enumerate(trace.calls):
        if call.call_id in call_positions:
            findings.append(
                LineageFinding(
                    code="duplicate_call_id",
                    detail="call ID is not unique within the trace",
                    call_id=call.call_id,
                )
            )
        else:
            call_positions[call.call_id] = position

    call_success = {call.call_id: call.succeeded for call in trace.calls}
    depths: dict[str, int] = {}
    evidence_sources: dict[str, str] = {}
    evidence_count = 0

    for position, call in enumerate(trace.calls):
        duplicate_dependencies = _duplicates(call.depends_on)
        for dependency_id in duplicate_dependencies:
            findings.append(
                LineageFinding(
                    code="duplicate_dependency",
                    detail="dependency is repeated on the same call",
                    call_id=call.call_id,
                    dependency_id=dependency_id,
                )
            )

        valid_depths: list[int] = []
        for dependency_id in call.depends_on:
            dependency_position = call_positions.get(dependency_id)
            if dependency_position is None:
                findings.append(
                    LineageFinding(
                        code="unknown_dependency",
                        detail="dependency does not reference a call in this trace",
                        call_id=call.call_id,
                        dependency_id=dependency_id,
                    )
                )
                continue
            if dependency_position >= position:
                findings.append(
                    LineageFinding(
                        code="non_causal_dependency",
                        detail="dependency must reference an earlier call",
                        call_id=call.call_id,
                        dependency_id=dependency_id,
                    )
                )
                continue
            if not call_success[dependency_id]:
                findings.append(
                    LineageFinding(
                        code="failed_dependency",
                        detail="call depends on a failed prerequisite",
                        call_id=call.call_id,
                        dependency_id=dependency_id,
                    )
                )
            valid_depths.append(depths.get(dependency_id, 0))

        depth = 1 + max(valid_depths) if valid_depths else 0
        depths.setdefault(call.call_id, depth)
        if depth > active_policy.max_dependency_depth:
            findings.append(
                LineageFinding(
                    code="dependency_depth_exceeded",
                    detail=(
                        f"dependency depth {depth} exceeds maximum "
                        f"{active_policy.max_dependency_depth}"
                    ),
                    call_id=call.call_id,
                )
            )

        for evidence in call.evidence:
            evidence_count += 1
            if not SHA256_PATTERN.fullmatch(evidence.content_sha256):
                findings.append(
                    LineageFinding(
                        code="invalid_content_hash",
                        detail="content_sha256 must be a canonical lowercase SHA-256 digest",
                        call_id=call.call_id,
                        evidence_id=evidence.evidence_id,
                    )
                )
            if not call.succeeded:
                findings.append(
                    LineageFinding(
                        code="evidence_from_failed_call",
                        detail="failed calls cannot produce trusted evidence",
                        call_id=call.call_id,
                        evidence_id=evidence.evidence_id,
                    )
                )
            if evidence.evidence_id in evidence_sources:
                findings.append(
                    LineageFinding(
                        code="duplicate_evidence_id",
                        detail="evidence ID is produced by more than one call",
                        call_id=call.call_id,
                        evidence_id=evidence.evidence_id,
                    )
                )
            else:
                evidence_sources[evidence.evidence_id] = call.call_id

    cited = set(trace.cited_evidence_ids)
    for evidence_id in _duplicates(trace.cited_evidence_ids):
        findings.append(
            LineageFinding(
                code="duplicate_citation",
                detail="evidence ID is cited more than once",
                evidence_id=evidence_id,
            )
        )
    for evidence_id in sorted(cited):
        if evidence_id not in evidence_sources:
            findings.append(
                LineageFinding(
                    code="unknown_citation",
                    detail="citation is not produced by any recorded call",
                    evidence_id=evidence_id,
                )
            )
    for evidence_id in sorted(active_policy.required_evidence_ids - cited):
        findings.append(
            LineageFinding(
                code="required_evidence_not_cited",
                detail="required evidence is absent from the final citation set",
                evidence_id=evidence_id,
            )
        )
    if active_policy.require_all_produced_evidence_cited:
        for evidence_id in sorted(evidence_sources.keys() - cited):
            findings.append(
                LineageFinding(
                    code="uncited_evidence",
                    detail="produced evidence is not cited by the final response",
                    evidence_id=evidence_id,
                )
            )

    return LineageReport(
        trace_id=trace.trace_id,
        passed=not findings,
        call_count=len(trace.calls),
        evidence_count=evidence_count,
        cited_evidence_count=len(cited),
        max_observed_dependency_depth=max(depths.values(), default=0),
        evidence_sources=evidence_sources,
        findings=tuple(findings),
    )


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_reliability.lineage",
        description="Audit tool-call and evidence provenance in a recorded agent trace.",
    )
    parser.add_argument("trace", help="JSON evidence-lineage artifact")
    parser.add_argument("--required-evidence", action="append", default=[])
    parser.add_argument("--max-calls", type=int, default=32)
    parser.add_argument("--max-dependency-depth", type=int, default=8)
    parser.add_argument("--require-all-produced-evidence-cited", action="store_true")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.trace).read_text(encoding="utf-8"))
        trace = LineageTrace.from_dict(payload)
        policy = LineagePolicy(
            required_evidence_ids=frozenset(args.required_evidence),
            max_calls=args.max_calls,
            max_dependency_depth=args.max_dependency_depth,
            require_all_produced_evidence_cited=args.require_all_produced_evidence_cited,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, sort_keys=True))
        return 2

    report = audit_evidence_lineage(trace, policy)
    output = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
