"""Fail-closed metamorphic audits for paired agent traces.

The audit compares behavioral projections, not answer wording. This makes it useful
for checking that semantics-preserving prompt transformations do not change routing,
while policy-changing transformations produce only predeclared changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agent_reliability.models import AgentTrace


class BehaviorDimension(StrEnum):
    TOOL_SEQUENCE = "tool_sequence"
    TOOL_ARGUMENTS = "tool_arguments"
    TOOL_OUTCOMES = "tool_outcomes"
    EVIDENCE = "evidence"
    ESCALATION = "escalation"
    ERROR_STATE = "error_state"


class MetamorphicRelation(StrEnum):
    INVARIANT = "invariant"
    RESPONSIVE = "responsive"


class FindingCode(StrEnum):
    MISSING_BASELINE_TRACE = "missing_baseline_trace"
    MISSING_TRANSFORMED_TRACE = "missing_transformed_trace"
    MALFORMED_TRACE = "malformed_trace"
    UNEXPECTED_BEHAVIOR_CHANGE = "unexpected_behavior_change"
    REQUIRED_CHANGE_MISSING = "required_change_missing"


@dataclass(frozen=True, slots=True)
class MetamorphicPairSpec:
    pair_id: str
    baseline_case_id: str
    transformed_case_id: str
    relation: MetamorphicRelation
    required_changes: frozenset[BehaviorDimension] = frozenset()
    allowed_changes: frozenset[BehaviorDimension] = frozenset()

    def __post_init__(self) -> None:
        _validate_identifier(self.pair_id, "pair_id")
        _validate_identifier(self.baseline_case_id, "baseline_case_id")
        _validate_identifier(self.transformed_case_id, "transformed_case_id")
        if self.baseline_case_id == self.transformed_case_id:
            raise ValueError("baseline_case_id and transformed_case_id must differ")

        try:
            relation = MetamorphicRelation(self.relation)
            required = frozenset(BehaviorDimension(item) for item in self.required_changes)
            allowed = frozenset(BehaviorDimension(item) for item in self.allowed_changes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pair spec contains an unknown relation or behavior dimension"
            ) from exc
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "required_changes", required)
        object.__setattr__(self, "allowed_changes", allowed)

        if relation is MetamorphicRelation.INVARIANT and (required or allowed):
            raise ValueError("invariant pairs cannot declare required or allowed changes")
        if relation is MetamorphicRelation.RESPONSIVE:
            if not required:
                raise ValueError("responsive pairs must declare at least one required change")
            if not required <= allowed:
                raise ValueError("required changes must be a subset of allowed changes")


@dataclass(frozen=True, slots=True)
class MetamorphicPolicy:
    min_pairs: int = 1
    max_pairs: int = 10_000
    max_calls_per_trace: int = 100
    max_evidence_ids: int = 1_000
    max_identifier_chars: int = 256
    max_argument_bytes: int = 65_536
    max_argument_depth: int = 16
    max_argument_nodes: int = 10_000

    def __post_init__(self) -> None:
        limits = {
            "min_pairs": (self.min_pairs, 1, 10_000),
            "max_pairs": (self.max_pairs, 1, 100_000),
            "max_calls_per_trace": (self.max_calls_per_trace, 1, 10_000),
            "max_evidence_ids": (self.max_evidence_ids, 1, 100_000),
            "max_identifier_chars": (self.max_identifier_chars, 1, 4_096),
            "max_argument_bytes": (self.max_argument_bytes, 2, 10_000_000),
            "max_argument_depth": (self.max_argument_depth, 1, 64),
            "max_argument_nodes": (self.max_argument_nodes, 1, 1_000_000),
        }
        for name, (value, minimum, maximum) in limits.items():
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if self.min_pairs > self.max_pairs:
            raise ValueError("min_pairs cannot exceed max_pairs")


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: FindingCode
    dimension: BehaviorDimension | None
    message: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": str(self.code),
            "dimension": None if self.dimension is None else str(self.dimension),
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class MetamorphicPairResult:
    pair_id: str
    relation: MetamorphicRelation
    passed: bool
    changed_dimensions: tuple[BehaviorDimension, ...]
    findings: tuple[AuditFinding, ...]
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "relation": str(self.relation),
            "passed": self.passed,
            "changed_dimensions": [str(item) for item in self.changed_dimensions],
            "findings": [item.to_dict() for item in self.findings],
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class MetamorphicAuditReport:
    results: tuple[MetamorphicPairResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(item.passed for item in self.results)

    @property
    def total_pairs(self) -> int:
        return len(self.results)

    @property
    def passed_pairs(self) -> int:
        return sum(item.passed for item in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed_pairs / self.total_pairs if self.total_pairs else 0.0

    def to_dict(self) -> dict[str, Any]:
        finding_counts: dict[str, int] = {}
        for result in self.results:
            for finding in result.findings:
                code = str(finding.code)
                finding_counts[code] = finding_counts.get(code, 0) + 1
        return {
            "summary": {
                "decision": "accept" if self.passed else "reject",
                "total_pairs": self.total_pairs,
                "passed_pairs": self.passed_pairs,
                "pass_rate": round(self.pass_rate, 6),
                "finding_counts": dict(sorted(finding_counts.items())),
            },
            "results": [item.to_dict() for item in self.results],
        }


class _MalformedTrace(ValueError):
    pass


def audit_metamorphic_behavior(
    specs: Iterable[MetamorphicPairSpec],
    traces: Iterable[AgentTrace],
    policy: MetamorphicPolicy | None = None,
) -> MetamorphicAuditReport:
    """Audit paired traces under invariant or responsive relations.

    Malformed collections raise ``ValueError``. Missing or malformed trace evidence
    produces a rejected pair result so a release decision cannot pass open.
    """

    active_policy = policy or MetamorphicPolicy()
    pair_specs = tuple(specs)
    if not active_policy.min_pairs <= len(pair_specs) <= active_policy.max_pairs:
        raise ValueError(
            f"pair count must be in [{active_policy.min_pairs}, {active_policy.max_pairs}]"
        )
    _reject_duplicate_ids((item.pair_id for item in pair_specs), "pair_id")

    trace_items = tuple(traces)
    _reject_duplicate_ids((item.case_id for item in trace_items), "trace case_id")
    trace_by_id = {item.case_id: item for item in trace_items}

    results = tuple(
        _audit_pair(spec, trace_by_id, active_policy)
        for spec in sorted(pair_specs, key=lambda item: item.pair_id)
    )
    return MetamorphicAuditReport(results=results)


def _audit_pair(
    spec: MetamorphicPairSpec,
    trace_by_id: dict[str, AgentTrace],
    policy: MetamorphicPolicy,
) -> MetamorphicPairResult:
    baseline = trace_by_id.get(spec.baseline_case_id)
    transformed = trace_by_id.get(spec.transformed_case_id)
    findings: list[AuditFinding] = []
    if baseline is None:
        findings.append(
            AuditFinding(FindingCode.MISSING_BASELINE_TRACE, None, "baseline trace is missing")
        )
    if transformed is None:
        findings.append(
            AuditFinding(
                FindingCode.MISSING_TRANSFORMED_TRACE, None, "transformed trace is missing"
            )
        )
    if findings:
        return _result(spec, (), findings, _fingerprint(spec, None, None))

    assert baseline is not None and transformed is not None
    try:
        baseline_projection = _project_trace(baseline, policy)
        transformed_projection = _project_trace(transformed, policy)
    except _MalformedTrace:
        findings.append(
            AuditFinding(FindingCode.MALFORMED_TRACE, None, "trace violates the audit contract")
        )
        return _result(spec, (), findings, _fingerprint(spec, None, None))

    changed = _changed_dimensions(baseline_projection, transformed_projection)
    changed_set = frozenset(changed)
    if spec.relation is MetamorphicRelation.INVARIANT:
        for dimension in changed:
            findings.append(
                AuditFinding(
                    FindingCode.UNEXPECTED_BEHAVIOR_CHANGE,
                    dimension,
                    "invariant pair changed a protected behavior dimension",
                )
            )
    else:
        for dimension in sorted(spec.required_changes - changed_set, key=str):
            findings.append(
                AuditFinding(
                    FindingCode.REQUIRED_CHANGE_MISSING,
                    dimension,
                    "responsive pair did not change a required behavior dimension",
                )
            )
        for dimension in sorted(changed_set - spec.allowed_changes, key=str):
            findings.append(
                AuditFinding(
                    FindingCode.UNEXPECTED_BEHAVIOR_CHANGE,
                    dimension,
                    "responsive pair changed a behavior dimension outside its allowlist",
                )
            )

    fingerprint = _fingerprint(spec, baseline_projection, transformed_projection)
    return _result(spec, changed, findings, fingerprint)


def _result(
    spec: MetamorphicPairSpec,
    changed: tuple[BehaviorDimension, ...],
    findings: list[AuditFinding],
    fingerprint: str,
) -> MetamorphicPairResult:
    return MetamorphicPairResult(
        pair_id=spec.pair_id,
        relation=spec.relation,
        passed=not findings,
        changed_dimensions=changed,
        findings=tuple(findings),
        evidence_sha256=fingerprint,
    )


def _project_trace(trace: AgentTrace, policy: MetamorphicPolicy) -> dict[str, Any]:
    _validate_bounded_identifier(trace.case_id, "trace case_id", policy.max_identifier_chars)
    if len(trace.tool_calls) > policy.max_calls_per_trace:
        raise _MalformedTrace("tool-call budget exceeded")
    if len(trace.evidence_ids) > policy.max_evidence_ids:
        raise _MalformedTrace("evidence-ID budget exceeded")
    if type(trace.escalated) is not bool or not (
        trace.error is None or isinstance(trace.error, str)
    ):
        raise _MalformedTrace("invalid trace status")

    names: list[str] = []
    arguments: list[str] = []
    outcomes: list[bool] = []
    for call in trace.tool_calls:
        _validate_bounded_identifier(call.name, "tool name", policy.max_identifier_chars)
        if type(call.succeeded) is not bool:
            raise _MalformedTrace("invalid tool outcome")
        names.append(call.name)
        arguments.append(_canonical_arguments(call.arguments, policy))
        outcomes.append(call.succeeded)

    evidence: list[str] = []
    for evidence_id in trace.evidence_ids:
        _validate_bounded_identifier(evidence_id, "evidence ID", policy.max_identifier_chars)
        evidence.append(evidence_id)
    return {
        str(BehaviorDimension.TOOL_SEQUENCE): names,
        str(BehaviorDimension.TOOL_ARGUMENTS): arguments,
        str(BehaviorDimension.TOOL_OUTCOMES): outcomes,
        str(BehaviorDimension.EVIDENCE): sorted(evidence),
        str(BehaviorDimension.ESCALATION): trace.escalated,
        str(BehaviorDimension.ERROR_STATE): trace.error is not None,
    }


def _changed_dimensions(
    baseline: dict[str, Any], transformed: dict[str, Any]
) -> tuple[BehaviorDimension, ...]:
    changes: list[BehaviorDimension] = []
    same_sequence = (
        baseline[str(BehaviorDimension.TOOL_SEQUENCE)]
        == transformed[str(BehaviorDimension.TOOL_SEQUENCE)]
    )
    for dimension in BehaviorDimension:
        if dimension in {BehaviorDimension.TOOL_ARGUMENTS, BehaviorDimension.TOOL_OUTCOMES}:
            if not same_sequence:
                continue
        if baseline[str(dimension)] != transformed[str(dimension)]:
            changes.append(dimension)
    return tuple(changes)


def _canonical_arguments(arguments: Any, policy: MetamorphicPolicy) -> str:
    node_count = [0]
    _validate_json_value(arguments, policy, node_count, set(), 0)
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode("utf-8")) > policy.max_argument_bytes:
        raise _MalformedTrace("argument byte budget exceeded")
    return encoded


def _validate_json_value(
    value: Any,
    policy: MetamorphicPolicy,
    node_count: list[int],
    active_containers: set[int],
    depth: int,
) -> None:
    node_count[0] += 1
    if node_count[0] > policy.max_argument_nodes:
        raise _MalformedTrace("argument node budget exceeded")
    if depth > policy.max_argument_depth:
        raise _MalformedTrace("argument depth budget exceeded")
    if value is None or type(value) in {bool, str}:
        return
    if type(value) is int:
        if abs(value) > 2**53 - 1:
            raise _MalformedTrace("integer is outside interoperable JSON range")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise _MalformedTrace("non-finite number")
        return
    if type(value) not in {dict, list}:
        raise _MalformedTrace("non-JSON argument value")

    identity = id(value)
    if identity in active_containers:
        raise _MalformedTrace("cyclic argument value")
    active_containers.add(identity)
    try:
        if type(value) is dict:
            for key, child in value.items():
                if type(key) is not str:
                    raise _MalformedTrace("argument object keys must be strings")
                _validate_json_value(child, policy, node_count, active_containers, depth + 1)
        else:
            for child in value:
                _validate_json_value(child, policy, node_count, active_containers, depth + 1)
    finally:
        active_containers.remove(identity)


def _fingerprint(
    spec: MetamorphicPairSpec,
    baseline: dict[str, Any] | None,
    transformed: dict[str, Any] | None,
) -> str:
    payload = {
        "pair_id": spec.pair_id,
        "baseline_case_id": spec.baseline_case_id,
        "transformed_case_id": spec.transformed_case_id,
        "relation": str(spec.relation),
        "required_changes": sorted(map(str, spec.required_changes)),
        "allowed_changes": sorted(map(str, spec.allowed_changes)),
        "baseline": baseline,
        "transformed": transformed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_identifier(value: Any, name: str) -> None:
    _validate_bounded_identifier(value, name, 256)


def _validate_bounded_identifier(value: Any, name: str, max_chars: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > max_chars
        or any(ord(char) < 32 for char in value)
    ):
        error = f"{name} must be a non-empty bounded string without control characters"
        if name in {"pair_id", "baseline_case_id", "transformed_case_id"}:
            raise ValueError(error)
        raise _MalformedTrace(error)


def _reject_duplicate_ids(values: Iterable[str], name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {name}")
        seen.add(value)
