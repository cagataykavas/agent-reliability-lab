"""Risk-stratified release gates for agent evaluation reports."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .models import CaseResult, EvaluationReport, FailureType

_SEGMENT_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


def _rate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return float(value)


@dataclass(frozen=True, slots=True)
class SegmentPolicy:
    """Release requirements for one evaluation segment."""

    min_cases: int = 1
    min_pass_rate: float = 1.0
    min_mean_score: float = 0.8
    blocked_failures: frozenset[FailureType | str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.min_cases, bool) or not isinstance(self.min_cases, int):
            raise TypeError("min_cases must be an integer")
        if self.min_cases < 1:
            raise ValueError("min_cases must be at least one")
        object.__setattr__(self, "min_pass_rate", _rate("min_pass_rate", self.min_pass_rate))
        object.__setattr__(self, "min_mean_score", _rate("min_mean_score", self.min_mean_score))
        try:
            normalized = frozenset(FailureType(item) for item in self.blocked_failures)
        except ValueError as exc:
            raise ValueError(f"unknown blocked failure type: {exc.args[0]}") from exc
        object.__setattr__(self, "blocked_failures", normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_cases": self.min_cases,
            "min_pass_rate": self.min_pass_rate,
            "min_mean_score": self.min_mean_score,
            "blocked_failures": sorted(str(item) for item in self.blocked_failures),
        }


class SegmentFindingCode(StrEnum):
    MISSING_REQUIRED_SEGMENT = "missing_required_segment"
    INSUFFICIENT_CASES = "insufficient_cases"
    PASS_RATE_BELOW_MINIMUM = "pass_rate_below_minimum"
    MEAN_SCORE_BELOW_MINIMUM = "mean_score_below_minimum"
    BLOCKED_FAILURE_PRESENT = "blocked_failure_present"


@dataclass(frozen=True, slots=True)
class SegmentFinding:
    code: SegmentFindingCode
    detail: str
    case_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "detail": self.detail,
            "case_ids": list(self.case_ids),
        }


@dataclass(frozen=True, slots=True)
class SegmentResult:
    name: str
    passed: bool
    total_cases: int
    passed_cases: int
    pass_rate: float
    mean_score: float
    failed_case_ids: tuple[str, ...]
    failure_counts: tuple[tuple[str, int], ...]
    policy: SegmentPolicy
    findings: tuple[SegmentFinding, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "pass_rate": self.pass_rate,
                "mean_score": self.mean_score,
            },
            "failed_case_ids": list(self.failed_case_ids),
            "failure_counts": dict(self.failure_counts),
            "policy": self.policy.as_dict(),
            "findings": [item.as_dict() for item in self.findings],
        }


@dataclass(frozen=True, slots=True)
class SegmentAuditReport:
    passed: bool
    evaluated_cases: int
    segments: tuple[SegmentResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "evaluated_cases": self.evaluated_cases,
            "segments": [item.as_dict() for item in self.segments],
        }


def audit_segments(
    report: EvaluationReport,
    assignments: Mapping[str, Collection[str]],
    policies: Mapping[str, SegmentPolicy],
    *,
    default_policy: SegmentPolicy | None = None,
) -> SegmentAuditReport:
    """Audit overlapping evaluation segments without hiding small risky slices.

    ``assignments`` maps every case ID in the report to one or more segment names.
    Explicit policies are also treated as required segments: a configured segment
    with no cases is rejected rather than silently omitted from the report.
    """

    case_map = _validated_cases(report)
    normalized_assignments = _validated_assignments(case_map, assignments)
    normalized_policies = _validated_policies(policies)
    if default_policy is not None and not isinstance(default_policy, SegmentPolicy):
        raise TypeError("default_policy must be a SegmentPolicy")

    by_segment: dict[str, list[CaseResult]] = {}
    for case_id, names in normalized_assignments.items():
        for name in names:
            by_segment.setdefault(name, []).append(case_map[case_id])

    observed_without_policy = by_segment.keys() - normalized_policies.keys()
    if observed_without_policy and default_policy is None:
        names = ", ".join(sorted(observed_without_policy))
        raise ValueError(f"no policy configured for observed segment(s): {names}")

    segment_names = sorted(by_segment.keys() | normalized_policies.keys())
    results = tuple(
        _audit_segment(
            name,
            sorted(by_segment.get(name, []), key=lambda item: item.case_id),
            normalized_policies.get(name, default_policy),
            required=name in normalized_policies,
        )
        for name in segment_names
    )
    return SegmentAuditReport(
        passed=all(item.passed for item in results),
        evaluated_cases=len(case_map),
        segments=results,
    )


def _validated_cases(report: EvaluationReport) -> dict[str, CaseResult]:
    output: dict[str, CaseResult] = {}
    for result in report.results:
        if not result.case_id:
            raise ValueError("report contains an empty case ID")
        if result.case_id in output:
            raise ValueError(f"report contains duplicate case ID: {result.case_id}")
        if not math.isfinite(result.score) or not 0 <= result.score <= 1:
            raise ValueError(f"case {result.case_id!r} has an invalid score")
        output[result.case_id] = result
    if not output:
        raise ValueError("report must contain at least one case")
    return output


def _validated_assignments(
    cases: Mapping[str, CaseResult],
    assignments: Mapping[str, Collection[str]],
) -> dict[str, tuple[str, ...]]:
    unknown = assignments.keys() - cases.keys()
    missing = cases.keys() - assignments.keys()
    if unknown:
        raise ValueError(f"assignments reference unknown case(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"cases have no segment assignment: {', '.join(sorted(missing))}")

    output: dict[str, tuple[str, ...]] = {}
    for case_id, raw_names in assignments.items():
        if isinstance(raw_names, (str, bytes)):
            raise TypeError(f"segments for case {case_id!r} must be a collection, not a string")
        names = tuple(sorted(set(raw_names)))
        if not names:
            raise ValueError(f"case {case_id!r} must belong to at least one segment")
        for name in names:
            _validate_segment_name(name)
        output[case_id] = names
    return output


def _validated_policies(policies: Mapping[str, SegmentPolicy]) -> dict[str, SegmentPolicy]:
    output: dict[str, SegmentPolicy] = {}
    for name, policy in policies.items():
        _validate_segment_name(name)
        if not isinstance(policy, SegmentPolicy):
            raise TypeError(f"policy for segment {name!r} must be a SegmentPolicy")
        output[name] = policy
    return output


def _validate_segment_name(name: str) -> None:
    if not isinstance(name, str) or not _SEGMENT_NAME.fullmatch(name):
        raise ValueError(
            "segment names must start with a lowercase letter and contain only "
            "lowercase letters, digits, '.', '_' or '-' (maximum 64 characters)"
        )


def _audit_segment(
    name: str,
    results: list[CaseResult],
    policy: SegmentPolicy | None,
    *,
    required: bool,
) -> SegmentResult:
    assert policy is not None
    total = len(results)
    passed_cases = sum(item.passed for item in results)
    pass_rate = passed_cases / total if total else 0.0
    mean_score = sum(item.score for item in results) / total if total else 0.0
    failed_ids = tuple(item.case_id for item in results if not item.passed)
    failure_counts = Counter(str(failure) for item in results for failure in item.failures)
    findings: list[SegmentFinding] = []

    if required and total == 0:
        findings.append(
            SegmentFinding(
                SegmentFindingCode.MISSING_REQUIRED_SEGMENT,
                "configured segment has no evaluation cases",
            )
        )
    if total < policy.min_cases:
        findings.append(
            SegmentFinding(
                SegmentFindingCode.INSUFFICIENT_CASES,
                f"observed {total} case(s); policy requires at least {policy.min_cases}",
            )
        )
    if pass_rate < policy.min_pass_rate:
        findings.append(
            SegmentFinding(
                SegmentFindingCode.PASS_RATE_BELOW_MINIMUM,
                f"pass rate {pass_rate:.6f} is below {policy.min_pass_rate:.6f}",
                failed_ids,
            )
        )
    if mean_score < policy.min_mean_score:
        findings.append(
            SegmentFinding(
                SegmentFindingCode.MEAN_SCORE_BELOW_MINIMUM,
                f"mean score {mean_score:.6f} is below {policy.min_mean_score:.6f}",
                tuple(item.case_id for item in results),
            )
        )

    blocked_case_ids = tuple(
        item.case_id for item in results if policy.blocked_failures.intersection(item.failures)
    )
    if blocked_case_ids:
        present = sorted(
            str(failure)
            for failure in policy.blocked_failures
            if any(failure in item.failures for item in results)
        )
        findings.append(
            SegmentFinding(
                SegmentFindingCode.BLOCKED_FAILURE_PRESENT,
                f"blocked failure type(s) observed: {', '.join(present)}",
                blocked_case_ids,
            )
        )

    return SegmentResult(
        name=name,
        passed=not findings,
        total_cases=total,
        passed_cases=passed_cases,
        pass_rate=round(pass_rate, 6),
        mean_score=round(mean_score, 6),
        failed_case_ids=failed_ids,
        failure_counts=tuple(sorted(failure_counts.items())),
        policy=policy,
        findings=tuple(findings),
    )
