"""Inter-rater agreement gates for agent-evaluation labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any


@dataclass(frozen=True, slots=True)
class Rating:
    case_id: str
    rater_id: str
    label: str

    def __post_init__(self) -> None:
        for field in ("case_id", "rater_id", "label"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            if value != value.strip() or len(value) > 200:
                raise ValueError(f"{field} must be trimmed and at most 200 characters")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Rating:
        required = {"case_id", "rater_id", "label"}
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError("rating must contain exactly case_id, rater_id and label")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class AgreementPolicy:
    min_shared_cases: int = 20
    min_overlap_ratio: float = 0.8
    min_kappa: float = 0.6

    def __post_init__(self) -> None:
        if isinstance(self.min_shared_cases, bool) or not isinstance(self.min_shared_cases, int):
            raise ValueError("min_shared_cases must be an integer")
        if self.min_shared_cases < 2:
            raise ValueError("min_shared_cases must be at least 2")
        for field in ("min_overlap_ratio", "min_kappa"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be numeric")
        if not 0.0 <= self.min_overlap_ratio <= 1.0:
            raise ValueError("min_overlap_ratio must be between 0 and 1")
        if not -1.0 <= self.min_kappa <= 1.0:
            raise ValueError("min_kappa must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class PairAgreement:
    raters: tuple[str, str]
    shared_cases: int
    union_cases: int
    overlap_ratio: float
    observed_agreement: float
    expected_agreement: float
    kappa: float | None
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["raters"] = list(self.raters)
        values["reasons"] = list(self.reasons)
        return values


@dataclass(frozen=True, slots=True)
class AgreementReport:
    accepted: bool
    total_ratings: int
    total_cases: int
    raters: tuple[str, ...]
    policy: AgreementPolicy
    pairs: tuple[PairAgreement, ...]
    disagreement_cases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        defined_kappas = [pair.kappa for pair in self.pairs if pair.kappa is not None]
        return {
            "accepted": self.accepted,
            "summary": {
                "total_ratings": self.total_ratings,
                "total_cases": self.total_cases,
                "rater_count": len(self.raters),
                "pair_count": len(self.pairs),
                "disagreement_case_count": len(self.disagreement_cases),
                "minimum_kappa": min(defined_kappas) if defined_kappas else None,
            },
            "raters": list(self.raters),
            "policy": asdict(self.policy),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "disagreement_cases": list(self.disagreement_cases),
        }


def _round(value: float) -> float:
    return round(value, 6)


def _evaluate_pair(
    first_id: str,
    second_id: str,
    first: dict[str, str],
    second: dict[str, str],
    policy: AgreementPolicy,
) -> PairAgreement:
    shared = sorted(first.keys() & second.keys())
    union_count = len(first.keys() | second.keys())
    shared_count = len(shared)
    overlap = shared_count / union_count if union_count else 0.0
    reasons: list[str] = []
    if shared_count < policy.min_shared_cases:
        reasons.append("shared_cases_below_minimum")
    if overlap < policy.min_overlap_ratio:
        reasons.append("overlap_ratio_below_minimum")

    if not shared:
        observed = 0.0
        expected = 0.0
        kappa = None
    else:
        first_counts = Counter(first[case_id] for case_id in shared)
        second_counts = Counter(second[case_id] for case_id in shared)
        labels = first_counts.keys() | second_counts.keys()
        observed = sum(first[case_id] == second[case_id] for case_id in shared) / shared_count
        expected = sum(
            (first_counts[label] / shared_count) * (second_counts[label] / shared_count)
            for label in labels
        )
        denominator = 1.0 - expected
        kappa = None if abs(denominator) < 1e-12 else (observed - expected) / denominator

    if kappa is None:
        reasons.append("kappa_undefined_prevalence_collapse")
    elif kappa < policy.min_kappa:
        reasons.append("kappa_below_minimum")
    return PairAgreement(
        raters=(first_id, second_id),
        shared_cases=shared_count,
        union_cases=union_count,
        overlap_ratio=_round(overlap),
        observed_agreement=_round(observed),
        expected_agreement=_round(expected),
        kappa=None if kappa is None else _round(kappa),
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def evaluate_agreement(
    ratings: Iterable[Rating], policy: AgreementPolicy | None = None
) -> AgreementReport:
    """Evaluate pairwise Cohen's kappa, overlap and label coverage for every rater pair."""

    active_policy = policy or AgreementPolicy()
    materialized = list(ratings)
    by_rater: dict[str, dict[str, str]] = {}
    by_case: dict[str, set[str]] = {}
    for rating in materialized:
        rater_cases = by_rater.setdefault(rating.rater_id, {})
        if rating.case_id in rater_cases:
            raise ValueError(
                f"duplicate rating for rater {rating.rater_id!r} and case {rating.case_id!r}"
            )
        rater_cases[rating.case_id] = rating.label
        by_case.setdefault(rating.case_id, set()).add(rating.label)
    raters = tuple(sorted(by_rater))
    if len(raters) < 2:
        raise ValueError("agreement evaluation requires at least two raters")

    pairs = tuple(
        _evaluate_pair(first, second, by_rater[first], by_rater[second], active_policy)
        for first, second in combinations(raters, 2)
    )
    disagreements = tuple(sorted(case_id for case_id, labels in by_case.items() if len(labels) > 1))
    return AgreementReport(
        accepted=all(pair.accepted for pair in pairs),
        total_ratings=len(materialized),
        total_cases=len(by_case),
        raters=raters,
        policy=active_policy,
        pairs=pairs,
        disagreement_cases=disagreements,
    )
