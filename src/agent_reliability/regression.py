from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from statistics import NormalDist
from typing import Any

from .models import EvaluationReport


@dataclass(frozen=True, slots=True)
class RegressionPolicy:
    max_mean_score_drop: float = 0.01
    max_pass_rate_drop: float = 0.0
    max_case_regressions: int = 0
    case_score_tolerance: float = 1e-9
    directional_significance_level: float | None = None
    directional_confidence_level: float = 0.95
    min_directional_changes: int = 5

    def __post_init__(self) -> None:
        if not 0 <= self.max_mean_score_drop <= 1:
            raise ValueError("max_mean_score_drop must be between 0 and 1")
        if not 0 <= self.max_pass_rate_drop <= 1:
            raise ValueError("max_pass_rate_drop must be between 0 and 1")
        if self.max_case_regressions < 0:
            raise ValueError("max_case_regressions must be non-negative")
        if self.case_score_tolerance < 0:
            raise ValueError("case_score_tolerance must be non-negative")
        if self.directional_significance_level is not None and not (
            0 < self.directional_significance_level < 1
        ):
            raise ValueError("directional_significance_level must be between 0 and 1")
        if not 0 < self.directional_confidence_level < 1:
            raise ValueError("directional_confidence_level must be between 0 and 1")
        if self.min_directional_changes < 1:
            raise ValueError("min_directional_changes must be positive")


@dataclass(frozen=True, slots=True)
class PairedRegressionEvidence:
    """Case-paired directional evidence, excluding unchanged score ties."""

    shared_cases: int
    tied_cases: int
    directional_changes: int
    regressed_cases: int
    improved_cases: int
    regression_share: float | None
    confidence_level: float
    regression_share_interval: tuple[float, float] | None
    one_sided_p_value: float
    significance_level: float | None
    minimum_directional_changes: int
    significant_regression: bool

    def to_dict(self) -> dict[str, Any]:
        interval = self.regression_share_interval
        return {
            "shared_cases": self.shared_cases,
            "tied_cases": self.tied_cases,
            "directional_changes": self.directional_changes,
            "regressed_cases": self.regressed_cases,
            "improved_cases": self.improved_cases,
            "regression_share": self.regression_share,
            "confidence_level": self.confidence_level,
            "regression_share_interval": (
                {"lower": interval[0], "upper": interval[1]} if interval else None
            ),
            "one_sided_p_value": self.one_sided_p_value,
            "significance_level": self.significance_level,
            "minimum_directional_changes": self.minimum_directional_changes,
            "significant_regression": self.significant_regression,
        }


@dataclass(frozen=True, slots=True)
class RegressionResult:
    passed: bool
    baseline_mean_score: float
    candidate_mean_score: float
    mean_score_delta: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_delta: float
    regressed_cases: tuple[str, ...]
    improved_cases: tuple[str, ...]
    new_cases: tuple[str, ...]
    missing_cases: tuple[str, ...]
    paired_evidence: PairedRegressionEvidence
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": {
                "baseline_mean_score": self.baseline_mean_score,
                "candidate_mean_score": self.candidate_mean_score,
                "mean_score_delta": self.mean_score_delta,
                "baseline_pass_rate": self.baseline_pass_rate,
                "candidate_pass_rate": self.candidate_pass_rate,
                "pass_rate_delta": self.pass_rate_delta,
            },
            "cases": {
                "regressed": list(self.regressed_cases),
                "improved": list(self.improved_cases),
                "new": list(self.new_cases),
                "missing": list(self.missing_cases),
            },
            "statistical_evidence": self.paired_evidence.to_dict(),
            "reasons": list(self.reasons),
        }


def _one_sided_sign_test(regressions: int, improvements: int) -> float:
    """Return P(X >= regressions) for X ~ Binomial(changes, 0.5)."""

    changes = regressions + improvements
    if changes == 0:
        return 1.0
    numerator = sum(comb(changes, count) for count in range(regressions, changes + 1))
    return numerator / (2**changes)


def _wilson_interval(successes: int, total: int, confidence: float) -> tuple[float, float] | None:
    if total == 0:
        return None
    probability = successes / total
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2)
    z_squared = z_score**2
    denominator = 1 + z_squared / total
    center = (probability + z_squared / (2 * total)) / denominator
    half_width = (
        z_score
        * sqrt(probability * (1 - probability) / total + z_squared / (4 * total**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _paired_evidence(
    *,
    shared_cases: int,
    regressions: int,
    improvements: int,
    policy: RegressionPolicy,
) -> PairedRegressionEvidence:
    changes = regressions + improvements
    p_value = _one_sided_sign_test(regressions, improvements)
    interval = _wilson_interval(regressions, changes, policy.directional_confidence_level)
    significance_level = policy.directional_significance_level
    significant = (
        significance_level is not None
        and changes >= policy.min_directional_changes
        and regressions > improvements
        and p_value <= significance_level
    )
    return PairedRegressionEvidence(
        shared_cases=shared_cases,
        tied_cases=shared_cases - changes,
        directional_changes=changes,
        regressed_cases=regressions,
        improved_cases=improvements,
        regression_share=round(regressions / changes, 6) if changes else None,
        confidence_level=policy.directional_confidence_level,
        regression_share_interval=(
            (round(interval[0], 6), round(interval[1], 6)) if interval else None
        ),
        one_sided_p_value=p_value,
        significance_level=significance_level,
        minimum_directional_changes=policy.min_directional_changes,
        significant_regression=significant,
    )


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    policy: RegressionPolicy | None = None,
) -> RegressionResult:
    active_policy = policy or RegressionPolicy()
    baseline_map = {item.case_id: item for item in baseline.results}
    candidate_map = {item.case_id: item for item in candidate.results}
    shared_ids = baseline_map.keys() & candidate_map.keys()
    missing = tuple(sorted(baseline_map.keys() - candidate_map.keys()))
    new = tuple(sorted(candidate_map.keys() - baseline_map.keys()))

    regressed = tuple(
        sorted(
            case_id
            for case_id in shared_ids
            if candidate_map[case_id].score
            < baseline_map[case_id].score - active_policy.case_score_tolerance
        )
    )
    improved = tuple(
        sorted(
            case_id
            for case_id in shared_ids
            if candidate_map[case_id].score
            > baseline_map[case_id].score + active_policy.case_score_tolerance
        )
    )
    mean_delta = candidate.mean_score - baseline.mean_score
    pass_delta = candidate.pass_rate - baseline.pass_rate
    evidence = _paired_evidence(
        shared_cases=len(shared_ids),
        regressions=len(regressed),
        improvements=len(improved),
        policy=active_policy,
    )
    reasons: list[str] = []
    if missing:
        reasons.append(f"candidate omitted {len(missing)} baseline case(s)")
    if mean_delta < -active_policy.max_mean_score_drop:
        reasons.append(
            f"mean score dropped {abs(mean_delta):.4f}; "
            f"allowed drop is {active_policy.max_mean_score_drop:.4f}"
        )
    if pass_delta < -active_policy.max_pass_rate_drop:
        reasons.append(
            f"pass rate dropped {abs(pass_delta):.4f}; "
            f"allowed drop is {active_policy.max_pass_rate_drop:.4f}"
        )
    if len(regressed) > active_policy.max_case_regressions:
        reasons.append(
            f"{len(regressed)} case(s) regressed; "
            f"allowed count is {active_policy.max_case_regressions}"
        )
    if evidence.significant_regression:
        reasons.append(
            "paired case directions indicate a significant regression "
            f"(one-sided p={evidence.one_sided_p_value:.6g}, "
            f"alpha={active_policy.directional_significance_level:.6g})"
        )

    return RegressionResult(
        passed=not reasons,
        baseline_mean_score=round(baseline.mean_score, 6),
        candidate_mean_score=round(candidate.mean_score, 6),
        mean_score_delta=round(mean_delta, 6),
        baseline_pass_rate=round(baseline.pass_rate, 6),
        candidate_pass_rate=round(candidate.pass_rate, 6),
        pass_rate_delta=round(pass_delta, 6),
        regressed_cases=regressed,
        improved_cases=improved,
        new_cases=new,
        missing_cases=missing,
        paired_evidence=evidence,
        reasons=tuple(reasons),
    )
