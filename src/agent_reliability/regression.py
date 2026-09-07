from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import EvaluationReport


@dataclass(frozen=True, slots=True)
class RegressionPolicy:
    max_mean_score_drop: float = 0.01
    max_pass_rate_drop: float = 0.0
    max_case_regressions: int = 0
    case_score_tolerance: float = 1e-9


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
            "reasons": list(self.reasons),
        }


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
        reasons=tuple(reasons),
    )
