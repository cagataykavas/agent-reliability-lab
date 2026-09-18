from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    lower: float
    upper: float
    confidence_level: float
    samples: int
    seed: int
    paired_cases: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "samples": self.samples,
            "seed": self.seed,
            "paired_cases": self.paired_cases,
        }


def paired_bootstrap_mean_delta(
    baseline_scores: list[float],
    candidate_scores: list[float],
    *,
    samples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 17,
) -> BootstrapInterval:
    """Percentile interval for the paired candidate-minus-baseline mean delta."""
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("baseline_scores and candidate_scores must have equal length")
    if not baseline_scores:
        raise ValueError("at least one paired score is required")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if any(not math.isfinite(value) for value in baseline_scores + candidate_scores):
        raise ValueError("scores must be finite")

    deltas = [
        candidate - baseline
        for baseline, candidate in zip(baseline_scores, candidate_scores, strict=True)
    ]
    rng = random.Random(seed)
    size = len(deltas)
    bootstrapped = sorted(
        sum(deltas[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples)
    )
    alpha = (1 - confidence_level) / 2
    return BootstrapInterval(
        lower=round(_quantile(bootstrapped, alpha), 6),
        upper=round(_quantile(bootstrapped, 1 - alpha), 6),
        confidence_level=confidence_level,
        samples=samples,
        seed=seed,
        paired_cases=size,
    )


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight
