import pytest

from agent_reliability.statistics import paired_bootstrap_mean_delta


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    baseline = [0.9, 0.7, 0.8, 1.0]
    candidate = [0.8, 0.6, 0.7, 0.9]

    first = paired_bootstrap_mean_delta(baseline, candidate, samples=500, seed=42)
    second = paired_bootstrap_mean_delta(baseline, candidate, samples=500, seed=42)

    assert first == second
    assert first.lower == pytest.approx(-0.1)
    assert first.upper == pytest.approx(-0.1)
    assert first.paired_cases == 4


def test_interval_captures_mixed_case_level_effects() -> None:
    result = paired_bootstrap_mean_delta(
        [0.9, 0.9, 0.9, 0.9],
        [0.7, 0.8, 0.9, 1.0],
        samples=2_000,
        seed=7,
    )

    assert result.lower < 0
    assert result.upper >= 0


@pytest.mark.parametrize(
    ("baseline", "candidate", "samples", "confidence"),
    [
        ([], [], 100, 0.95),
        ([1.0], [], 100, 0.95),
        ([1.0], [1.0], 0, 0.95),
        ([1.0], [1.0], 100, 1.0),
        ([float("nan")], [1.0], 100, 0.95),
    ],
)
def test_invalid_inputs_are_rejected(
    baseline: list[float],
    candidate: list[float],
    samples: int,
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_mean_delta(
            baseline,
            candidate,
            samples=samples,
            confidence_level=confidence,
        )
