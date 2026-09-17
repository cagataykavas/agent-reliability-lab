import pytest

from agent_reliability.capacity import CapacityInput, estimate_capacity


def test_capacity_estimate_is_deterministic():
    values = CapacityInput(
        requests_per_day=1_000_000,
        spans_per_request=8,
        bytes_per_span=1500,
        retention_days=30,
        compression_ratio=0.4,
        replication_factor=2,
        peak_factor=4,
    )
    result = estimate_capacity(values)
    assert result.spans_per_day == 8_000_000
    assert result.peak_spans_per_second == pytest.approx(370.37037)
    assert result.retained_gb == pytest.approx(288.0)
    assert result.monthly_total_cost > 0


def test_invalid_compression_ratio_rejected():
    with pytest.raises(ValueError):
        estimate_capacity(CapacityInput(1, 1, 1, 1, compression_ratio=0))
