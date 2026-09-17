"""Capacity and TCO estimation for LLM observability workloads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityInput:
    requests_per_day: int
    spans_per_request: float
    bytes_per_span: int
    retention_days: int
    compression_ratio: float = 0.35
    replication_factor: int = 2
    peak_factor: float = 3.0
    storage_cost_per_gb_month: float = 0.03
    ingest_cost_per_million_spans: float = 0.20


@dataclass(frozen=True)
class CapacityEstimate:
    spans_per_day: float
    average_spans_per_second: float
    peak_spans_per_second: float
    retained_gb: float
    monthly_storage_cost: float
    monthly_ingest_cost: float
    monthly_total_cost: float


def estimate_capacity(values: CapacityInput) -> CapacityEstimate:
    """Estimate trace throughput, retained storage and a simple monthly TCO.

    This is deliberately vendor-neutral. Cost coefficients are explicit inputs so
    interview discussions can swap in ClickHouse/cloud/vendor assumptions.
    """
    if values.requests_per_day < 0 or values.spans_per_request < 0:
        raise ValueError("traffic values must be non-negative")
    if values.bytes_per_span < 0 or values.retention_days < 0:
        raise ValueError("storage values must be non-negative")
    if not 0 < values.compression_ratio <= 1:
        raise ValueError("compression_ratio must be in (0, 1]")
    if values.replication_factor < 1 or values.peak_factor < 1:
        raise ValueError("replication_factor and peak_factor must be >= 1")

    spans_day = values.requests_per_day * values.spans_per_request
    avg_sps = spans_day / 86_400
    peak_sps = avg_sps * values.peak_factor
    retained_bytes = (
        spans_day
        * values.bytes_per_span
        * values.compression_ratio
        * values.retention_days
        * values.replication_factor
    )
    retained_gb = retained_bytes / 1_000_000_000
    storage_cost = retained_gb * values.storage_cost_per_gb_month
    ingest_cost = spans_day * 30 / 1_000_000 * values.ingest_cost_per_million_spans
    total = storage_cost + ingest_cost
    return CapacityEstimate(
        spans_per_day=spans_day,
        average_spans_per_second=avg_sps,
        peak_spans_per_second=peak_sps,
        retained_gb=retained_gb,
        monthly_storage_cost=storage_cost,
        monthly_ingest_cost=ingest_cost,
        monthly_total_cost=total,
    )
