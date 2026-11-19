from dataclasses import replace

import pytest

from agent_reliability.telemetry import InvalidTelemetry, TraceSummary, trace_fingerprint
from agent_reliability.telemetry_manifest import (
    ProviderUsage,
    TelemetryManifest,
    admit_manifest,
    build_manifest,
)

REVISION = "a" * 40


def summary(trace_id: str, *, input_tokens: int = 10, cost: float = 0.01) -> TraceSummary:
    return TraceSummary(
        trace_id=trace_id,
        latency_ms=120.0,
        span_count=3,
        error_count=0,
        model_calls=1,
        tool_calls=1,
        input_tokens=input_tokens,
        output_tokens=4,
        total_cost_usd=cost,
        prompt_versions=("support-v1",),
    )


def test_manifest_admission_is_order_independent_and_provider_verified():
    items = [summary("trace-b", input_tokens=20), summary("trace-a")]
    manifest = build_manifest(
        items,
        exporter_name="otel-agent-exporter",
        exporter_version="1.4.2",
        source_revision=REVISION,
    )
    usage = ProviderUsage(input_tokens=30, output_tokens=8, total_cost_usd=0.02)

    first = admit_manifest(items, manifest, provider_usage=usage)
    second = admit_manifest(list(reversed(items)), manifest, provider_usage=usage)

    assert first == second
    assert first.trace_count == 2
    assert first.provider_usage_verified
    assert manifest.trace_fingerprints == tuple(sorted(manifest.trace_fingerprints))
    assert len(first.manifest_sha256) == 64
    assert len(first.evidence_sha256) == 64


def test_manifest_rejects_tampered_trace_or_claimed_aggregate():
    items = [summary("trace-a")]
    manifest = build_manifest(
        items,
        exporter_name="exporter",
        exporter_version="1.0.0",
        source_revision=REVISION,
    )

    with pytest.raises(InvalidTelemetry, match="fingerprints"):
        admit_manifest([replace(items[0], tool_calls=0)], manifest)
    with pytest.raises(InvalidTelemetry, match="input_tokens"):
        admit_manifest(items, replace(manifest, input_tokens=11))


def test_manifest_rejects_noncanonical_or_unbound_identity():
    items = [summary("trace-a"), summary("trace-b")]
    manifest = build_manifest(
        items,
        exporter_name="exporter",
        exporter_version="1.0.0",
        source_revision=REVISION,
    )

    with pytest.raises(InvalidTelemetry, match="canonical"):
        admit_manifest(
            items,
            replace(manifest, trace_fingerprints=tuple(reversed(manifest.trace_fingerprints))),
        )
    with pytest.raises(InvalidTelemetry, match="source_revision"):
        admit_manifest(items, replace(manifest, source_revision="main"))
    with pytest.raises(InvalidTelemetry, match="exporter_name"):
        admit_manifest(items, replace(manifest, exporter_name="bad exporter"))


def test_manifest_rejects_duplicate_or_malformed_summary_evidence():
    item = summary("trace-a")

    with pytest.raises(InvalidTelemetry, match="unique"):
        build_manifest(
            [item, item],
            exporter_name="exporter",
            exporter_version="1",
            source_revision=REVISION,
        )
    with pytest.raises(InvalidTelemetry, match="error_count"):
        build_manifest(
            [replace(item, error_count=4)],
            exporter_name="exporter",
            exporter_version="1",
            source_revision=REVISION,
        )
    with pytest.raises(InvalidTelemetry, match="prompt_versions"):
        build_manifest(
            [replace(item, prompt_versions=("z", "a"))],
            exporter_name="exporter",
            exporter_version="1",
            source_revision=REVISION,
        )


def test_manifest_rejects_provider_usage_mismatch_or_invalid_number():
    items = [summary("trace-a")]
    manifest = build_manifest(
        items,
        exporter_name="exporter",
        exporter_version="1",
        source_revision=REVISION,
    )

    with pytest.raises(InvalidTelemetry, match="provider input_tokens"):
        admit_manifest(
            items,
            manifest,
            provider_usage=ProviderUsage(11, 4, 0.01),
        )
    with pytest.raises(InvalidTelemetry, match="provider total_cost_usd"):
        admit_manifest(
            items,
            manifest,
            provider_usage=ProviderUsage(10, 4, float("nan")),
        )


def test_trace_fingerprint_covers_all_summary_fields():
    item = summary("trace-a")

    assert trace_fingerprint(item) != trace_fingerprint(replace(item, model_calls=0))
    assert trace_fingerprint(item) != trace_fingerprint(replace(item, tool_calls=0))


def test_manifest_requires_supported_schema_and_typed_counts():
    item = summary("trace-a")
    manifest = TelemetryManifest(
        schema_version=2,
        exporter_name="exporter",
        exporter_version="1",
        source_revision=REVISION,
        trace_fingerprints=(trace_fingerprint(item),),
        input_tokens=10,
        output_tokens=4,
        total_cost_usd=0.01,
    )

    with pytest.raises(InvalidTelemetry, match="schema version"):
        admit_manifest([item], manifest)
    with pytest.raises(InvalidTelemetry, match="schema version"):
        admit_manifest([item], replace(manifest, schema_version=True))
    with pytest.raises(InvalidTelemetry, match="non-negative integer"):
        admit_manifest([item], replace(manifest, schema_version=1, input_tokens=True))
