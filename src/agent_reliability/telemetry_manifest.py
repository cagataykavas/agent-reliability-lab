from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from .telemetry import InvalidTelemetry, TraceSummary, trace_fingerprint

MANIFEST_SCHEMA_VERSION = 1
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int
    total_cost_usd: float


@dataclass(frozen=True, slots=True)
class TelemetryManifest:
    schema_version: int
    exporter_name: str
    exporter_version: str
    source_revision: str
    trace_fingerprints: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    total_cost_usd: float


@dataclass(frozen=True, slots=True)
class TelemetryAdmissionReceipt:
    manifest_sha256: str
    evidence_sha256: str
    schema_version: int
    exporter_name: str
    exporter_version: str
    source_revision: str
    trace_count: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    provider_usage_verified: bool


def build_manifest(
    summaries: tuple[TraceSummary, ...] | list[TraceSummary],
    *,
    exporter_name: str,
    exporter_version: str,
    source_revision: str,
) -> TelemetryManifest:
    """Build a canonical manifest from admitted trace summaries.

    The returned object is deterministic for the same evidence set, regardless of
    input order. ``admit_manifest`` must still run at the trust boundary: building a
    manifest is not proof that the exporter or source revision is authentic.
    """
    items = tuple(summaries)
    aggregates = _validate_and_aggregate_summaries(items)
    manifest = TelemetryManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        exporter_name=exporter_name,
        exporter_version=exporter_version,
        source_revision=source_revision,
        trace_fingerprints=aggregates["trace_fingerprints"],
        input_tokens=aggregates["input_tokens"],
        output_tokens=aggregates["output_tokens"],
        total_cost_usd=aggregates["total_cost_usd"],
    )
    _validate_manifest(manifest)
    return manifest


def admit_manifest(
    summaries: tuple[TraceSummary, ...] | list[TraceSummary],
    manifest: TelemetryManifest,
    *,
    provider_usage: ProviderUsage | None = None,
) -> TelemetryAdmissionReceipt:
    """Fail closed unless a canonical manifest exactly matches trace evidence.

    When independent provider usage is supplied, its token and cost totals must also
    match. The receipt contains identities and aggregates only; it does not retain
    trace IDs, prompt versions, spans, or provider payloads.
    """
    items = tuple(summaries)
    aggregates = _validate_and_aggregate_summaries(items)
    _validate_manifest(manifest)

    if manifest.trace_fingerprints != aggregates["trace_fingerprints"]:
        raise InvalidTelemetry("manifest trace fingerprints do not match evidence")
    for field_name in ("input_tokens", "output_tokens", "total_cost_usd"):
        if getattr(manifest, field_name) != aggregates[field_name]:
            raise InvalidTelemetry(f"manifest {field_name} does not match evidence")

    provider_verified = provider_usage is not None
    if provider_usage is not None:
        _validate_provider_usage(provider_usage)
        for field_name in ("input_tokens", "output_tokens", "total_cost_usd"):
            if getattr(provider_usage, field_name) != aggregates[field_name]:
                raise InvalidTelemetry(f"provider {field_name} does not match evidence")

    evidence_payload = {
        "trace_fingerprints": aggregates["trace_fingerprints"],
        "input_tokens": aggregates["input_tokens"],
        "output_tokens": aggregates["output_tokens"],
        "total_cost_usd": aggregates["total_cost_usd"],
    }
    return TelemetryAdmissionReceipt(
        manifest_sha256=_digest(asdict(manifest)),
        evidence_sha256=_digest(evidence_payload),
        schema_version=manifest.schema_version,
        exporter_name=manifest.exporter_name,
        exporter_version=manifest.exporter_version,
        source_revision=manifest.source_revision,
        trace_count=len(items),
        input_tokens=aggregates["input_tokens"],
        output_tokens=aggregates["output_tokens"],
        total_cost_usd=aggregates["total_cost_usd"],
        provider_usage_verified=provider_verified,
    )


def _validate_and_aggregate_summaries(items: tuple[TraceSummary, ...]) -> dict[str, Any]:
    if not items:
        raise InvalidTelemetry("manifest admission requires at least one trace summary")

    trace_ids: set[str] = set()
    fingerprints: list[str] = []
    for item in items:
        if not isinstance(item, TraceSummary):
            raise InvalidTelemetry("manifest evidence must contain TraceSummary values")
        if not isinstance(item.trace_id, str) or not item.trace_id.strip():
            raise InvalidTelemetry("trace summaries require non-empty trace IDs")
        if item.trace_id in trace_ids:
            raise InvalidTelemetry("trace summaries require unique trace IDs")
        trace_ids.add(item.trace_id)
        _require_non_negative_finite(item.latency_ms, "latency_ms")
        for field_name in (
            "span_count",
            "error_count",
            "model_calls",
            "tool_calls",
            "input_tokens",
            "output_tokens",
        ):
            _require_non_negative_integer(getattr(item, field_name), field_name)
        if item.error_count > item.span_count:
            raise InvalidTelemetry("error_count cannot exceed span_count")
        if item.model_calls > item.span_count or item.tool_calls > item.span_count:
            raise InvalidTelemetry("model_calls and tool_calls cannot exceed span_count")
        _require_non_negative_finite(item.total_cost_usd, "total_cost_usd")
        if (
            not isinstance(item.prompt_versions, tuple)
            or item.prompt_versions != tuple(sorted(set(item.prompt_versions)))
            or any(not isinstance(value, str) or not value for value in item.prompt_versions)
        ):
            raise InvalidTelemetry("prompt_versions must be canonical non-empty strings")
        fingerprints.append(trace_fingerprint(item))

    try:
        total_cost = math.fsum(item.total_cost_usd for item in items)
    except OverflowError as exc:
        raise InvalidTelemetry("aggregate manifest cost is not finite") from exc
    if not math.isfinite(total_cost):
        raise InvalidTelemetry("aggregate manifest cost is not finite")

    return {
        "trace_fingerprints": tuple(sorted(fingerprints)),
        "input_tokens": sum(item.input_tokens for item in items),
        "output_tokens": sum(item.output_tokens for item in items),
        "total_cost_usd": round(total_cost, 8),
    }


def _validate_manifest(manifest: TelemetryManifest) -> None:
    if not isinstance(manifest, TelemetryManifest):
        raise InvalidTelemetry("manifest must be a TelemetryManifest")
    if (
        isinstance(manifest.schema_version, bool)
        or not isinstance(manifest.schema_version, int)
        or manifest.schema_version != MANIFEST_SCHEMA_VERSION
    ):
        raise InvalidTelemetry(f"unsupported manifest schema version: {manifest.schema_version!r}")
    for field_name in ("exporter_name", "exporter_version"):
        value = getattr(manifest, field_name)
        if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
            raise InvalidTelemetry(f"manifest {field_name} is invalid")
    if not isinstance(manifest.source_revision, str) or not REVISION_PATTERN.fullmatch(
        manifest.source_revision
    ):
        raise InvalidTelemetry("manifest source_revision must be a lowercase Git SHA-1")
    if (
        not isinstance(manifest.trace_fingerprints, tuple)
        or not manifest.trace_fingerprints
        or manifest.trace_fingerprints != tuple(sorted(set(manifest.trace_fingerprints)))
        or any(
            not isinstance(value, str) or not FINGERPRINT_PATTERN.fullmatch(value)
            for value in manifest.trace_fingerprints
        )
    ):
        raise InvalidTelemetry("manifest trace_fingerprints must be canonical SHA-256 values")
    _require_non_negative_integer(manifest.input_tokens, "manifest input_tokens")
    _require_non_negative_integer(manifest.output_tokens, "manifest output_tokens")
    _require_non_negative_finite(manifest.total_cost_usd, "manifest total_cost_usd")


def _validate_provider_usage(usage: ProviderUsage) -> None:
    if not isinstance(usage, ProviderUsage):
        raise InvalidTelemetry("provider_usage must be ProviderUsage")
    _require_non_negative_integer(usage.input_tokens, "provider input_tokens")
    _require_non_negative_integer(usage.output_tokens, "provider output_tokens")
    _require_non_negative_finite(usage.total_cost_usd, "provider total_cost_usd")


def _require_non_negative_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidTelemetry(f"{field_name} must be a non-negative integer")


def _require_non_negative_finite(value: Any, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise InvalidTelemetry(f"{field_name} must be a non-negative finite number")


def _digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()
