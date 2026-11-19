from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

REDACTED = "[REDACTED]"
MAX_TIMESTAMP_NS = 2**63 - 1
DEFAULT_SAFE_METRIC_KEYS = frozenset(
    {
        "llm.cost.usd",
        "llm.usage.input_tokens",
        "llm.usage.output_tokens",
    }
)
DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "email",
        "phone",
        "prompt",
        "request.body",
        "response.body",
        "session_id",
        "user_id",
    }
)
DEFAULT_SENSITIVE_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "api_key",
    "credential",
    "pii",
    "email",
    "phone",
    "user_id",
)


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    exact_keys: frozenset[str] = DEFAULT_SENSITIVE_KEYS
    key_fragments: tuple[str, ...] = DEFAULT_SENSITIVE_FRAGMENTS
    replacement: str = REDACTED
    safe_metric_keys: frozenset[str] = DEFAULT_SAFE_METRIC_KEYS

    def should_redact(self, key: str) -> bool:
        normalized = key.strip().lower()
        if normalized in self.exact_keys:
            return True
        if normalized in self.safe_metric_keys:
            return False
        return any(fragment in normalized for fragment in self.key_fragments)


@dataclass(frozen=True, slots=True)
class LLMSpan:
    trace_id: str
    span_id: str
    name: str
    start_ns: int
    end_ns: int
    parent_span_id: str | None = None
    status: str = "OK"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LLMSpan:
        return cls(
            trace_id=str(values["trace_id"]),
            span_id=str(values["span_id"]),
            parent_span_id=_optional_str(values.get("parent_span_id")),
            name=str(values["name"]),
            start_ns=int(values["start_ns"]),
            end_ns=int(values["end_ns"]),
            status=str(values.get("status", "OK")).upper(),
            attributes=dict(values.get("attributes", {})),
        )

    @property
    def duration_ms(self) -> float:
        return max(self.end_ns - self.start_ns, 0) / 1_000_000

    def sanitized(self, policy: RedactionPolicy | None = None) -> LLMSpan:
        selected_policy = policy or RedactionPolicy()
        return LLMSpan(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            status=self.status,
            attributes=redact_attributes(self.attributes, selected_policy),
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["attributes"] = dict(self.attributes)
        return values


@dataclass(frozen=True, slots=True)
class TraceSummary:
    trace_id: str
    latency_ms: float
    span_count: int
    error_count: int
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    prompt_versions: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.error_count == 0


@dataclass(frozen=True, slots=True)
class SLOTarget:
    availability: float = 0.99
    p95_latency_ms: float = 2_000.0
    max_mean_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class SLOResult:
    total_traces: int
    successful_traces: int
    availability: float
    p95_latency_ms: float
    mean_cost_usd: float
    availability_ok: bool
    latency_ok: bool
    cost_ok: bool

    @property
    def passed(self) -> bool:
        return self.availability_ok and self.latency_ok and self.cost_ok


class InvalidTelemetry(ValueError):
    """Recorded telemetry cannot support a trustworthy summary."""


def redact_attributes(
    attributes: Mapping[str, Any],
    policy: RedactionPolicy | None = None,
) -> dict[str, Any]:
    selected_policy = policy or RedactionPolicy()
    return {
        str(key): _redact_value(str(key), value, selected_policy)
        for key, value in attributes.items()
    }


def summarize_trace(spans: Iterable[LLMSpan]) -> TraceSummary:
    items = tuple(spans)
    if not items:
        raise ValueError("cannot summarize an empty trace")
    _validate_spans(items)

    start_ns = min(item.start_ns for item in items)
    end_ns = max(item.end_ns for item in items)
    attributes = [item.attributes for item in items]

    input_tokens = sum(_usage_int_attribute(item, "llm.usage.input_tokens") for item in attributes)
    output_tokens = sum(
        _usage_int_attribute(item, "llm.usage.output_tokens") for item in attributes
    )
    try:
        total_cost_usd = math.fsum(_cost_attribute(item) for item in attributes)
    except OverflowError as exc:
        raise InvalidTelemetry("aggregate llm.cost.usd is not finite") from exc
    if not math.isfinite(total_cost_usd):
        raise InvalidTelemetry("aggregate llm.cost.usd is not finite")
    prompt_versions = tuple(
        sorted(
            {
                str(item["llm.prompt.version"])
                for item in attributes
                if item.get("llm.prompt.version") not in {None, ""}
            }
        )
    )

    return TraceSummary(
        trace_id=items[0].trace_id,
        latency_ms=(end_ns - start_ns) / 1_000_000,
        span_count=len(items),
        error_count=sum(item.status not in {"OK", "UNSET"} for item in items),
        model_calls=sum(_is_model_span(item) for item in items),
        tool_calls=sum(_is_tool_span(item) for item in items),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=round(total_cost_usd, 8),
        prompt_versions=prompt_versions,
    )


def evaluate_slos(
    summaries: Iterable[TraceSummary],
    target: SLOTarget | None = None,
) -> SLOResult:
    items = tuple(summaries)
    if not items:
        raise ValueError("cannot evaluate SLOs without traces")

    selected_target = target or SLOTarget()
    _validate_slo_inputs(items, selected_target)
    successful = sum(item.succeeded for item in items)
    availability = successful / len(items)
    p95 = percentile([item.latency_ms for item in items], 0.95)
    try:
        mean_cost = math.fsum(item.total_cost_usd for item in items) / len(items)
    except OverflowError as exc:
        raise InvalidTelemetry("aggregate SLO cost is not finite") from exc
    if not math.isfinite(mean_cost):
        raise InvalidTelemetry("aggregate SLO cost is not finite")
    cost_ok = (
        True
        if selected_target.max_mean_cost_usd is None
        else mean_cost <= selected_target.max_mean_cost_usd
    )

    return SLOResult(
        total_traces=len(items),
        successful_traces=successful,
        availability=round(availability, 6),
        p95_latency_ms=round(p95, 6),
        mean_cost_usd=round(mean_cost, 8),
        availability_ok=availability >= selected_target.availability,
        latency_ok=p95 <= selected_target.p95_latency_ms,
        cost_ok=cost_ok,
    )


def trace_fingerprint(summary: TraceSummary) -> str:
    payload = json.dumps(
        asdict(summary),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def percentile(values: Iterable[float], quantile: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if len(items) == 1:
        return items[0]

    position = quantile * (len(items) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return items[lower_index]

    weight = position - lower_index
    return items[lower_index] * (1 - weight) + items[upper_index] * weight


def _redact_value(key: str, value: Any, policy: RedactionPolicy) -> Any:
    if policy.should_redact(key):
        return policy.replacement
    if isinstance(value, Mapping):
        return redact_attributes(value, policy)
    if isinstance(value, list):
        return [
            redact_attributes(item, policy) if isinstance(item, Mapping) else item for item in value
        ]
    return value


def _is_model_span(span: LLMSpan) -> bool:
    kind = str(span.attributes.get("gen_ai.operation.name", "")).lower()
    return kind in {"chat", "completion", "embeddings", "invoke_model"} or span.name.startswith(
        "llm."
    )


def _is_tool_span(span: LLMSpan) -> bool:
    kind = str(span.attributes.get("gen_ai.operation.name", "")).lower()
    return kind in {"execute_tool", "tool"} or span.name.startswith("tool.")


def _validate_spans(items: tuple[LLMSpan, ...]) -> None:
    trace_id: str | None = None
    span_ids: set[str] = set()
    for item in items:
        if not isinstance(item.trace_id, str) or not item.trace_id.strip():
            raise InvalidTelemetry("all spans must have one non-empty trace ID")
        if trace_id is None:
            trace_id = item.trace_id
        elif item.trace_id != trace_id:
            raise InvalidTelemetry("all spans must have one non-empty trace ID")
        if not isinstance(item.span_id, str) or not item.span_id.strip():
            raise InvalidTelemetry("span IDs must be non-empty")
        if item.span_id in span_ids:
            raise InvalidTelemetry(f"duplicate span ID: {item.span_id}")
        span_ids.add(item.span_id)
        if not isinstance(item.name, str) or not item.name.strip():
            raise InvalidTelemetry(f"{item.span_id}: span name must be non-empty")
        if (
            isinstance(item.start_ns, bool)
            or isinstance(item.end_ns, bool)
            or not isinstance(item.start_ns, int)
            or not isinstance(item.end_ns, int)
            or item.start_ns < 0
            or item.end_ns < item.start_ns
            or item.end_ns > MAX_TIMESTAMP_NS
        ):
            raise InvalidTelemetry(f"{item.span_id}: invalid monotonic timestamp range")
        if item.parent_span_id is not None and not isinstance(item.parent_span_id, str):
            raise InvalidTelemetry(f"{item.span_id}: parent span ID must be a string")
        if item.parent_span_id == item.span_id:
            raise InvalidTelemetry(f"{item.span_id}: span cannot parent itself")
        if not isinstance(item.status, str) or item.status not in {"OK", "UNSET", "ERROR"}:
            raise InvalidTelemetry(f"{item.span_id}: unsupported span status {item.status!r}")
        if not isinstance(item.attributes, Mapping):
            raise InvalidTelemetry(f"{item.span_id}: attributes must be a mapping")
        _usage_int_attribute(item.attributes, "llm.usage.input_tokens")
        _usage_int_attribute(item.attributes, "llm.usage.output_tokens")
        _cost_attribute(item.attributes)


def _usage_int_attribute(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidTelemetry(f"{key} must be a non-negative integer")
    return value


def _cost_attribute(values: Mapping[str, Any]) -> float:
    value = values.get("llm.cost.usd", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTelemetry("llm.cost.usd must be a non-negative finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise InvalidTelemetry("llm.cost.usd must be a non-negative finite number")
    return number


def _validate_slo_inputs(items: tuple[TraceSummary, ...], target: SLOTarget) -> None:
    if not _is_finite_number(target.availability) or not 0 <= target.availability <= 1:
        raise InvalidTelemetry("availability target must be finite and between 0 and 1")
    if not _is_finite_number(target.p95_latency_ms) or target.p95_latency_ms < 0:
        raise InvalidTelemetry("latency target must be a non-negative finite number")
    if target.max_mean_cost_usd is not None and (
        not _is_finite_number(target.max_mean_cost_usd) or target.max_mean_cost_usd < 0
    ):
        raise InvalidTelemetry("cost target must be a non-negative finite number")
    trace_ids: set[str] = set()
    for item in items:
        if (
            not isinstance(item.trace_id, str)
            or not item.trace_id.strip()
            or item.trace_id in trace_ids
        ):
            raise InvalidTelemetry("SLO inputs require unique, non-empty trace IDs")
        trace_ids.add(item.trace_id)
        if not _is_finite_number(item.latency_ms) or item.latency_ms < 0:
            raise InvalidTelemetry(f"{item.trace_id}: invalid latency summary")
        if not _is_finite_number(item.total_cost_usd) or item.total_cost_usd < 0:
            raise InvalidTelemetry(f"{item.trace_id}: invalid cost summary")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
