import math

import pytest

from agent_reliability.telemetry import (
    InvalidTelemetry,
    LLMSpan,
    RedactionPolicy,
    SLOTarget,
    evaluate_slos,
    redact_attributes,
    summarize_trace,
    trace_fingerprint,
)


def make_span(
    *,
    trace_id: str = "trace-1",
    span_id: str,
    name: str,
    start_ns: int,
    end_ns: int,
    status: str = "OK",
    attributes: dict | None = None,
) -> LLMSpan:
    return LLMSpan(
        trace_id=trace_id,
        span_id=span_id,
        name=name,
        start_ns=start_ns,
        end_ns=end_ns,
        status=status,
        attributes=attributes or {},
    )


def test_redaction_masks_sensitive_nested_attributes():
    values = {
        "service.name": "assistant",
        "enduser.email": "person@example.com",
        "http": {"authorization": "Bearer abc", "status": 200},
        "custom_api_key_hint": "secret-value",
    }

    cleaned = redact_attributes(values)

    assert cleaned["service.name"] == "assistant"
    assert cleaned["enduser.email"] == "[REDACTED]"
    assert cleaned["http"]["authorization"] == "[REDACTED]"
    assert cleaned["custom_api_key_hint"] == "[REDACTED]"


def test_redaction_policy_can_mask_domain_specific_keys():
    policy = RedactionPolicy(
        exact_keys=frozenset({"customer_number"}),
        key_fragments=(),
        safe_metric_keys=frozenset(),
    )
    cleaned = redact_attributes({"customer_number": "C-42", "region": "TR"}, policy)

    assert cleaned == {"customer_number": "[REDACTED]", "region": "TR"}


def test_redaction_preserves_aggregate_usage_but_masks_secret_tokens():
    cleaned = redact_attributes(
        {
            "llm.usage.input_tokens": 120,
            "llm.usage.output_tokens": 40,
            "llm.cost.usd": 0.006,
            "oauth_token": "secret",
        }
    )

    assert cleaned == {
        "llm.usage.input_tokens": 120,
        "llm.usage.output_tokens": 40,
        "llm.cost.usd": 0.006,
        "oauth_token": "[REDACTED]",
    }

    strict = RedactionPolicy(exact_keys=frozenset({"llm.cost.usd"}))
    assert redact_attributes({"llm.cost.usd": 0.006}, strict) == {"llm.cost.usd": "[REDACTED]"}


def test_trace_summary_aggregates_llm_tool_cost_and_tokens():
    spans = [
        make_span(
            span_id="root",
            name="agent.run",
            start_ns=0,
            end_ns=2_000_000_000,
            attributes={"llm.prompt.version": "support-v3"},
        ),
        make_span(
            span_id="model",
            name="llm.chat",
            start_ns=100_000_000,
            end_ns=900_000_000,
            attributes={
                "gen_ai.operation.name": "chat",
                "llm.usage.input_tokens": 120,
                "llm.usage.output_tokens": 40,
                "llm.cost.usd": 0.006,
            },
        ),
        make_span(
            span_id="tool",
            name="tool.search_docs",
            start_ns=1_000_000_000,
            end_ns=1_300_000_000,
            attributes={"gen_ai.operation.name": "execute_tool"},
        ),
    ]

    summary = summarize_trace(spans)

    assert summary.trace_id == "trace-1"
    assert summary.latency_ms == 2000.0
    assert summary.model_calls == 1
    assert summary.tool_calls == 1
    assert summary.input_tokens == 120
    assert summary.output_tokens == 40
    assert summary.total_cost_usd == 0.006
    assert summary.prompt_versions == ("support-v3",)


def test_slo_evaluation_flags_latency_and_error_budget():
    healthy = summarize_trace(
        [make_span(span_id="a", name="agent.run", start_ns=0, end_ns=1_000_000_000)]
    )
    failed = summarize_trace(
        [
            make_span(
                trace_id="trace-2",
                span_id="b",
                name="llm.chat",
                start_ns=0,
                end_ns=4_000_000_000,
                status="ERROR",
                attributes={"llm.cost.usd": 0.08},
            )
        ]
    )

    result = evaluate_slos(
        [healthy, failed],
        SLOTarget(availability=0.99, p95_latency_ms=2_500, max_mean_cost_usd=0.02),
    )

    assert result.availability == 0.5
    assert not result.availability_ok
    assert not result.latency_ok
    assert not result.cost_ok
    assert not result.passed


def test_fingerprint_is_deterministic():
    summary = summarize_trace(
        [make_span(span_id="a", name="agent.run", start_ns=0, end_ns=1_000_000)]
    )

    assert trace_fingerprint(summary) == trace_fingerprint(summary)


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"llm.usage.input_tokens": -1}, "input_tokens"),
        ({"llm.usage.output_tokens": "4"}, "output_tokens"),
        ({"llm.cost.usd": math.nan}, "cost"),
        ({"llm.cost.usd": math.inf}, "cost"),
        ({"llm.cost.usd": -0.01}, "cost"),
    ],
)
def test_trace_summary_rejects_invalid_usage_evidence(attributes, message):
    span = make_span(
        span_id="model",
        name="llm.chat",
        start_ns=0,
        end_ns=1,
        attributes=attributes,
    )

    with pytest.raises(InvalidTelemetry, match=message):
        summarize_trace([span])


def test_trace_summary_rejects_duplicate_spans_and_invalid_time():
    first = make_span(span_id="same", name="agent.run", start_ns=0, end_ns=1)
    duplicate = make_span(span_id="same", name="llm.chat", start_ns=1, end_ns=2)
    invalid_time = make_span(span_id="later", name="tool.call", start_ns=2, end_ns=1)

    with pytest.raises(InvalidTelemetry, match="duplicate span ID"):
        summarize_trace([first, duplicate])
    with pytest.raises(InvalidTelemetry, match="timestamp range"):
        summarize_trace([invalid_time])


def test_trace_summary_rejects_invalid_identity_and_aggregate_cost():
    invalid_identity = make_span(span_id="", name="agent.run", start_ns=0, end_ns=1)
    expensive = [
        make_span(
            span_id=str(index),
            name="llm.chat",
            start_ns=index,
            end_ns=index + 1,
            attributes={"llm.cost.usd": 1e308},
        )
        for index in range(2)
    ]

    with pytest.raises(InvalidTelemetry, match="span IDs"):
        summarize_trace([invalid_identity])
    with pytest.raises(InvalidTelemetry, match="aggregate llm.cost"):
        summarize_trace(expensive)


def test_slo_evaluation_rejects_invalid_targets_and_duplicate_summaries():
    summary = summarize_trace(
        [make_span(span_id="a", name="agent.run", start_ns=0, end_ns=1_000_000)]
    )

    with pytest.raises(InvalidTelemetry, match="availability target"):
        evaluate_slos([summary], SLOTarget(availability=1.1))
    with pytest.raises(InvalidTelemetry, match="unique"):
        evaluate_slos([summary, summary])
