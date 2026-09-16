from agent_reliability.telemetry import (
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
    policy = RedactionPolicy(exact_keys=frozenset({"customer_number"}), key_fragments=())
    cleaned = redact_attributes({"customer_number": "C-42", "region": "TR"}, policy)

    assert cleaned == {"customer_number": "[REDACTED]", "region": "TR"}


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
