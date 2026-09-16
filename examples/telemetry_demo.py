from __future__ import annotations

from dataclasses import asdict
from pprint import pprint

from agent_reliability.telemetry import LLMSpan, SLOTarget, evaluate_slos, summarize_trace


def main() -> None:
    values = [
        LLMSpan(
            trace_id="demo-trace",
            span_id="root",
            name="agent.run",
            start_ns=0,
            end_ns=1_480_000_000,
            attributes={"llm.prompt.version": "support-v4", "enduser.email": "demo@example.com"},
        ),
        LLMSpan(
            trace_id="demo-trace",
            span_id="model",
            parent_span_id="root",
            name="llm.chat",
            start_ns=100_000_000,
            end_ns=780_000_000,
            attributes={
                "gen_ai.operation.name": "chat",
                "llm.usage.input_tokens": 310,
                "llm.usage.output_tokens": 92,
                "llm.cost.usd": 0.011,
            },
        ),
        LLMSpan(
            trace_id="demo-trace",
            span_id="tool",
            parent_span_id="root",
            name="tool.search_documents",
            start_ns=820_000_000,
            end_ns=1_120_000_000,
            attributes={"gen_ai.operation.name": "execute_tool"},
        ),
    ]

    sanitized = [span.sanitized() for span in values]
    summary = summarize_trace(sanitized)
    slo = evaluate_slos(
        [summary],
        SLOTarget(availability=0.99, p95_latency_ms=2_000, max_mean_cost_usd=0.03),
    )

    print("Sanitized root attributes:")
    pprint(dict(sanitized[0].attributes))
    print("\nTrace summary:")
    pprint(asdict(summary))
    print("\nSLO evaluation:")
    pprint(asdict(slo))


if __name__ == "__main__":
    main()
