from __future__ import annotations

from dataclasses import replace
from enum import StrEnum

from .models import AgentTrace, ToolCall


class FaultType(StrEnum):
    DROP_EVIDENCE = "drop_evidence"
    CORRUPT_ARGUMENT = "corrupt_argument"
    INSERT_UNEXPECTED_TOOL = "insert_unexpected_tool"
    EXCEED_LATENCY = "exceed_latency"
    EXCEED_COST = "exceed_cost"
    FLIP_ESCALATION = "flip_escalation"
    UNHANDLED_ERROR = "unhandled_error"


def inject_fault(
    trace: AgentTrace,
    fault: FaultType,
    *,
    magnitude: float = 10.0,
) -> AgentTrace:
    """Return a deterministic faulty copy without mutating the source trace."""
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    if fault is FaultType.DROP_EVIDENCE:
        return replace(trace, evidence_ids=frozenset())
    if fault is FaultType.CORRUPT_ARGUMENT:
        if not trace.tool_calls:
            raise ValueError("cannot corrupt arguments on a trace without tool calls")
        first, *rest = trace.tool_calls
        arguments = dict(first.arguments)
        if arguments:
            key = sorted(arguments)[0]
            arguments[key] = f"CORRUPTED::{arguments[key]}"
        else:
            arguments["injected_fault"] = True
        return replace(trace, tool_calls=(replace(first, arguments=arguments), *rest))
    if fault is FaultType.INSERT_UNEXPECTED_TOOL:
        injected = ToolCall(name="fault_injection.unknown_tool", arguments={})
        return replace(trace, tool_calls=(*trace.tool_calls, injected))
    if fault is FaultType.EXCEED_LATENCY:
        return replace(trace, total_latency_ms=max(trace.total_latency_ms, 1.0) * magnitude)
    if fault is FaultType.EXCEED_COST:
        return replace(trace, total_cost_usd=max(trace.total_cost_usd, 0.001) * magnitude)
    if fault is FaultType.FLIP_ESCALATION:
        return replace(trace, escalated=not trace.escalated)
    if fault is FaultType.UNHANDLED_ERROR:
        return replace(trace, error="injected provider timeout")
    raise ValueError(f"unsupported fault: {fault}")


def inject_suite(
    traces: list[AgentTrace],
    fault: FaultType,
    *,
    every: int = 1,
    magnitude: float = 10.0,
) -> list[AgentTrace]:
    if every < 1:
        raise ValueError("every must be at least 1")
    return [
        inject_fault(trace, fault, magnitude=magnitude) if index % every == 0 else trace
        for index, trace in enumerate(traces)
    ]
