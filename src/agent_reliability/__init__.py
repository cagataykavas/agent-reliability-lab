"""Deterministic evaluation primitives for tool-using AI agents."""

from .models import AgentTrace, EvaluationCase, EvaluationReport, ToolCall
from .scoring import evaluate_suite, evaluate_trace
from .segments import SegmentAuditReport, SegmentPolicy, audit_segments

__all__ = [
    "AgentTrace",
    "EvaluationCase",
    "EvaluationReport",
    "SegmentAuditReport",
    "SegmentPolicy",
    "ToolCall",
    "audit_segments",
    "evaluate_suite",
    "evaluate_trace",
]

__version__ = "0.3.0"
