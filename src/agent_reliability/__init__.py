"""Deterministic evaluation primitives for tool-using AI agents."""

from .models import AgentTrace, EvaluationCase, EvaluationReport, ToolCall
from .progress import ProgressAuditReport, ProgressPolicy, audit_progress_trace
from .scoring import evaluate_suite, evaluate_trace

__all__ = [
    "AgentTrace",
    "EvaluationCase",
    "EvaluationReport",
    "ProgressAuditReport",
    "ProgressPolicy",
    "ToolCall",
    "audit_progress_trace",
    "evaluate_suite",
    "evaluate_trace",
]

__version__ = "0.3.0"
