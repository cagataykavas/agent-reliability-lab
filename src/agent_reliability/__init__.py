"""Deterministic evaluation primitives for tool-using AI agents."""

from .models import AgentTrace, EvaluationCase, EvaluationReport, ToolCall
from .scoring import evaluate_suite, evaluate_trace

__all__ = [
    "AgentTrace",
    "EvaluationCase",
    "EvaluationReport",
    "ToolCall",
    "evaluate_suite",
    "evaluate_trace",
]

__version__ = "0.1.0"
