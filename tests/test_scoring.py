from agent_reliability.models import AgentTrace, EvaluationCase, FailureType, ToolCall
from agent_reliability.scoring import evaluate_suite, evaluate_trace


def test_perfect_trace_passes() -> None:
    case = EvaluationCase(
        case_id="one",
        prompt="lookup",
        expected_tools=(ToolCall("lookup", {"id": 7}),),
        required_evidence=frozenset({"record:7"}),
        allowed_tools=frozenset({"lookup"}),
        max_latency_ms=500,
        max_cost_usd=0.01,
        reference_answer_contains=("approved",),
    )
    trace = AgentTrace(
        case_id="one",
        final_answer="The request is approved.",
        tool_calls=(ToolCall("lookup", {"id": 7}),),
        evidence_ids=frozenset({"record:7"}),
        total_latency_ms=200,
        total_cost_usd=0.002,
    )
    result = evaluate_trace(case, trace)
    assert result.passed
    assert result.score == 1.0
    assert result.failures == ()


def test_failures_are_explainable() -> None:
    case = EvaluationCase(
        case_id="two",
        prompt="refund",
        expected_tools=(ToolCall("get_order", {"id": "O-1"}),),
        required_evidence=frozenset({"order:O-1"}),
        allowed_tools=frozenset({"get_order"}),
        max_cost_usd=0.01,
        should_escalate=True,
    )
    trace = AgentTrace(
        case_id="two",
        final_answer="done",
        tool_calls=(ToolCall("delete_order", {"id": "O-1"}),),
        total_cost_usd=0.03,
    )
    result = evaluate_trace(case, trace)
    assert not result.passed
    assert FailureType.MISSING_TOOL in result.failures
    assert FailureType.UNEXPECTED_TOOL in result.failures
    assert FailureType.MISSING_EVIDENCE in result.failures
    assert FailureType.BUDGET_EXCEEDED in result.failures
    assert FailureType.MISSING_ESCALATION in result.failures
    assert result.diagnostics


def test_missing_trace_is_reported() -> None:
    report = evaluate_suite([EvaluationCase(case_id="missing", prompt="x")], [])
    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert FailureType.UNHANDLED_ERROR in report.results[0].failures


def test_duplicate_ids_are_rejected() -> None:
    case = EvaluationCase(case_id="duplicate", prompt="x")
    try:
        evaluate_suite([case, case], [])
    except ValueError as error:
        assert "duplicate case ID" in str(error)
    else:
        raise AssertionError("duplicate ID should fail")
