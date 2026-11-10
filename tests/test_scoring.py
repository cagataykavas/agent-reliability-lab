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


def test_failed_expected_tool_cannot_satisfy_contract() -> None:
    case = EvaluationCase(
        case_id="failed",
        prompt="lookup",
        expected_tools=(ToolCall("lookup", {"id": 7}),),
        allowed_tools=frozenset({"lookup"}),
    )
    trace = AgentTrace(
        case_id="failed",
        final_answer="done",
        tool_calls=(ToolCall("lookup", {"id": 7}, succeeded=False),),
    )

    result = evaluate_trace(case, trace)

    assert not result.passed
    assert result.breakdown.tool_selection == 0.0
    assert FailureType.MISSING_TOOL in result.failures
    assert FailureType.TOOL_EXECUTION_FAILED in result.failures
    assert FailureType.TOOL_FAILURE_BUDGET_EXCEEDED in result.failures


def test_successful_retry_within_reviewed_failure_budget_passes() -> None:
    case = EvaluationCase(
        case_id="retry",
        prompt="lookup",
        expected_tools=(ToolCall("lookup", {"id": 7}),),
        allowed_tools=frozenset({"lookup"}),
        max_failed_tool_calls=1,
    )
    trace = AgentTrace(
        case_id="retry",
        final_answer="done",
        tool_calls=(
            ToolCall("lookup", {"id": 7}, succeeded=False),
            ToolCall("lookup", {"id": 7}),
        ),
    )

    result = evaluate_trace(case, trace)

    assert result.passed
    assert result.score == 1.0
    assert result.failures == ()


def test_successful_retry_over_default_failure_budget_is_hard_failure() -> None:
    case = EvaluationCase(
        case_id="retry",
        prompt="lookup",
        expected_tools=(ToolCall("lookup"),),
        allowed_tools=frozenset({"lookup"}),
    )
    trace = AgentTrace(
        case_id="retry",
        final_answer="done",
        tool_calls=(
            ToolCall("lookup", succeeded=False),
            ToolCall("lookup"),
        ),
    )

    result = evaluate_trace(case, trace)

    assert result.score == 0.9
    assert not result.passed
    assert result.failures == (FailureType.TOOL_FAILURE_BUDGET_EXCEEDED,)


def test_failed_disallowed_tool_is_still_unexpected_when_failure_is_budgeted() -> None:
    case = EvaluationCase(
        case_id="policy",
        prompt="lookup",
        expected_tools=(ToolCall("lookup"),),
        allowed_tools=frozenset({"lookup"}),
        max_failed_tool_calls=1,
    )
    trace = AgentTrace(
        case_id="policy",
        final_answer="done",
        tool_calls=(ToolCall("delete", succeeded=False), ToolCall("lookup")),
    )

    result = evaluate_trace(case, trace)

    assert not result.passed
    assert FailureType.UNEXPECTED_TOOL in result.failures
    assert "delete" in result.diagnostics[0]


def test_negative_tool_failure_budget_is_rejected() -> None:
    case = EvaluationCase(
        case_id="invalid",
        prompt="lookup",
        max_failed_tool_calls=-1,
    )

    try:
        evaluate_trace(case, AgentTrace(case_id="invalid", final_answer=""))
    except ValueError as error:
        assert "max_failed_tool_calls" in str(error)
    else:
        raise AssertionError("negative failure budget should be rejected")


def test_failure_budget_loads_from_case_contract() -> None:
    case = EvaluationCase.from_dict(
        {
            "case_id": "loaded",
            "prompt": "lookup",
            "max_failed_tool_calls": 2,
        }
    )

    assert case.max_failed_tool_calls == 2


def test_expected_tools_cannot_declare_failed_outcomes() -> None:
    case = EvaluationCase(
        case_id="invalid-outcome",
        prompt="lookup",
        expected_tools=(ToolCall("lookup", succeeded=False),),
    )

    try:
        evaluate_trace(case, AgentTrace(case_id="invalid-outcome", final_answer=""))
    except ValueError as error:
        assert "expected_tools must describe successful calls" in str(error)
    else:
        raise AssertionError("failed expected outcome should be rejected")


def test_duplicate_ids_are_rejected() -> None:
    case = EvaluationCase(case_id="duplicate", prompt="x")
    try:
        evaluate_suite([case, case], [])
    except ValueError as error:
        assert "duplicate case ID" in str(error)
    else:
        raise AssertionError("duplicate ID should fail")
