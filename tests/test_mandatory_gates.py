from dataclasses import replace

import pytest

from agent_reliability.models import AgentTrace, EvaluationCase, FailureType, ToolCall
from agent_reliability.scoring import evaluate_trace


@pytest.mark.parametrize('threshold', [0.0, 0.8, 0.9])
@pytest.mark.parametrize('kind', ['escalation', 'evidence', 'allowlist', 'error'])
def test_mandatory_constraints_cannot_be_compensated(kind, threshold):
    case = EvaluationCase('case', 'synthetic')
    trace = AgentTrace('case', 'answer')
    if kind == 'escalation':
        case = replace(case, should_escalate=True)
    elif kind == 'evidence':
        case = replace(case, required_evidence=frozenset({'required'}))
    elif kind == 'allowlist':
        # Matching expectations must never override an explicit allowlist.
        case = replace(case, expected_tools=(ToolCall('delete'),), allowed_tools=frozenset({'read'}))
        trace = replace(trace, tool_calls=(ToolCall('delete'),))
    else:
        trace = replace(trace, error='failed')
    result = evaluate_trace(case, trace, pass_threshold=threshold)
    assert not result.passed
    assert result.hard_failures
    assert result.to_dict()['hard_failures']


def test_quality_score_is_retained_and_valid_case_passes():
    case = EvaluationCase('case', 'synthetic', should_escalate=True)
    result = evaluate_trace(case, AgentTrace('case', 'answer'))
    assert result.score == 0.9
    assert result.hard_failures == (FailureType.MISSING_ESCALATION,)
    assert evaluate_trace(case, AgentTrace('case', 'answer', escalated=True)).passed


@pytest.mark.parametrize('changes,trace_changes', [
    ({'max_latency_ms': 1}, {'total_latency_ms': 2}),
    ({'max_cost_usd': 1}, {'total_cost_usd': 2}),
    ({}, {'escalated': True}),
    ({'reference_answer_contains': ('fact',)}, {}),
])
def test_soft_quality_diagnostics_are_not_automatically_safety_failures(changes, trace_changes):
    result = evaluate_trace(EvaluationCase('case', 'synthetic', **changes),
                            AgentTrace('case', 'answer', **trace_changes))
    assert result.passed and result.failures and not result.hard_failures


def test_case_can_explicitly_make_budget_mandatory():
    case = EvaluationCase.from_dict({'case_id': 'case', 'prompt': 'x',
        'max_latency_ms': 1, 'hard_failure_types': ['budget_exceeded']})
    assert not evaluate_trace(case, AgentTrace('case', '', total_latency_ms=2)).passed
    with pytest.raises(ValueError):
        EvaluationCase.from_dict({'case_id': 'case', 'prompt': 'x', 'hard_failure_types': ['typo']})


def test_unexpected_but_allowed_extra_call_is_not_a_hard_failure():
    case = EvaluationCase('case', 'x', expected_tools=(ToolCall('read'),),
                          allowed_tools=frozenset({'read', 'search'}))
    trace = AgentTrace('case', '', tool_calls=(ToolCall('read'), ToolCall('search')))
    result = evaluate_trace(case, trace)
    assert result.passed and FailureType.UNEXPECTED_TOOL in result.failures
    assert not result.hard_failures


def test_default_policy_survives_json_roundtrip_and_empty_additional_policy():
    case = EvaluationCase.from_dict({'case_id': 'case', 'prompt': 'x',
                                     'should_escalate': True, 'hard_failure_types': []})
    result = evaluate_trace(case, AgentTrace.from_dict({'case_id': 'case'}))
    assert result.to_dict()['hard_failures'] == ['missing_escalation']
    assert not result.passed


def test_api_reports_mandatory_failure_even_with_passing_quality_score(tmp_path):
    from fastapi.testclient import TestClient

    from agent_reliability.api import create_app

    client = TestClient(create_app(tmp_path / 'runs.db'))
    response = client.post('/v1/evaluations', json={
        'suite_name': 'synthetic', 'candidate_name': 'test',
        'cases': [{'case_id': 'review', 'prompt': 'x', 'should_escalate': True}],
        'traces': [{'case_id': 'review', 'final_answer': 'done'}],
    })
    assert response.status_code == 201
    report = response.json()['report']
    assert report['summary']['passed_cases'] == 0
    assert report['results'][0]['score'] == 0.9
    assert report['results'][0]['hard_failures'] == ['missing_escalation']
