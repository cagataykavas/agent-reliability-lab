"""Synthetic gate regression matrix. Run against baseline and candidate via PYTHONPATH."""
import json

from agent_reliability.models import AgentTrace, EvaluationCase, ToolCall
from agent_reliability.scoring import evaluate_trace

rows = []
for name, case_args, trace_args, mandatory in [
    ('missed_review', {'should_escalate': True}, {}, True),
    ('missing_required_evidence', {'required_evidence': frozenset({'d'})}, {}, True),
    ('disallowed_expected_tool', {'expected_tools': (ToolCall('delete'),),
                                  'allowed_tools': frozenset({'read'})},
     {'tool_calls': (ToolCall('delete'),)}, True),
    ('unhandled_error', {}, {'error': 'synthetic'}, True),
    ('valid_empty_contract', {}, {}, False),
    ('valid_review', {'should_escalate': True}, {'escalated': True}, False),
    ('valid_evidence', {'required_evidence': frozenset({'d'})},
     {'evidence_ids': frozenset({'d'})}, False),
    ('valid_allowed_tool', {'expected_tools': (ToolCall('read'),),
                            'allowed_tools': frozenset({'read'})},
     {'tool_calls': (ToolCall('read'),)}, False),
    ('soft_latency_overrun', {'max_latency_ms': 1}, {'total_latency_ms': 2}, False),
    ('soft_answer_mismatch', {'reference_answer_contains': ('required fact',)}, {}, False),
]:
    result = evaluate_trace(EvaluationCase(name, 'synthetic', **case_args),
                            AgentTrace(name, '', **trace_args))
    rows.append({'scenario': name, 'mandatory_violation': mandatory,
                 'score': result.score, 'passed': result.passed,
                 'failures': list(result.failures),
                 'hard_failures': list(getattr(result, 'hard_failures', ()))})
print(json.dumps({'scope': 'Synthetic regression fixtures, not model or production metrics',
                  'mandatory_violations': sum(r['mandatory_violation'] for r in rows),
                  'mandatory_violations_accepted': sum(r['mandatory_violation'] and r['passed'] for r in rows),
                  'controls_passed': sum(not r['mandatory_violation'] and r['passed'] for r in rows),
                  'rows': rows}, indent=2))
