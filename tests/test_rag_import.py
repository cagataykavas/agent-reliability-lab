from copy import deepcopy

import pytest

from agent_reliability.rag_import import adapt_record
from agent_reliability.scoring import evaluate_trace


def example():
    record = {'id': 'agent:1', 'claim': 'public synthetic example', 'status': 'completed',
              'answer': {'verdict': 'SUPPORT', 'citations': [{'doc_id': 'd', 'sentence': 0}]},
              'trace': [{'action': {'tool': 'read', 'doc_id': 'd'},
                         'result': {'doc_id': 'd', 'sentences': [{'sentence': 0}]}}]}
    labels = {'claim_id': 1, 'predicted': 'SUPPORT', 'citations': [['d', 0]],
              'status': 'completed', 'label_correct': False, 'evidence_complete_correct': False,
              'wall_seconds_this_invocation': 0.5}
    return record, labels


def test_preserves_actual_calls_without_inventing_expectations_or_truth():
    raw, labels = example()
    before = deepcopy((raw, labels))
    case, trace, findings = adapt_record(raw, labels, candidate='agent')
    assert (raw, labels) == before
    assert case.expected_tools == () and case.reference_answer_contains == ()
    assert trace.tool_calls[0].arguments == {'doc_id': 'd'}
    assert trace.total_latency_ms == 500
    assert not findings and not evaluate_trace(case, trace).hard_failures


def test_abstention_is_not_an_unsafe_completion():
    raw, labels = example()
    raw['answer'] = {'verdict': 'NOT_ENOUGH_INFO', 'citations': []}
    labels.update(predicted='NOT_ENOUGH_INFO', citations=[])
    case, trace, findings = adapt_record(raw, labels, candidate='agent')
    assert not findings and not evaluate_trace(case, trace).hard_failures


def test_citation_membership_is_checked_separately_from_correctness():
    raw, labels = example()
    raw['trace'] = []
    _, _, findings = adapt_record(raw, labels, candidate='agent')
    assert findings == ['citation_not_in_recorded_read']


@pytest.mark.parametrize('field,value', [('claim_id', '../evil'), ('predicted', 'CONTRADICT'),
                                        ('citations', []), ('status', 'failed'),
                                        ('label_correct', 'true')])
def test_mismatched_or_malformed_labels_are_rejected(field, value):
    raw, labels = example()
    labels[field] = value
    with pytest.raises(ValueError):
        adapt_record(raw, labels, candidate='agent')


def test_directory_import_preserves_labels_hashes_and_reports(tmp_path):
    import hashlib
    import json

    from agent_reliability.rag_import import import_directory

    raw, labels = example()
    pair = {}
    for candidate in ('agent', 'fixed_top3'):
        record = deepcopy(raw)
        record['id'] = f'{candidate}:1'
        (tmp_path / f'{candidate}-1.json').write_text(json.dumps(record))
        pair[candidate] = deepcopy(labels)
    results = {'protocol': {'prior_exposure': 'dev'}, 'policy': 'recorded policy', 'cases': [pair]}
    source = json.dumps(results).encode()
    (tmp_path / 'results.json').write_bytes(source)
    imported = import_directory(tmp_path)
    assert imported['results_sha256'] == hashlib.sha256(source).hexdigest()
    assert imported['records'][0]['source_labels'] == labels
    assert imported['summary']['agent']['source_label_correct'] == 0
    assert imported['summary']['agent']['records_with_contract_findings'] == 0
    assert 'score' not in imported['records'][0]
    # Exported projection can be loaded using the existing schemas.
    from agent_reliability.models import AgentTrace, EvaluationCase
    EvaluationCase.from_dict(imported['records'][0]['case'])
    AgentTrace.from_dict(imported['records'][0]['trace'])
    results['cases'].append(pair)
    (tmp_path / 'results.json').write_text(json.dumps(results))
    with pytest.raises(ValueError, match='duplicate'):
        import_directory(tmp_path)
