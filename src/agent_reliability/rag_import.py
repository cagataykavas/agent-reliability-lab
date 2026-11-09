"""Read-only import of recorded rag-document-intelligence development experiments.

This is not a model runner, semantic judge, or source of trusted outcome labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import AgentTrace, EvaluationCase, ToolCall
from .scoring import evaluate_trace

LIMITATIONS = [
    'Imported historical development runs, not new model executions or held-out estimates.',
    'Raw source labels are preserved, not independently adjudicated.',
    'Citation membership is structural linkage, not entailment or factual truth.',
    'No gold tool trajectory, human-review policy, cost measurement or latency ceiling is supplied.',
    'Generic weighted scores are omitted: absent gold trajectories make them unsuitable for quality comparison.',
    'Contract checks passing does not imply answer correctness or general safety.',
]


def _read(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f'{path.name}: expected object')
    return value, hashlib.sha256(data).hexdigest()


def adapt_record(record: dict[str, Any], labels: dict[str, Any], *,
                 candidate: str) -> tuple[EvaluationCase, AgentTrace, list[str]]:
    """Project actual calls; never manufacture expected calls from observed calls."""
    claim_id = labels['claim_id']
    if not isinstance(claim_id, int) or isinstance(claim_id, bool):
        raise ValueError('claim_id must be an integer')
    if record['id'] != f'{candidate}:{claim_id}':
        raise ValueError('source trace/label identity mismatch')
    answer = record.get('answer') or {}
    if answer.get('verdict') != labels.get('predicted'):
        raise ValueError('source trace/label prediction mismatch')
    citations = answer.get('citations', [])
    pairs = [[str(item['doc_id']), item['sentence']] for item in citations]
    if pairs != labels.get('citations', []):
        raise ValueError('source trace/label citation mismatch')
    if record['status'] != labels['status']:
        raise ValueError('source trace/label status mismatch')
    for name in ('label_correct', 'evidence_complete_correct'):
        if type(labels.get(name)) is not bool:
            raise ValueError(f'{name} must be a source boolean')

    calls = []
    available = set()
    findings = []
    for step in record['trace']:
        action = step.get('action')
        if not action:
            continue  # rejected model text is not an executed tool call
        name = action['tool']
        if name == 'finish':
            continue  # final decision, not an external tool invocation
        if 'result' not in step:
            continue  # proposed but not recorded as executed
        result = step['result']
        succeeded = not step.get('error') and not (
            isinstance(result, dict) and result.get('error'))
        calls.append(ToolCall(name, {k: v for k, v in action.items() if k != 'tool'},
                              succeeded=bool(succeeded)))
        if name == 'read' and succeeded and isinstance(result, dict):
            available.update((str(result['doc_id']), item['sentence'])
                             for item in result.get('sentences', []))
    if any((doc, sentence) not in available for doc, sentence in pairs):
        findings.append('citation_not_in_recorded_read')
    if answer.get('verdict') == 'NOT_ENOUGH_INFO' and citations:
        findings.append('abstention_has_citations')
    if answer.get('verdict') in {'SUPPORT', 'CONTRADICT'} and not citations:
        findings.append('non_abstention_missing_citation')
    case = EvaluationCase(record['id'], record['claim'],
                          allowed_tools=frozenset({'search', 'read'}))
    trace = AgentTrace(
        record['id'], str(answer.get('verdict', '')), tool_calls=tuple(calls),
        evidence_ids=frozenset(f'{doc}:{sentence}' for doc, sentence in pairs),
        total_latency_ms=float(labels['wall_seconds_this_invocation']) * 1000,
        error=None if record['status'] == 'completed' else f"source status: {record['status']}",
    )
    return case, trace, findings


def import_directory(directory: str | Path) -> dict[str, Any]:
    source = Path(directory)
    results, results_hash = _read(source / 'results.json')
    records = []
    seen = set()
    for pair in results['cases']:
        for candidate in ('agent', 'fixed_top3'):
            labels = pair[candidate]
            claim_id = labels['claim_id']
            if not isinstance(claim_id, int) or isinstance(claim_id, bool):
                raise ValueError('claim_id must be an integer')
            filename = f'{candidate}-{claim_id}.json'
            raw, digest = _read(source / filename)
            case, trace, findings = adapt_record(raw, labels, candidate=candidate)
            if case.case_id in seen:
                raise ValueError(f'duplicate source ID: {case.case_id}')
            seen.add(case.case_id)
            result = evaluate_trace(case, trace)
            case_dict = asdict(case)
            for field in ('required_evidence', 'allowed_tools', 'hard_failure_types'):
                case_dict[field] = sorted(case_dict[field])
            records.append({
                'source_file': filename, 'source_sha256': digest, 'candidate': candidate,
                'case': case_dict, 'trace': trace.to_dict(), 'source_labels': labels,
                'contract_findings': findings + list(map(str, result.hard_failures)),
            })
    summaries = {}
    for candidate in ('agent', 'fixed_top3'):
        rows = [row for row in records if row['candidate'] == candidate]
        summaries[candidate] = {
            'records': len(rows),
            'source_label_correct': sum(r['source_labels']['label_correct'] for r in rows),
            'source_evidence_complete_correct': sum(
                r['source_labels']['evidence_complete_correct'] for r in rows),
            'source_abstentions': sum(r['trace']['final_answer'] == 'NOT_ENOUGH_INFO' for r in rows),
            'records_with_contract_findings': sum(bool(r['contract_findings']) for r in rows),
        }
    return {
        'schema': 'rag-reliability-import/v1', 'source_directory': source.name,
        'results_sha256': results_hash, 'source_protocol': results['protocol'],
        'source_policy': results['policy'], 'limitations': LIMITATIONS,
        'summary': summaries, 'records': records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    imported = import_directory(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(imported, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(imported['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
