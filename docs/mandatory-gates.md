# Mandatory constraints are not compensable

Previously, a missing required escalation produced score 0.9 and passed the default
0.8 threshold. Missing required evidence could likewise pass at 0.84. An expected
tool outside the explicit allowlist could even score 1.0 and pass.

The evaluator now keeps the original quality score and component breakdown but
requires both the score threshold and absence of mandatory failures for `passed`.
Built-in hard gates are missing required escalation, missing required evidence,
explicit tool allowlist violations, and unhandled errors (including missing traces).
They cannot be disabled with an empty additional policy. Extra calls that are allowed
are not hard failures merely because the expected trajectory omitted them.

`hard_failure_types` is an optional list of existing failure names on each case.
It adds zero-tolerance policy, for example `['budget_exceeded', 'wrong_arguments']`.
Unknown names fail validation. Empty lists preserve default behavior. Existing
JSON report fields and score arithmetic remain intact; `hard_failures` is additive.
Existing consumers must use `passed`, rather than recompute it from `score`.

Latency/cost limits, trajectory match, unnecessary escalation and answer keyword
coverage remain weighted quality checks unless explicitly promoted. The historical
`unsafe_completion` name remains for compatibility with keyword checks; it is not a
semantic safety verdict. No change here proves that evidence supports a claim.
The existing JSON loader's inferred allowlist (expected tool names when omitted)
remains unchanged. An empty allowlist means unrestricted, not deny-all.

Case gates do not override suite-level `--fail-under`: a policy permitting a fraction
of failed cases can still admit that suite. Use 1.0 when no failed case is acceptable,
or the separate risk-stratified branch for independently reviewed segment gates.

## Synthetic reproduction

Run `PYTHONPATH=src python scripts/gate_matrix.py` from this checkout.
Run the same script with PYTHONPATH pointing to baseline commit 49f7af4's `src`
to reproduce the defect. Four mandatory-violation fixtures include three previously
accepted violations plus an already-rejected error control. Six controls cover
valid contracts and intentionally soft quality diagnostics. These are evaluator
regressions, not model-quality or production-safety measurements.

## RAG adapter

```
PYTHONPATH=src python -m agent_reliability.rag_import \
  /path/to/rag-agent/evidence/agent-v2-dev --output artifacts/rag-v2.json
```

Also accepts the existing `agent-v3-capacity` directory. Each directory contains
24 agent and 24 fixed-top3 execution records. The adapter reads only `results.json`
and the matching trace JSON files, never SQLite, credentials, or model weights.
Output includes SHA-256 provenance, the complete source protocol/policy and raw
per-record labels. It rejects mismatched identity, prediction, citation and status
joins, duplicate IDs, and non-boolean correctness labels.

Actual recorded search/read calls become `AgentTrace.tool_calls`; finish is the
final answer, not an external tool call. Proposed actions without recorded results
are not fabricated as executed calls. Wall time is explicitly source invocation
wall time. Tool latency and cost are unknown; schema defaults are placeholders,
not measurements. No gold trajectory or human-review expectation is inferred.
No expected evidence is copied from the agent's own citations. Generic weighted
scores are omitted because absent gold trajectories would distort them.

Read-only allowlist and execution-error gates use the shared evaluator. Citation
membership in recorded read results, NEI citation emptiness and non-NEI citation
presence are checked separately. A structurally valid wrong answer remains wrong
in `source_labels`, without being relabeled `unsafe_completion`. Ordinary NEI
abstention is not escalation or an execution error. Source truth labels are not
independently adjudicated, and citation linkage is not entailment.

These runs reuse development cases with prior exposure and differ in model/policy.
The adapter performs no new inference, no generalization estimate and no production
validation. It deliberately does not merge the richer feature-branch lineage,
recovery or statistical frameworks merely to inflate scope.
