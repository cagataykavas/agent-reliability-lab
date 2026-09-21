# Evidence-lineage gate

An agent can cite the expected evidence ID while the recorded execution does not prove where that evidence came from. The lineage gate checks the causal path from tool calls to content-addressed evidence and the final citation set before a trace is admitted as release evidence.

## Artifact contract

```json
{
  "trace_id": "refund-review",
  "calls": [
    {
      "call_id": "order",
      "tool_name": "get_order",
      "depends_on": [],
      "succeeded": true,
      "evidence": [
        {
          "evidence_id": "order:O-77",
          "content_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }
      ]
    },
    {
      "call_id": "policy",
      "tool_name": "get_refund_policy",
      "depends_on": ["order"],
      "succeeded": true,
      "evidence": [
        {
          "evidence_id": "policy:refund:TR",
          "content_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        }
      ]
    }
  ],
  "cited_evidence_ids": ["order:O-77", "policy:refund:TR"]
}
```

Run the dependency-free gate directly from the installed package:

```bash
python -m agent_reliability.lineage trace-lineage.json \
  --required-evidence order:O-77 \
  --required-evidence policy:refund:TR \
  --max-calls 16 \
  --max-dependency-depth 4 \
  --require-all-produced-evidence-cited \
  --output artifacts/lineage-report.json
```

The command exits `0` when the trace passes and `2` for a malformed artifact or policy rejection. Its JSON report includes stable finding codes, call/evidence counts, maximum dependency depth and a deterministic evidence-to-source-call map.

## Enforced invariants

- call and evidence IDs are unique;
- dependencies reference earlier calls, preventing self, forward and cyclic lineage;
- a call cannot silently rely on a failed prerequisite;
- failed calls cannot produce trusted evidence;
- evidence includes a canonical lowercase SHA-256 content digest;
- citations resolve to evidence produced by a recorded call;
- required evidence appears in the final citation set;
- optional strict mode rejects produced-but-uncited evidence;
- call count and causal depth remain inside explicit budgets;
- malformed types, empty identifiers and invalid policies fail closed.

## Trust boundary and limitations

The gate validates internal consistency, not factual truth. A SHA-256 digest proves that the same bytes were referenced; it does not prove that those bytes are correct, current or authorized. Trace creation must therefore happen in a server-controlled recorder rather than from model-authored metadata.

For stronger non-repudiation, sign the completed artifact and store it in an append-only audit system. Distributed and parallel agents also need span timestamps or vector-clock semantics; list order here is treated as the recorded causal order. Content is intentionally excluded from the report, but evidence IDs themselves must still avoid secrets or personal data.
