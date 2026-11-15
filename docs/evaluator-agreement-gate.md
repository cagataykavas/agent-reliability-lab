# Evaluator agreement gate

Agent regression scores are only as trustworthy as the labels behind them. Two judges can report a
high raw match rate simply because almost every case receives the same label, while sparse overlap
can hide systematic disagreement. `agent-reliability-agreement` adds a deterministic admission gate
before labels are used as release evidence.

Each JSONL record contains exactly a `case_id`, `rater_id`, and categorical `label`:

```json
{"case_id":"refund-001","rater_id":"reviewer-a","label":"pass"}
```

```bash
agent-reliability-agreement ratings.jsonl \
  --min-shared-cases 20 \
  --min-overlap-ratio 0.8 \
  --min-kappa 0.6 \
  --output artifacts/agreement.json
```

The gate evaluates every rater pair with Cohen's kappa, shared-case count, and Jaccard overlap. It
rejects low overlap, insufficient shared cases, kappa below policy, duplicate ratings, and undefined
kappa caused by single-label prevalence collapse. The report lists disagreement case IDs for review.
Exit code `0` means accepted, `1` means valid evidence rejected by policy, and `2` means malformed
evidence or policy.

## Limitations

Cohen's kappa measures pairwise categorical consistency, not correctness. It is sensitive to label
prevalence and does not establish that human or model judges are calibrated against ground truth.
Ordinal or continuous scores need a different statistic. Production use should pair this gate with
adjudicated gold cases, blinded rating order, judge-version provenance, and periodic drift review.
