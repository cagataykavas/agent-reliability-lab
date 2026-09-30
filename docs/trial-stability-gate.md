# Repeated-trial stability gate

A stochastic agent can pass one evaluation run and still be unsafe to promote. A
single seed can hide intermittent tool misuse, missed escalation, or a case that
oscillates around the pass threshold. Aggregate pass rate alone can also hide one
unstable high-risk case.

`agent_reliability.trial_stability` audits repeated runs of the exact same candidate,
dataset, evaluator and case set before release. It provides:

- unique trial IDs and random seeds so duplicated evidence cannot inflate confidence;
- exact case-set reconciliation across every trial;
- a Wilson lower confidence bound for every case's pass probability;
- minimum pass rate for every complete trial;
- per-case score-span and suite-mean-span limits;
- zero-tolerance failure classes such as unsafe completion or missed escalation;
- bounded input, trial, case, failure and finding budgets;
- deterministic, content-addressed evidence without raw case or trial IDs;
- distinct accepted (`0`), policy-rejected (`2`) and malformed (`3`) exits.

## Artifact contract

The producer runs the existing evaluator repeatedly and records only its structured
results:

```json
{
  "schema_version": 1,
  "candidate_sha256": "<64 lowercase hex characters>",
  "dataset_sha256": "<64 lowercase hex characters>",
  "evaluator_sha256": "<64 lowercase hex characters>",
  "pass_threshold": 0.8,
  "trials": [
    {
      "trial_id": "seed-001",
      "seed": 1,
      "results": [
        {"case_id": "refund-review", "score": 1.0, "passed": true, "failures": []}
      ]
    }
  ]
}
```

`trial_from_report(...)` converts the repository's `EvaluationReport` into a trial
observation. The producer is responsible for setting the model/provider seed before
each real execution and for binding the three manifest digests to immutable release
inputs.

Run the gate with defaults or an explicit JSON policy:

```bash
python -m agent_reliability.trial_stability repeated-trials.json \
  --policy stability-policy.json \
  --output artifacts/trial-stability.json
```

The default policy requires ten trials, a 0.70 per-case Wilson lower bound, at least
0.95 pass rate in every trial, no case score span above 0.20, and no suite mean-score
span above 0.10. Thresholds are release policy, not universal statistical truth.

## Operational boundaries

- A unique seed is evidence of requested variation, not proof that a provider honored
  it. Capture provider request IDs and signed run manifests where available.
- Wilson intervals quantify observed Bernoulli uncertainty. They do not repair biased
  cases, correlated failures, shared infrastructure, or an unrepresentative suite.
- Zero observed failures does not prove zero production risk. More trials narrow the
  lower bound; ten successes yield roughly 0.72 at `z=1.96`.
- Score-span checks expose variation but do not identify its cause. Correlate failures
  with model, prompt, tool-schema, dependency and environment versions.
- SHA-256 identifies exact bytes but does not authenticate the producer. Sign or
  append-only persist manifests before using them as promotion authority.
- Case and trial references are SHA-256 hashes. They reduce accidental disclosure,
  but low-entropy IDs remain guessable and reports still require access control.

The next integration step is a runner adapter that reserves independent seeds,
executes the real benchmark in isolated workers, persists each report before retry,
and transactionally binds the accepted evidence digest to release promotion.
