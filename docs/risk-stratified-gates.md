# Risk-stratified agent release gates

An aggregate score can hide a production-critical regression. If nine routine cases
pass and one write-capable payment case fails, the suite still reports a 90% pass
rate. `audit_segments` makes each reviewed slice an independent release boundary.

```python
from agent_reliability import SegmentPolicy, audit_segments, evaluate_suite
from agent_reliability.models import FailureType

evaluation = evaluate_suite(cases, traces)
audit = audit_segments(
    evaluation,
    assignments={
        "faq-read": {"routine", "read-only"},
        "refund-write": {"high-risk", "write-tools"},
        "payment-write": {"high-risk", "write-tools"},
    },
    policies={
        "routine": SegmentPolicy(min_cases=1, min_pass_rate=0.95),
        "read-only": SegmentPolicy(min_cases=1, min_pass_rate=0.95),
        "write-tools": SegmentPolicy(min_cases=2, min_pass_rate=1.0),
        "high-risk": SegmentPolicy(
            min_cases=2,
            min_pass_rate=1.0,
            min_mean_score=0.95,
            blocked_failures=frozenset(
                {
                    FailureType.MISSING_ESCALATION,
                    FailureType.UNHANDLED_ERROR,
                }
            ),
        ),
    },
)

if not audit.passed:
    publish_evidence(audit.as_dict())
    raise SystemExit(2)
```

## Contract

- Every evaluated case must have at least one segment assignment.
- Assignments may overlap so teams can audit task family, tool risk, geography or
  another prespecified dimension at the same time.
- Every observed segment needs an explicit policy or a supplied default policy.
- Every explicitly configured segment is required. Missing evidence is reported as
  a rejection instead of disappearing from aggregate metrics.
- Minimum case count, pass rate and mean score are checked independently.
- Selected failure types can be zero-tolerance even when the weighted case score
  remains above its pass threshold.
- Output order, case IDs, failure counts and finding codes are deterministic and
  JSON-ready for CI artifacts.

Malformed manifests, unknown case IDs, invalid/non-finite scores and unreviewed
segment names fail closed before a release decision is produced.

## Calibration and limits

Segments and thresholds are policy choices, not learned facts. Prespecify them before
running the candidate to avoid choosing only favorable slices. Small segments have
high variance: `min_cases` enforces an evidence floor but does not provide statistical
confidence by itself. Use paired confidence intervals when comparing model versions.

Overlapping segments are intentionally not statistically independent, and this gate
does not adjust for multiple comparisons. It also audits recorded evaluator results;
it does not prove that the case suite represents live traffic or that segment labels
are correct. Version the assignment manifest with the dataset and review label changes
as carefully as prompt, tool-policy and model changes.

The next step is a signed JSON manifest loader that binds assignments and policies to
the evaluated dataset fingerprint in the run registry.
