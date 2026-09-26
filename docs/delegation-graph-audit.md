# Delegation graph release audit

`agent_reliability.delegation` validates the topology, causal timing and resource envelope of a completed multi-agent run before its evidence is admitted to a release report. It targets delegation amplification: a planner can remain individually within budget while recursive children multiply token, tool and concurrency cost.

## Contract and controls

The versioned artifact records one root plus every delegated agent, parent and delegation identifiers, terminal state, timezone-aware start/end timestamps, token usage and tool-call count. The audit rejects duplicate identities, orphan/self parents, cycles, multiple roots and impossible parent/child timing. Policy limits independently cover agent count, depth, fan-out, per-agent and total tokens, per-agent and total tool calls, and root duration.

JSON parsing rejects duplicate fields, non-finite constants, unknown schema fields and oversized artifacts. Reports have deterministic finding codes, exact aggregate metrics and canonical SHA-256 identities. Raw run, policy, agent and delegation identifiers never enter the report. The CLI returns `0` for acceptance, `2` for a policy rejection and `64` for malformed evidence; `--output` uses atomic replacement.

```bash
python -m agent_reliability.delegation run.json --output delegation-report.json
```

## Trust boundaries and limitations

The audit evaluates declared traces. It does not authenticate the collector, prove token billing, detect undeclared agents or tool calls, establish task correctness, or stop a live runaway process. Production enforcement should reserve budgets before spawning work, attach cryptographically authenticated telemetry to a run manifest, and terminate work when live counters reach the same limits.

Wall-clock timestamps establish basic causal consistency but are not a distributed happens-before proof. Cross-host traces need trusted clock synchronization or logical/vector clocks. A single process-local report also does not coordinate budgets across concurrent runs or replicas.

## Next integration step

Emit this artifact from the agent runner, bind it to model/prompt/tool-schema versions in the run registry, and compare pre-execution reservations with post-run usage. Add low-cardinality rejection counters by finding code without agent or customer identifiers.
