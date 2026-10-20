# Replay-safe agent tool effects

Tool-using agents can fail between sending a request and recording its response. Retrying a payment,
refund, message, provisioning action, or other side effect at that point can execute it twice. The
tool-effect ledger makes this ambiguity explicit instead of treating every exception as retryable.

## State model

```text
prepared -> dispatched -> committed
                      \-> retryable_failure -> dispatched
                      \-> terminal_failure
                      \-> not_applied -> dispatched
```

`dispatched` without a terminal event is an unknown outcome. After journal recovery, another
dispatch is rejected until an external reconciliation step proves either that the effect was
applied (`reconcile_applied`) or was not applied (`reconcile_not_applied`).

Every transition is appended to a SHA-256 hash chain. Recovery rejects gaps, altered hashes,
unknown or duplicate fields, illegal state transitions, non-finite JSON, and configured capacity
breaches. The persisted evidence contains only digests of effect IDs, tool names, arguments, and
results; it does not serialize raw tool arguments or credentials.

```python
ledger.prepare("refund:order-77", "issue_refund", {"order_id": "O-77"})
ledger.dispatch("refund:order-77")
journal, trusted_head = ledger.export_checkpoint()

# Persist before the external call. If the process now crashes, recovery blocks a blind retry.
recovered = ToolEffectLedger.from_jsonl(
    journal,
    expected_head_hash=trusted_head,
)
recovered.reconcile_not_applied("refund:order-77")
recovered.dispatch("refund:order-77")
recovered.commit("refund:order-77", {"receipt_id": "R-19"})
```

Run the deterministic crash/recovery scenario:

```bash
python -m agent_reliability.tool_effects --output artifacts/tool-effects.json
```

## Durability boundary

The implementation is an in-memory state machine with a portable JSONL journal. It does not make
file writes atomic, call an external tool, acquire a cross-process lock, or guarantee exactly-once
delivery by itself. `export_checkpoint()` captures the journal and head under the same lock. Store
that head in an authenticated, rollback-protected location and pass it back as
`expected_head_hash`; recovery then rejects both truncation and a fully rewritten, internally valid
journal. Without that external checkpoint, the chain remains integrity evidence rather than
authentication. Production integration must durably append `dispatched` before the side effect,
durably append the terminal event afterward, atomically advance the trusted checkpoint, use one
writer or transactional compare-and-set, and reconcile unknown outcomes through the provider's
idempotency or status API.

The next step is a SQLite or PostgreSQL adapter that atomically claims prepared work, fences stale
workers with a lease generation, and persists each hash-chained transition before acknowledging it.
