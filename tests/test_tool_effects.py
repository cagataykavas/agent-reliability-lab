from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_reliability.tool_effects import (
    EffectConflict,
    EffectState,
    InvalidTransition,
    JournalCorruption,
    LedgerCapacityExceeded,
    OutcomeUnknown,
    ToolEffectLedger,
    _demo,
)


def prepared() -> ToolEffectLedger:
    ledger = ToolEffectLedger()
    ledger.prepare("effect-1", "send_payment", {"amount": 10, "currency": "TRY"})
    return ledger


def test_prepare_dispatch_commit_and_idempotent_replay() -> None:
    ledger = prepared()
    assert ledger.dispatch("effect-1").state is EffectState.DISPATCHED
    committed = ledger.commit("effect-1", {"receipt": "R-1"})

    assert committed.state is EffectState.COMMITTED
    assert ledger.dispatch("effect-1") == committed
    assert ledger.commit("effect-1", {"receipt": "R-1"}) == committed
    assert len(ledger.events) == 3


def test_effect_id_cannot_be_reused_for_different_work() -> None:
    ledger = prepared()

    with pytest.raises(EffectConflict):
        ledger.prepare("effect-1", "send_payment", {"amount": 11, "currency": "TRY"})


def test_dispatched_effect_blocks_blind_retry_after_recovery() -> None:
    ledger = prepared()
    ledger.dispatch("effect-1")
    recovered = ToolEffectLedger.from_jsonl(ledger.dump_jsonl())

    with pytest.raises(OutcomeUnknown):
        recovered.dispatch("effect-1")


def test_reconcile_not_applied_allows_new_attempt() -> None:
    ledger = prepared()
    ledger.dispatch("effect-1")
    ledger.reconcile_not_applied("effect-1")

    record = ledger.dispatch("effect-1")

    assert record.state is EffectState.DISPATCHED
    assert record.attempt == 2


def test_reconcile_applied_commits_unknown_outcome() -> None:
    ledger = prepared()
    ledger.dispatch("effect-1")

    record = ledger.reconcile_applied("effect-1", {"receipt": "R-1"})

    assert record.state is EffectState.COMMITTED


def test_retryable_failure_allows_retry_but_terminal_failure_does_not() -> None:
    ledger = prepared()
    ledger.dispatch("effect-1")
    assert ledger.fail("effect-1", retryable=True).state is EffectState.RETRYABLE_FAILURE
    ledger.dispatch("effect-1")
    assert ledger.fail("effect-1", retryable=False).state is EffectState.TERMINAL_FAILURE

    with pytest.raises(InvalidTransition):
        ledger.dispatch("effect-1")


def test_commit_with_conflicting_result_is_rejected() -> None:
    ledger = prepared()
    ledger.dispatch("effect-1")
    ledger.commit("effect-1", {"receipt": "R-1"})

    with pytest.raises(EffectConflict):
        ledger.commit("effect-1", {"receipt": "R-2"})


def test_hash_chain_detects_event_tampering() -> None:
    ledger = prepared()
    ledger.dispatch("effect-1")
    lines = ledger.dump_jsonl().splitlines()
    event = json.loads(lines[0])
    event["attempt"] = 9
    lines[0] = json.dumps(event)

    with pytest.raises(JournalCorruption, match="hash"):
        ToolEffectLedger.from_jsonl("\n".join(lines))


def test_trusted_head_accepts_exact_checkpoint() -> None:
    ledger = prepared()
    ledger.dispatch("effect-1")
    content, trusted_head = ledger.export_checkpoint()

    recovered = ToolEffectLedger.from_jsonl(
        content,
        expected_head_hash=trusted_head,
    )

    assert recovered.head_hash == trusted_head
    assert recovered.snapshot()["head_hash"] == trusted_head


def test_trusted_head_rejects_valid_but_truncated_journal() -> None:
    ledger = prepared()
    ledger.dispatch("effect-1")
    ledger.commit("effect-1", {"receipt": "R-1"})
    content, trusted_head = ledger.export_checkpoint()
    truncated = "\n".join(content.splitlines()[:-1]) + "\n"

    with pytest.raises(JournalCorruption, match="trusted checkpoint"):
        ToolEffectLedger.from_jsonl(
            truncated,
            expected_head_hash=trusted_head,
        )


def test_trusted_head_rejects_fully_rewritten_valid_journal() -> None:
    original = prepared()
    original.dispatch("effect-1")
    _, trusted_head = original.export_checkpoint()

    rewritten = ToolEffectLedger()
    rewritten.prepare("effect-1", "send_payment", {"amount": 999, "currency": "TRY"})
    rewritten.dispatch("effect-1")

    with pytest.raises(JournalCorruption, match="trusted checkpoint"):
        ToolEffectLedger.from_jsonl(
            rewritten.dump_jsonl(),
            expected_head_hash=trusted_head,
        )


@pytest.mark.parametrize(
    "expected_head_hash",
    ["short", "A" * 64, 7, True],
)
def test_invalid_trusted_head_fails_closed(expected_head_hash: object) -> None:
    with pytest.raises(ValueError, match="expected_head_hash"):
        ToolEffectLedger.from_jsonl(
            "",
            expected_head_hash=expected_head_hash,
        )


def test_journal_rejects_unknown_and_duplicate_fields() -> None:
    line = prepared().dump_jsonl().strip()

    with pytest.raises(JournalCorruption, match="schema"):
        ToolEffectLedger.from_jsonl(line[:-1] + ',"extra":1}')
    with pytest.raises(JournalCorruption, match="duplicate JSON key"):
        ToolEffectLedger.from_jsonl(line[:-1] + ',"sequence":1}')


def test_non_finite_arguments_fail_before_prepare() -> None:
    ledger = ToolEffectLedger()

    with pytest.raises(ValueError, match="finite canonical JSON"):
        ledger.prepare("effect-1", "tool", {"value": float("nan")})
    assert ledger.snapshot()["effects"] == 0


def test_journal_contains_digests_not_raw_arguments_or_results() -> None:
    ledger = ToolEffectLedger()
    ledger.prepare("secret-effect", "private-tool", {"secret": "do-not-persist"})
    ledger.dispatch("secret-effect")
    ledger.commit("secret-effect", {"token": "sensitive-result"})

    journal = ledger.dump_jsonl()

    assert "secret-effect" not in journal
    assert "private-tool" not in journal
    assert "do-not-persist" not in journal
    assert "sensitive-result" not in journal


def test_capacity_limits_are_explicit() -> None:
    effect_limited = ToolEffectLedger(max_effects=1)
    effect_limited.prepare("one", "tool", {})
    with pytest.raises(LedgerCapacityExceeded):
        effect_limited.prepare("two", "tool", {})

    event_limited = ToolEffectLedger(max_events=1)
    event_limited.prepare("one", "tool", {})
    with pytest.raises(LedgerCapacityExceeded):
        event_limited.dispatch("one")


def test_concurrent_duplicate_prepare_emits_one_event() -> None:
    ledger = ToolEffectLedger()

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(
            executor.map(
                lambda _: ledger.prepare("same", "tool", {"value": 1}),
                range(100),
            )
        )

    assert len(set(records)) == 1
    assert len(ledger.events) == 1


def test_unprepared_completion_is_rejected() -> None:
    with pytest.raises(InvalidTransition, match="prepared first"):
        ToolEffectLedger().commit("missing", {"result": True})


def test_demo_proves_unknown_outcome_recovery_path() -> None:
    report = _demo()

    assert report["blind_retry_blocked"] is True
    assert report["final_record"]["state"] == "committed"
    assert report["final_record"]["attempt"] == 2
    assert report["ledger"]["states"]["committed"] == 1
