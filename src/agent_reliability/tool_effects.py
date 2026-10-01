"""Hash-chained state machine for replay-safe agent tool effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64
MAX_JOURNAL_BYTES = 8 * 1024 * 1024
_EVENT_KEYS = {
    "schema_version",
    "sequence",
    "previous_hash",
    "event_type",
    "effect_ref",
    "tool_ref",
    "arguments_digest",
    "attempt",
    "result_digest",
    "retryable",
    "event_hash",
}


class EffectError(RuntimeError):
    """Base class for effect-ledger failures."""


class EffectConflict(EffectError):
    """Raised when one effect ID is reused for different work."""


class OutcomeUnknown(EffectError):
    """Raised when retrying could duplicate an already-applied side effect."""


class InvalidTransition(EffectError):
    """Raised when an effect does not permit the requested transition."""


class JournalCorruption(EffectError):
    """Raised when persisted journal evidence cannot be trusted."""


class LedgerCapacityExceeded(EffectError):
    """Raised when configured in-memory evidence bounds are reached."""


class EffectState(StrEnum):
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    COMMITTED = "committed"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    NOT_APPLIED = "not_applied"


class EventType(StrEnum):
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    COMMITTED = "committed"
    FAILED = "failed"
    NOT_APPLIED = "not_applied"


@dataclass(frozen=True, slots=True)
class EffectRecord:
    effect_ref: str
    tool_ref: str
    arguments_digest: str
    state: EffectState
    attempt: int
    result_digest: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "effect_ref": self.effect_ref,
            "tool_ref": self.tool_ref,
            "arguments_digest": self.arguments_digest,
            "state": self.state,
            "attempt": self.attempt,
            "result_digest": self.result_digest,
        }


@dataclass(frozen=True, slots=True)
class EffectEvent:
    schema_version: int
    sequence: int
    previous_hash: str
    event_type: EventType
    effect_ref: str
    tool_ref: str
    arguments_digest: str
    attempt: int
    result_digest: str | None
    retryable: bool | None
    event_hash: str

    def hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "previous_hash": self.previous_hash,
            "event_type": self.event_type,
            "effect_ref": self.effect_ref,
            "tool_ref": self.tool_ref,
            "arguments_digest": self.arguments_digest,
            "attempt": self.attempt,
            "result_digest": self.result_digest,
            "retryable": self.retryable,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.hash_payload(), "event_hash": self.event_hash}


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reference(value: str, field: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ValueError(f"{field} must contain 1..256 characters")
    return hashlib.sha256(value.encode()).hexdigest()


def _event_hash(payload: Mapping[str, Any]) -> str:
    return _canonical_digest(payload)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JournalCorruption(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class ToolEffectLedger:
    """Record tool effects so an unknown outcome cannot be retried blindly.

    The ledger stores only SHA-256 references for effect IDs, tool names,
    arguments, and results. Callers must persist ``dump_jsonl`` before and
    after external dispatch according to their durability requirements.
    """

    def __init__(self, *, max_effects: int = 10_000, max_events: int = 100_000) -> None:
        if type(max_effects) is not int or max_effects < 1:
            raise ValueError("max_effects must be a positive integer")
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self._max_effects = max_effects
        self._max_events = max_events
        self._records: dict[str, EffectRecord] = {}
        self._events: list[EffectEvent] = []
        self._lock = threading.RLock()

    @property
    def events(self) -> tuple[EffectEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def get(self, effect_id: str) -> EffectRecord | None:
        effect_ref = _reference(effect_id, "effect_id")
        with self._lock:
            return self._records.get(effect_ref)

    def prepare(self, effect_id: str, tool_name: str, arguments: Any) -> EffectRecord:
        effect_ref = _reference(effect_id, "effect_id")
        tool_ref = _reference(tool_name, "tool_name")
        arguments_digest = _canonical_digest(arguments)
        with self._lock:
            current = self._records.get(effect_ref)
            if current is not None:
                if current.tool_ref != tool_ref or current.arguments_digest != arguments_digest:
                    raise EffectConflict("effect ID reused for different tool work")
                return current
            if len(self._records) >= self._max_effects:
                raise LedgerCapacityExceeded("effect capacity exhausted")
            return self._emit(
                EventType.PREPARED,
                effect_ref=effect_ref,
                tool_ref=tool_ref,
                arguments_digest=arguments_digest,
                attempt=0,
            )

    def dispatch(self, effect_id: str) -> EffectRecord:
        effect_ref = _reference(effect_id, "effect_id")
        with self._lock:
            current = self._require(effect_ref)
            if current.state is EffectState.DISPATCHED:
                raise OutcomeUnknown("tool outcome is unknown; reconcile it before retrying")
            if current.state is EffectState.COMMITTED:
                return current
            if current.state is EffectState.TERMINAL_FAILURE:
                raise InvalidTransition("terminally failed effect cannot be dispatched")
            if current.state not in {
                EffectState.PREPARED,
                EffectState.RETRYABLE_FAILURE,
                EffectState.NOT_APPLIED,
            }:
                raise InvalidTransition(f"cannot dispatch from {current.state}")
            return self._emit_from(
                current,
                EventType.DISPATCHED,
                attempt=current.attempt + 1,
            )

    def commit(self, effect_id: str, result: Any) -> EffectRecord:
        return self._complete(effect_id, EventType.COMMITTED, result=result)

    def reconcile_applied(self, effect_id: str, result: Any) -> EffectRecord:
        return self._complete(effect_id, EventType.COMMITTED, result=result)

    def reconcile_not_applied(self, effect_id: str) -> EffectRecord:
        return self._complete(effect_id, EventType.NOT_APPLIED)

    def fail(self, effect_id: str, *, retryable: bool) -> EffectRecord:
        if type(retryable) is not bool:
            raise ValueError("retryable must be a boolean")
        return self._complete(effect_id, EventType.FAILED, retryable=retryable)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = {state.value: 0 for state in EffectState}
            for record in self._records.values():
                counts[record.state.value] += 1
            return {
                "effects": len(self._records),
                "events": len(self._events),
                "states": counts,
                "head_hash": self._events[-1].event_hash if self._events else GENESIS_HASH,
            }

    def dump_jsonl(self) -> str:
        with self._lock:
            return "".join(
                json.dumps(event.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                for event in self._events
            )

    @classmethod
    def from_jsonl(
        cls,
        content: str,
        *,
        max_effects: int = 10_000,
        max_events: int = 100_000,
    ) -> ToolEffectLedger:
        if not isinstance(content, str):
            raise JournalCorruption("journal must be text")
        if len(content.encode()) > MAX_JOURNAL_BYTES:
            raise JournalCorruption("journal exceeds byte bound")
        ledger = cls(max_effects=max_effects, max_events=max_events)
        lines = content.splitlines()
        if len(lines) > max_events:
            raise JournalCorruption("journal exceeds event bound")
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise JournalCorruption(f"blank journal line {line_number}")
            try:
                data = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except JournalCorruption:
                raise
            except json.JSONDecodeError as exc:
                raise JournalCorruption(f"invalid JSON on line {line_number}") from exc
            event = ledger._parse_event(data, line_number)
            ledger._verify_and_apply(event)
        return ledger

    def _complete(
        self,
        effect_id: str,
        event_type: EventType,
        *,
        result: Any | None = None,
        retryable: bool | None = None,
    ) -> EffectRecord:
        effect_ref = _reference(effect_id, "effect_id")
        with self._lock:
            current = self._require(effect_ref)
            if current.state is EffectState.COMMITTED and event_type is EventType.COMMITTED:
                candidate_digest = _canonical_digest(result)
                if candidate_digest != current.result_digest:
                    raise EffectConflict("committed effect has a different result")
                return current
            if current.state is not EffectState.DISPATCHED:
                raise InvalidTransition(f"cannot complete from {current.state}")
            result_digest = _canonical_digest(result) if event_type is EventType.COMMITTED else None
            return self._emit_from(
                current,
                event_type,
                attempt=current.attempt,
                result_digest=result_digest,
                retryable=retryable,
            )

    def _require(self, effect_ref: str) -> EffectRecord:
        try:
            return self._records[effect_ref]
        except KeyError as exc:
            raise InvalidTransition("effect must be prepared first") from exc

    def _emit_from(
        self,
        current: EffectRecord,
        event_type: EventType,
        *,
        attempt: int,
        result_digest: str | None = None,
        retryable: bool | None = None,
    ) -> EffectRecord:
        return self._emit(
            event_type,
            effect_ref=current.effect_ref,
            tool_ref=current.tool_ref,
            arguments_digest=current.arguments_digest,
            attempt=attempt,
            result_digest=result_digest,
            retryable=retryable,
        )

    def _emit(
        self,
        event_type: EventType,
        *,
        effect_ref: str,
        tool_ref: str,
        arguments_digest: str,
        attempt: int,
        result_digest: str | None = None,
        retryable: bool | None = None,
    ) -> EffectRecord:
        if len(self._events) >= self._max_events:
            raise LedgerCapacityExceeded("event capacity exhausted")
        payload = {
            "schema_version": 1,
            "sequence": len(self._events) + 1,
            "previous_hash": self._events[-1].event_hash if self._events else GENESIS_HASH,
            "event_type": event_type,
            "effect_ref": effect_ref,
            "tool_ref": tool_ref,
            "arguments_digest": arguments_digest,
            "attempt": attempt,
            "result_digest": result_digest,
            "retryable": retryable,
        }
        event = EffectEvent(**payload, event_hash=_event_hash(payload))
        record = self._apply(event)
        self._events.append(event)
        return record

    def _verify_and_apply(self, event: EffectEvent) -> None:
        expected_sequence = len(self._events) + 1
        expected_previous = self._events[-1].event_hash if self._events else GENESIS_HASH
        if event.sequence != expected_sequence:
            raise JournalCorruption("journal sequence is not contiguous")
        if event.previous_hash != expected_previous:
            raise JournalCorruption("journal hash chain is broken")
        if event.event_hash != _event_hash(event.hash_payload()):
            raise JournalCorruption("event hash does not match its contents")
        try:
            self._apply(event)
        except (EffectError, ValueError) as exc:
            raise JournalCorruption(f"invalid {event.event_type} transition") from exc
        self._events.append(event)

    def _apply(self, event: EffectEvent) -> EffectRecord:
        current = self._records.get(event.effect_ref)
        if event.event_type is EventType.PREPARED:
            if (
                current is not None
                or event.attempt != 0
                or event.result_digest is not None
                or event.retryable is not None
            ):
                raise InvalidTransition("prepared event must create a new effect")
            if len(self._records) >= self._max_effects:
                raise LedgerCapacityExceeded("effect capacity exhausted")
            state = EffectState.PREPARED
        else:
            if current is None:
                raise InvalidTransition("effect has no prepared event")
            if (
                current.tool_ref != event.tool_ref
                or current.arguments_digest != event.arguments_digest
            ):
                raise EffectConflict("event metadata changed within an effect")
            state = self._next_state(current, event)
        record = EffectRecord(
            effect_ref=event.effect_ref,
            tool_ref=event.tool_ref,
            arguments_digest=event.arguments_digest,
            state=state,
            attempt=event.attempt,
            result_digest=event.result_digest,
        )
        self._records[event.effect_ref] = record
        return record

    @staticmethod
    def _next_state(current: EffectRecord, event: EffectEvent) -> EffectState:
        if event.event_type is EventType.DISPATCHED:
            if (
                current.state
                not in {
                    EffectState.PREPARED,
                    EffectState.RETRYABLE_FAILURE,
                    EffectState.NOT_APPLIED,
                }
                or event.attempt != current.attempt + 1
            ):
                raise InvalidTransition("invalid dispatch")
            if event.result_digest is not None or event.retryable is not None:
                raise InvalidTransition("dispatch cannot contain outcome evidence")
            return EffectState.DISPATCHED
        if current.state is not EffectState.DISPATCHED or event.attempt != current.attempt:
            raise InvalidTransition("terminal event requires the current dispatch attempt")
        if event.event_type is EventType.COMMITTED:
            if event.result_digest is None or event.retryable is not None:
                raise InvalidTransition("commit requires only a result digest")
            return EffectState.COMMITTED
        if event.event_type is EventType.FAILED:
            if event.result_digest is not None or type(event.retryable) is not bool:
                raise InvalidTransition("failure requires a retryable decision")
            return (
                EffectState.RETRYABLE_FAILURE if event.retryable else EffectState.TERMINAL_FAILURE
            )
        if event.event_type is EventType.NOT_APPLIED:
            if event.result_digest is not None or event.retryable is not None:
                raise InvalidTransition("not-applied cannot contain outcome evidence")
            return EffectState.NOT_APPLIED
        raise InvalidTransition("unsupported event type")

    @staticmethod
    def _parse_event(data: Any, line_number: int) -> EffectEvent:
        if not isinstance(data, dict) or set(data) != _EVENT_KEYS:
            raise JournalCorruption(f"line {line_number} has an invalid event schema")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise JournalCorruption("unsupported schema version")
        if type(data["sequence"]) is not int or data["sequence"] < 1:
            raise JournalCorruption("sequence must be a positive integer")
        if type(data["attempt"]) is not int or data["attempt"] < 0:
            raise JournalCorruption("attempt must be a non-negative integer")
        for field in ("previous_hash", "effect_ref", "tool_ref", "arguments_digest", "event_hash"):
            value = data[field]
            if not _is_sha256(value):
                raise JournalCorruption(f"{field} must be a SHA-256 reference")
        result_digest = data["result_digest"]
        if result_digest is not None and not _is_sha256(result_digest):
            raise JournalCorruption("result_digest must be null or a SHA-256 reference")
        if data["retryable"] is not None and type(data["retryable"]) is not bool:
            raise JournalCorruption("retryable must be null or boolean")
        try:
            event_type = EventType(data["event_type"])
        except (TypeError, ValueError) as exc:
            raise JournalCorruption("unknown event type") from exc
        return EffectEvent(
            schema_version=1,
            sequence=data["sequence"],
            previous_hash=data["previous_hash"],
            event_type=event_type,
            effect_ref=data["effect_ref"],
            tool_ref=data["tool_ref"],
            arguments_digest=data["arguments_digest"],
            attempt=data["attempt"],
            result_digest=result_digest,
            retryable=data["retryable"],
            event_hash=data["event_hash"],
        )


def _demo() -> dict[str, Any]:
    ledger = ToolEffectLedger()
    ledger.prepare("refund:order-77", "issue_refund", {"order_id": "O-77", "amount": 25})
    ledger.dispatch("refund:order-77")
    recovered = ToolEffectLedger.from_jsonl(ledger.dump_jsonl())
    blocked_retry = False
    try:
        recovered.dispatch("refund:order-77")
    except OutcomeUnknown:
        blocked_retry = True
    recovered.reconcile_not_applied("refund:order-77")
    recovered.dispatch("refund:order-77")
    committed = recovered.commit("refund:order-77", {"receipt_id": "R-19"})
    return {
        "schema_version": 1,
        "scenario": "crash-after-tool-dispatch",
        "blind_retry_blocked": blocked_retry,
        "final_record": committed.as_dict(),
        "ledger": recovered.snapshot(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the tool-effect recovery scenario")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(_demo(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
