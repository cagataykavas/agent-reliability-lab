from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .progress import (
    ProgressAuditError,
    ProgressPolicy,
    audit_progress_trace,
    load_progress_trace,
)


def _bounded_int(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 10_000:
        raise argparse.ArgumentTypeError("value must be between 0 and 10000")
    return parsed


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit recorded agent progress and loop budgets")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=_bounded_int, default=128)
    parser.add_argument("--max-no-progress-steps", type=_bounded_int, default=2)
    parser.add_argument("--max-action-state-repeats", type=_bounded_int, default=2)
    parser.add_argument("--max-state-visits", type=_bounded_int, default=3)
    parser.add_argument("--allow-unterminated", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = audit_progress_trace(
            load_progress_trace(args.trace),
            ProgressPolicy(
                max_steps=args.max_steps,
                max_no_progress_steps=args.max_no_progress_steps,
                max_action_state_repeats=args.max_action_state_repeats,
                max_state_visits=args.max_state_visits,
                require_terminal=not args.allow_unterminated,
            ),
        )
    except ProgressAuditError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 3
    payload = report.to_dict()
    if args.output is not None:
        _write_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
