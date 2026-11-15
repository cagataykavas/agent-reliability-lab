"""CLI for gating agent-evaluation labels on inter-rater agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agreement import AgreementPolicy, Rating, evaluate_agreement

MAX_RATINGS = 100_000


def _load_ratings(path: str | Path) -> list[Rating]:
    source = Path(path)
    ratings: list[Rating] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if len(ratings) >= MAX_RATINGS:
                raise ValueError(f"rating count exceeds {MAX_RATINGS}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
            try:
                ratings.append(Rating.from_dict(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: invalid rating: {error}") from error
    return ratings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate agent-evaluation labels on pairwise inter-rater agreement"
    )
    parser.add_argument("ratings", help="JSONL ratings with case_id, rater_id and label")
    parser.add_argument("--output", default="artifacts/agreement.json")
    parser.add_argument("--min-shared-cases", type=int, default=20)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.8)
    parser.add_argument("--min-kappa", type=float, default=0.6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_agreement(
            _load_ratings(args.ratings),
            AgreementPolicy(
                min_shared_cases=args.min_shared_cases,
                min_overlap_ratio=args.min_overlap_ratio,
                min_kappa=args.min_kappa,
            ),
        )
    except (OSError, TypeError, ValueError) as error:
        payload = {"accepted": False, "status": "invalid", "error": str(error)}
        print(json.dumps(payload), file=sys.stderr)
        return 2

    payload = report.to_dict()
    target = Path(args.output)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as error:
        invalid = {"accepted": False, "status": "invalid", "error": str(error)}
        print(json.dumps(invalid), file=sys.stderr)
        return 2
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
