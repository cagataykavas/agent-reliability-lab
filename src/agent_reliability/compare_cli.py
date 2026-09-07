from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_cases, load_traces
from .regression import RegressionPolicy, compare_reports
from .scoring import evaluate_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare candidate agent traces with a baseline")
    parser.add_argument("cases")
    parser.add_argument("baseline_traces")
    parser.add_argument("candidate_traces")
    parser.add_argument("--output", default="artifacts/regression.json")
    parser.add_argument("--max-score-drop", type=float, default=0.01)
    parser.add_argument("--max-pass-rate-drop", type=float, default=0.0)
    parser.add_argument("--max-case-regressions", type=int, default=0)
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    baseline = evaluate_suite(cases, load_traces(args.baseline_traces))
    candidate = evaluate_suite(cases, load_traces(args.candidate_traces))
    result = compare_reports(
        baseline,
        candidate,
        policy=RegressionPolicy(
            max_mean_score_drop=args.max_score_drop,
            max_pass_rate_drop=args.max_pass_rate_drop,
            max_case_regressions=args.max_case_regressions,
        ),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
