from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_cases, load_traces
from .reporting import write_html_report, write_json_report
from .scoring import evaluate_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-reliability",
        description="Evaluate recorded AI-agent traces against deterministic reliability contracts.",
    )
    parser.add_argument("cases", help="JSONL evaluation-case file")
    parser.add_argument("traces", help="JSONL recorded-agent-trace file")
    parser.add_argument("--threshold", type=float, default=0.8, help="case pass threshold")
    parser.add_argument("--json-output", default="artifacts/report.json")
    parser.add_argument("--html-output", default="artifacts/report.html")
    parser.add_argument("--fail-under", type=float, default=None, help="exit 1 below suite pass rate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_suite(
        load_cases(args.cases),
        load_traces(args.traces),
        pass_threshold=args.threshold,
    )
    json_path = write_json_report(report, args.json_output)
    html_path = write_html_report(report, args.html_output)
    summary = report.to_dict()["summary"]
    print(json.dumps(summary, indent=2))
    print(f"JSON report: {Path(json_path).resolve()}")
    print(f"HTML report: {Path(html_path).resolve()}")
    if args.fail_under is not None and report.pass_rate < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
