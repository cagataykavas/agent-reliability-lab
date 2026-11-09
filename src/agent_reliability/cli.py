from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import DEFAULT_JSONL_LIMITS, JsonlLimits, load_cases, load_traces
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
    parser.add_argument(
        "--fail-under", type=float, default=None, help="exit 1 below suite pass rate"
    )
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=DEFAULT_JSONL_LIMITS.max_file_bytes,
        help="maximum bytes accepted per JSONL input file",
    )
    parser.add_argument(
        "--max-line-bytes",
        type=int,
        default=DEFAULT_JSONL_LIMITS.max_line_bytes,
        help="maximum bytes accepted for one JSONL record",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_JSONL_LIMITS.max_records,
        help="maximum non-comment records accepted per input file",
    )
    parser.add_argument(
        "--max-json-depth",
        type=int,
        default=DEFAULT_JSONL_LIMITS.max_json_depth,
        help="maximum object/array nesting depth accepted per record",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    limits = JsonlLimits(
        max_file_bytes=args.max_input_bytes,
        max_line_bytes=args.max_line_bytes,
        max_records=args.max_records,
        max_json_depth=args.max_json_depth,
    )
    report = evaluate_suite(
        load_cases(args.cases, limits=limits),
        load_traces(args.traces, limits=limits),
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
