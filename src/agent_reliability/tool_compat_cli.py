"""Command-line interface for tool manifest compatibility checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_reliability.tool_compatibility import (
    ToolCompatibilityError,
    audit_tool_compatibility,
    load_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail when a candidate agent tool manifest breaks baseline callers."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_tool_compatibility(
            load_manifest(args.baseline), load_manifest(args.candidate)
        )
    except ToolCompatibilityError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2

    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            if args.output.is_symlink():
                raise OSError("output must not be a symlink")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(json.dumps({"error": f"cannot write report: {exc}"}, sort_keys=True))
            return 2
    print(rendered, end="")
    return 0 if report.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
