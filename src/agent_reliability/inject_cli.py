from __future__ import annotations

import argparse
import json
from pathlib import Path

from .faults import FaultType, inject_suite
from .io import load_traces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inject deterministic faults into agent traces")
    parser.add_argument("traces")
    parser.add_argument("output")
    parser.add_argument("--fault", required=True, choices=[item.value for item in FaultType])
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--magnitude", type=float, default=10.0)
    args = parser.parse_args(argv)

    traces = inject_suite(
        load_traces(args.traces),
        FaultType(args.fault),
        every=args.every,
        magnitude=args.magnitude,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(trace.to_dict(), sort_keys=True) for trace in traces]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(traces)} fault-injected traces to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
