from __future__ import annotations

import html
import json
from pathlib import Path

from .models import EvaluationReport


def write_json_report(report: EvaluationReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def write_html_report(report: EvaluationReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = report.to_dict()["summary"]
    rows = []
    for result in report.results:
        failures = ", ".join(map(str, result.failures)) or "none"
        diagnostics = "<br>".join(html.escape(item) for item in result.diagnostics) or "—"
        status = "PASS" if result.passed else "FAIL"
        rows.append(
            f"<tr><td>{html.escape(result.case_id)}</td>"
            f"<td><span class='{status.lower()}'>{status}</span></td>"
            f"<td>{result.score:.3f}</td><td>{html.escape(failures)}</td>"
            f"<td>{diagnostics}</td></tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Reliability Report</title><style>
:root{{--bg:#07111f;--panel:#0d1b2d;--ink:#eef5ff;--muted:#9cb0c8;--good:#74f0c7;--bad:#ff8f9c}}
*{{box-sizing:border-box}}body{{margin:0;padding:40px;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui}}
main{{max-width:1100px;margin:auto}}h1{{font-size:38px;margin-bottom:8px}}.subtitle{{color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}.metric{{padding:18px;border:1px solid #ffffff20;border-radius:14px;background:var(--panel)}}.metric strong{{display:block;font-size:26px}}.metric span{{color:var(--muted)}}
.table-wrap{{overflow:auto;border:1px solid #ffffff20;border-radius:14px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:14px;text-align:left;border-bottom:1px solid #ffffff16;vertical-align:top}}th{{color:var(--muted);font-size:11px;text-transform:uppercase}}.pass{{color:var(--good);font-weight:800}}.fail{{color:var(--bad);font-weight:800}}@media(max-width:700px){{body{{padding:20px}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><h1>Agent Reliability Report</h1><p class="subtitle">Deterministic evaluation of tool use, evidence, budgets and escalation behavior.</p>
<section class="metrics"><div class="metric"><strong>{summary['total_cases']}</strong><span>cases</span></div><div class="metric"><strong>{summary['passed_cases']}</strong><span>passed</span></div><div class="metric"><strong>{summary['pass_rate']:.1%}</strong><span>pass rate</span></div><div class="metric"><strong>{summary['mean_score']:.3f}</strong><span>mean score</span></div></section>
<div class="table-wrap"><table><thead><tr><th>Case</th><th>Status</th><th>Score</th><th>Failures</th><th>Diagnostics</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</main></body></html>"""
    target.write_text(document, encoding="utf-8")
    return target
