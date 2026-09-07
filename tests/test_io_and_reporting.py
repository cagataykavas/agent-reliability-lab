import json

from agent_reliability.io import load_cases, load_traces
from agent_reliability.reporting import write_html_report, write_json_report
from agent_reliability.scoring import evaluate_suite


def test_example_suite_and_reports(tmp_path) -> None:
    cases = load_cases("examples/cases.jsonl")
    traces = load_traces("examples/traces.jsonl")
    report = evaluate_suite(cases, traces)
    assert report.pass_rate == 1.0

    json_path = write_json_report(report, tmp_path / "report.json")
    html_path = write_html_report(report, tmp_path / "report.html")
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["passed_cases"] == 3
    assert "Agent Reliability Report" in html_path.read_text()
