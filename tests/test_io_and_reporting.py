import json

import pytest

from agent_reliability.cli import main
from agent_reliability.io import JsonlLimits, load_cases, load_traces
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


@pytest.mark.parametrize(
    ("payload", "limits", "message"),
    [
        (
            '{"case_id":"first","case_id":"second","prompt":"hello"}\n',
            JsonlLimits(),
            "duplicate JSON key 'case_id'",
        ),
        (
            '{"case_id":"case-1","prompt":"hello","max_cost_usd":NaN}\n',
            JsonlLimits(),
            "non-finite JSON number 'NaN'",
        ),
        (
            '{"case_id":"case-1","prompt":"hello","metadata":{"nested":{"too_deep":true}}}\n',
            JsonlLimits(max_json_depth=2),
            "JSON nesting depth exceeds 2",
        ),
    ],
)
def test_strict_jsonl_rejects_ambiguous_records(tmp_path, payload, limits, message) -> None:
    source = tmp_path / "cases.jsonl"
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cases(source, limits=limits)


def test_jsonl_limits_bound_file_line_and_record_count(tmp_path) -> None:
    source = tmp_path / "cases.jsonl"
    record = '{"case_id":"case-1","prompt":"hello"}\n'
    source.write_text(record + record.replace("case-1", "case-2"), encoding="utf-8")

    with pytest.raises(ValueError, match="file exceeds 10 byte input budget"):
        load_cases(source, limits=JsonlLimits(max_file_bytes=10))
    with pytest.raises(ValueError, match="line exceeds 12 byte budget"):
        load_cases(source, limits=JsonlLimits(max_line_bytes=12))
    with pytest.raises(ValueError, match="record count exceeds 1"):
        load_cases(source, limits=JsonlLimits(max_records=1))


def test_jsonl_loader_rejects_symlinks(tmp_path) -> None:
    target = tmp_path / "cases.jsonl"
    target.write_text('{"case_id":"case-1","prompt":"hello"}\n', encoding="utf-8")
    link = tmp_path / "linked-cases.jsonl"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic links are not accepted"):
        load_cases(link)


def test_cli_applies_configured_input_limits() -> None:
    with pytest.raises(ValueError, match="record count exceeds 2"):
        main(
            [
                "examples/cases.jsonl",
                "examples/traces.jsonl",
                "--max-records",
                "2",
            ]
        )


@pytest.mark.parametrize(
    "field", ["max_file_bytes", "max_line_bytes", "max_records", "max_json_depth"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_jsonl_limits_require_positive_integers(field, value) -> None:
    values = {field: value}
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        JsonlLimits(**values)
