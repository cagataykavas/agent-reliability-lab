from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent_reliability.tool_compat_cli import main
from agent_reliability.tool_compatibility import (
    MANIFEST_VERSION,
    ToolCompatibilityError,
    audit_tool_compatibility,
    load_manifest,
)

BASELINE = {
    "manifest_version": MANIFEST_VERSION,
    "tools": [
        {
            "name": "issue_refund",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "order_id": {"type": "string", "minLength": 1, "maxLength": 32},
                    "amount": {"type": "number", "minimum": 0, "maximum": 1000},
                    "reason": {"type": "string", "enum": ["damaged", "late"]},
                    "lines": {
                        "type": "array",
                        "maxItems": 10,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1},
                            },
                            "required": ["sku"],
                        },
                    },
                },
                "required": ["order_id", "amount"],
            },
        }
    ],
}


def candidate_schema() -> dict:
    return copy.deepcopy(BASELINE)


def schema(manifest: dict) -> dict:
    return manifest["tools"][0]["input_schema"]


def rules(report) -> set[str]:
    return {change.rule for change in report.breaking_changes}


def test_additive_and_relaxed_changes_are_compatible() -> None:
    candidate = candidate_schema()
    tool_schema = schema(candidate)
    tool_schema["properties"]["note"] = {"type": "string"}
    tool_schema["properties"]["amount"]["maximum"] = 2000
    tool_schema["properties"]["reason"]["enum"].append("other")
    tool_schema["properties"]["lines"]["items"]["properties"]["quantity"]["type"] = "number"

    report = audit_tool_compatibility(BASELINE, candidate)

    assert report.compatible
    assert report.breaking_changes == ()
    assert report.checked_tool_count == 1


def test_removed_tool_is_breaking() -> None:
    candidate = {
        "manifest_version": MANIFEST_VERSION,
        "tools": [{"name": "lookup_order", "input_schema": {"type": "object"}}],
    }

    report = audit_tool_compatibility(BASELINE, candidate)

    assert not report.compatible
    assert rules(report) == {"tool_removed"}


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda value: value["required"].append("reason"), "required_added"),
        (lambda value: value["properties"].pop("reason"), "property_removed"),
        (
            lambda value: value["properties"]["amount"].update(type="integer"),
            "type_narrowed",
        ),
        (
            lambda value: value["properties"]["reason"].update(enum=["damaged"]),
            "enum_narrowed",
        ),
        (
            lambda value: value["properties"]["amount"].update(minimum=10),
            "minimum_tightened",
        ),
        (
            lambda value: value["properties"]["order_id"].update(maxLength=8),
            "maxLength_tightened",
        ),
    ],
)
def test_breaking_changes_are_classified(mutate, expected: str) -> None:
    candidate = candidate_schema()
    mutate(schema(candidate))

    report = audit_tool_compatibility(BASELINE, candidate)

    assert expected in rules(report)


def test_nested_array_schema_is_checked_recursively() -> None:
    candidate = candidate_schema()
    item_schema = schema(candidate)["properties"]["lines"]["items"]
    item_schema["required"].append("quantity")

    report = audit_tool_compatibility(BASELINE, candidate)

    assert [(item.path, item.rule) for item in report.breaking_changes] == [
        ("$.lines[].quantity", "required_added")
    ]


def test_report_digest_is_independent_of_json_key_order() -> None:
    reordered = json.loads(json.dumps(BASELINE, sort_keys=True))

    report = audit_tool_compatibility(BASELINE, reordered)

    assert report.baseline_sha256 == report.candidate_sha256


def test_equivalent_cross_keyword_and_integer_bounds_are_compatible() -> None:
    baseline = candidate_schema()
    baseline_amount = schema(baseline)["properties"]["amount"]
    baseline_amount.pop("minimum")
    baseline_amount["exclusiveMinimum"] = 0
    baseline_quantity = schema(baseline)["properties"]["lines"]["items"]["properties"]["quantity"]
    baseline_quantity.pop("minimum")
    baseline_quantity["exclusiveMinimum"] = 0

    assert audit_tool_compatibility(baseline, BASELINE).compatible


def test_const_to_superset_enum_is_compatible() -> None:
    baseline = candidate_schema()
    baseline_reason = schema(baseline)["properties"]["reason"]
    baseline_reason.pop("enum")
    baseline_reason["const"] = "damaged"

    assert audit_tool_compatibility(baseline, BASELINE).compatible


def test_unsupported_keywords_and_budgets_fail_closed() -> None:
    unsupported = candidate_schema()
    schema(unsupported)["oneOf"] = []
    with pytest.raises(ToolCompatibilityError, match="unsupported keywords"):
        audit_tool_compatibility(BASELINE, unsupported)

    with pytest.raises(ToolCompatibilityError, match="depth budget"):
        audit_tool_compatibility(BASELINE, BASELINE, max_schema_depth=0)

    with pytest.raises(ToolCompatibilityError, match="tool budget"):
        audit_tool_compatibility(BASELINE, BASELINE, max_tools=0)

    contradictory = candidate_schema()
    schema(contradictory)["properties"]["amount"].update(minimum=10, maximum=5)
    with pytest.raises(ToolCompatibilityError, match="contradictory numeric bounds"):
        audit_tool_compatibility(BASELINE, contradictory)


@pytest.mark.parametrize(
    "contents, message",
    [
        ('{"manifest_version":"x","manifest_version":"y"}', "duplicate JSON key"),
        ('{"value": NaN}', "non-finite JSON number"),
    ],
)
def test_loader_rejects_ambiguous_json(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ToolCompatibilityError, match=message):
        load_manifest(path)


def test_loader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps(BASELINE), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ToolCompatibilityError, match="symlink"):
        load_manifest(link)


def test_in_memory_manifest_must_be_finite_json() -> None:
    candidate = candidate_schema()
    schema(candidate)["default"] = float("nan")

    with pytest.raises(ToolCompatibilityError, match="not finite JSON"):
        audit_tool_compatibility(BASELINE, candidate)


def test_cli_exit_codes_and_report_output(tmp_path: Path, capsys) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "artifacts" / "report.json"
    baseline_path.write_text(json.dumps(BASELINE), encoding="utf-8")
    candidate_path.write_text(json.dumps(BASELINE), encoding="utf-8")

    assert main([str(baseline_path), str(candidate_path), "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text())["compatible"] is True
    capsys.readouterr()

    breaking = candidate_schema()
    schema(breaking)["required"].append("reason")
    candidate_path.write_text(json.dumps(breaking), encoding="utf-8")
    assert main([str(baseline_path), str(candidate_path)]) == 1
    assert json.loads(capsys.readouterr().out)["breaking_changes"][0]["rule"] == "required_added"

    candidate_path.write_text('{"manifest_version":', encoding="utf-8")
    assert main([str(baseline_path), str(candidate_path)]) == 2
    assert "error" in json.loads(capsys.readouterr().out)

    candidate_path.write_text(json.dumps(BASELINE), encoding="utf-8")
    symlink_output = tmp_path / "report-link.json"
    symlink_output.symlink_to(output_path)
    assert main([str(baseline_path), str(candidate_path), "--output", str(symlink_output)]) == 2
    assert "symlink" in json.loads(capsys.readouterr().out)["error"]
