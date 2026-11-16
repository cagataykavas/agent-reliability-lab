from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_reliability.models import EvaluationCase
from agent_reliability.overlap import OverlapAuditError, audit_overlap, main


def case(case_id: str, prompt: str) -> EvaluationCase:
    return EvaluationCase(case_id=case_id, prompt=prompt)


def write_cases(path: Path, cases: list[EvaluationCase]) -> None:
    lines = [json.dumps({"case_id": item.case_id, "prompt": item.prompt}) for item in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_unrelated_splits_are_admitted() -> None:
    report = audit_overlap(
        [case("refund", "Check refund eligibility for order O-77")],
        [case("weather", "Find tomorrow's weather in Istanbul")],
    )

    assert report.admitted is True
    assert report.comparisons == 1
    assert report.overlap_count == 0
    assert report.findings == ()


def test_case_id_collision_is_rejected_even_when_prompts_differ() -> None:
    report = audit_overlap(
        [case("shared-id", "Look up order")],
        [case("shared-id", "Summarize account")],
    )

    assert report.admitted is False
    assert report.findings[0].reason == "case_id_collision"
    assert report.findings[0].similarity == 0.0


def test_unicode_and_formatting_normalization_finds_exact_prompt() -> None:
    report = audit_overlap(
        [case("reference", "Review  POLICY\u3000for Café")],
        [case("candidate", "review policy for cafe\u0301")],
    )

    assert report.admitted is False
    assert report.findings[0].reason == "exact_prompt_match"
    assert report.findings[0].similarity == 1.0


def test_near_duplicate_prompt_crosses_reviewed_threshold() -> None:
    report = audit_overlap(
        [case("reference", "Refund order O-77 after checking regional refund policy")],
        [case("candidate", "Refund order O-77 after checking regional return policy")],
        similarity_threshold=0.75,
    )

    assert report.admitted is False
    assert report.findings[0].reason == "near_duplicate_prompt"
    assert report.findings[0].similarity == pytest.approx(7 / 8, abs=1e-6)


def test_reviewed_overlap_allowance_can_admit_known_pair() -> None:
    report = audit_overlap(
        [case("reference", "Check refund policy")],
        [case("candidate", "Check refund policy")],
        max_overlaps=1,
    )

    assert report.admitted is True
    assert report.overlap_count == 1


def test_findings_are_bounded_without_hiding_total() -> None:
    reference = [case(f"reference-{index}", "duplicate prompt") for index in range(3)]
    candidate = [case(f"candidate-{index}", "duplicate prompt") for index in range(3)]

    report = audit_overlap(reference, candidate, max_findings=2)

    assert report.overlap_count == 9
    assert len(report.findings) == 2
    assert report.truncated_findings == 7


def test_comparison_budget_fails_closed() -> None:
    with pytest.raises(OverlapAuditError, match="exceeding"):
        audit_overlap(
            [case("reference-a", "one"), case("reference-b", "two")],
            [case("candidate-a", "three"), case("candidate-b", "four")],
            max_comparisons=3,
        )


def test_duplicate_ids_within_split_are_invalid() -> None:
    with pytest.raises(OverlapAuditError, match="duplicate case_id"):
        audit_overlap(
            [case("duplicate", "one"), case("duplicate", "two")],
            [case("candidate", "three")],
        )


def test_cli_returns_denial_code_and_json_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    write_cases(reference_path, [case("reference", "Check refund policy")])
    write_cases(candidate_path, [case("candidate", "Check refund policy")])

    assert main([str(reference_path), str(candidate_path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["admitted"] is False
    assert report["overlap_count"] == 1


def test_cli_returns_error_code_for_unsafe_workload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    write_cases(reference_path, [case("reference", "one")])
    write_cases(candidate_path, [case("candidate", "two"), case("other", "three")])

    assert (
        main(
            [
                str(reference_path),
                str(candidate_path),
                "--max-comparisons",
                "1",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert "exceeding" in error["error"]
