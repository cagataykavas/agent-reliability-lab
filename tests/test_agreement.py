from __future__ import annotations

import json

import pytest

from agent_reliability.agreement import AgreementPolicy, Rating, evaluate_agreement
from agent_reliability.agreement_cli import main


def ratings(first: list[str], second: list[str]) -> list[Rating]:
    output: list[Rating] = []
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        case_id = f"case-{index:03d}"
        output.extend((Rating(case_id, "alice", left), Rating(case_id, "bob", right)))
    return output


def test_balanced_identical_labels_pass_agreement_gate():
    labels = ["pass", "fail"] * 12
    report = evaluate_agreement(ratings(labels, labels))
    pair = report.pairs[0]
    assert report.accepted is True
    assert pair.observed_agreement == 1.0
    assert pair.kappa == 1.0
    assert pair.reasons == ()


def test_chance_level_disagreement_is_rejected():
    first = ["pass", "pass", "fail", "fail"] * 6
    second = ["pass", "fail", "pass", "fail"] * 6
    report = evaluate_agreement(ratings(first, second))
    pair = report.pairs[0]
    assert report.accepted is False
    assert pair.kappa == 0.0
    assert pair.reasons == ("kappa_below_minimum",)
    assert len(report.disagreement_cases) == 12


def test_low_case_overlap_fails_even_with_perfect_shared_labels():
    values = [
        Rating(f"case-{index}", "alice", "pass" if index % 2 else "fail") for index in range(30)
    ]
    values.extend(
        Rating(f"case-{index}", "bob", "pass" if index % 2 else "fail") for index in range(20, 50)
    )
    report = evaluate_agreement(
        values,
        AgreementPolicy(min_shared_cases=5, min_overlap_ratio=0.8, min_kappa=0.6),
    )
    pair = report.pairs[0]
    assert pair.shared_cases == 10
    assert pair.union_cases == 50
    assert pair.kappa == 1.0
    assert pair.reasons == ("overlap_ratio_below_minimum",)


def test_small_shared_sample_is_rejected_independently_of_kappa():
    report = evaluate_agreement(
        ratings(["pass", "fail"], ["pass", "fail"]),
        AgreementPolicy(min_shared_cases=5, min_overlap_ratio=1.0, min_kappa=0.6),
    )
    pair = report.pairs[0]
    assert pair.kappa == 1.0
    assert pair.reasons == ("shared_cases_below_minimum",)


def test_single_label_prevalence_collapse_is_not_perfect_reliability():
    labels = ["pass"] * 24
    report = evaluate_agreement(ratings(labels, labels))
    pair = report.pairs[0]
    assert pair.observed_agreement == 1.0
    assert pair.kappa is None
    assert pair.reasons == ("kappa_undefined_prevalence_collapse",)
    assert report.accepted is False


def test_every_rater_pair_is_evaluated_and_disagreements_are_sorted():
    values = ratings(["pass", "fail"] * 10, ["pass", "fail"] * 10)
    values.extend(Rating(f"case-{index:03d}", "carol", "fail") for index in range(20))
    report = evaluate_agreement(
        values,
        AgreementPolicy(min_shared_cases=10, min_overlap_ratio=1.0, min_kappa=-1.0),
    )
    assert [pair.raters for pair in report.pairs] == [
        ("alice", "bob"),
        ("alice", "carol"),
        ("bob", "carol"),
    ]
    assert report.disagreement_cases == tuple(f"case-{index:03d}" for index in range(0, 20, 2))


def test_duplicate_rating_and_single_rater_input_fail_closed():
    duplicate = Rating("case-1", "alice", "pass")
    with pytest.raises(ValueError, match="duplicate rating"):
        evaluate_agreement([duplicate, duplicate, Rating("case-1", "bob", "pass")])
    with pytest.raises(ValueError, match="at least two raters"):
        evaluate_agreement([duplicate])


def test_rating_schema_is_strict():
    with pytest.raises(ValueError, match="exactly"):
        Rating.from_dict({"case_id": "case", "rater_id": "alice", "label": "pass", "score": 1})
    with pytest.raises(ValueError, match="trimmed"):
        Rating.from_dict({"case_id": " case", "rater_id": "alice", "label": "pass"})
    with pytest.raises(ValueError, match="non-empty"):
        Rating("case", "alice", "")


def test_cli_writes_rejected_report_and_returns_one(tmp_path):
    path = tmp_path / "ratings.jsonl"
    rows = ratings(["pass", "fail"] * 10, ["fail", "pass"] * 10)
    path.write_text(
        "".join(
            json.dumps({"case_id": item.case_id, "rater_id": item.rater_id, "label": item.label})
            + "\n"
            for item in rows
        ),
        encoding="utf-8",
    )
    output = tmp_path / "agreement.json"
    assert main([str(path), "--output", str(output)]) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["accepted"] is False
    assert payload["pairs"][0]["kappa"] == -1.0
