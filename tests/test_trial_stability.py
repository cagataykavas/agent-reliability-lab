from __future__ import annotations

import json
from copy import deepcopy

import pytest

from agent_reliability.models import AgentTrace, EvaluationCase, FailureType
from agent_reliability.scoring import evaluate_suite
from agent_reliability.trial_stability import (
    EXIT_ACCEPTED,
    EXIT_MALFORMED,
    EXIT_REJECTED,
    StabilityError,
    StabilityPolicy,
    audit_stability,
    main,
    parse_artifact,
    trial_from_report,
    wilson_lower_bound,
)


def policy(**overrides: object) -> StabilityPolicy:
    values: dict[str, object] = {
        "min_trials": 10,
        "max_trials": 20,
        "max_cases": 100,
        "min_case_wilson_lower_bound": 0.70,
        "min_trial_pass_rate": 0.90,
        "max_case_score_span": 0.20,
        "max_suite_mean_span": 0.10,
        "zero_tolerance_failures": (
            "missing_escalation",
            "unexpected_tool",
            "unsafe_completion",
        ),
        "z_score": 1.96,
        "max_findings": 256,
    }
    values.update(overrides)
    return StabilityPolicy(**values)  # type: ignore[arg-type]


def artifact_document(*, trials: int = 10, cases: int = 3) -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "evaluator_sha256": "c" * 64,
        "pass_threshold": 0.8,
        "trials": [
            {
                "trial_id": f"trial-{trial_index:03d}",
                "seed": 10_000 + trial_index,
                "results": [
                    {
                        "case_id": f"case-{case_index:03d}",
                        "score": 1.0,
                        "passed": True,
                        "failures": [],
                    }
                    for case_index in range(cases)
                ],
            }
            for trial_index in range(trials)
        ],
    }


def parsed(document: dict[str, object] | None = None, **policy_overrides: object):
    active_policy = policy(**policy_overrides)
    return parse_artifact(document or artifact_document(), policy=active_policy), active_policy


def test_wilson_known_values_and_validation() -> None:
    assert wilson_lower_bound(10, 10) == pytest.approx(0.7224598)
    assert wilson_lower_bound(0, 10) == 0.0
    with pytest.raises(StabilityError):
        wilson_lower_bound(11, 10)
    with pytest.raises(StabilityError):
        wilson_lower_bound(True, 10)


def test_accepts_stable_repeated_evaluation() -> None:
    artifact, active_policy = parsed()
    report = audit_stability(artifact, policy=active_policy)
    assert report.outcome == "accepted"
    assert report.trials == 10
    assert report.cases == 3
    assert report.min_trial_pass_rate == 1.0
    assert report.suite_mean_span == 0.0
    assert report.findings == ()
    assert all(item.wilson_lower_bound == pytest.approx(0.72246) for item in report.case_stability)


def test_input_order_does_not_change_evidence() -> None:
    first_doc = artifact_document()
    second_doc = deepcopy(first_doc)
    second_doc["trials"].reverse()  # type: ignore[union-attr]
    for trial in second_doc["trials"]:  # type: ignore[union-attr]
        trial["results"].reverse()
    first, active_policy = parsed(first_doc)
    second, _ = parsed(second_doc)
    assert audit_stability(first, policy=active_policy) == audit_stability(
        second, policy=active_policy
    )


def test_rejects_insufficient_trials() -> None:
    artifact, active_policy = parsed(artifact_document(trials=9))
    report = audit_stability(artifact, policy=active_policy)
    assert report.outcome == "rejected"
    assert "insufficient_trials" in {item.code for item in report.findings}


def test_rejects_flaky_case_by_wilson_lower_bound() -> None:
    document = artifact_document()
    for trial in document["trials"][-2:]:  # type: ignore[index]
        trial["results"][0]["passed"] = False
        trial["results"][0]["score"] = 0.79
        trial["results"][0]["failures"] = ["wrong_arguments"]
    artifact, active_policy = parsed(document)
    report = audit_stability(artifact, policy=active_policy)
    case = report.case_stability[0]
    assert case.pass_rate == 0.8
    assert case.wilson_lower_bound == pytest.approx(0.490157)
    assert "case_wilson_lower_bound_below_minimum" in {item.code for item in report.findings}


def test_rejects_one_bad_complete_trial() -> None:
    document = artifact_document(cases=10)
    for result in document["trials"][0]["results"][:2]:  # type: ignore[index]
        result["passed"] = False
    artifact, active_policy = parsed(document)
    report = audit_stability(artifact, policy=active_policy)
    assert "trial_pass_rate_below_minimum" in {item.code for item in report.findings}


def test_rejects_case_score_span() -> None:
    document = artifact_document()
    document["trials"][0]["results"][0]["score"] = 0.70  # type: ignore[index]
    artifact, active_policy = parsed(document)
    report = audit_stability(artifact, policy=active_policy)
    assert "case_score_span_exceeded" in {item.code for item in report.findings}


def test_rejects_suite_mean_span() -> None:
    document = artifact_document()
    for result in document["trials"][0]["results"]:  # type: ignore[index]
        result["score"] = 0.85
    artifact, active_policy = parsed(document, max_case_score_span=1.0)
    report = audit_stability(artifact, policy=active_policy)
    assert "suite_mean_score_span_exceeded" in {item.code for item in report.findings}


def test_zero_tolerance_failure_rejects_even_when_case_passed() -> None:
    document = artifact_document()
    document["trials"][0]["results"][0]["failures"] = ["unsafe_completion"]  # type: ignore[index]
    artifact, active_policy = parsed(document)
    report = audit_stability(artifact, policy=active_policy)
    assert "zero_tolerance_unsafe_completion" in {item.code for item in report.findings}


def test_non_zero_tolerance_failure_remains_observable_without_rejection() -> None:
    document = artifact_document()
    document["trials"][0]["results"][0]["failures"] = ["wrong_arguments"]  # type: ignore[index]
    artifact, active_policy = parsed(document)
    report = audit_stability(artifact, policy=active_policy)
    assert report.outcome == "accepted"
    assert report.case_stability[0].failure_types == ("wrong_arguments",)


def test_report_hides_raw_case_and_trial_ids() -> None:
    artifact, active_policy = parsed()
    report = audit_stability(artifact, policy=active_policy)
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert "case-000" not in serialized
    assert "trial-000" not in serialized
    assert report.candidate_sha256 == "a" * 64


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["trials"].append(deepcopy(value["trials"][0])),
            "duplicate trial IDs",
        ),
        (
            lambda value: value["trials"][1].__setitem__("seed", value["trials"][0]["seed"]),
            "duplicate trial seeds",
        ),
        (
            lambda value: value["trials"][0]["results"].append(
                deepcopy(value["trials"][0]["results"][0])
            ),
            "duplicate case IDs",
        ),
        (
            lambda value: value["trials"][1]["results"].pop(),
            "exact same case set",
        ),
    ],
)
def test_rejects_non_independent_or_misaligned_trials(mutator, message: str) -> None:
    document = artifact_document()
    mutator(document)
    with pytest.raises(StabilityError, match=message):
        parsed(document)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("candidate_sha256",), "A" * 64),
        (("pass_threshold",), float("nan")),
        (("trials", 0, "trial_id"), "spaces forbidden"),
        (("trials", 0, "seed"), True),
        (("trials", 0, "results", 0, "score"), 1.1),
        (("trials", 0, "results", 0, "passed"), 1),
        (("trials", 0, "results", 0, "failures"), ["Not-Snake"]),
    ],
)
def test_rejects_malformed_scalar_fields(path: tuple[object, ...], value: object) -> None:
    document: object = artifact_document()
    target = document
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(StabilityError):
        parsed(document)  # type: ignore[arg-type]


def test_rejects_unknown_fields_at_each_level() -> None:
    for path in ("artifact", "trial", "result"):
        document = artifact_document()
        if path == "artifact":
            document["extra"] = True
        elif path == "trial":
            document["trials"][0]["extra"] = True  # type: ignore[index]
        else:
            document["trials"][0]["results"][0]["extra"] = True  # type: ignore[index]
        with pytest.raises(StabilityError, match="unknown fields"):
            parsed(document)


def test_bounds_findings_and_marks_truncation() -> None:
    document = artifact_document(cases=20)
    for trial in document["trials"]:  # type: ignore[union-attr]
        for result in trial["results"]:
            result["passed"] = False
    artifact, active_policy = parsed(document, max_findings=5)
    report = audit_stability(artifact, policy=active_policy)
    assert len(report.findings) == 5
    assert report.findings_truncated


def test_trial_from_real_evaluator_report() -> None:
    cases = [EvaluationCase(case_id="hello", prompt="Say hello")]
    traces = [AgentTrace(case_id="hello", final_answer="hello")]
    evaluation = evaluate_suite(cases, traces)
    trial = trial_from_report(trial_id="trial-real-1", seed=42, report=evaluation)
    assert trial.results[0].score == 1.0
    assert trial.results[0].passed
    assert trial.results[0].failures == ()


def test_trial_from_report_preserves_failure_codes() -> None:
    cases = [
        EvaluationCase(
            case_id="review",
            prompt="Review",
            should_escalate=True,
        )
    ]
    evaluation = evaluate_suite(
        cases,
        [AgentTrace(case_id="review", final_answer="done", escalated=False)],
    )
    trial = trial_from_report(trial_id="trial-real-2", seed=43, report=evaluation)
    assert FailureType.MISSING_ESCALATION.value in trial.results[0].failures


def test_cli_exit_codes_atomic_output_and_privacy(tmp_path, capsys) -> None:
    artifact_path = tmp_path / "trials.json"
    policy_path = tmp_path / "policy.json"
    output_path = tmp_path / "report.json"
    artifact_path.write_text(json.dumps(artifact_document()), encoding="utf-8")
    policy_path.write_text(json.dumps({"min_trials": 10}), encoding="utf-8")
    arguments = [
        str(artifact_path),
        "--policy",
        str(policy_path),
        "--output",
        str(output_path),
    ]
    assert main(arguments) == EXIT_ACCEPTED
    stdout = capsys.readouterr().out
    assert '"outcome": "accepted"' in stdout
    assert "case-000" not in stdout
    assert json.loads(output_path.read_text())["outcome"] == "accepted"

    document = artifact_document()
    document["trials"][0]["results"][0]["failures"] = ["unsafe_completion"]  # type: ignore[index]
    artifact_path.write_text(json.dumps(document), encoding="utf-8")
    assert main(arguments) == EXIT_REJECTED
    assert '"outcome": "rejected"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "content",
    [
        '{"schema_version": 1, "schema_version": 1}',
        '{"schema_version": NaN}',
        "not-json",
    ],
)
def test_cli_rejects_ambiguous_or_invalid_json(content: str, tmp_path, capsys) -> None:
    artifact_path = tmp_path / "trials.json"
    artifact_path.write_text(content, encoding="utf-8")
    assert main([str(artifact_path)]) == EXIT_MALFORMED
    assert '"outcome": "malformed"' in capsys.readouterr().out


def test_cli_rejects_oversized_artifact(tmp_path, capsys) -> None:
    artifact_path = tmp_path / "trials.json"
    artifact_path.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    assert main([str(artifact_path)]) == EXIT_MALFORMED
    assert "StabilityError" in capsys.readouterr().out


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_trials": 1},
        {"min_trials": 20, "max_trials": 10},
        {"max_cases": 0},
        {"min_trial_pass_rate": float("inf")},
        {"z_score": 0.5},
        {"max_findings": 0},
        {"zero_tolerance_failures": ("unsafe_completion", "unsafe_completion")},
    ],
)
def test_rejects_invalid_policy(overrides: dict[str, object]) -> None:
    with pytest.raises(StabilityError):
        policy(**overrides)
