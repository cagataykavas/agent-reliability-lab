from agent_reliability.models import AgentTrace, EvaluationCase
from agent_reliability.registry import RunRegistry, fingerprint_dataset
from agent_reliability.scoring import evaluate_suite


def test_registry_round_trip_and_filters(tmp_path) -> None:
    report = evaluate_suite(
        [EvaluationCase(case_id="one", prompt="hello")],
        [AgentTrace(case_id="one", final_answer="hello")],
    )
    registry = RunRegistry(tmp_path / "runs.db")
    saved = registry.save(
        report,
        suite_name="support",
        candidate_name="agent-v2",
        dataset_fingerprint="sha256:test",
    )

    loaded = registry.get(saved.run_id)
    assert loaded == saved
    assert registry.list_runs(suite_name="support") == [saved]
    assert registry.list_runs(candidate_name="other") == []
    assert registry.list_runs(limit=1) == [saved]


def test_dataset_fingerprint_is_stable_and_content_sensitive() -> None:
    first = fingerprint_dataset([{"b": 2, "a": 1}])
    reordered = fingerprint_dataset([{"a": 1, "b": 2}])
    changed = fingerprint_dataset([{"a": 1, "b": 3}])
    assert first == reordered
    assert first != changed
    assert first.startswith("sha256:")
