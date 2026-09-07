from fastapi.testclient import TestClient

from agent_reliability.api import create_app


def _payload() -> dict:
    return {
        "suite_name": "support-agent",
        "candidate_name": "candidate-v1",
        "cases": [{"case_id": "one", "prompt": "Say hello"}],
        "traces": [{"case_id": "one", "final_answer": "hello"}],
    }


def test_evaluation_lifecycle(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "api.db")) as client:
        assert client.get("/health").json() == {"status": "ok"}
        created = client.post("/v1/evaluations", json=_payload())
        assert created.status_code == 201
        run = created.json()
        assert run["pass_rate"] == 1.0
        assert run["dataset_fingerprint"].startswith("sha256:")

        listing = client.get("/v1/runs", params={"suite_name": "support-agent"})
        assert listing.status_code == 200
        assert listing.json()["runs"][0]["run_id"] == run["run_id"]
        assert "report" not in listing.json()["runs"][0]

        fetched = client.get(f"/v1/runs/{run['run_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["report"]["summary"]["passed_cases"] == 1


def test_api_rejects_unknown_trace_and_returns_not_found(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "api.db")) as client:
        payload = _payload()
        payload["traces"] = [{"case_id": "unknown", "final_answer": "hello"}]
        response = client.post("/v1/evaluations", json=payload)
        assert response.status_code == 422
        assert "unknown cases" in response.json()["detail"]
        assert client.get("/v1/runs/missing").status_code == 404
