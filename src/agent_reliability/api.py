from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field
except ImportError as error:  # pragma: no cover - exercised by dependency-free installs
    raise RuntimeError("API dependencies are missing; install with: pip install -e '.[api]'") from error

from .models import AgentTrace, EvaluationCase
from .registry import RunRegistry, fingerprint_dataset
from .scoring import evaluate_suite


class EvaluationRequest(BaseModel):
    suite_name: str = Field(min_length=1, max_length=120)
    candidate_name: str = Field(min_length=1, max_length=120)
    cases: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)
    traces: list[dict[str, Any]] = Field(max_length=10_000)
    pass_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


def create_app(database_path: str | Path | None = None) -> FastAPI:
    path = Path(database_path or os.getenv("AGENT_RELIABILITY_DB", "artifacts/evaluations.db"))
    registry = RunRegistry(path)
    application = FastAPI(
        title="Agent Reliability Lab",
        version="0.3.0",
        description="Deterministic evaluation and run registry for recorded AI-agent traces.",
    )
    application.state.registry = registry

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/v1/evaluations", status_code=201)
    def create_evaluation(request: EvaluationRequest) -> dict[str, Any]:
        try:
            cases = [EvaluationCase.from_dict(item) for item in request.cases]
            traces = [AgentTrace.from_dict(item) for item in request.traces]
            report = evaluate_suite(cases, traces, pass_threshold=request.pass_threshold)
            run = registry.save(
                report,
                suite_name=request.suite_name,
                candidate_name=request.candidate_name,
                dataset_fingerprint=fingerprint_dataset(request.cases),
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return run.to_dict()

    @application.get("/v1/runs")
    def list_runs(
        suite_name: str | None = None,
        candidate_name: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        runs = registry.list_runs(
            suite_name=suite_name,
            candidate_name=candidate_name,
            limit=limit,
        )
        return {"runs": [run.to_dict(include_report=False) for run in runs]}

    @application.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = registry.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="evaluation run not found")
        return run.to_dict()

    return application


app = create_app()
