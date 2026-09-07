from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import EvaluationReport


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    run_id: str
    created_at: str
    suite_name: str
    candidate_name: str
    dataset_fingerprint: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    mean_score: float
    report: dict[str, Any]

    def to_dict(self, *, include_report: bool = True) -> dict[str, Any]:
        values: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "suite_name": self.suite_name,
            "candidate_name": self.candidate_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
        }
        if include_report:
            values["report"] = self.report
        return values


class RunRegistry:
    """Small SQLite persistence boundary for immutable evaluation runs."""

    def __init__(self, path: str | Path = "artifacts/evaluations.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save(
        self,
        report: EvaluationReport,
        *,
        suite_name: str,
        candidate_name: str,
        dataset_fingerprint: str,
    ) -> EvaluationRun:
        if not suite_name.strip() or not candidate_name.strip():
            raise ValueError("suite_name and candidate_name must not be blank")
        report_values = json.loads(json.dumps(report.to_dict(), sort_keys=True))
        run = EvaluationRun(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            suite_name=suite_name.strip(),
            candidate_name=candidate_name.strip(),
            dataset_fingerprint=dataset_fingerprint,
            total_cases=report.total_cases,
            passed_cases=report.passed_cases,
            pass_rate=round(report.pass_rate, 6),
            mean_score=round(report.mean_score, 6),
            report=report_values,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, created_at, suite_name, candidate_name, dataset_fingerprint,
                    total_cases, passed_cases, pass_rate, mean_score, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.created_at,
                    run.suite_name,
                    run.candidate_name,
                    run.dataset_fingerprint,
                    run.total_cases,
                    run.passed_cases,
                    run.pass_rate,
                    run.mean_score,
                    json.dumps(run.report, separators=(",", ":"), sort_keys=True),
                ),
            )
        return run

    def get(self, run_id: str) -> EvaluationRun | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _row_to_run(row) if row else None

    def list_runs(
        self,
        *,
        suite_name: str | None = None,
        candidate_name: str | None = None,
        limit: int = 50,
    ) -> list[EvaluationRun]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        clauses: list[str] = []
        parameters: list[Any] = []
        if suite_name is not None:
            clauses.append("suite_name = ?")
            parameters.append(suite_name)
        if candidate_name is not None:
            clauses.append("candidate_name = ?")
            parameters.append(candidate_name)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM evaluation_runs{where} ORDER BY created_at DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
        return [_row_to_run(row) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    suite_name TEXT NOT NULL,
                    candidate_name TEXT NOT NULL,
                    dataset_fingerprint TEXT NOT NULL,
                    total_cases INTEGER NOT NULL,
                    passed_cases INTEGER NOT NULL,
                    pass_rate REAL NOT NULL,
                    mean_score REAL NOT NULL,
                    report_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_suite_created
                ON evaluation_runs(suite_name, created_at DESC)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def fingerprint_dataset(values: list[dict[str, Any]]) -> str:
    canonical = json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _row_to_run(row: sqlite3.Row) -> EvaluationRun:
    return EvaluationRun(
        run_id=row["run_id"],
        created_at=row["created_at"],
        suite_name=row["suite_name"],
        candidate_name=row["candidate_name"],
        dataset_fingerprint=row["dataset_fingerprint"],
        total_cases=row["total_cases"],
        passed_cases=row["passed_cases"],
        pass_rate=row["pass_rate"],
        mean_score=row["mean_score"],
        report=json.loads(row["report_json"]),
    )
