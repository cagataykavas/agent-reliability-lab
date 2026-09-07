from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from .models import AgentTrace, EvaluationCase

T = TypeVar("T")


def load_cases(path: str | Path) -> list[EvaluationCase]:
    return _load_jsonl(path, EvaluationCase.from_dict)


def load_traces(path: str | Path) -> list[AgentTrace]:
    return _load_jsonl(path, AgentTrace.from_dict)


def _load_jsonl(path: str | Path, factory: Callable[[dict[str, Any]], T]) -> list[T]:
    source = Path(path)
    values: list[T] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(item, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        try:
            values.append(factory(item))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: invalid record: {error}") from error
    return values
