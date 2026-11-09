from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .models import AgentTrace, EvaluationCase

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class JsonlLimits:
    """Resource and structure budgets for one benchmark JSONL file."""

    max_file_bytes: int = 64 * 1024 * 1024
    max_line_bytes: int = 1024 * 1024
    max_records: int = 10_000
    max_json_depth: int = 32

    def __post_init__(self) -> None:
        for name, value in (
            ("max_file_bytes", self.max_file_bytes),
            ("max_line_bytes", self.max_line_bytes),
            ("max_records", self.max_records),
            ("max_json_depth", self.max_json_depth),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_JSONL_LIMITS = JsonlLimits()


def load_cases(
    path: str | Path, *, limits: JsonlLimits = DEFAULT_JSONL_LIMITS
) -> list[EvaluationCase]:
    return _load_jsonl(path, EvaluationCase.from_dict, limits=limits)


def load_traces(
    path: str | Path, *, limits: JsonlLimits = DEFAULT_JSONL_LIMITS
) -> list[AgentTrace]:
    return _load_jsonl(path, AgentTrace.from_dict, limits=limits)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key, value in pairs:
        if key in item:
            raise ValueError(f"duplicate JSON key {key!r}")
        item[key] = value
    return item


def _reject_non_finite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _validate_json_depth(value: Any, *, maximum: int) -> None:
    pending: list[tuple[Any, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        if depth > maximum:
            raise ValueError(f"JSON nesting depth exceeds {maximum}")
        if isinstance(current, dict):
            pending.extend(
                (child, depth + 1) for child in current.values() if isinstance(child, (dict, list))
            )
        elif isinstance(current, list):
            pending.extend(
                (child, depth + 1) for child in current if isinstance(child, (dict, list))
            )


def _load_jsonl(
    path: str | Path,
    factory: Callable[[dict[str, Any]], T],
    *,
    limits: JsonlLimits,
) -> list[T]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{source}: symbolic links are not accepted")
    values: list[T] = []
    total_bytes = 0
    with source.open("rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{source}: expected a regular file")
        if metadata.st_size > limits.max_file_bytes:
            raise ValueError(f"{source}: file exceeds {limits.max_file_bytes} byte input budget")

        line_number = 0
        while raw_line := stream.readline(limits.max_line_bytes + 2):
            line_number += 1
            total_bytes += len(raw_line)
            if total_bytes > limits.max_file_bytes:
                raise ValueError(
                    f"{source}: file exceeds {limits.max_file_bytes} byte input budget"
                )
            if len(raw_line) > limits.max_line_bytes:
                raise ValueError(
                    f"{source}:{line_number}: line exceeds {limits.max_line_bytes} byte budget"
                )
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid UTF-8") from error
            if not line or line.startswith("#"):
                continue
            if len(values) >= limits.max_records:
                raise ValueError(
                    f"{source}:{line_number}: record count exceeds {limits.max_records}"
                )
            try:
                item = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_non_finite_number,
                )
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            try:
                _validate_json_depth(item, maximum=limits.max_json_depth)
                values.append(factory(item))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: invalid record: {error}") from error
    return values
