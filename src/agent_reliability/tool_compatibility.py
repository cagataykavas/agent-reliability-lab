"""Fail-closed compatibility checks for versioned agent tool manifests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "agent-tool-manifest/v1"
REPORT_VERSION = "agent-tool-compatibility/v1"
MAX_MANIFEST_BYTES = 1_000_000

_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "title",
    "description",
    "default",
}
_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


class ToolCompatibilityError(ValueError):
    """Raised when a manifest is invalid or outside the supported contract."""


@dataclass(frozen=True, order=True)
class BreakingChange:
    tool: str
    path: str
    rule: str
    detail: str


@dataclass(frozen=True)
class CompatibilityReport:
    report_version: str
    compatible: bool
    baseline_sha256: str
    candidate_sha256: str
    checked_tool_count: int
    breaking_changes: tuple[BreakingChange, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: Path, *, max_bytes: int = MAX_MANIFEST_BYTES) -> dict[str, Any]:
    """Load a JSON manifest with size, symlink, duplicate-key and finite-number checks."""

    if path.is_symlink():
        raise ToolCompatibilityError(f"manifest must not be a symlink: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ToolCompatibilityError(f"cannot read manifest {path}: {exc}") from exc
    if size > max_bytes:
        raise ToolCompatibilityError(f"manifest exceeds {max_bytes} bytes: {path}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ToolCompatibilityError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ToolCompatibilityError(f"non-finite JSON number: {value}")

    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=reject_duplicate,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ToolCompatibilityError(f"invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ToolCompatibilityError("manifest root must be an object")
    return value


def audit_tool_compatibility(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_tools: int = 256,
    max_schema_depth: int = 16,
    max_schema_nodes: int = 10_000,
) -> CompatibilityReport:
    """Check whether every call accepted by a baseline tool is accepted by a candidate."""

    for label, manifest in (("baseline", baseline), ("candidate", candidate)):
        try:
            _canonical(manifest)
        except (TypeError, ValueError) as exc:
            raise ToolCompatibilityError(f"{label} manifest is not finite JSON: {exc}") from exc

    baseline_tools = _validate_manifest(
        baseline,
        label="baseline",
        max_tools=max_tools,
        max_depth=max_schema_depth,
        max_nodes=max_schema_nodes,
    )
    candidate_tools = _validate_manifest(
        candidate,
        label="candidate",
        max_tools=max_tools,
        max_depth=max_schema_depth,
        max_nodes=max_schema_nodes,
    )

    changes: list[BreakingChange] = []
    for name, baseline_schema in baseline_tools.items():
        candidate_schema = candidate_tools.get(name)
        if candidate_schema is None:
            changes.append(BreakingChange(name, "$", "tool_removed", "tool is absent"))
            continue
        _compare_schema(name, "$", baseline_schema, candidate_schema, changes)

    ordered = tuple(sorted(changes))
    return CompatibilityReport(
        report_version=REPORT_VERSION,
        compatible=not ordered,
        baseline_sha256=_digest(baseline),
        candidate_sha256=_digest(candidate),
        checked_tool_count=len(baseline_tools),
        breaking_changes=ordered,
    )


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    label: str,
    max_tools: int,
    max_depth: int,
    max_nodes: int,
) -> dict[str, dict[str, Any]]:
    if set(manifest) != {"manifest_version", "tools"}:
        raise ToolCompatibilityError(
            f"{label} manifest must contain only 'manifest_version' and 'tools'"
        )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ToolCompatibilityError(f"{label} manifest_version must be {MANIFEST_VERSION!r}")
    tools = manifest["tools"]
    if not isinstance(tools, list) or not tools:
        raise ToolCompatibilityError(f"{label} tools must be a non-empty array")
    if len(tools) > max_tools:
        raise ToolCompatibilityError(f"{label} tools exceeds the {max_tools} tool budget")

    validated: dict[str, dict[str, Any]] = {}
    nodes = [0]
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or set(tool) != {"name", "input_schema"}:
            raise ToolCompatibilityError(
                f"{label} tools[{index}] must contain only 'name' and 'input_schema'"
            )
        name = tool["name"]
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or not name.isprintable()
            or len(name) > 128
        ):
            raise ToolCompatibilityError(f"{label} tools[{index}].name is invalid")
        if name in validated:
            raise ToolCompatibilityError(f"{label} has duplicate tool name {name!r}")
        schema = tool["input_schema"]
        _validate_schema(
            schema,
            path=f"{label}.tools[{index}].input_schema",
            depth=0,
            max_depth=max_depth,
            nodes=nodes,
            max_nodes=max_nodes,
        )
        if schema.get("type") != "object":
            raise ToolCompatibilityError(f"{label} tool {name!r} input_schema must be an object")
        validated[name] = schema
    return validated


def _validate_schema(
    schema: Any,
    *,
    path: str,
    depth: int,
    max_depth: int,
    nodes: list[int],
    max_nodes: int,
) -> None:
    if not isinstance(schema, dict):
        raise ToolCompatibilityError(f"{path} must be an object")
    if depth > max_depth:
        raise ToolCompatibilityError(f"{path} exceeds schema depth budget {max_depth}")
    nodes[0] += 1
    if nodes[0] > max_nodes:
        raise ToolCompatibilityError(f"schema exceeds node budget {max_nodes}")
    unsupported = sorted(set(schema) - _SCHEMA_KEYS)
    if unsupported:
        raise ToolCompatibilityError(f"{path} uses unsupported keywords: {', '.join(unsupported)}")

    schema_type = schema.get("type")
    if schema_type not in _TYPES:
        raise ToolCompatibilityError(f"{path}.type must be one supported scalar type")

    properties = schema.get("properties")
    required = schema.get("required")
    additional = schema.get("additionalProperties")
    items = schema.get("items")

    if schema_type == "object":
        if properties is not None and not isinstance(properties, dict):
            raise ToolCompatibilityError(f"{path}.properties must be an object")
        if required is not None and (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)
            or len(set(required)) != len(required)
        ):
            raise ToolCompatibilityError(f"{path}.required must contain unique strings")
        if additional is not None and not isinstance(additional, bool):
            raise ToolCompatibilityError(f"{path}.additionalProperties must be boolean")
        property_names = set((properties or {}).keys())
        if any(not isinstance(name, str) for name in property_names):
            raise ToolCompatibilityError(f"{path}.properties keys must be strings")
        if not set(required or []).issubset(property_names):
            raise ToolCompatibilityError(f"{path}.required contains an undeclared property")
        for name, child in (properties or {}).items():
            _validate_schema(
                child,
                path=f"{path}.properties.{name}",
                depth=depth + 1,
                max_depth=max_depth,
                nodes=nodes,
                max_nodes=max_nodes,
            )
    elif any(value is not None for value in (properties, required, additional)):
        raise ToolCompatibilityError(f"{path} uses object keywords with type {schema_type!r}")

    if schema_type == "array":
        if items is None:
            raise ToolCompatibilityError(f"{path}.items is required for array schemas")
        _validate_schema(
            items,
            path=f"{path}.items",
            depth=depth + 1,
            max_depth=max_depth,
            nodes=nodes,
            max_nodes=max_nodes,
        )
    elif items is not None:
        raise ToolCompatibilityError(f"{path}.items requires type 'array'")

    _validate_schema_values(schema, path=path, schema_type=schema_type)


def _validate_schema_values(schema: dict[str, Any], *, path: str, schema_type: str) -> None:
    numeric_keys = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
    length_keys = {"minLength", "maxLength"}
    item_keys = {"minItems", "maxItems"}
    if numeric_keys & schema.keys() and schema_type not in {"number", "integer"}:
        raise ToolCompatibilityError(f"{path} uses numeric bounds with type {schema_type!r}")
    if length_keys & schema.keys() and schema_type != "string":
        raise ToolCompatibilityError(f"{path} uses string bounds with type {schema_type!r}")
    if item_keys & schema.keys() and schema_type != "array":
        raise ToolCompatibilityError(f"{path} uses array bounds with type {schema_type!r}")
    for key in ("title", "description"):
        if key in schema and not isinstance(schema[key], str):
            raise ToolCompatibilityError(f"{path}.{key} must be a string")
    if schema_type in {"object", "array"} and ("enum" in schema or "const" in schema):
        raise ToolCompatibilityError(f"{path} supports enum/const only for scalar types")

    for key in numeric_keys:
        if key in schema and (
            not isinstance(schema[key], (int, float))
            or isinstance(schema[key], bool)
            or not math.isfinite(schema[key])
        ):
            raise ToolCompatibilityError(f"{path}.{key} must be a finite number")
    for key in length_keys | item_keys:
        if key in schema and (
            not isinstance(schema[key], int) or isinstance(schema[key], bool) or schema[key] < 0
        ):
            raise ToolCompatibilityError(f"{path}.{key} must be a non-negative integer")
    if "enum" in schema and (
        not isinstance(schema["enum"], list)
        or not schema["enum"]
        or len({_canonical(item) for item in schema["enum"]}) != len(schema["enum"])
    ):
        raise ToolCompatibilityError(f"{path}.enum must contain unique values")
    for key in ("enum", "const"):
        values = (
            schema[key]
            if key == "enum" and key in schema
            else [schema[key]]
            if key in schema
            else []
        )
        if any(not _matches_type(value, schema_type) for value in values):
            raise ToolCompatibilityError(
                f"{path}.{key} contains a value outside type {schema_type!r}"
            )
    if schema_type in {"number", "integer"}:
        lower = _numeric_lower(schema, schema_type)
        upper = _numeric_upper(schema, schema_type)
        if (
            lower is not None
            and upper is not None
            and (lower[0] > upper[0] or (lower[0] == upper[0] and (lower[1] or upper[1])))
        ):
            raise ToolCompatibilityError(f"{path} has contradictory numeric bounds")
    for minimum, maximum in (("minLength", "maxLength"), ("minItems", "maxItems")):
        if minimum in schema and maximum in schema and schema[minimum] > schema[maximum]:
            raise ToolCompatibilityError(f"{path} has contradictory {minimum}/{maximum} bounds")


def _compare_schema(
    tool: str,
    path: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    changes: list[BreakingChange],
) -> None:
    baseline_type = baseline["type"]
    candidate_type = candidate["type"]
    if baseline_type != candidate_type and not (
        baseline_type == "integer" and candidate_type == "number"
    ):
        changes.append(
            BreakingChange(
                tool,
                path,
                "type_narrowed",
                f"type changed from {baseline_type} to {candidate_type}",
            )
        )
        return

    _compare_value_domain(tool, path, baseline, candidate, changes)
    if baseline_type == "object":
        _compare_object(tool, path, baseline, candidate, changes)
    elif baseline_type == "array":
        _compare_schema(tool, f"{path}[]", baseline["items"], candidate["items"], changes)


def _compare_value_domain(
    tool: str,
    path: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    changes: list[BreakingChange],
) -> None:
    baseline_domain = _finite_domain(baseline)
    candidate_domain = _finite_domain(candidate)
    if candidate_domain is not None and (
        baseline_domain is None or not baseline_domain.issubset(candidate_domain)
    ):
        rule = "const_changed" if "const" in candidate else "enum_narrowed"
        changes.append(BreakingChange(tool, path, rule, "candidate narrows allowed values"))

    schema_type = baseline["type"]
    if schema_type in {"number", "integer"}:
        if _lower_tightened(
            _numeric_lower(baseline, schema_type), _numeric_lower(candidate, candidate["type"])
        ):
            changes.append(
                BreakingChange(tool, path, "minimum_tightened", "candidate lower bound is stricter")
            )
        if _upper_tightened(
            _numeric_upper(baseline, schema_type), _numeric_upper(candidate, candidate["type"])
        ):
            changes.append(
                BreakingChange(tool, path, "maximum_tightened", "candidate upper bound is stricter")
            )
    elif schema_type == "string":
        _compare_count_bound(tool, path, baseline, candidate, "minLength", True, changes)
        _compare_count_bound(tool, path, baseline, candidate, "maxLength", False, changes)
    elif schema_type == "array":
        _compare_count_bound(tool, path, baseline, candidate, "minItems", True, changes)
        _compare_count_bound(tool, path, baseline, candidate, "maxItems", False, changes)


def _compare_object(
    tool: str,
    path: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    changes: list[BreakingChange],
) -> None:
    baseline_required = set(baseline.get("required", []))
    candidate_required = set(candidate.get("required", []))
    for name in sorted(candidate_required - baseline_required):
        changes.append(
            BreakingChange(
                tool, f"{path}.{name}", "required_added", "optional property became required"
            )
        )
    if baseline.get("additionalProperties", True) and not candidate.get(
        "additionalProperties", True
    ):
        changes.append(
            BreakingChange(tool, path, "additional_properties_closed", "open object became closed")
        )

    baseline_properties = baseline.get("properties", {})
    candidate_properties = candidate.get("properties", {})
    for name, baseline_child in baseline_properties.items():
        candidate_child = candidate_properties.get(name)
        child_path = f"{path}.{name}"
        if candidate_child is None:
            if not candidate.get("additionalProperties", True):
                changes.append(
                    BreakingChange(
                        tool, child_path, "property_removed", "property is no longer accepted"
                    )
                )
            continue
        _compare_schema(tool, child_path, baseline_child, candidate_child, changes)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "array":
        return isinstance(value, list)
    return isinstance(value, dict)


def _finite_domain(schema: dict[str, Any]) -> set[str] | None:
    if "const" in schema:
        return {_canonical(schema["const"])}
    if "enum" in schema:
        return {_canonical(value) for value in schema["enum"]}
    return None


def _numeric_lower(schema: dict[str, Any], schema_type: str) -> tuple[int | float, bool] | None:
    choices: list[tuple[int | float, bool]] = []
    if "minimum" in schema:
        choices.append((schema["minimum"], False))
    if "exclusiveMinimum" in schema:
        choices.append((schema["exclusiveMinimum"], True))
    selected = max(choices) if choices else None
    if selected is not None and schema_type == "integer":
        value, exclusive = selected
        return (math.floor(value) + 1 if exclusive else math.ceil(value), False)
    return selected


def _numeric_upper(schema: dict[str, Any], schema_type: str) -> tuple[int | float, bool] | None:
    choices: list[tuple[int | float, bool]] = []
    if "maximum" in schema:
        choices.append((schema["maximum"], False))
    if "exclusiveMaximum" in schema:
        choices.append((schema["exclusiveMaximum"], True))
    selected = min(choices, key=lambda item: (item[0], not item[1])) if choices else None
    if selected is not None and schema_type == "integer":
        value, exclusive = selected
        return (math.ceil(value) - 1 if exclusive else math.floor(value), False)
    return selected


def _lower_tightened(
    baseline: tuple[int | float, bool] | None, candidate: tuple[int | float, bool] | None
) -> bool:
    if candidate is None:
        return False
    if baseline is None:
        return True
    return candidate[0] > baseline[0] or (
        candidate[0] == baseline[0] and candidate[1] and not baseline[1]
    )


def _upper_tightened(
    baseline: tuple[int | float, bool] | None, candidate: tuple[int | float, bool] | None
) -> bool:
    if candidate is None:
        return False
    if baseline is None:
        return True
    return candidate[0] < baseline[0] or (
        candidate[0] == baseline[0] and candidate[1] and not baseline[1]
    )


def _compare_count_bound(
    tool: str,
    path: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    key: str,
    minimum: bool,
    changes: list[BreakingChange],
) -> None:
    candidate_value = candidate.get(key)
    if candidate_value is None:
        return
    baseline_value = baseline.get(key, 0 if minimum else math.inf)
    tightened = candidate_value > baseline_value if minimum else candidate_value < baseline_value
    if tightened:
        changes.append(
            BreakingChange(tool, path, f"{key}_tightened", f"candidate {key} is stricter")
        )
