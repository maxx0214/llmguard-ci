"""Project arbitrary system outputs onto the fields understood by the evaluator."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


FIELDS = ("answer", "label", "retrieved_ids", "contexts", "latency_ms", "usage", "cost", "metadata")
TASK_FIELDS = {"qa": "answer", "classification": "label", "retrieval": "retrieved_ids"}
MISSING = object()


class OutputMappingError(ValueError):
    """A required standardized field could not be read."""


def validate_paths(mapping: Mapping[str, str] | None, *, output: bool = False) -> dict[str, str]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(path, str) or not all(path.split("."))
        for key, path in mapping.items()
    ):
        raise ValueError("Field mappings must contain nonempty names and dot-separated paths")
    if output and any(key not in FIELDS and not (key.startswith("metadata.") and key[9:]) for key in mapping):
        raise ValueError("Output mappings must use standardized field names")
    return dict(mapping)


def select(value: Any, parts: list[str]) -> Any:
    if not parts:
        return value
    key, *tail = parts
    if key == "*" and isinstance(value, (list, tuple)):
        found = [select(item, tail) for item in value]
        return MISSING if any(item is MISSING for item in found) else found
    if isinstance(value, Mapping):
        child = value.get(key, MISSING)
    elif isinstance(value, (list, tuple)):
        index = int(key) if key.isascii() and key.isdecimal() and len(key) <= 18 else -1
        child = value[index] if 0 <= index < len(value) else MISSING
    else:
        child = getattr(value, key, MISSING)
    return MISSING if child is MISSING else select(child, tail)


def project(value: Any, mapping: Mapping[str, str]) -> dict:
    return {field: found for field, path in mapping.items()
            if (found := select(value, path.split("."))) is not MISSING}


def project_output(value: Any, mapping: Mapping[str, str]) -> dict:
    fields = project(value, {key: path for key, path in mapping.items() if key in FIELDS})
    additions = {key[9:]: select(value, path.split(".")) for key, path in mapping.items() if key.startswith("metadata.")}
    if additions:
        base = fields.get("metadata", select(value, ["metadata"]) if "metadata" not in mapping else None)
        fields["metadata"] = {**(dict(base) if isinstance(base, Mapping) else {}),
                              **{key: None if item is MISSING else item for key, item in additions.items()}}
    return fields


def finite_number(value: Any) -> bool:
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def nonnegative(value: Any) -> bool:
    return finite_number(value) and value >= 0


def json_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def normalize_output(value: Any, task: str, mapping: Mapping[str, str]) -> dict:
    field = TASK_FIELDS[task]
    if isinstance(value, str) and task != "retrieval":
        output = {field: value}
    elif isinstance(value, (list, tuple)) and task == "retrieval":
        output = {field: value}
    else:
        output = {
            name: found for name in FIELDS if (found := select(value, [name])) is not MISSING
        }
    # An explicit path takes precedence, including when it is absent from the response.
    for name in mapping:
        output.pop(name, None)
    output.update(project_output(value, mapping))
    for name in ("cost", "latency_ms"):
        if not nonnegative(output.get(name)):
            output.pop(name, None)
    usage = output.get("usage")
    if not isinstance(usage, Mapping) or not usage or any(
        not isinstance(key, str) or not key or not nonnegative(amount) for key, amount in usage.items()
    ):
        output.pop("usage", None)
    else:
        output["usage"] = dict(usage)
    if output.get("metadata") is None:
        output.pop("metadata", None)
    else:
        metadata = output["metadata"]
        if not isinstance(metadata, Mapping) or any(not isinstance(key, str) or not key for key in metadata):
            raise OutputMappingError("Output metadata must be an object with nonempty string keys")
        output["metadata"] = json_snapshot(dict(metadata))
    return output


def prediction_for(output: dict, task: str):
    field = TASK_FIELDS[task]
    prediction = output.get(field)
    if prediction is None:
        raise OutputMappingError(f"Missing required output field '{field}'. Set output_mapping or an adapter.")
    if task == "retrieval" and isinstance(prediction, (list, tuple)):
        return list(prediction)
    if task != "retrieval" and isinstance(prediction, str):
        return prediction
    expected = "a list of document IDs" if task == "retrieval" else "a string"
    raise OutputMappingError(f"Output field '{field}' must be {expected}.")
