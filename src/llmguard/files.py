"""Input files for offline evaluation and case-ID alignment."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .output import TASK_FIELDS, normalize_output, project, select, validate_paths


def _records(data, suffix, rows_path):
    if rows_path is not None and (not isinstance(rows_path, str) or not all(rows_path.split("."))):
        raise ValueError("rows_path must be a dot-separated path")
    if suffix == ".json":
        rows = json.loads(data)
        if rows_path is not None:
            rows = select(rows, rows_path.split("."))
    elif rows_path is not None:
        raise ValueError("rows_path applies to JSON files")
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    elif suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(data, newline="")))
    else:
        raise ValueError("Data must be JSON, JSONL or CSV")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected a nonempty list of records")
    return rows


def read_records(path: Path, rows_path: str | None = None) -> list[dict]:
    return _records(path.read_text(encoding="utf-8-sig"), path.suffix.lower(), rows_path)


def index_records(rows, *, label, id_path="id") -> dict:
    if not isinstance(id_path, str) or not all(id_path.split(".")):
        raise ValueError("id_path must be a dot-separated path")
    indexed = {}
    for position, row in enumerate(rows, 1):
        case_id = select(row, id_path.split("."))
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{label} record {position} requires a nonempty string ID")
        if case_id in indexed:
            raise ValueError(f"Duplicate {label} case ID {case_id!r}")
        indexed[case_id] = row
    return indexed


def map_dataset(rows, mapping):
    paths = {"id": "id", "input": "input", "expected": "expected"}
    overrides = validate_paths(mapping)
    if overrides.keys() - paths.keys():
        raise ValueError("dataset_mapping fields must be id, input or expected")
    paths.update(overrides)
    result = []
    for position, row in enumerate(rows, 1):
        projected = project(row, paths)
        if missing := paths.keys() - projected.keys():
            raise ValueError(f"Dataset record {position} is missing mapped fields: {', '.join(sorted(missing))}")
        result.append(projected)
    index_records(result, label="dataset")
    return result


@dataclass(frozen=True)
class PredictionFile:
    """Saved predictions in JSON, JSONL or CSV, joined to the dataset by ID."""

    path: str | Path
    prediction_mapping: Mapping[str, str] | None = None
    id_path: str = "id"
    rows_path: str | None = None
    model_version: str | None = None

    def _load(self, task: str, expected_ids: set[str]):
        if self.model_version is not None and (not isinstance(self.model_version, str) or not self.model_version.strip()):
            raise ValueError("Prediction file model_version must be a nonempty string")
        path = Path(self.path).resolve()
        raw = path.read_bytes()
        rows = _records(raw.decode("utf-8-sig"), path.suffix.lower(), self.rows_path)
        indexed = index_records(rows, label="prediction", id_path=self.id_path)
        missing, unexpected = expected_ids - indexed.keys(), indexed.keys() - expected_ids
        if missing or unexpected:
            raise ValueError(f"Prediction file IDs do not match dataset: missing ({len(missing)}) {sorted(missing)[:5]}, "
                             f"unexpected ({len(unexpected)}) {sorted(unexpected)[:5]}")
        field = TASK_FIELDS[task]
        mapping = validate_paths(self.prediction_mapping)
        if "prediction" in mapping:
            if field in mapping:
                raise ValueError(f"Map either prediction or {field}")
            mapping[field] = mapping.pop("prediction")
        mapping = validate_paths(mapping, output=True)
        predictions = {}
        for case_id, row in indexed.items():
            output = normalize_output(row, task, mapping)
            if field not in mapping and field not in output and "prediction" in row:
                value = row["prediction"]
                output = {**normalize_output(value, task, {}), **output}
                output.setdefault(field, value)
            if task == "retrieval" and isinstance(output.get(field), str) and path.suffix.lower() == ".csv":
                try:
                    output[field] = json.loads(output[field])
                except ValueError:
                    pass  # The normal prediction validator reports malformed rows.
            predictions[case_id] = output
        source = {"kind": "prediction_file", "model_version": self.model_version or path.stem,
                  "sha256": hashlib.sha256(raw).hexdigest(), "rows_path": self.rows_path,
                  "id_path": self.id_path, "prediction_mapping": mapping, "is_api": False}
        return path, predictions, source
