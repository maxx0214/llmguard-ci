"""Local evaluation of Python callables and compatible saved runs."""
from __future__ import annotations

import asyncio
import json
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from uuid import uuid4

from . import __version__
from .eval.aggregates import classification_macro_f1, percentile, sum_cost
from .execution import ExecutionOptions, RequestLimiter, invoke, request, run_workers
from .files import PredictionFile, map_dataset, read_records
from .metrics import Metric, summarize_metadata
from .output import (OutputMappingError, TASK_FIELDS, finite_number, json_snapshot, nonnegative, normalize_output,
                     prediction_for, validate_paths)
from .regression.gates import LOWER_IS_BETTER, GatePolicy
from .rest import RESTAdapter
from .statistics import StatisticsOptions, _require_statistics, case_changes, paired_statistics, validate_case_mean
from .storage.case_results import CaseResults, fingerprint
from .suite import evaluate_suite, validate_prediction, write_suite_report


_SCHEMA_VERSION = 1
_RETRIEVAL_METRIC = re.compile(r"(?:hit|recall|mrr|ndcg)@([1-9][0-9]*)")
_DEFAULT_METRICS = {"qa": ("exact_match", "token_f1"), "classification": ("accuracy", "macro_f1"),
                    "retrieval": ("hit@5", "recall@5", "mrr@5", "ndcg@5")}
_INVALID_PREDICTION = {"qa": "Output field 'answer' must be a valid answer string.",
                       "classification": "Output field 'label' must be a known dataset label.",
                       "retrieval": "Output field 'retrieved_ids' must contain unique nonempty string IDs."}
_CASE_PAYLOADS = {"input", "expected", "prediction", "output"}


@dataclass(frozen=True)
class System:
    """Attach output handling and reproducibility metadata to one callable."""

    predict: Callable[[Any], Any]
    output_mapping: Mapping[str, str] | None = None
    adapter: Callable[[Any], Any] | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    settings: Mapping | None = None
    is_api: bool = False


@dataclass(frozen=True)
class _LoadedPredictions:
    path: Path
    outputs: dict
    source: dict


class EvaluationResult:
    """A run or comparison, with optional local JSON and HTML export."""

    def __init__(self, data: dict, protected_paths: tuple[Path, ...] = ()):
        self._data = data
        self._protected_paths = protected_paths

    @property
    def status(self) -> str:
        return self._data["status"]

    @property
    def metrics(self) -> dict:
        return dict(self._data["metrics"])

    @property
    def cases(self) -> list[dict]:
        return deepcopy(self._data["cases"])

    def report(self, path: str | Path | None = None) -> dict:
        if path is not None:
            output = Path(path)
            if output.suffix.lower() != ".json":
                raise ValueError("Report path must have a .json extension")
            if {output.resolve(), output.with_suffix(".html").resolve()} & set(self._protected_paths):
                raise ValueError("Report cannot overwrite its dataset, predictions, baseline or execution state")
            write_suite_report(self._data, output)
        return deepcopy(self._data)


def _run_sync(method: Callable, *args, **kwargs):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(method(*args, **kwargs))
    raise RuntimeError("An event loop is already running. Await aevaluate() or acompare().")


def _usage_total(values: list[dict | None]) -> dict | None:
    if not values:
        return {}
    first = values[0]
    if first is None or any(value is None or value.keys() != first.keys() for value in values):
        return None
    totals = {key: sum(value[key] for value in values) for key in sorted(first)}
    return totals if all(nonnegative(value) for value in totals.values()) else None


def _payload(output: dict, prediction, field: str, latency_ms: float | None) -> dict:
    result = {"latency_ms": output.get("latency_ms", latency_ms), "usage": output.get("usage"), "cost": output.get("cost")}
    for name in ("answer", "label", "retrieved_ids", "contexts"):
        value = prediction if name == field and prediction is not None else output.get(name)
        if value is not None:
            try:
                result[name] = json_snapshot(value)
            except (TypeError, ValueError, RecursionError):
                # Optional application objects can be mapped to JSON by the caller.
                continue
    return result


class Evaluator:
    """Score a QA, classification or retrieval dataset using local suite metrics."""

    def __init__(self, dataset: str | Path | list[dict], task: str, metrics: Sequence[str] | None = None,
                 *, profile: str | None = None, output_mapping: Mapping[str, str] | None = None,
                 adapter: Callable[[Any], Any] | None = None, quality_gate: Mapping | None = None,
                 include_payloads: bool = False, execution: ExecutionOptions | Mapping | None = None,
                 cache: str | Path | None = None, statistics: StatisticsOptions | Mapping | bool | None = None,
                 dataset_mapping: Mapping[str, str] | None = None, dataset_rows_path: str | None = None,
                 custom_metrics: Mapping[str, Metric | Callable] | None = None,
                 metadata_fields: Sequence[str] = ()):
        if not isinstance(task, str) or task not in _DEFAULT_METRICS:
            raise ValueError("Task must be qa, classification or retrieval")
        if metrics is not None and (isinstance(metrics, str) or not isinstance(metrics, Sequence)):
            raise ValueError("metrics must be a nonempty sequence of metric names")
        if custom_metrics is not None and not isinstance(custom_metrics, Mapping):
            raise TypeError("custom_metrics must be a mapping")
        registered = {}
        reserved = LOWER_IS_BETTER | {name for names in _DEFAULT_METRICS.values() for name in names}
        for name, metric in (custom_metrics or {}).items():
            if not isinstance(name, str) or not name.strip() or name in reserved or _RETRIEVAL_METRIC.fullmatch(name):
                raise ValueError(f"Invalid or reserved custom metric name: {name!r}")
            registered[name] = metric if isinstance(metric, Metric) else Metric(metric)
        self._metrics = (*_DEFAULT_METRICS[task], *registered) if metrics is None else tuple(metrics)
        if (not self._metrics or any(not isinstance(name, str) for name in self._metrics)
                or len(self._metrics) != len(set(self._metrics))):
            raise ValueError("metrics must contain unique nonempty metric names")
        self._custom_metrics = {name: registered[name] for name in self._metrics if name in registered}
        if (not isinstance(metadata_fields, Sequence) or isinstance(metadata_fields, str)
                or any(not isinstance(name, str) or not name.strip() for name in metadata_fields)):
            raise ValueError("metadata_fields must be a sequence of nonempty names")
        self._metadata_fields = set(metadata_fields)
        quality = set(self._metrics) - LOWER_IS_BETTER - self._custom_metrics.keys()
        depths = set()
        for name in quality:
            if task == "retrieval" and (match := _RETRIEVAL_METRIC.fullmatch(name)):
                depths.add(int(match[1]))
            elif name not in _DEFAULT_METRICS[task]:
                raise ValueError(f"Unsupported metric for {task}: {name}")
        self._scoring = {"k": sorted(depths) or 5, "profile": profile, "structured_input": True}
        self._task = task
        self._quality_metrics = quality
        self._policy = GatePolicy(quality_gate, quality, {name: spec.direction for name, spec in self._custom_metrics.items()})
        self._mapping = validate_paths(output_mapping, output=True)
        self._adapter = adapter
        if adapter is not None and not callable(adapter):
            raise TypeError("adapter must be callable")
        if type(include_payloads) is not bool:
            raise TypeError("include_payloads must be a boolean")
        self._include_payloads = include_payloads
        if execution is not None and not isinstance(execution, (ExecutionOptions, Mapping)):
            raise TypeError("execution must be a mapping or ExecutionOptions")
        self._execution = execution if isinstance(execution, ExecutionOptions) else ExecutionOptions(**(execution or {}))
        self._cache = Path(cache).resolve() if cache is not None and not self._execution.independent_trials else None
        if statistics is True:
            statistics = StatisticsOptions()
        elif isinstance(statistics, Mapping):
            statistics = StatisticsOptions(**statistics)
        elif statistics is False:
            statistics = None
        if statistics is not None and not isinstance(statistics, StatisticsOptions):
            raise TypeError("statistics must be a boolean, mapping or StatisticsOptions")
        if statistics is not None and isinstance(statistics.min_effect, Mapping) and statistics.min_effect.keys() - quality:
            raise ValueError("Statistics min_effect must refer to selected quality metrics")
        self._statistics = statistics

        self._dataset_path = Path(dataset).resolve() if isinstance(dataset, (str, Path)) else None
        if self._dataset_path is None and dataset_rows_path is not None:
            raise ValueError("dataset_rows_path applies to a JSON dataset file")
        raw_rows = read_records(self._dataset_path, dataset_rows_path) if self._dataset_path is not None else dataset
        if not isinstance(raw_rows, list) or any(not isinstance(row, dict) for row in raw_rows):
            raise ValueError("Dataset must be a file or a list of case dictionaries")
        rows = json_snapshot(map_dataset(raw_rows, dataset_mapping))
        self._preflight = evaluate_suite(task, rows, {}, **self._scoring)
        self._rows = self._preflight["cases"]
        self._labels = {row["expected"] for row in self._rows} if task == "classification" else None
        self._class_indices = {label: index for index, label in enumerate(sorted(self._labels or []))}
        self._versions = {name: (f"suite-v1/{profile or 'default'}" if name in quality else "runtime-v1")
                          for name in sorted(quality | LOWER_IS_BETTER)}
        self._versions.update({name: f"custom-v1/{spec.direction}/{spec.version}" for name, spec in self._custom_metrics.items()})

    def _prepare(self, value):
        if isinstance(value, PredictionFile):
            return _LoadedPredictions(*value._load(self._task, {row["id"] for row in self._rows}))
        return self._system(value)

    def _system(self, value: System | Callable) -> System:
        spec = value if isinstance(value, System) else System(value)
        adapter = self._adapter if spec.adapter is None else spec.adapter
        if not callable(spec.predict) or (adapter is not None and not callable(adapter)):
            raise TypeError("A system and its custom adapter must be callable")
        for version in (spec.model_version, spec.prompt_version):
            if version is not None and (not isinstance(version, str) or not version.strip()):
                raise ValueError("System versions must be nonempty strings")
        mapping = self._mapping if spec.output_mapping is None else validate_paths(spec.output_mapping, output=True)
        if type(spec.is_api) is not bool or (spec.settings is not None and not isinstance(spec.settings, Mapping)):
            raise TypeError("System is_api must be a boolean and settings must be a JSON mapping")
        settings = json_snapshot(dict(spec.settings)) if spec.settings is not None else {}
        return System(spec.predict, mapping, adapter, spec.model_version, spec.prompt_version, settings, spec.is_api)

    async def _record(self, system: System, value, limiter: RequestLimiter) -> dict:
        response = await request(system, value, self._execution, limiter)
        output, prediction, error = {}, None, response.error
        if error is None:
            output, prediction, error = await self._adapt(system, response.value)
        return self._observation(output, prediction, error, response.latency_ms, response.attempts, response.api_calls)

    def _observation(self, output, prediction, error, latency, attempts, api_calls):
        record = {"error": error, "latency_ms": latency, "cost": output.get("cost"),
                  "reported_latency_ms": output.get("latency_ms"), "usage": output.get("usage"),
                  "prediction": prediction, "attempts": attempts, "api_calls": api_calls}
        if "metadata" in output:
            record["metadata"] = output["metadata"]
        if self._include_payloads:
            record["output"] = _payload(output, prediction, TASK_FIELDS[self._task], latency)
        return record

    async def _adapt(self, system: System, raw) -> tuple[dict, Any, dict | None]:
        try:
            if system.adapter is not None:
                raw = await invoke(system.adapter, raw)
            output = normalize_output(raw, self._task, system.output_mapping)
        except Exception as exc:
            return {}, None, {"code": "adapter_failed", "message": "Output adaptation failed.",
                              "exception_type": type(exc).__name__}
        return self._validate_output(output)

    def _validate_output(self, output):
        if not self._quality_metrics and not self._custom_metrics:
            return output, None, None
        try:
            prediction = prediction_for(output, self._task)
        except OutputMappingError as exc:
            error = {"code": "missing_or_invalid_output", "message": str(exc)} if self._quality_metrics else None
            return output, None, error
        try:
            prediction = validate_prediction(self._task, self._scoring["profile"], prediction, self._labels)
        except (ValueError, TypeError):
            error = {"code": "invalid_prediction", "message": _INVALID_PREDICTION[self._task]} if self._quality_metrics else None
            return output, None, error
        return output, prediction, None

    def _source(self, system: System) -> dict:
        source = system.predict.configuration() if isinstance(system.predict, RESTAdapter) else {"kind": "callable"}
        source.update(model_version=system.model_version, prompt_version=system.prompt_version,
                      output_mapping=dict(system.output_mapping), custom_adapter=system.adapter is not None,
                      system_settings_fingerprint=fingerprint(system.settings),
                      is_api=system.is_api or isinstance(system.predict, RESTAdapter))
        return source

    def _state_options(self, system: System | _LoadedPredictions, checkpoint, retry_failed: bool):
        path = Path(checkpoint).resolve() if checkpoint is not None else None
        if isinstance(system, _LoadedPredictions):
            if path is not None or retry_failed is not False:
                raise ValueError("Checkpoints and retry_failed apply to live systems")
            return None
        if (self._cache is not None or path is not None) and (system.model_version is None or system.prompt_version is None):
            raise ValueError("Caching and checkpoints require explicit model_version and prompt_version")
        if path is not None and self._cache in (path, path.with_suffix(path.suffix + ".lease")):
            raise ValueError("Cache and checkpoint must use separate files")
        if type(retry_failed) is not bool or (retry_failed and path is None):
            raise ValueError("retry_failed requires a checkpoint")
        return path

    def _run_fingerprint(self, contract, *, cache_enabled):
        return fingerprint({"contract": contract, "metrics": sorted(self._metrics), "versions": self._versions,
                            "gates": self._policy.settings(),
                            "statistics": asdict(self._statistics) if self._statistics else None,
                            "cache_enabled": cache_enabled, "dataset": self._preflight["dataset_fingerprint"]})

    async def _observations(self, system: System, source: dict, checkpoint: Path | None, retry_failed: bool):
        rest = isinstance(system.predict, RESTAdapter)
        state_contract = {"version": 2 if rest else 1, "source": source, "execution": asdict(self._execution),
                          "task": self._task, "profile": self._scoring["profile"], "labels": sorted(self._labels or []),
                          "requires_prediction": bool(self._quality_metrics),
                          "payloads": self._include_payloads}
        if self._custom_metrics or self._metadata_fields:
            state_contract.update(custom_metrics=bool(self._custom_metrics), metadata_fields=sorted(self._metadata_fields))
        run_fingerprint = self._run_fingerprint(state_contract, cache_enabled=self._cache is not None)
        limiter = RequestLimiter(self._execution.requests_per_second)
        locks = {}
        observations, origins = {}, {}
        cache_lookups = 0
        cache_namespace = fingerprint(state_contract)
        state_scope = system.predict._state_scope if rest else None
        with CaseResults(self._cache, checkpoint, run_fingerprint, state_scope) as state:
            if state.cache_scope is not None:
                cache_namespace = fingerprint([cache_namespace, state.cache_scope])
            completed = state.completed()
            if completed.keys() - {row["id"] for row in self._rows}:
                raise ValueError("Checkpoint contains unknown case IDs")

            async def process(row):
                nonlocal cache_lookups
                case_id = row["id"]
                previous = completed.get(case_id)
                force = previous is not None and previous["error"] is not None and retry_failed
                if previous is not None and not force:
                    observations[case_id], origins[case_id] = previous, "checkpoint"
                    return
                origin = "live"
                if state.cache is None:
                    observation = await self._record(system, row["input"], limiter)
                else:
                    key = fingerprint([cache_namespace, row["input"]])
                    # Coalesce identical successful predictions within this run.
                    async with locks.setdefault(key, asyncio.Lock()):
                        observation = None
                        if not force:
                            cache_lookups += 1
                            observation = state.lookup(key)
                        if observation is not None:
                            origin = "cache"
                        else:
                            observation = await self._record(system, row["input"], limiter)
                            if observation["error"] is None:
                                state.remember(key, observation)
                state.save(case_id, observation)
                observations[case_id], origins[case_id] = observation, origin

            await run_workers(self._rows, self._execution.max_concurrency, process)
        return observations, origins, cache_lookups, run_fingerprint

    def _score_custom(self, case):
        scores = {}
        for name, metric in self._custom_metrics.items():
            try:
                value = metric.score(deepcopy(case))
            except Exception as exc:
                raise ValueError(f"Custom metric {name!r} failed for case {case['id']!r}: {type(exc).__name__}") from None
            if isinstance(value, bool):
                value = float(value)
            if value is not None and not finite_number(value):
                raise ValueError(f"Custom metric {name!r} must return a finite number, bool or None")
            scores[name] = value
        return scores

    async def _evaluate(self, system: System | _LoadedPredictions, checkpoint: Path | None = None, retry_failed: bool = False) -> dict:
        started = perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        saved = isinstance(system, _LoadedPredictions)
        cache_enabled = self._cache is not None and not saved
        if saved:
            source = system.source
            observations = {case_id: self._observation(*self._validate_output(output), output.get("latency_ms"), 0, 0)
                            for case_id, output in system.outputs.items()}
            origins = dict.fromkeys(observations, "prediction_file")
            cache_lookups = 0
            run_fingerprint = self._run_fingerprint(source, cache_enabled=False)
        else:
            source = self._source(system)
            observations, origins, cache_lookups, run_fingerprint = await self._observations(system, source, checkpoint, retry_failed)
        predictions = {case_id: observation["prediction"] for case_id, observation in observations.items()}
        scoring_started = perf_counter()
        scores = evaluate_suite(self._task, self._rows, predictions, **self._scoring) if self._quality_metrics else self._preflight
        cases = []
        metadata_fields = self._metadata_fields | {name for observation in observations.values() for name in observation.get("metadata", {})}
        for case in scores["cases"]:
            case_id = case["id"]
            record = {key: value for key, value in observations[case_id].items() if key != "prediction"}
            record.update(id=case_id, execution_origin=origins[case_id])
            record["metrics"] = {name: value for name, value in case["metrics"].items() if name in self._quality_metrics}
            if metadata_fields:
                metadata = record.get("metadata", {})
                record["metadata"] = {name: metadata.get(name) for name in sorted(metadata_fields)}
            if self._custom_metrics:
                record["metrics"].update(self._score_custom({**case, "prediction": predictions[case_id],
                                                             "metrics": record["metrics"], "error": record["error"],
                                                             "metadata": record.get("metadata", {})}))
            if self._include_payloads:
                record.update(input=case["input"], expected=case["expected"], prediction=predictions[case_id])
            if self._task == "classification" and self._quality_metrics:
                record["class_truth_index"] = self._class_indices[case["expected"]]
                record["class_prediction_index"] = self._class_indices.get(case["prediction"])
                record["class_correct"] = bool(case["metrics"]["accuracy"])
            cases.append(record)
        errors = sum(case["error"] is not None for case in cases)
        latencies = [case["latency_ms"] for case in cases if case["latency_ms"] is not None]
        complete_latency = len(latencies) == len(cases)
        cost = sum_cost([case["cost"] for case in cases])
        operational = {"latency_ms": max(latencies) if complete_latency else None,
                       "latency_p95_ms": percentile(latencies, .95) if complete_latency else None,
                       "cost": cost if nonnegative(cost) else None, "error_rate": errors / len(cases)}
        custom_values = {name: [case["metrics"][name] for case in cases] for name in self._custom_metrics}
        custom_scores = {name: mean(values) if None not in values else None for name, values in custom_values.items()}
        all_metrics = {**scores["metrics"], **operational, **custom_scores}
        timing = {"elapsed_ms": (perf_counter() - started) * 1000,
                  "scoring_ms": (perf_counter() - scoring_started) * 1000}
        live = [case for case in cases if case["execution_origin"] == "live" and case["attempts"]]
        cached = [case for case in cases if case["execution_origin"] == "cache"]
        actual_cost = sum_cost([case["cost"] if case["attempts"] == 1 else None for case in live])
        mode = "prediction_file" if saved else "parallel" if self._execution.max_concurrency > 1 else "sequential"
        execution = {"mode": mode,
                     "settings": asdict(self._execution), "wall_time_ms": timing["elapsed_ms"],
                     "cache_enabled": cache_enabled, "cache_hits": len(cached), "cache_lookups": cache_lookups,
                     "cache_hit_rate": len(cached) / cache_lookups if cache_lookups else None,
                     "api_calls_saved": sum(case["api_calls"] > 0 for case in cached),
                     "checkpoint_enabled": checkpoint is not None, "checkpoint_reused": list(origins.values()).count("checkpoint"),
                     "request_attempts": sum(case["attempts"] for case in live), "api_calls": sum(case["api_calls"] for case in live),
                     "actual_cost": actual_cost if nonnegative(actual_cost) else None,
                     "actual_usage": _usage_total([case["usage"] if case["attempts"] == 1 else None for case in live])}
        return {
            "sdk_schema_version": _SCHEMA_VERSION, "run_id": str(uuid4()), "started_at": started_at,
            "task": self._task, "profile": self._scoring["profile"], "k": scores["k"],
            "dataset_fingerprint": self._preflight["dataset_fingerprint"], "metric_versions": self._versions,
            "run_fingerprint": run_fingerprint,
            "settings": {"metrics": list(self._metrics), "quality_gate": self._policy.settings(),
                         "include_payloads": self._include_payloads, "execution": execution["mode"]},
            "environment": {"llmguard_version": __version__, "python": platform.python_version(),
                            "os": platform.system(), "os_release": platform.release(), "architecture": platform.machine()},
            "source": source, "total_cases": len(cases), "error_count": errors, "cases": cases,
            "metrics": {name: all_metrics[name] for name in self._metrics}, "operational": operational,
            **({"metadata_summary": summarize_metadata(cases)} if metadata_fields else {}),
            "usage": _usage_total([case["usage"] for case in cases]),
            "timing": timing, "execution": execution,
        }

    def _baseline_metrics(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Invalid SDK baseline report")
        required = {"sdk_schema_version": _SCHEMA_VERSION, "task": self._task, "profile": self._scoring["profile"],
                    "dataset_fingerprint": self._preflight["dataset_fingerprint"], "metric_versions": self._versions,
                    "total_cases": len(self._rows), "k": self._preflight["k"]}
        for name, expected in required.items():
            if name not in data or data[name] != expected:
                raise ValueError(f"Incompatible SDK baseline: {name}")
        if data.get("error_count") != 0:
            raise ValueError("A baseline must be error-free")
        cases = data.get("cases")
        if (not isinstance(cases, list) or any(not isinstance(case, dict) or case.get("error") is not None for case in cases)
                or [case.get("id") for case in cases] != [case["id"] for case in self._rows]):
            raise ValueError("SDK baseline case records must match the dataset")
        if any(not isinstance(case.get("metrics"), dict) for case in cases):
            raise ValueError("Invalid SDK baseline case metrics")
        metrics, operational = data.get("metrics"), data.get("operational")
        if (not isinstance(metrics, dict) or metrics.keys() != set(self._metrics)
                or not isinstance(operational, dict) or operational.keys() != LOWER_IS_BETTER):
            raise ValueError("Invalid SDK baseline metrics")
        for name, value in {**metrics, **operational}.items():
            if value is None and name in {"cost", "latency_ms", "latency_p95_ms", *self._custom_metrics}:
                continue
            valid_number = finite_number(value) if name in self._custom_metrics else nonnegative(value)
            if (not valid_number or (name in self._quality_metrics | {"error_rate"} and value > 1)
                    or (name == "error_rate" and value != 0)):
                raise ValueError(f"Invalid SDK baseline metric: {name}")
        if any(metrics[name] != operational[name] for name in metrics.keys() & operational.keys()):
            raise ValueError("Inconsistent SDK baseline operational metrics")
        for name in self._quality_metrics - {"macro_f1"}:
            if not validate_case_mean(data, name):
                raise ValueError(f"Missing SDK baseline case scores for {name}")
        if "macro_f1" in self._quality_metrics and all("class_prediction_index" in case for case in cases):
            truth = [self._class_indices[row["expected"]] for row in self._rows]
            predictions = [case["class_prediction_index"] for case in cases]
            if any(type(index) is not int or not 0 <= index < len(self._class_indices) for index in predictions):
                raise ValueError("Invalid SDK baseline class predictions")
            if classification_macro_f1(truth, predictions) != metrics["macro_f1"]:
                raise ValueError("Aggregate and per-case scores disagree for macro_f1")
        return {**metrics, **operational}

    def _result(self, run: dict, baseline: dict | None = None, protected: tuple[Path | None, ...] = ()) -> EvaluationResult:
        previous = self._baseline_metrics(baseline) if baseline is not None else None
        observed = {**run["metrics"], **run["operational"]}
        run["status"], run["gates"], run["comparison"] = self._policy.assess(observed, previous)
        if baseline is not None:
            run["case_changes"] = case_changes(baseline, run, lower_is_better=self._policy.lower)
            if self._statistics is not None:
                started = perf_counter()
                truth = {row["id"]: self._class_indices[row["expected"]] for row in self._rows} if self._class_indices else None
                run["statistics"] = paired_statistics(baseline, run, self._statistics, classification_truth=truth)
                run["timing"]["statistical_analysis_ms"] = (perf_counter() - started) * 1000
                if self._statistics.enforce and run["statistics"]["regressions"]:
                    run["status"] = "FAIL"
            run["baseline"] = {key: baseline[key] for key in ("run_id", "source", "environment", "settings", "timing",
                                                               "metrics", "operational", "usage", "total_cases", "error_count",
                                                               "execution", "run_fingerprint", "metadata_summary")
                               if key in baseline}
            run["baseline"]["cases"] = baseline["cases"] if self._include_payloads else [
                {key: value for key, value in case.items() if key not in _CASE_PAYLOADS} for case in baseline["cases"]
            ]
        paths = tuple(path for path in (self._dataset_path, self._cache, *protected) if path is not None)
        return EvaluationResult(run, paths)

    async def aevaluate(self, system: System | Callable | PredictionFile, *, checkpoint: str | Path | None = None,
                        retry_failed: bool = False) -> EvaluationResult:
        spec = self._prepare(system)
        path = self._state_options(spec, checkpoint, retry_failed)
        source_path = spec.path if isinstance(spec, _LoadedPredictions) else None
        lease_path = path.with_suffix(path.suffix + ".lease") if path else None
        return self._result(await self._evaluate(spec, path, retry_failed), protected=(path, lease_path, source_path))

    def evaluate(self, system: System | Callable | PredictionFile, *, checkpoint: str | Path | None = None,
                 retry_failed: bool = False) -> EvaluationResult:
        return _run_sync(self.aevaluate, system, checkpoint=checkpoint, retry_failed=retry_failed)

    async def acompare(self, baseline: System | Callable | PredictionFile | EvaluationResult | str | Path,
                       candidate: System | Callable | PredictionFile, *, checkpoint: str | Path | None = None,
                       retry_failed: bool = False) -> EvaluationResult:
        if self._statistics is not None:
            _require_statistics()
        challenger = self._prepare(candidate)
        candidate_path = self._state_options(challenger, checkpoint, retry_failed)
        lease_path = candidate_path.with_suffix(candidate_path.suffix + ".lease") if candidate_path else None
        candidate_source_path = challenger.path if isinstance(challenger, _LoadedPredictions) else None
        baseline_paths = ()
        if isinstance(baseline, (str, Path)):
            baseline_path = Path(baseline).resolve()
            baseline_paths = (baseline_path,)
            previous = json.loads(baseline_path.read_text(encoding="utf-8"))
        elif isinstance(baseline, EvaluationResult):
            previous = baseline.report()
            baseline_paths = baseline._protected_paths
        else:
            incumbent = self._prepare(baseline)
            self._state_options(incumbent, None, False)
            if isinstance(incumbent, _LoadedPredictions):
                baseline_paths = (incumbent.path,)
            previous = await self._evaluate(incumbent)
        self._baseline_metrics(previous)  # Check before sending candidate requests.
        return self._result(await self._evaluate(challenger, candidate_path, retry_failed), previous,
                            (*baseline_paths, candidate_path, lease_path, candidate_source_path))

    def compare(self, baseline: System | Callable | PredictionFile | EvaluationResult | str | Path,
                candidate: System | Callable | PredictionFile, *, checkpoint: str | Path | None = None,
                retry_failed: bool = False) -> EvaluationResult:
        return _run_sync(self.acompare, baseline, candidate, checkpoint=checkpoint, retry_failed=retry_failed)
