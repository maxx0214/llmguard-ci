"""Generic, deterministic evaluation suites; legacy BANKING77 CLI is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Protocol

from ._privacy import private_request_logs
from .eval.aggregates import classification_macro_f1
from .files import index_records, read_records
from .regression.gates import degradation, exceeds_limit


SQUAD_PUNCTUATION = str.maketrans("", "", string.punctuation)


class Provider(Protocol):
    def predict(self, text: str) -> str: ...


class CompatibleProvider:
    """Chat completions adapter for OpenAI and explicitly configured compatible APIs."""

    def __init__(self, config: dict, sdk=None, *, max_retries: int = 2):
        self.config = config
        for name in ("name", "model", "api_key_env"):
            if not isinstance(config.get(name), str) or not config[name].strip():
                raise ValueError(f"provider.{name} is required")
        if config["name"] != "openai" and not config.get("base_url"):
            raise ValueError("Non-OpenAI providers require an explicit base_url")
        if sdk is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                if exc.name != "openai":
                    raise
                raise ValueError("Chat providers require pip install 'llmguard-ci[openai]'") from None
            key = os.getenv(config["api_key_env"])
            if not key:
                raise ValueError(f"Missing environment variable: {config['api_key_env']}")
            sdk = OpenAI(api_key=key, base_url=config.get("base_url"),
                         timeout=30, max_retries=max_retries)
        self.sdk = sdk
        self.last_usage = None

    def predict(self, text: str) -> str:
        self.last_usage = None
        options = {key: self.config[key] for key in ("max_completion_tokens", "reasoning_effort")
                   if key in self.config}
        with private_request_logs():
            response = self.sdk.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": self.config.get("system_prompt", "Answer the question directly.")},
                    {"role": "user", "content": text},
                ],
                **options,
            )
        self.last_usage = getattr(response, "usage", None)
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ValueError("Incomplete model response")
        return choice.message.content.strip()

    def close(self):
        self.sdk.close()


def load_rows(path: Path) -> list[dict]:
    rows = read_records(path)
    index_records(rows, label="data")
    return rows


def ranked_ids(value, *, allow_empty=False) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if (not isinstance(value, list) or (not value and not allow_empty)
            or any(not isinstance(x, str) or not x.strip() for x in value)
            or len(set(value)) != len(value)):
        raise ValueError("Ranked IDs must be a list of unique nonempty strings")
    return value


def tokens(value: str) -> list[str]:
    return unicodedata.normalize("NFKC", value).casefold().split()


def text_f1(expected: str, predicted: str) -> float:
    a, b = tokens(expected), tokens(predicted)
    overlap = sum((Counter(a) & Counter(b)).values())
    return 2 * overlap / (len(a) + len(b)) if a or b else 1.0


def squad_tokens(value: str) -> list[str]:
    value = value.lower().translate(SQUAD_PUNCTUATION)
    return re.sub(r"\b(a|an|the)\b", " ", value).split()


def squad_scores(expected: list[str], predicted: str) -> dict:
    prediction = squad_tokens(predicted)
    prediction_counts = Counter(prediction)
    references = [squad_tokens(reference) for reference in expected]
    overlaps = [sum((prediction_counts & Counter(reference)).values()) for reference in references]
    return {
        "exact_match": float(prediction in references),
        "token_f1": max(2 * overlap / (len(prediction) + len(reference)) if overlap else 0.0
                        for reference, overlap in zip(references, overlaps)),
    }


def relevance_grades(value) -> dict[str, int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or not value or any(
        not isinstance(doc, str) or not doc.strip() or type(grade) is not int or grade < 1
        for doc, grade in value.items()
    ):
        raise ValueError("Expected a mapping of document IDs to positive integer grades")
    return value


def retrieval_scores(expected: list[str] | dict[str, int], predicted: list[str], k: int) -> dict:
    gold = expected if isinstance(expected, dict) else dict.fromkeys(expected, 1)
    gains = [gold.get(doc, 0) for doc in predicted[:k]]
    ideal = sum(grade / math.log2(i + 2) for i, grade in enumerate(sorted(gold.values(), reverse=True)[:k]))
    return {
        f"hit@{k}": float(any(gains)),
        f"recall@{k}": sum(grade > 0 for grade in gains) / len(gold),
        f"mrr@{k}": next((1 / (i + 1) for i, grade in enumerate(gains) if grade), 0.0),
        f"ndcg@{k}": sum(grade / math.log2(i + 2) for i, grade in enumerate(gains)) / ideal,
    }


def validate_prediction(task: str, profile: str | None, prediction, labels: set[str] | None):
    if task == "retrieval":
        return ranked_ids(prediction, allow_empty=True)
    if not isinstance(prediction, str) or (profile != "squad_v1" and not prediction.strip()):
        raise ValueError("Missing or empty prediction")
    if task == "classification":
        prediction = prediction.strip()
        if prediction not in labels:
            raise ValueError("Unknown predicted label")
    return prediction


def evaluate_suite(task: str, rows: list[dict], predictions: dict, k: int | list[int] = 5,
                   *, profile: str | None = None, structured_input: bool = False) -> dict:
    if task not in {"classification", "qa", "retrieval"}:
        raise ValueError("Unknown task")
    if profile is not None and (task, profile) not in (("qa", "squad_v1"), ("retrieval", "beir")):
        raise ValueError("Unknown scoring profile for task")
    depths = k if task == "retrieval" and isinstance(k, list) else [k]
    if not depths or any(type(depth) is not int or depth < 1 for depth in depths):
        raise ValueError("k must contain positive integers")
    if len(set(depths)) != len(depths):
        raise ValueError("Retrieval depths must be unique")
    depths = sorted(depths)
    if not rows:
        raise ValueError("Dataset is empty")
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows) or set(predictions) - ids:
        raise ValueError("Duplicate dataset IDs or predictions for unknown case IDs")
    canonical = []
    for row in rows:
        text, expected = row.get("input"), row.get("expected")
        if not (isinstance(text, str) and text.strip()) and not (
            structured_input and isinstance(text, (dict, list)) and text
        ):
            raise ValueError("Each case needs a nonempty input")
        if task == "retrieval":
            expected = relevance_grades(expected) if profile == "beir" else ranked_ids(expected)
        elif profile == "squad_v1":
            if isinstance(expected, str):
                expected = json.loads(expected)
            if (not isinstance(expected, list) or not expected
                    or any(not isinstance(answer, str) or not answer.strip() for answer in expected)):
                raise ValueError("SQuAD cases require a nonempty list of reference answers")
        elif not isinstance(expected, str) or not expected.strip():
            raise ValueError("Each case needs a nonempty expected answer")
        canonical.append({"id": row["id"], "input": text, "expected": expected})
    canonical.sort(key=lambda row: row["id"])
    labels = {row["expected"] for row in canonical} if task == "classification" else None
    results = []
    for row in canonical:
        expected = row["expected"]
        prediction = predictions.get(row["id"])
        error = None
        try:
            prediction = validate_prediction(task, profile, prediction, labels)
            if task == "retrieval":
                scores = {key: value for depth in depths
                          for key, value in retrieval_scores(expected, prediction, depth).items()}
            elif task == "classification":
                scores = {"accuracy": float(prediction == expected)}
            elif profile == "squad_v1":
                scores = squad_scores(expected, prediction)
            else:
                scores = {"exact_match": float(tokens(expected) == tokens(prediction)),
                          "token_f1": text_f1(expected, prediction)}
        except (ValueError, TypeError):
            error = "invalid_or_missing_prediction"
            prediction = None
            if task == "retrieval":
                scores = {key: value for depth in depths
                          for key, value in retrieval_scores(expected, [], depth).items()}
            elif task == "classification":
                scores = {"accuracy": 0.0}
            else:
                scores = {"exact_match": 0.0, "token_f1": 0.0}
        results.append({**row, "prediction": prediction, "metrics": scores, "error": error})
    metrics = {key: mean(r["metrics"][key] for r in results) for key in results[0]["metrics"]}
    if task == "classification":
        metrics["macro_f1"] = classification_macro_f1([r["expected"] for r in results], [r["prediction"] for r in results])
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    errors = sum(r["error"] is not None for r in results)
    report_depth = depths if isinstance(k, list) else k
    run = {"schema_version": 1, "task": task, "k": report_depth if task == "retrieval" else None,
            "dataset_fingerprint": fingerprint, "total_cases": len(results),
            "error_count": errors, "metrics": metrics, "cases": results,
            "status": "FAIL" if errors else "PASS"}
    if profile is not None:
        run["profile"] = profile
    return run


def compare_suite(baseline: dict, candidate: dict, tolerances: dict) -> dict:
    for key in ("schema_version", "task", "k", "profile", "dataset_fingerprint"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Incompatible baseline: {key}")
    if baseline.get("error_count") != 0 or baseline.get("status") != "PASS":
        raise ValueError("Baseline must be a successful error-free run")
    if set(baseline["metrics"]) != set(candidate["metrics"]):
        raise ValueError("Incompatible metric set")
    if set(tolerances) - set(candidate["metrics"]):
        raise ValueError("Unknown metric in tolerances")
    deltas = {}
    for metric, value in candidate["metrics"].items():
        tolerance = tolerances.get(metric, 0.0)
        if (not isinstance(tolerance, (float, int)) or isinstance(tolerance, bool)
                or not math.isfinite(tolerance) or not 0 <= tolerance <= 1):
            raise ValueError("Tolerances must be finite values in [0, 1]")
        old = baseline["metrics"][metric]
        if not isinstance(old, (int, float)) or not math.isfinite(old) or not 0 <= old <= 1:
            raise ValueError("Invalid baseline metric")
        delta = value - old
        failed = exceeds_limit(degradation(old, value), tolerance)
        deltas[metric] = {"baseline": old, "candidate": value, "delta": delta,
                          "tolerance": tolerance, "status": "FAIL" if failed else "PASS"}
    candidate["comparison"] = deltas
    if any(d["status"] == "FAIL" for d in deltas.values()):
        candidate["status"] = "FAIL"
    return candidate


def render_html(run: dict) -> str:
    escape = lambda x: html.escape(json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x)

    def number(value, *, signed=False):
        return "unknown" if value is None else format(value, "+.4f" if signed else ".4f")

    def table(headers, rows):
        headings = "".join(f"<th scope='col'>{escape(name)}</th>" for name in headers)
        body = "".join(f"<tr><th scope='row'>{first}</th>"
                       + "".join(f"<td>{value}</td>" for value in rest) + "</tr>" for first, *rest in rows)
        return f"<div class='table-scroll'><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>"

    def section(title, content):
        return f"<section><h2>{escape(title)}</h2>{content}</section>"

    sdk = "sdk_schema_version" in run
    baseline = run.get("baseline")
    comparisons = run.get("comparison", {})
    comparing = baseline is not None or bool(comparisons)
    sides = (("Baseline", baseline), ("Candidate", run)) if baseline is not None else (
        ("Candidate" if comparing else "Run", run),)
    status_label = "Comparison status" if comparing else "Run status"
    summary = f"<p>{status_label} <strong>{escape(run['status'])}</strong> · {run['total_cases']} cases</p>"
    if baseline is not None:
        summary += table(("Run", "Model version", "Errors"), (
            (label, escape(data.get("source", {}).get("model_version") or "unknown"), escape(data["error_count"]))
            for label, data in sides))
    else:
        summary += f"<p>{run['error_count']} errors</p>"

    def score_row(name, label, score):
        if not comparing:
            return (label, number(score))
        change = comparisons[name]
        gate = "—" if sdk and not change["thresholds"] else escape(change["status"])
        return (label, number(change["baseline"]), number(score), number(change["delta"], signed=True),
                gate)

    score_headers = ("Metric", "Baseline", "Candidate", "Delta (Candidate − Baseline)", "Gate")
    metric_rows = [score_row(name, escape(name), score) for name, score in run["metrics"].items()
                   if name not in run.get("operational", {})]
    metrics = section("Metrics", table(score_headers if comparing else ("Metric", "Score"), metric_rows)) if metric_rows else ""
    operations = gates = ""
    if sdk:
        labels = {"latency_ms": "Max latency (ms)", "latency_p95_ms": "P95 latency (ms)",
                  "cost": "Cost (USD)", "error_rate": "Error rate"}
        ops = [score_row(name, escape(labels.get(name, name)), value) for name, value in run["operational"].items()]
        usage = tuple(escape(data["usage"]) if data.get("usage") is not None else "unknown" for _, data in sides)
        if comparing and baseline is None:
            usage = ("unknown", *usage)
        ops.append(("Usage", *usage, "—", "—") if comparing else ("Usage", *usage))
        operations = section("Operations", table(score_headers if comparing else ("Metric", "Value"), ops))
        gates = section("Candidate quality gates" if comparing else "Quality gates",
                        table(("Metric", "Value", "Thresholds", "Status"), (
                            (escape(labels.get(name, name)), number(gate["value"]), escape(gate["thresholds"]), escape(gate["status"]))
                            for name, gate in run["gates"].items())))

    def case_values(case):
        if case is None:
            return "unknown"
        fields = {"metadata": "Metadata", "latency_ms": "Latency (ms)", "reported_latency_ms": "Reported latency (ms)", "cost": "Cost (USD)",
                  "usage": "Usage", "execution_origin": "Origin", "attempts": "Attempts", "api_calls": "API calls"}
        if not sdk or run["settings"]["include_payloads"]:
            fields.update(input="Input", expected="Expected", prediction="Prediction", output="Output")
        measurements = "".join(f"<dt>{label}</dt><dd>{'unknown' if case[key] is None else escape(case[key])}</dd>"
                               for key, label in fields.items() if key in case)
        extra = f"<details><summary>Case details</summary><dl>{measurements}</dl></details>" if measurements else ""
        scores = "<br>".join(f"{escape(metric)} {number(score)}" for metric, score in case["metrics"].items())
        scores = f"<details><summary>Case scores</summary>{scores}</details>" if scores else ""
        error = f"<p>Error {escape(case['error'])}</p>" if case.get("error") is not None else ""
        return f"{scores}{error}{extra}"

    cases_by_side = [{case["id"]: case for case in data["cases"]} for _, data in sides]
    case_ids = dict.fromkeys(case_id for cases in reversed(cases_by_side) for case_id in cases)
    cases = section("Cases", table(("Case ID", *(label for label, _ in sides)), (
        (escape(case_id), *(case_values(items.get(case_id)) for items in cases_by_side)) for case_id in case_ids)))

    metadata_names = sorted({name for _, data in sides for name in data.get("metadata_summary", {})})
    metadata = section("Metadata", table(("Field", *(label for label, _ in sides)), (
        (escape(name), *(escape(data["metadata_summary"][name]) if name in data.get("metadata_summary", {})
                         else "unknown" for _, data in sides)) for name in metadata_names))) if metadata_names else ""

    fields = ("source", "settings", "timing", "api_usage", "environment", "run_id", "run_fingerprint")
    provenance = "<details><summary>Run details</summary>" + table(
        ("Field", *(label for label, _ in sides)), (
            (escape(name), *(escape(data[name]) if data.get(name) is not None else "—" for _, data in sides))
            for name in fields if any(data.get(name) is not None for _, data in sides)))
    shared = "".join(f"<dt>{escape(name)}</dt><dd>{escape(run[name])}</dd>"
                     for name in ("profile", "k", "dataset_fingerprint", "metric_versions") if run.get(name) is not None)
    provenance += f"<dl>{shared}</dl></details>"
    inference = ""
    if "statistics" in run:
        result = run["statistics"]
        entries = []
        for name, item in result["metrics"].items():
            interval = item["confidence_interval"]
            bounds = "unknown" if interval is None else f"[{number(interval[0])}, {number(interval[1])}]"
            p_value = "unknown" if item["p_value"] is None else f"{item['p_value']:.4g}"
            adjusted = "unknown" if item["adjusted_p_value"] is None else f"{item['adjusted_p_value']:.4g}"
            verdict = "detected" if item["regression"] else "not detected" if item["status"] == "ok" else "uncertain"
            entries.append((escape(name), number(item["effect_size"], signed=True), bounds, p_value, adjusted,
                            number(item["min_effect"]), escape(item["status"]), verdict))
        inference = section("Paired statistics", f"<p>{escape(result['settings'])}</p>" + table(
            ("Metric", "Effect", "Confidence interval", "p-value", "Adjusted p", "Minimum effect", "Evidence", "Regression"), entries))
    execution = ""
    if any("execution" in data for _, data in sides):
        entries = []
        for label, source_run in sides:
            data = source_run.get("execution")
            if data is None:
                entries.append((label, *("unknown",) * 8))
            else:
                usage = "unknown" if data["actual_usage"] is None else escape(data["actual_usage"])
                entries.append((label, escape(data["mode"]), number(data["wall_time_ms"]), number(data["cache_hit_rate"]),
                                escape(data["api_calls_saved"]), escape(data["checkpoint_reused"]), escape(data["api_calls"]),
                                number(data["actual_cost"]), usage))
        execution = section("Execution", table(("Run", "Mode", "Wall time (ms)", "Cache hit rate", "API calls saved",
                                                "Resumed cases", "API calls", "Actual cost", "Actual usage"), entries))
    changes = ""
    if run.get("case_changes"):
        sections = []
        for name, groups in run["case_changes"].items():
            before, after = len(groups["improved"]), len(groups["worsened"])
            rows_html = table(("Case", "Direction", "Delta"), (
                (escape(case["id"]), kind, number(case["delta"], signed=True))
                for kind in ("improved", "worsened") for case in groups[kind]))
            unknown = f" · {groups['unknown']} unknown" if groups.get("unknown") else ""
            sections.append(f"<details><summary>{escape(name)} · {before} improved · {after} worsened{unknown}</summary>"
                            f"{rows_html}</details>")
        changes = section("Case changes", "".join(sections))
    title = f"LLMGuard {'comparison' if comparing else 'evaluation'} · {escape(run['task'])}"
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{title}</title>"
            "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#182437}"
            "section{margin:1.6rem 0}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;"
            "font-variant-numeric:tabular-nums}td,th{border-bottom:1px solid #d8dee8;padding:.5rem;text-align:left;vertical-align:top}"
            "thead th{background:#eef2f7}td,dd{overflow-wrap:anywhere}"
            "summary{cursor:pointer;margin:.5rem 0}dt{font-weight:600}dd{margin:0 0 .4rem}</style></head><body>"
            f"<h1>{title}</h1>{summary}{metrics}{operations}{execution}{gates}{inference}{changes}{metadata}{cases}{provenance}</body></html>")


def write_suite_report(run: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".html").write_text(render_html(run), encoding="utf-8")


def _run_rest_suite(cfg: dict, root: Path, output: Path, baseline: Path | None) -> int:
    from .rest import RESTAdapter
    from .sdk import Evaluator, System

    evaluator_keys = {"profile", "metrics", "output_mapping", "quality_gate", "include_payloads", "execution", "statistics",
                      "dataset_mapping", "dataset_rows_path", "metadata_fields"}
    if cfg.keys() - {"task", "dataset", "rest", "model_version", "prompt_version", "system_settings", "cache", "checkpoint",
                     "retry_failed", *evaluator_keys}:
        raise ValueError("Unsupported REST suite configuration keys")
    cache = root / cfg["cache"] if cfg.get("cache") is not None else None
    checkpoint = root / cfg["checkpoint"] if cfg.get("checkpoint") is not None else None
    evaluator = Evaluator(root / cfg["dataset"], cfg["task"], cache=cache, **{
        key: cfg[key] for key in evaluator_keys if key in cfg
    })
    with RESTAdapter(**cfg["rest"]) as adapter:
        system = System(adapter, model_version=cfg.get("model_version"), prompt_version=cfg.get("prompt_version"),
                        settings=cfg.get("system_settings"))
        options = {"checkpoint": checkpoint, "retry_failed": cfg.get("retry_failed", False)}
        result = evaluator.evaluate(system, **options) if baseline is None else evaluator.compare(baseline, system, **options)
        result.report(output)
    print(f"{result.status}: {result.metrics}")
    return 1 if result.status == "FAIL" else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="llmguard suite")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Result JSON; HTML uses the same stem")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix.lower() != ".json":
            raise ValueError("Output must use a .json extension")
        if args.baseline and args.baseline.resolve() in {args.output.resolve(), args.output.with_suffix('.html').resolve()}:
            raise ValueError("Output must differ from baseline")
        config = args.config.read_text(encoding="utf-8")
        if args.config.suffix.lower() == ".json":
            cfg = json.loads(config)
        else:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                if exc.name != "yaml":
                    raise
                print("YAML configurations require pip install 'llmguard-ci[yaml]'", file=sys.stderr)
                return 2
            cfg = yaml.safe_load(config)
        root = args.config.parent
        if "rest" in cfg:
            if args.config.resolve() in {args.output.resolve(), args.output.with_suffix(".html").resolve()}:
                raise ValueError("Output must differ from configuration")
            return _run_rest_suite(cfg, root, args.output, args.baseline)
        task = cfg["task"]
        rows = load_rows(root / cfg["dataset"])
        # Validate ground truth before making any model requests.
        profile = cfg.get("profile")
        depths = cfg.get("k", 5)
        preflight = evaluate_suite(task, rows, {}, depths, profile=profile)
        baseline = None
        if args.baseline:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            compare_suite(baseline, preflight, cfg.get("tolerances", {}))
        protected = {args.config.resolve(), (root / cfg["dataset"]).resolve()}
        if "predictions" in cfg:
            protected.add((root / cfg["predictions"]).resolve())
        if protected & {args.output.resolve(), args.output.with_suffix(".html").resolve()}:
            raise ValueError("Output must not overwrite input data")
        if ("predictions" in cfg) == ("provider" in cfg):
            raise ValueError("Specify exactly one of predictions or provider")
        if "predictions" in cfg:
            predictions = {r["id"]: r.get("prediction") for r in load_rows(root / cfg["predictions"])}
            metadata = {"mode": "imported"}
        else:
            if task == "retrieval":
                raise ValueError("Retrieval suites require imported ranked predictions")
            adapter = CompatibleProvider(cfg["provider"])
            predictions = {}
            try:
                for row in rows:
                    try:
                        predictions[row["id"]] = adapter.predict(row["input"])
                    except Exception:
                        # Never leak SDK response bodies or credentials.
                        predictions[row["id"]] = None
            finally:
                adapter.close()
            metadata = {"mode": "live", "provider": cfg["provider"]["name"], "model": cfg["provider"]["model"],
                        "prompt_sha256": hashlib.sha256(cfg["provider"].get("system_prompt", "Answer the question directly.").encode()).hexdigest()}
        started = perf_counter()
        run = evaluate_suite(task, rows, predictions, depths, profile=profile)
        run["timing"] = {"scoring_seconds": perf_counter() - started}
        run["source"] = metadata
        if baseline is not None:
            compare_suite(baseline, run, cfg.get("tolerances", {}))
        write_suite_report(run, args.output)
        print(f"{run['status']}: {run['metrics']}")
        return 1 if run["status"] == "FAIL" else 0
    except Exception as exc:
        print(f"Suite execution/configuration error: {type(exc).__name__}", file=sys.stderr)
        return 2

