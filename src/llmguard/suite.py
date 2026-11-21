"""Generic, deterministic evaluation suites; legacy BANKING77 CLI is unchanged."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Protocol

import yaml


class Provider(Protocol):
    def predict(self, text: str) -> str: ...


class CompatibleProvider:
    """Chat completions adapter for OpenAI and explicitly configured compatible APIs."""

    def __init__(self, config: dict, sdk=None):
        self.config = config
        for name in ("name", "model", "api_key_env"):
            if not isinstance(config.get(name), str) or not config[name].strip():
                raise ValueError(f"provider.{name} is required")
        if config["name"] != "openai" and not config.get("base_url"):
            raise ValueError("Non-OpenAI providers require an explicit base_url")
        if sdk is None:
            from openai import OpenAI
            key = os.getenv(config["api_key_env"])
            if not key:
                raise ValueError(f"Missing environment variable: {config['api_key_env']}")
            sdk = OpenAI(api_key=key, base_url=config.get("base_url"),
                         timeout=30, max_retries=2)
        self.sdk = sdk

    def predict(self, text: str) -> str:
        response = self.sdk.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": self.config.get("system_prompt", "Answer the question directly.")},
                {"role": "user", "content": text},
            ],
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ValueError("Incomplete model response")
        return choice.message.content.strip()

    def close(self):
        self.sdk.close()


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Data must be JSON, JSONL or CSV")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Expected a nonempty list of objects")
    ids = [r.get("id") for r in rows]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Each row must have a unique nonempty string id")
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


def retrieval_scores(expected: list[str], predicted: list[str], k: int) -> dict:
    gold = set(expected)
    hits = [int(doc in gold) for doc in predicted[:k]]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {
        f"hit@{k}": float(any(hits)),
        f"recall@{k}": sum(hits) / len(gold),
        f"mrr@{k}": next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0),
        f"ndcg@{k}": sum(h / math.log2(i + 2) for i, h in enumerate(hits)) / ideal,
    }


def evaluate_suite(task: str, rows: list[dict], predictions: dict, k: int = 5) -> dict:
    if task not in {"classification", "qa", "retrieval"}:
        raise ValueError("Unknown task")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not rows:
        raise ValueError("Dataset is empty")
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows) or set(predictions) - ids:
        raise ValueError("Duplicate dataset IDs or predictions for unknown case IDs")
    canonical = []
    for row in rows:
        text, expected = row.get("input"), row.get("expected")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each case needs a nonempty input")
        if task == "retrieval":
            expected = ranked_ids(expected)
        elif not isinstance(expected, str) or not expected.strip():
            raise ValueError("Each case needs a nonempty expected answer")
        canonical.append({"id": row["id"], "input": text, "expected": expected})
    canonical.sort(key=lambda row: row["id"])
    if task == "classification":
        labels = {row["expected"] for row in canonical}
    results = []
    for row in canonical:
        expected = row["expected"]
        prediction = predictions.get(row["id"])
        error = None
        try:
            if task == "retrieval":
                prediction = ranked_ids(prediction, allow_empty=True)
                scores = retrieval_scores(expected, prediction, k)
            elif not isinstance(prediction, str) or not prediction.strip():
                raise ValueError("Missing or empty prediction")
            elif task == "classification":
                prediction = prediction.strip()
                if prediction not in labels:
                    raise ValueError("Unknown predicted label")
                scores = {"accuracy": float(prediction == expected)}
            else:
                scores = {"exact_match": float(tokens(expected) == tokens(prediction)),
                          "token_f1": text_f1(expected, prediction)}
        except (ValueError, TypeError):
            error = "invalid_or_missing_prediction"
            prediction = None
            if task == "retrieval":
                scores = retrieval_scores(expected, [], k)
            elif task == "classification":
                scores = {"accuracy": 0.0}
            else:
                scores = {"exact_match": 0.0, "token_f1": 0.0}
        results.append({**row, "prediction": prediction, "metrics": scores, "error": error})
    metrics = {key: mean(r["metrics"][key] for r in results) for key in results[0]["metrics"]}
    if task == "classification":
        f1 = []
        for label in sorted(labels):
            tp = sum(r["expected"] == label and r["prediction"] == label for r in results)
            fp = sum(r["expected"] != label and r["prediction"] == label for r in results)
            fn = sum(r["expected"] == label and r["prediction"] != label for r in results)
            f1.append(2 * tp / (2 * tp + fp + fn))
        metrics["macro_f1"] = mean(f1)
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":")).encode()).hexdigest()
    errors = sum(r["error"] is not None for r in results)
    return {"schema_version": 1, "task": task, "k": k if task == "retrieval" else None,
            "dataset_fingerprint": fingerprint, "total_cases": len(results),
            "error_count": errors, "metrics": metrics, "cases": results,
            "status": "FAIL" if errors else "PASS"}


def compare_suite(baseline: dict, candidate: dict, tolerances: dict) -> dict:
    for key in ("schema_version", "task", "k", "dataset_fingerprint"):
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
        deltas[metric] = {"baseline": old, "candidate": value, "delta": delta,
                          "tolerance": tolerance, "status": "FAIL" if delta < -tolerance - 1e-12 else "PASS"}
    candidate["comparison"] = deltas
    if any(d["status"] == "FAIL" for d in deltas.values()):
        candidate["status"] = "FAIL"
    return candidate


def render_html(run: dict) -> str:
    escape = lambda x: html.escape(json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x)
    rows = "".join(f"<tr><td>{escape(name)}</td><td>{value:.4f}</td></tr>"
                   for name, value in run["metrics"].items())
    comparison = "".join(f"<tr><td>{escape(name)}</td><td>{delta['baseline']:.4f}</td>"
                         f"<td>{delta['delta']:+.4f}</td><td>{escape(delta['status'])}</td></tr>"
                         for name, delta in run.get("comparison", {}).items())
    cases = "".join(f"<tr><td>{escape(r['id'])}</td><td>{escape(r['expected'])}</td>"
                    f"<td>{escape(r['prediction'])}</td><td>{escape(r['error'])}</td></tr>"
                    for r in run["cases"])
    return ("<!doctype html><html><head><meta charset='utf-8'><title>LLMGuard Suite</title>"
            "<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.5rem;text-align:left}"
            "td{overflow-wrap:anywhere}</style></head><body>"
            f"<h1>LLMGuard — {escape(run['task'])}</h1><h2>{escape(run['status'])}</h2>"
            f"<p>{run['total_cases']} cases; {run['error_count']} errors</p>"
            f"<h2>Metrics</h2><table><tr><th>Metric</th><th>Score</th></tr>{rows}</table>"
            f"<h2>Comparison</h2><table><tr><th>Metric</th><th>Baseline</th><th>Delta</th><th>Status</th></tr>{comparison}</table>"
            f"<h2>Cases</h2><table><tr><th>ID</th><th>Expected</th><th>Prediction</th><th>Error</th></tr>{cases}</table></body></html>")


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
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        root = args.config.parent
        task = cfg["task"]
        rows = load_rows(root / cfg["dataset"])
        # Validate ground truth before making any model requests.
        preflight = evaluate_suite(task, rows, {}, cfg.get("k", 5))
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
        run = evaluate_suite(task, rows, predictions, cfg.get("k", 5))
        run["source"] = metadata
        if baseline is not None:
            compare_suite(baseline, run, cfg.get("tolerances", {}))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        args.output.with_suffix(".html").write_text(render_html(run), encoding="utf-8")
        print(f"{run['status']}: {run['metrics']}")
        return 1 if run["status"] == "FAIL" else 0
    except Exception as exc:
        print(f"Suite execution/configuration error: {type(exc).__name__}", file=sys.stderr)
        return 2
