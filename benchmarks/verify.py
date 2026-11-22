"""Compare LLMGuard reports with the released evaluators and verify regression gates."""
import argparse
import html
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from llmguard.suite import compare_suite, evaluate_suite, load_rows, write_suite_report

from .download import SHA256, sha256
from .prepare import DEPTHS, read_json, write_json


def official_squad(root: Path, label: str, predictions: dict) -> dict:
    script = root / "raw/evaluate-v1.1.py"
    if sha256(script) != SHA256[script.name]:
        raise ValueError("SQuAD evaluator checksum mismatch")
    folder = root / "squad"
    answers = folder / f"{label}-official-answers.json"
    write_json(answers, {key: value for key, value in predictions.items() if value is not None})
    result = subprocess.run([sys.executable, "-X", "utf8", str(script), str(folder / "official-sample.json"),
                             str(answers)], check=True, text=True, capture_output=True)
    metrics = json.loads(result.stdout)
    return {"exact_match": metrics["exact_match"] / 100, "token_f1": metrics["f1"] / 100}


def official_beir(qrels: dict, predictions: dict) -> dict:
    from beir.retrieval.evaluation import EvaluateRetrieval

    # The official scorer accepts scores. Unique ordinal scores preserve the exact suite ranking.
    ranks = {query: {doc: -float(rank) for rank, doc in enumerate(docs)} for query, docs in predictions.items()}
    ndcg, _, recall, _ = EvaluateRetrieval.evaluate(qrels, ranks, DEPTHS, ignore_identical_ids=False)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, ranks, DEPTHS, metric="mrr")
    hit = EvaluateRetrieval.evaluate_custom(qrels, ranks, DEPTHS, metric="top_k_accuracy")
    return {f"{metric}@{k}": values[f"{name}@{k}"] for k in DEPTHS
            for metric, name, values in (("hit", "Accuracy", hit), ("recall", "Recall", recall),
                                         ("mrr", "MRR", mrr), ("ndcg", "NDCG", ndcg))}


def metric_agreement(metrics: dict, reference: dict, tolerance: float) -> dict:
    if metrics.keys() != reference.keys():
        raise ValueError("Framework and reference metric sets differ")
    differences = {key: abs(metrics[key] - reference[key]) for key in metrics}
    delta = max(differences.values())
    if delta > tolerance:
        raise ValueError(f"Reference mismatch: {differences}")
    return {"llmguard": metrics, "official": reference, "max_absolute_difference": delta,
            "absolute_tolerance": tolerance}


def check_gate(baseline: dict, candidate: dict) -> dict:
    expected_drops = sorted(key for key, value in candidate["metrics"].items()
                            if baseline["metrics"][key] - value > 1e-12)
    flagged = sorted(key for key, delta in candidate["comparison"].items() if delta["status"] == "FAIL")
    expected_status = "FAIL" if expected_drops or candidate["error_count"] else "PASS"
    if flagged != expected_drops or candidate["status"] != expected_status:
        raise ValueError(f"Unexpected regression gate: {flagged}, {candidate['status']}")
    return {"status": expected_status, "failed_metrics": flagged}


def verify_dataset(root: Path, dataset: str) -> dict:
    started = perf_counter()
    folder = root / dataset
    data_file = folder / "dataset.json"
    metadata = read_json(folder / "dataset-meta.json")
    if sha256(data_file) != metadata["dataset_sha256"]:
        raise ValueError("Prepared dataset checksum mismatch")
    rows = load_rows(data_file)
    qa = dataset == "squad"
    task = "qa" if qa else "retrieval"
    labels = ("baseline", "candidate") if qa else ("bm25", "dense")
    options = {"profile": "squad_v1"} if qa else {"profile": "beir", "k": DEPTHS}
    tolerance = 1e-12 if qa else 5.000001e-6  # BEIR publishes metrics rounded to five decimals.
    if qa:
        legacy_rows = [{**row, "expected": row["expected"][0]} for row in rows]
        evaluator = {"name": "SQuAD v1.1 evaluate-v1.1.py", "sha256": SHA256["evaluate-v1.1.py"]}
    else:
        qrels = read_json(folder / "qrels.json")
        legacy_rows = [{**row, "expected": list(row["expected"])} for row in rows]
        evaluator = {"name": "BEIR EvaluateRetrieval", "beir": version("beir"),
                     "pytrec_eval_terrier": version("pytrec-eval-terrier")}

    results, runs, saved_predictions = {}, {}, {}
    for label in labels:
        artifact = read_json(folder / f"{label}-predictions.json")
        predictions = ({case_id: record["prediction"] for case_id, record in artifact["records"].items()}
                       if qa else artifact["predictions"])
        saved_predictions[label] = predictions
        run = read_json(folder / f"{label}.json")
        runs[label] = run
        replay = evaluate_suite(task, rows[::-1], dict(reversed(list(predictions.items()))), **options)
        for key in ("cases", "dataset_fingerprint", "metrics", "error_count"):
            if run[key] != replay[key]:
                raise ValueError(f"Non-reproducible {label} report field: {key}")
        if run["error_count"]:
            raise ValueError(f"{label} has {run['error_count']} prediction errors")
        if qa and run["api_usage"]["responses_with_usage"] != run["total_cases"]:
            raise ValueError(f"{label} lacks usage data for one or more QA responses")

        official_started = perf_counter()
        reference = official_squad(root, label, predictions) if qa else official_beir(qrels, predictions)
        official_seconds = perf_counter() - official_started
        legacy = evaluate_suite(task, legacy_rows, predictions, **({} if qa else {"k": DEPTHS}))
        results[label] = {
            **metric_agreement(run["metrics"], reference, tolerance), "legacy_metrics": legacy["metrics"],
            "timing": {**run["timing"], "official_evaluator_seconds": official_seconds},
            "api_usage": run["api_usage"], "replay_identical": True,
        }
        repeat_path = folder / f"{label}-repeat-predictions.json"
        if not qa and repeat_path.exists():
            repeat = read_json(repeat_path)
            if repeat["predictions"] != predictions or repeat["source"] != artifact["source"]:
                raise ValueError(f"{label} repeated retrieval produced different rankings")
            results[label]["retrieval_repeat_identical"] = True
            results[label]["repeat_timing"] = repeat["timing"]

    baseline, candidate = (runs[label] for label in labels)
    replay_gate = check_gate(baseline, compare_suite(baseline, {**baseline}, {}))
    if replay_gate["status"] != "PASS":
        raise ValueError("Comparing a run to itself triggered the regression gate")
    checks = {"candidate": check_gate(baseline, candidate), "identical_predictions": replay_gate}
    if qa and checks["candidate"]["status"] != "FAIL":
        raise ValueError("The verbose QA candidate did not produce the intended degradation")
    if not qa:
        predictions = {key: docs[:1] for key, docs in saved_predictions["bm25"].items()}
        control = evaluate_suite(task, rows, predictions, **options)
        control_before_gate = official_beir(qrels, predictions)
        report = compare_suite(baseline, control, {})
        check = check_gate(baseline, report)
        if (check["status"] != "FAIL" or not check["failed_metrics"]
                or any(key.endswith("@1") for key in check["failed_metrics"]) or report["error_count"]):
            raise ValueError("Top-1 truncation did not trigger the expected retrieval regression")
        report["source"] = {"mode": "BM25 truncated to top 1"}
        write_suite_report(report, folder / "truncated-bm25.json")
        checks["truncated_bm25"] = {
            **check, **metric_agreement(report["metrics"], control_before_gate, tolerance),
        }

    return {"dataset": metadata, "official_evaluator": evaluator, "runs": results,
            "regression_checks": checks, "verification_seconds": perf_counter() - started}


def summary_html(verification: dict) -> str:
    sections = []
    for dataset, result in verification["datasets"].items():
        rows = []
        execution = []
        for label, run in result["runs"].items():
            for key, value in run["llmguard"].items():
                rows.append(f"<tr><td><a href='{dataset}/{label}.html'>{label}</a></td>"
                            f"<td>{html.escape(key)}</td><td>{value:.6f}</td>"
                            f"<td>{run['official'][key]:.6f}</td><td>{run['legacy_metrics'][key]:.6f}</td></tr>")
            timing, usage = run["timing"], run["api_usage"]
            prediction_seconds = (timing["api_request_seconds"] if dataset == "squad"
                                  else timing["index_seconds"] + timing["query_seconds"])
            cost = usage["estimated_cost_usd"]
            cost_label = "Unavailable" if cost is None else f"${cost:.6f}"
            execution.append(f"<tr><td>{label}</td><td>{prediction_seconds:.3f}</td>"
                             f"<td>{timing['scoring_seconds']:.4f}</td><td>{usage['requests']}</td>"
                             f"<td>{cost_label}</td><td>{run['max_absolute_difference']:.3g}</td></tr>")
        gates = []
        for name, gate in result["regression_checks"].items():
            metrics = html.escape(", ".join(gate["failed_metrics"]) or "None")
            gates.append(f"<li>{html.escape(name.replace('_', ' '))} — {gate['status']}. Failed metrics: {metrics}</li>")
        evaluator = html.escape(result["official_evaluator"]["name"])
        title = "SQuAD v1.1" if dataset == "squad" else "BEIR NFCorpus"
        sections.append(f"<section><h2>{title}</h2><p>Checked against {evaluator}.</p>"
                        "<table><thead><tr><th>Run</th><th>Metric</th><th>LLMGuard</th>"
                        f"<th>Official</th><th>Legacy default</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
                        "<h3>Execution and agreement</h3><table><thead><tr><th>Run</th><th>Prediction (s)</th>"
                        "<th>Scoring (s)</th><th>API requests</th><th>Estimated USD</th><th>Max score difference</th>"
                        f"</tr></thead><tbody>{''.join(execution)}</tbody></table>"
                        f"<h3>Regression gates</h3><ul>{''.join(gates)}</ul></section>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Public benchmark verification</title><style>body{font:16px system-ui;max-width:900px;"
            "margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}"
            "td,th{border:1px solid #d0d7de;padding:.4rem;text-align:left}section{margin:2rem 0}</style></head>"
            "<body><h1>Public benchmark verification</h1><p>Scores are reported on a 0–1 scale. "
            "<a href='verification.json'>Download verification JSON</a></p>"
            f"{''.join(sections)}</body></html>")


def verify(root: Path, datasets: list[str]) -> None:
    verification = {"python": platform.python_version(), "platform": platform.platform(), "datasets": {}}
    for dataset in datasets:
        result = verify_dataset(root, dataset)
        verification["datasets"][dataset] = result
        write_json(root / dataset / "verification.json", result)
        print(f"{dataset}: official metric parity and regression gates verified", flush=True)
    write_json(root / "verification.json", verification)
    (root / "verification.html").write_text(summary_html(verification), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", choices=("squad", "nfcorpus", "both"), default="both")
    args = parser.parse_args()
    verify(args.root, ["squad", "nfcorpus"] if args.dataset == "both" else [args.dataset])
