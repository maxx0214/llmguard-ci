"""Replay public predictions and measure SDK execution using local simulated requests."""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import random
import tempfile
from pathlib import Path
from statistics import median

import httpx

from llmguard import Evaluator, RESTAdapter, StatisticsOptions, System, paired_statistics
from llmguard.suite import evaluate_suite, load_rows

from .prepare import DEPTHS, read_json, write_json


SEED = 42
TRIALS = 12


def _predictions(folder, label):
    artifact = read_json(folder / f"{label}-predictions.json")
    if "records" in artifact:
        return {case_id: item["prediction"] for case_id, item in artifact["records"].items()}
    return artifact["predictions"]


def _public_dataset(root, dataset):
    folder = root / dataset
    qa = dataset == "squad"
    task, profile = ("qa", "squad_v1") if qa else ("retrieval", "beir")
    labels = ("baseline", "candidate") if qa else ("bm25", "dense")
    metric = "exact_match" if qa else "ndcg@10"
    rows = load_rows(folder / "dataset.json")
    before, after = (_predictions(folder, label) for label in labels)

    def score(sample, predictions, only=None):
        values = {row["id"]: predictions[row["id"]] for row in sample}
        run = evaluate_suite(task, sample, values, 5 if qa else DEPTHS, profile=profile)
        if only is not None:
            run["metrics"] = {only: run["metrics"][only]}
        return run

    def degrade(ids):
        replacement = "llmguard benchmark deliberately mismatched answer" if qa else []
        return {**before, **dict.fromkeys(ids, replacement)}

    baseline, candidate = score(rows, before), score(rows, after)
    for label, run in zip(labels, (baseline, candidate), strict=True):
        saved = read_json(folder / f"{label}.json")
        if run["metrics"] != saved["metrics"] or run["dataset_fingerprint"] != saved["dataset_fingerprint"]:
            raise ValueError(f"Could not reproduce saved {dataset}/{label} scores")
    sampled_ids = random.Random(SEED).sample(sorted(before), round(len(rows) * .4))
    options = StatisticsOptions(seed=SEED)
    same = paired_statistics(baseline, baseline, options)
    actual = paired_statistics(baseline, candidate, options)
    corrupted = score(rows, degrade(sampled_ids))
    degraded = paired_statistics(baseline, corrupted, options)
    if same["regressions"] or metric not in degraded["regressions"]:
        raise ValueError(f"Unexpected identical or degraded verdict for {dataset}")
    if degraded != paired_statistics(baseline, corrupted, options):
        raise ValueError("Statistical replay differs at fixed seed")

    sensitivity = []
    sizes = [12, 30, 60, len(rows)] if qa else [12, 30, 100, len(rows)]
    small_options = StatisticsOptions(seed=SEED, bootstrap_samples=300, permutation_samples=999)
    for size in sizes:
        by_fraction = {fraction: [] for fraction in (.05, .15, .35)}
        for trial in range(TRIALS):
            rng = random.Random(SEED + trial)
            sample = rng.sample(rows, size)
            control = score(sample, before, metric)
            order = rng.sample([row["id"] for row in sample], size)
            for fraction, results in by_fraction.items():
                changed = score(sample, degrade(order[:round(size * fraction)]), metric)
                results.append(paired_statistics(control, changed, small_options)["metrics"][metric])
        for fraction, items in by_fraction.items():
            sensitivity.append({"pairs": size, "corrupted_fraction": fraction, "trials": TRIALS,
                                "median_effect": median(item["effect_size"] for item in items),
                                "detections": sum(item["regression"] for item in items),
                                "uncertain": sum(item["status"] != "ok" for item in items)})
    return {"cases": len(rows), "dataset_fingerprint": baseline["dataset_fingerprint"], "metric": metric,
            "original_models": list(labels), "saved_scores_reproduced": True, "fixed_seed_reproduced": True,
            "identical": same, "original_comparison": actual, "degraded": degraded,
            "controlled_corrupted_cases": len(sampled_ids), "sensitivity": sensitivity}


async def _execution_benchmark():
    rows = [{"id": str(index), "input": str(index), "expected": "yes"} for index in range(64)]

    async def system(_):
        await asyncio.sleep(.012)
        return "yes"

    timing = {"sequential": [], "parallel": []}
    reference = None
    for _ in range(3):
        for name, concurrency in (("sequential", 1), ("parallel", 8)):
            run = await Evaluator(rows, "qa", execution={"max_concurrency": concurrency}).aevaluate(system)
            scores = [(case["id"], case["metrics"]) for case in run.cases]
            if reference is None:
                reference = scores
            elif scores != reference:
                raise ValueError("Parallel and sequential runs differ")
            timing[name].append(run.report()["execution"]["wall_time_ms"])
    medians = {name: median(values) for name, values in timing.items()}
    return {"cases": len(rows), "delay_ms": 12, "trials_ms": timing, "median_ms": medians,
            "parallel_workers": 8, "identical_results": True, "speedup": medians["sequential"] / medians["parallel"]}


def _cache_benchmark():
    rows = [{"id": str(index), "input": str(index), "expected": "yes"} for index in range(24)]
    requests = 0

    def simulate(_):
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"answer": "yes", "usage": {"simulated_tokens": 2}})

    with tempfile.TemporaryDirectory() as state, RESTAdapter(
        "https://example.invalid/evaluate", transport=httpx.MockTransport(simulate),
    ) as rest:
        evaluator = Evaluator(rows, "qa", cache=Path(state) / "cache.sqlite", execution={"max_concurrency": 8})
        system = System(rest, model_version="mock-v1", prompt_version="none")
        cold, warm = [evaluator.evaluate(system).report()["execution"] for _ in range(2)]
    if requests != len(rows) or warm["cache_hits"] != len(rows):
        raise ValueError("Cache did not suppress the expected simulated requests")
    return {"cases": len(rows), "transport": "httpx.MockTransport", "cold": cold, "warm": warm}


def _render(result):
    timing = result["execution"]
    seq, par = (timing["median_ms"][name] for name in ("sequential", "parallel"))
    warm = result["cache"]["warm"]
    tables = []
    for name, dataset in result["datasets"].items():
        metric = dataset["metric"]
        runs = "".join(
            f"<tr><td>{label}</td><td>{item['effect_size']:+.5f}</td>"
            f"<td>{html.escape(json.dumps(item['confidence_interval']))}</td><td>{item['adjusted_p_value']:.5g}</td>"
            f"<td>{item['regression']}</td></tr>"
            for label in ("identical", "original_comparison", "degraded") for item in [dataset[label]["metrics"][metric]]
        )
        sensitivity = "".join(
            f"<tr><td>{row['pairs']}</td><td>{row['corrupted_fraction']:.0%}</td><td>{row['median_effect']:+.5f}</td>"
            f"<td>{row['detections']}/{row['trials']}</td><td>{row['uncertain']}</td></tr>" for row in dataset["sensitivity"]
        )
        tables.append(f"<h2>{name}</h2><p>Primary metric {metric}</p><table><tr><th>Comparison</th><th>Effect</th>"
                      f"<th>95% interval</th><th>Holm p</th><th>Regression</th></tr>{runs}</table>"
                      "<h3>Sensitivity</h3><table><tr><th>Pairs</th><th>Corrupted</th><th>Median effect</th>"
                      f"<th>Detected</th><th>Uncertain</th></tr>{sensitivity}</table>")
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Efficiency and regression verification</title>"
            "<style>body{max-width:900px;margin:2rem auto;padding:0 1rem;font:16px system-ui}"
            "table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #ccc;padding:.4rem}</style>"
            "<h1>Efficiency and statistical regression verification</h1>"
            f"<p>Median local wall time was {seq:.1f} ms sequential and {par:.1f} ms parallel ({timing['speedup']:.2f}×). "
            f"The warm cache saved {warm['api_calls_saved']} simulated calls.</p>{''.join(tables)}</html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Folder with saved SQuAD and NFCorpus predictions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must end with .json")
    result = {"seed": SEED, "execution": asyncio.run(_execution_benchmark()), "cache": _cache_benchmark(),
              "datasets": {name: _public_dataset(args.root, name) for name in ("squad", "nfcorpus")}}
    write_json(args.output, result)
    args.output.with_suffix(".html").write_text(_render(result), encoding="utf-8")
    print(f"Saved benchmark evidence to {args.output}")


if __name__ == "__main__":
    main()
