"""Paired regression inference on existing per-case evaluation results."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isclose
from statistics import mean

from .eval.aggregates import classification_macro_f1
from .output import nonnegative
from .regression.gates import LOWER_IS_BETTER


_BINARY_SCORES = {"accuracy", "exact_match"}


@dataclass(frozen=True)
class StatisticsOptions:
    seed: int = 42
    confidence: float = 0.95
    alpha: float = 0.05
    bootstrap_samples: int = 2000
    permutation_samples: int = 2000
    min_pairs: int = 20
    min_effect: float | Mapping[str, float] = 0.01
    correction: str = "holm"
    enforce: bool = False

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Statistics seed must be a nonnegative integer")
        for name in ("confidence", "alpha"):
            value = getattr(self, name)
            if not nonnegative(value) or not 0 < value < 1:
                raise ValueError(f"Statistics {name} must be between 0 and 1")
        for name in ("bootstrap_samples", "permutation_samples", "min_pairs"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"Statistics {name} must be an integer of at least 2")
        effects = self.min_effect.values() if isinstance(self.min_effect, Mapping) else (self.min_effect,)
        if any(not nonnegative(value) or value > 1 for value in effects):
            raise ValueError("Statistics min_effect values must be in [0, 1]")
        if isinstance(self.min_effect, Mapping):
            if any(not isinstance(name, str) or not name for name in self.min_effect):
                raise ValueError("Statistics min_effect requires metric names")
            object.__setattr__(self, "min_effect", dict(self.min_effect))
        if self.correction not in ("none", "holm", "bonferroni") or type(self.enforce) is not bool:
            raise ValueError("Statistics correction or enforce setting is invalid")


def _pairs(baseline: dict, candidate: dict) -> list[tuple[dict, dict]]:
    if not baseline.get("dataset_fingerprint") or any(
        baseline.get(key) != candidate.get(key) for key in ("dataset_fingerprint", "task", "profile")
    ):
        raise ValueError("Paired comparisons require the same dataset, task and scoring profile")
    before, after = ({case["id"]: case for case in run["cases"]} for run in (baseline, candidate))
    if (not before or len(before) != len(baseline["cases"]) or len(after) != len(candidate["cases"])
            or before.keys() != after.keys()):
        raise ValueError("Paired comparisons require matching unique case IDs")
    return [(before[case_id], after[case_id]) for case_id in sorted(before)]


def case_changes(baseline: dict, candidate: dict, *, lower_is_better=()) -> dict:
    changes = {}
    for old, new in _pairs(baseline, candidate):
        old_metrics, new_metrics = (dict(case["metrics"]) for case in (old, new))
        if "class_correct" in old and "class_correct" in new:
            old_metrics.setdefault("accuracy", float(old["class_correct"]))
            new_metrics.setdefault("accuracy", float(new["class_correct"]))
        for name in old_metrics.keys() & new_metrics.keys():
            before, after = old_metrics[name], new_metrics[name]
            by_metric = changes.setdefault(name, {"improved": [], "worsened": [], "unchanged": 0})
            if before is None or after is None:
                by_metric["unknown"] = by_metric.get("unknown", 0) + 1
                continue
            if isclose(before, after, rel_tol=0, abs_tol=1e-12):
                by_metric["unchanged"] += 1
            else:
                improved = after < before if name in lower_is_better else after > before
                direction = "improved" if improved else "worsened"
                by_metric[direction].append({"id": new["id"], "baseline": before, "candidate": after, "delta": after - before})
    return dict(sorted(changes.items()))


def _batches(total, n):
    size = max(1, min(256, 1_000_000 // max(1, n)))
    for start in range(0, total, size):
        yield start, min(size, total - start)


def _bootstrap(np, n, statistic, options):
    rng = np.random.default_rng(options.seed)
    values = np.concatenate([statistic(rng.integers(n, size=(size, n)))
                             for _, size in _batches(options.bootstrap_samples, n)])
    tail = (1 - options.confidence) / 2
    return [float(value) for value in np.quantile(values, [tail, 1 - tail])]


def _permutation(np, changed, statistic, observed, options):
    if not changed:
        return 1.0, True
    exact = changed < options.permutation_samples.bit_length() and 1 << changed <= options.permutation_samples
    total = (1 << changed) if exact else options.permutation_samples
    rng = np.random.default_rng(options.seed + 1)
    hits = 0
    for start, size in _batches(total, changed):
        if exact:
            swaps = ((np.arange(start, start + size)[:, None] >> np.arange(changed)) & 1).astype(bool)
        else:
            swaps = rng.integers(0, 2, size=(size, changed), dtype=np.int8).astype(bool)
        hits += int(np.count_nonzero(np.abs(statistic(swaps)) >= abs(observed) - 1e-12))
    return (hits / total if exact else (hits + 1) / (total + 1)), exact


def _macro_data(pairs, classification_truth):
    if classification_truth is None and all("class_truth_index" in case for pair in pairs for case in pair):
        if any(old["class_truth_index"] != new["class_truth_index"] for old, new in pairs):
            raise ValueError("Paired classification truth differs between runs")
        classification_truth = {old["id"]: old["class_truth_index"] for old, _ in pairs}
    if classification_truth is not None:
        indices = {case["expected"]: classification_truth[case["id"]]
                   for pair in pairs for case in pair if "expected" in case}
        if all("class_prediction_index" in case or "prediction" in case for pair in pairs for case in pair):
            def predicted(case):
                if "class_prediction_index" in case:
                    return case["class_prediction_index"]
                return None if case["prediction"] is None else indices[case["prediction"]]

            return ([classification_truth[old["id"]] for old, _ in pairs],
                    [predicted(old) for old, _ in pairs], [predicted(new) for _, new in pairs])
    if all("expected" in old and "prediction" in old and "prediction" in new for old, new in pairs):
        return ([old["expected"] for old, _ in pairs], [old["prediction"] for old, _ in pairs],
                [new["prediction"] for _, new in pairs])
    return None


def _macro_inference(np, data, effect, options):
    truth, before, after = (np.asarray(values, dtype=object) for values in data)

    def bootstrap(indices):
        return np.asarray([classification_macro_f1(truth[ix], after[ix]) - classification_macro_f1(truth[ix], before[ix])
                           for ix in indices])

    changed = np.flatnonzero(before != after)

    def permutation(swaps):
        deltas = []
        for mask in swaps:
            selected = changed[mask]
            a, b = before.copy(), after.copy()
            a[selected], b[selected] = after[selected], before[selected]
            deltas.append(classification_macro_f1(truth, b) - classification_macro_f1(truth, a))
        return np.asarray(deltas)

    interval = _bootstrap(np, len(truth), bootstrap, options)
    p_value, exact = _permutation(np, len(changed), permutation, effect, options)
    return interval, p_value, "paired_permutation_exact" if exact else "paired_permutation_monte_carlo", len(changed)


def validate_case_mean(run: dict, name: str) -> bool:
    """Validate observed scores and report whether every case supplies the metric."""
    values = [case["metrics"].get(name) for case in run["cases"]]
    for value in values:
        if value is not None and (not nonnegative(value) or value > 1
                                  or (name in _BINARY_SCORES and value not in (0, 1))):
            raise ValueError(f"Invalid per-case score for {name}")
    if None in values:
        return False
    aggregate = run["metrics"][name]
    if not nonnegative(aggregate) or aggregate > 1 or not isclose(mean(values), aggregate, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Aggregate and per-case scores disagree for {name}")
    return True


def _case_inference(np, name, pairs, effect, options):
    differences = np.asarray([new["metrics"][name] - old["metrics"][name] for old, new in pairs])
    changed = differences[np.abs(differences) > 1e-12]
    interval = _bootstrap(np, len(pairs), lambda ix: differences[ix].mean(axis=1), options)
    if name in _BINARY_SCORES:
        from scipy.stats import binomtest

        wins = int(np.count_nonzero(changed > 0))
        p_value = float(binomtest(wins, len(changed), 0.5).pvalue) if len(changed) else 1.0
        return interval, p_value, "mcnemar_exact", len(changed)
    p_value, exact = _permutation(np, len(changed), lambda swaps: ((1 - 2 * swaps) * changed).sum(axis=1) / len(pairs),
                                  effect, options)
    return interval, p_value, "paired_permutation_exact" if exact else "paired_permutation_monte_carlo", len(changed)


def _adjust(results, correction):
    # Count the entire planned family, including metrics without sufficient evidence.
    family_size = len(results)
    available = sorted((item["p_value"], name) for name, item in results.items() if item["p_value"] is not None)
    previous = 0.0
    for rank, (p_value, name) in enumerate(available):
        if correction == "holm":
            previous = max(previous, min(1.0, (family_size - rank) * p_value))
            adjusted = previous
        elif correction == "bonferroni":
            adjusted = min(1.0, family_size * p_value)
        else:
            adjusted = p_value
        results[name]["adjusted_p_value"] = adjusted


def _require_statistics():
    try:
        import numpy as np
        import scipy.stats
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "scipy"}:
            raise
        raise ValueError("Paired statistics require pip install 'llmguard-ci[statistics]'") from None
    return np


def paired_statistics(baseline: dict, candidate: dict, options: StatisticsOptions | None = None,
                      *, classification_truth: dict | None = None) -> dict:
    """Compare case-aligned suite or SDK reports. Positive effects favor the candidate."""
    np = _require_statistics()
    options = options if options is not None else StatisticsOptions()
    pairs = _pairs(baseline, candidate)

    def supported(run):
        versions = run.get("metric_versions", {})
        return {name for name in run["metrics"].keys() - LOWER_IS_BETTER
                if not versions.get(name, "").startswith("custom-v1/")}

    metrics = supported(candidate)
    if metrics != supported(baseline):
        raise ValueError("Paired comparisons require matching quality metrics")
    if isinstance(options.min_effect, Mapping) and options.min_effect.keys() - metrics:
        raise ValueError("Statistics min_effect must refer to selected quality metrics")
    results = {}
    for name in sorted(metrics):
        has_case_scores = True
        if name != "macro_f1":
            baseline_complete = validate_case_mean(baseline, name)
            candidate_complete = validate_case_mean(candidate, name)
            has_case_scores = baseline_complete and candidate_complete
        effect = candidate["metrics"][name] - baseline["metrics"][name]
        threshold = options.min_effect.get(name, 0.01) if isinstance(options.min_effect, Mapping) else options.min_effect
        result = {"effect_size": effect, "min_effect": threshold, "pairs": len(pairs), "changed_pairs": None,
                  "confidence_interval": None, "p_value": None, "adjusted_p_value": None, "test": None,
                  "status": "ok", "practical_regression": effect < -1e-12 and -effect + 1e-12 >= threshold,
                  "statistically_significant": None, "regression": False}
        if len(pairs) < options.min_pairs:
            result["status"] = "insufficient_pairs"
        elif name == "macro_f1":
            data = _macro_data(pairs, classification_truth)
            if data is None:
                result["status"] = "missing_class_predictions"
            else:
                truth, before, after = data
                for run, predictions in ((baseline, before), (candidate, after)):
                    if not isclose(classification_macro_f1(truth, predictions), run["metrics"][name], rel_tol=0, abs_tol=1e-12):
                        raise ValueError("Aggregate and per-case scores disagree for macro_f1")
                result["confidence_interval"], result["p_value"], result["test"], result["changed_pairs"] = _macro_inference(
                    np, data, effect, options)
        elif not has_case_scores:
            result["status"] = "missing_case_scores"
        else:
            result["confidence_interval"], result["p_value"], result["test"], result["changed_pairs"] = _case_inference(
                np, name, pairs, effect, options)
        results[name] = result
    _adjust(results, options.correction)
    for result in results.values():
        if result["adjusted_p_value"] is not None:
            result["statistically_significant"] = result["adjusted_p_value"] <= options.alpha
            result["regression"] = result["statistically_significant"] and result["practical_regression"]
    return {"version": "paired-v1", "settings": asdict(options), "metrics": results,
            "regressions": [name for name, result in results.items() if result["regression"]]}
