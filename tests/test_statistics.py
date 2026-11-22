import numpy as np
import pytest
from sklearn.metrics import f1_score

from llmguard import Evaluator, StatisticsOptions, paired_statistics
from llmguard.suite import evaluate_suite


def test_exact_mcnemar_holm_and_unchanged_pairs():
    rows = [{"id": str(i), "input": str(i), "expected": "yes"} for i in range(20)]
    baseline = evaluate_suite("qa", rows, {row["id"]: "yes" for row in rows})
    candidate = evaluate_suite("qa", rows, {row["id"]: "no" if int(row["id"]) < 8 else "yes" for row in rows})
    options = StatisticsOptions(seed=9, bootstrap_samples=300, permutation_samples=256)
    same = paired_statistics(baseline, baseline, options)
    assert same["regressions"] == []
    assert all(item["p_value"] == 1 and item["confidence_interval"] == [0, 0] for item in same["metrics"].values())
    changed = paired_statistics(baseline, candidate, options)
    assert changed == paired_statistics(baseline, candidate, options)
    assert changed["regressions"] == ["exact_match", "token_f1"]
    em = changed["metrics"]["exact_match"]
    assert em["test"] == "mcnemar_exact" and em["changed_pairs"] == 8
    assert em["p_value"] == 2 / 256 and em["adjusted_p_value"] == 4 / 256
    assert em["effect_size"] == pytest.approx(-.4) and em["confidence_interval"][1] < 0
    assert changed["metrics"]["token_f1"]["test"] == "paired_permutation_exact"


def test_statistical_significance_practical_effect_and_insufficient_pairs():
    def runs(count):
        rows = [{"id": str(i), "input": str(i), "expected": ["a", "b"]} for i in range(count)]
        baseline = evaluate_suite("retrieval", rows, {row["id"]: ["a", "b"] for row in rows}, [3])
        candidate = evaluate_suite("retrieval", rows, {row["id"]: ["a", "z", "b"] for row in rows}, [3])
        return baseline, candidate

    options = StatisticsOptions(seed=7, min_effect=.10, bootstrap_samples=100, permutation_samples=300,
                                correction="bonferroni")
    result = paired_statistics(*runs(40), options)
    ndcg = result["metrics"]["ndcg@3"]
    assert ndcg["statistically_significant"] and not ndcg["practical_regression"] and not ndcg["regression"]
    assert ndcg["adjusted_p_value"] == 4 * ndcg["p_value"]
    assert ndcg["test"] == "paired_permutation_monte_carlo"
    small = paired_statistics(*runs(4), options)["metrics"]["ndcg@3"]
    assert small["status"] == "insufficient_pairs" and small["p_value"] is None and small["confidence_interval"] is None


def test_sdk_macro_f1_bootstrap_recalculates_full_samples(tmp_path):
    truth = np.array(["a", "a", "b", "b", "c"])
    predicted = np.array(["a", "b", "b", "b", "b"])
    rows = [{"id": str(i), "input": str(i), "expected": label} for i, label in enumerate(truth.tolist())]
    options = StatisticsOptions(seed=12, min_pairs=2, bootstrap_samples=80, permutation_samples=64)
    evaluator = Evaluator(rows, "classification", ["macro_f1"], statistics=options)
    baseline = evaluator.evaluate(lambda x: str(truth[int(x)]))
    saved = tmp_path / "baseline.json"
    baseline.report(saved)
    result = evaluator.compare(saved, lambda x: str(predicted[int(x)]))
    report = result.report(tmp_path / "candidate.json")
    assert "prediction" not in baseline.cases[0] and "class_prediction_index" in baseline.cases[0]
    indices = np.random.default_rng(options.seed).integers(len(truth), size=(80, len(truth)))
    differences = [f1_score(truth[ix], predicted[ix], average="macro", labels=np.unique(truth[ix]), zero_division=0) - 1
                   for ix in indices]
    item = report["statistics"]["metrics"]["macro_f1"]
    assert item["confidence_interval"] == pytest.approx(np.quantile(differences, [.025, .975]))
    assert item["test"] == "paired_permutation_exact"
    assert paired_statistics(baseline.report(), report, options) == report["statistics"]
    assert [case["id"] for case in report["case_changes"]["accuracy"]["worsened"]] == ["1", "4"]
    html = (tmp_path / "candidate.html").read_text(encoding="utf-8")
    assert "Confidence interval" in html and "Case changes" in html and "Actual usage" in html


def test_optional_enforcement_preserves_threshold_gate():
    rows = [{"id": str(i), "input": str(i), "expected": "yes"} for i in range(24)]
    before, after = (lambda _: "yes"), (lambda _: "no")
    for enforce, expected in ((False, "PASS"), (True, "FAIL")):
        evaluator = Evaluator(rows, "qa", ["exact_match"], quality_gate={"max_drop": {"exact_match": 1}},
                              statistics=StatisticsOptions(enforce=enforce, bootstrap_samples=20))
        report = evaluator.compare(before, after).report()
        assert report["comparison"]["exact_match"]["status"] == "PASS"
        assert report["statistics"]["regressions"] == ["exact_match"] and report["status"] == expected
    threshold_only = Evaluator(rows, "qa", ["exact_match"]).compare(before, after).report()
    assert threshold_only["status"] == "FAIL" and "statistics" not in threshold_only


def test_legacy_redacted_macro_baseline_is_reported_unavailable():
    rows = [{"id": str(i), "input": str(i), "expected": "yes"} for i in range(4)]
    evaluator = Evaluator(rows, "classification", statistics={"min_pairs": 2})
    baseline = evaluator.evaluate(lambda _: "yes").report()
    for case in baseline["cases"]:
        case.pop("class_prediction_index")
    result = paired_statistics(baseline, evaluator.evaluate(lambda _: "yes").report(), StatisticsOptions(min_pairs=2))
    assert result["metrics"]["macro_f1"]["status"] == "missing_class_predictions"
    assert result["metrics"]["accuracy"]["p_value"] == 1


@pytest.mark.parametrize(("changed", "message"), [
    ("aggregate", "Aggregate and per-case scores disagree"),
    ("binary", "Invalid per-case score"),
])
def test_inconsistent_scores_are_rejected_before_candidate_requests(tmp_path, changed, message):
    import json

    rows = [{"id": "1", "input": "q", "expected": "yes"}]
    evaluator = Evaluator(rows, "qa", ["exact_match"])
    report = evaluator.evaluate(lambda _: "yes").report()
    clean = evaluator.evaluate(lambda _: "yes").report()
    if changed == "aggregate":
        report["metrics"]["exact_match"] = 0
    else:
        report["cases"][0]["metrics"]["exact_match"] = 0.5
    baseline = tmp_path / "bad-baseline.json"
    baseline.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        evaluator.compare(baseline, lambda _: pytest.fail("Candidate must not be called"))
    with pytest.raises(ValueError, match=message):
        paired_statistics(report, clean)
