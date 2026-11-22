import asyncio
import csv
import json

import httpx
import pytest

from llmguard import Evaluator, Metric, PredictionFile, RESTAdapter, StatisticsOptions


CASES = [{"id": "a", "input": "same question", "expected": ["doc-a"]},
         {"id": "b", "input": "same question", "expected": ["doc-b"]}]


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".csv"])
def test_replays_files_by_id_in_dataset_order(tmp_path, suffix):
    rows = [{"case": "b", "rank": ["doc-b"]}, {"case": "a", "rank": ["doc-a"]}]
    path = tmp_path / f"predictions{suffix}"
    if suffix == ".csv":
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case", "rank"])
            writer.writeheader()
            writer.writerows({**row, "rank": json.dumps(row["rank"])} for row in rows)
    else:
        serialized = json.dumps({"results": {"rows": rows}}) if suffix == ".json" else "\n".join(map(json.dumps, rows))
        path.write_text(serialized, encoding="utf-8-sig")
    file = PredictionFile(path, {"prediction": "rank"}, id_path="case",
                          rows_path="results.rows" if suffix == ".json" else None)
    evaluator = Evaluator(CASES, "retrieval", ["hit@1", "latency_ms", "cost"], cache=tmp_path / "cache.sqlite")
    report = evaluator.compare(file, file).report(tmp_path / "comparison.json")
    assert report["metrics"] == {"hit@1": 1, "latency_ms": None, "cost": None}
    assert [case["id"] for case in report["cases"]] == ["a", "b"]
    assert all(case["execution_origin"] == "prediction_file" for case in report["cases"])
    assert report["execution"]["request_attempts"] == report["execution"]["api_calls"] == 0
    assert report["usage"] is None
    assert not (tmp_path / "cache.sqlite").exists()
    assert report["baseline"]["source"]["sha256"] == report["source"]["sha256"]


@pytest.mark.parametrize(("ids", "message"), [
    (["a", "a"], "Duplicate prediction case ID"),
    (["a"], r"missing \(1\) \['b'\]"),
    (["a", "c"], r"missing \(1\).*unexpected \(1\) \['c'\]"),
    (["a", None], "requires a nonempty string ID"),
])
def test_invalid_prediction_id_does_not_run_live_baseline(tmp_path, ids, message):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps([{"id": case_id, "prediction": []} for case_id in ids]), encoding="utf-8")

    def forbidden(_):
        pytest.fail("Preflight should reject mismatched IDs before calling the baseline")

    with pytest.raises(ValueError, match=message):
        Evaluator(CASES, "retrieval").compare(forbidden, PredictionFile(path))


def test_nested_dataset_matches_canonical_fingerprint_and_protects_sources(tmp_path):
    original = tmp_path / "dataset.json"
    payload = {"cases": [
        {"item": {"case_id": case["id"], "ask": case["input"]},
         "gold_chunks": [{"chunk_id": chunk} for chunk in case["expected"]]}
        for case in CASES
    ]}
    original.write_text(json.dumps(payload), encoding="utf-8")
    snapshot = original.read_bytes()
    evaluator = Evaluator(original, "retrieval", dataset_rows_path="cases",
                          dataset_mapping={"id": "item.case_id", "input": "item.ask",
                                           "expected": "gold_chunks.*.chunk_id"})
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps([{"id": "b", "prediction": ["doc-b"]},
                                      {"id": "a", "prediction": {"retrieved_ids": ["doc-a"]}}]), encoding="utf-8")
    evaluated = evaluator.evaluate(PredictionFile(predictions))
    expected = Evaluator(CASES, "retrieval").evaluate(PredictionFile(predictions))
    mapped_report = evaluated.report()
    assert mapped_report["dataset_fingerprint"] == expected.report()["dataset_fingerprint"]
    assert mapped_report["run_fingerprint"] == expected.report()["run_fingerprint"]
    assert original.read_bytes() == snapshot
    comparison = evaluator.compare(evaluated, PredictionFile(predictions))
    for source in (original, predictions):
        with pytest.raises(ValueError, match="cannot overwrite"):
            comparison.report(source)
    with pytest.raises(ValueError, match="missing mapped fields: expected"):
        Evaluator(original, "retrieval", dataset_rows_path="cases",
                  dataset_mapping={"id": "item.case_id", "input": "item.ask", "expected": "gold_chunks.*.doc_id"})


def test_mixed_archive_async_and_rest(tmp_path):
    data = [{"id": "x", "input": "hello", "expected": "hello"}]
    saved = tmp_path / "qa.jsonl"
    saved.write_text('{"id": "x", "prediction": "hello", "latency_ms": 2.5}', encoding="utf-8")
    evaluator = Evaluator(data, "qa", ["exact_match"], metadata_fields=["rerank_failed"])

    async def local(question):
        return question

    archive = PredictionFile(saved)
    assert asyncio.run(evaluator.acompare(local, archive)).status == "PASS"
    with RESTAdapter("https://example.test/predict", response_mapping={
            "answer": "result.answer", "metadata.rerank_failed": "result.failed"},
            transport=httpx.MockTransport(lambda _: httpx.Response(
                200, json={"result": {"answer": "hello", "failed": False}}))) as rest:
        report = evaluator.compare(archive, rest).report()
    assert report["baseline"]["operational"]["latency_ms"] == 2.5
    assert report["execution"]["api_calls"] == 1
    assert report["metadata_summary"]["rerank_failed"] == {
        "known": 1, "unknown": 0, "values": [{"value": False, "count": 1}]}


def test_metadata_custom_gate_and_unknown_policy(tmp_path):
    def make_archive(name, values):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps([{"id": case["id"], "rank": case["expected"], "fallback": flag}
                                    for case, flag in zip(CASES, values, strict=True)]), encoding="utf-8")
        return PredictionFile(path, {"retrieved_ids": "rank", "metadata.rerank_failed": "fallback"})

    before = make_archive("before", [False, False])
    worse = make_archive("worse", [True, False])
    unknown = make_archive("unknown", [None, False])
    evaluator = Evaluator(
        CASES, "retrieval", ["hit@1", "fallback_rate"],
        custom_metrics={"fallback_rate": Metric(lambda case: case["metadata"]["rerank_failed"],
                                                  direction="lower_is_better", version="release-test")},
        metadata_fields=["rerank_failed", "unavailable"],
        quality_gate={"max_scores": {"fallback_rate": .3}, "max_drop": {"fallback_rate": .1}, "on_unknown": "warn"},
        statistics=StatisticsOptions(bootstrap_samples=10, permutation_samples=10),
    )
    report = evaluator.compare(before, worse).report(tmp_path / "compare.json")
    assert report["status"] == "FAIL"
    assert report["metrics"]["fallback_rate"] == .5
    assert report["metric_versions"]["fallback_rate"] == "custom-v1/lower_is_better/release-test"
    assert report["comparison"]["fallback_rate"]["status"] == report["gates"]["fallback_rate"]["status"] == "FAIL"
    assert report["case_changes"]["fallback_rate"]["worsened"] == [
        {"id": "a", "baseline": 0., "candidate": 1., "delta": 1.}]
    assert set(report["statistics"]["metrics"]) == {"hit@1"}
    assert report["metadata_summary"]["unavailable"] == {"known": 0, "unknown": 2, "values": []}
    html = (tmp_path / "compare.html").read_text(encoding="utf-8")
    assert "<h2>Metadata</h2>" in html and "rerank_failed" in html
    warned = evaluator.compare(before, unknown).report()
    assert warned["status"] == "WARN"
    assert warned["metrics"]["fallback_rate"] is None
    assert warned["case_changes"]["fallback_rate"]["unknown"] == 1
    assert warned["metadata_summary"]["rerank_failed"]["unknown"] == 1


def test_signed_higher_is_better_custom_metric_and_saved_baseline(tmp_path):
    evaluator = Evaluator(CASES, "retrieval", ["signed"],
                          custom_metrics={"signed": lambda case: case["metadata"].get("score")},
                          quality_gate={"min_scores": {"signed": -1.5}, "max_drop": {"signed": .5}})
    baseline = tmp_path / "baseline.json"
    evaluator.evaluate(lambda _: {"metadata": {"score": -1}}).report(baseline)
    report = evaluator.compare(baseline, lambda _: {"metadata": {"score": -2}}).report()
    assert report["metrics"] == {"signed": -2}
    assert report["status"] == "FAIL"
    assert len(report["case_changes"]["signed"]["worsened"]) == 2
    with pytest.raises(ValueError, match="must return a finite number"):
        evaluator.evaluate(lambda _: {"metadata": {"score": "unavailable"}})
