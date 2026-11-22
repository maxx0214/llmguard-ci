import asyncio
import json
from dataclasses import dataclass
from time import sleep

import pytest

from llmguard import Evaluator, System


def test_rag_regression_and_saved_baseline(tmp_path):
    dataset = tmp_path / "rag.jsonl"
    rows = [{"id": "q1", "input": "private question", "expected": ["doc-a"]},
            {"id": "q2", "input": "other question", "expected": ["doc-b"]}]
    dataset.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    seen = []

    def baseline(question):
        seen.append(question)
        return ["doc-a" if question == "private question" else "doc-b"]

    def candidate(question):
        return [] if question == "private question" else ["doc-b"]

    evaluator = Evaluator(dataset, "retrieval", ["recall@5", "mrr@5", "ndcg@5"])
    path = tmp_path / "baseline.json"
    first = evaluator.evaluate(System(baseline, model_version="rag-v1", prompt_version="p1"))
    assert first.status == "PASS"
    first.report(path)
    result = evaluator.compare(baseline=path, candidate=System(candidate, model_version="rag-v2"))
    report = result.report(tmp_path / "candidate.json")
    assert seen == ["private question", "other question"]
    assert report["baseline"]["source"]["model_version"] == "rag-v1"
    assert report["baseline"]["source"]["prompt_version"] == "p1"
    assert report["source"]["model_version"] == "rag-v2"
    assert report["dataset_fingerprint"] == first.report()["dataset_fingerprint"]
    assert report["metric_versions"]["ndcg@5"].startswith("suite-v1")
    assert report["environment"]["python"]
    assert result.status == "FAIL"
    assert result.metrics == {"recall@5": .5, "mrr@5": .5, "ndcg@5": .5}
    assert report["comparison"]["recall@5"]["status"] == "FAIL"
    assert len(result.cases) == 2 and all(case["latency_ms"] >= 0 for case in result.cases)
    assert report["baseline"]["cases"][0]["metrics"]["recall@5"] == 1
    for artifact in (path, path.with_suffix(".html"), tmp_path / "candidate.json", tmp_path / "candidate.html"):
        assert "private question" not in artifact.read_text(encoding="utf-8")
    for protected in (path, dataset):
        with pytest.raises(ValueError):
            result.report(protected)


def test_comparing_sync_and_async_systems_in_an_event_loop():
    dataset = [{"id": "q", "input": "question", "expected": "answer"}]
    evaluator = Evaluator(dataset, "qa")

    async def answer(text):
        await asyncio.sleep(0)
        return "ANSWER"

    def regular_returns_awaitable(text):
        return answer(text)

    async def run():
        with pytest.raises(RuntimeError, match="aevaluate"):
            evaluator.evaluate(answer)
        single = await evaluator.aevaluate(answer)
        result = await evaluator.acompare(single, regular_returns_awaitable)
        assert result.status == "PASS"
        assert result.metrics == {"exact_match": 1, "token_f1": 1}
        assert result.report()["operational"]["cost"] is None
        assert result.report()["usage"] is None

    asyncio.run(run())


def test_generate_method_and_custom_adapter():
    @dataclass
    class Generated:
        tokens: list[int]

    class FineTunedModel:
        def generate(self, value):
            assert value == {"prompt": "decode"}
            return Generated([42, 7])

    async def decode(result):
        assert result.tokens == [42, 7]
        return {"answer": "decoded", "usage": {"generated_tokens": len(result.tokens)}, "cost": 0}

    dataset = [{"id": "q", "input": {"prompt": "decode"}, "expected": "decoded"}]
    evaluator = Evaluator(dataset, "qa", include_payloads=True)
    result = evaluator.evaluate(System(FineTunedModel().generate, adapter=decode, model_version="checkpoint-42"))
    assert result.status == "PASS"
    assert result.report()["usage"] == {"generated_tokens": 2}
    assert result.report()["operational"]["cost"] == 0
    assert result.cases[0]["output"]["answer"] == "decoded"


def test_langchain_runnable_with_document_attributes():
    docs = pytest.importorskip("langchain_core.documents")
    runnables = pytest.importorskip("langchain_core.runnables")
    chain = runnables.RunnableLambda(lambda query: [docs.Document(page_content="context", metadata={"doc_id": "d"})])
    evaluator = Evaluator([{"id": "q", "input": "question", "expected": ["d"]}], "retrieval",
                          ["recall@1"], output_mapping={"retrieved_ids": "*.metadata.doc_id",
                                                       "contexts": "*.page_content"}, include_payloads=True)
    result = evaluator.compare(chain.invoke, chain.ainvoke)
    assert result.status == "PASS" and result.metrics["recall@1"] == 1
    assert result.cases[0]["output"]["contexts"] == ["context"]


def test_missing_field_and_safe_errors_mapped_only_if_required():
    dataset = [{"id": "q", "input": "question", "expected": "class-a"}]
    evaluator = Evaluator(dataset, "classification", output_mapping={"label": "result.0.label"})
    result = evaluator.evaluate(lambda _: {"label": "class-a", "usage": {"tokens": 8}, "cost": 0.5})
    assert result.status == "FAIL"
    assert "Missing required output field 'label'" in result.cases[0]["error"]["message"]
    assert result.cases[0]["cost"] == .5 and result.report()["usage"] == {"tokens": 8}
    assert evaluator.evaluate(lambda _: {"result": [{"label": "class-a"}]}).metrics["accuracy"] == 1

    def crash(_):
        raise RuntimeError("secret header value")

    failed = evaluator.evaluate(crash).report()
    assert "secret header value" not in json.dumps(failed)
    assert failed["cases"][0]["error"]["code"] == "call_failed"
    operations = Evaluator(dataset, "classification", ["latency_ms"]).evaluate(lambda _: None)
    assert operations.status == "PASS" and operations.cases[0]["error"] is None


def test_absolute_quality_operational_and_directional_gates():
    rows = [{"id": "q1", "input": "one", "expected": "right"},
            {"id": "q2", "input": "two", "expected": "right"}]
    evaluator = Evaluator(rows, "qa", ["exact_match", "cost"], quality_gate={
        "min_scores": {"exact_match": {"warn": .8, "fail": .4}},
        "max_drop": {"exact_match": {"warn": .1, "fail": .5}, "cost": .1},
        "max_cost": 1.5,
    })
    baseline = evaluator.evaluate(lambda _: {"answer": "right", "cost": .6})
    cheap = evaluator.compare(baseline, lambda x: {"answer": "right" if x == "one" else "wrong", "cost": .2})
    assert cheap.status == "WARN"  # A drop equal to the failure bound is allowed.
    assert cheap.report()["gates"]["exact_match"]["status"] == "WARN"
    assert cheap.report()["comparison"]["cost"]["status"] == "PASS"
    expensive = evaluator.compare(baseline, lambda _: {"answer": "right", "cost": .7})
    assert expensive.status == "FAIL"
    assert expensive.report()["gates"]["cost"]["status"] == "PASS"
    assert expensive.report()["comparison"]["cost"]["direction"] == "lower"
    assert expensive.report()["comparison"]["cost"]["status"] == "FAIL"
    assert evaluator.evaluate(lambda _: {"answer": "right", "cost": .9}).report()["gates"]["cost"]["status"] == "FAIL"


def test_latency_budget_uses_elapsed_time_and_missing_cost_is_unknown():
    rows = [{"id": "q", "input": "q", "expected": "yes"}]

    def slow(_):
        sleep(.02)
        return {"answer": "yes", "latency_ms": 0, "cost": 0}

    timed = Evaluator(rows, "qa", quality_gate={"max_latency_ms": 5}).evaluate(slow)
    assert timed.report()["gates"]["latency_ms"]["status"] == "FAIL"
    assert timed.cases[0]["latency_ms"] > 5 and timed.cases[0]["reported_latency_ms"] == 0
    for on_unknown, status in (("warn", "WARN"), ("fail", "FAIL")):
        evaluator = Evaluator(rows, "qa", quality_gate={"max_cost": .05, "on_unknown": on_unknown})
        result = evaluator.evaluate(lambda _: "yes")
        assert result.status == status
        assert result.report()["operational"]["cost"] is None


def test_error_rate_limit_and_partial_telemetry():
    rows = [{"id": "a", "input": "one", "expected": "yes"}, {"id": "b", "input": "two", "expected": "yes"}]

    def system(text):
        if text == "one":
            return {"answer": "yes", "cost": 0, "usage": {"tokens": 1}}
        raise ValueError("private input")

    evaluator = Evaluator(rows, "qa", quality_gate={"max_error_rate": .5})
    report = evaluator.evaluate(system).report()
    assert report["status"] == "PASS" and report["operational"]["error_rate"] == .5
    assert report["operational"]["cost"] is None and report["usage"] is None
    assert report["cases"][1]["metrics"]["exact_match"] == 0
    assert Evaluator(rows, "qa").evaluate(system).status == "FAIL"


def test_saved_baseline_mismatch_is_rejected_before_candidate_call(tmp_path):
    baseline = Evaluator([{"id": "q", "input": "question", "expected": "answer"}], "qa")
    path = tmp_path / "baseline.json"
    baseline.evaluate(lambda _: "answer").report(path)
    calls = []
    changed = Evaluator([{"id": "q", "input": "changed", "expected": "answer"}], "qa")
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        changed.compare(path, lambda text: calls.append(text))
    assert not calls


def test_saved_baseline_payloads_remain_private_in_default_comparison(tmp_path):
    rows = [{"id": "q", "input": "private question", "expected": "answer"}]
    path = tmp_path / "private.json"
    Evaluator(rows, "qa", include_payloads=True).evaluate(lambda _: "answer").report(path)
    assert "private question" in path.read_text(encoding="utf-8")
    result = Evaluator(rows, "qa").compare(path, lambda _: "answer")
    report = result.report(tmp_path / "public.json")
    assert "input" not in report["baseline"]["cases"][0]
    assert "private question" not in (tmp_path / "public.html").read_text(encoding="utf-8")


def test_comparison_html_displays_both_models_in_each_table(tmp_path):
    rows = [{"id": "case-1", "input": "question", "expected": "yes"}]
    evaluator = Evaluator(rows, "qa", ["exact_match"], include_payloads=True)
    baseline = System(lambda _: {"answer": "yes", "cost": .02, "usage": {"tokens": 7}}, model_version="original")
    candidate = System(lambda _: {"answer": "no", "usage": {"tokens": 11}}, model_version="updated")
    evaluator.compare(baseline, candidate).report(tmp_path / "comparison.json")
    page = (tmp_path / "comparison.html").read_text(encoding="utf-8")

    def section(title):
        return page.split(f"<h2>{title}</h2>", 1)[1].split("</section>", 1)[0]

    both = "<th scope='col'>Baseline</th><th scope='col'>Candidate</th>"
    assert "<td>original</td>" in page and "<td>updated</td>" in page
    assert both in section("Metrics") and "<td>1.0000</td><td>0.0000</td><td>-1.0000</td>" in section("Metrics")
    assert both in section("Operations") and "Cost (USD)</th><td>0.0200</td><td>unknown</td>" in section("Operations")
    assert "Usage</th><td>{&quot;tokens&quot;: 7}</td><td>{&quot;tokens&quot;: 11}</td>" in section("Operations")
    cases = section("Cases")
    assert both in cases and cases.count("case-1</th>") == 1
    assert "<dt>Prediction</dt><dd>yes</dd>" in cases and "<dt>Prediction</dt><dd>no</dd>" in cases
