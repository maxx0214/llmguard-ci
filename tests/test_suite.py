from pathlib import Path
import json
from types import SimpleNamespace

import pytest
from llmguard.suite import (CompatibleProvider, compare_suite, evaluate_suite, load_rows,
                            main, render_html, retrieval_scores)


def test_retrieval_metrics():
    result = retrieval_scores(["a", "b"], ["x", "a", "b"], 2)
    assert result["hit@2"] == 1
    assert result["recall@2"] == .5
    assert result["mrr@2"] == .5
    assert 0 < result["ndcg@2"] < 1


def test_classification_missing_and_regression():
    rows = [{"id": "1", "input": "one", "expected": "a"},
            {"id": "2", "input": "two", "expected": "b"}]
    baseline = evaluate_suite("classification", rows, {"1": "a", "2": "b"})
    candidate = evaluate_suite("classification", rows, {"1": "a"})
    assert candidate["metrics"] == {"accuracy": .5, "macro_f1": .5}
    assert candidate["error_count"] == 1
    assert compare_suite(baseline, candidate, {})["status"] == "FAIL"
    with pytest.raises(ValueError):
        compare_suite(baseline, evaluate_suite("classification", rows[:1], {"1":"a"}), {})


def test_retrieval_duplicate_ids_are_errors():
    rows = [{"id": "1", "input": "one", "expected": ["a", "b"]}]
    assert evaluate_suite("retrieval", rows, {"1": ["a", "a"]})["status"] == "FAIL"


def test_qa_and_report_escaping():
    rows = [{"id": "<script>", "input": "q", "expected": "hello world"}]
    run = evaluate_suite("qa", rows, {"<script>": "HELLO"})
    assert run["metrics"]["token_f1"] == pytest.approx(2/3)
    assert "<script>" not in render_html(run)


def test_custom_csv_jsonl(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,input,expected\n1,question,answer\n", encoding="utf-8")
    assert load_rows(csv_path)[0]["expected"] == "answer"
    p = tmp_path / "predictions.jsonl"
    p.write_text('{"id":"1","prediction":"answer"}\n', encoding="utf-8")
    cfg = tmp_path / "suite.yaml"
    cfg.write_text("task: qa\ndataset: data.csv\npredictions: predictions.jsonl\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    assert main(["--config", str(cfg), "--output", str(baseline)]) == 0
    assert baseline.with_suffix(".html").exists()
    assert json.loads(baseline.read_text())["metrics"]["exact_match"] == 1
    p.write_text('{"id":"1","prediction":"wrong"}\n', encoding="utf-8")
    assert main(["--config", str(cfg), "--output", str(tmp_path / "candidate.json"),
                 "--baseline", str(baseline)]) == 1


def test_provider_mock():
    class Chat:
        def create(self, **kwargs):
            assert kwargs["model"] == "mock"
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                  message=SimpleNamespace(content="answer"))])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=Chat()), close=lambda: None)
    provider = CompatibleProvider({"name": "compatible", "model": "mock",
                                   "base_url": "https://example.invalid/v1", "api_key_env": "MOCK_KEY"}, sdk)
    assert provider.predict("question") == "answer"
    provider.close()
