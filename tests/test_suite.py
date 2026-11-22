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


def test_squad_official_normalization_and_multiple_answers():
    rows = [{"id": "one", "input": "q", "expected": ["another answer", "The New York! City"]},
            {"id": "two", "input": "q", "expected": ["a a b", "a b b c"]},
            {"id": "three", "input": "q", "expected": ["The"]},
            {"id": "four", "input": "q", "expected": ["Paris"]}]
    run = evaluate_suite("qa", rows, {"one": "new york city", "two": "b b", "three": ""}, profile="squad_v1")
    cases = {row["id"]: row for row in run["cases"]}
    assert cases["one"]["metrics"] == {"exact_match": 1, "token_f1": 1}
    assert cases["two"]["metrics"] == {"exact_match": 0, "token_f1": pytest.approx(.8)}
    assert cases["three"]["metrics"] == {"exact_match": 1, "token_f1": 0}
    assert cases["four"]["metrics"] == {"exact_match": 0, "token_f1": 0}
    assert run["error_count"] == 1


def test_graded_retrieval_depths_and_regression_gate():
    rows = [{"id": "q", "input": "q", "expected": {"high": 2, "low": 1}}]
    baseline = evaluate_suite("retrieval", rows, {"q": ["low", "high"]}, [3, 1], profile="beir")
    candidate = evaluate_suite("retrieval", rows, {"q": ["low"]}, [1, 3], profile="beir")
    assert baseline["k"] == [1, 3]
    assert baseline["metrics"]["ndcg@1"] == .5
    assert baseline["metrics"]["ndcg@3"] == pytest.approx(.85971869985)
    assert baseline["metrics"]["hit@3"] == baseline["metrics"]["mrr@3"] == 1
    assert compare_suite(baseline, candidate, {})["status"] == "FAIL"
    assert {key for key, delta in candidate["comparison"].items() if delta["status"] == "FAIL"} == {
        "recall@3", "ndcg@3",
    }


def test_suite_cli_squad_profile_and_baseline_profile_check(tmp_path):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{"id": "q", "input": "q", "expected": ["Paris", "The City of Light"]}]))
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps([{"id": "q", "prediction": "city of light!"}]))
    cfg = tmp_path / "suite.yaml"
    cfg.write_text("task: qa\nprofile: squad_v1\ndataset: data.json\npredictions: predictions.json\n")
    output = tmp_path / "baseline.json"
    assert main(["--config", str(cfg), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["profile"] == "squad_v1"
    assert report["metrics"] == {"exact_match": 1, "token_f1": 1}
    with pytest.raises(ValueError, match="profile"):
        compare_suite(report, {**report, "profile": None}, {})


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
    usage = SimpleNamespace(prompt_tokens=8, completion_tokens=2)

    class Chat:
        def create(self, **kwargs):
            assert kwargs["model"] == "mock"
            assert kwargs["max_completion_tokens"] == 10
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                  message=SimpleNamespace(content="answer"))], usage=usage)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=Chat()), close=lambda: None)
    provider = CompatibleProvider({"name": "compatible", "model": "mock",
                                   "base_url": "https://example.invalid/v1", "api_key_env": "MOCK_KEY",
                                   "max_completion_tokens": 10}, sdk)
    assert provider.predict("question") == "answer"
    assert provider.last_usage is usage
    provider.close()

