import json
import logging

import httpx
import pytest

from llmguard import Evaluator, RESTAdapter, System
from llmguard.cli import main


def test_post_nested_json_and_environment_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMGUARD_TEST_AUTH", "Bearer dont-persist-this")

    def send(request):
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer dont-persist-this"
        assert request.headers["accept"] == "application/json"
        assert json.loads(request.content) == {"request": {"query": "q", "limit": 2}}
        assert request.extensions["timeout"]["read"] == 4
        return httpx.Response(200, json={"data": {"hits": [{"key": "doc"}], "body": "private response"},
                                         "billing": {"usd": 0.01, "units": {"input_tokens": 5}}})

    with RESTAdapter("https://example.invalid/query?secret=endpoint-secret", timeout=4,
                     request_mapping={"request.query": "input.query", "request.limit": "input.limit"},
                     response_mapping={"retrieved_ids": "data.hits.*.key", "cost": "billing.usd", "usage": "billing.units"},
                     auth_headers={"Authorization": "LLMGUARD_TEST_AUTH"}, transport=httpx.MockTransport(send)) as adapter:
        dataset = [{"id": "q", "input": {"query": "q", "limit": 2}, "expected": ["doc"]}]
        result = Evaluator(dataset, "retrieval", ["recall@1"]).evaluate(adapter)
    report = result.report(tmp_path / "report.json")
    assert result.status == "PASS" and result.metrics == {"recall@1": 1}
    assert report["operational"]["cost"] == .01 and report["usage"] == {"input_tokens": 5}
    assert report["source"]["method"] == "POST" and len(report["source"]["endpoint_sha256"]) == 64
    for artifact in ("report.json", "report.html"):
        contents = (tmp_path / artifact).read_text(encoding="utf-8")
        assert not any(secret in contents for secret in ("dont-persist-this", "private response", "endpoint-secret"))


def test_yaml_rest_get_regression_uses_saved_baseline(monkeypatch, tmp_path):
    client = httpx.Client
    requests = []

    def send(request):
        requests.append(request)
        assert request.method == "GET" and request.url.params == httpx.QueryParams({"q": "q with spaces"})
        return httpx.Response(200, json={"results": {"answer": "correct" if request.url.path == "/baseline" else "wrong"}})

    monkeypatch.setattr("llmguard.rest.httpx.Client", lambda **kw: client(**{**kw, "transport": httpx.MockTransport(send)}))
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"id":"q","input":"q with spaces","expected":"correct"}\n', encoding="utf-8")
    for label, exit_code in (("baseline", 0), ("candidate", 1)):
        config = tmp_path / f"{label}.yaml"
        config.write_text(f"""task: qa
dataset: data.jsonl
metrics: [exact_match]
model_version: {label}
rest:
  url: https://example.invalid/{label}
  method: GET
  request_mapping:
    q: input
  response_mapping:
    answer: results.answer
""", encoding="utf-8")
        output = tmp_path / f"{label}.json"
        argv = ["suite", "--config", str(config), "--output", str(output)]
        if label == "candidate":
            argv += ["--baseline", str(tmp_path / "baseline.json")]
        assert main(argv) == exit_code
        assert output.with_suffix(".html").is_file()
    assert len(requests) == 2
    report = json.loads((tmp_path / "candidate.json").read_text(encoding="utf-8"))
    assert report["baseline"]["source"]["model_version"] == "baseline"
    assert report["comparison"]["exact_match"]["status"] == "FAIL"


def test_yaml_execution_cache_checkpoint_and_statistics(monkeypatch, tmp_path):
    client = httpx.Client
    requests = []

    def send(request):
        requests.append(request)
        return httpx.Response(200, json="yes")

    monkeypatch.setattr("llmguard.rest.httpx.Client", lambda **kw: client(**{**kw, "transport": httpx.MockTransport(send)}))
    (tmp_path / "data.jsonl").write_text(
        '\n'.join(json.dumps({"id": str(i), "input": str(i), "expected": "yes"}) for i in range(3)), encoding="utf-8")
    config = tmp_path / "run.yaml"
    config.write_text("""task: qa
dataset: data.jsonl
model_version: test
prompt_version: none
system_settings: {temperature: 0}
cache: responses.sqlite
checkpoint: run.sqlite
execution: {max_concurrency: 2}
statistics: {min_pairs: 2, bootstrap_samples: 20}
rest: {url: 'https://example.invalid/'}
""", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    argv = ["suite", "--config", str(config)]
    assert main([*argv, "--output", str(baseline)]) == 0
    candidate = tmp_path / "candidate.json"
    assert main([*argv, "--output", str(candidate), "--baseline", str(baseline)]) == 0
    data = json.loads(candidate.read_text(encoding="utf-8"))
    assert len(requests) == 3 and data["execution"]["checkpoint_reused"] == 3
    assert data["statistics"]["regressions"] == []


@pytest.mark.parametrize(("failures", "retries", "error", "attempts"), [
    ([503, 200], 1, None, 2), ([401, 200], 3, "http_error", 1),
    (["timeout", "timeout", 200], 1, "timeout", 2), (["invalid_json"], 3, "invalid_json", 1),
])
def test_retry_budget_and_sanitized_rest_errors(monkeypatch, failures, retries, error, attempts):
    monkeypatch.setattr("llmguard.rest.sleep", lambda _: None)
    sent = []

    def send(request):
        state = failures[len(sent)]
        sent.append(state)
        if state == "timeout":
            raise httpx.ReadTimeout("private transport error", request=request)
        if state == "invalid_json":
            return httpx.Response(200, content=b"private invalid content")
        return httpx.Response(state, json="yes" if state == 200 else {"error": "private response"})

    with RESTAdapter("https://example.invalid", retries=retries, transport=httpx.MockTransport(send)) as adapter:
        result = Evaluator([{"id": "q", "input": "q", "expected": "yes"}], "qa").evaluate(adapter)
    assert len(sent) == attempts
    report = result.report()
    assert "private" not in json.dumps(report)
    assert result.status == ("PASS" if error is None else "FAIL")
    if error:
        case_error = result.cases[0]["error"]
        assert case_error["code"] == error and case_error["attempts"] == attempts


def test_missing_auth_and_request_field_are_checked_early(monkeypatch):
    monkeypatch.delenv("LLMGUARD_MISSING_KEY", raising=False)
    with pytest.raises(ValueError, match="LLMGUARD_MISSING_KEY"):
        RESTAdapter("https://example.invalid", auth_headers={"X-Key": "LLMGUARD_MISSING_KEY"})
    with pytest.raises(ValueError, match="Overlapping"):
        RESTAdapter("https://example.invalid", request_mapping={"query": "input", "query.text": "input.text"})
    calls = []
    with RESTAdapter("https://example.invalid", request_mapping={"q": "input.missing"},
                     transport=httpx.MockTransport(lambda req: calls.append(req))) as adapter:
        result = Evaluator([{"id": "q", "input": {"query": "x"}, "expected": "x"}], "qa").evaluate(adapter)
    assert calls == [] and result.cases[0]["error"]["code"] == "request_field_missing"


def test_rest_queries_and_low_level_headers_do_not_leak_to_logs(caplog):
    def send(request):
        assert request.url.params["input"] == "private question"
        logging.getLogger("httpcore.http11").debug("private response header")
        return httpx.Response(200, json="answer")

    with caplog.at_level(logging.DEBUG):
        with RESTAdapter("https://example.invalid", method="GET", transport=httpx.MockTransport(send)) as adapter:
            assert adapter("private question") == "answer"
        logging.getLogger("httpx").info("unrelated application log")
    assert "private" not in caplog.text
    assert "unrelated application log" in caplog.text


def test_rest_cache_and_checkpoint_do_not_cross_authentication_contexts(monkeypatch, tmp_path):
    rows = [{"id": "q", "input": "q", "expected": "first"}]
    cache = tmp_path / "cache.sqlite"
    checkpoint = tmp_path / "checkpoint.sqlite"
    requests = []
    tokens = {"sample-first-token": "first", "sample-second-token": "second"}

    def send(request):
        authorization = request.headers.get("Authorization")
        requests.append(authorization)
        return httpx.Response(200, json=tokens.get(authorization, "anonymous"))

    def run(token, checkpoint_path=None):
        monkeypatch.setenv("LLMGUARD_TEST_AUTH", token or "unused")
        with RESTAdapter("https://example.invalid", transport=httpx.MockTransport(send),
                         auth_headers={"Authorization": "LLMGUARD_TEST_AUTH"} if token else None) as adapter:
            evaluator = Evaluator(rows, "qa", cache=cache)
            return evaluator.evaluate(System(adapter, model_version="same", prompt_version="none"),
                                      checkpoint=checkpoint_path).report()

    first = run("sample-first-token", checkpoint)
    assert run("sample-first-token", checkpoint)["execution"]["checkpoint_reused"] == 1
    with pytest.raises(ValueError, match="Checkpoint fingerprint"):
        run("sample-second-token", checkpoint)
    assert run("sample-first-token")["execution"]["cache_hits"] == 1
    second = run("sample-second-token")
    assert run("sample-second-token")["execution"]["cache_hits"] == 1
    assert run(None)["execution"]["cache_hits"] == 0
    assert requests == ["sample-first-token", "sample-second-token", None]
    assert first["metrics"]["exact_match"] == 1 and second["metrics"]["exact_match"] == 0
    assert not any(token in json.dumps([first, second]) for token in tokens)
    assert not any(token.encode() in path.read_bytes() for token in tokens for path in (cache, checkpoint))
