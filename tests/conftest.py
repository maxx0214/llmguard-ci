from pathlib import Path
import socket
import pytest

from llmguard.config import PricingConfig, load_prompt, fingerprint
from llmguard.eval.metrics import compute_metrics
from llmguard.eval.runner import run_metadata
from llmguard.config import JudgeConfig
from llmguard.models import DatasetInfo, EvaluationRun, TestCaseResult
from llmguard.regression.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    original_connect = socket.socket.connect
    def guarded_connect(sock, address):
        # Windows asyncio creates a loopback socket pair for its wakeup pipe.
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return original_connect(sock, address)
        raise AssertionError("Unit tests must not access external networks")
    def forbidden(*args, **kwargs):
        raise AssertionError("Unit tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def prompt():
    return load_prompt(ROOT / "prompts/banking77_v1.yaml", model="fixture-model")


@pytest.fixture
def pricing():
    return PricingConfig(as_of="test", source="synthetic test prices", models={
        "openai/fixture-model": {"input_cost_per_million_tokens": 1, "output_cost_per_million_tokens": 2}})


@pytest.fixture
def thresholds():
    return load_thresholds(ROOT / "config/thresholds.yaml")


def case_result(case_id="BANKING77-TEST-0001", expected="card_arrival", predicted="card_arrival",
                error=None, kind=None):
    return TestCaseResult(test_case_id=case_id, input_text=f"Query {case_id}", expected_intent=expected,
                          predicted_intent=predicted, correct=expected == predicted and error is None,
                          error=error, error_kind=kind, latency_ms=100, estimated_cost=.001)


@pytest.fixture
def make_run(prompt, pricing):
    def factory(cases=None, baseline=False):
        cases = cases if cases is not None else [case_result()]
        data = [{"id": c.test_case_id, "text": c.input_text, "expected_intent": c.expected_intent} for c in cases]
        return EvaluationRun(prompt=prompt, cases=cases, metrics=compute_metrics(cases),
                             dataset=DatasetInfo(sample_count=len(cases), revision="test", fingerprint=fingerprint(data)),
                             is_baseline=baseline,
                             metadata=run_metadata(prompt, pricing, prompt.model, JudgeConfig(), True, 3))
    return factory

