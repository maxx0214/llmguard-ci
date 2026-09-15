import pytest
from pydantic import ValidationError
from conftest import ROOT
from llmguard.cli import parser, effective_settings
from llmguard.config import PricingConfig, load_prompt, load_yaml
from llmguard.models import PromptConfig, TokenUsage
from llmguard.regression.thresholds import Thresholds


def test_prompt_loading_and_model_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_CLASSIFIER_MODEL", "from-env")
    assert load_prompt(ROOT / "prompts/banking77_v1.yaml").model == "from-env"
    assert load_prompt(ROOT / "prompts/banking77_v1.yaml", "from-cli").model == "from-cli"
    assert load_prompt(ROOT / "prompts/banking77_v2.yaml").version == 2


def test_model_required(monkeypatch):
    monkeypatch.delenv("OPENAI_CLASSIFIER_MODEL", raising=False)
    with pytest.raises(ValueError, match="CLASSIFIER_MODEL"):
        load_prompt(ROOT / "prompts/banking77_v1.yaml")


@pytest.mark.parametrize("field,value", [("version", 0), ("version", "1"), ("provider", "bad"),
    ("temperature", 3), ("temperature", float("nan")), ("system_prompt", " "), ("created_at", "bad")])
def test_invalid_prompt(prompt, field, value):
    data = prompt.model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(data)


def test_invalid_yaml_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- list\n- not-a-map\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)


def test_inverted_thresholds(thresholds):
    data = thresholds.model_dump()
    data["cost"]["warning_increase_pct"] = 80
    with pytest.raises(ValidationError):
        Thresholds.model_validate(data)


def test_missing_pricing_is_unavailable():
    usage = [TokenUsage(provider="openai", model="unknown", input_tokens=100, output_tokens=10)]
    assert PricingConfig().estimate(usage) is None
    assert PricingConfig().estimate([]) == 0


def test_incomplete_usage_has_unknown_cost(pricing):
    usage = [TokenUsage(provider="openai", model="fixture-model", input_tokens=100, output_tokens=10, complete=False)]
    assert pricing.estimate(usage) is None


def test_cli_settings(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("LLMGUARD_CONCURRENCY", "4")
    args = parser().parse_args(["evaluate", "--prompt", "x", "--tier", "regression", "--seed", "19",
                               "--judge-max-cases", "7", "--disable-judge"])
    settings = effective_settings(args)
    assert settings.evaluation.seed == 19
    assert settings.evaluation.concurrency == 4
    assert settings.judge.max_cases == 7 and not settings.judge.enabled


@pytest.mark.parametrize("flag,value", [("--concurrency","0"),("--judge-max-cases","-1"),("--timeout","nan")])
def test_cli_invalid_values(monkeypatch, flag, value):
    monkeypatch.chdir(ROOT)
    args = parser().parse_args(["evaluate","--prompt","x",flag,value])
    with pytest.raises(ValueError):
        effective_settings(args)
