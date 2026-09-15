"""Strict, deterministic configuration loaded before provider requests."""
import hashlib
import json
import os
from pathlib import Path

import yaml
from pydantic import Field
from llmguard.models import Model, PromptConfig, TokenUsage


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_prompt(path: Path, model: str | None = None) -> PromptConfig:
    prompt = PromptConfig.model_validate(load_yaml(path))
    prefix = prompt.provider.upper()
    chosen = model or os.getenv(f"{prefix}_CLASSIFIER_MODEL") or prompt.model
    if not chosen or not chosen.strip():
        raise ValueError(f"Set {prefix}_CLASSIFIER_MODEL, --model, or prompt.model")
    return prompt.model_copy(update={"model": chosen})


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


class EvaluationConfig(Model):
    seed: int = 42
    smoke_size: int = Field(default=100, ge=1)
    regression_size: int = Field(default=500, ge=1)
    concurrency: int = Field(default=3, ge=1, le=32)
    timeout: float = Field(default=30, gt=0)


class JudgeConfig(Model):
    enabled: bool = True
    max_cases: int = Field(default=100, ge=0)
    model: str | None = None


class Settings(Model):
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class ModelPrice(Model):
    input_cost_per_million_tokens: float = Field(ge=0)
    output_cost_per_million_tokens: float = Field(ge=0)


class PricingConfig(Model):
    as_of: str | None = None
    source: str | None = None
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    def estimate(self, usage: list[TokenUsage]) -> float | None:
        if any(not u.complete or f"{u.provider}/{u.model}" not in self.models for u in usage):
            return None
        return sum((u.input_tokens * self.models[f"{u.provider}/{u.model}"].input_cost_per_million_tokens
                    + u.output_tokens * self.models[f"{u.provider}/{u.model}"].output_cost_per_million_tokens)
                   / 1_000_000 for u in usage)
