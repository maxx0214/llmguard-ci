from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from llmguard.config import load_yaml
from llmguard.models import Model


class DropThreshold(Model):
    warning_drop: float = Field(gt=0)
    critical_drop: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.warning_drop > self.critical_drop:
            raise ValueError("warning_drop must not exceed critical_drop")
        return self


class IncreaseThreshold(Model):
    warning_increase_pct: float = Field(gt=0)
    critical_increase_pct: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.warning_increase_pct > self.critical_increase_pct:
            raise ValueError("warning increase must not exceed critical increase")
        return self


class JudgeThreshold(DropThreshold):
    enabled_as_gate: bool = False


class Thresholds(Model):
    accuracy: DropThreshold
    macro_f1: DropThreshold
    latency_p95: IncreaseThreshold
    cost: IncreaseThreshold
    judge_semantic_fit: JudgeThreshold
    max_error_rate: float = Field(default=0, ge=0, lt=1)


def load_thresholds(path: Path) -> Thresholds:
    return Thresholds.model_validate(load_yaml(path))

