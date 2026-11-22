"""Validated contracts shared by evaluation, storage, and reporting."""
from datetime import date, datetime, timezone
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from llmguard.data.intents import Intent

Status = Literal["PASS", "WARN", "FAIL"]
Tier = Literal["smoke", "regression", "full"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PromptConfig(Model):
    id: str = Field(min_length=1, pattern=r"\S")
    version: int = Field(ge=1, strict=True)
    provider: Literal["openai"] = "openai"
    model: str | None = Field(default=None, min_length=1, pattern=r"\S")
    temperature: float = Field(ge=0, le=2)
    created_at: date
    description: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1, pattern=r"\S")


class IntentClassificationResult(Model):
    intent: Intent


class Banking77Case(Model):
    id: str = Field(pattern=r"^BANKING77-(TEST|TRAIN|FIXTURE)-\d{4,}$")
    text: str = Field(min_length=1)
    expected_intent: Intent


class IntentJudgeResult(Model):
    semantic_fit: int = Field(ge=1, le=5, strict=True)
    severity: Literal["minor", "moderate", "critical"]
    ambiguity: Literal["low", "medium", "high"]
    error_type: Literal["semantically_close_intents", "missed_key_detail",
        "wrong_transaction_type", "wrong_payment_state", "wrong_card_state",
        "wrong_transfer_state", "identity_or_account_confusion", "unrelated_intent",
        "ambiguous_query", "other"]
    explanation: str = Field(min_length=1, max_length=2000, pattern=r"\S")


class TokenUsage(Model):
    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    complete: bool = True


class TestCaseResult(Model):
    __test__ = False
    test_case_id: str
    input_text: str
    expected_intent: Intent
    predicted_intent: Intent | None = None
    correct: bool = False
    latency_ms: float = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    error: str | None = None
    error_kind: Literal["invalid_output", "execution"] | None = None
    usage: list[TokenUsage] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.correct != (self.predicted_intent == self.expected_intent and self.error is None):
            raise ValueError("Correctness must match authoritative ground truth")
        if (self.error is None) != (self.error_kind is None):
            raise ValueError("error and error_kind must be present together")
        if self.predicted_intent is None and self.error is None:
            raise ValueError("Missing prediction requires an evaluation error")
        return self


class JudgeCaseResult(Model):
    test_case_id: str
    result: IntentJudgeResult | None = None
    error: str | None = None
    latency_ms: float = Field(default=0, ge=0)
    usage: list[TokenUsage] = Field(default_factory=list)
    estimated_cost: float | None = Field(default=None, ge=0)


class ClassMetrics(Model):
    support: int
    precision: float
    recall: float
    f1: float


class ConfusionPair(Model):
    expected_intent: str
    predicted_intent: str
    count: int


class Metrics(Model):
    total_cases: int
    valid_predictions: int
    invalid_predictions: int
    error_count: int
    error_rate: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_labels: list[str]
    confusion_matrix: list[list[int]]
    top_confusions: list[ConfusionPair]
    latency_mean: float
    latency_p50: float
    latency_p95: float
    input_tokens: int
    output_tokens: int
    estimated_cost: float | None
    judge_count: int = 0
    judge_error_count: int = 0
    judge_semantic_fit: float | None = None
    judge_severity_counts: dict[str, int] = Field(default_factory=dict)
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_cost: float | None = 0
    total_estimated_cost: float | None = None


class DatasetInfo(Model):
    dataset: str = "PolyAI/banking77"
    split: Literal["test", "train"] = "test"
    tier: Tier = "smoke"
    sample_count: int = Field(ge=1)
    seed: int = 42
    revision: str
    fingerprint: str
    sampling: str = "stratified-round-robin-v1"
    synthetic: bool = False


class EvaluationRun(Model):
    schema_version: Literal[2] = 2
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: PromptConfig
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset: DatasetInfo
    metadata: dict = Field(default_factory=dict)
    metrics: Metrics
    cases: list[TestCaseResult]
    judgments: list[JudgeCaseResult] = Field(default_factory=list)
    is_baseline: bool = False
    status: Status = "PASS"


class MetricDelta(Model):
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    change_pct: float | None = None
    severity: Literal["PASS", "WARN", "FAIL", "N/A"] = "PASS"
    note: str = ""


class RegressionResult(Model):
    baseline_run_id: str
    candidate_run_id: str
    status: Status
    metrics: list[MetricDelta]
    transitions: dict[str, list[str]]
    reasons: list[str] = Field(default_factory=list)

