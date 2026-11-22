"""Deterministic infrastructure fixtures. Never evidence of model quality."""
import asyncio

from llmguard.config import fingerprint
from llmguard.data.intents import INTENTS
from llmguard.llm.client import Reply
from llmguard.models import (Banking77Case, DatasetInfo, IntentClassificationResult,
                             IntentJudgeResult, PromptConfig, TokenUsage)


def fixture_dataset() -> tuple[list[Banking77Case], DatasetInfo]:
    cases = [Banking77Case(id=f"BANKING77-FIXTURE-{i+1:04d}",
                          text=f"Offline infrastructure fixture {i+1}.",
                          expected_intent=INTENTS[i % 77]) for i in range(100)]
    return cases, DatasetInfo(dataset="synthetic/banking77-fixture", sample_count=len(cases),
                              revision="fixture-v2", synthetic=True,
                              fingerprint=fingerprint([c.model_dump() for c in cases]))


class OfflineClient:
    def __init__(self, cases: list[Banking77Case]):
        self.cases = {case.text: case for case in cases}

    async def classify(self, query: str, prompt: PromptConfig) -> Reply[IntentClassificationResult]:
        await asyncio.sleep(0)
        case = self.cases[query]
        intent = case.expected_intent
        index = int(case.id.rsplit("-", 1)[1])
        wrong = index % (8 if prompt.version > 1 else 20) == 0
        if wrong:
            intent = INTENTS[(INTENTS.index(intent) + (2 if prompt.version > 1 else 1)) % 77]
        return Reply(IntentClassificationResult(intent=intent),
                     [TokenUsage(provider=prompt.provider, model=str(prompt.model),
                                 input_tokens=350, output_tokens=12)])

    async def judge_intent(self, system: str, payload: dict, model: str) -> Reply[IntentJudgeResult]:
        await asyncio.sleep(0)
        return Reply(IntentJudgeResult(semantic_fit=2, severity="moderate", ambiguity="low",
                                       error_type="other", explanation="Synthetic fixture analysis; no model judgment."),
                     [TokenUsage(provider=payload["provider"], model=model, input_tokens=200, output_tokens=50)])

    async def close(self) -> None:
        pass

