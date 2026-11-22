"""Provider-neutral structured calls; evaluation never imports a provider SDK."""
import json
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel
from llmguard.data.intents import INTENTS
from llmguard.models import IntentClassificationResult, IntentJudgeResult, PromptConfig, TokenUsage

T = TypeVar("T", bound=BaseModel)


@dataclass
class Reply(Generic[T]):
    value: T
    usage: list[TokenUsage]


class LLMError(RuntimeError):
    def __init__(self, message: str, usage: list[TokenUsage] | None = None,
                 kind: str = "execution"):
        super().__init__(message)
        self.usage = usage or []
        self.kind = kind


class LLMClient(Protocol):
    async def classify(self, query: str, prompt: PromptConfig) -> Reply[IntentClassificationResult]: ...
    async def judge_intent(self, system: str, payload: dict,
                           model: str) -> Reply[IntentJudgeResult]: ...
    async def close(self) -> None: ...


class StructuredClient:
    """Shared prompt assembly for provider adapters."""
    async def _request(self, model: str, temperature: float, system: str,
                       user: str, schema: type[T]) -> Reply[T]:
        raise NotImplementedError

    async def classify(self, query: str, prompt: PromptConfig) -> Reply[IntentClassificationResult]:
        system = prompt.system_prompt + "\nAllowed BANKING77 intents:\n" + "\n".join(INTENTS)
        return await self._request(str(prompt.model), prompt.temperature, system,
                                   json.dumps({"query": query}, ensure_ascii=False), IntentClassificationResult)

    async def judge_intent(self, system: str, payload: dict, model: str) -> Reply[IntentJudgeResult]:
        return await self._request(model, 0, system, json.dumps(payload, ensure_ascii=False), IntentJudgeResult)

