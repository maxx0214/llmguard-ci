"""OpenAI structured-output adapter."""
import os

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import ValidationError
from llmguard.llm.client import LLMError, Reply, StructuredClient, T
from llmguard.models import TokenUsage


class OpenAIClient(StructuredClient):
    def __init__(self, timeout: float = 30, sdk: AsyncOpenAI | None = None):
        if sdk is None and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        self.sdk = sdk or AsyncOpenAI(timeout=timeout, max_retries=2)

    async def close(self) -> None:
        await self.sdk.close()

    async def _request(self, model: str, temperature: float, system: str,
                       user: str, schema: type[T]) -> Reply[T]:
        usage: list[TokenUsage] = []
        for attempt in range(2):
            try:
                response = await self.sdk.chat.completions.create(
                    model=model, temperature=temperature,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}},
                )
            except (APIStatusError, APIConnectionError) as exc:
                usage.append(TokenUsage(provider="openai", model=model, input_tokens=0,
                                        output_tokens=0, complete=False))
                detail = f"HTTP {exc.status_code}" if isinstance(exc, APIStatusError) else "connection/timeout"
                raise LLMError(f"OpenAI {detail} failure", usage) from exc
            observed = response.usage
            usage.append(TokenUsage(provider="openai", model=model,
                                    input_tokens=observed.prompt_tokens if observed else 0,
                                    output_tokens=observed.completion_tokens if observed else 0,
                                    complete=observed is not None))
            try:
                if not response.choices:
                    raise ValueError("No response choices")
                choice = response.choices[0]
                if choice.message.refusal or choice.finish_reason != "stop":
                    raise ValueError("Refused or incomplete response")
                return Reply(schema.model_validate_json(choice.message.content or ""), usage)
            except (ValidationError, ValueError) as exc:
                if attempt == 1:
                    raise LLMError("Invalid structured output after two attempts", usage, "invalid_output") from exc
        raise AssertionError("Unreachable")
