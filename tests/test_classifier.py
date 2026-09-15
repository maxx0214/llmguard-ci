import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from llmguard.data.intents import INTENTS
from llmguard.llm.openai import OpenAIClient
from llmguard.llm.classifier import classify_query
from llmguard.llm.client import LLMError
from llmguard.models import IntentClassificationResult


def response(text='{"intent":"card_arrival"}', finish="stop", usage=True):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(refusal=None, content=text))], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5) if usage else None)


def fake_openai(responses):
    create = AsyncMock(side_effect=responses)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock())
    return OpenAIClient(sdk=sdk), create


def test_openai_valid_schema_and_canonical_labels(prompt):
    client, create = fake_openai([response()])
    reply = asyncio.run(client.classify("Where is my card?", prompt))
    assert reply.value.intent == "card_arrival"
    assert reply.usage[0].output_tokens == 5
    schema = create.call_args.kwargs["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["intent"]["enum"] == list(INTENTS)
    assert all(label in create.call_args.kwargs["messages"][0]["content"] for label in INTENTS)


@pytest.mark.parametrize("intent", ["unknown", "Card_arrival", "card_arrival ", 1, None])
def test_invalid_intent(intent):
    with pytest.raises(ValidationError):
        IntentClassificationResult(intent=intent)


def test_retry_and_usage(prompt):
    client, create = fake_openai([response('{"intent":"bad"}'), response()])
    reply = asyncio.run(client.classify("query", prompt))
    assert create.await_count == 2
    assert sum(u.input_tokens for u in reply.usage) == 20


def test_invalid_after_retry(prompt):
    client, create = fake_openai([response("not-json"), response("{}")])
    with pytest.raises(LLMError) as caught:
        asyncio.run(client.classify("query", prompt))
    assert caught.value.kind == "invalid_output"
    assert create.await_count == 2


def test_transport_failure_not_retried(prompt):
    client, create = fake_openai([RuntimeError("connection")])
    with pytest.raises(RuntimeError):
        asyncio.run(client.classify("query", prompt))
    assert create.await_count == 1


def test_missing_usage_keeps_prediction(prompt):
    client, _ = fake_openai([response(usage=False)])
    reply = asyncio.run(client.classify("query", prompt))
    assert reply.value.intent == "card_arrival"
    assert not reply.usage[0].complete


def test_public_facade_and_empty_input(prompt):
    client, _ = fake_openai([response()])
    assert asyncio.run(classify_query("where is my card", prompt, client)).intent == "card_arrival"
    with pytest.raises(ValueError):
        asyncio.run(classify_query(" ", prompt, client))


def test_openai_key_required():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_openai_sdk_serialization_without_network(prompt):
    import json
    import httpx
    from openai import AsyncOpenAI
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": '{"intent":"card_arrival"}', "refusal": None}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    async def check():
        sdk = AsyncOpenAI(api_key="offline-test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        client = OpenAIClient(sdk=sdk)
        try:
            assert (await client.classify("Where is my card?", prompt)).value.intent == "card_arrival"
        finally:
            await client.close()
    asyncio.run(check())
    assert len(requests) == 1
    assert requests[0]["response_format"]["json_schema"]["schema"]["properties"]["intent"]["enum"] == list(INTENTS)
