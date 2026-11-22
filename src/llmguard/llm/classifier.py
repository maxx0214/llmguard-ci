"""Public feature facade, independent of provider selection."""
from llmguard.llm.client import LLMClient
from llmguard.models import IntentClassificationResult, PromptConfig


async def classify_query(query: str, prompt: PromptConfig,
                         client: LLMClient) -> IntentClassificationResult:
    if not query.strip():
        raise ValueError("Query must not be blank")
    return (await client.classify(query, prompt)).value

