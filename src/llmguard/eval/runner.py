"""Bounded classifier workers followed by deterministic metrics and selective judging."""
import asyncio
import logging
from time import perf_counter

from llmguard.config import JudgeConfig, PricingConfig, fingerprint
from llmguard.eval.intent_judge import JUDGE_PROMPT, analyze_cases
from llmguard.eval.metrics import compute_metrics
from llmguard.llm.client import LLMClient, LLMError
from llmguard.models import Banking77Case, DatasetInfo, EvaluationRun, PromptConfig, TestCaseResult
from llmguard.regression.comparator import ensure_compatible

logger = logging.getLogger(__name__)
METRIC_CONTRACT = "banking77-all-cases-macro77-v1"


def run_metadata(prompt: PromptConfig, pricing: PricingConfig, judge_model: str,
                 judge: JudgeConfig, offline: bool, concurrency: int) -> dict:
    return {"mode": "offline" if offline else "live", "metric_contract": METRIC_CONTRACT,
            "judge_model": judge_model, "judge_provider": prompt.provider,
            "judge_prompt_hash": fingerprint(JUDGE_PROMPT), "judge_config": judge.model_dump(),
            "pricing": pricing.model_dump(), "concurrency": concurrency,
            "prompt_hash": fingerprint(prompt.model_dump(mode="json")),
            "latency_scope": "classifier only, including retries, excluding queue wait",
            "cost_scope": "classifier and judge reported separately; observed usage only"}


async def evaluate(prompt: PromptConfig, cases: list[Banking77Case], dataset: DatasetInfo,
                   client: LLMClient, pricing: PricingConfig, judge_model: str,
                   judge: JudgeConfig | None = None, concurrency: int = 3,
                   baseline: EvaluationRun | None = None, offline: bool = False) -> EvaluationRun:
    judge = judge or JudgeConfig()
    if not 1 <= concurrency <= 32:
        raise ValueError("Concurrency must be between 1 and 32")
    if not cases or len({c.id for c in cases}) != len(cases) or dataset.sample_count != len(cases):
        raise ValueError("Dataset must be nonempty, correctly counted, with unique IDs")
    if dataset.fingerprint != fingerprint([c.model_dump() for c in cases]):
        raise ValueError("Dataset fingerprint does not match cases")
    run = EvaluationRun(prompt=prompt, dataset=dataset, cases=[], metrics=compute_metrics([]),
                        metadata=run_metadata(prompt, pricing, judge_model, judge, offline, concurrency))
    if baseline:
        ensure_compatible(baseline, run)
    results = []
    iterator = iter(cases)
    async def worker() -> None:
        for case in iterator:
            start = perf_counter()
            predicted, error, kind, usage = None, None, None, []
            try:
                reply = await client.classify(case.text, prompt)
                predicted, usage = reply.value.intent, reply.usage
            except LLMError as exc:
                error, kind, usage = str(exc), exc.kind, exc.usage
                logger.warning("case_error case_id=%s kind=%s", case.id, kind)
            results.append(TestCaseResult(test_case_id=case.id, input_text=case.text,
                expected_intent=case.expected_intent, predicted_intent=predicted,
                correct=predicted == case.expected_intent and error is None,
                latency_ms=100 if offline else (perf_counter() - start) * 1000,
                input_tokens=sum(u.input_tokens for u in usage),
                output_tokens=sum(u.output_tokens for u in usage),
                estimated_cost=pricing.estimate(usage) if usage else None,
                error=error, error_kind=kind, usage=usage))
    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(cases))):
            group.create_task(worker())
    order = {case.id: i for i, case in enumerate(cases)}
    run.cases = sorted(results, key=lambda c: order[c.test_case_id])
    run.metrics = compute_metrics(run.cases)
    run.judgments, selected = await analyze_cases(run.cases, baseline, client, pricing, prompt.provider,
                                                judge_model, judge, dataset.seed, concurrency, offline)
    run.metadata["judge_selected_count"] = selected
    run.metrics = compute_metrics(run.cases, run.judgments)
    run.status = "FAIL" if run.metrics.error_count or run.metrics.invalid_predictions else "PASS"
    return run
