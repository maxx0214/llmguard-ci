import asyncio
import threading
from time import monotonic, sleep

import httpx
import pytest

from llmguard import Evaluator, ExecutionOptions, RESTAdapter, System


def dataset(count=4):
    return [{"id": str(index), "input": str(index), "expected": "yes"} for index in range(count)]


def versioned(function, *, model="v1", prompt="p1", settings=None, is_api=False):
    return System(function, model_version=model, prompt_version=prompt, settings=settings, is_api=is_api)


def test_parallel_preserves_canonical_ids_and_limits_in_flight():
    active = peak = 0
    completed = []

    async def predict(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.025 if value == "0" else .001)
        active -= 1
        completed.append(value)
        return "yes"

    result = Evaluator(dataset()[::-1], "qa", execution=ExecutionOptions(max_concurrency=2)).evaluate(predict)
    assert peak == 2 and completed[0] != "0"
    assert [case["id"] for case in result.cases] == ["0", "1", "2", "3"]
    assert result.report()["execution"]["mode"] == "parallel"
    assert result.metrics == {"exact_match": 1, "token_f1": 1}


def test_sync_timeout_drains_threads_before_retry_and_limits_concurrency():
    lock = threading.Lock()
    active = peak = calls = 0

    def slow(_):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        sleep(.025)
        with lock:
            active -= 1
        return "yes"

    evaluator = Evaluator(dataset(3), "qa", execution={"max_concurrency": 2, "timeout": .005, "retries": 1,
                                                        "retry_backoff": 0})
    result = evaluator.evaluate(slow)
    assert peak == 2 and calls == 6 and active == 0
    assert result.status == "FAIL"
    assert all(case["error"]["code"] == "timeout" and case["attempts"] == 2 for case in result.cases)


def test_async_timeout_retry_and_nontransient_error():
    calls = []

    async def predict(value):
        calls.append(value)
        if value == "1":
            raise ValueError("permanent")
        if calls.count(value) == 1:
            await asyncio.sleep(10)
        return "yes"

    evaluator = Evaluator(dataset(2), "qa", execution={"timeout": .01, "retries": 2, "retry_backoff": 0})
    result = evaluator.evaluate(predict)
    assert calls == ["0", "0", "1"]
    assert result.cases[0]["error"] is None and result.cases[1]["error"]["code"] == "call_failed"


def test_rest_retry_after_and_spacing():
    timestamps = []

    def send(_):
        timestamps.append(monotonic())
        if len(timestamps) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.03"})
        return httpx.Response(200, json={"answer": "yes", "usage": {"tokens": 2}, "cost": .01})

    options = ExecutionOptions(retries=1, requests_per_second=100, retry_backoff=0)
    evaluator = Evaluator(dataset(2), "qa", execution=options)
    with RESTAdapter("http://example.invalid", transport=httpx.MockTransport(send)) as rest:
        result = evaluator.evaluate(rest)
    execution = result.report()["execution"]
    assert len(timestamps) == execution["api_calls"] == 3
    assert timestamps[1] - timestamps[0] >= .028
    assert timestamps[2] - timestamps[1] >= .008
    assert execution["actual_cost"] is None  # Failed attempt cost was not reported.
    assert result.status == "PASS"


def test_long_server_cooldown_does_not_send_more_requests():
    attempts = []

    def send(request):
        attempts.append(request)
        return httpx.Response(429, headers={"Retry-After": "100"})

    evaluator = Evaluator(dataset(2), "qa", execution={"retries": 2, "max_retry_delay": 0.1})
    with RESTAdapter("http://example.invalid", transport=httpx.MockTransport(send)) as rest:
        result = evaluator.evaluate(rest)
    assert len(attempts) == 1
    assert result.cases[1]["error"]["code"] == "rate_limit_wait_exceeded"


def test_sqlite_cache_persists_deduplicates_and_invalidates(tmp_path):
    path = tmp_path / "responses.sqlite"
    rows = [{**row, "input": "same private input"} for row in dataset()]
    calls = []

    def predict(value):
        calls.append(value)
        return {"answer": "yes", "cost": .03, "usage": {"tokens": 3}}

    def evaluate(system, cache=path):
        return Evaluator(rows, "qa", cache=cache, execution={"max_concurrency": 4}).evaluate(system).report()

    system = versioned(predict, is_api=True, settings={"temperature": 0})
    first = evaluate(system)
    assert first["execution"]["cache_hits"] == 3 and first["execution"]["api_calls_saved"] == 3
    assert first["execution"]["actual_cost"] == .03 and first["execution"]["actual_usage"] == {"tokens": 3}
    second = evaluate(system)
    assert second["execution"]["cache_hit_rate"] == 1 and second["execution"]["api_calls"] == 0
    assert second["execution"]["actual_cost"] == 0 and second["operational"]["cost"] == .12
    assert len(calls) == 1
    evaluate(versioned(predict, model="new", is_api=True, settings={"temperature": 0}))
    evaluate(versioned(predict, prompt="new", is_api=True, settings={"temperature": 0}))
    evaluate(versioned(predict, is_api=True, settings={"temperature": .5}))
    assert len(calls) == 4
    uncached = evaluate(system, cache=None)
    assert len(calls) == 8 and uncached["execution"]["cache_hit_rate"] is None
    independent = Evaluator(rows, "qa", cache=path, execution={"max_concurrency": 4, "independent_trials": True})
    trial = independent.evaluate(system).report()["execution"]
    assert len(calls) == 12 and not trial["cache_enabled"] and trial["api_calls"] == 4
    assert b"same private input" not in path.read_bytes()


def test_parallel_checkpoint_interrupt_resume_and_exclusive_owner(tmp_path):
    path = tmp_path / "run.sqlite"
    evaluator = Evaluator(dataset(), "qa", execution={"max_concurrency": 2})

    async def exercise():
        both_pending = asyncio.Event()
        pending = []

        async def interrupted(value):
            if value in ("2", "3"):
                pending.append(value)
                if len(pending) == 2:
                    both_pending.set()
                await asyncio.Event().wait()
            return "yes"

        task = asyncio.create_task(evaluator.aevaluate(versioned(interrupted), checkpoint=path))
        await asyncio.wait_for(both_pending.wait(), 3)
        with pytest.raises(ValueError, match="already open"):
            await evaluator.aevaluate(versioned(interrupted), checkpoint=path)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        needed = []

        async def resumed(value):
            needed.append(value)
            return "yes"

        result = await evaluator.aevaluate(versioned(resumed), checkpoint=path)
        assert sorted(needed) == ["2", "3"]
        assert result.status == "PASS" and len(result.cases) == 4
        assert result.report()["execution"]["checkpoint_reused"] == 2
        assert [case["execution_origin"] for case in result.cases] == ["checkpoint"] * 2 + ["live"] * 2

    asyncio.run(exercise())


def test_checkpoint_only_retries_failures_and_validates_settings(tmp_path):
    path = tmp_path / "run.sqlite"
    cache = tmp_path / "cache.sqlite"
    evaluator = Evaluator(dataset(2), "qa", cache=cache)
    calls = []

    def initial(value):
        calls.append(value)
        return "yes" if value == "0" else {"answer": 1}

    assert evaluator.evaluate(versioned(initial), checkpoint=path).status == "FAIL"
    assert evaluator.evaluate(versioned(initial), checkpoint=path).status == "FAIL"
    assert calls == ["0", "1"]

    def fixed(value):
        calls.append(value)
        return "yes"

    retry = evaluator.evaluate(versioned(fixed), checkpoint=path, retry_failed=True)
    assert retry.status == "PASS" and calls == ["0", "1", "1"]
    assert retry.report()["execution"]["checkpoint_reused"] == 1
    for changed in (Evaluator(dataset(3), "qa", cache=cache),
                    Evaluator(dataset(2), "qa", cache=cache, execution={"max_concurrency": 2})):
        with pytest.raises(ValueError, match="Checkpoint fingerprint"):
            changed.evaluate(versioned(fixed), checkpoint=path)
    with pytest.raises(ValueError, match="Checkpoint fingerprint"):
        evaluator.evaluate(versioned(fixed, model="v2"), checkpoint=path)
    assert calls == ["0", "1", "1"]
