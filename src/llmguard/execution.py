"""Bounded asynchronous execution of system requests."""
from __future__ import annotations

import asyncio
import inspect
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic, perf_counter

from .output import nonnegative
from .rest import RESTAdapter, RESTError, TRANSIENT_STATUS_CODES


@dataclass(frozen=True)
class ExecutionOptions:
    max_concurrency: int = 1
    requests_per_second: float | None = None
    timeout: float | None = None
    retries: int | None = None
    retry_backoff: float = 0.1
    max_retry_delay: float = 30.0
    independent_trials: bool = False

    def __post_init__(self):
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        for name in ("requests_per_second", "timeout"):
            value = getattr(self, name)
            if value is not None and (not nonnegative(value) or value == 0):
                raise ValueError(f"{name} must be positive")
        if self.retries is not None and (type(self.retries) is not int or not 0 <= self.retries <= 3):
            raise ValueError("retries must be an integer in [0, 3]")
        if not all(nonnegative(value) for value in (self.retry_backoff, self.max_retry_delay)):
            raise ValueError("Retry delays must be finite and nonnegative")
        if type(self.independent_trials) is not bool:
            raise ValueError("independent_trials must be a boolean")


class RequestLimiter:
    def __init__(self, requests_per_second: float | None):
        self.interval = 1 / requests_per_second if requests_per_second else 0
        if not nonnegative(self.interval):
            raise ValueError("requests_per_second is too small")
        self.next_request = 0.0
        self.blocked = False
        self.lock = asyncio.Lock()
        self.changed = asyncio.Event()

    async def wait(self):
        while True:
            async with self.lock:
                if self.blocked:
                    return False
                now = monotonic()
                pause = self.next_request - now
                if pause <= 0:
                    self.next_request = now + self.interval
                    return True
                changed = self.changed
            try:
                await asyncio.wait_for(changed.wait(), pause)
            except TimeoutError:
                pass

    def defer(self, seconds: float):
        # Called only from the event loop, without yielding before the assignment.
        self.next_request = max(self.next_request, monotonic() + seconds)
        self._wake()

    def stop(self):
        self.blocked = True
        self._wake()

    def _wake(self):
        previous, self.changed = self.changed, asyncio.Event()
        previous.set()


async def invoke(function, value):
    if inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(getattr(function, "__call__", None)):
        result = function(value)
    else:
        worker = asyncio.create_task(asyncio.to_thread(function, value))
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Keep the worker slot occupied until a synchronous call actually returns.
            try:
                late = await worker
                if inspect.iscoroutine(late):
                    late.close()
            except Exception:
                pass
            raise
    return await result if inspect.isawaitable(result) else result


@dataclass
class Response:
    value: object = None
    latency_ms: float = 0.0
    attempts: int = 0
    api_calls: int = 0
    error: dict | None = None


class RateLimitError(ConnectionError):
    """Signal that a provider-independent Python callable has been rate limited."""

    def __init__(self, retry_after: float | None = None):
        if retry_after is not None and not nonnegative(retry_after):
            raise ValueError("retry_after must be finite and nonnegative")
        self.retry_after = retry_after
        super().__init__("System rate limit reached.")


def _request_error(exc: Exception) -> tuple[dict, bool, float | None]:
    if isinstance(exc, RESTError):
        error = {"code": exc.code, "message": str(exc), "attempts": exc.attempts, "http_status": exc.status_code}
        retry = exc.code in {"timeout", "transport_error"} or exc.status_code in TRANSIENT_STATUS_CODES
        return error, retry, exc.retry_after if retry else None
    if isinstance(exc, TimeoutError):
        return {"code": "timeout", "message": "System request timed out."}, True, None
    if isinstance(exc, RateLimitError):
        return {"code": "rate_limited", "message": str(exc)}, True, exc.retry_after
    if isinstance(exc, ConnectionError):
        return {"code": "transport_error", "message": "System connection failed."}, True, None
    return {"code": "call_failed", "message": "System callable failed.", "exception_type": type(exc).__name__}, False, None


async def request(system, value, options: ExecutionOptions, limiter: RequestLimiter) -> Response:
    rest = isinstance(system.predict, RESTAdapter)
    function = system.predict.request_once if rest else system.predict
    retries = options.retries if options.retries is not None else system.predict.retries if rest else 0
    response = Response()
    first_started = None
    for attempt in range(retries + 1):
        if not await limiter.wait():
            response.error = {"code": "rate_limit_wait_exceeded", "message": "Server wait exceeds the retry delay limit.",
                              "attempts": response.attempts}
            if first_started is not None:
                response.latency_ms = (perf_counter() - first_started) * 1000
            return response
        started = perf_counter()
        if first_started is None:
            first_started = started
        response.attempts += 1
        if rest or system.is_api:
            response.api_calls += 1
        try:
            async with asyncio.timeout(options.timeout):
                response.value = await invoke(function, deepcopy(value))
                if options.timeout is not None and perf_counter() - started > options.timeout:
                    raise TimeoutError
        except Exception as exc:
            response.error, retryable, retry_after = _request_error(exc)
            if rest and isinstance(exc, RESTError) and exc.attempts == 0:
                response.api_calls -= 1
            delay = min(options.max_retry_delay, max(options.retry_backoff * 2 ** attempt, retry_after or 0))
            if retry_after is not None and retry_after > options.max_retry_delay:
                limiter.stop()
                retryable = False
            shared_cooldown = (retry_after is not None or response.error.get("http_status") == 429
                               or isinstance(exc, RateLimitError))
            if shared_cooldown:
                limiter.defer(delay)
        else:
            response.error = None
            return response
        finally:
            response.latency_ms = (perf_counter() - first_started) * 1000
        if not retryable or attempt == retries:
            response.error["attempts"] = response.attempts
            return response
        if not shared_cooldown:
            await asyncio.sleep(delay)
    return response


async def run_workers(rows, max_concurrency: int, process):
    queue = iter(rows)

    async def worker():
        for row in queue:
            await process(row)

    # TaskGroup cancels and drains peer requests before storage is closed on interruption.
    async with asyncio.TaskGroup() as group:
        for _ in range(min(len(rows), max_concurrency)):
            group.create_task(worker())
