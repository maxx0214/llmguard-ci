"""Keep SDK-owned requests out of third-party HTTP debug logs."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar


_PRIVATE_REQUEST = ContextVar("private_llmguard_request", default=False)
_LOGGERS = ("openai._base_client", "httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2",
            "httpcore.proxy", "httpcore.socks")


@contextmanager
def private_request_logs():
    token = _PRIVATE_REQUEST.set(True)
    loggers = [logging.getLogger(name) for name in _LOGGERS]

    def outside_request(_record):
        return not _PRIVATE_REQUEST.get()

    for logger in loggers:
        logger.addFilter(outside_request)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(outside_request)
        _PRIVATE_REQUEST.reset(token)
