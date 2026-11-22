"""A provider-independent HTTP adapter for local and deployed systems."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import sleep

try:
    import httpx
except ModuleNotFoundError as exc:
    if exc.name != "httpx":
        raise
    httpx = None

from ._privacy import private_request_logs
from .output import MISSING, nonnegative, project, project_output, select, validate_paths


TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_HEADER = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ENVIRONMENT_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_ERROR_MESSAGES = {"timeout": "HTTP request timed out.", "transport_error": "HTTP transport failed.",
                   "invalid_json": "HTTP response was not valid JSON.", "http_error": "HTTP request failed.",
                   "request_field_missing": "A configured REST request field is missing from the input."}


class RESTError(RuntimeError):
    """A safe error that carries no endpoint, authentication or response values."""

    def __init__(self, code: str, attempts: int, status_code: int | None = None, retry_after: float | None = None):
        self.code = code
        self.attempts = attempts
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"HTTP {status_code}" if status_code is not None else _ERROR_MESSAGES[code])


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if nonnegative(seconds) else None


def _request_body(mapping: dict[str, str], value) -> dict:
    body = {}
    for target, path in mapping.items():
        found = select({"input": value}, path.split("."))
        if found is MISSING:
            raise RESTError("request_field_missing", 0)
        parts = target.split(".")
        parent = body
        for key in parts[:-1]:
            parent = parent.setdefault(key, {})
        parent[parts[-1]] = found
    return body


class RESTAdapter:
    """Call a GET/POST JSON service with optional request and response mappings."""

    def __init__(self, url: str, *, method: str = "POST", request_mapping: Mapping[str, str] | None = None,
                 response_mapping: Mapping[str, str] | None = None, auth_headers: Mapping[str, str] | None = None,
                 timeout: float = 10.0, retries: int = 0, transport: httpx.BaseTransport | None = None):
        if httpx is None:
            raise ImportError("RESTAdapter requires pip install 'llmguard-ci[rest]'")
        if not isinstance(url, str):
            raise ValueError("REST URL must be an absolute HTTP(S) URL")
        try:
            endpoint = httpx.URL(url)
        except (httpx.InvalidURL, ValueError):
            raise ValueError("REST URL must be an absolute HTTP(S) URL") from None
        if endpoint.scheme not in {"http", "https"} or not endpoint.host or endpoint.userinfo or endpoint.fragment:
            raise ValueError("REST URL must be absolute HTTP(S) without credentials or fragments")
        if not isinstance(method, str) or method.upper() not in {"GET", "POST"}:
            raise ValueError("REST method must be GET or POST")
        if not nonnegative(timeout) or not 0 < timeout <= 300:
            raise ValueError("REST timeout must be in (0, 300] seconds")
        if type(retries) is not int or not 0 <= retries <= 3:
            raise ValueError("REST retries must be an integer in [0, 3]")
        self._url = endpoint
        self._method = method.upper()
        self._request_mapping = validate_paths({"input": "input"} if request_mapping is None else request_mapping)
        for target, path in self._request_mapping.items():
            if path.split(".")[0] != "input" or not all(target.split(".")):
                raise ValueError("REST request paths must start with 'input' and targets must have nonempty names")
            if self._method == "POST" and any(other.startswith(target + ".") for other in self._request_mapping):
                raise ValueError("Overlapping REST request field targets")
        self._response_mapping = validate_paths(response_mapping, output=True)
        self._retries = retries
        self._timeout = timeout

        if auth_headers is not None and not isinstance(auth_headers, Mapping):
            raise ValueError("REST auth_headers must map header names to environment variable names")
        headers = {"Accept": "application/json"}
        for header, env_name in (auth_headers or {}).items():
            if (not isinstance(header, str) or not _HEADER.fullmatch(header)
                    or not isinstance(env_name, str) or not _ENVIRONMENT_VARIABLE.fullmatch(env_name)):
                raise ValueError("Invalid REST authentication header or environment variable name")
            value = os.environ.get(env_name)
            if not value:
                raise ValueError(f"Missing REST authentication environment variable: {env_name}")
            if not value.isascii() or any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ValueError(f"Invalid REST authentication value in environment variable: {env_name}")
            headers[header] = value
        authenticators = sorted((name.lower(), headers[name]) for name in (auth_headers or {}))
        self._authentication = json.dumps(authenticators).encode() if authenticators else None
        self._client = httpx.Client(headers=headers, timeout=timeout, transport=transport,
                                    follow_redirects=False, trust_env=False)

    @property
    def _state_scope(self):
        if self._authentication is None:
            return None
        return self._sign_state

    def _sign_state(self, secret: bytes) -> str:
        return hmac.digest(secret, self._authentication, "sha256").hex()

    def configuration(self) -> dict:
        """Settings suitable for report metadata, with the endpoint represented by a hash."""
        return {"kind": "rest", "method": self._method,
                "endpoint_sha256": hashlib.sha256(str(self._url).encode()).hexdigest(),
                "request_mapping": dict(self._request_mapping), "response_mapping": dict(self._response_mapping),
                "timeout": self._timeout, "retries": self._retries}

    @property
    def retries(self) -> int:
        return self._retries

    def request_once(self, value):
        if self._method == "GET":
            fields = project({"input": value}, self._request_mapping)
            if fields.keys() != self._request_mapping.keys():
                raise RESTError("request_field_missing", 0)
            payload = {"params": {key: json.dumps(item, ensure_ascii=False, allow_nan=False)
                                  if isinstance(item, (dict, list)) else item for key, item in fields.items()}}
        else:
            payload = {"json": _request_body(self._request_mapping, value)}
        try:
            with private_request_logs():
                response = self._client.request(self._method, self._url, **payload)
        except httpx.TimeoutException:
            raise RESTError("timeout", 1) from None
        except httpx.TransportError:
            raise RESTError("transport_error", 1) from None
        if not response.is_success:
            raise RESTError("http_error", 1, response.status_code, _retry_after(response.headers.get("Retry-After")))
        try:
            result = response.json()
        except ValueError:
            raise RESTError("invalid_json", 1) from None
        return project_output(result, self._response_mapping) if self._response_mapping else result

    def __call__(self, value):
        for attempt in range(1, self._retries + 2):
            try:
                return self.request_once(value)
            except RESTError as exc:
                retryable = exc.code in {"timeout", "transport_error"} or exc.status_code in TRANSIENT_STATUS_CODES
                if not retryable or attempt > self._retries or (exc.retry_after or 0) > 30:
                    if exc.attempts:
                        exc.attempts = attempt
                    raise
                sleep(max(0.1 * 2 ** (attempt - 1), exc.retry_after or 0))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RESTAdapter:
        return self

    def __exit__(self, *_):
        self.close()
