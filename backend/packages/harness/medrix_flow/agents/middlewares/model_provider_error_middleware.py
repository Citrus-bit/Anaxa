"""Retry transient model provider failures during agent model calls."""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, override

import httpx
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}
_BASE_RETRY_DELAY_SECONDS = 2.0
_MAX_RETRY_DELAY_SECONDS = 300.0
_MAX_RETRY_AFTER_DELAY_SECONDS = 600.0
_RETRY_SLEEP_CHUNK_SECONDS = 15.0
_TRANSIENT_PATTERNS = (
    "system_cpu_overloaded",
    "system cpu overloaded",
    "cpu overloaded",
    "overloaded",
    "service unavailable",
    "temporarily unavailable",
    "temporarily overloaded",
    "rate limit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection error",
    "concurrency limit exceeded",
    "please retry later",
    "model unavailable",
    "model is unavailable",
    "provider unavailable",
)


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    if retry_after is None:
        return None

    retry_after_text = str(retry_after).strip()
    if not retry_after_text:
        return None

    try:
        return max(0.0, float(retry_after_text))
    except ValueError:
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(retry_after_text)
    except (TypeError, ValueError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _retry_delay_seconds(attempt: int, exc: BaseException) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, _MAX_RETRY_AFTER_DELAY_SECONDS)

    backoff = _BASE_RETRY_DELAY_SECONDS * (2 ** max(0, attempt - 1))
    return min(backoff, _MAX_RETRY_DELAY_SECONDS)


def _retry_at_iso(delay_seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delay_seconds)).isoformat()


def _provider_error_code(exc: BaseException) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str) and code:
                return code
        code = body.get("code")
        if isinstance(code, str) and code:
            return code
    return None


def _error_text(exc: BaseException) -> str:
    parts = [exc.__class__.__name__, str(exc)]
    status = _status_code(exc)
    if status is not None:
        parts.append(str(status))
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(_safe_json(body))
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text:
        parts.append(response_text[:2000])
    return "\n".join(parts).lower()


def is_transient_model_provider_error(exc: BaseException) -> bool:
    """Return true for model-provider failures that should be retried."""

    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True

    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
        return True

    if isinstance(exc, (InternalServerError, APIStatusError)):
        status = _status_code(exc)
        if status in _TRANSIENT_STATUS_CODES:
            return True
        text = _error_text(exc)
        return any(pattern in text for pattern in _TRANSIENT_PATTERNS)

    if isinstance(exc, APIError):
        text = _error_text(exc)
        return any(pattern in text for pattern in _TRANSIENT_PATTERNS)

    status = _status_code(exc)
    if status in _TRANSIENT_STATUS_CODES:
        return True

    return False


class ModelProviderErrorMiddleware(AgentMiddleware[AgentState]):
    """Retry transient provider failures until the model call succeeds or is stopped."""

    def _log_retry(self, *, attempt: int, exc: BaseException, delay_seconds: float) -> None:
        status = _status_code(exc)
        logger.warning(
            "Transient model provider error; retrying model call: attempt=%s class=%s status=%s code=%s delay=%.1fs",
            attempt,
            exc.__class__.__name__,
            status,
            _provider_error_code(exc),
            delay_seconds,
        )

    def _emit_stream_event(self, event: dict[str, Any], *, failure_message: str) -> None:
        try:
            writer = get_stream_writer()
        except Exception:
            logger.debug("Model retry stream writer is unavailable", exc_info=True)
            return

        try:
            writer(event)
        except Exception:
            logger.debug(failure_message, exc_info=True)

    def _emit_retry_event(self, *, attempt: int, exc: BaseException, delay_seconds: float) -> None:
        status = _status_code(exc)
        self._emit_stream_event(
            {
                "type": "model_retry",
                "attempt": attempt,
                "delay_seconds": delay_seconds,
                "error_class": exc.__class__.__name__,
                "status_code": status,
                "provider_code": _provider_error_code(exc),
                "retry_at": _retry_at_iso(delay_seconds),
            },
            failure_message="Failed to emit model retry stream event",
        )

    def _emit_retry_heartbeat_event(self, *, attempt: int, exc: BaseException, remaining_seconds: float) -> None:
        self._emit_stream_event(
            {
                "type": "model_retry_heartbeat",
                "attempt": attempt,
                "remaining_seconds": remaining_seconds,
                "error_class": exc.__class__.__name__,
                "status_code": _status_code(exc),
                "provider_code": _provider_error_code(exc),
            },
            failure_message="Failed to emit model retry heartbeat stream event",
        )

    def _sleep_with_heartbeats(self, *, attempt: int, exc: BaseException, delay_seconds: float) -> None:
        remaining = delay_seconds
        while remaining > 0:
            chunk = min(remaining, _RETRY_SLEEP_CHUNK_SECONDS)
            time.sleep(chunk)
            remaining = max(0.0, remaining - chunk)
            if remaining > 0:
                self._emit_retry_heartbeat_event(attempt=attempt, exc=exc, remaining_seconds=remaining)

    async def _asleep_with_heartbeats(self, *, attempt: int, exc: BaseException, delay_seconds: float) -> None:
        remaining = delay_seconds
        while remaining > 0:
            chunk = min(remaining, _RETRY_SLEEP_CHUNK_SECONDS)
            await asyncio.sleep(chunk)
            remaining = max(0.0, remaining - chunk)
            if remaining > 0:
                self._emit_retry_heartbeat_event(attempt=attempt, exc=exc, remaining_seconds=remaining)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        attempt = 1
        while True:
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if not is_transient_model_provider_error(exc):
                    raise
                delay_seconds = _retry_delay_seconds(attempt, exc)
                self._log_retry(attempt=attempt, exc=exc, delay_seconds=delay_seconds)
                self._emit_retry_event(attempt=attempt, exc=exc, delay_seconds=delay_seconds)
                self._sleep_with_heartbeats(attempt=attempt, exc=exc, delay_seconds=delay_seconds)
                attempt += 1

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        attempt = 1
        while True:
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if not is_transient_model_provider_error(exc):
                    raise
                delay_seconds = _retry_delay_seconds(attempt, exc)
                self._log_retry(attempt=attempt, exc=exc, delay_seconds=delay_seconds)
                self._emit_retry_event(attempt=attempt, exc=exc, delay_seconds=delay_seconds)
                await self._asleep_with_heartbeats(attempt=attempt, exc=exc, delay_seconds=delay_seconds)
                attempt += 1
