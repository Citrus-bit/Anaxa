from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

GOOGLE_IMAGE_SMOKE_MODEL = "gemini-2.5-flash-image"
GOOGLE_IMAGE_SMOKE_PROMPT = "Generate a simple blue scientific icon on a white background."
GOOGLE_IMAGE_SMOKE_IMAGE_SIZE = "1K"
GOOGLE_IMAGE_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
GOOGLE_IMAGE_RETRY_DELAYS_SECONDS = (1.0, 3.0)
GOOGLE_IMAGE_MAX_ATTEMPTS = 3
# Setup validation is allowed to use most of the 3 minute proxy budget. The
# total wall-clock time remains under that ceiling even if the first attempt
# times out and the retry runs.
GOOGLE_SETTINGS_VALIDATION_TIMEOUT_SECONDS = 85
GOOGLE_SETTINGS_VALIDATION_RETRY_DELAYS_SECONDS = (1.0,)
GOOGLE_SETTINGS_VALIDATION_MAX_ATTEMPTS = 2


@dataclass(slots=True)
class GoogleImageRequestResult:
    payload: dict[str, Any]
    retry_count: int


class GoogleImageRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_count: int = 0,
        payload: dict[str, Any] | None = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_count = retry_count
        self.payload = payload or {}
        self.transient = transient


def build_google_generation_request(
    *,
    prompt_text: str,
    inline_parts: list[dict],
    aspect_ratio: str,
    model: str,
    image_size: str,
    force_image_size: bool = False,
) -> tuple[str, dict[str, Any]]:
    image_config: dict[str, str] = {"aspectRatio": aspect_ratio}
    if image_size and (force_image_size or model != GOOGLE_IMAGE_SMOKE_MODEL):
        image_config["imageSize"] = image_size
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "generationConfig": {"imageConfig": image_config},
            "contents": [{"parts": [*inline_parts, {"text": prompt_text}]}],
        },
    )


def build_google_image_smoke_request(*, model: str) -> dict[str, Any]:
    _, payload = build_google_generation_request(
        prompt_text=GOOGLE_IMAGE_SMOKE_PROMPT,
        inline_parts=[],
        aspect_ratio="1:1",
        model=model,
        image_size=GOOGLE_IMAGE_SMOKE_IMAGE_SIZE,
        force_image_size=True,
    )
    return payload


def has_google_image_content(payload: dict[str, Any]) -> bool:
    candidates = payload.get("candidates") or []
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") if candidates else []
    return any(
        isinstance(part, dict) and (("inlineData" in part and isinstance(part["inlineData"], dict) and part["inlineData"].get("data")) or ("inline_data" in part and isinstance(part["inline_data"], dict) and part["inline_data"].get("data")))
        for part in (parts or [])
    )


def summarize_google_image_response(payload: dict[str, Any]) -> str:
    prompt_feedback = payload.get("promptFeedback") or payload.get("prompt_feedback") or {}
    block_reason = prompt_feedback.get("blockReason") or prompt_feedback.get("block_reason")
    candidates = payload.get("candidates") or []
    first_candidate = candidates[0] if candidates else {}
    finish_reason = first_candidate.get("finishReason") or first_candidate.get("finish_reason")
    parts = ((first_candidate or {}).get("content") or {}).get("parts") or []
    text_parts = [part.get("text", "").strip() for part in parts if isinstance(part, dict) and part.get("text")]

    details: list[str] = []
    if block_reason:
        details.append(f"block_reason={block_reason}")
    if finish_reason:
        details.append(f"finish_reason={finish_reason}")
    if text_parts:
        details.append(f"text={text_parts[0][:160]}")
    if not details:
        return "No additional response details were provided."
    return "; ".join(details)


def extract_google_image_part(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Image generation returned no candidates")
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    for part in parts:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict) and inline.get("data"):
            return inline
    raise RuntimeError("Image generation returned no image data")


def extract_google_error_message(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def format_google_status_message(status_code: int, payload: dict[str, Any] | None = None) -> str:
    detail = extract_google_error_message(payload or {})
    if status_code == 400:
        return _append_google_detail("Google AI Studio returned status 400.", detail)
    if status_code in {401, 403}:
        prefix = f"Google AI Studio returned status {status_code}. The API key is invalid or does not have permission for image generation."
        return _append_google_detail(prefix, detail)
    if status_code == 429:
        prefix = "Google AI Studio returned status 429. The image generation service is rate-limiting this request; this is not an API key configuration error. Retry later or choose another image model."
        return _append_google_detail(prefix, detail)
    if status_code == 503:
        prefix = "Google AI Studio returned status 503. The configured image model is temporarily unavailable or overloaded; this is not an API key configuration error. Retry later or choose another image model."
        return _append_google_detail(prefix, detail)
    if status_code in {500, 502, 503, 504}:
        prefix = f"Google AI Studio returned status {status_code}. The upstream image service is temporarily unavailable or overloaded; this is not an API key configuration error. Retry later or choose another image model."
        return _append_google_detail(prefix, detail)
    return _append_google_detail(f"Google AI Studio returned status {status_code}.", detail)


def format_google_timeout_message(timeout_seconds: int) -> str:
    return f"Google AI Studio request timed out after {timeout_seconds} seconds. The upstream image service may be overloaded; please retry later."


def execute_google_image_request(
    *,
    requests_module: Any,
    api_key: str,
    model: str,
    prompt_text: str,
    inline_parts: list[dict],
    aspect_ratio: str,
    image_size: str,
    timeout_seconds: int,
    force_image_size: bool = False,
    max_attempts: int = GOOGLE_IMAGE_MAX_ATTEMPTS,
    retry_delays_seconds: tuple[float, ...] = GOOGLE_IMAGE_RETRY_DELAYS_SECONDS,
    sleep_fn: Any = time.sleep,
) -> GoogleImageRequestResult:
    url, request_payload = build_google_generation_request(
        prompt_text=prompt_text,
        inline_parts=inline_parts,
        aspect_ratio=aspect_ratio,
        model=model,
        image_size=image_size,
        force_image_size=force_image_size,
    )

    retry_count = 0
    for attempt in range(max_attempts):
        try:
            response = requests_module.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            if attempt < max_attempts - 1:
                sleep_fn(retry_delays_seconds[min(attempt, len(retry_delays_seconds) - 1)])
                retry_count += 1
                continue
            raise GoogleImageRequestError(
                format_google_timeout_message(timeout_seconds),
                retry_count=retry_count,
                transient=True,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise GoogleImageRequestError(
                f"Google AI Studio request failed before a response was received: {exc}",
                retry_count=retry_count,
            ) from exc

        if response.status_code == 200:
            return GoogleImageRequestResult(
                payload=_safe_response_json(response),
                retry_count=retry_count,
            )

        error_payload = _safe_response_json(response)
        if response.status_code in GOOGLE_IMAGE_TRANSIENT_STATUS_CODES and attempt < max_attempts - 1:
            sleep_fn(retry_delays_seconds[min(attempt, len(retry_delays_seconds) - 1)])
            retry_count += 1
            continue

        raise GoogleImageRequestError(
            format_google_status_message(response.status_code, error_payload),
            status_code=response.status_code,
            retry_count=retry_count,
            payload=error_payload,
            transient=response.status_code in GOOGLE_IMAGE_TRANSIENT_STATUS_CODES,
        )

    raise GoogleImageRequestError("Google AI Studio request failed without a response payload.")


def execute_google_settings_validation_request(
    *,
    requests_module: Any,
    api_key: str,
    model: str,
    sleep_fn: Any = time.sleep,
) -> GoogleImageRequestResult:
    return execute_google_image_request(
        requests_module=requests_module,
        api_key=api_key,
        model=model,
        prompt_text=GOOGLE_IMAGE_SMOKE_PROMPT,
        inline_parts=[],
        aspect_ratio="1:1",
        image_size=GOOGLE_IMAGE_SMOKE_IMAGE_SIZE,
        timeout_seconds=GOOGLE_SETTINGS_VALIDATION_TIMEOUT_SECONDS,
        force_image_size=True,
        max_attempts=GOOGLE_SETTINGS_VALIDATION_MAX_ATTEMPTS,
        retry_delays_seconds=GOOGLE_SETTINGS_VALIDATION_RETRY_DELAYS_SECONDS,
        sleep_fn=sleep_fn,
    )


def _append_google_detail(prefix: str, detail: str | None) -> str:
    if not detail:
        return prefix
    return f"{prefix} {detail}"


def _safe_response_json(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
