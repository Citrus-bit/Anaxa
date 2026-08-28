"""Validation helpers for the setup gateway endpoints."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from medrix_flow.config import get_app_config

_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_TOOL_KEY_ENV_VARS = {
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "INFOQUEST_API_KEY",
    "JINA_API_KEY",
    "OPENALEX_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "TAVILY_API_KEY",
}
_DEFAULT_ALLOWED_MODEL_PROVIDERS = {
    "langchain_anthropic:ChatAnthropic",
    "langchain_google_genai:ChatGoogleGenerativeAI",
    "langchain_openai:ChatOpenAI",
    "medrix_flow.models.claude_provider:ClaudeChatModel",
    "medrix_flow.models.openai_codex_provider:CodexChatModel",
    "medrix_flow.models.patched_deepseek:PatchedChatDeepSeek",
    "medrix_flow.models.patched_minimax:PatchedChatMiniMax",
}


def get_allowed_setup_model_providers() -> set[str]:
    """Return setup-provider paths allowed by the current app config."""

    allowed = set(_DEFAULT_ALLOWED_MODEL_PROVIDERS)
    try:
        allowed.update(model.use for model in get_app_config().models if model.use)
    except Exception:
        pass
    return allowed


def validate_setup_model_provider(provider: str) -> None:
    if provider not in get_allowed_setup_model_providers():
        raise ValueError(f"Unsupported model provider '{provider}'. Only built-in providers and providers already present in config.yaml are allowed.")


def validate_optional_base_url(base_url: str | None) -> None:
    if not base_url:
        return

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Base URL must be an absolute http(s) URL.")


def validate_env_var_name(name: str, *, allow_tool_key: bool = False) -> None:
    if not _ENV_VAR_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid environment variable name '{name}'.")
    if allow_tool_key and name not in _ALLOWED_TOOL_KEY_ENV_VARS:
        raise ValueError(f"Unsupported tool API key env var '{name}'.")
