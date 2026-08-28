from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv, set_key
from pydantic import BaseModel, Field

from medrix_flow.config.app_config import AppConfig, reload_app_config
from medrix_flow.setup.security import (
    validate_env_var_name,
    validate_optional_base_url,
    validate_setup_model_provider,
)

_ENV_VAR_PATTERN = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
_VALID_ENV_VAR_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
IMAGE_PROVIDER_GOOGLE = "google-ai-studio"
IMAGE_PROVIDER_OPENAI = "openai-compatible"
IMAGE_PROVIDER_KINDS = {IMAGE_PROVIDER_GOOGLE, IMAGE_PROVIDER_OPENAI}
DEFAULT_GOOGLE_IMAGE_MODEL = "gemini-3-pro-image-preview"
IMAGE_GEN_ACTIVE_PROVIDER_ENV = "IMAGE_GEN_ACTIVE_PROVIDER"
IMAGE_GEN_GOOGLE_MODEL_ENV = "IMAGE_GEN_GOOGLE_MODEL"
IMAGE_GEN_OPENAI_MODEL_ENV = "IMAGE_GEN_OPENAI_MODEL"
IMAGE_GEN_OPENAI_BASE_URL_ENV = "IMAGE_GEN_OPENAI_BASE_URL"
IMAGE_GEN_OPENAI_API_KEY_ENV = "IMAGE_GEN_OPENAI_API_KEY"


class ModelSetupItem(BaseModel):
    name: str = Field(..., description="Unique model identifier")
    provider: str = Field("langchain_openai:ChatOpenAI", description="Provider class path")
    model: str = Field(..., description="Model ID sent to the provider")
    base_url: str | None = Field(None, description="Custom API base URL")
    api_key: str | None = Field(None, description="API key (plain text on write, masked on read)")
    api_key_env_var: str | None = Field(None, description="Env var name that holds the API key")
    max_tokens: int | None = Field(None, description="Max tokens")
    temperature: float | None = Field(None, description="Sampling temperature")
    supports_thinking: bool = Field(False)
    supports_reasoning_effort: bool = Field(False)
    supports_vision: bool = Field(False)


class ToolKeyItem(BaseModel):
    service: str = Field(..., description="Service name: tavily, jina, openalex, semantic-scholar, or google-ai-studio")
    api_key: str | None = Field(None, description="API key (plain on write, masked on read)")
    env_var: str = Field(..., description="Environment variable name")


class ImageProviderConfig(BaseModel):
    provider: Literal["google-ai-studio", "openai-compatible"]
    enabled: bool = Field(False)
    model: str | None = Field(None)
    base_url: str | None = Field(None)
    api_key: str | None = Field(None)
    api_key_env_var: str = Field(...)


class ImageGenerationConfig(BaseModel):
    active_provider: Literal["google-ai-studio", "openai-compatible"] = Field(IMAGE_PROVIDER_GOOGLE)
    google_ai_studio: ImageProviderConfig
    openai_compatible: ImageProviderConfig


class SetupConfigResponse(BaseModel):
    models: list[ModelSetupItem]
    tool_keys: list[ToolKeyItem]
    image_generation: ImageGenerationConfig


class SaveModelsRequest(BaseModel):
    models: list[ModelSetupItem]
    tool_keys: list[ToolKeyItem] | None = None
    image_generation: ImageGenerationConfig | None = None


def find_config_path() -> Path:
    return AppConfig.resolve_config_path()


def find_env_path() -> Path:
    return find_config_path().parent / ".env"


def read_raw_config() -> dict[str, Any]:
    config_path = find_config_path()
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_raw_config(data: dict[str, Any]) -> None:
    config_path = find_config_path()
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(data, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)


def refresh_env() -> None:
    load_dotenv(find_env_path(), override=True)


def get_env_value(var_name: str) -> str | None:
    return os.getenv(var_name)


def get_non_empty_env_value(var_name: str) -> str | None:
    value = get_env_value(var_name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def set_env_value(var_name: str, value: str) -> None:
    env_path = find_env_path()
    if not env_path.exists():
        env_path.touch()
    set_key(str(env_path), var_name, value)
    os.environ[var_name] = value


def set_optional_env_value(var_name: str, value: str | None) -> None:
    set_env_value(var_name, value or "")


def get_google_ai_studio_key() -> str:
    return get_non_empty_env_value("GEMINI_API_KEY") or get_non_empty_env_value("GOOGLE_API_KEY") or ""


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    return base_url.strip().rstrip("/") or None


def normalize_model_env_var_name(name: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").upper())
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if not candidate:
        candidate = "MODEL_API_KEY"
    if not candidate[0].isalpha():
        candidate = f"MODEL_{candidate}"
    if not candidate.endswith("_API_KEY"):
        candidate = f"{candidate}_API_KEY"
    return candidate


def resolve_model_env_var_name(raw_name: str | None) -> str | None:
    if not raw_name:
        return None
    if _VALID_ENV_VAR_PATTERN.fullmatch(raw_name):
        return raw_name
    return normalize_model_env_var_name(raw_name)


def get_model_api_key(raw_env_var: str | None) -> tuple[str | None, str]:
    if not raw_env_var:
        return None, ""
    normalized_env_var = resolve_model_env_var_name(raw_env_var)
    actual_key = get_non_empty_env_value(raw_env_var) or ""
    if not actual_key and normalized_env_var and normalized_env_var != raw_env_var:
        actual_key = get_non_empty_env_value(normalized_env_var) or ""
    return normalized_env_var, actual_key


def resolve_image_provider_kind(provider: str | None) -> str:
    if provider in IMAGE_PROVIDER_KINDS:
        return provider
    return IMAGE_PROVIDER_GOOGLE


def default_image_provider_config(provider: Literal["google-ai-studio", "openai-compatible"]) -> ImageProviderConfig:
    api_key_env_var = "GEMINI_API_KEY" if provider == IMAGE_PROVIDER_GOOGLE else IMAGE_GEN_OPENAI_API_KEY_ENV
    model = DEFAULT_GOOGLE_IMAGE_MODEL if provider == IMAGE_PROVIDER_GOOGLE else None
    return ImageProviderConfig(
        provider=provider,
        enabled=False,
        model=model,
        base_url=None,
        api_key=None,
        api_key_env_var=api_key_env_var,
    )


def build_image_generation_config(raw: dict[str, Any]) -> ImageGenerationConfig:
    raw_image_generation = raw.get("image_generation") or {}
    raw_google = raw_image_generation.get("google_ai_studio") or {}
    raw_openai = raw_image_generation.get("openai_compatible") or {}
    active_provider = resolve_image_provider_kind(raw_image_generation.get("active_provider") or get_non_empty_env_value(IMAGE_GEN_ACTIVE_PROVIDER_ENV))
    google_model = raw_google.get("model") or get_non_empty_env_value(IMAGE_GEN_GOOGLE_MODEL_ENV) or DEFAULT_GOOGLE_IMAGE_MODEL
    openai_model = raw_openai.get("model") or get_non_empty_env_value(IMAGE_GEN_OPENAI_MODEL_ENV)
    openai_base_url = normalize_base_url(raw_openai.get("base_url") or get_non_empty_env_value(IMAGE_GEN_OPENAI_BASE_URL_ENV))

    return ImageGenerationConfig(
        active_provider=active_provider,
        google_ai_studio=ImageProviderConfig(
            provider=IMAGE_PROVIDER_GOOGLE,
            enabled=active_provider == IMAGE_PROVIDER_GOOGLE,
            model=google_model,
            api_key=get_google_ai_studio_key(),
            api_key_env_var="GEMINI_API_KEY",
        ),
        openai_compatible=ImageProviderConfig(
            provider=IMAGE_PROVIDER_OPENAI,
            enabled=active_provider == IMAGE_PROVIDER_OPENAI,
            model=openai_model,
            base_url=openai_base_url,
            api_key=get_non_empty_env_value(IMAGE_GEN_OPENAI_API_KEY_ENV) or "",
            api_key_env_var=IMAGE_GEN_OPENAI_API_KEY_ENV,
        ),
    )


def apply_legacy_google_image_key(
    image_generation: ImageGenerationConfig,
    tool_keys: list[ToolKeyItem] | None,
) -> ImageGenerationConfig:
    if not tool_keys:
        return image_generation
    for tool_key in tool_keys:
        if tool_key.service == IMAGE_PROVIDER_GOOGLE and tool_key.api_key and tool_key.api_key.strip():
            image_generation.google_ai_studio.api_key = tool_key.api_key.strip()
            break
    return image_generation


def validate_image_generation_config(
    image_generation: ImageGenerationConfig,
    *,
    require_active_fields: bool = True,
) -> ImageGenerationConfig:
    active_provider = resolve_image_provider_kind(image_generation.active_provider)
    image_generation.active_provider = active_provider

    image_generation.google_ai_studio.provider = IMAGE_PROVIDER_GOOGLE
    image_generation.google_ai_studio.enabled = active_provider == IMAGE_PROVIDER_GOOGLE
    image_generation.google_ai_studio.api_key_env_var = "GEMINI_API_KEY"
    image_generation.google_ai_studio.base_url = None
    image_generation.google_ai_studio.model = (image_generation.google_ai_studio.model or "").strip() or DEFAULT_GOOGLE_IMAGE_MODEL
    image_generation.google_ai_studio.api_key = (image_generation.google_ai_studio.api_key or "").strip() or None

    image_generation.openai_compatible.provider = IMAGE_PROVIDER_OPENAI
    image_generation.openai_compatible.enabled = active_provider == IMAGE_PROVIDER_OPENAI
    image_generation.openai_compatible.api_key_env_var = IMAGE_GEN_OPENAI_API_KEY_ENV
    image_generation.openai_compatible.model = (image_generation.openai_compatible.model or "").strip() or None
    image_generation.openai_compatible.api_key = (image_generation.openai_compatible.api_key or "").strip() or None
    if image_generation.openai_compatible.base_url:
        validate_optional_base_url(image_generation.openai_compatible.base_url)
    image_generation.openai_compatible.base_url = normalize_base_url(image_generation.openai_compatible.base_url)

    if not require_active_fields:
        return image_generation

    if active_provider == IMAGE_PROVIDER_GOOGLE:
        if not image_generation.google_ai_studio.model:
            raise ValueError("Active image provider 'google-ai-studio' requires a model.")
        if not image_generation.google_ai_studio.api_key:
            raise ValueError("Active image provider 'google-ai-studio' requires an API key.")
    elif active_provider == IMAGE_PROVIDER_OPENAI:
        if not image_generation.openai_compatible.model:
            raise ValueError("Active image provider 'openai-compatible' requires a model.")
        if not image_generation.openai_compatible.base_url:
            raise ValueError("Active image provider 'openai-compatible' requires a base URL.")
        if not image_generation.openai_compatible.api_key:
            raise ValueError("Active image provider 'openai-compatible' requires an API key.")

    return image_generation


def get_setup_config_data() -> SetupConfigResponse:
    refresh_env()
    raw = read_raw_config()
    models_raw: list[dict[str, Any]] = raw.get("models") or []

    models: list[ModelSetupItem] = []
    for item in models_raw:
        api_key_raw = item.get("api_key", "")
        env_var: str | None = None
        actual_key = ""

        if isinstance(api_key_raw, str) and api_key_raw.startswith("$"):
            env_var, actual_key = get_model_api_key(api_key_raw[1:])
        elif api_key_raw:
            actual_key = str(api_key_raw)

        models.append(
            ModelSetupItem(
                name=item.get("name", ""),
                provider=item.get("use", "langchain_openai:ChatOpenAI"),
                model=item.get("model", ""),
                base_url=item.get("base_url"),
                api_key=actual_key,
                api_key_env_var=env_var,
                max_tokens=item.get("max_tokens"),
                temperature=item.get("temperature"),
                supports_thinking=item.get("supports_thinking", True),
                supports_reasoning_effort=item.get("supports_reasoning_effort", False),
                supports_vision=item.get("supports_vision", True),
            )
        )

    tool_keys = [
        ToolKeyItem(service="tavily", api_key=get_env_value("TAVILY_API_KEY") or "", env_var="TAVILY_API_KEY"),
        ToolKeyItem(service="jina", api_key=get_env_value("JINA_API_KEY") or "", env_var="JINA_API_KEY"),
        ToolKeyItem(service="openalex", api_key=get_env_value("OPENALEX_API_KEY") or "", env_var="OPENALEX_API_KEY"),
        ToolKeyItem(
            service="semantic-scholar",
            api_key=get_env_value("SEMANTIC_SCHOLAR_API_KEY") or "",
            env_var="SEMANTIC_SCHOLAR_API_KEY",
        ),
    ]

    return SetupConfigResponse(
        models=models,
        tool_keys=tool_keys,
        image_generation=build_image_generation_config(raw),
    )


def save_setup_config_data(payload: SaveModelsRequest) -> None:
    raw = read_raw_config()
    models: list[dict[str, Any]] = []

    for model in payload.models:
        validate_setup_model_provider(model.provider)
        validate_optional_base_url(model.base_url)
        env_var = resolve_model_env_var_name(model.api_key_env_var) or normalize_model_env_var_name(model.name or model.model)
        validate_env_var_name(env_var)

        if model.api_key and model.api_key.strip():
            set_env_value(env_var, model.api_key.strip())

        entry: dict[str, Any] = {
            "name": model.name or model.model,
            "display_name": model.model,
            "use": model.provider,
            "model": model.model,
            "api_key": f"${env_var}",
            "supports_thinking": model.supports_thinking,
            "supports_reasoning_effort": model.supports_reasoning_effort,
            "supports_vision": model.supports_vision,
        }
        if model.base_url:
            entry["base_url"] = model.base_url
        if model.max_tokens is not None:
            entry["max_tokens"] = model.max_tokens
        if model.temperature is not None:
            entry["temperature"] = model.temperature
        models.append(entry)

    raw["models"] = models

    require_active_image_provider_fields = payload.image_generation is not None
    image_generation = payload.image_generation or build_image_generation_config(raw)
    image_generation = apply_legacy_google_image_key(image_generation, payload.tool_keys)
    image_generation = validate_image_generation_config(
        image_generation,
        require_active_fields=require_active_image_provider_fields,
    )
    raw["image_generation"] = {
        "active_provider": image_generation.active_provider,
        "google_ai_studio": {
            "model": image_generation.google_ai_studio.model,
        },
        "openai_compatible": {
            "model": image_generation.openai_compatible.model,
            "base_url": image_generation.openai_compatible.base_url,
        },
    }

    set_env_value(IMAGE_GEN_ACTIVE_PROVIDER_ENV, image_generation.active_provider)
    set_env_value(IMAGE_GEN_GOOGLE_MODEL_ENV, image_generation.google_ai_studio.model or DEFAULT_GOOGLE_IMAGE_MODEL)
    set_optional_env_value(IMAGE_GEN_OPENAI_MODEL_ENV, image_generation.openai_compatible.model)
    set_optional_env_value(IMAGE_GEN_OPENAI_BASE_URL_ENV, image_generation.openai_compatible.base_url)
    if image_generation.google_ai_studio.api_key:
        set_env_value("GEMINI_API_KEY", image_generation.google_ai_studio.api_key)
        set_env_value("GOOGLE_API_KEY", image_generation.google_ai_studio.api_key)
    if image_generation.openai_compatible.api_key:
        set_env_value(IMAGE_GEN_OPENAI_API_KEY_ENV, image_generation.openai_compatible.api_key)

    if payload.tool_keys:
        for tool_key in payload.tool_keys:
            validate_env_var_name(tool_key.env_var, allow_tool_key=True)
            if tool_key.api_key and tool_key.api_key.strip():
                value = tool_key.api_key.strip()
                if tool_key.service == "google-ai-studio":
                    set_env_value("GEMINI_API_KEY", value)
                    set_env_value("GOOGLE_API_KEY", value)
                else:
                    set_env_value(tool_key.env_var, value)

    write_raw_config(raw)
    reload_app_config()


def collect_referenced_env_vars(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, str):
        match = _ENV_VAR_PATTERN.match(node.strip())
        if match:
            found.add(match.group(1))
    elif isinstance(node, dict):
        for value in node.values():
            found.update(collect_referenced_env_vars(value))
    elif isinstance(node, list):
        for item in node:
            found.update(collect_referenced_env_vars(item))
    return found


def ensure_setup_files(project_root: Path) -> list[Path]:
    created: list[Path] = []
    pairs = [
        (project_root / "config.example.yaml", project_root / "config.yaml"),
        (project_root / ".env.example", project_root / ".env"),
        (project_root / "frontend" / ".env.example", project_root / "frontend" / ".env"),
        (project_root / "extensions_config.example.json", project_root / "extensions_config.json"),
    ]
    for source, target in pairs:
        if target.exists():
            continue
        if not source.exists():
            raise FileNotFoundError(f"Missing template file: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        created.append(target)
    return created
