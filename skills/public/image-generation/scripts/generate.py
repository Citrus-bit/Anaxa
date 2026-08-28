import base64
import json
import mimetypes
import os
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import requests

try:
    from PIL import Image
except ImportError:  # pragma: no cover - only exercised in minimal skill runners
    Image = None  # type: ignore[assignment]

try:
    from medrix_flow.utils import google_image as google_image_utils
except ModuleNotFoundError:
    try:
        from deerflow.utils import google_image as google_image_utils
    except ModuleNotFoundError:
        harness_path = Path(__file__).resolve().parents[4] / "backend" / "packages" / "harness"
        if harness_path.exists():
            sys.path.insert(0, str(harness_path))
        try:
            from medrix_flow.utils import google_image as google_image_utils
        except ModuleNotFoundError:
            from deerflow.utils import google_image as google_image_utils

IMAGE_PROVIDER_GOOGLE = "google-ai-studio"
IMAGE_PROVIDER_OPENAI = "openai-compatible"
IMAGE_PROVIDER_MINIMAX = "minimax"
IMAGE_GENERATION_PROVIDER_ENV = "IMAGE_GENERATION_PROVIDER"
MINIMAX_DEFAULT_HOST = "https://api.minimaxi.com"
# MiniMax image-01 accepts a single prompt string with a 1500-character limit.
MINIMAX_PROMPT_MAX_CHARS = 1500
MINIMAX_IMAGE_MODEL_ENV = "MINIMAX_IMAGE_MODEL"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"
MINIMAX_API_HOST_ENV = "MINIMAX_API_HOST"
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
DEFAULT_DRAFT_MODEL = google_image_utils.GOOGLE_IMAGE_SMOKE_MODEL
DEFAULT_IMAGE_SIZE = "2K"
SCIENTIFIC_IMAGE_SIZE = "4K"
DEFAULT_OUTPUT_MIME_TYPE = "image/jpeg"
SCIENTIFIC_OUTPUT_MIME_TYPE = "image/png"
SUPPORTED_IMAGE_SIZES = {"1K", "2K", "4K"}
SUPPORTED_OUTPUT_MIME_TYPES = {"image/jpeg", "image/png"}
SUPPORTED_OPENAI_COMPATIBLE_ASPECT_RATIOS = {"1:1", "16:9", "9:16"}
OPENAI_COMPATIBLE_SIZE_MAP = {
    "1:1": {"1K": "1024x1024", "2K": "2048x2048", "4K": "4096x4096"},
    "16:9": {"1K": "1024x576", "2K": "2048x1152", "4K": "4096x2304"},
    "9:16": {"1K": "576x1024", "2K": "1152x2048", "4K": "2304x4096"},
}
IMAGE_GEN_ACTIVE_PROVIDER_ENV = "IMAGE_GEN_ACTIVE_PROVIDER"
IMAGE_GEN_GOOGLE_MODEL_ENV = "IMAGE_GEN_GOOGLE_MODEL"
IMAGE_GEN_OPENAI_MODEL_ENV = "IMAGE_GEN_OPENAI_MODEL"
IMAGE_GEN_OPENAI_BASE_URL_ENV = "IMAGE_GEN_OPENAI_BASE_URL"
IMAGE_GEN_OPENAI_API_KEY_ENV = "IMAGE_GEN_OPENAI_API_KEY"
OPENAI_COMPATIBLE_IMAGE_GENERATIONS_PATH = "/images/generations"
OPENAI_COMPATIBLE_BASE_URL_HINT = (
    "Use the API root path such as https://provider.example.com/v1; "
    "MedrixFlow appends /images/generations."
)
SCIENTIFIC_GUARDRAIL = (
    "This is a scientific illustration request, not a quantitative data figure. "
    "Do not fabricate plots, axes, p-values, ROC curves, heatmaps, volcano plots, or other measured results. "
    "Generate only a conceptual, explanatory, or graphical-abstract style image consistent with scientific communication."
)

_PROVIDER_ALIASES = {
    "gemini": IMAGE_PROVIDER_GOOGLE,
    "google": IMAGE_PROVIDER_GOOGLE,
    "google-ai-studio": IMAGE_PROVIDER_GOOGLE,
    "openai-compatible": IMAGE_PROVIDER_OPENAI,
    "minimax": IMAGE_PROVIDER_MINIMAX,
}


def validate_image(image_path: str) -> bool:
    """
    Validate if an image file can be opened and is not corrupted.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if the image is valid and can be opened, False otherwise
    """
    if Image is None:
        print("Warning: Pillow is not installed; reference image validation is unavailable")
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it's a valid image
        # Re-open to check if it can be fully loaded (verify() may not catch all issues)
        with Image.open(image_path) as img:
            img.load()  # Force load the image data
        return True
    except Exception as e:
        print(f"Warning: Image '{image_path}' is invalid or corrupted: {e}")
        return False


def get_non_empty_env_value(var_name: str) -> str | None:
    value = os.getenv(var_name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _canonical_provider(provider: str | None) -> str:
    """Normalize provider names used by the settings UI and skill scripts."""
    candidate = (provider or "").strip().lower().replace("_", "-")
    resolved = _PROVIDER_ALIASES.get(candidate)
    if resolved is None:
        raise ValueError(
            f"Unsupported provider: {provider}. Use google-ai-studio, openai-compatible, or minimax."
        )
    return resolved


def _resolve_provider(override_env: str, existing_provider: str, has_existing_creds: bool) -> str:
    """Resolve a DeerFlow-style provider override.

    The helper is intentionally kept public-ish because the generation skill
    tests and other skills use it directly.
    """
    override = os.getenv(override_env)
    if override:
        return override.strip().lower()
    if has_existing_creds:
        return existing_provider
    if os.getenv(MINIMAX_API_KEY_ENV):
        return IMAGE_PROVIDER_MINIMAX
    raise ValueError(
        f"No credentials found. Set GEMINI_API_KEY for {existing_provider}, "
        f"or MINIMAX_API_KEY for minimax (optionally force with {override_env})."
    )


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    return base_url.strip().rstrip("/") or None


def validate_openai_compatible_base_url(base_url: str) -> None:
    if base_url.rstrip("/").endswith(OPENAI_COMPATIBLE_IMAGE_GENERATIONS_PATH):
        raise RuntimeError(
            "OpenAI-compatible image provider base URL must be the API root path, not the full "
            f"{OPENAI_COMPATIBLE_IMAGE_GENERATIONS_PATH} endpoint. {OPENAI_COMPATIBLE_BASE_URL_HINT}"
        )


def openai_compatible_generations_url(base_url: str) -> str:
    return f"{base_url}{OPENAI_COMPATIBLE_IMAGE_GENERATIONS_PATH}"


def trim_for_message(value: str, limit: int = 500) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def response_content_type(response: requests.Response) -> str:
    headers = getattr(response, "headers", {}) or {}
    if hasattr(headers, "get"):
        content_type = headers.get("Content-Type") or headers.get("content-type")
        if content_type:
            return str(content_type)
    return "unknown"


def response_preview(response: requests.Response) -> str:
    text = getattr(response, "text", None)
    if text is None:
        content = getattr(response, "content", b"")
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        elif content:
            text = str(content)
    if not text:
        return "<empty>"
    return trim_for_message(text)


def json_preview(payload: object) -> str:
    try:
        return trim_for_message(json.dumps(payload, ensure_ascii=False))
    except TypeError:
        return trim_for_message(str(payload))


def resolve_google_ai_studio_api_key() -> str | None:
    return get_non_empty_env_value("GEMINI_API_KEY") or get_non_empty_env_value("GOOGLE_API_KEY")


def resolve_openai_compatible_api_key() -> str | None:
    return get_non_empty_env_value(IMAGE_GEN_OPENAI_API_KEY_ENV)


def resolve_provider_name(provider: str | None = None) -> str:
    """Resolve the active image provider across Anaxa and DeerFlow settings.

    Explicit function arguments win.  The legacy Anaxa setting is honored,
    followed by DeerFlow's ``IMAGE_GENERATION_PROVIDER`` override.  With no
    setting, Google credentials win and MiniMax is used as the automatic
    fallback when no Google credential is present.
    """
    if provider:
        return _canonical_provider(provider)

    # DeerFlow's override is checked first when present.  This lets a caller
    # opt into MiniMax even when an older Anaxa checkout left a Google setting
    # in the environment.
    configured = get_non_empty_env_value(IMAGE_GENERATION_PROVIDER_ENV)
    if configured is not None:
        return _canonical_provider(configured)

    # Anaxa's setup UI writes IMAGE_GEN_ACTIVE_PROVIDER. Persisted settings
    # remain authoritative even when credentials for multiple providers are
    # present.
    configured = get_non_empty_env_value(IMAGE_GEN_ACTIVE_PROVIDER_ENV)
    if configured is not None:
        return _canonical_provider(configured)

    if resolve_google_ai_studio_api_key():
        return IMAGE_PROVIDER_GOOGLE
    if get_non_empty_env_value(MINIMAX_API_KEY_ENV):
        return IMAGE_PROVIDER_MINIMAX
    # Default to Google so the missing-key error remains actionable and
    # backwards compatible with the original Anaxa script.
    return IMAGE_PROVIDER_GOOGLE


def resolve_output_mime_type(output_file: str, requested_mime_type: str | None, scientific_mode: bool) -> str:
    if requested_mime_type:
        if requested_mime_type not in SUPPORTED_OUTPUT_MIME_TYPES:
            raise ValueError(f"Unsupported output_mime_type: {requested_mime_type}")
        return requested_mime_type
    if scientific_mode:
        return SCIENTIFIC_OUTPUT_MIME_TYPE
    guessed, _ = mimetypes.guess_type(output_file)
    if guessed in SUPPORTED_OUTPUT_MIME_TYPES:
        return guessed
    return DEFAULT_OUTPUT_MIME_TYPE


def resolve_image_size(requested_image_size: str | None, scientific_mode: bool) -> str:
    image_size = requested_image_size or (SCIENTIFIC_IMAGE_SIZE if scientific_mode else DEFAULT_IMAGE_SIZE)
    if image_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(f"Unsupported image_size: {image_size}")
    return image_size


def resolve_model_name(
    provider: str = IMAGE_PROVIDER_GOOGLE,
    model: str | None = None,
    draft_mode: bool = False,
) -> str:
    provider = _canonical_provider(provider)
    if model:
        return model.strip()
    if provider == IMAGE_PROVIDER_GOOGLE:
        if draft_mode:
            return DEFAULT_DRAFT_MODEL
        return get_non_empty_env_value(IMAGE_GEN_GOOGLE_MODEL_ENV) or DEFAULT_IMAGE_MODEL
    if provider == IMAGE_PROVIDER_OPENAI:
        resolved = get_non_empty_env_value(IMAGE_GEN_OPENAI_MODEL_ENV)
        if not resolved:
            raise RuntimeError(
                "OpenAI-compatible image provider model is not configured. "
                "Set IMAGE_GEN_OPENAI_MODEL or pass --model."
            )
        return resolved
    return get_non_empty_env_value(MINIMAX_IMAGE_MODEL_ENV) or "image-01"


def resolve_provider_base_url(provider: str, base_url: str | None = None) -> str | None:
    provider = _canonical_provider(provider)
    if provider == IMAGE_PROVIDER_MINIMAX:
        return normalize_base_url(
            base_url or get_non_empty_env_value(MINIMAX_API_HOST_ENV) or MINIMAX_DEFAULT_HOST
        )
    if provider != IMAGE_PROVIDER_OPENAI:
        return None
    resolved = normalize_base_url(base_url) or normalize_base_url(get_non_empty_env_value(IMAGE_GEN_OPENAI_BASE_URL_ENV))
    if not resolved:
        raise RuntimeError(
            "OpenAI-compatible image provider base URL is not configured. "
            "Set IMAGE_GEN_OPENAI_BASE_URL or pass --base-url."
        )
    validate_openai_compatible_base_url(resolved)
    return resolved


def resolve_provider_api_key(provider: str) -> str:
    provider = _canonical_provider(provider)
    if provider == IMAGE_PROVIDER_GOOGLE:
        api_key = resolve_google_ai_studio_api_key()
        if not api_key:
            raise RuntimeError("Google AI Studio API key is not set. Configure GEMINI_API_KEY or GOOGLE_API_KEY.")
        return api_key
    if provider == IMAGE_PROVIDER_OPENAI:
        api_key = resolve_openai_compatible_api_key()
        if not api_key:
            raise RuntimeError(
                "OpenAI-compatible image provider API key is not set. "
                "Configure IMAGE_GEN_OPENAI_API_KEY."
            )
        return api_key
    api_key = get_non_empty_env_value(MINIMAX_API_KEY_ENV)
    if not api_key:
        raise RuntimeError("MiniMax image provider API key is not set. Configure MINIMAX_API_KEY.")
    return api_key


def load_prompt_payload(prompt_file: str) -> tuple[str, dict | None]:
    raw = Path(prompt_file).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw, None
    return json.dumps(payload, ensure_ascii=False, indent=2), payload if isinstance(payload, dict) else None


def guess_reference_mime_type(reference_image: str) -> str:
    guessed, _ = mimetypes.guess_type(reference_image)
    return guessed or "image/jpeg"


def map_openai_compatible_size(aspect_ratio: str, image_size: str) -> str:
    if aspect_ratio not in SUPPORTED_OPENAI_COMPATIBLE_ASPECT_RATIOS:
        raise RuntimeError(
            f"OpenAI-compatible image provider does not support aspect ratio '{aspect_ratio}'. "
            "Supported values are 1:1, 16:9, and 9:16."
        )
    return OPENAI_COMPATIBLE_SIZE_MAP[aspect_ratio][image_size]


def build_openai_compatible_generation_request(
    *,
    prompt_text: str,
    model: str,
    image_size: str,
    aspect_ratio: str,
) -> dict:
    return {
        "model": model,
        "prompt": prompt_text,
        "n": 1,
        "size": map_openai_compatible_size(aspect_ratio, image_size),
        "response_format": "b64_json",
    }


def build_generation_request(
    *,
    prompt_text: str,
    inline_parts: list[dict],
    aspect_ratio: str,
    model: str = DEFAULT_IMAGE_MODEL,
    image_size: str = DEFAULT_IMAGE_SIZE,
    output_mime_type: str = DEFAULT_OUTPUT_MIME_TYPE,
) -> tuple[str, dict]:
    """Backward-compatible Google request builder used by older callers.

    The shared utility owns the current Gemini payload shape; this wrapper
    keeps the pre-provider-split helper name available to integrations.
    ``output_mime_type`` is accepted for API compatibility because Gemini's
    endpoint negotiates the returned MIME type itself.
    """
    del output_mime_type
    return google_image_utils.build_google_generation_request(
        prompt_text=prompt_text,
        inline_parts=inline_parts,
        aspect_ratio=aspect_ratio,
        model=model,
        image_size=image_size,
    )


def extract_openai_compatible_image(payload: dict, timeout_seconds: int) -> tuple[bytes, str | None]:
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("OpenAI-compatible image provider returned no data")
    first = data[0] or {}
    if isinstance(first, dict):
        b64_payload = first.get("b64_json")
        if b64_payload:
            return base64.b64decode(b64_payload), None
        image_url = first.get("url")
        if image_url:
            response = requests.get(image_url, timeout=timeout_seconds)
            response.raise_for_status()
            return response.content, response.headers.get("Content-Type")
    raise RuntimeError("OpenAI-compatible image provider returned no image content")


def parse_openai_compatible_response_json(response: requests.Response, endpoint: str) -> dict:
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"OpenAI-compatible image provider at {endpoint} returned a non-JSON response. "
            f"status={response.status_code}; content_type={response_content_type(response)}; "
            f"preview={response_preview(response)}. {OPENAI_COMPATIBLE_BASE_URL_HINT}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"OpenAI-compatible image provider at {endpoint} returned JSON that is not an object. "
            f"preview={json_preview(payload)}. {OPENAI_COMPATIBLE_BASE_URL_HINT}"
        )
    return payload


def format_openai_compatible_status_error(response: requests.Response, endpoint: str, model: str, size: str) -> str:
    return (
        f"OpenAI-compatible image provider at {endpoint} returned status {response.status_code} "
        f"for model '{model}' and size {size}. content_type={response_content_type(response)}; "
        f"preview={response_preview(response)}. {OPENAI_COMPATIBLE_BASE_URL_HINT}"
    )


def save_generated_image_bytes(image_bytes: bytes, output_file: str) -> dict[str, int | None]:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_bytes(image_bytes)

    if Image is None:
        return {"width": None, "height": None}
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return {"width": image.width, "height": image.height}
    except Exception:
        return {"width": None, "height": None}


def extract_image_part(payload: dict) -> dict:
    """Compatibility alias for the original Google-only helper."""
    return google_image_utils.extract_google_image_part(payload)


def save_generated_image(image_part: dict, output_file: str) -> dict[str, int | None]:
    """Save a Google inline image part using the current byte-saving helper."""
    return save_generated_image_bytes(base64.b64decode(image_part["data"]), output_file)


def write_generation_manifest(manifest_file: str, manifest: dict) -> None:
    manifest_path = Path(manifest_file)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_manifest_prompt(prompt_payload: dict | None, prompt_text: str) -> str:
    if not prompt_payload:
        return prompt_text
    prompt_value = prompt_payload.get("prompt")
    if isinstance(prompt_value, str) and prompt_value.strip():
        return prompt_value
    return prompt_text


def _minimax_host() -> str:
    return (
        get_non_empty_env_value(MINIMAX_API_HOST_ENV) or MINIMAX_DEFAULT_HOST
    ).rstrip("/")


def _check_base_resp(payload: dict) -> None:
    base = payload.get("base_resp") or {}
    if base.get("status_code", 0) != 0:
        raise RuntimeError(
            f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}"
        )


def _guess_mime(image_path: str) -> str:
    """Return a stable MIME type for MiniMax reference images."""
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")


def _to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{_guess_mime(image_path)};base64,{encoded}"


def _ensure_output_dir(output_file: str) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)


def _minimax_prompt(raw: str) -> str:
    """Extract the consolidated prompt MiniMax image-01 accepts."""
    text = raw.strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    if isinstance(payload, dict):
        prompt_value = payload.get("prompt")
        if isinstance(prompt_value, str) and prompt_value.strip():
            return prompt_value.strip()
    return text


def _generate_image_minimax(
    prompt: str,
    reference_images: list[str],
    output_file: str,
    aspect_ratio: str,
    *,
    model: str | None = None,
    scientific_mode: bool = False,
    timeout_seconds: int = 120,
) -> str:
    """Generate an image through MiniMax while retaining the DeerFlow API."""
    api_key = get_non_empty_env_value(MINIMAX_API_KEY_ENV)
    if not api_key:
        # This exact return form is part of the original DeerFlow skill
        # contract; provider selection itself still raises for missing Google
        # credentials so callers get an actionable error in that path.
        return "MINIMAX_API_KEY is not set"

    minimax_prompt = _minimax_prompt(prompt)
    if scientific_mode:
        minimax_prompt = f"{SCIENTIFIC_GUARDRAIL}\n\n{minimax_prompt}"
    if len(minimax_prompt) > MINIMAX_PROMPT_MAX_CHARS:
        return (
            f"Prompt is {len(minimax_prompt)} characters but MiniMax image-01 accepts at most "
            f"{MINIMAX_PROMPT_MAX_CHARS}. Shorten the prompt to stay within the limit; "
            "reference images plus a tighter description usually recover the detail."
        )

    body: dict[str, object] = {
        "model": model or get_non_empty_env_value(MINIMAX_IMAGE_MODEL_ENV) or "image-01",
        "prompt": minimax_prompt,
        "aspect_ratio": aspect_ratio,
        "response_format": "base64",
        "n": 1,
        "prompt_optimizer": True,
    }
    if reference_images:
        # MiniMax accepts reference subjects as data URLs.  Keep this path
        # intentionally permissive: malformed files are reported by the API,
        # matching DeerFlow's original behavior.
        body["subject_reference"] = [
            {"type": "character", "image_file": _to_data_url(path)}
            for path in reference_images
        ]

    request_timeout = min(max(int(timeout_seconds), 1), 60)
    response = requests.post(
        f"{_minimax_host()}/v1/image_generation",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=request_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("MiniMax returned an invalid response payload")
    _check_base_resp(payload)

    data = payload.get("data") or {}
    images: list[str] = []
    if isinstance(data, dict):
        candidate_images = data.get("image_base64") or []
        if isinstance(candidate_images, list):
            images = [value for value in candidate_images if isinstance(value, str)]
    elif isinstance(data, list):
        # A few MiniMax-compatible gateways return the OpenAI-style list shape.
        for item in data:
            if isinstance(item, dict):
                value = item.get("b64_json") or item.get("image_base64")
                if isinstance(value, str):
                    images.append(value)
    if not images:
        raise RuntimeError("MiniMax returned no image data")

    _ensure_output_dir(output_file)
    try:
        image_bytes = base64.b64decode(images[0], validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("MiniMax returned invalid base64 image data") from exc
    Path(output_file).write_bytes(image_bytes)
    return f"Successfully generated image to {output_file}"


def generate_image(
    prompt_file: str,
    reference_images: list[str],
    output_file: str,
    aspect_ratio: str = "16:9",
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    image_size: str | None = None,
    output_mime_type: str | None = None,
    scientific_mode: bool = False,
    manifest_file: str | None = None,
    draft_mode: bool = False,
    timeout_seconds: int = 120,
) -> str:
    raw_prompt_text, prompt_payload = load_prompt_payload(prompt_file)
    prompt_text = raw_prompt_text
    if scientific_mode:
        prompt_text = f"{SCIENTIFIC_GUARDRAIL}\n\n{prompt_text}"
    resolved_provider = resolve_provider_name(provider)
    resolved_model = resolve_model_name(resolved_provider, model, draft_mode=draft_mode)
    resolved_base_url = resolve_provider_base_url(resolved_provider, base_url)
    resolved_image_size = resolve_image_size(image_size, scientific_mode)
    resolved_output_mime_type = resolve_output_mime_type(output_file, output_mime_type, scientific_mode)

    # MiniMax accepts references without local decoding.  Keep the historical
    # validation behavior for Google/OpenAI, where inline image parts are built
    # locally and invalid files should be skipped.
    if resolved_provider == IMAGE_PROVIDER_MINIMAX:
        valid_reference_images = list(reference_images)
        parts: list[dict] = []
    else:
        parts = []
        valid_reference_images = []
        for ref_img in reference_images:
            if validate_image(ref_img):
                valid_reference_images.append(ref_img)
            else:
                print(f"Skipping invalid reference image: {ref_img}")
        if len(valid_reference_images) < len(reference_images):
            skipped = len(reference_images) - len(valid_reference_images)
            print(f"Note: {skipped} reference image(s) were skipped due to validation failure.")
        for reference_image in valid_reference_images:
            with open(reference_image, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode("ascii")
            parts.append(
                {
                    "inlineData": {
                        "mimeType": guess_reference_mime_type(reference_image),
                        "data": image_b64,
                    }
                }
            )

    if resolved_provider == IMAGE_PROVIDER_OPENAI and valid_reference_images:
        raise RuntimeError(
            "OpenAI-compatible image provider v1 does not support reference-guided image generation. "
            "Switch to Google AI Studio or remove reference images."
        )

    actual_mime_type = resolved_output_mime_type
    retry_count = 0
    if resolved_provider == IMAGE_PROVIDER_MINIMAX:
        result_message = _generate_image_minimax(
            raw_prompt_text,
            valid_reference_images,
            output_file,
            aspect_ratio,
            model=resolved_model,
            scientific_mode=scientific_mode,
            timeout_seconds=timeout_seconds,
        )
        if not Path(output_file).is_file():
            return result_message
        image_bytes = Path(output_file).read_bytes()
        actual_mime_type = _guess_mime(output_file)
    elif resolved_provider == IMAGE_PROVIDER_GOOGLE:
        api_key = resolve_provider_api_key(resolved_provider)
        result = google_image_utils.execute_google_image_request(
            requests_module=requests,
            api_key=api_key,
            model=resolved_model,
            prompt_text=prompt_text,
            inline_parts=parts,
            aspect_ratio=aspect_ratio,
            image_size=resolved_image_size,
            timeout_seconds=timeout_seconds,
            force_image_size=scientific_mode,
            sleep_fn=google_image_utils.time.sleep,
        )
        payload = result.payload
        retry_count = result.retry_count
        image_part = google_image_utils.extract_google_image_part(payload)
        actual_mime_type = image_part.get("mimeType") or image_part.get("mime_type") or resolved_output_mime_type
        image_bytes = base64.b64decode(image_part["data"])
    else:
        api_key = resolve_provider_api_key(resolved_provider)
        assert resolved_base_url is not None
        endpoint = openai_compatible_generations_url(resolved_base_url)
        request_size = map_openai_compatible_size(aspect_ratio, resolved_image_size)
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=build_openai_compatible_generation_request(
                    prompt_text=prompt_text,
                    model=resolved_model,
                    image_size=resolved_image_size,
                    aspect_ratio=aspect_ratio,
                ),
                timeout=timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"OpenAI-compatible image provider request to {endpoint} timed out after {timeout_seconds} seconds. "
                f"{OPENAI_COMPATIBLE_BASE_URL_HINT}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"OpenAI-compatible image provider request to {endpoint} failed before a response was received: {exc}. "
                f"{OPENAI_COMPATIBLE_BASE_URL_HINT}"
            ) from exc
        if response.status_code != 200:
            raise RuntimeError(
                format_openai_compatible_status_error(response, endpoint, resolved_model, request_size)
            )
        payload = parse_openai_compatible_response_json(response, endpoint)
        image_bytes, actual_mime_type = extract_openai_compatible_image(payload, timeout_seconds)
        actual_mime_type = actual_mime_type or resolved_output_mime_type

    actual_dimensions = save_generated_image_bytes(image_bytes, output_file)

    manifest_path = manifest_file or str(Path(output_file).with_name("generation_manifest.json"))
    config_source = "explicit-override" if any([provider, model, base_url]) else "settings-default"
    manifest = {
        "provider": resolved_provider,
        "model": resolved_model,
        "resolved_model": resolved_model,
        "resolved_base_url": resolved_base_url,
        "config_source": config_source,
        "image_size": resolved_image_size,
        "output_mime_type": resolved_output_mime_type,
        "aspect_ratio": aspect_ratio,
        "scientific_mode": scientific_mode,
        "draft_mode": draft_mode,
        "prompt_file": prompt_file,
        "prompt": resolve_manifest_prompt(prompt_payload, prompt_text),
        "rendered_prompt": prompt_text,
        "negative_prompt": (prompt_payload or {}).get("negative_prompt") if prompt_payload else None,
        "figure_type": (prompt_payload or {}).get("figure_type") if prompt_payload else None,
        "references": valid_reference_images,
        "output_file": output_file,
        "actual_output": {
            "width": actual_dimensions["width"],
            "height": actual_dimensions["height"],
            "mime_type": actual_mime_type,
        },
        "manifest_created_at": datetime.now(UTC).isoformat(),
        "retry_count": retry_count,
        "retried": retry_count > 0,
    }
    write_generation_manifest(manifest_path, manifest)
    return f"Successfully generated image to {output_file}. Manifest written to {manifest_path}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate images using Google AI Studio, OpenAI-compatible, or MiniMax APIs"
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Absolute path to JSON prompt file",
    )
    parser.add_argument(
        "--reference-images",
        nargs="*",
        default=[],
        help="Absolute paths to reference images (space-separated)",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output path for generated image",
    )
    parser.add_argument(
        "--aspect-ratio",
        required=False,
        default="16:9",
        help="Aspect ratio of the generated image",
    )
    parser.add_argument(
        "--provider",
        required=False,
        default=None,
        help="Image provider override: google-ai-studio, openai-compatible, or minimax",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Google AI Studio image generation model name",
    )
    parser.add_argument(
        "--base-url",
        required=False,
        default=None,
        help="OpenAI-compatible image generation base URL override",
    )
    parser.add_argument(
        "--image-size",
        required=False,
        default=None,
        help="Requested output size: 1K, 2K, or 4K",
    )
    parser.add_argument(
        "--output-mime-type",
        required=False,
        default=None,
        help="Requested output MIME type: image/png or image/jpeg",
    )
    parser.add_argument(
        "--scientific-mode",
        action="store_true",
        help="Enable scientific illustration guardrails and defaults",
    )
    parser.add_argument(
        "--manifest-file",
        required=False,
        default=None,
        help="Optional path for generation manifest JSON output",
    )
    parser.add_argument(
        "--draft-mode",
        action="store_true",
        help="Use lower-cost draft defaults when no model is explicitly provided",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Network timeout in seconds (MiniMax caps each request at 60 seconds)",
    )

    args = parser.parse_args()

    try:
        print(
            generate_image(
                args.prompt_file,
                args.reference_images,
                args.output_file,
                args.aspect_ratio,
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
                image_size=args.image_size,
                output_mime_type=args.output_mime_type,
                scientific_mode=args.scientific_mode,
                manifest_file=args.manifest_file,
                draft_mode=args.draft_mode,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except Exception as e:
        print(f"Error while generating image: {e}")
