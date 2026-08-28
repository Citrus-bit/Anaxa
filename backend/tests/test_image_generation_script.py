from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
import requests
from PIL import Image


def _load_image_generation_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "skills" / "public" / "image-generation" / "scripts" / "generate.py"
    spec = importlib.util.spec_from_file_location("image_generation_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_bytes(color: str = "blue", size: tuple[int, int] = (8, 8)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeResponse:
    def __init__(
        self,
        payload: object,
        status_code: int = 200,
        *,
        headers: dict[str, str] | None = None,
        text: str | None = None,
        json_exc: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        self.text = text if text is not None else (json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload))
        self.content = self.text.encode("utf-8")
        self._json_exc = json_exc

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> object:
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


def test_generate_image_uses_google_ai_studio_alias_and_writes_manifest(tmp_path, monkeypatch):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(
        json.dumps(
            {
                "figure_type": "graphical_abstract",
                "prompt": "A clean scientific graphical abstract about multimodal agents.",
                "negative_prompt": "charts, axes, p-values",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "scientific-illustration-4k.png"
    manifest_file = tmp_path / "generation_manifest.json"

    captured: dict = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(_png_bytes(size=(32, 24))).decode("utf-8"),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-studio-key")
    monkeypatch.setattr(module.requests, "post", fake_post)

    message = module.generate_image(
        str(prompt_file),
        [],
        str(output_file),
        scientific_mode=True,
        manifest_file=str(manifest_file),
    )

    assert "Successfully generated image" in message
    assert captured["url"].endswith("/models/gemini-3-pro-image-preview:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "google-studio-key"
    assert captured["json"]["generationConfig"]["imageConfig"]["imageSize"] == "4K"
    assert captured["json"]["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
    assert output_file.exists()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["provider"] == "google-ai-studio"
    assert manifest["scientific_mode"] is True
    assert manifest["image_size"] == "4K"
    assert manifest["output_mime_type"] == "image/png"
    assert manifest["figure_type"] == "graphical_abstract"
    assert manifest["negative_prompt"] == "charts, axes, p-values"
    assert manifest["actual_output"]["width"] == 32
    assert manifest["actual_output"]["height"] == 24
    assert manifest["retry_count"] == 0
    assert manifest["retried"] is False


def test_generate_image_raises_when_no_api_key(monkeypatch, tmp_path):
    module = _load_image_generation_module()
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("simple prompt", encoding="utf-8")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Google AI Studio API key is not set"):
        module.generate_image(str(prompt_file), [], str(tmp_path / "out.png"))


def test_generate_image_raises_when_no_image_data_returned(monkeypatch, tmp_path):
    module = _load_image_generation_module()
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("simple prompt", encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse({"candidates": [{"content": {"parts": [{"text": "no image"}]}}]}),
    )

    with pytest.raises(RuntimeError, match="no image data"):
        module.generate_image(str(prompt_file), [], str(tmp_path / "out.png"))


def test_generate_image_manifest_prompt_falls_back_to_rendered_prompt(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(
        json.dumps(
            {
                "figure_type": "concept_explainer",
                "composition": "clean three-part composition",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "scientific-illustration-4k.png"
    manifest_file = tmp_path / "generation_manifest.json"

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(_png_bytes()).decode("utf-8"),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    )

    module.generate_image(
        str(prompt_file),
        [],
        str(output_file),
        scientific_mode=True,
        manifest_file=str(manifest_file),
    )

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["prompt"] == manifest["rendered_prompt"]


def test_generate_image_omits_image_size_for_flash_image_smoke_requests(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("simple draft prompt", encoding="utf-8")
    output_file = tmp_path / "draft.png"
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(_png_bytes()).decode("utf-8"),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(module.requests, "post", fake_post)

    module.generate_image(
        str(prompt_file),
        [],
        str(output_file),
        draft_mode=True,
    )

    assert captured["json"]["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
    assert "imageSize" not in captured["json"]["generationConfig"]["imageConfig"]


def test_generate_image_retries_google_503_and_records_retry_count(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("simple prompt", encoding="utf-8")
    output_file = tmp_path / "out.png"
    manifest_file = tmp_path / "generation_manifest.json"

    responses = [
        _FakeResponse(
            {"error": {"code": 503, "message": "This model is currently experiencing high demand."}},
            status_code=503,
        ),
        _FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(_png_bytes(size=(24, 24))).decode("utf-8"),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    ]

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(module.google_image_utils.time, "sleep", lambda *_args, **_kwargs: None)

    call_count = {"value": 0}

    def fake_post(*args, **kwargs):
        idx = call_count["value"]
        call_count["value"] += 1
        return responses[idx]

    monkeypatch.setattr(module.requests, "post", fake_post)

    message = module.generate_image(
        str(prompt_file),
        [],
        str(output_file),
        manifest_file=str(manifest_file),
    )

    assert "Successfully generated image" in message
    assert call_count["value"] == 2
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["retry_count"] == 1
    assert manifest["retried"] is True


def test_generate_image_raises_when_google_timeouts_are_exhausted(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("simple prompt", encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(module.google_image_utils.time, "sleep", lambda *_args, **_kwargs: None)

    call_count = {"value": 0}

    def fake_post(*args, **kwargs):
        call_count["value"] += 1
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(module.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="timed out after 120 seconds"):
        module.generate_image(str(prompt_file), [], str(tmp_path / "out.png"))

    assert call_count["value"] == 3


def test_generate_image_uses_openai_compatible_provider_from_settings(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Draw a clean scientific concept figure.", encoding="utf-8")
    output_file = tmp_path / "output.png"
    manifest_file = tmp_path / "generation_manifest.json"

    captured: dict = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse(
            {
                "data": [
                    {
                        "b64_json": base64.b64encode(_png_bytes(size=(64, 36))).decode("utf-8"),
                    }
                ]
            }
        )

    monkeypatch.setenv("IMAGE_GEN_ACTIVE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_BASE_URL", "https://images.example.com/v1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_API_KEY", "openai-image-key")
    monkeypatch.setattr(module.requests, "post", fake_post)

    message = module.generate_image(
        str(prompt_file),
        [],
        str(output_file),
        aspect_ratio="16:9",
        manifest_file=str(manifest_file),
    )

    assert "Successfully generated image" in message
    assert captured["url"] == "https://images.example.com/v1/images/generations"
    assert captured["headers"]["Authorization"] == "Bearer openai-image-key"
    assert captured["json"]["model"] == "gpt-image-1"
    assert captured["json"]["size"] == "2048x1152"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["provider"] == "openai-compatible"
    assert manifest["resolved_model"] == "gpt-image-1"
    assert manifest["resolved_base_url"] == "https://images.example.com/v1"
    assert manifest["config_source"] == "settings-default"


def test_generate_image_supports_openai_compatible_url_payload(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Draw a clean scientific concept figure.", encoding="utf-8")
    output_file = tmp_path / "output.png"

    monkeypatch.setenv("IMAGE_GEN_ACTIVE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_BASE_URL", "https://images.example.com/v1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_API_KEY", "openai-image-key")

    def fake_post(*args, **kwargs):
        return _FakeResponse({"data": [{"url": "https://cdn.example.com/image.png"}]})

    def fake_get(url, *, timeout):
        assert url == "https://cdn.example.com/image.png"
        return type(
            "DownloadResponse",
            (),
            {
                "content": _png_bytes(size=(40, 20)),
                "headers": {"Content-Type": "image/png"},
                "raise_for_status": staticmethod(lambda: None),
            },
        )()

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module.requests, "get", fake_get)

    module.generate_image(str(prompt_file), [], str(output_file))

    assert output_file.exists()


def test_generate_image_rejects_unsupported_openai_compatible_aspect_ratio(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Draw a clean scientific concept figure.", encoding="utf-8")

    monkeypatch.setenv("IMAGE_GEN_ACTIVE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_BASE_URL", "https://images.example.com/v1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_API_KEY", "openai-image-key")

    with pytest.raises(RuntimeError, match="does not support aspect ratio"):
        module.generate_image(str(prompt_file), [], str(tmp_path / "out.png"), aspect_ratio="4:3")


def test_generate_image_rejects_reference_images_for_openai_compatible_provider(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Draw a clean scientific concept figure.", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(_png_bytes())

    monkeypatch.setenv("IMAGE_GEN_ACTIVE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_BASE_URL", "https://images.example.com/v1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_API_KEY", "openai-image-key")

    with pytest.raises(RuntimeError, match="does not support reference-guided image generation"):
        module.generate_image(
            str(prompt_file),
            [str(reference_image)],
            str(tmp_path / "out.png"),
        )


def test_generate_image_rejects_openai_compatible_full_endpoint_base_url(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Draw a clean scientific concept figure.", encoding="utf-8")

    monkeypatch.setenv("IMAGE_GEN_ACTIVE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_BASE_URL", "https://images.example.com/v1/images/generations")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_API_KEY", "openai-image-key")

    with pytest.raises(RuntimeError, match="base URL must be the API root path"):
        module.generate_image(str(prompt_file), [], str(tmp_path / "out.png"))


def test_generate_image_surfaces_openai_compatible_non_json_response(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Draw a clean scientific concept figure.", encoding="utf-8")

    monkeypatch.setenv("IMAGE_GEN_ACTIVE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_BASE_URL", "https://images.example.com/v1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_API_KEY", "openai-image-key")
    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            "<html>not json</html>",
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html>not json</html>",
            json_exc=ValueError("Expecting value"),
        ),
    )

    with pytest.raises(RuntimeError, match="returned a non-JSON response"):
        module.generate_image(str(prompt_file), [], str(tmp_path / "out.png"))


def test_generate_image_surfaces_openai_compatible_status_with_preview(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Draw a clean scientific concept figure.", encoding="utf-8")

    monkeypatch.setenv("IMAGE_GEN_ACTIVE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_BASE_URL", "https://images.example.com/v1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_API_KEY", "openai-image-key")
    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            {"error": "not found"},
            status_code=404,
            text='{"error":"not found"}',
        ),
    )

    with pytest.raises(RuntimeError, match="returned status 404"):
        module.generate_image(str(prompt_file), [], str(tmp_path / "out.png"))


def test_generate_image_surfaces_openai_compatible_timeout(monkeypatch, tmp_path):
    module = _load_image_generation_module()

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Draw a clean scientific concept figure.", encoding="utf-8")

    monkeypatch.setenv("IMAGE_GEN_ACTIVE_PROVIDER", "openai-compatible")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_BASE_URL", "https://images.example.com/v1")
    monkeypatch.setenv("IMAGE_GEN_OPENAI_API_KEY", "openai-image-key")
    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.Timeout("timed out")),
    )

    with pytest.raises(RuntimeError, match="timed out after 120 seconds"):
        module.generate_image(str(prompt_file), [], str(tmp_path / "out.png"))


def test_image_generation_script_falls_back_to_deerflow_google_utility(monkeypatch):
    del monkeypatch
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "skills" / "public" / "image-generation" / "scripts" / "generate.py"
    probe = """
import importlib.util
import sys

class BlockMedrixImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'medrix_flow' or fullname.startswith('medrix_flow.'):
            raise ModuleNotFoundError('blocked for namespace fallback test', name=fullname)
        return None

sys.meta_path.insert(0, BlockMedrixImports())
spec = importlib.util.spec_from_file_location('image_generation_fallback_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.google_image_utils.__name__)
print(module.DEFAULT_DRAFT_MODEL)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(script)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["deerflow.utils.google_image", "gemini-2.5-flash-image"]
