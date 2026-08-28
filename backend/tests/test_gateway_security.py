from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.gateway.auth import get_proxy_authorization_token, is_loopback_request, require_admin_access
from app.gateway.routers import mcp, setup
from medrix_flow.setup.service import (
    DEFAULT_GOOGLE_IMAGE_MODEL,
    IMAGE_PROVIDER_GOOGLE,
    ImageGenerationConfig,
    ImageProviderConfig,
    SetupConfigResponse,
)


def _build_request(client_host: str, headers: dict[str, str] | None = None) -> Request:
    request_headers = [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in (headers or {}).items()]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/setup/config",
        "raw_path": b"/api/setup/config",
        "query_string": b"",
        "root_path": "",
        "headers": request_headers,
        "client": (client_host, 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_is_loopback_request_ignores_spoofed_x_forwarded_for() -> None:
    request = _build_request("203.0.113.10", headers={"x-forwarded-for": "127.0.0.1, 203.0.113.10"})

    assert not is_loopback_request(request)


def test_require_admin_access_rejects_remote_clients_without_token(monkeypatch) -> None:
    monkeypatch.delenv("BETTER_AUTH_SECRET", raising=False)
    monkeypatch.delenv("MEDRIX_GATEWAY_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("MEDRIX_FLOW_UI_PASSWORD", raising=False)
    request = _build_request("203.0.113.10", headers={"x-forwarded-for": "127.0.0.1"})

    with pytest.raises(HTTPException, match="restricted to loopback clients") as exc_info:
        asyncio.run(require_admin_access(request))

    assert exc_info.value.status_code == 403


def test_require_admin_access_accepts_remote_clients_with_admin_token(monkeypatch) -> None:
    monkeypatch.setenv("MEDRIX_GATEWAY_ADMIN_TOKEN", "secret-token")
    request = _build_request("203.0.113.10", headers={"x-medrix-admin-token": "secret-token"})

    asyncio.run(require_admin_access(request))


def test_require_admin_access_accepts_proxy_authorized_requests(monkeypatch) -> None:
    monkeypatch.delenv("MEDRIX_GATEWAY_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("BETTER_AUTH_SECRET", "shared-auth-secret")
    proxy_token = get_proxy_authorization_token()
    assert proxy_token is not None
    request = _build_request("172.18.0.10", headers={"x-medrix-proxy-authorized": proxy_token})

    asyncio.run(require_admin_access(request))


def test_proxy_token_does_not_fall_back_to_ui_password(monkeypatch) -> None:
    """UI password must not derive the proxy bypass token.

    If it did, anyone who knows the UI password could forge nginx-bypass tokens
    and reach admin endpoints on the gateway directly.
    """
    monkeypatch.delenv("BETTER_AUTH_SECRET", raising=False)
    monkeypatch.delenv("MEDRIX_GATEWAY_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("MEDRIX_FLOW_UI_PASSWORD", "hunter2")

    assert get_proxy_authorization_token() is None


def test_setup_routes_still_work_for_loopback_clients() -> None:
    app = FastAPI()
    app.include_router(setup.router)

    with patch(
        "app.gateway.routers.setup.get_setup_config_data",
        return_value=SetupConfigResponse(
            models=[],
            tool_keys=[],
            image_generation=ImageGenerationConfig(
                active_provider=IMAGE_PROVIDER_GOOGLE,
                google_ai_studio=ImageProviderConfig(
                    provider=IMAGE_PROVIDER_GOOGLE,
                    enabled=True,
                    model=DEFAULT_GOOGLE_IMAGE_MODEL,
                    api_key="",
                    api_key_env_var="GEMINI_API_KEY",
                ),
                openai_compatible=ImageProviderConfig(
                    provider="openai-compatible",
                    enabled=False,
                    model=None,
                    base_url=None,
                    api_key="",
                    api_key_env_var="IMAGE_GEN_OPENAI_API_KEY",
                ),
            ),
        ),
    ):
        with TestClient(app) as client:
            response = client.get("/api/setup/config")

    assert response.status_code == 200
    assert response.json()["models"] == []
    assert response.json()["tool_keys"] == []
    assert response.json()["image_generation"]["active_provider"] == IMAGE_PROVIDER_GOOGLE


def test_mcp_config_returns_raw_env_placeholders(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "github": {
                        "enabled": True,
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {"GITHUB_TOKEN": "$GITHUB_TOKEN"},
                        "description": "GitHub server",
                    }
                },
                "skills": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDRIX_FLOW_EXTENSIONS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("GITHUB_TOKEN", "real-secret-value")

    app = FastAPI()
    app.include_router(mcp.router)

    with TestClient(app) as client:
        response = client.get("/api/mcp/config")

    assert response.status_code == 200
    assert response.json()["mcp_servers"]["github"]["env"]["GITHUB_TOKEN"] == "$GITHUB_TOKEN"
