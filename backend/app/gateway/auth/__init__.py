"""Authentication module for DeerFlow.

This module provides:
- JWT-based authentication
- Provider Factory pattern for extensible auth methods
- UserRepository interface for storage backends (SQLite)
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import secrets

from fastapi import HTTPException, Request

from app.gateway.auth.config import AuthConfig, get_auth_config, set_auth_config
from app.gateway.auth.errors import AuthErrorCode, AuthErrorResponse, TokenError
from app.gateway.auth.jwt import TokenPayload, create_access_token, decode_token
from app.gateway.auth.local_provider import LocalAuthProvider
from app.gateway.auth.models import User, UserResponse
from app.gateway.auth.password import hash_password, verify_password
from app.gateway.auth.providers import AuthProvider
from app.gateway.auth.repositories.base import UserRepository

# These helpers were part of Anaxa's former ``app.gateway.auth`` module.  Keep
# them at the package boundary so existing management routers and integrations
# continue to work while the DeerFlow provider-based auth implementation is
# used for normal session authentication.
ADMIN_TOKEN_HEADER = "x-medrix-admin-token"
PROXY_AUTH_HEADER = "x-medrix-proxy-authorized"
_TRUSTED_LOOPBACK_HOSTS = {"localhost", "testclient"}


def _parse_ip(value: str | None) -> ipaddress._BaseAddress | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.startswith("[") and "]" in candidate:
        candidate = candidate[1 : candidate.index("]")]
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return None


def is_loopback_request(request: Request) -> bool:
    """Return whether the socket peer is a loopback client.

    Do not trust forwarded headers here: this helper protects direct gateway
    management endpoints and therefore must inspect the actual ASGI client.
    """
    client_host = request.client.host if request.client else None
    if not client_host:
        return False
    lowered = client_host.strip().lower()
    if lowered in _TRUSTED_LOOPBACK_HOSTS:
        return True
    ip = _parse_ip(client_host)
    return ip is not None and ip.is_loopback


def get_proxy_authorization_token() -> str | None:
    """Derive the shared nginx-to-gateway authorization token."""
    secret_value = os.getenv("BETTER_AUTH_SECRET", "").strip()
    if not secret_value:
        return None
    return hashlib.sha256(f"{secret_value}:medrix-flow-proxy-auth".encode()).hexdigest()


async def require_admin_access(request: Request) -> None:
    """Allow loopback callers or requests presenting a configured admin token."""
    if is_loopback_request(request):
        return

    expected_proxy_token = get_proxy_authorization_token()
    provided_proxy_token = request.headers.get(PROXY_AUTH_HEADER, "")
    if expected_proxy_token and secrets.compare_digest(provided_proxy_token, expected_proxy_token):
        return

    expected_token = os.getenv("MEDRIX_GATEWAY_ADMIN_TOKEN", "").strip()
    provided_token = request.headers.get(ADMIN_TOKEN_HEADER, "")
    if expected_token and secrets.compare_digest(provided_token, expected_token):
        return

    if expected_token or expected_proxy_token:
        raise HTTPException(status_code=401, detail="Admin authorization required for this endpoint.")
    raise HTTPException(
        status_code=403,
        detail=("This endpoint is restricted to loopback clients unless MEDRIX_GATEWAY_ADMIN_TOKEN is configured."),
    )


__all__ = [
    # Config
    "AuthConfig",
    "get_auth_config",
    "set_auth_config",
    # Errors
    "AuthErrorCode",
    "AuthErrorResponse",
    "TokenError",
    # JWT
    "TokenPayload",
    "create_access_token",
    "decode_token",
    # Password
    "hash_password",
    "verify_password",
    # Models
    "User",
    "UserResponse",
    # Providers
    "AuthProvider",
    "LocalAuthProvider",
    # Repository
    "UserRepository",
    # Anaxa management-endpoint compatibility helpers
    "ADMIN_TOKEN_HEADER",
    "PROXY_AUTH_HEADER",
    "get_proxy_authorization_token",
    "is_loopback_request",
    "require_admin_access",
]
