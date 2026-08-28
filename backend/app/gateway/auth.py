"""Gateway access-control helpers for sensitive management endpoints."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import secrets

from fastapi import HTTPException, Request

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
    client_host = request.client.host if request.client else None
    if not client_host:
        return False

    lowered = client_host.strip().lower()
    if lowered in _TRUSTED_LOOPBACK_HOSTS:
        return True

    ip = _parse_ip(client_host)
    return ip is not None and ip.is_loopback


def get_proxy_authorization_token() -> str | None:
    """Return the shared proxy-auth token used between nginx auth_request and the gateway.

    Only derives from BETTER_AUTH_SECRET: falling back to MEDRIX_FLOW_UI_PASSWORD would
    make the proxy token equal to sha256(user_password:...), letting anyone who knows
    the UI password forge nginx-bypass tokens. BETTER_AUTH_SECRET is already required
    by the frontend env schema in production.
    """

    secret_value = os.getenv("BETTER_AUTH_SECRET", "").strip()
    if not secret_value:
        return None
    return hashlib.sha256(f"{secret_value}:medrix-flow-proxy-auth".encode()).hexdigest()


async def require_admin_access(request: Request) -> None:
    """Allow loopback callers or requests presenting the admin token."""

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
