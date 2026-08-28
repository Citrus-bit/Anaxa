"""Contract tests for Anaxa's optional outer auth in the Helm chart.

The chart is often tested on machines without Helm (for example, local
backend-only runs), so the source-level checks below always run. Render checks
are enabled automatically when a Helm binary is available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART = REPO_ROOT / "deploy" / "helm" / "deer-flow"
NGINX_TEMPLATE = CHART / "templates" / "configmap-nginx.yaml"
FRONTEND_TEMPLATE = CHART / "templates" / "frontend-deployment.yaml"
GATEWAY_TEMPLATE = CHART / "templates" / "gateway-deployment.yaml"
SECRET_TEMPLATE = CHART / "templates" / "secret-app.yaml"
VALUES = CHART / "values.yaml"


def _render_chart(*settings: str) -> list[dict]:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is unavailable")
    command = [helm, "template", "deer-flow", str(CHART)]
    for setting in settings:
        command.extend(["--set-string", setting])
    rendered = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [document for document in yaml.safe_load_all(rendered) if isinstance(document, dict)]


def _named(documents: list[dict], kind: str, suffix: str) -> dict:
    return next(document for document in documents if document.get("kind") == kind and document.get("metadata", {}).get("name", "").endswith(suffix))


def _container(deployment: dict, name: str) -> dict:
    return next(item for item in deployment["spec"]["template"]["spec"]["containers"] if item["name"] == name)


def test_helm_auth_source_contract() -> None:
    nginx = NGINX_TEMPLATE.read_text(encoding="utf-8")
    frontend = FRONTEND_TEMPLATE.read_text(encoding="utf-8")
    gateway = GATEWAY_TEMPLATE.read_text(encoding="utf-8")
    secret = SECRET_TEMPLATE.read_text(encoding="utf-8")
    values = VALUES.read_text(encoding="utf-8")

    assert "location = /_auth_check" in nginx
    assert "internal;" in nginx
    assert "proxy_pass http://frontend_upstream/api/internal/auth-guard;" in nginx
    assert "location ^~ /api/auth/" in nginx
    assert "location ^~ /api/session/" in nginx
    assert "location ^~ /api/v1/auth/" in nginx
    assert "auth_request /_auth_check;" in nginx
    assert 'proxy_set_header X-Medrix-Admin-Token "";' in nginx
    assert "location /health" in nginx
    assert "location = /nginx-health" in nginx

    assert "MEDRIX_FLOW_ENV" in frontend
    assert "MEDRIX_FLOW_UI_PASSWORD" in frontend
    assert "MEDRIX_GATEWAY_ADMIN_TOKEN" in frontend
    assert "optional: true" in frontend
    assert "BETTER_AUTH_SECRET" in gateway
    assert "MEDRIX_FLOW_ENV" in gateway
    assert "MEDRIX_GATEWAY_ADMIN_TOKEN" in gateway

    assert "MEDRIX_FLOW_UI_PASSWORD" in secret
    assert "MEDRIX_GATEWAY_ADMIN_TOKEN" in secret
    assert "existingAppSecret" in values
    assert "uiPassword" in values
    assert "adminToken" in values


def test_rendered_default_chart_keeps_auth_fail_closed_and_health_public() -> None:
    documents = _render_chart()
    frontend = _container(_named(documents, "Deployment", "-frontend"), "frontend")
    env = {item["name"]: item for item in frontend["env"]}
    assert env["NODE_ENV"]["value"] == "production"
    assert env["MEDRIX_FLOW_ENV"]["value"] == "production"
    assert env["MEDRIX_FLOW_UI_PASSWORD"]["valueFrom"]["secretKeyRef"]["optional"] is True
    assert env["MEDRIX_GATEWAY_ADMIN_TOKEN"]["valueFrom"]["secretKeyRef"]["optional"] is True

    gateway = _container(_named(documents, "Deployment", "-gateway"), "gateway")
    gateway_env = {item["name"]: item for item in gateway["env"]}
    assert gateway_env["MEDRIX_FLOW_ENV"]["value"] == "production"
    assert gateway_env["BETTER_AUTH_SECRET"]["valueFrom"]["secretKeyRef"]["key"] == "BETTER_AUTH_SECRET"
    assert gateway_env["MEDRIX_GATEWAY_ADMIN_TOKEN"]["valueFrom"]["secretKeyRef"]["optional"] is True

    nginx = _named(documents, "ConfigMap", "-nginx")
    config = nginx["data"]["nginx.conf"]
    assert "auth_request /_auth_check;" in config
    assert "location /health {" in config
    assert "location = /nginx-health {" in config


def test_rendered_auth_values_land_in_app_secret_and_frontend() -> None:
    documents = _render_chart(
        "frontend.auth.environment=production",
        "frontend.auth.uiPassword=ui-secret-value",
        "frontend.auth.adminToken=admin-secret-value",
    )
    secret = _named(documents, "Secret", "-app")
    assert secret["stringData"]["MEDRIX_FLOW_UI_PASSWORD"] == "ui-secret-value"
    assert secret["stringData"]["MEDRIX_GATEWAY_ADMIN_TOKEN"] == "admin-secret-value"

    frontend = _container(_named(documents, "Deployment", "-frontend"), "frontend")
    refs = {item["name"]: item["valueFrom"]["secretKeyRef"] for item in frontend["env"] if "valueFrom" in item and "secretKeyRef" in item["valueFrom"]}
    assert refs["MEDRIX_FLOW_UI_PASSWORD"]["name"].endswith("-app")
    assert refs["MEDRIX_GATEWAY_ADMIN_TOKEN"]["name"].endswith("-app")


def test_existing_app_secret_is_used_by_workload_references() -> None:
    helper = (CHART / "templates" / "_helpers.tpl").read_text(encoding="utf-8")
    assert 'default (printf "%s-app" (include "deer-flow.fullname" .)) .Values.existingAppSecret' in helper
