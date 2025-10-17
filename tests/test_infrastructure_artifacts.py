from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path,description",
    [
        ("Dockerfile", "container runtime Dockerfile"),
        ("deploy/helm/agnitra-marketplace/Chart.yaml", "Helm chart descriptor"),
        ("deploy/helm/agnitra-marketplace/templates/deployment.yaml", "Helm deployment template"),
        ("deploy/terraform/aws_marketplace/main.tf", "AWS marketplace Terraform module"),
        ("deploy/terraform/gcp_marketplace/main.tf", "GCP marketplace Terraform module"),
        ("deploy/terraform/azure_marketplace/main.tf", "Azure marketplace Terraform module"),
    ],
)
def test_infrastructure_artifacts_exist(path: str, description: str) -> None:
    artifact = PROJECT_ROOT / path
    assert artifact.exists(), f"Missing {description} at {path}"


def test_dockerfile_targets_runtime_service() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "uvicorn" in dockerfile
    assert "agnitra.api.app:create_app" in dockerfile
    assert "EXPOSE 8080" in dockerfile


def test_usage_api_routes_are_exposed() -> None:
    app_module = (PROJECT_ROOT / "agnitra" / "api" / "app.py").read_text(encoding="utf-8")
    assert 'Route("/usage"' in app_module
    assert "create_default_dispatcher" in app_module
    assert "MarketplaceDispatcher" in app_module
