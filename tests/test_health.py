from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    ("service_name", "expected_name"),
    [
        ("vision", "avatar-vision"),
        ("tts", "avatar-tts"),
        ("video", "avatar-video"),
        ("orchestrator", "avatar-orchestrator"),
    ],
)
def test_service_health_does_not_load_models(
    load_service,
    service_name: str,
    expected_name: str,
) -> None:
    module = load_service(service_name)
    with TestClient(module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == expected_name
