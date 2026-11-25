from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common.errors import ServiceError


def _valid_generate_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "photo_path": tmp_path / "missing.png",
        "output_dir": tmp_path / "output",
        "height_cm": 172.0,
        "weight_kg": 64.0,
        "age": 28,
        "gender": "female",
        "style": "realistic",
    }


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("height_cm", 119.0, "invalid_height"),
        ("height_cm", 231.0, "invalid_height"),
        ("weight_kg", 29.0, "invalid_weight"),
        ("weight_kg", 251.0, "invalid_weight"),
        ("age", 15, "invalid_age"),
        ("age", 101, "invalid_age"),
    ],
)
def test_vision_rejects_invalid_profile_before_loading_runtime(
    load_service,
    tmp_path: Path,
    field: str,
    value: object,
    error_code: str,
) -> None:
    module = load_service("vision")
    kwargs = _valid_generate_kwargs(tmp_path)
    kwargs[field] = value

    with pytest.raises(ServiceError) as caught:
        module.ENGINE.generate(**kwargs)

    assert caught.value.code == error_code
    assert caught.value.status_code == 422


def test_vision_requires_existing_normalized_photo(
    load_service,
    tmp_path: Path,
) -> None:
    module = load_service("vision")

    with pytest.raises(ServiceError) as caught:
        module.ENGINE.generate(**_valid_generate_kwargs(tmp_path))

    assert caught.value.code == "invalid_image"


def test_vision_api_rejects_undecodable_image(
    load_service,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_service("vision")
    monkeypatch.setattr(
        "common.jobs.SHARED_JOBS_ROOT",
        tmp_path / "jobs",
    )

    with TestClient(module.app) as client:
        response = client.post(
            "/v1/avatar/image",
            data={
                "height_cm": "172",
                "weight_kg": "64",
                "age": "28",
            },
            files={
                "photo": ("photo.png", b"not-an-image", "image/png"),
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"
