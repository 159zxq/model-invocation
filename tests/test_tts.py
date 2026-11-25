from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common.errors import ServiceError


def _generate(module, tmp_path: Path, **overrides):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")
    kwargs = {
        "text": "你好。",
        "reference_wav": reference,
        "output_path": tmp_path / "speech.wav",
        "emotion": "neutral",
        "speed": 1.0,
    }
    kwargs.update(overrides)
    return module.ENGINE.generate(**kwargs)


def test_tts_rejects_invalid_text_before_importing_audio_stack(
    load_service,
    tmp_path: Path,
) -> None:
    module = load_service("tts")

    with pytest.raises(ServiceError) as caught:
        _generate(module, tmp_path, text=" \n\t ")

    assert caught.value.code == "empty_text"
    assert caught.value.status_code == 422


def test_tts_rejects_text_over_configured_limit(
    load_service,
    tmp_path: Path,
) -> None:
    module = load_service("tts")
    limit = module.ENGINE.settings.max_text_chars

    with pytest.raises(ServiceError) as caught:
        _generate(module, tmp_path, text="a" * (limit + 1))

    assert caught.value.code == "text_too_long"


@pytest.mark.parametrize("speed", [0.49, 2.01])
def test_tts_rejects_speed_outside_supported_range(
    load_service,
    tmp_path: Path,
    speed: float,
) -> None:
    module = load_service("tts")

    with pytest.raises(ServiceError) as caught:
        _generate(module, tmp_path, speed=speed)

    assert caught.value.code == "invalid_speed"


def test_tts_api_rejects_unsupported_reference_format(
    load_service,
) -> None:
    module = load_service("tts")
    with TestClient(module.app) as client:
        response = client.post(
            "/v1/avatar/speech",
            data={"text": "你好。"},
            files={
                "reference_audio": ("voice.txt", b"not audio", "text/plain"),
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_audio"
