from __future__ import annotations

import sys
from pathlib import Path

import pytest

from common.errors import ServiceError


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_video_command_uses_v100_safe_settings(
    load_service,
    tmp_path: Path,
) -> None:
    module = load_service("video")
    engine = module.ENGINE
    image = tmp_path / "image.png"
    audio = tmp_path / "speech.wav"
    output = tmp_path / "segment"

    command = engine._build_command(
        image_path=image,
        audio_path=audio,
        output_dir=output,
        prompt="A person is speaking naturally.",
        seed=43,
    )

    assert _value_after(command, "--weight_dtype") == "float16"
    assert _value_after(command, "--transformer_path") == str(
        engine.settings.transformer_path.resolve()
    )
    assert _value_after(command, "--sampler_name") == "Flow_Unipc"
    assert _value_after(command, "--save_path") == str(output.resolve())
    assert _value_after(command, "--seed") == "43"


def test_video_transformer_path_can_be_overridden(
    load_service,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    load_service("video")
    transformer = tmp_path / "custom-transformer.safetensors"
    monkeypatch.setenv("ECHOMIMIC_TRANSFORMER", str(transformer))

    settings = sys.modules["app.engine"].VideoSettings.from_env()

    assert settings.transformer_path == transformer


def test_video_rejects_overlong_prompt_before_model_installation(
    load_service,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_service("video")
    image = tmp_path / "image.png"
    audio = tmp_path / "speech.wav"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        sys.modules["app.engine"],
        "probe_duration",
        lambda path: 1.0,
    )

    with pytest.raises(ServiceError) as caught:
        module.ENGINE.generate(
            image_path=image,
            audio_path=audio,
            output_path=tmp_path / "avatar.mp4",
            prompt="x" * (module.ENGINE.settings.max_prompt_chars + 1),
            log_path=tmp_path / "video.log",
        )

    assert caught.value.code == "video_prompt_too_long"
