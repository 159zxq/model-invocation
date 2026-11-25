from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _post_job(client: TestClient):
    return client.post(
        "/v1/jobs",
        data={
            "height_cm": "172",
            "weight_kg": "64",
            "age": "28",
            "gender": "female",
            "style": "realistic",
            "user_text": "你好",
        },
        files={
            "photo": ("photo.jpg", b"photo", "image/jpeg"),
            "reference_audio": ("voice.wav", b"audio", "audio/wav"),
        },
    )


def _prepare_job(monkeypatch: pytest.MonkeyPatch, module, tmp_path: Path):
    job_dir = tmp_path / "job"
    monkeypatch.setattr(module, "new_job_dir", lambda prefix: job_dir)
    monkeypatch.setattr(module, "job_slot", nullcontext)

    async def personality_plan(**kwargs):
        return {
            "plan": {
                "text": "你好，很高兴认识你。",
                "emotion": "happy",
                "gesture": "greet",
                "speed": 1.0,
                "video_prompt": "A person greets the viewer.",
            },
            "source": "test",
        }

    monkeypatch.setattr(module, "create_personality_plan", personality_plan)
    return job_dir


def test_orchestrator_runs_vision_tts_video_in_order(
    load_service,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_service("orchestrator")
    job_dir = _prepare_job(monkeypatch, module, tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_post(stage, url, *, data, files):
        calls.append((stage, dict(data)))
        if stage == "vision":
            image = job_dir / "vision.png"
            image.write_bytes(b"vision")
            return {
                "fullbody_path": str(image),
                "video_reference_path": str(image),
                "pose_path": str(image),
            }
        if stage == "tts":
            audio = job_dir / "speech.wav"
            audio.write_bytes(b"audio")
            return {"audio_path": str(audio)}
        video = job_dir / "avatar.mp4"
        video.write_bytes(b"video")
        return {"video_path": str(video), "log_path": str(job_dir / "video.log")}

    monkeypatch.setattr(module, "_post_files", fake_post)

    with TestClient(module.app) as client:
        response = _post_job(client)

    assert response.status_code == 200
    assert [stage for stage, _ in calls] == ["vision", "tts", "video"]
    assert all("seed" not in data for _, data in calls)
    assert response.json()["status"] == "completed"
    assert json.loads((job_dir / "result.json").read_text("utf-8"))[
        "status"
    ] == "completed"


def test_orchestrator_records_unexpected_failure(
    load_service,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_service("orchestrator")
    job_dir = _prepare_job(monkeypatch, module, tmp_path)

    async def fail_post(stage, url, *, data, files):
        raise RuntimeError("unexpected test failure")

    monkeypatch.setattr(module, "_post_files", fail_post)

    with TestClient(module.app, raise_server_exceptions=False) as client:
        response = _post_job(client)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    saved = json.loads((job_dir / "result.json").read_text("utf-8"))
    assert saved["status"] == "failed"
    assert saved["error"]["code"] == "internal_error"


def test_form_fields_omits_empty_seed(load_service) -> None:
    module = load_service("orchestrator")

    fields = module.form_fields(
        {
            "height_cm": 172,
            "seed": "",
            "prompt": None,
            "style": "realistic",
        }
    )

    assert fields == {"height_cm": "172", "style": "realistic"}
