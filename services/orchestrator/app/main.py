from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import File, Form, UploadFile

from app.personality import create_personality_plan
from common.api import create_app
from common.errors import ServiceError
from common.gpu import job_slot
from common.jobs import atomic_write_json, new_job_dir
from common.media import AUDIO_EXTENSIONS, cleanup_old_jobs, save_upload_limited
from common.settings import (
    SHARED_JOBS_ROOT,
    env,
    env_float,
    env_int,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
app = create_app("avatar-orchestrator")


def form_fields(data: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in data.items():
        if value is None:
            continue
        cleaned = str(value)
        if key == "seed" and not cleaned.strip():
            continue
        fields[key] = cleaned
    return fields


async def _post_files(
    stage: str,
    url: str,
    *,
    data: dict[str, Any],
    files: dict[str, tuple[str, Any, str]],
) -> dict[str, Any]:
    timeout = env_float("SERVICE_REQUEST_TIMEOUT_SECONDS", 9000.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(
                url,
                data=form_fields(data),
                files=files,
            )
    except httpx.TimeoutException as exc:
        raise ServiceError(
            f"{stage} service timed out.",
            code="pipeline_stage_timeout",
            status_code=504,
            details={"stage": stage},
        ) from exc
    except httpx.HTTPError as exc:
        raise ServiceError(
            f"{stage} service is unreachable.",
            code="pipeline_stage_unreachable",
            status_code=503,
            details={"stage": stage, "error": type(exc).__name__},
        ) from exc

    if response.status_code >= 400:
        raise ServiceError(
            f"{stage} stage failed.",
            code="pipeline_stage_failed",
            status_code=502,
            details={
                "stage": stage,
                "upstream_status": response.status_code,
                "upstream_response": response.text[-1500:],
            },
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ServiceError(
            f"{stage} service returned invalid JSON.",
            code="pipeline_invalid_response",
            status_code=502,
            details={"stage": stage},
        ) from exc
    if not isinstance(payload, dict):
        raise ServiceError(
            f"{stage} service returned an unexpected response.",
            code="pipeline_invalid_response",
            status_code=502,
            details={"stage": stage},
        )
    return payload


def failure_payload(
    result: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    if isinstance(exc, ServiceError):
        error = {
            "code": exc.code,
            "message": str(exc),
            "details": exc.details,
        }
    else:
        error = {
            "code": "internal_error",
            "message": "Avatar pipeline failed unexpectedly.",
            "details": {"exception": type(exc).__name__},
        }
    return {**result, "status": "failed", "error": error}


@app.on_event("startup")
def cleanup_jobs() -> None:
    removed = cleanup_old_jobs(
        SHARED_JOBS_ROOT,
        env_int("JOB_RETENTION_DAYS", 7),
    )
    if removed:
        logger.info("Removed %d expired job directories", removed)


@app.post("/v1/jobs")
async def create_avatar_job(
    photo: UploadFile = File(...),
    reference_audio: UploadFile = File(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    age: int = Form(...),
    user_text: str = Form(""),
    gender: str = Form("female"),
    style: str = Form("realistic"),
    persona: str = Form("自然、友善、可靠的数字人助手"),
    scene: str = Form("日常对话"),
    emotion_hint: str = Form("auto"),
    gesture_hint: str = Form("auto"),
    video_prompt: str | None = Form(None),
    seed: int | None = Form(None),
) -> dict[str, Any]:
    if not photo.filename or not reference_audio.filename:
        raise ServiceError(
            "photo and reference_audio are required.",
            code="missing_input",
            status_code=422,
        )
    audio_suffix = Path(reference_audio.filename).suffix.lower()
    if audio_suffix not in AUDIO_EXTENSIONS:
        raise ServiceError(
            "Unsupported reference audio format.",
            code="unsupported_audio",
            status_code=422,
        )
    clean_user_text = " ".join(user_text.split())
    max_text_chars = env_int("MAX_TEXT_CHARS", 2000)
    if len(clean_user_text) > max_text_chars:
        raise ServiceError(
            f"user_text exceeds the {max_text_chars} character limit.",
            code="text_too_long",
            status_code=422,
        )
    if not 16 <= age <= 100:
        raise ServiceError(
            "age must be between 16 and 100.",
            code="invalid_age",
            status_code=422,
        )

    job_dir = new_job_dir("avatar")
    photo_path = job_dir / "input" / "photo.bin"
    audio_path = job_dir / "input" / f"reference{audio_suffix}"
    result: dict[str, Any] = {
        "job_id": job_dir.name,
        "status": "running",
        "stages": {},
    }
    atomic_write_json(job_dir / "result.json", result)

    try:
        await save_upload_limited(
            photo,
            photo_path,
            max_bytes=env_int("MAX_IMAGE_MB", 20) * 1024 * 1024,
        )
        await save_upload_limited(
            reference_audio,
            audio_path,
            max_bytes=env_int("MAX_AUDIO_MB", 50) * 1024 * 1024,
        )

        with job_slot():
            planning = await create_personality_plan(
                user_text=clean_user_text,
                persona=persona,
                scene=scene,
                age=age,
                gender=gender,
                emotion_hint=emotion_hint,
                gesture_hint=gesture_hint,
            )
            plan = planning["plan"]
            result["personality"] = planning
            atomic_write_json(job_dir / "result.json", result)

            vision_data: dict[str, Any] = {
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "age": age,
                "gender": gender,
                "style": style,
            }
            if seed is not None:
                vision_data["seed"] = seed
            with photo_path.open("rb") as photo_file:
                vision = await _post_files(
                    "vision",
                    f"{env('VISION_URL', 'http://127.0.0.1:8001').rstrip('/')}"
                    "/v1/avatar/image",
                    data=vision_data,
                    files={
                        "photo": (
                            "photo.png",
                            photo_file,
                            "application/octet-stream",
                        )
                    },
                )
            result["stages"]["vision"] = vision
            atomic_write_json(job_dir / "result.json", result)

            with audio_path.open("rb") as audio_file:
                speech = await _post_files(
                    "tts",
                    f"{env('TTS_URL', 'http://127.0.0.1:8002').rstrip('/')}"
                    "/v1/avatar/speech",
                    data={
                        "text": plan["text"],
                        "emotion": plan["emotion"],
                        "speed": plan["speed"],
                    },
                    files={
                        "reference_audio": (
                            f"reference{audio_suffix}",
                            audio_file,
                            "application/octet-stream",
                        )
                    },
                )
            result["stages"]["tts"] = speech
            atomic_write_json(job_dir / "result.json", result)

            video_image = Path(str(vision["video_reference_path"]))
            speech_audio = Path(str(speech["audio_path"]))
            if not video_image.is_file() or not speech_audio.is_file():
                raise ServiceError(
                    "Intermediate image or speech artifact is missing.",
                    code="pipeline_artifact_missing",
                    status_code=500,
                )
            video_data: dict[str, Any] = {
                "prompt": video_prompt or plan["video_prompt"],
            }
            if seed is not None:
                video_data["seed"] = seed
            with video_image.open("rb") as image_file, speech_audio.open(
                "rb"
            ) as speech_file:
                video = await _post_files(
                    "video",
                    f"{env('VIDEO_URL', 'http://127.0.0.1:8003').rstrip('/')}"
                    "/v1/avatar/video",
                    data=video_data,
                    files={
                        "image": (
                            "video_reference.png",
                            image_file,
                            "application/octet-stream",
                        ),
                        "audio": (
                            "speech.wav",
                            speech_file,
                            "application/octet-stream",
                        ),
                    },
                )
            result["stages"]["video"] = video
            result["artifacts"] = {
                "avatar_image": vision.get("fullbody_path"),
                "video_reference": vision.get("video_reference_path"),
                "pose": vision.get("pose_path"),
                "speech": speech.get("audio_path"),
                "video": video.get("video_path"),
                "video_log": video.get("log_path"),
            }
            result["status"] = "completed"
            result["result_path"] = str(job_dir / "result.json")
            atomic_write_json(job_dir / "result.json", result)
            return result
    except Exception as exc:
        if not isinstance(exc, ServiceError):
            logger.exception("Avatar pipeline failed unexpectedly")
        atomic_write_json(job_dir / "result.json", failure_payload(result, exc))
        raise
