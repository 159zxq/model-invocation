from __future__ import annotations

import logging
from pathlib import Path

from fastapi import File, Form, UploadFile

from app.engine import ENGINE
from common.api import create_app
from common.errors import ServiceError
from common.gpu import gpu_slot
from common.jobs import atomic_write_json, new_job_dir
from common.media import (
    AUDIO_EXTENSIONS,
    fit_square_image,
    normalize_audio,
    normalize_image,
    save_upload_limited,
    validate_audio,
)
from common.settings import env_float, env_int

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
app = create_app("avatar-video")


@app.post("/v1/avatar/video")
async def generate_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str | None = Form(None),
    seed: int | None = Form(None),
) -> dict[str, object]:
    if not image.filename or not audio.filename:
        raise ServiceError(
            "Both image and audio files are required.",
            code="missing_video_input",
            status_code=422,
        )
    audio_suffix = Path(audio.filename).suffix.lower()
    if audio_suffix not in AUDIO_EXTENSIONS:
        raise ServiceError(
            "Unsupported video audio format.",
            code="unsupported_audio",
            status_code=422,
        )

    job_dir = new_job_dir("video")
    input_image = job_dir / "input" / "image.bin"
    input_audio = job_dir / "input" / f"audio{audio_suffix}"
    await save_upload_limited(
        image,
        input_image,
        max_bytes=env_int("MAX_IMAGE_MB", 20) * 1024 * 1024,
    )
    await save_upload_limited(
        audio,
        input_audio,
        max_bytes=env_int("MAX_AUDIO_MB", 50) * 1024 * 1024,
    )
    validate_audio(
        input_audio,
        min_seconds=env_float("VIDEO_AUDIO_MIN_SECONDS", 0.4),
        max_seconds=env_float("VIDEO_AUDIO_MAX_SECONDS", 120.0),
    )

    normalized_image = job_dir / "input" / "image.png"
    normalize_image(input_image, normalized_image)
    reference_image = job_dir / "input" / "reference.png"
    fit_square_image(
        normalized_image,
        reference_image,
        size=ENGINE.settings.sample_size,
    )
    normalized_audio = normalize_audio(
        input_audio,
        job_dir / "input" / "speech.wav",
        sample_rate=24000,
    )
    output_path = job_dir / "output" / "avatar.mp4"
    log_path = job_dir / "output" / "echomimic.log"

    with gpu_slot():
        output = ENGINE.generate(
            image_path=reference_image,
            audio_path=normalized_audio,
            output_path=output_path,
            prompt=prompt,
            seed=seed,
            log_path=log_path,
        )

    response = {
        "job_id": job_dir.name,
        "output_dir": str(job_dir / "output"),
        **output,
    }
    atomic_write_json(job_dir / "result.json", response)
    return response
