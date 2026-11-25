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
    normalize_audio,
    save_upload_limited,
    validate_audio,
)
from common.settings import env_float, env_int

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
app = create_app("avatar-tts")


@app.post("/v1/avatar/speech")
async def generate_speech(
    reference_audio: UploadFile = File(...),
    text: str = Form(...),
    emotion: str = Form("neutral"),
    speed: float = Form(1.0),
) -> dict[str, object]:
    if not reference_audio.filename:
        raise ServiceError(
            "A reference audio file is required.",
            code="missing_reference_audio",
            status_code=422,
        )
    suffix = Path(reference_audio.filename).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise ServiceError(
            "Unsupported reference audio format.",
            code="unsupported_audio",
            status_code=422,
        )

    job_dir = new_job_dir("tts")
    input_path = job_dir / "input" / f"reference{suffix}"
    await save_upload_limited(
        reference_audio,
        input_path,
        max_bytes=env_int("MAX_AUDIO_MB", 50) * 1024 * 1024,
    )
    validate_audio(
        input_path,
        min_seconds=env_float("REFERENCE_AUDIO_MIN_SECONDS", 2.0),
        max_seconds=env_float("REFERENCE_AUDIO_MAX_SECONDS", 60.0),
    )
    reference_wav = normalize_audio(
        input_path,
        job_dir / "input" / "reference.wav",
        sample_rate=env_int("TTS_SAMPLE_RATE", 24000),
    )
    output_path = job_dir / "output" / "speech.wav"

    with gpu_slot():
        output = ENGINE.generate(
            text=text,
            reference_wav=reference_wav,
            output_path=output_path,
            emotion=emotion,
            speed=speed,
        )

    response = {
        "job_id": job_dir.name,
        "output_dir": str(job_dir / "output"),
        **output,
    }
    atomic_write_json(job_dir / "result.json", response)
    return response
