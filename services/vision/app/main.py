from __future__ import annotations

import logging

from fastapi import File, Form, UploadFile

from app.engine import ENGINE
from common.api import create_app
from common.errors import ServiceError
from common.gpu import gpu_slot
from common.jobs import atomic_write_json, new_job_dir
from common.media import normalize_image, save_upload_limited
from common.settings import env_int

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
app = create_app("avatar-vision")


@app.post("/v1/avatar/image")
async def generate_avatar(
    photo: UploadFile = File(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    age: int = Form(...),
    gender: str = Form("female"),
    style: str = Form("realistic"),
    seed: int | None = Form(None),
) -> dict[str, object]:
    if not photo.filename:
        raise ServiceError(
            "A photo file is required.",
            code="missing_photo",
            status_code=422,
        )
    job_dir = new_job_dir("vision")
    input_path = job_dir / "input" / "photo.bin"
    await save_upload_limited(
        photo,
        input_path,
        max_bytes=env_int("MAX_IMAGE_MB", 20) * 1024 * 1024,
    )
    normalized_path = job_dir / "input" / "photo.png"
    width, height = normalize_image(input_path, normalized_path)
    input_path.unlink(missing_ok=True)

    with gpu_slot():
        output = ENGINE.generate(
            photo_path=normalized_path,
            output_dir=job_dir / "output",
            height_cm=height_cm,
            weight_kg=weight_kg,
            age=age,
            gender=gender,
            style=style,
            seed=seed,
        )

    response = {
        "job_id": job_dir.name,
        "output_dir": str(job_dir / "output"),
        "input_width": width,
        "input_height": height,
        **output,
    }
    atomic_write_json(job_dir / "result.json", response)
    return response
