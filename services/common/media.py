from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from common.errors import ServiceError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


async def save_upload_limited(
    upload: UploadFile,
    destination: Path,
    *,
    max_bytes: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                output.close()
                destination.unlink(missing_ok=True)
                raise ServiceError(
                    f"Uploaded file exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                    code="upload_too_large",
                    status_code=413,
                )
            output.write(chunk)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise ServiceError(
            "Uploaded file is empty.",
            code="empty_upload",
            status_code=422,
        )
    return destination


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ServiceError(
            "ffmpeg is not installed. Run scripts/setup_system.sh.",
            code="ffmpeg_missing",
            status_code=503,
        )
    return ffmpeg


def normalize_image(
    source: Path,
    destination: Path,
    *,
    max_side: int = 4096,
) -> tuple[int, int]:
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ServiceError(
            "The uploaded image cannot be decoded.",
            code="invalid_image",
            status_code=422,
        ) from exc

    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return image.size


def fit_square_image(
    source: Path,
    destination: Path,
    *,
    size: int,
) -> tuple[int, int]:
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ServiceError(
            "The uploaded image cannot be decoded.",
            code="invalid_image",
            status_code=422,
        ) from exc
    square = contain_square_image(image, size=size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    square.save(destination, format="PNG", optimize=True)
    return square.size


def contain_square_image(
    image: Image.Image,
    *,
    size: int,
    background: tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Fit the whole image into a square without cropping the subject."""
    if size <= 0:
        raise ValueError("size must be positive")
    rgb = image.convert("RGB")
    contained = ImageOps.contain(
        rgb,
        (size, size),
        method=Image.Resampling.LANCZOS,
    )
    square = Image.new("RGB", (size, size), background)
    left = (size - contained.width) // 2
    top = (size - contained.height) // 2
    square.paste(contained, (left, top))
    return square


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ServiceError(
            "Generated image is invalid.",
            code="invalid_generated_image",
            status_code=500,
        ) from exc


def validate_audio(
    path: Path,
    *,
    min_seconds: float = 2.0,
    max_seconds: float = 60.0,
) -> float:
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ServiceError(
            "Unsupported audio format. Use WAV, MP3, FLAC, M4A or OGG.",
            code="unsupported_audio",
            status_code=422,
        )
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ServiceError(
            "ffprobe is not installed. Run scripts/setup_system.sh.",
            code="ffprobe_missing",
            status_code=503,
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ServiceError(
            "The uploaded audio cannot be decoded.",
            code="invalid_audio",
            status_code=422,
            details={"ffprobe": result.stderr[-500:]},
        )
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ServiceError(
            "The uploaded audio has no usable duration.",
            code="invalid_audio",
            status_code=422,
        ) from exc
    if duration < min_seconds:
        raise ServiceError(
            f"Reference audio is too short; use at least {min_seconds:g} seconds.",
            code="audio_too_short",
            status_code=422,
            details={"duration_seconds": duration},
        )
    if duration > max_seconds:
        raise ServiceError(
            f"Reference audio is too long; keep it under {max_seconds:g} seconds.",
            code="audio_too_long",
            status_code=422,
            details={"duration_seconds": duration},
        )
    return duration


def normalize_audio(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    max_seconds: float | None = None,
) -> Path:
    ffmpeg = _require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
    ]
    if max_seconds is not None:
        command.extend(["-t", f"{max_seconds:.3f}"])
    command.append(str(destination))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ServiceError(
            "The audio file could not be converted to WAV.",
            code="invalid_audio",
            status_code=422,
            details={"ffmpeg": result.stderr[-500:]},
        )
    return destination


def probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ServiceError(
            "ffprobe is not installed. Run scripts/setup_system.sh.",
            code="ffprobe_missing",
            status_code=503,
        )
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ServiceError(
            "The media duration cannot be read.",
            code="invalid_media",
            status_code=422,
        )
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ServiceError(
            "The media file has no usable duration.",
            code="invalid_media",
            status_code=422,
        ) from exc


def media_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_chars))
            return file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def validate_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 1024:
        raise ServiceError(
            "Video generation produced no usable file.",
            code="invalid_generated_video",
            status_code=500,
        )
    with path.open("rb") as file:
        header = file.read(32)
    if b"ftyp" not in header:
        raise ServiceError(
            "Generated output is not an MP4 file.",
            code="invalid_generated_video",
            status_code=500,
        )


def cleanup_old_jobs(root: Path, retention_days: int) -> int:
    import time

    if retention_days <= 0 or not root.exists():
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            if candidate.stat().st_mtime < cutoff:
                shutil.rmtree(candidate)
                removed += 1
        except OSError:
            continue
    return removed
