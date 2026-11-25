from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from fastapi import UploadFile
from PIL import Image

from common.errors import ServiceError
from common.media import contain_square_image, save_upload_limited


def test_contain_square_image_preserves_full_frame() -> None:
    source = Image.new("RGB", (400, 100), (255, 0, 0))
    result = contain_square_image(source, size=200)

    assert result.size == (200, 200)
    assert result.getpixel((0, 0)) == (127, 127, 127)
    assert result.getpixel((100, 100)) == (255, 0, 0)
    assert result.getpixel((100, 74)) == (127, 127, 127)


def test_save_upload_limited_rejects_oversized_file(tmp_path: Path) -> None:
    upload = UploadFile(filename="photo.png", file=io.BytesIO(b"x" * 11))

    with pytest.raises(ServiceError) as caught:
        asyncio.run(
            save_upload_limited(
                upload,
                tmp_path / "photo.png",
                max_bytes=10,
            )
        )

    assert caught.value.code == "upload_too_large"
    assert caught.value.status_code == 413
    assert not (tmp_path / "photo.png").exists()
