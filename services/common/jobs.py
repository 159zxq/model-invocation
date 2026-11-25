from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from common.settings import SHARED_JOBS_ROOT


def new_job_dir(prefix: str) -> Path:
    safe_prefix = "".join(c for c in prefix if c.isalnum() or c in "-_") or "job"
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    job_id = f"{safe_prefix}-{stamp}-{uuid.uuid4().hex[:10]}"
    path = SHARED_JOBS_ROOT / job_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
