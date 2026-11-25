from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from filelock import FileLock

from common.errors import ServiceError
from common.settings import GPU_LOCK_FILE, JOB_LOCK_FILE


@contextmanager
def gpu_slot(timeout: float | None = None) -> Iterator[None]:
    GPU_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_timeout = timeout or float(os.getenv("GPU_LOCK_TIMEOUT_SECONDS", "7500"))
    try:
        with FileLock(str(GPU_LOCK_FILE), timeout=lock_timeout):
            yield
    except TimeoutError as exc:
        raise ServiceError(
            "The GPU is busy. Retry after the current inference finishes.",
            code="gpu_busy",
            status_code=503,
        ) from exc


@contextmanager
def job_slot(timeout: float | None = None) -> Iterator[None]:
    JOB_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_timeout = timeout or float(
        os.getenv("JOB_LOCK_TIMEOUT_SECONDS", "14400")
    )
    try:
        with FileLock(str(JOB_LOCK_FILE), timeout=lock_timeout):
            yield
    except TimeoutError as exc:
        raise ServiceError(
            "Another avatar job is running. Retry later.",
            code="job_queue_timeout",
            status_code=503,
        ) from exc
