from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def runtime_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, str(PROJECT_ROOT / "runtime" / default))).resolve()


RUNTIME_ROOT = runtime_path("RUNTIME_ROOT", "")
MODELS_ROOT = runtime_path("MODELS_ROOT", "models")
REPOS_ROOT = runtime_path("REPOS_ROOT", "repos")
VENVS_ROOT = runtime_path("VENVS_ROOT", "venvs")
LOG_ROOT = runtime_path("LOG_ROOT", "logs")
PID_ROOT = runtime_path("PID_ROOT", "pids")
SHARED_JOBS_ROOT = runtime_path("SHARED_JOBS_ROOT", "jobs")
GPU_LOCK_FILE = Path(
    os.getenv("GPU_LOCK_FILE", str(RUNTIME_ROOT / "gpu.lock"))
).resolve()
JOB_LOCK_FILE = Path(
    os.getenv("JOB_LOCK_FILE", str(RUNTIME_ROOT / "job.lock"))
).resolve()


def ensure_runtime_dirs() -> None:
    for path in (RUNTIME_ROOT, MODELS_ROOT, LOG_ROOT, PID_ROOT, SHARED_JOBS_ROOT):
        path.mkdir(parents=True, exist_ok=True)
