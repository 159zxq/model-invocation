from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVICES_ROOT = ROOT / "services"
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))


def _clear_app_modules() -> None:
    for name in tuple(sys.modules):
        if name == "app" or name.startswith("app."):
            sys.modules.pop(name, None)


@pytest.fixture(scope="session", autouse=True)
def isolated_runtime(tmp_path_factory: pytest.TempPathFactory):
    runtime = tmp_path_factory.mktemp("runtime")
    values = {
        "RUNTIME_ROOT": runtime,
        "MODELS_ROOT": runtime / "models",
        "REPOS_ROOT": runtime / "repos",
        "VENVS_ROOT": runtime / "venvs",
        "LOG_ROOT": runtime / "logs",
        "PID_ROOT": runtime / "pids",
        "SHARED_JOBS_ROOT": runtime / "jobs",
        "GPU_LOCK_FILE": runtime / "gpu.lock",
        "JOB_LOCK_FILE": runtime / "job.lock",
        "SILICONFLOW_API_KEY": "",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def clean_service_package():
    _clear_app_modules()
    yield
    _clear_app_modules()


@pytest.fixture
def load_service(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], object]:
    def load(name: str):
        service_root = SERVICES_ROOT / name
        if not (service_root / "app" / "main.py").is_file():
            raise AssertionError(f"unknown service: {name}")
        monkeypatch.syspath_prepend(str(SERVICES_ROOT))
        monkeypatch.syspath_prepend(str(service_root))
        return importlib.import_module("app.main")

    return load
