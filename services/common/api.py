from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI

from common.errors import install_error_handlers
from common.settings import ensure_runtime_dirs


def create_app(name: str, version: str = "0.1.0") -> FastAPI:
    ensure_runtime_dirs()
    app = FastAPI(
        title=name,
        version=version,
        docs_url="/docs",
        redoc_url=None,
    )
    install_error_handlers(app)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": name,
            "version": version,
            "pid": os.getpid(),
            "time": time.time(),
        }

    return app
