from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "service_error",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details


def error_body(
    code: str,
    message: str,
    *,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if request_id:
        error["request_id"] = request_id
    if details:
        error["details"] = details
    return {"error": error}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def handle_service_error(
        request: Request, exc: ServiceError
    ) -> JSONResponse:
        request_id = request.headers.get("x-request-id")
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                exc.code,
                str(exc),
                request_id=request_id,
                details=exc.details,
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = request.headers.get("x-request-id")
        logger.exception("Unhandled request error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=error_body(
                "internal_error",
                "Internal service error. Check the service log for details.",
                request_id=request_id,
            ),
        )
