"""Request logging middleware, split out so ``api.py`` holds routes and nothing else.

The access log is structured (see ``src/logging_config.py``, ``LOG_FORMAT=json``)
because the container's stdout is the only observability this project has. There
is no APM, no tracing backend and no cloud budget for one. Latency and status code
per request is the minimum that makes a running container debuggable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)

#: Polled every 30s by the compose healthcheck. Logging it would drown the log.
_SILENT_PATHS = {"/health"}


def install_request_logging(app: FastAPI) -> None:
    """Attach the access-log middleware to ``app``."""

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in _SILENT_PATHS:
            return await call_next(request)

        started = time.time()
        response = await call_next(request)
        latency_ms = round((time.time() - started) * 1000, 1)
        message = f"{request.method} {request.url.path} {response.status_code} {latency_ms}ms"

        if response.status_code >= 500:
            logger.error(message)
        elif response.status_code >= 400:
            logger.warning(message)
        else:
            logger.info(message)
        return response
