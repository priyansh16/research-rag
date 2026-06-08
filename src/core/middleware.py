import time
import uuid
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Per-request middleware that:
    - Assigns a unique correlation ID to every request
    - Logs method, path, status code, and wall-clock latency
    - Attaches X-Request-ID header to the response

    The correlation ID flows through loguru context so every log
    line emitted during a request carries the same ID — making
    distributed tracing trivial to add later.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]  # short ID, easy to grep
        start = time.perf_counter()

        # Bind the request ID into loguru context for this request's lifetime
        with logger.contextualize(request_id=request_id):
            logger.info(
                f"→ {request.method} {request.url.path}",
                request_id=request_id,
            )

            try:
                response: Response = await call_next(request)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.exception(
                    f"✗ {request.method} {request.url.path} "
                    f"UNHANDLED ERROR ({elapsed_ms:.1f}ms)"
                )
                raise exc

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"← {request.method} {request.url.path} "
                f"{response.status_code} ({elapsed_ms:.1f}ms)"
            )

            response.headers["X-Request-ID"] = request_id
            return response