import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique request ID
    and logs the lifecycle of every HTTP request.
    """

    async def dispatch(
        self, 
        request: Request, 
        call_next
        ) -> Response:
        
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        
        start = time.perf_counter()

        with logger.contextualize(request_id=request_id):
            
            logger.info(
                "Started {} {}",
                request.method,
                request.url.path,
            )

            try:
                response: Response = await call_next(request)
                
            except Exception as exc:
                
                elapsed_ms = (
                    time.perf_counter() - start
                    ) * 1000
                
                logger.exception(
                    "Unhandled exception while processing {} {} ({:.2f} ms)",
                    request.method,
                    request.url.path,
                    elapsed_ms,
                )
                raise

            elapsed_ms = (
                time.perf_counter() - start
                ) * 1000
            logger.info(
                "Completed {} {} -> {} ({:.2f} ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )

            response.headers["X-Request-ID"] = request_id
            return response