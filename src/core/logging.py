import sys
import logging
from loguru import logger


def setup_logging(debug: bool = False) -> None:
    """
    Configure loguru for structured, leveled output.

    In production (debug=False): JSON-style, WARNING and above.
    In development (debug=True): human-readable, DEBUG and above.

    Call once from main.py at startup.
    """

    logger.remove()

    level = "DEBUG" if debug else "WARNING"

    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=debug,  
    )

    # Also write WARNING+ to a file for persistence
    logger.add(
        "logs/app.log",
        level="WARNING",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
        backtrace=True,
        diagnose=False,  # prevent writing variable values to disk
    )

    # Intercept stdlib logging (uvicorn, sqlalchemy, etc.) into loguru
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back  # type: ignore[assignment]
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)