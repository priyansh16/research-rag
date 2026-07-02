import sys
import logging
from loguru import logger

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:<8}</level> | "
    "[<cyan>{extra[request_id]}</cyan>] | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

def setup_logging(debug: bool = False) -> None:
    """
    Configure Loguru for the application.

    Console:
        - INFO/DEBUG depending on DEBUG flag

    logs/app.log:
        - INFO+

    logs/error.log:
        - ERROR+

    Also redirects stdlib logging (uvicorn, FastAPI, etc.)
    into Loguru.
    
    """

    logger.remove()
    
    # Default value for logs outside request context
    logger.configure(extra={"request_id": "-"})

    console_level = "DEBUG" if debug else "INFO"
    
    # -----------------------------
    # Console
    # -----------------------------
    logger.add(
        sys.stdout,
        level=console_level,
        format=LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=debug,  
    )

    # -----------------------------
    # Application logs
    # -----------------------------   
    logger.add(
        "logs/app.log",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format=LOG_FORMAT,
        backtrace=True,
        diagnose=False,
    )
    
    # -----------------------------
    # Error logs
    # -----------------------------
    logger.add(
        "logs/error.log",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format=LOG_FORMAT,
        backtrace=True,
        diagnose=False,
    )
    
    # -----------------------------
    # Intercept stdlib logging
    # -----------------------------
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
                
            frame, depth = logging.currentframe(), 2
            
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(
                depth=depth, 
                exception=record.exc_info
                ).log(level, record.getMessage()
            )

    logging.basicConfig(
        handlers=[InterceptHandler()], 
        level=0, 
        force=True)
    
    # Redirect uvicorn loggers
    for logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
    ):
        uv_logger = logging.getLogger(logger_name)
        uv_logger.handlers = [InterceptHandler()]
        uv_logger.propagate = False

    
    