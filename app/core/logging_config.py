"""
Logging configuration for the LangGraph Helper Agent.
Configures logging to file only, not to stdout/stderr.
"""

import logging
import logging.handlers
from pathlib import Path
from app.core import settings


def configure_logging():
    """
    Configure logging to write to a file only.
    All logs go to a file, stdout is quiet.
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Remove any existing handlers to avoid duplicates
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set root logger to not output to console
    logger.setLevel(logging.INFO)

    # Create file handler that logs everything
    log_file = logs_dir / "app.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Add file handler to root logger
    logger.addHandler(file_handler)

    return log_file


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance configured for file-only output.
    """
    return logging.getLogger(name)