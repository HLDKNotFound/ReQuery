from __future__ import annotations
import logging
import os
import sys
from typing import Optional

def get_logger(
    name: str = "ReQuery",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a standardized logger with console and optional file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False

    if log_file:
        # Check if file handler for this file already exists
        existing_file_paths = [
            getattr(h, "baseFilename", None)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        abs_log_path = os.path.abspath(log_file)
        if abs_log_path not in existing_file_paths:
            parent = os.path.dirname(abs_log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            file_handler = logging.FileHandler(abs_log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
