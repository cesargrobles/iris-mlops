# src/logger.py
"""Central logging configuration for the MLOps pipeline."""

import logging
from pathlib import Path


def get_logger(name: str = "mlops") -> logging.Logger:
    """Return a logger that writes to stdout and a local file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path("logs") / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
