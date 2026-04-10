"""Logging setup with Rich for pretty output."""

import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console

console = Console()


def setup_logger(
    name: str = "company_analysis",
    level: str = "INFO",
    log_file: str | None = None,
    console_output: bool = True,
) -> logging.Logger:
    """Set up a logger with Rich console + optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    if console_output:
        handler = RichHandler(
            console=console, show_time=True, show_path=False, markup=True,
        )
        handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    return logger
