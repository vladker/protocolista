#!/usr/bin/env python3
"""
Logging Configuration Module
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", format: Optional[str] = None, datefmt: Optional[str] = None) -> None:
    """
    Настройка логирования

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        format: Формат сообщений
        datefmt: Формат даты/времени
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    log_format = format or "%(asctime)s | %(levelname)-8s | %(message)s"
    log_datefmt = datefmt or "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
