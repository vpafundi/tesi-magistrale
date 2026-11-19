"""Logging condiviso per pipeline ed esperimenti."""

from .core import PIPELINE_LOGGER_NAME, get_pipeline_logger, setup_pipeline_logger

__all__ = [
    "PIPELINE_LOGGER_NAME",
    "get_pipeline_logger",
    "setup_pipeline_logger",
]
