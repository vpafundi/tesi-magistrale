"""Configurazione logger unificato (stderr, formato compatto)."""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

PIPELINE_LOGGER_NAME = "slm_baseline.pipeline"

_configured = False


def pipeline_log_level(default: int = logging.INFO) -> int:
    """Legge ``TESI_LOG_LEVEL`` (DEBUG, INFO, WARNING, …)."""
    raw = os.environ.get("TESI_LOG_LEVEL", "").strip().upper()
    if not raw:
        return default
    return getattr(logging, raw, default)


def setup_pipeline_logger(
    logger: logging.Logger | None = None,
    *,
    level: int | None = None,
    stream: TextIO | None = None,
    name: str = PIPELINE_LOGGER_NAME,
) -> logging.Logger:
    """Restituisce il logger pipeline, configurando handler e formato se assenti.

    Args:
        logger: Istanza esterna da riusare; se ``None``, usa ``logging.getLogger(name)``.
        level: Livello minimo (``INFO`` per run operative).
        stream: Destinazione log; default ``stderr``.
        name: Nome logger root pipeline.

    Returns:
        Logger pronto per ``info`` / ``error`` in orchestrator, stage e CLI.
    """
    global _configured
    resolved = pipeline_log_level(logging.INFO) if level is None else level
    log = logger or logging.getLogger(name)
    log.setLevel(resolved)
    if stream is None:
        stream = sys.stderr
    if not log.handlers:
        handler = logging.StreamHandler(stream)
        handler.setLevel(resolved)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        log.addHandler(handler)
        log.propagate = False
        _configured = True
    else:
        for h in log.handlers:
            h.setLevel(resolved)
    return log


def get_pipeline_logger() -> logging.Logger:
    """Logger pipeline già configurato o con setup default.

    Returns:
        Stesso logger usato da :func:`setup_pipeline_logger` senza argomenti.
    """
    return setup_pipeline_logger()
