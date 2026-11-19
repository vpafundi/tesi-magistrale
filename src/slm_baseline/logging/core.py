"""Configurazione logger unificato (stderr, formato compatto)."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

PIPELINE_LOGGER_NAME = "slm_baseline.pipeline"

_configured = False


def setup_pipeline_logger(
    logger: logging.Logger | None = None,
    *,
    level: int = logging.INFO,
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
    log = logger or logging.getLogger(name)
    log.setLevel(level)
    if stream is None:
        stream = sys.stderr
    if not log.handlers:
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        log.addHandler(handler)
        log.propagate = False
        _configured = True
    return log


def get_pipeline_logger() -> logging.Logger:
    """Logger pipeline già configurato o con setup default.

    Returns:
        Stesso logger usato da :func:`setup_pipeline_logger` senza argomenti.
    """
    return setup_pipeline_logger()
