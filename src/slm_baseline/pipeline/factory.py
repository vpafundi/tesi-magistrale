"""Costruzione di :class:`ExperimentContext` da un file YAML."""

from __future__ import annotations

import logging
from pathlib import Path

from ..artifacts.paths import experiment_paths
from ..logging.core import setup_pipeline_logger
from ..config.experiment import (
    ExperimentConfig,
    canonical_config_hash,
    experiment_output_dir,
    load_merged_experiment_dict,
)
from .context import ExperimentContext


def build_context(
    config_path: Path,
    *,
    force: bool = False,
    rerun: bool = False,
    dry_run_slm: bool = False,
    logger: logging.Logger | None = None,
) -> ExperimentContext:
    """Carica YAML mergeato e inizializza path/hash per gli stage.

    Args:
        config_path: Percorso a ``configs/experiments/<id>.yaml`` (o equivalente).
        force: Propagato a ``ExperimentContext.force`` (prepare).
        rerun: Propagato a ``ExperimentContext.rerun`` (classic/slm).
        dry_run_slm: Propagato a ``ExperimentContext.dry_run_slm`` (solo stampa comandi).
        logger: Logger condiviso; default da :func:`~slm_baseline.logging.setup_pipeline_logger`.

    Returns:
        Contesto pronto per :class:`~slm_baseline.pipeline.stages.prepare.PreparePipeline`
        o pipeline di valutazione.

    Raises:
        FileNotFoundError: YAML o file referenziato da ``extends`` assente.
        KeyError: YAML senza campi obbligatori (es. ``experiment_id``).
    """
    config_path = config_path.resolve()
    merged = load_merged_experiment_dict(config_path)
    cfg = ExperimentConfig.from_dict(merged)
    exp_dir = experiment_output_dir(cfg)
    log = setup_pipeline_logger(logger)
    log.info(
        "context: experiment_id=%s config=%s",
        cfg.experiment_id,
        config_path,
    )
    return ExperimentContext(
        config_path=config_path,
        merged=merged,
        cfg=cfg,
        exp_dir=exp_dir,
        paths=experiment_paths(cfg),
        config_hash=canonical_config_hash(merged),
        force=force,
        rerun=rerun,
        dry_run_slm=dry_run_slm,
        logger=log,
    )
