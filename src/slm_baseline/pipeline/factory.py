"""Factory for :class:`~slm_baseline.pipeline.context.ExperimentContext`.

Loads and merges experiment YAML, resolves output paths and config hash, and
packages runtime flags so prepare, evaluation, and training stages share one
object. Called by :class:`~slm_baseline.pipeline.orchestrator.BaselineExperimentOrchestrator`
and tests before running any stage.
"""

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
    """Load merged YAML and build paths, hash, and flags for pipeline stages.

    Args:
        config_path: Path to an experiment YAML (e.g. ``configs/experiments/<id>.yaml``).
        force: Stored on the context; when ``True``, prepare may overwrite existing artifacts.
        rerun: Stored on the context; when ``True``, classic/SLM stages re-run instead of skipping.
        dry_run_slm: Stored on the context; when ``True``, SLM stage only prints subprocess commands.
        logger: Shared logger; defaults to :func:`~slm_baseline.logging.setup_pipeline_logger`.

    Returns:
        Context ready for :class:`~slm_baseline.pipeline.stages.prepare.PreparePipeline`
        and evaluation or training pipelines.

    Raises:
        FileNotFoundError: The YAML or a file referenced by ``extends`` is missing.
        KeyError: Required YAML fields (e.g. ``experiment_id``) are absent.
    """
    log = setup_pipeline_logger(logger)

    log.info("context: start config=%s force=%s rerun=%s dry_run_slm=%s", config_path, force, rerun, dry_run_slm)
    try:
        config_path = config_path.resolve()
        # load the merged experiment dictionary from the config file
        merged = load_merged_experiment_dict(config_path)
        # create the experiment config from the merged dictionary
        cfg = ExperimentConfig.from_dict(merged)
        # create the experiment output directory
        exp_dir = experiment_output_dir(cfg)
    except FileNotFoundError as e:
        log.error("context: error loading config=%s: %s", config_path, e)
        raise
    except KeyError as e:
        log.error("context: error loading config=%s: %s", config_path, e)
        raise
    except Exception as e:
        log.error("context: error loading config=%s: %s", config_path, e)
        raise
    

    log.info(
        "context: experiment_id=%s config=%s",
        cfg.experiment_id,
        config_path,
    )
    # create the experiment context
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