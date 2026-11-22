"""Shared state passed between baseline pipeline stages.

Defines dataclasses that hold experiment configuration, on-disk paths, loaded
data frames, and run flags. :class:`ExperimentContext` is built by
:func:`~slm_baseline.pipeline.factory.build_context` and mutated in place as
prepare and evaluation stages run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts.paths import ExperimentPaths
from ..config.experiment import ExperimentConfig
from ..logging.core import get_pipeline_logger


@dataclass
class PrepareStats:
    """Summary metrics produced by the prepare stage.

    Attributes:
        n_segments: Segments in scope after filtering and sampling.
        n_splits_rows: Rows in the split table (holdout or k-fold).
        n_empty_text_dropped: Rows dropped because OCR text was empty.
        max_seq_len: Resolved token budget for SLM runs (from ``run_config``).
        splits_hash: Hash of the split table for cross-run comparability.
        dataset_path: Absolute path to the segments parquet used.
    """

    n_segments: int = 0
    n_splits_rows: int = 0
    n_empty_text_dropped: int = 0
    max_seq_len: int = 0
    splits_hash: str = ""
    dataset_path: Path | None = None


@dataclass
class ExperimentContext:
    """Mutable carrier for one experiment loaded from a single YAML.

    Created by :func:`~slm_baseline.pipeline.factory.build_context` and filled
    by prepare and evaluation stages (segments, manifest, run_config, etc.).

    Attributes:
        config_path: Resolved path to the experiment YAML.
        merged: YAML dict after ``extends`` merge and variable expansion.
        cfg: Typed config (:class:`~slm_baseline.config.experiment.ExperimentConfig`).
        exp_dir: Root directory ``outputs/experiments/<experiment_id>/``.
        paths: Standard artifact paths (manifest, splits, runs, …).
        config_hash: Canonical config hash for immutability checks.
        force: When ``True``, prepare may overwrite even when the hash differs.
        rerun: When ``True``, evaluation stages re-run instead of skipping existing metrics.
        dry_run_slm: When ``True``, the SLM stage only prints subprocess commands.
        segments: Curated segments dataframe (set after prepare/load).
        split_table: Holdout or k-fold assignment table.
        run_config: Payload written to ``run_config.json`` (max_seq_len, limits, …).
        manifest: Payload written to ``manifest.json``.
        prepare_stats: Counters updated during prepare.
        parent_splits_hash: Hash of parent splits for macro-scoped child experiments, if any.
        logger: Shared logger (set by the orchestrator or ``build_context``).
    """

    config_path: Path
    merged: dict[str, Any]
    cfg: ExperimentConfig
    exp_dir: Path
    paths: ExperimentPaths
    config_hash: str
    force: bool = False
    rerun: bool = False
    dry_run_slm: bool = False
    segments: pd.DataFrame | None = None
    split_table: pd.DataFrame | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    prepare_stats: PrepareStats = field(default_factory=PrepareStats)
    parent_splits_hash: str | None = None
    logger: logging.Logger = field(default_factory=get_pipeline_logger)


@dataclass
class EvalFoldContext:
    """Train and test frames for one k-fold or a single holdout evaluation.

    Attributes:
        fold_id: Fold index ``0 .. k-1``, or ``None`` for holdout.
        df: Segments with the ``split`` column already applied.
        train: Rows where ``split == \"train\"``.
        test: Rows in the evaluation split (e.g. ``test``).
    """

    fold_id: int | None
    df: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame