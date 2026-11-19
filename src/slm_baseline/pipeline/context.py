"""Stato condiviso tra gli stage della pipeline baseline."""

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
    """Metriche riassuntive emesse dalla fase prepare.

    Attributes:
        n_segments: Segmenti in perimetro dopo filtro e sample.
        n_splits_rows: Righe nella tabella split (holdout o k-fold).
        n_empty_text_dropped: Righe scartate per testo OCR vuoto.
        max_seq_len: Budget token risolto per SLM (da ``run_config``).
        splits_hash: Hash della tabella split (confrontabilità tra run).
        dataset_path: Percorso assoluto del parquet usato.
    """

    n_segments: int = 0
    n_splits_rows: int = 0
    n_empty_text_dropped: int = 0
    max_seq_len: int = 0
    splits_hash: str = ""
    dataset_path: Path | None = None


@dataclass
class ExperimentContext:
    """Contesto mutabile per un esperimento da un singolo YAML.

    Viene creato da :func:`build_context` e arricchito dagli stage prepare/evaluate.

    Attributes:
        config_path: File YAML esperimento (path risolto).
        merged: Dizionario YAML dopo ``extends`` ed espansione variabili.
        cfg: Config tipizzata (:class:`~slm_baseline.config.experiment.ExperimentConfig`).
        exp_dir: Directory radice ``outputs/experiments/<experiment_id>/``.
        paths: Percorsi standard (manifest, splits, runs, …).
        config_hash: Hash canonico della config per immutabilità.
        force: Se True, prepare sovrascrive anche con hash diverso.
        rerun: Se True, le valutazioni non saltano metriche esistenti.
        dry_run_slm: Se True, ``slm`` stampa solo i comandi subprocess.
        segments: DataFrame segmenti curati (dopo prepare, load).
        split_table: Tabella assegnazione split o fold.
        run_config: Dict scritto in ``run_config.json`` (max_seq_len, limit, …).
        manifest: Dict scritto in ``manifest.json``.
        prepare_stats: Contatori aggiornati durante prepare.
        logger: Logger condiviso (impostato da orchestrator o ``build_context``).
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
    logger: logging.Logger = field(default_factory=get_pipeline_logger)


@dataclass
class EvalFoldContext:
    """Frame train/test per un fold k-fold o per holdout.

    Attributes:
        fold_id: Indice fold ``0..k-1``, oppure ``None`` in holdout.
        df: Segmenti con colonna ``split`` già applicata.
        train: Righe con ``split == \"train\"``.
        test: Righe dello split di valutazione (es. ``test``).
    """

    fold_id: int | None
    df: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
