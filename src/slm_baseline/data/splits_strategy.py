"""Strategie holdout vs k-fold per la tabella split in prepare."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..config.experiment import ExperimentConfig, SplitConfig
from ..pipeline.exceptions import UnknownSplitStrategyError
from .core import make_kfold_splits, make_splits


class SplitAssignmentStrategy(Protocol):
    """Contratto per costruire la tabella assegnazioni da segmenti in perimetro."""

    def assign(self, segments: pd.DataFrame, split_cfg: SplitConfig) -> pd.DataFrame:
        """Args:
            segments: Righe con ``record_key``, ``document_id``, macro/micro.
            split_cfg: Parametri split da YAML (seed, frazioni, k).

        Returns:
            DataFrame da salvare in ``splits.parquet`` (schema dipende dalla strategia).
        """


class HoldoutSplitStrategy:
    """Una sola partizione train/val/test stratificata per gruppo documento."""

    @property
    def name(self) -> str:
        return "holdout"

    def assign(self, segments: pd.DataFrame, split_cfg: SplitConfig) -> pd.DataFrame:
        """Args:
            segments: Segmenti in perimetro.
            split_cfg: ``test_frac``, ``val_frac``, ``seed``.

        Returns:
            Tabella con colonna ``split`` (train/val/test).
        """
        return make_splits(
            segments,
            test_frac=split_cfg.test_frac,
            val_frac=split_cfg.val_frac,
            seed=split_cfg.seed,
        )


class KFoldSplitStrategy:
    """Colonna ``fold`` per riga; train/val/test derivati a valutazione."""

    @property
    def name(self) -> str:
        return "kfold"

    def assign(self, segments: pd.DataFrame, split_cfg: SplitConfig) -> pd.DataFrame:
        """Args:
            segments: Segmenti in perimetro.
            split_cfg: ``k``, ``seed``.

        Returns:
            Tabella con colonna ``fold`` in ``0..k-1``.
        """
        return make_kfold_splits(segments, k=split_cfg.k, seed=split_cfg.seed)


def split_strategy_for(cfg: ExperimentConfig) -> SplitAssignmentStrategy:
    """Factory da ``cfg.split.strategy``.

    Args:
        cfg: Config esperimento completa.

    Returns:
        Istanza :class:`HoldoutSplitStrategy` o :class:`KFoldSplitStrategy`.

    Raises:
        UnknownSplitStrategyError: Valore strategy non riconosciuto.
    """
    strategy = cfg.split.strategy
    if strategy == "holdout":
        return HoldoutSplitStrategy()
    if strategy == "kfold":
        return KFoldSplitStrategy()
    raise UnknownSplitStrategyError(f"split.strategy sconosciuta: {strategy!r}")
