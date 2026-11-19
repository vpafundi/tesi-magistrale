"""Layout directory esperimento e nomi file standard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.experiment import ExperimentConfig, experiment_output_dir

CONFIG_NAME = "config.resolved.yaml"
MANIFEST_NAME = "manifest.json"
SPLITS_NAME = "splits.parquet"
RUN_CONFIG_NAME = "run_config.json"
RUNS_SUBDIR = "runs"


@dataclass(frozen=True)
class ExperimentPaths:
    """Percorsi sotto ``outputs/experiments/<experiment_id>/``.

    Attributes:
        dir: Radice esperimento.
        config: YAML risolto scritto da prepare.
        manifest: JSON tracciabilità (hash, dataset, prompt).
        splits: Parquet assegnazioni split/fold.
        run_config: JSON parametri condivisi per classic/SLM.
        runs: Sottocartella metriche e predizioni per slug.
    """

    dir: Path
    config: Path
    manifest: Path
    splits: Path
    run_config: Path
    runs: Path


def experiment_paths(cfg: ExperimentConfig) -> ExperimentPaths:
    """Deriva i path standard da :class:`ExperimentConfig`.

    Args:
        cfg: Config con ``experiment_id`` e ``paths.output_root``.

    Returns:
        Bundle :class:`ExperimentPaths` (directory può non esistere ancora).
    """
    exp_dir = experiment_output_dir(cfg)
    return _paths_for_dir(exp_dir)


def experiment_paths_from_dir(exp_dir: Path) -> ExperimentPaths:
    """Costruisce i path da una directory esperimento già esistente.

    Args:
        exp_dir: Es. ``outputs/experiments/smoke_sample150``.

    Returns:
        Stesso schema di :func:`experiment_paths`.
    """
    return _paths_for_dir(exp_dir)


def _paths_for_dir(exp_dir: Path) -> ExperimentPaths:
    return ExperimentPaths(
        dir=exp_dir,
        config=exp_dir / CONFIG_NAME,
        manifest=exp_dir / MANIFEST_NAME,
        splits=exp_dir / SPLITS_NAME,
        run_config=exp_dir / RUN_CONFIG_NAME,
        runs=exp_dir / RUNS_SUBDIR,
    )
