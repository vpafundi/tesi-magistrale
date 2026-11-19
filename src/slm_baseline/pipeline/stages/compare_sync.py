"""Confronto esperimenti e sync opzionale su S3."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ...artifacts.paths import CONFIG_NAME
from ...artifacts.reporting import load_experiments
from ...config.experiment import (
    ExperimentConfig,
    experiment_output_dir,
    load_experiment_config,
    load_merged_experiment_dict,
    repo_root,
)
from ...logging.core import get_pipeline_logger
from ...pipeline.exceptions import SyncConfigError


class ComparePipeline:
    """Aggrega le run di tutti gli esperimenti in un unico CSV."""

    def __init__(
        self,
        output_root: Path | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Args:
            output_root: Directory ``outputs/``; default da ``TESI_OUTPUT_DIR``.
            logger: Logger condiviso con l'orchestrator.
        """
        self.output_root = output_root or Path(
            os.environ.get("TESI_OUTPUT_DIR", str(repo_root() / "outputs"))
        ).resolve()
        self.logger = logger or get_pipeline_logger()

    def run(self) -> Path:
        """Scansiona ``output_root/experiments`` e scrive ``comparison/summary.csv``.

        Returns:
            Percorso assoluto del file CSV creato (anche se tabella vuota).

        Raises:
            OSError: Errori di scrittura su disco.
        """
        table, _ = load_experiments(self.output_root)
        out = self.output_root / "comparison" / "summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(table):
            table.to_csv(out, index=False)
            self.logger.info("compare table:\n%s", table.to_string(index=False))
        else:
            self.logger.info("compare: no runs found under %s", self.output_root)
        self.logger.info("compare: summary %s", out)
        return out


class SyncPipeline:
    """Copia ricorsiva delle cartelle esperimento verso S3 (s3fs)."""

    def __init__(
        self,
        output_root: Path | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Args:
            output_root: Radice locale degli esperimenti.
            logger: Logger condiviso con l'orchestrator.
        """
        self.output_root = output_root or Path(
            os.environ.get("TESI_OUTPUT_DIR", str(repo_root() / "outputs"))
        ).resolve()
        self.logger = logger or get_pipeline_logger()

    def sync_one(self, exp_dir: Path) -> None:
        """Sincronizza una singola cartella ``outputs/experiments/<id>/``.

        Args:
            exp_dir: Deve contenere ``config.resolved.yaml`` con ``sync.s3_prefix``.

        Returns:
            None.

        Raises:
            SyncConfigError: s3fs assente o prefisso S3 vuoto.
        """
        try:
            import s3fs
        except ImportError as e:
            self.logger.error("sync: s3fs not installed")
            raise SyncConfigError("Installa s3fs: uv sync --group sagemaker") from e
        merged = load_merged_experiment_dict(exp_dir / CONFIG_NAME)
        cfg = ExperimentConfig.from_dict(merged)
        pre = (cfg.sync.s3_prefix or "").strip()
        if not pre:
            self.logger.error("sync: empty s3_prefix experiment_id=%s", cfg.experiment_id)
            raise SyncConfigError(
                f"sync.s3_prefix vuoto per {cfg.experiment_id} (imposta TESI_S3_PREFIX o YAML)."
            )
        fs = s3fs.S3FileSystem()
        dest = f"{pre.rstrip('/')}/experiments/{cfg.experiment_id}"
        self.logger.info("sync: %s -> %s", exp_dir, dest)
        fs.put(str(exp_dir), dest, recursive=True)

    def sync_from_config(self, config_path: Path) -> None:
        """Risolve la directory output dal YAML e chiama :meth:`sync_one`.

        Args:
            config_path: File YAML esperimento.

        Returns:
            None.
        """
        cfg = load_experiment_config(config_path)
        self.sync_one(experiment_output_dir(cfg))

    def sync_all(self) -> None:
        """Sincronizza ogni esperimento con ``s3_prefix`` valorizzato.

        Returns:
            None; stampa ``skip`` per esperimenti senza prefisso.

        Raises:
            SyncConfigError: ``output_root/experiments`` assente.
        """
        root = self.output_root / "experiments"
        if not root.is_dir():
            self.logger.error("sync: experiments dir missing %s", root)
            raise SyncConfigError(f"Nessuna cartella: {root}")
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / CONFIG_NAME).is_file():
                merged = load_merged_experiment_dict(d / CONFIG_NAME)
                cfg = ExperimentConfig.from_dict(merged)
                if not (cfg.sync.s3_prefix or "").strip():
                    self.logger.info("sync: skip %s (no s3_prefix)", d.name)
                    continue
                self.sync_one(d)
