"""Facade per le pipeline baseline (punto d'ingresso CLI e notebook)."""

from __future__ import annotations

import logging
from pathlib import Path

from ..evaluation.slm.runner import SlmEvaluatePipeline
from ..logging.core import setup_pipeline_logger
from .context import PrepareStats
from .exceptions import SyncConfigError
from .factory import build_context
from .stages.classic import ClassicEvaluatePipeline
from .stages.compare_sync import ComparePipeline, SyncPipeline
from .stages.prepare import PreparePipeline, print_prepare_summary
from .stages.train import run_train_stage


class BaselineExperimentOrchestrator:
    """Coordina prepare, valutazione classic/SLM, confronto e sync S3."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Args:
            logger: Logger condiviso; se ``None``, viene creato con :func:`~slm_baseline.logging.setup_pipeline_logger`.
        """
        self.logger = setup_pipeline_logger(logger)
        self._prepare = PreparePipeline()
        self._classic = ClassicEvaluatePipeline()
        self._slm = SlmEvaluatePipeline()

    def prepare(self, config_path: Path, *, force: bool = False) -> PrepareStats:
        """Materializza ``outputs/experiments/<id>/`` (split, manifest, run_config)."""
        self.logger.info("prepare: start config=%s force=%s", config_path, force)
        ctx = build_context(config_path, force=force, logger=self.logger)
        stats = self._prepare.run(ctx)
        print_prepare_summary(ctx)
        self.logger.info("prepare: done experiment_id=%s", ctx.cfg.experiment_id)
        return stats

    def evaluate_classic(self, config_path: Path, *, rerun: bool = False) -> None:
        """Esegue le baseline CPU definite in ``runs.classic``."""
        self.logger.info("classic: start config=%s rerun=%s", config_path, rerun)
        ctx = build_context(config_path, rerun=rerun, logger=self.logger)
        self._classic.run(ctx)
        self.logger.info("classic: done experiment_id=%s", ctx.cfg.experiment_id)

    def train(self, config_path: Path, *, rerun: bool = False, dry_run: bool = False) -> None:
        """Export JSONL e subprocess Unsloth quando ``training.enabled``."""
        self.logger.info("train: start config=%s rerun=%s dry_run=%s", config_path, rerun, dry_run)
        ctx = build_context(config_path, rerun=rerun, dry_run_slm=dry_run, logger=self.logger)
        run_train_stage(ctx)
        self.logger.info("train: done experiment_id=%s", ctx.cfg.experiment_id)

    def evaluate_slm(self, config_path: Path, *, rerun: bool = False, dry_run: bool = False) -> None:
        """Avvia un subprocess vLLM per ogni (modello, modalità, fold)."""
        self.logger.info("slm: start config=%s rerun=%s dry_run=%s", config_path, rerun, dry_run)
        ctx = build_context(config_path, rerun=rerun, dry_run_slm=dry_run, logger=self.logger)
        self._slm.run(ctx)
        self.logger.info("slm: done experiment_id=%s", ctx.cfg.experiment_id)

    def run_all(self, config_path: Path, *, force: bool = False, rerun: bool = False) -> None:
        """Prepare, poi classiche, poi SLM (equivalente a ``cli run``)."""
        self.logger.info("run_all: start config=%s", config_path)
        self.prepare(config_path, force=force)
        self.evaluate_classic(config_path, rerun=rerun)
        self.evaluate_slm(config_path, rerun=rerun)
        self.logger.info("run_all: complete")

    def compare_family(
        self,
        slug: str,
        *,
        family_prefix: str = "holdout_s42",
        output_root: Path | None = None,
    ) -> Path:
        """Unisce predizioni dei 4 esperimenti macro e scrive metriche globali."""
        from ..artifacts.merge import write_family_merge
        from ..config.experiment import repo_root
        import os

        root = output_root or Path(os.environ.get("TESI_OUTPUT_DIR", str(repo_root() / "outputs"))).resolve()
        self.logger.info("compare_family: slug=%s prefix=%s", slug, family_prefix)
        out = write_family_merge(root, slug=slug, family_prefix=family_prefix)
        self.logger.info("compare_family: wrote %s", out)
        return out

    def compare(self, output_root: Path | None = None) -> Path:
        """Aggrega tutte le run in ``comparison/summary.csv``."""
        self.logger.info("compare: start output_root=%s", output_root)
        out = ComparePipeline(output_root, logger=self.logger).run()
        self.logger.info("compare: wrote %s", out)
        return out

    def sync(
        self,
        config_path: Path | None = None,
        *,
        all_experiments: bool = False,
        output_root: Path | None = None,
    ) -> None:
        """Copia cartelle esperimento su S3."""
        self.logger.info(
            "sync: start config=%s all_experiments=%s",
            config_path,
            all_experiments,
        )
        pipe = SyncPipeline(output_root, logger=self.logger)
        if all_experiments:
            pipe.sync_all()
            self.logger.info("sync: all experiments done")
            return
        if config_path is None:
            self.logger.error("sync: missing --config and --all not set")
            raise SyncConfigError("Specifica --config o --all per sync.")
        pipe.sync_from_config(config_path)
        self.logger.info("sync: done for %s", config_path)
