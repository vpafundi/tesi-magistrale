"""High-level coordinator for baseline experiment pipelines.

Wires together prepare, classic evaluation, LoRA training, SLM evaluation,
cross-experiment comparison, and S3 sync stages. Used by :mod:`slm_baseline.cli`
and notebooks as the single facade over the per-stage pipeline classes.
"""

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
    """Orchestrates the full baseline workflow from config YAML to artifacts on disk."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize stage pipelines and a shared logger.

        Args:
            logger: Shared logger; if ``None``, created via
                :func:`~slm_baseline.logging.setup_pipeline_logger`.
        """
        self.logger = setup_pipeline_logger(logger)
        self._prepare = PreparePipeline()
        self._classic = ClassicEvaluatePipeline()
        self._slm = SlmEvaluatePipeline()

    def prepare(self, config_path: Path, *, force: bool = False) -> PrepareStats:
        """Materialize ``outputs/experiments/<id>/`` with splits, manifest, and run config.
        This is the first step in the pipeline; all subsequent stages depend on it.
        This stage at first reads the experiment config file and then creates the necessary directories and files.
        It then creates the splits and the manifest.
        Finally, it creates the run config file.
        The run config file is used by the subsequent stages to run the experiments.

        Args:
            config_path: Path to the experiment YAML.
            force: Recreate artifacts even if the experiment directory already exists.

        Returns:
            Summary counts from the prepare stage (e.g. split sizes).
        """
        self.logger.info("prepare: start config=%s force=%s", config_path, force)
        ctx = build_context(config_path, force=force, logger=self.logger)
        stats = self._prepare.run(ctx)
        print_prepare_summary(ctx)
        self.logger.info("prepare: done experiment_id=%s", ctx.cfg.experiment_id)
        return stats

    def evaluate_classic(self, config_path: Path, *, rerun: bool = False) -> None:
        """Run CPU baselines listed under ``runs.classic`` in the experiment config.

        Args:
            config_path: Path to the experiment YAML (experiment must already be prepared).
            rerun: Overwrite existing classic run outputs when set.
        """
        self.logger.info("classic: start config=%s rerun=%s", config_path, rerun)
        ctx = build_context(config_path, rerun=rerun, logger=self.logger)
        self._classic.run(ctx)
        self.logger.info("classic: done experiment_id=%s", ctx.cfg.experiment_id)

    def train(self, config_path: Path, *, rerun: bool = False, dry_run: bool = False) -> None:
        """Export SFT JSONL and launch Unsloth LoRA training when ``training.enabled``.

        Args:
            config_path: Path to the experiment YAML.
            rerun: Re-run training even if checkpoints already exist.
            dry_run: Log planned commands without executing training.
        """
        self.logger.info("train: start config=%s rerun=%s dry_run=%s", config_path, rerun, dry_run)
        ctx = build_context(config_path, rerun=rerun, dry_run_slm=dry_run, logger=self.logger)
        run_train_stage(ctx)
        self.logger.info("train: done experiment_id=%s", ctx.cfg.experiment_id)

    def evaluate_slm(self, config_path: Path, *, rerun: bool = False, dry_run: bool = False) -> None:
        """Evaluate configured SLMs via vLLM subprocesses (one per model/mode/fold).

        Args:
            config_path: Path to the experiment YAML.
            rerun: Overwrite existing SLM run outputs when set.
            dry_run: Print subprocess commands without launching vLLM.
        """
        self.logger.info("slm: start config=%s rerun=%s dry_run=%s", config_path, rerun, dry_run)
        ctx = build_context(config_path, rerun=rerun, dry_run_slm=dry_run, logger=self.logger)
        self._slm.run(ctx)
        self.logger.info("slm: done experiment_id=%s", ctx.cfg.experiment_id)

    def run_all(self, config_path: Path, *, force: bool = False, rerun: bool = False) -> None:
        """Run prepare, classic baselines, and SLM evaluation in order (same as ``cli run``).

        Args:
            config_path: Path to the experiment YAML.
            force: Passed to :meth:`prepare`.
            rerun: Passed to classic and SLM evaluation stages.
        """
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
        """Merge predictions across the four macro experiments and write family-level metrics.

        Args:
            slug: Run slug shared by the macro-scoped experiments (e.g. model/mode/length).
            family_prefix: Directory prefix that groups the four macro experiment folders.
            output_root: Base ``outputs`` directory; defaults to ``TESI_OUTPUT_DIR`` or repo ``outputs/``.

        Returns:
            Path to the written merge artifact.
        """
        from ..artifacts.merge import write_family_merge
        from ..config.experiment import repo_root
        import os

        root = output_root or Path(os.environ.get("TESI_OUTPUT_DIR", str(repo_root() / "outputs"))).resolve()
        self.logger.info("compare_family: slug=%s prefix=%s", slug, family_prefix)
        out = write_family_merge(root, slug=slug, family_prefix=family_prefix)
        self.logger.info("compare_family: wrote %s", out)
        return out

    def compare(self, output_root: Path | None = None) -> Path:
        """Aggregate metrics from all experiments into ``comparison/summary.csv``.

        Args:
            output_root: Base ``outputs`` directory; ``None`` uses the compare pipeline default.

        Returns:
            Path to the summary CSV (or primary compare output).
        """
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
        """Upload experiment folder(s) to S3.

        Args:
            config_path: Experiment YAML used to locate a single experiment directory.
            all_experiments: When ``True``, sync every experiment under ``output_root``.
            output_root: Base ``outputs`` directory passed to :class:`SyncPipeline`.

        Raises:
            SyncConfigError: Neither ``config_path`` nor ``all_experiments`` was provided.
        """
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