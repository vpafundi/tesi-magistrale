"""Prepare stage: materialize the experiment directory and on-disk artifacts.

Runs a fixed sequence of sub-stages on :class:`~slm_baseline.pipeline.context.ExperimentContext`:
immutability guard, dataset load/curation, split assignment, sequence budget, and persistence.
Invoked by :class:`~slm_baseline.pipeline.orchestrator.BaselineExperimentOrchestrator.prepare`.
"""

from __future__ import annotations

import pandas as pd

from ...artifacts.io import load_json, write_json, write_yaml
from ...config.experiment import (
    build_manifest,
    experiment_output_dir_for_id,
    file_sha256,
    resolved_dataset_path,
    splits_hash,
)
from ...data.core import inherit_split_assignments, load_segments_cured
from ...data.splits_strategy import split_strategy_for
from ...pipeline.context import ExperimentContext, PrepareStats
from ...pipeline.exceptions import (
    DatasetNotFoundError,
    EmptyScopeMacroError,
    ImmutableExperimentError,
    MasterSplitsNotFoundError,
)
from ...prompt.core import prompt_footprint
from ...sequence.core import compute_max_seq_len_for_experiment


class GuardImmutabilityStage:
    """Refuse prepare when an existing manifest has a different ``config_hash``."""

    @property
    def name(self) -> str:
        return "guard_immutability"

    def run(self, ctx: ExperimentContext) -> None:
        """Compare the current config hash with any existing manifest.

        Args:
            ctx: Must have ``config_hash`` and ``paths.manifest``; honors ``ctx.force``.

        Raises:
            ImmutableExperimentError: Hashes differ and ``force`` is ``False``.
        """
        mpath = ctx.paths.manifest
        if not mpath.is_file() or ctx.force:
            ctx.logger.info("guard_immutability: manifest not found or force is True")
            return
        prev = load_json(mpath).get("config_hash")
        if prev and prev != ctx.config_hash:
            ctx.logger.error(
                "guard_immutability: config_hash conflict prev=%s new=%s",
                prev,
                ctx.config_hash,
            )
            raise ImmutableExperimentError(prev, ctx.config_hash)
        ctx.logger.info("guard_immutability: manifest found and force is False")

class LoadAndCureDatasetStage:
    """Load the parquet, apply scope filters, sample, and normalize OCR text."""

    @property
    def name(self) -> str:
        return "load_and_cure_dataset"

    def run(self, ctx: ExperimentContext) -> None:
        """Populate ``ctx.segments`` and counters on ``ctx.prepare_stats``.

        Args:
            ctx: Uses ``ctx.cfg.dataset`` and the resolved parquet path.

        Side effects:
            Sets ``ctx.segments``, ``prepare_stats.dataset_path``,
            and ``prepare_stats.n_empty_text_dropped``.

        Raises:
            DatasetNotFoundError: Parquet file is missing.
            EmptyScopeMacroError: Macro scope filter yields no rows.
        """
        ds_path = resolved_dataset_path(ctx.cfg)
        if not ds_path.is_file():
            ctx.logger.error("load_dataset: not found path=%s", ds_path)
            raise DatasetNotFoundError(f"Dataset non trovato: {ds_path}")
        try:
            segments, dropped, _ = load_segments_cured(ctx.cfg)
        except ValueError as e:
            if ctx.cfg.scope.macro_category:
                raise EmptyScopeMacroError(str(e)) from e
            raise
        ctx.segments = segments
        ctx.prepare_stats.n_empty_text_dropped = len(dropped)
        ctx.prepare_stats.dataset_path = ds_path
        ctx.logger.info(
            "load_dataset: segments=%s dropped_empty=%s path=%s scope=%s",
            len(segments),
            len(dropped),
            ds_path,
            ctx.cfg.scope.macro_category,
        )


class AssignSplitStage:
    """Build a holdout or k-fold split table from YAML strategy or parent experiment."""

    @property
    def name(self) -> str:
        return "assign_split"

    def run(self, ctx: ExperimentContext) -> None:
        """Write split assignments to ``ctx.split_table``.

        Args:
            ctx: Requires ``ctx.segments`` to be loaded.

        Side effects:
            Updates ``ctx.split_table``, ``prepare_stats.n_segments``,
            ``prepare_stats.n_splits_rows``, and optionally ``parent_splits_hash``.

        Raises:
            MasterSplitsNotFoundError: Child experiment references a parent without ``splits.parquet``.
            UnknownSplitStrategyError: Propagated when ``split.strategy`` is unsupported.
        """
        assert ctx.segments is not None
        master_id = ctx.cfg.split.assignment_from
        if master_id:
            ctx.logger.info("assign_split: master_id=%s", master_id)
            parent_dir = experiment_output_dir_for_id(ctx.cfg, master_id)
            ctx.logger.info("assign_split: parent_dir=%s", parent_dir)
            parent_splits_path = parent_dir / "splits.parquet"
            if not parent_splits_path.is_file():
                ctx.logger.error("assign_split: parent_splits_path not found=%s", parent_splits_path)
                raise MasterSplitsNotFoundError(
                    f"Manca {parent_splits_path} — esegui prepare su {master_id!r} prima."
                )
            parent_splits = pd.read_parquet(parent_splits_path)
            ctx.parent_splits_hash = splits_hash(parent_splits)
            split_tbl = inherit_split_assignments(ctx.segments, parent_splits)
            ctx.logger.info("assign_split: split_tbl=%s", split_tbl)
        else:
            ctx.logger.info("assign_split: no master_id, using strategy")
            strategy = split_strategy_for(ctx.cfg)
            ctx.logger.info("assign_split: strategy=%s", strategy)
            split_tbl = strategy.assign(ctx.segments, ctx.cfg.split)
            ctx.logger.info("assign_split: split_tbl=%s", split_tbl)
        ctx.split_table = split_tbl
        ctx.prepare_stats.n_segments = len(ctx.segments)
        ctx.prepare_stats.n_splits_rows = len(split_tbl)
        ctx.logger.info(
            "assign_split: strategy=%s rows=%s",
            ctx.cfg.split.strategy,
            len(split_tbl),
        )


class ComputeSequenceBudgetStage:
    """Resolve ``max_seq_len`` and build the shared ``run_config`` dict."""

    @property
    def name(self) -> str:
        return "compute_sequence_budget"

    def run(self, ctx: ExperimentContext) -> None:
        """Compute token budget and run parameters for classic and SLM stages.

        Args:
            ctx: Requires ``segments`` and ``split_table``.

        Side effects:
            Sets ``ctx.run_config`` and ``prepare_stats.max_seq_len``.
        """
        assert ctx.segments is not None and ctx.split_table is not None
        ctx.logger.info("compute_sequence_budget: start")
        # merge the segments and the split table on the 'on' columns
        on = ["record_key", "document_id", "macro_category", "sub_category"]
        ctx.logger.info("compute_sequence_budget: on=%s", on)
        merged_df = ctx.segments.merge(ctx.split_table, on=on, how="inner", validate="1:1")
        prompt_version = ctx.cfg.prompt.version
        ctx.logger.info("compute_sequence_budget: prompt_version=%s", prompt_version)
        seq = ctx.cfg.sequence
        ctx.logger.info("compute_sequence_budget: seq=%s", seq)
        macros = (
            [ctx.cfg.scope.macro_category]
            if ctx.cfg.scope.macro_category
            else sorted(merged_df["macro_category"].unique().tolist())
        )
        ctx.logger.info("compute_sequence_budget: macros=%s", macros)
        max_len, dbg = compute_max_seq_len_for_experiment(
            mode=seq.max_seq_len_mode,
            base_max_seq_len=seq.max_seq_len,
            df=merged_df,
            tokenizer_ids=list(seq.tokenizers),
            headroom=seq.headroom,
            multiple=seq.multiple,
            prompt_version=prompt_version,
            token_sample_rows=seq.token_sample_rows,
            macros=macros,
        )
        ctx.logger.info("compute_sequence_budget: max_len=%s dbg=%s", max_len, dbg)
        ds_path = resolved_dataset_path(ctx.cfg)
        ctx.run_config = {
            "max_seq_len": max_len,
            "max_seq_len_mode": seq.max_seq_len_mode,
            "prompt_version": prompt_version,
            "sequence_debug": dbg,
            "split_strategy": ctx.cfg.split.strategy,
            "eval_split": ctx.cfg.runs.eval_split,
            "limit": ctx.cfg.runs.limit,
            "seed": ctx.cfg.runs.seed,
            "gpu_mem": ctx.cfg.runs.gpu_mem,
            "text_column": ctx.cfg.dataset.text_column,
            "dataset_path": str(ds_path),
            "scope_macro": ctx.cfg.scope.macro_category,
        }

        ctx.prepare_stats.max_seq_len = max_len
        ctx.logger.info("sequence_budget: max_seq_len=%s mode=%s", max_len, seq.max_seq_len_mode)


class PersistPrepareArtifactsStage:
    """Write resolved YAML, split parquet, manifest, and run_config to disk."""

    @property
    def name(self) -> str:
        return "persist_prepare_artifacts"

    def run(self, ctx: ExperimentContext) -> None:
        """Persist all prepare artifacts under ``ctx.paths.dir``.

        Args:
            ctx: Requires ``split_table``, ``run_config``, and valid config fields.

        Side effects:
            Creates the experiment directory, sets ``ctx.manifest`` and ``prepare_stats.splits_hash``.
        """
        assert ctx.split_table is not None
        ds_path = resolved_dataset_path(ctx.cfg)
        dataset_hash = file_sha256(ds_path)
        sh = splits_hash(ctx.split_table)
        ctx.prepare_stats.splits_hash = sh
        manifest = build_manifest(
            ctx.cfg,
            config_hash=ctx.config_hash,
            dataset_path=ds_path,
            dataset_hash=dataset_hash,
            splits_hash_value=sh,
            prompt_hash=prompt_footprint(ctx.cfg.prompt.version),
            prompt_version=ctx.cfg.prompt.version,
            parent_splits_hash=ctx.parent_splits_hash,
        )
        ctx.manifest = manifest
        paths = ctx.paths
        paths.dir.mkdir(parents=True, exist_ok=True)
        write_yaml(paths.config, ctx.merged)
        ctx.split_table.to_parquet(paths.splits, index=False)
        write_json(paths.manifest, manifest)
        write_json(paths.run_config, ctx.run_config)
        ctx.logger.info(
            "persist: dir=%s splits_hash=%s",
            paths.dir,
            sh,
        )


class PreparePipeline:
    """Run prepare sub-stages in order from immutability guard through persistence."""

    def __init__(self) -> None:
        """Build the fixed list of prepare sub-stages."""
        self.stages = [
            GuardImmutabilityStage(),
            LoadAndCureDatasetStage(),
            AssignSplitStage(),
            ComputeSequenceBudgetStage(),
            PersistPrepareArtifactsStage(),
        ]

    def run(self, ctx: ExperimentContext) -> PrepareStats:
        """Execute every prepare sub-stage in sequence.

        Args:
            ctx: Context from :func:`~slm_baseline.pipeline.factory.build_context`.

        Returns:
            ``ctx.prepare_stats`` after the last sub-stage completes.
        """
        for stage in self.stages:
            ctx.logger.info("prepare stage: %s", stage.name)
            stage.run(ctx)
        return ctx.prepare_stats


def print_prepare_summary(ctx: ExperimentContext) -> None:
    """Log a human-readable prepare summary (legacy CLI behavior).

    Args:
        ctx: Context after :meth:`PreparePipeline.run`.
    """
    stats = ctx.prepare_stats
    seq_mode = ctx.cfg.sequence.max_seq_len_mode
    log = ctx.logger
    log.info("prepare summary: %s -> %s", ctx.cfg.experiment_id, ctx.paths.dir)
    if ctx.cfg.meta.description:
        log.info("  description: %s", ctx.cfg.meta.description.replace("\n", " ").strip())
    if ctx.cfg.meta.sagemaker_instance_type:
        log.info("  sagemaker_instance_type: %s", ctx.cfg.meta.sagemaker_instance_type)
    log.info(
        "  config_hash=%s splits_hash=%s max_seq_len=%s (%s)",
        ctx.config_hash,
        stats.splits_hash,
        stats.max_seq_len,
        seq_mode,
    )
    log.info(
        "  n_segments=%s n_splits_rows=%s empty_text_dropped=%s",
        stats.n_segments,
        stats.n_splits_rows,
        stats.n_empty_text_dropped,
    )