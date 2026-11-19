"""Pipeline prepare: materializza directory esperimento e artefatti."""

from __future__ import annotations

import pandas as pd

from ...artifacts.io import load_json, write_json, write_yaml
from ...config.experiment import (
    build_manifest,
    file_sha256,
    resolved_dataset_path,
    splits_hash,
)
from ...data.core import filter_perimeter, load_segments, prepare_text, sample_dataset
from ...data.splits_strategy import split_strategy_for
from ...pipeline.context import ExperimentContext, PrepareStats
from ...pipeline.exceptions import DatasetNotFoundError, ImmutableExperimentError
from ...prompt.core import prompt_footprint
from ...sequence.core import compute_max_seq_len_for_experiment


class GuardImmutabilityStage:
    """Blocca prepare se esiste un manifest con ``config_hash`` diverso."""

    @property
    def name(self) -> str:
        return "guard_immutability"

    def run(self, ctx: ExperimentContext) -> None:
        """Confronta hash config con manifest esistente.

        Args:
            ctx: Deve avere ``config_hash`` e ``paths.manifest``; rispetta ``ctx.force``.

        Returns:
            None.

        Raises:
            ImmutableExperimentError: Hash diverso e ``force`` False.
        """
        mpath = ctx.paths.manifest
        if not mpath.is_file() or ctx.force:
            return
        prev = load_json(mpath).get("config_hash")
        if prev and prev != ctx.config_hash:
            ctx.logger.error(
                "guard_immutability: config_hash conflict prev=%s new=%s",
                prev,
                ctx.config_hash,
            )
            raise ImmutableExperimentError(prev, ctx.config_hash)


class LoadAndCureDatasetStage:
    """Carica parquet, filtra perimetro, campiona e normalizza il testo OCR."""

    @property
    def name(self) -> str:
        return "load_and_cure_dataset"

    def run(self, ctx: ExperimentContext) -> None:
        """Popola ``ctx.segments`` e contatori in ``ctx.prepare_stats``.

        Args:
            ctx: Usa ``ctx.cfg.dataset`` e path risolto del parquet.

        Returns:
            None. Imposta ``ctx.segments``, ``prepare_stats.dataset_path``,
            ``prepare_stats.n_empty_text_dropped``.

        Raises:
            DatasetNotFoundError: File parquet assente.
        """
        ds_path = resolved_dataset_path(ctx.cfg)
        if not ds_path.is_file():
            ctx.logger.error("load_dataset: not found path=%s", ds_path)
            raise DatasetNotFoundError(f"Dataset non trovato: {ds_path}")
        segments, _ = filter_perimeter(load_segments(ds_path))
        samp = ctx.cfg.dataset.sample
        segments = sample_dataset(
            segments, samp.get("n"), samp.get("frac"), seed=int(samp.get("seed", 42))
        )
        segments, dropped = prepare_text(segments, ctx.cfg.dataset.text_column)
        ctx.segments = segments
        ctx.prepare_stats.n_empty_text_dropped = len(dropped)
        ctx.prepare_stats.dataset_path = ds_path
        ctx.logger.info(
            "load_dataset: segments=%s dropped_empty=%s path=%s",
            len(segments),
            len(dropped),
            ds_path,
        )


class AssignSplitStage:
    """Costruisce tabella holdout o k-fold tramite strategia da YAML."""

    @property
    def name(self) -> str:
        return "assign_split"

    def run(self, ctx: ExperimentContext) -> None:
        """Scrive assegnazioni in ``ctx.split_table``.

        Args:
            ctx: Richiede ``ctx.segments`` già popolato.

        Returns:
            None. Aggiorna ``ctx.split_table``, ``prepare_stats.n_segments``,
            ``prepare_stats.n_splits_rows``.

        Raises:
            UnknownSplitStrategyError: Strategia non supportata.
        """
        assert ctx.segments is not None
        strategy = split_strategy_for(ctx.cfg)
        split_tbl = strategy.assign(ctx.segments, ctx.cfg.split)
        ctx.split_table = split_tbl
        ctx.prepare_stats.n_segments = len(ctx.segments)
        ctx.prepare_stats.n_splits_rows = len(split_tbl)
        ctx.logger.info(
            "assign_split: strategy=%s rows=%s",
            ctx.cfg.split.strategy,
            len(split_tbl),
        )


class ComputeSequenceBudgetStage:
    """Risolve ``max_seq_len`` e costruisce il dict ``run_config``."""

    @property
    def name(self) -> str:
        return "compute_sequence_budget"

    def run(self, ctx: ExperimentContext) -> None:
        """Calcola budget token e parametri condivisi per classic/SLM.

        Args:
            ctx: Richiede ``segments`` e ``split_table``.

        Returns:
            None. Imposta ``ctx.run_config`` e ``prepare_stats.max_seq_len``.
        """
        assert ctx.segments is not None and ctx.split_table is not None
        on = ["record_key", "document_id", "macro_category", "sub_category"]
        merged_df = ctx.segments.merge(ctx.split_table, on=on, how="inner", validate="1:1")
        prompt_version = ctx.cfg.prompt.version
        seq = ctx.cfg.sequence
        max_len, dbg = compute_max_seq_len_for_experiment(
            mode=seq.max_seq_len_mode,
            base_max_seq_len=seq.max_seq_len,
            df=merged_df,
            tokenizer_ids=list(seq.tokenizers),
            headroom=seq.headroom,
            multiple=seq.multiple,
            prompt_version=prompt_version,
            token_sample_rows=seq.token_sample_rows,
        )
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
        }
        ctx.prepare_stats.max_seq_len = max_len
        ctx.logger.info("sequence_budget: max_seq_len=%s mode=%s", max_len, seq.max_seq_len_mode)


class PersistPrepareArtifactsStage:
    """Persiste YAML, parquet split, manifest e run_config su disco."""

    @property
    def name(self) -> str:
        return "persist_prepare_artifacts"

    def run(self, ctx: ExperimentContext) -> None:
        """Scrive tutti gli artefatti sotto ``ctx.paths.dir``.

        Args:
            ctx: Richiede ``split_table``, ``run_config`` e campi cfg già validi.

        Returns:
            None. Crea directory, aggiorna ``ctx.manifest`` e ``prepare_stats.splits_hash``.
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
    """Esegue in ordine gli stage 3–7 del piano prepare (dopo load context)."""

    def __init__(self) -> None:
        """Costruisce la lista fissa di stage (immutabilità → persist)."""
        self.stages = [
            GuardImmutabilityStage(),
            LoadAndCureDatasetStage(),
            AssignSplitStage(),
            ComputeSequenceBudgetStage(),
            PersistPrepareArtifactsStage(),
        ]

    def run(self, ctx: ExperimentContext) -> PrepareStats:
        """Esegue tutti gli stage prepare in sequenza.

        Args:
            ctx: Contesto da :func:`~slm_baseline.pipeline.factory.build_context`.

        Returns:
            ``ctx.prepare_stats`` dopo l'ultimo stage.
        """
        for stage in self.stages:
            ctx.logger.info("prepare stage: %s", stage.name)
            stage.run(ctx)
        return ctx.prepare_stats


def print_prepare_summary(ctx: ExperimentContext) -> None:
    """Stampa su stdout il riepilogo (comportamento legacy CLI).

    Args:
        ctx: Dopo :meth:`PreparePipeline.run`.

    Returns:
        None.
    """
    stats = ctx.prepare_stats
    seq_mode = ctx.cfg.sequence.max_seq_len_mode
    log = ctx.logger
    log.info("prepare summary: %s -> %s", ctx.cfg.experiment_id, ctx.paths.dir)
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
