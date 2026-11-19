"""Pipeline valutazione baseline classiche (CPU)."""

from __future__ import annotations

import pandas as pd

from ...artifacts.io import load_json
from ...artifacts.reporting import save_run
from ...config.experiment import resolved_dataset_path
from ...data.core import (
    apply_kfold_assignments,
    attach_splits,
    filter_perimeter,
    load_segments,
    prepare_text,
    sample_dataset,
)
from ...evaluation.classic.registry import ClassicMethodRegistry
from ...pipeline.context import ExperimentContext
from ...pipeline.exceptions import PrepareArtifactsMissingError
from ...sequence.core import CHAT_TEMPLATE_KWARGS


def classic_slug(method: str, fold_id: int | None) -> str:
    """Nome file run per una baseline classica.

    Args:
        method: Nome in ``runs.classic`` (es. ``tfidf_logreg``).
        fold_id: Indice fold k-fold, oppure ``None`` per holdout.

    Returns:
        Slug senza prefisso ``metrics__`` (es. ``majority__fold2``).
    """
    if fold_id is None:
        return method
    return f"{method}__fold{fold_id}"


class DatasetLoader:
    """Ricarica i segmenti con le stesse regole di prepare (per valutazione)."""

    @staticmethod
    def load_segments_for_eval(ctx: ExperimentContext) -> pd.DataFrame:
        """Applica perimetro, sample e normalizzazione testo come in prepare.

        Args:
            ctx: Config esperimento (dataset, text_column, sample).

        Returns:
            DataFrame pronto per merge con ``splits.parquet``.
        """
        ds_path = resolved_dataset_path(ctx.cfg)
        segments, _ = filter_perimeter(load_segments(ds_path))
        samp = ctx.cfg.dataset.sample
        segments = sample_dataset(
            segments, samp.get("n"), samp.get("frac"), seed=int(samp.get("seed", 42))
        )
        segments, _ = prepare_text(segments, ctx.cfg.dataset.text_column)
        return segments


class ClassicEvaluatePipeline:
    """Esegue ogni metodo in ``runs.classic`` per holdout o per ogni fold k."""

    def __init__(self, registry: ClassicMethodRegistry | None = None) -> None:
        """Args:
            registry: Registry preveditori; default include majority/random/tfidf.
        """
        self.registry = registry or ClassicMethodRegistry()

    def run(self, ctx: ExperimentContext) -> None:
        """Scrive ``predictions__`` e ``metrics__`` sotto ``ctx.paths.runs``.

        Args:
            ctx: Richiede artefatti prepare; rispetta ``ctx.rerun`` per skip.

        Returns:
            None.

        Raises:
            PrepareArtifactsMissingError: Manca ``splits.parquet``.
            UnknownClassicMethodError: Metodo YAML non registrato.
        """
        if not ctx.paths.splits.is_file():
            ctx.logger.error("classic: missing splits at %s", ctx.paths.splits)
            raise PrepareArtifactsMissingError("Esegui prima prepare (manca splits.parquet).")
        run_cfg = load_json(ctx.paths.run_config)
        segments = DatasetLoader.load_segments_for_eval(ctx)
        splits_tbl = pd.read_parquet(ctx.paths.splits)
        eval_split = ctx.cfg.runs.eval_split
        limit = ctx.cfg.runs.limit
        seed = ctx.cfg.runs.seed
        strategy = ctx.cfg.split.strategy
        ctx.logger.info(
            "classic: experiment_id=%s methods=%s strategy=%s",
            ctx.cfg.experiment_id,
            ctx.cfg.runs.classic,
            strategy,
        )

        def one_fold(fold_id: int | None) -> None:
            if strategy == "kfold":
                assert fold_id is not None
                df = apply_kfold_assignments(
                    segments, splits_tbl, fold_id=fold_id, val_frac=ctx.cfg.split.val_frac, seed=ctx.cfg.split.seed
                )
            else:
                df = attach_splits(segments, splits_tbl)

            train_df = df[df["split"] == "train"]
            test_df = df[df["split"] == eval_split].sort_values("record_key").reset_index(drop=True)
            if limit is not None and len(test_df) > limit:
                test_df = test_df.sample(n=min(limit, len(test_df)), random_state=seed).reset_index(drop=True)

            manifest = load_json(ctx.paths.manifest) if ctx.paths.manifest.is_file() else ctx.manifest
            exp_fields = {
                "experiment_id": ctx.cfg.experiment_id,
                "config_hash": manifest.get("config_hash"),
                "split_hash": manifest.get("splits_hash"),
                "prompt_version": run_cfg.get("prompt_version"),
                "fold": fold_id,
                "comparable_group": manifest.get("splits_hash"),
            }

            for method in ctx.cfg.runs.classic:
                slug = classic_slug(method, fold_id)
                metrics_path = ctx.paths.runs / f"metrics__{slug}.json"
                if metrics_path.is_file() and not ctx.rerun:
                    ctx.logger.info("classic: skip %s (metrics exist)", slug)
                    continue
                pred = self.registry.predict(method, train_df, test_df, seed)
                pred_df = test_df[["record_key", "document_id", "macro_category", "sub_category"]].copy()
                pred_df["pred"] = pred.values
                pred_df["raw_output"] = ""
                pred_df["prompt_tokens"] = 0
                pred_df["text_tokens"] = 0
                pred_df["truncated"] = False
                run_info = {
                    "model": None,
                    "mode": f"classic_{method}",
                    "split": eval_split,
                    "max_seq_len": run_cfg.get("max_seq_len"),
                    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                    **exp_fields,
                }
                ctx.paths.runs.mkdir(parents=True, exist_ok=True)
                save_run(ctx.paths.runs, slug, pred_df, run_info)
                ctx.logger.info("classic: done %s", slug)

        if strategy == "kfold":
            for fi in range(ctx.cfg.split.k):
                one_fold(fi)
        else:
            one_fold(None)
