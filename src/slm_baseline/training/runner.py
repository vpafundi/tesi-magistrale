"""Subprocess Unsloth per esperimento macro-scoped."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from ..artifacts.io import load_json, write_json
from ..artifacts.model_card import build_lora_model_card, write_lora_model_card
from ..evaluation.classic.persistence import lora_adapter_ready
from ..logging.core import get_pipeline_logger
from ..pipeline.context import ExperimentContext
from ..pipeline.exceptions import PrepareArtifactsMissingError, TrainRunnerError
from .dataset import export_sft_jsonl


def train_slug(base_model: str) -> str:
    return f"{base_model.split('/')[-1]}__lora"


def _emit_lora_model_card(
    ctx: ExperimentContext,
    *,
    slug: str,
    base_model: str,
    adapter_dir: Path,
    metrics_path: Path,
    run_cfg: dict,
) -> None:
    if not metrics_path.is_file():
        return
    training_payload = load_json(metrics_path)
    manifest = load_json(ctx.paths.manifest) if ctx.paths.manifest.is_file() else {}
    u = ctx.cfg.training.unsloth
    card = build_lora_model_card(
        experiment_id=ctx.cfg.experiment_id,
        slug=slug,
        base_model=base_model,
        adapter_dir=adapter_dir,
        training_payload=training_payload if isinstance(training_payload, dict) else {},
        manifest=manifest,
        run_cfg=run_cfg,
        description=ctx.cfg.meta.description,
        unsloth_cfg={
            "lora_r": u.lora_r,
            "lora_alpha": u.lora_alpha,
            "epochs": u.epochs,
            "batch_size": u.batch_size,
            "grad_accum": u.grad_accum,
            "learning_rate": u.learning_rate,
            "early_stopping_patience": u.early_stopping_patience,
            "packing": u.packing,
        },
    )
    training_dir = ctx.exp_dir / "training"
    write_lora_model_card(adapter_dir, training_dir, slug, card)
    ctx.logger.info("train: model_card %s", adapter_dir / "MODEL_CARD.md")


class SubprocessTrainRunner:
    def __init__(self, python: str | None = None, logger: logging.Logger | None = None) -> None:
        self.python = python or sys.executable
        self.logger = logger or get_pipeline_logger()

    def launch(self, cmd: list[str], *, packing: bool = False, dry_run: bool = False) -> None:
        if dry_run:
            self.logger.info("train dry-run: %s", " ".join(cmd))
            return
        self.logger.info("train launch: packing=%s %s", packing, " ".join(cmd))
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if not packing:
            env["UNSLOTH_DISABLE_AUTO_PADDING_FREE"] = "1"
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, check=False, env=env)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            self.logger.error("train failed: exit=%s after %.1fs", proc.returncode, elapsed)
            raise TrainRunnerError(f"run_unsloth failed: {' '.join(cmd)}", proc.returncode)
        self.logger.info("train subprocess finished in %.1fs", elapsed)


class TrainLoRAPipeline:
    def __init__(self, runner: SubprocessTrainRunner | None = None) -> None:
        self.runner = runner or SubprocessTrainRunner()

    def run(self, ctx: ExperimentContext) -> Path:
        if not ctx.cfg.training.enabled:
            ctx.logger.info("train: skipped (training.enabled=false)")
            return ctx.exp_dir
        if not ctx.paths.run_config.is_file():
            raise PrepareArtifactsMissingError("Esegui prima prepare (manca run_config.json).")
        run_cfg = load_json(ctx.paths.run_config)
        base_model = ctx.cfg.training.base_model or (
            ctx.cfg.runs.slm[0]["model"] if ctx.cfg.runs.slm else ""
        )
        if not base_model:
            raise TrainRunnerError("training.base_model o runs.slm[0].model richiesto")

        prompt_version = str(run_cfg.get("prompt_version", ctx.cfg.prompt.version))
        samp = ctx.cfg.training.sampling
        if samp.enabled:
            ctx.logger.info(
                "train: sampling enabled max_altro=%s min_non_altro=%s seed=%s",
                samp.max_altro,
                samp.min_non_altro,
                samp.seed,
            )
        train_p, val_p, n_train, n_val = export_sft_jsonl(ctx.cfg, exp_dir=ctx.exp_dir, prompt_version=prompt_version)
        ctx.logger.info("train: jsonl train=%s val=%s rows=%s/%s", train_p, val_p, n_train, n_val)

        slug = train_slug(base_model)
        adapter_dir = ctx.exp_dir / ctx.cfg.training.output_subdir / slug
        metrics_path = ctx.exp_dir / "training" / f"metrics__{slug}.json"
        adapter_ok = lora_adapter_ready(adapter_dir)
        if not ctx.rerun and (metrics_path.is_file() or adapter_ok):
            reason = "metrics exist" if metrics_path.is_file() else "adapter on disk"
            ctx.logger.info("train: skip %s (%s)", slug, reason)
            if adapter_ok:
                manifest = load_json(ctx.paths.manifest) if ctx.paths.manifest.is_file() else {}
                manifest["adapter_path"] = str(adapter_dir)
                manifest["training_slug"] = slug
                write_json(ctx.paths.manifest, manifest)
            from ..artifacts.model_card import lora_model_card_json

            if not lora_model_card_json(adapter_dir).is_file() and metrics_path.is_file():
                _emit_lora_model_card(
                    ctx, slug=slug, base_model=base_model, adapter_dir=adapter_dir,
                    metrics_path=metrics_path, run_cfg=run_cfg,
                )
            return adapter_dir

        if ctx.rerun and adapter_dir.is_dir():
            import shutil

            shutil.rmtree(adapter_dir)
            ctx.logger.info("train: cleared adapter dir for rerun %s", adapter_dir)

        max_seq = ctx.cfg.training.unsloth.max_seq_len or int(run_cfg["max_seq_len"])
        u = ctx.cfg.training.unsloth
        cmd = [
            self.runner.python,
            "-u",
            "-m",
            "slm_baseline.training.run_unsloth",
            "--base-model",
            base_model,
            "--train-jsonl",
            str(train_p),
            "--val-jsonl",
            str(val_p),
            "--output-dir",
            str(adapter_dir),
            "--max-seq-len",
            str(max_seq),
            "--lora-r",
            str(u.lora_r),
            "--lora-alpha",
            str(u.lora_alpha),
            "--epochs",
            str(u.epochs),
            "--batch-size",
            str(u.batch_size),
            "--grad-accum",
            str(u.grad_accum),
            "--learning-rate",
            str(u.learning_rate),
            "--early-stopping-patience",
            str(u.early_stopping_patience),
            "--metrics-out",
            str(metrics_path),
        ]
        if u.packing:
            cmd.append("--packing")
        self.runner.logger = ctx.logger
        self.runner.launch(cmd, packing=u.packing, dry_run=ctx.dry_run_slm)

        manifest = load_json(ctx.paths.manifest) if ctx.paths.manifest.is_file() else {}
        manifest["adapter_path"] = str(adapter_dir)
        manifest["training_slug"] = slug
        write_json(ctx.paths.manifest, manifest)
        _emit_lora_model_card(
            ctx, slug=slug, base_model=base_model, adapter_dir=adapter_dir,
            metrics_path=metrics_path, run_cfg=run_cfg,
        )
        return adapter_dir