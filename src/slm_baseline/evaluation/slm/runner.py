"""Schedulazione run SLM via subprocess (un modello = un processo)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from ...artifacts.io import load_json
from ...logging.core import get_pipeline_logger
from ...pipeline.context import ExperimentContext
from ...pipeline.exceptions import PrepareArtifactsMissingError, SlmRunnerError


def slm_slug(model: str, mode: str, max_seq_len: int, fold_id: int | None, *, lora: bool = False) -> str:
    """Identificatore file run per una valutazione vLLM.

    Args:
        model: Id Hugging Face (solo il nome finale entra nello slug).
        mode: ``constrained``, ``free``, o few-shot.
        max_seq_len: Budget token dalla prepare.
        fold_id: Fold k-fold, oppure ``None``.

    Returns:
        Slug senza prefisso ``metrics__``.
    """
    base = model.split("/")[-1]
    lora_sfx = "__lora" if lora else ""
    if fold_id is None:
        return f"{base}__{mode}__L{max_seq_len}{lora_sfx}"
    return f"{base}__{mode}__L{max_seq_len}__fold{fold_id}{lora_sfx}"


class SubprocessVllmRunner:
    """Adapter che invoca ``python -m slm_baseline.run_baseline``."""

    def __init__(self, python: str | None = None, logger: logging.Logger | None = None) -> None:
        """Args:
            python: Interprete da usare; default ``sys.executable``.
            logger: Logger condiviso; default logger pipeline.
        """
        self.python = python or sys.executable
        self.logger = logger or get_pipeline_logger()

    def launch(
        self,
        exp_dir: Path,
        model: str,
        mode: str,
        fold_id: int | None,
        *,
        lora_adapter: str | None = None,
        dry_run: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """Avvia (o stampa) una singola run SLM.

        Args:
            exp_dir: Cartella dopo prepare (contiene ``run_config.json``).
            model: Id modello Hugging Face.
            mode: Modalità inferenza (constrained/free/…).
            fold_id: Richiesto se split strategy è k-fold.
            dry_run: Se True, stampa il comando e non esegue.
            logger: Override logger per questa chiamata (es. da ``ctx.logger``).

        Returns:
            None.

        Raises:
            SlmRunnerError: Exit code subprocess non zero (solo se non dry_run).
        """
        log = logger or self.logger
        cmd = [
            self.python,
            "-u",
            "-m",
            "slm_baseline.run_baseline",
            "--experiment-dir",
            str(exp_dir),
            "--model",
            model,
            "--mode",
            mode,
        ]
        if fold_id is not None:
            cmd.extend(["--fold", str(fold_id)])
        if lora_adapter:
            cmd.extend(["--lora-adapter", lora_adapter])
        if dry_run:
            log.info("slm dry-run: %s", " ".join(cmd))
            return
        log.info("slm launch: %s", " ".join(cmd))
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, check=False, env=env)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            log.error(
                "slm failed: exit=%s after %.1fs cmd=%s",
                proc.returncode,
                elapsed,
                " ".join(cmd),
            )
            raise SlmRunnerError(f"run_baseline failed: {' '.join(cmd)}", proc.returncode)
        log.info(
            "slm finished: model=%s mode=%s fold=%s lora=%s elapsed=%.1fs",
            model,
            mode,
            fold_id,
            lora_adapter or "(none)",
            elapsed,
        )


class SlmEvaluatePipeline:
    """Pianifica tutte le combinazioni modello × modalità × fold da YAML."""

    def __init__(self, runner: SubprocessVllmRunner | None = None) -> None:
        """Args:
            runner: Adapter subprocess; default :class:`SubprocessVllmRunner`.
        """
        self.runner = runner or SubprocessVllmRunner()

    def run(self, ctx: ExperimentContext) -> None:
        """Itera ``runs.slm`` e rispetta skip/resume su metriche esistenti."""
        if not ctx.paths.run_config.is_file():
            ctx.logger.error("slm: missing run_config at %s", ctx.paths.run_config)
            raise PrepareArtifactsMissingError("Esegui prima prepare (manca run_config.json).")
        run_cfg = load_json(ctx.paths.run_config)
        max_seq_len = int(run_cfg["max_seq_len"])
        strategy = ctx.cfg.split.strategy
        exp_dir = ctx.exp_dir
        manifest = load_json(ctx.paths.manifest) if ctx.paths.manifest.is_file() else {}
        default_adapter = manifest.get("adapter_path")
        ctx.logger.info(
            "slm: schedule n_models=%s max_seq_len=%s strategy=%s",
            len(ctx.cfg.runs.slm),
            max_seq_len,
            strategy,
        )
        self.runner.logger = ctx.logger

        def maybe_launch(
            model: str,
            mode: str,
            fold_id: int | None,
            *,
            lora_adapter: str | None = None,
        ) -> None:
            use_lora = bool(lora_adapter)
            slug = slm_slug(model, mode, max_seq_len, fold_id, lora=use_lora)
            if (ctx.paths.runs / f"metrics__{slug}.json").is_file() and not ctx.rerun:
                ctx.logger.info("slm: skip %s (metrics exist)", slug)
                return
            ctx.logger.info(
                "slm: queue slug=%s model=%s mode=%s lora=%s",
                slug,
                model,
                mode,
                lora_adapter or "(none)",
            )
            self.runner.launch(
                exp_dir,
                model,
                mode,
                fold_id,
                lora_adapter=lora_adapter,
                dry_run=ctx.dry_run_slm,
                logger=ctx.logger,
            )

        for spec in ctx.cfg.runs.slm:
            model = spec["model"]
            use_adapter = bool(spec.get("use_adapter", ctx.cfg.training.enabled))
            adapter = str(spec.get("adapter_path") or default_adapter or "") if use_adapter else ""
            adapter = adapter if adapter and Path(adapter).is_dir() else None
            for mode in spec["modes"]:
                if strategy == "kfold":
                    for fold_id in range(ctx.cfg.split.k):
                        maybe_launch(model, mode, fold_id, lora_adapter=adapter)
                else:
                    maybe_launch(model, mode, None, lora_adapter=adapter)
