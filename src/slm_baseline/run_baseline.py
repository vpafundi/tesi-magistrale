"""Zero/few-shot baseline for one model with vLLM. One process per model so GPU memory is freed.

    python -m slm_baseline.run_baseline --model Qwen/Qwen3-1.7B --mode constrained \
        --data data/segments.parquet --splits data/splits_v1.parquet --out-dir outputs/baseline

    Esperimento (cartella da `cli prepare`):

    python -m slm_baseline.run_baseline --experiment-dir outputs/experiments/<id> \
        --model Qwen/Qwen3-1.7B --mode constrained
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data import apply_kfold_assignments, attach_splits, filter_perimeter, load_segments, prepare_text
from .logging.core import setup_pipeline_logger
from .inference import CHAT_TEMPLATE_KWARGS, build_prompts, parse_free_output, select_few_shot
from .labels import MICRO_PER_MACRO
from .prompts import PROMPT_VERSION as LEGACY_PROMPT_TAG
from .reporting import save_run

MODES = ("constrained", "free", "fewshot_constrained")

_MANIFEST = "manifest.json"
_RUN_CONFIG = "run_config.json"
_SPLITS = "splits.parquet"
_CONFIG_RESOLVED = "config.resolved.yaml"


def run_slug(model: str, mode: str, max_seq_len: int, fold_id: int | None = None) -> str:
    base = model.split("/")[-1]
    if fold_id is None:
        return f"{base}__{mode}__L{max_seq_len}"
    return f"{base}__{mode}__L{max_seq_len}__fold{fold_id}"


def _sampling_params(allowed: list[str] | None, max_tokens: int, seed: int) -> Any:
    from vllm import SamplingParams

    if allowed is None:
        return SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
            structured_outputs=StructuredOutputsParams(choice=allowed),
        )
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
            guided_decoding=GuidedDecodingParams(choice=allowed),
        )


def _model_revision(model: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import model_info

        return model_info(model, revision=revision).sha
    except Exception:
        return revision


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch
        import vllm

        env["vllm"] = vllm.__version__
        env["torch"] = torch.__version__
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env


class SlmEvaluationJob:
    """Valutazione SLM in-process con vLLM (un modello per processo).

    Entry point Studio e target del subprocess avviato da :class:`~slm_baseline.evaluation.slm.runner.SubprocessVllmRunner`.
    """

    @staticmethod
    def run(argv: list[str] | None = None) -> None:
        """Parsa argv CLI, carica dati/split e scrive metriche sotto ``runs/``.

        Args:
            argv: Argomenti stile ``sys.argv[1:]``; ``None`` usa ``sys.argv``.

        Returns:
            None. Termina con ``SystemExit`` se mancano artefatti prepare.

        Raises:
            SystemExit: Config o path invalidi (comportamento CLI legacy).
        """
        _run_slm_evaluation(argv)


def main(argv: list[str] | None = None) -> None:
    """Alias CLI modulo ``run_baseline``."""
    SlmEvaluationJob.run(argv)


def _run_slm_evaluation(argv: list[str] | None = None) -> None:
    log = setup_pipeline_logger()
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=None, help="Cartella esperimento (dopo cli prepare)")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--mode", choices=MODES, default="constrained")
    p.add_argument("--data", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--split", default=None, help="Split di valutazione (default: test o run_config.eval_split)")
    p.add_argument("--text-column", default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--few-shot-k", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpu-mem", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--fold", type=int, default=None, help="Indice fold (kfold)")
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--rerun", action="store_true")
    args = p.parse_args(argv)
    log.info("run_baseline: model=%s mode=%s experiment_dir=%s", args.model, args.mode, args.experiment_dir)

    exp_dir: Path | None = Path(args.experiment_dir).resolve() if args.experiment_dir else None
    manifest: dict[str, Any] = {}

    if exp_dir is not None:
        mpath = exp_dir / _MANIFEST
        rpath = exp_dir / _RUN_CONFIG
        if not rpath.is_file():
            raise SystemExit(f"Manca {rpath} (esegui: python -m slm_baseline.cli prepare --config ...).")
        with open(rpath, encoding="utf-8") as f:
            run_cfg = json.load(f)
        if mpath.is_file():
            with open(mpath, encoding="utf-8") as f:
                manifest = json.load(f)

        data_path = run_cfg.get("dataset_path")
        if not data_path:
            raise SystemExit("run_config.json senza dataset_path")
        splits_path = str(exp_dir / _SPLITS)
        out_dir = str(exp_dir / "runs")
        text_column = args.text_column or run_cfg.get("text_column") or "ocr_surrogates"
        max_seq_len = int(run_cfg["max_seq_len"] if args.max_seq_len is None else args.max_seq_len)
        limit = run_cfg.get("limit") if args.limit is None else args.limit
        gpu_mem = float(run_cfg.get("gpu_mem", 0.85) if args.gpu_mem is None else args.gpu_mem)
        seed = int(run_cfg.get("seed", 42) if args.seed is None else args.seed)
        eval_split = run_cfg.get("eval_split", "test") if args.split is None else args.split
        prompt_version = run_cfg.get("prompt_version", "minimal") if args.prompt_version is None else args.prompt_version
        strategy = str(run_cfg.get("split_strategy", "holdout"))

        fold_id: int | None = args.fold
        if strategy == "kfold":
            if fold_id is None:
                raise SystemExit("Per split kfold serve --fold <0..k-1>")
            cfg_resolved = exp_dir / _CONFIG_RESOLVED
            if not cfg_resolved.is_file():
                raise SystemExit(f"Manca {cfg_resolved}")
            with open(cfg_resolved, encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}
            val_frac = float(yaml_cfg.get("split", {}).get("val_frac", 0.05))
            split_seed = int(yaml_cfg.get("split", {}).get("seed", 42))
        else:
            fold_id = None
            val_frac = 0.05
            split_seed = 42
    else:
        run_cfg = {}
        if not args.data or not args.splits:
            raise SystemExit("Serve --experiment-dir oppure entrambi --data e --splits")
        data_path = args.data
        splits_path = args.splits
        out_dir = args.out_dir or "outputs/baseline"
        text_column = args.text_column or "ocr_surrogates"
        max_seq_len = 16384 if args.max_seq_len is None else args.max_seq_len
        limit = args.limit
        gpu_mem = 0.85 if args.gpu_mem is None else args.gpu_mem
        seed = 42 if args.seed is None else args.seed
        eval_split = "test" if args.split is None else args.split
        pv = LEGACY_PROMPT_TAG if args.prompt_version is None else args.prompt_version
        prompt_version = "minimal" if pv in ("v1", "minimal") else pv
        strategy = "holdout"
        fold_id = None
        val_frac = 0.05
        split_seed = 42

    segments, _ = filter_perimeter(load_segments(data_path))
    segments, dropped = prepare_text(segments, text_column)
    splits_df = pd.read_parquet(splits_path)

    if exp_dir is not None and strategy == "kfold":
        assert fold_id is not None
        df = apply_kfold_assignments(
            segments,
            splits_df,
            fold_id=fold_id,
            val_frac=val_frac,
            seed=split_seed,
        )
    else:
        df = attach_splits(segments, splits_df)

    eval_df = df[df["split"] == eval_split].sort_values("record_key").reset_index(drop=True)
    if limit:
        eval_df = eval_df.sample(n=min(int(limit), len(eval_df)), random_state=seed).reset_index(drop=True)

    slug = run_slug(args.model, args.mode, max_seq_len, fold_id if strategy == "kfold" else None)
    metrics_path = Path(out_dir) / f"metrics__{slug}.json"
    if metrics_path.is_file() and not args.rerun:
        log.info("run_baseline: skip %s (metrics exist, use --rerun)", slug)
        return

    from vllm import LLM

    llm_kwargs: dict[str, Any] = {}
    if "gemma-3" in args.model.lower():
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0}
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        max_model_len=max_seq_len,
        gpu_memory_utilization=gpu_mem,
        seed=seed,
        enable_prefix_caching=True,
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer()

    few_shot = None
    if args.mode.startswith("fewshot"):
        few_shot = select_few_shot(
            df[df["split"] == "train"],
            tokenizer,
            k=args.few_shot_k,
            seed=seed,
            prompt_version=prompt_version,
        )

    prompts = build_prompts(tokenizer, eval_df, max_seq_len, few_shot=few_shot, prompt_version=prompt_version)
    constrained = args.mode.endswith("constrained")
    params = [
        _sampling_params(MICRO_PER_MACRO[m] if constrained else None, 16 if constrained else 32, seed)
        for m in eval_df["macro_category"]
    ]

    t0 = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompts["prompt_token_ids"]], params, use_tqdm=True
    )
    elapsed = time.perf_counter() - t0

    raw = [o.outputs[0].text for o in outputs]
    pred_df = eval_df[["record_key", "document_id", "macro_category", "sub_category"]].copy()
    pred_df["raw_output"] = raw
    pred_df["pred"] = [parse_free_output(r, m) for r, m in zip(raw, eval_df["macro_category"])]
    pred_df = pred_df.join(prompts[["prompt_tokens", "text_tokens", "truncated"]])

    exp_fields: dict[str, Any] = {}
    if exp_dir is not None and manifest:
        exp_fields = {
            "experiment_id": manifest.get("experiment_id"),
            "config_hash": manifest.get("config_hash"),
            "split_hash": manifest.get("splits_hash"),
            "prompt_version": prompt_version,
            "fold": fold_id if strategy == "kfold" else None,
            "comparable_group": manifest.get("splits_hash"),
        }

    run_info = {
        "model": args.model,
        "revision": _model_revision(args.model, args.revision),
        "mode": args.mode,
        "few_shot_k": args.few_shot_k if few_shot else 0,
        "split": eval_split,
        "splits_file": splits_path,
        "text_column": text_column,
        "max_seq_len": max_seq_len,
        "prompt_version": prompt_version,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "limit": limit,
        "n_eval": len(pred_df),
        "n_dropped_empty_text": int(len(dropped)),
        "truncated_rate": float(pred_df["truncated"].mean()),
        "mean_prompt_tokens": float(pred_df["prompt_tokens"].mean()),
        "elapsed_s": elapsed,
        "examples_per_s": len(pred_df) / elapsed if elapsed else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        **exp_fields,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    metrics = save_run(out_dir, slug, pred_df, run_info)

    eps = run_info["examples_per_s"]
    eps_s = f"{eps:.1f}" if eps is not None else "n/a"
    log.info(
        "%s: acc=%.4f macro_f1=%.4f invalid=%.4f n=%s %s ex/s",
        slug,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["invalid_rate"],
        metrics["n"],
        eps_s,
    )


if __name__ == "__main__":
    main()
