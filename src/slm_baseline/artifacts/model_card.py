"""Model card riassuntive (JSON + Markdown) per artefatti trainati."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def model_card_json_path(directory: Path, slug: str) -> Path:
    return directory / f"model_card__{slug}.json"


def model_card_md_path(directory: Path, slug: str) -> Path:
    return directory / f"model_card__{slug}.md"


def lora_model_card_json(adapter_dir: Path) -> Path:
    return adapter_dir / "MODEL_CARD.json"


def lora_model_card_md(adapter_dir: Path) -> Path:
    return adapter_dir / "MODEL_CARD.md"


def _evaluation_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": metrics.get("n"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
    }
    boot = metrics.get("bootstrap") or {}
    if boot:
        out["macro_f1_bootstrap"] = {
            "mean": boot.get("macro_f1_mean"),
            "ci95_low": boot.get("macro_f1_ci95_low"),
            "ci95_high": boot.get("macro_f1_ci95_high"),
        }
    per_class = metrics.get("per_class") or metrics.get("classification_report")
    if per_class:
        out["per_class"] = per_class
    return {k: v for k, v in out.items() if v is not None}


def build_classic_model_card(
    *,
    experiment_id: str,
    slug: str,
    method: str,
    run_info: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    description: str = "",
) -> dict[str, Any]:
    tuning = run_info.get("tuning") if isinstance(run_info.get("tuning"), dict) else None
    card: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "classic_cpu",
        "experiment_id": experiment_id,
        "slug": slug,
        "method": method,
        "description": description.strip() or None,
        "scope_macro": run_info.get("scope_macro") or manifest.get("scope_macro"),
        "prompt_version": run_info.get("prompt_version"),
        "eval_split": run_info.get("split"),
        "fold": run_info.get("fold"),
        "hyperparameters": run_info.get("best_params") or run_info.get("params"),
        "tuning": tuning,
        "evaluation": _evaluation_summary(metrics),
        "provenance": {
            "config_hash": run_info.get("config_hash") or manifest.get("config_hash"),
            "split_hash": run_info.get("split_hash") or manifest.get("splits_hash"),
            "comparable_group": run_info.get("comparable_group"),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "git_commit": manifest.get("git_commit"),
        },
        "artifacts": {
            "weights": run_info.get("model") or run_info.get("classic_model"),
            "metrics": f"metrics__{slug}.json",
            "predictions": f"predictions__{slug}.parquet",
            "model_meta": f"model__{slug}.json" if run_info.get("classic_model") else None,
        },
        "loaded_from_disk": bool(run_info.get("loaded_from_disk")),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {k: v for k, v in card.items() if v is not None}


def render_classic_model_card_md(card: dict[str, Any]) -> str:
    ev = card.get("evaluation") or {}
    hp = card.get("hyperparameters") or {}
    tuning = card.get("tuning") or {}
    prov = card.get("provenance") or {}
    lines = [
        f"# Model card — {card.get('method')} (`{card.get('slug')}`)",
        "",
        f"**Esperimento:** `{card.get('experiment_id')}`",
    ]
    if card.get("description"):
        lines.extend(["", card["description"].strip()])
    lines.extend(
        [
            "",
            "## Scopo",
            f"- Macro scope: **{card.get('scope_macro') or 'tutte'}**",
            f"- Split di valutazione: **{card.get('eval_split')}**",
            f"- Prompt: **{card.get('prompt_version')}**",
            "",
            "## Iperparametri",
            "```json",
            json.dumps(hp, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    if tuning:
        summary = {
            k: tuning.get(k)
            for k in (
                "cv_macro_f1",
                "cv_macro_f1_std",
                "n_trials",
                "n_completed_trials",
                "strategy",
                "refit_on",
            )
            if tuning.get(k) is not None
        }
        if summary:
            lines.extend(["", "## Tuning (Optuna)", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```"])
    lines.extend(
        [
            "",
            "## Metriche (holdout)",
            f"- **n** = {ev.get('n')}",
            f"- **accuracy** = {ev.get('accuracy')}",
            f"- **macro-F1** = {ev.get('macro_f1')}",
        ]
    )
    if ev.get("macro_f1_bootstrap"):
        b = ev["macro_f1_bootstrap"]
        lines.append(
            f"- macro-F1 bootstrap 95% CI: [{b.get('ci95_low')}, {b.get('ci95_high')}]"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            f"- `config_hash`: `{prov.get('config_hash')}`",
            f"- `split_hash`: `{prov.get('split_hash')}`",
            f"- `comparable_group`: `{prov.get('comparable_group')}`",
        ]
    )
    arts = card.get("artifacts") or {}
    if arts.get("weights"):
        lines.extend(["", "## Artefatti", f"- Pesi: `{arts.get('weights')}`", f"- Metriche: `{arts.get('metrics')}`"])
    lines.append(f"\n_Generato: {card.get('created_at')}_")
    return "\n".join(lines) + "\n"


def write_classic_model_card(directory: Path, slug: str, card: dict[str, Any]) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    jp = model_card_json_path(directory, slug)
    mp = model_card_md_path(directory, slug)
    jp.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(render_classic_model_card_md(card), encoding="utf-8")
    return jp, mp


def write_classic_model_card_from_run(
    runs_dir: Path,
    slug: str,
    *,
    experiment_id: str,
    method: str,
    run_info: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    description: str = "",
) -> tuple[Path, Path]:
    card = build_classic_model_card(
        experiment_id=experiment_id,
        slug=slug,
        method=method,
        run_info=run_info,
        metrics=metrics,
        manifest=manifest,
        description=description,
    )
    artifacts = card.setdefault("artifacts", {})
    jp, mp = write_classic_model_card(runs_dir, slug, card)
    artifacts["model_card_json"] = jp.name
    artifacts["model_card_md"] = mp.name
    jp.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return jp, mp


def build_lora_model_card(
    *,
    experiment_id: str,
    slug: str,
    base_model: str,
    adapter_dir: Path,
    training_payload: dict[str, Any],
    manifest: dict[str, Any],
    run_cfg: dict[str, Any],
    description: str = "",
    unsloth_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "lora_adapter",
        "experiment_id": experiment_id,
        "slug": slug,
        "description": description.strip() or None,
        "base_model": base_model,
        "scope_macro": manifest.get("scope_macro"),
        "prompt_version": run_cfg.get("prompt_version"),
        "training": {
            "train_loss": training_payload.get("train_loss"),
            "global_step": training_payload.get("global_step"),
            "epochs": training_payload.get("epochs"),
            "jsonl_rows_train": training_payload.get("jsonl_rows_train"),
            "sft_train_examples": training_payload.get("sft_train_examples"),
            "max_seq_len": training_payload.get("max_seq_len"),
            "packing": training_payload.get("packing"),
        },
        "hyperparameters": unsloth_cfg,
        "provenance": {
            "config_hash": manifest.get("config_hash"),
            "split_hash": manifest.get("splits_hash"),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "git_commit": manifest.get("git_commit"),
        },
        "artifacts": {
            "adapter_dir": str(adapter_dir),
            "training_metrics": f"training/metrics__{slug}.json",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {k: v for k, v in card.items() if v is not None}


def render_lora_model_card_md(card: dict[str, Any]) -> str:
    tr = card.get("training") or {}
    hp = card.get("hyperparameters") or {}
    lines = [
        f"# Model card — LoRA `{card.get('slug')}`",
        "",
        f"**Esperimento:** `{card.get('experiment_id')}`",
        f"**Base model:** `{card.get('base_model')}`",
    ]
    if card.get("description"):
        lines.extend(["", card["description"].strip()])
    lines.extend(
        [
            "",
            "## Training",
            f"- train loss: **{tr.get('train_loss')}**",
            f"- step: **{tr.get('global_step')}**",
            f"- epoche (config): **{tr.get('epochs')}**",
            f"- esempi SFT: **{tr.get('sft_train_examples')}** (jsonl {tr.get('jsonl_rows_train')})",
            f"- max_seq_len: **{tr.get('max_seq_len')}**",
            f"- packing: **{tr.get('packing')}**",
        ]
    )
    if hp:
        lines.extend(["", "## Iperparametri LoRA / Unsloth", "```json", json.dumps(hp, ensure_ascii=False, indent=2), "```"])
    lines.append(f"\n_Valutazione SLM: metriche in `runs/metrics__*__lora.json`._")
    lines.append(f"\n_Generato: {card.get('created_at')}_")
    return "\n".join(lines) + "\n"


def write_lora_model_card(
    adapter_dir: Path,
    training_dir: Path,
    slug: str,
    card: dict[str, Any],
) -> tuple[Path, Path, Path]:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    adapter_json = lora_model_card_json(adapter_dir)
    adapter_md = lora_model_card_md(adapter_dir)
    train_json = model_card_json_path(training_dir, slug)
    adapter_json.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    adapter_md.write_text(render_lora_model_card_md(card), encoding="utf-8")
    train_json.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return adapter_json, adapter_md, train_json