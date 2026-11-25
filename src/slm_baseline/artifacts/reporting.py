"""One on-disk format for every run (classic, zero-shot SLM, fine-tuned) so they compare directly."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import compute_metrics, summary_row

PRED_COLUMNS = ["record_key", "document_id", "macro_category", "sub_category", "pred"]


def save_run(out_dir: str | Path, slug: str, pred_df: pd.DataFrame, run_info: dict[str, Any]) -> dict[str, Any]:
    missing = set(PRED_COLUMNS) - set(pred_df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nelle predizioni: {sorted(missing)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(pred_df)
    pred_df.to_parquet(out / f"predictions__{slug}.parquet", index=False)
    payload = {
        "run_info": {
            "slug": slug,
            **run_info,
            # Default for older callers / comparable_group
            "comparable_group": run_info.get("comparable_group") or run_info.get("split_hash"),
        },
        "metrics": metrics,
    }
    with open(out / f"metrics__{slug}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return metrics


def load_runs(out_dir: str | Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Returns (summary table sorted by macro-F1, {slug: {"run_info", "metrics"}})."""
    runs: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(out_dir).glob("metrics__*.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        runs[payload["run_info"]["slug"]] = payload
    rows = [
        summary_row(
            slug,
            p["metrics"],
            model=p["run_info"].get("model"),
            mode=p["run_info"].get("mode"),
            max_seq_len=p["run_info"].get("max_seq_len"),
            truncated_rate=p["run_info"].get("truncated_rate"),
            examples_per_s=p["run_info"].get("examples_per_s"),
        )
        for slug, p in runs.items()
    ]
    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return summary, runs


def load_predictions(out_dir: str | Path, slug: str) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / f"predictions__{slug}.parquet")


def per_macro_table(runs: dict[str, dict[str, Any]], metric: str = "macro_f1") -> pd.DataFrame:
    return pd.DataFrame(
        {slug: {m: v[metric] for m, v in p["metrics"]["per_macro"].items()} for slug, p in runs.items()}
    ).T


def load_experiments(output_root: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Scans outputs/experiments/*/runs/metrics__*.json into one flat table.

    comparable_group = split_hash from manifest (same test split -> comparable SLM/classic runs).
    """
    root = Path(output_root) / "experiments"
    rows: list[dict[str, Any]] = []
    raw_payloads: list[dict[str, Any]] = []

    if not root.is_dir():
        return pd.DataFrame(), []

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        runs_dir = exp_dir / "runs"
        if not runs_dir.is_dir():
            continue
        manifest: dict[str, Any] = {}
        mf = exp_dir / "manifest.json"
        if mf.is_file():
            with open(mf, encoding="utf-8") as f:
                manifest = json.load(f)

        for path in sorted(runs_dir.glob("metrics__*.json")):
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            raw_payloads.append(payload)
            ri = payload["run_info"]
            metrics = payload["metrics"]
            comparable = ri.get("comparable_group") or ri.get("split_hash") or manifest.get("splits_hash")
            row: dict[str, Any] = {
                "experiment_id": ri.get("experiment_id") or manifest.get("experiment_id"),
                "description": manifest.get("description"),
                "sagemaker_instance_type": manifest.get("sagemaker_instance_type"),
                "slug": ri.get("slug"),
                "model": ri.get("model"),
                "mode": ri.get("mode"),
                "fold": ri.get("fold"),
                "prompt_version": ri.get("prompt_version") or manifest.get("prompt_version"),
                "max_seq_len": ri.get("max_seq_len"),
                "n": metrics.get("n"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "invalid_rate": metrics.get("invalid_rate"),
                "config_hash": ri.get("config_hash") or manifest.get("config_hash"),
                "split_hash": ri.get("split_hash") or manifest.get("splits_hash"),
                "comparable_group": comparable,
                "truncated_rate": ri.get("truncated_rate"),
                "examples_per_s": ri.get("examples_per_s"),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame(), raw_payloads

    df = pd.DataFrame(rows)
    # K-fold: one row per fold — add aggregated mean ± std (same comparable_group)
    fold_rows = df[df["fold"].notna()]
    if len(fold_rows):
        pat = re.compile(r"__fold\d+$")

        def _base_slug(slug: str) -> str:
            return pat.sub("", slug) if isinstance(slug, str) else slug

        fold_rows = fold_rows.copy()
        fold_rows["_base_slug"] = fold_rows["slug"].map(_base_slug)
        agg_list: list[dict[str, Any]] = []
        for (exp_id, base_s), g in fold_rows.groupby(["experiment_id", "_base_slug"], dropna=False):
            agg_list.append(
                {
                    "experiment_id": exp_id,
                    "slug": f"{base_s}__kfold_agg",
                    "model": g["model"].iloc[0],
                    "mode": g["mode"].iloc[0],
                    "fold": None,
                    "prompt_version": g["prompt_version"].iloc[0],
                    "max_seq_len": g["max_seq_len"].iloc[0],
                    "n": int(g["n"].sum()),
                    "accuracy": float(g["accuracy"].mean()),
                    "macro_f1": float(g["macro_f1"].mean()),
                    "macro_f1_std": float(g["macro_f1"].std(ddof=0)) if len(g) > 1 else 0.0,
                    "invalid_rate": float(g["invalid_rate"].mean()),
                    "config_hash": g["config_hash"].iloc[0],
                    "split_hash": g["split_hash"].iloc[0],
                    "comparable_group": g["comparable_group"].iloc[0],
                    "truncated_rate": float(g["truncated_rate"].mean()) if g["truncated_rate"].notna().any() else None,
                    "examples_per_s": float(g["examples_per_s"].mean()) if g["examples_per_s"].notna().any() else None,
                }
            )
        if agg_list:
            df = pd.concat([df, pd.DataFrame(agg_list)], ignore_index=True)

    df = df.sort_values(["experiment_id", "macro_f1"], ascending=[True, False]).reset_index(drop=True)
    return df, raw_payloads
