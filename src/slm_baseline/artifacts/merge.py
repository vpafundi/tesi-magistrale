"""Unione predizioni da esperimenti macro-scoped."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..domain.labels import MICRO_PER_MACRO
from .metrics import compute_metrics
from .reporting import PRED_COLUMNS

DEFAULT_FAMILY_PREFIX = "holdout_s42"


def macro_experiment_ids(family_prefix: str = DEFAULT_FAMILY_PREFIX) -> dict[str, str]:
    return {macro: f"{family_prefix}__{macro}" for macro in MICRO_PER_MACRO}


def merge_family_predictions(
    output_root: Path,
    *,
    slug: str,
    family_prefix: str = DEFAULT_FAMILY_PREFIX,
    eval_split: str = "test",
) -> tuple[pd.DataFrame, dict]:
    """Concatena ``predictions__<slug>.parquet`` dai quattro esperimenti figli."""
    root = Path(output_root) / "experiments"
    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for macro, exp_id in macro_experiment_ids(family_prefix).items():
        pred_path = root / exp_id / "runs" / f"predictions__{slug}.parquet"
        if not pred_path.is_file():
            missing.append(exp_id)
            continue
        parts.append(pd.read_parquet(pred_path))
    if missing:
        raise FileNotFoundError(f"Predizioni mancanti per slug={slug!r}: {missing}")
    merged = pd.concat(parts, ignore_index=True)
    for col in PRED_COLUMNS:
        if col not in merged.columns:
            raise ValueError(f"Colonna mancante in merge: {col}")
    if not merged["record_key"].is_unique:
        raise ValueError("record_key duplicati dopo merge macro")
    metrics = compute_metrics(merged)
    return merged, metrics


def write_family_merge(
    output_root: Path,
    *,
    slug: str,
    family_prefix: str = DEFAULT_FAMILY_PREFIX,
) -> Path:
    merged, metrics = merge_family_predictions(output_root, slug=slug, family_prefix=family_prefix)
    out_dir = Path(output_root) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"merged__{family_prefix}__{slug}.json"
    out_parquet = out_dir / f"merged__{family_prefix}__{slug}.parquet"
    merged.to_parquet(out_parquet, index=False)
    payload = {"slug": slug, "family_prefix": family_prefix, "n_rows": len(merged), "metrics": metrics}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json