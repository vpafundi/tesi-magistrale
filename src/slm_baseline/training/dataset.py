"""Export JSONL SFT da segmenti + split per training Unsloth."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data.core import attach_splits, load_segments_cured
from ..config.experiment import ExperimentConfig
from ..prompt.core import build_messages


def _rows_to_jsonl(df: pd.DataFrame, prompt_version: str) -> list[dict]:
    records: list[dict] = []
    for _, row in df.iterrows():
        messages = build_messages(
            str(row["macro_category"]),
            str(row["text"]),
            micro=str(row["sub_category"]),
            prompt_version=prompt_version,
        )
        records.append({"messages": messages})
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_sft_jsonl(
    cfg: ExperimentConfig,
    *,
    exp_dir: Path,
    prompt_version: str,
    train_path: Path | None = None,
    val_path: Path | None = None,
) -> tuple[Path, Path, int, int]:
    """Scrive ``training/train.jsonl`` e ``training/val.jsonl`` sotto l'esperimento."""
    segments, _, _ = load_segments_cured(cfg)
    splits = pd.read_parquet(exp_dir / "splits.parquet")
    merged = attach_splits(segments, splits)
    train_df = merged[merged["split"] == "train"]
    val_df = merged[merged["split"] == "val"]
    train_path = train_path or exp_dir / "training" / "train.jsonl"
    val_path = val_path or exp_dir / "training" / "val.jsonl"
    write_jsonl(_rows_to_jsonl(train_df, prompt_version), train_path)
    write_jsonl(_rows_to_jsonl(val_df, prompt_version), val_path)
    return train_path, val_path, len(train_df), len(val_df)