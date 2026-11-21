"""Load the segment parquet, filter to perimeter, build fixed splits."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from ..config.experiment import ExperimentConfig, ScopeConfig, resolved_dataset_path
from ..domain.labels import is_in_perimeter

REQUIRED_COLUMNS = [
    "record_key",
    "document_id",
    "segment_id",
    "macro_category",
    "sub_category",
    "pages_list",
]

SPLIT_COLUMNS = ["record_key", "document_id", "macro_category", "sub_category", "split"]


def normalize_text(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).replace("\x00", "").replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(re.sub(r"[ \t\f\v]+", " ", line) for line in s.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def load_segments(path: str | Path, *, macro_category: str | None = None) -> pd.DataFrame:
    """Read parquet (local path or s3:// with s3fs) and check schema."""
    path = Path(path)
    filters = [("macro_category", "==", macro_category)] if macro_category is not None else None
    try:
        df = pd.read_parquet(path, filters=filters)
    except Exception:
        df = pd.read_parquet(path)
        if macro_category is not None:
            df = df[df["macro_category"] == macro_category].copy()
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nel parquet: {sorted(missing)}")
    if not df["record_key"].is_unique:
        raise ValueError("record_key non univoco")
    for col in ("macro_category", "sub_category"):
        df[col] = df[col].astype(str).str.strip()
    return df


def apply_scope_filter(df: pd.DataFrame, scope: ScopeConfig) -> pd.DataFrame:
    """Keep rows for ``scope.macro_category`` when set."""
    if not scope.macro_category:
        return df
    out = df[df["macro_category"] == scope.macro_category].copy()
    if len(out) == 0:
        raise ValueError(f"nessun segmento per scope.macro_category={scope.macro_category!r}")
    return out


def load_segments_cured(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load parquet, perimeter, scope, sample, text — same order as prepare/classic."""
    ds_path = resolved_dataset_path(cfg)
    macro = cfg.scope.macro_category
    raw = load_segments(ds_path, macro_category=macro)
    in_perimeter, out_perimeter = filter_perimeter(raw)
    scoped = apply_scope_filter(in_perimeter, cfg.scope)
    samp = cfg.dataset.sample
    sampled = sample_dataset(
        scoped, samp.get("n"), samp.get("frac"), seed=int(samp.get("seed", 42))
    )
    usable, dropped = prepare_text(sampled, cfg.dataset.text_column)
    return usable, dropped, out_perimeter


def inherit_split_assignments(segments: pd.DataFrame, parent_splits: pd.DataFrame) -> pd.DataFrame:
    """Join master split table with scoped segments on ``record_key``."""
    assign_cols = ["record_key"] + [c for c in ("split", "fold") if c in parent_splits.columns]
    parent = parent_splits[assign_cols].drop_duplicates("record_key")
    base_cols = ["record_key", "document_id", "macro_category", "sub_category"]
    seg = segments[base_cols].copy()
    merged = seg.merge(parent, on="record_key", how="inner")
    if len(merged) != len(seg):
        missing = len(seg) - len(merged)
        raise ValueError(f"{missing} segmenti senza assegnazione split nel master")
    return merged


def filter_perimeter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (in_perimeter, out_of_perimeter)."""
    mask = [is_in_perimeter(m, mi) for m, mi in zip(df["macro_category"], df["sub_category"])]
    mask = np.array(mask, dtype=bool)
    return df[mask].copy(), df[~mask].copy()


def prepare_text(df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds `text` (normalized). Returns (usable, dropped_empty_text)."""
    out = df.copy()
    out["text"] = out[text_column].map(normalize_text)
    empty = out["text"].str.len() == 0
    return out[~empty].copy(), out[empty].copy()


def make_splits(
    df: pd.DataFrame,
    test_frac: float = 0.10,
    val_frac: float = 0.05,
    seed: int = 42,
    group_col: str = "document_id",
) -> pd.DataFrame:
    """Stratified by macro::micro, grouped by document (no leakage across splits).

    Build it once on the whole in-perimeter set (independent of the text column)
    and reuse the saved file for baseline, fine-tuning and final evaluation.
    """
    strat = (df["macro_category"] + "::" + df["sub_category"]).to_numpy()
    groups = df[group_col].to_numpy()
    idx = np.arange(len(df))

    def _take(frac: float, sub_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_splits = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rest, held = next(sgkf.split(sub_idx, strat[sub_idx], groups[sub_idx]))
        return sub_idx[rest], sub_idx[held]

    trainval_idx, test_idx = _take(test_frac, idx)
    train_idx, val_idx = _take(val_frac / (1 - test_frac), trainval_idx)

    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    out = df[["record_key", "document_id", "macro_category", "sub_category"]].copy()
    out["split"] = split
    return out


def sample_dataset(
    df: pd.DataFrame,
    n: int | None,
    frac: float | None,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified random sample of rows by macro::micro. Preserves all columns."""
    if n is None and frac is None:
        return df
    strat = (df["macro_category"] + "::" + df["sub_category"]).to_numpy()
    if n is not None:
        n = min(int(n), len(df))
        if n >= len(df):
            return df
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
        train_idx, _ = next(sss.split(np.zeros(len(df)), strat))
        return df.iloc[train_idx].reset_index(drop=True)
    assert frac is not None
    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    train_idx, _ = next(sss.split(np.zeros(len(df)), strat))
    return df.iloc[train_idx].reset_index(drop=True)


def make_kfold_splits(
    df: pd.DataFrame,
    k: int,
    seed: int = 42,
    group_col: str = "document_id",
) -> pd.DataFrame:
    """Each row is assigned exactly one test fold index `fold` in 0..k-1 (StratifiedGroupKFold).

    Train/val for a given fold are derived at runtime with `split_for_kfold_fold`.
    """
    idx = np.arange(len(df))
    strat = (df["macro_category"] + "::" + df["sub_category"]).to_numpy()
    groups = df[group_col].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_col = np.full(len(df), -1, dtype=int)
    for fold_i, (_, test_idx) in enumerate(sgkf.split(idx, strat, groups)):
        fold_col[test_idx] = fold_i
    if (fold_col < 0).any():
        raise RuntimeError("make_kfold_splits: some rows were not assigned a fold")
    out = df[["record_key", "document_id", "macro_category", "sub_category"]].copy()
    out["fold"] = fold_col
    return out


def split_for_kfold_fold(
    splits: pd.DataFrame,
    fold_id: int,
    val_frac: float,
    seed: int = 42,
    group_col: str = "document_id",
) -> pd.Series:
    """Assign train / val / test for one outer fold. `splits` must have column `fold`."""
    if "fold" not in splits.columns:
        raise ValueError("splits must contain a fold column (kfold)")
    df = splits.copy()
    strat = (df["macro_category"] + "::" + df["sub_category"]).to_numpy()
    groups = df[group_col].to_numpy()
    idx = np.arange(len(df))

    def _take(frac: float, sub_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_splits = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rest, held = next(sgkf.split(sub_idx, strat[sub_idx], groups[sub_idx]))
        return sub_idx[rest], sub_idx[held]

    test_mask = df["fold"].to_numpy() == fold_id
    test_idx = idx[test_mask]
    trainval_idx = idx[~test_mask]
    if len(trainval_idx) == 0:
        raise ValueError("empty trainval for fold")
    test_frac_actual = len(test_idx) / len(df)
    train_idx, val_idx = _take(val_frac / (1 - test_frac_actual), trainval_idx)
    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    return pd.Series(split, index=df.index, name="split")


def check_no_group_leakage(splits: pd.DataFrame, group_col: str = "document_id") -> None:
    if "split" not in splits.columns:
        return
    per_group = splits.groupby(group_col)["split"].nunique()
    leaked = per_group[per_group > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} documenti in più split")


def attach_splits(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Merge segment rows with split assignments (train/val/test)."""
    return df.merge(splits[["record_key", "split"]], on="record_key", how="inner", validate="1:1")


def apply_kfold_assignments(
    df: pd.DataFrame,
    splits: pd.DataFrame,
    fold_id: int,
    val_frac: float,
    seed: int = 42,
    group_col: str = "document_id",
) -> pd.DataFrame:
    """Attach `split` column for outer fold `fold_id` (kfold master table + segment features)."""
    if "fold" not in splits.columns:
        raise ValueError("splits must be kfold table with fold column")
    split_series = split_for_kfold_fold(splits, fold_id, val_frac, seed=seed, group_col=group_col)
    wide = splits[["record_key", "document_id", "macro_category", "sub_category"]].copy()
    wide["split"] = split_series.values
    return df.merge(wide[["record_key", "split"]], on="record_key", how="inner", validate="1:1")


def label_distribution(splits: pd.DataFrame) -> pd.DataFrame:
    return (
        splits.groupby(["macro_category", "sub_category", "split"])
        .size()
        .unstack("split", fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )
