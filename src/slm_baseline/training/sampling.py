"""Train-set resampling for SFT (cap ``*_altro``, boost rare micro)."""

from __future__ import annotations

import pandas as pd

from ..config.experiment import TrainSamplingConfig


def _is_altro_sub(sub_category: str) -> bool:
    return sub_category.endswith("_altro")


def apply_train_sampling(df: pd.DataFrame, cfg: TrainSamplingConfig) -> pd.DataFrame:
    if not cfg.enabled or df.empty:
        return df
    parts: list[pd.DataFrame] = []
    for sub, group in df.groupby("sub_category", sort=False):
        g = group
        if _is_altro_sub(str(sub)) and cfg.max_altro is not None and len(g) > cfg.max_altro:
            g = g.sample(n=cfg.max_altro, random_state=cfg.seed)
        elif not _is_altro_sub(str(sub)) and cfg.min_non_altro is not None and len(g) < cfg.min_non_altro:
            g = g.sample(n=cfg.min_non_altro, replace=True, random_state=cfg.seed)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)