"""Metrics shared by baseline and post-fine-tuning evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ..domain.labels import INVALID_LABEL, MICRO_PER_MACRO, composite_label


def _composite(macro: pd.Series, micro: pd.Series) -> np.ndarray:
    return np.array(
        [INVALID_LABEL if mi == INVALID_LABEL else composite_label(m, mi) for m, mi in zip(macro, micro)]
    )


def compute_metrics(
    pred_df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """`pred_df` columns: macro_category, sub_category (gold), pred (micro or INVALID_LABEL).

    Macro-F1 averages over the gold classes present, so an absent class does not count as 0.
    """
    y_true = _composite(pred_df["macro_category"], pred_df["sub_category"])
    y_pred = _composite(pred_df["macro_category"], pred_df["pred"])
    labels = sorted(set(y_true))

    def _scores(t: np.ndarray, p: np.ndarray) -> tuple[float, float]:
        return (
            accuracy_score(t, p),
            f1_score(t, p, labels=sorted(set(t)), average="macro", zero_division=0),
        )

    acc, macro_f1 = _scores(y_true, y_pred)
    rng = np.random.default_rng(seed)
    boot = np.array(
        [_scores(y_true[i], y_pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot))]
    )

    per_macro = {}
    for macro in MICRO_PER_MACRO:
        mask = pred_df["macro_category"].to_numpy() == macro
        if not mask.any():
            continue
        per_macro[macro] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "macro_f1": float(
                f1_score(y_true[mask], y_pred[mask], labels=sorted(set(y_true[mask])), average="macro", zero_division=0)
            ),
        }

    return {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "accuracy_ci95": [float(x) for x in np.percentile(boot[:, 0], [2.5, 97.5])],
        "macro_f1": float(macro_f1),
        "macro_f1_ci95": [float(x) for x in np.percentile(boot[:, 1], [2.5, 97.5])],
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "invalid_rate": float((y_pred == INVALID_LABEL).mean()),
        "per_macro": per_macro,
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def confusion_for_macro(pred_df: pd.DataFrame, macro: str) -> pd.DataFrame:
    sub = pred_df[pred_df["macro_category"] == macro]
    labels = MICRO_PER_MACRO[macro] + ([INVALID_LABEL] if (sub["pred"] == INVALID_LABEL).any() else [])
    cm = confusion_matrix(sub["sub_category"], sub["pred"], labels=labels)
    return pd.DataFrame(cm, index=pd.Index(labels, name="gold"), columns=pd.Index(labels, name="pred"))


def summary_row(name: str, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "run": name,
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "acc_lo": metrics["accuracy_ci95"][0],
        "acc_hi": metrics["accuracy_ci95"][1],
        "macro_f1": metrics["macro_f1"],
        "f1_lo": metrics["macro_f1_ci95"][0],
        "f1_hi": metrics["macro_f1_ci95"][1],
        "invalid_rate": metrics["invalid_rate"],
        **extra,
    }
