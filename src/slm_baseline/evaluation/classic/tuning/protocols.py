"""Protocols for pluggable classic tuners."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd


class TunableMethod(Protocol):
    name: str

    def suggest_params(self, trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
        """Sample hyperparameters for one Optuna trial."""

    def fit(
        self,
        train_df: pd.DataFrame,
        params: dict[str, Any],
        *,
        fold_key: str,
    ) -> Any:
        """Train on ``train_df``; return opaque model handle."""

    def predict_proba(self, model: Any, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Return (n_samples, n_classes) probabilities and ``classes_`` order."""