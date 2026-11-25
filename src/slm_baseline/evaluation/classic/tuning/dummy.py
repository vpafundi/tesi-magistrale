"""Minimal tunable method for engine smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier


class DummyTunable:
    name = "dummy_tunable"

    def suggest_params(self, trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
        return {"strategy": trial.suggest_categorical("strategy", ["most_frequent", "stratified"])}

    def fit(self, train_df: pd.DataFrame, params: dict[str, Any], *, fold_key: str) -> DummyClassifier:
        y = train_df["sub_category"]
        clf = DummyClassifier(strategy=params["strategy"])
        clf.fit(np.zeros((len(y), 1)), y)
        return clf

    def predict_proba(self, model: DummyClassifier, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        x = np.zeros((len(df), 1))
        return model.predict_proba(x), list(model.classes_)

    def refit(self, train_df: pd.DataFrame, params: dict[str, Any], *, fold_key: str = "refit") -> DummyClassifier:
        return self.fit(train_df, params, fold_key=fold_key)