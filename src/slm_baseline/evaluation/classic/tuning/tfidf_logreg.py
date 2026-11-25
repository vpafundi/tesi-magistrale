"""TF-IDF + LogisticRegression adapter for Optuna tuning."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from ..predictors import tfidf_docs, tfidf_labels

FEATURE_PRESETS: dict[str, dict[str, Any]] = {
    "char35": dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=300_000),
    "char36": dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500_000),
    "word12": dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=300_000),
}


def build_vectorizer(feature_name: str) -> TfidfVectorizer | FeatureUnion:
    if "+" in feature_name:
        parts = feature_name.split("+")
        transformers = []
        for p in parts:
            p = p.strip()
            if p not in FEATURE_PRESETS:
                raise ValueError(f"Unknown feature preset: {p!r}")
            transformers.append((p, TfidfVectorizer(**FEATURE_PRESETS[p])))
        return FeatureUnion(transformers)
    if feature_name not in FEATURE_PRESETS:
        raise ValueError(f"Unknown feature preset: {feature_name!r}")
    return TfidfVectorizer(**FEATURE_PRESETS[feature_name])


@dataclass
class TfidfLogRegModel:
    vectorizer: Any
    clf: LogisticRegression


class FoldMatrixCache:
    """Disk cache for sparse TF-IDF matrices per (fold_key, features)."""

    def __init__(self, cache_dir: Path | None, logger: logging.Logger | None = None) -> None:
        self.cache_dir = cache_dir
        self.logger = logger or logging.getLogger("slm_baseline.pipeline")
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, fold_key: str, features: str, split: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = features.replace("+", "_plus_")
        return self.cache_dir / f"{fold_key}__{safe}__{split}.npz"

    def _vec_joblib(self, fold_key: str, features: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = features.replace("+", "_plus_")
        return self.cache_dir / f"{fold_key}__{safe}__vectorizer.joblib"

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        features: str,
        fold_key: str,
        *,
        macro_scope: str | None,
    ) -> tuple[Any, sparse.csr_matrix, sparse.csr_matrix]:
        xtr_path = self._paths(fold_key, features, "x_train")
        xev_path = self._paths(fold_key, features, "x_eval")
        vec_joblib = self._vec_joblib(fold_key, features)
        if xtr_path is not None and xtr_path.is_file() and xev_path is not None and xev_path.is_file():
            self.logger.info("tuning cache hit fold_key=%s features=%s", fold_key, features)
            import joblib

            x_train = load_npz(xtr_path)
            x_eval = load_npz(xev_path)
            if vec_joblib is not None and vec_joblib.is_file():
                vec = joblib.load(vec_joblib)
            else:
                vec = build_vectorizer(features)
                vec.fit(tfidf_docs(train_df, macro_scope))
            return vec, x_train, x_eval

        t0 = time.perf_counter()
        vec = build_vectorizer(features)
        x_train = vec.fit_transform(tfidf_docs(train_df, macro_scope))
        x_eval = vec.transform(tfidf_docs(eval_df, macro_scope))
        elapsed = time.perf_counter() - t0
        self.logger.info(
            "tuning vectorize fold_key=%s features=%s done %.1fs shape_train=%s shape_eval=%s",
            fold_key,
            features,
            elapsed,
            x_train.shape,
            x_eval.shape,
        )
        if xtr_path is not None and xev_path is not None:
            save_npz(xtr_path, x_train)
            save_npz(xev_path, x_eval)
            import joblib

            if vec_joblib is not None:
                joblib.dump(vec, vec_joblib)
        return vec, x_train, x_eval


class TfidfLogRegTunable:
    name = "tfidf_logreg_tuned"

    def __init__(
        self,
        *,
        macro_scope: str | None,
        seed: int,
        cache_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.macro_scope = macro_scope
        self.seed = seed
        self._cache = FoldMatrixCache(cache_dir, logger=logger)
        self.logger = logger or logging.getLogger("slm_baseline.pipeline")

    def suggest_params(self, trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
        features_list = search_space.get("features") or list(FEATURE_PRESETS.keys())
        cw_list = search_space.get("class_weight") or ["balanced", None]
        cw_choices: list[str] = []
        for c in cw_list:
            cw_choices.append("__none__" if c in (None, "null", "none") else str(c))
        c_cfg = search_space.get("C") or {"low": 0.1, "high": 100.0, "log": True}
        cw_raw = trial.suggest_categorical("class_weight", cw_choices)
        params: dict[str, Any] = {
            "features": trial.suggest_categorical("features", list(features_list)),
            "class_weight": None if cw_raw == "__none__" else cw_raw,
        }
        if c_cfg.get("log", True):
            params["C"] = trial.suggest_float("C", float(c_cfg["low"]), float(c_cfg["high"]), log=True)
        else:
            params["C"] = trial.suggest_float("C", float(c_cfg["low"]), float(c_cfg["high"]))
        return params

    def fit(
        self,
        train_df: pd.DataFrame,
        params: dict[str, Any],
        *,
        fold_key: str,
    ) -> TfidfLogRegModel:
        features = params["features"]
        vec, x_train, _ = self._cache.fit_transform(
            train_df,
            train_df.iloc[:0],
            features,
            fold_key,
            macro_scope=self.macro_scope,
        )
        y_train = tfidf_labels(train_df, self.macro_scope)
        clf = LogisticRegression(
            max_iter=2000,
            C=float(params["C"]),
            class_weight=params.get("class_weight"),
            random_state=self.seed,
        )
        t0 = time.perf_counter()
        clf.fit(x_train, y_train)
        self.logger.info(
            "tuning clf fit fold_key=%s features=%s C=%s cw=%s done %.1fs",
            fold_key,
            features,
            params["C"],
            params.get("class_weight"),
            time.perf_counter() - t0,
        )
        return TfidfLogRegModel(vectorizer=vec, clf=clf)

    def fit_fold_eval(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        params: dict[str, Any],
        *,
        fold_key: str,
    ) -> tuple[TfidfLogRegModel, sparse.csr_matrix]:
        features = params["features"]
        vec, x_train, x_eval = self._cache.fit_transform(
            train_df, eval_df, features, fold_key, macro_scope=self.macro_scope
        )
        y_train = tfidf_labels(train_df, self.macro_scope)
        clf = LogisticRegression(
            max_iter=2000,
            C=float(params["C"]),
            class_weight=params.get("class_weight"),
            random_state=self.seed,
        )
        t0 = time.perf_counter()
        clf.fit(x_train, y_train)
        self.logger.info(
            "tuning clf fit fold_key=%s features=%s C=%s done %.1fs",
            fold_key,
            features,
            params["C"],
            time.perf_counter() - t0,
        )
        return TfidfLogRegModel(vectorizer=vec, clf=clf), x_eval

    def predict_proba(self, model: TfidfLogRegModel, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        x = model.vectorizer.transform(tfidf_docs(df, self.macro_scope))
        proba = model.clf.predict_proba(x)
        return proba, list(model.clf.classes_)

    def predict_proba_from_x(self, model: TfidfLogRegModel, x_eval: sparse.csr_matrix) -> tuple[np.ndarray, list[str]]:
        proba = model.clf.predict_proba(x_eval)
        return proba, list(model.clf.classes_)

    def refit(
        self,
        train_df: pd.DataFrame,
        params: dict[str, Any],
        *,
        fold_key: str = "refit",
    ) -> TfidfLogRegModel:
        features = params["features"]
        t0 = time.perf_counter()
        vec = build_vectorizer(features)
        x_train = vec.fit_transform(tfidf_docs(train_df, self.macro_scope))
        y_train = tfidf_labels(train_df, self.macro_scope)
        clf = LogisticRegression(
            max_iter=2000,
            C=float(params["C"]),
            class_weight=params.get("class_weight"),
            random_state=self.seed,
        )
        clf.fit(x_train, y_train)
        self.logger.info(
            "tuning refit features=%s n=%s done %.1fs shape=%s",
            features,
            len(train_df),
            time.perf_counter() - t0,
            x_train.shape,
        )
        return TfidfLogRegModel(vectorizer=vec, clf=clf)