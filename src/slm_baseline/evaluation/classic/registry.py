"""Registry nomi YAML → funzioni previsionali classiche."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ...config.experiment import ClassicMethodSpec
from ...pipeline.exceptions import UnknownClassicMethodError
from .predictors import predict_majority, predict_random, predict_tfidf_logreg
from .tuning import ClassicEvalData, tune_and_predict
from .tuning.tfidf_logreg import TfidfLogRegTunable

ClassicPredictFn = Callable[[pd.DataFrame, pd.DataFrame, int], pd.Series]


@dataclass
class ClassicResult:
    pred: pd.Series
    proba_df: pd.DataFrame | None = None
    extra_run_info: dict[str, Any] = field(default_factory=dict)


class ClassicMethodRegistry:
    """Mappa ``runs.classic`` a predictor CPU (majority, random, tfidf_logreg)."""

    def __init__(self) -> None:
        """Registra i tre metodi predefiniti; estendi ``_methods`` per nuove baseline."""
        self._methods: dict[str, ClassicPredictFn] = {
            "majority": lambda train, test, seed: predict_majority(train, test),
            "random": lambda train, test, seed: predict_random(test, seed=seed),
            "tfidf_logreg": lambda train, test, seed: predict_tfidf_logreg(train, test, seed=seed),
        }

    def predict(
        self,
        method: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        seed: int,
        *,
        macro_scope: str | None = None,
    ) -> pd.Series:
        """Esegue il predictor registrato (legacy API).

        Args:
            method: Nome come nel YAML (es. ``random``).
            train: Frame training con colonne macro/micro e ``text``.
            test: Frame test da predire (stesso schema).
            seed: Random seed per metodi stocastici.

        Returns:
            Serie ``pred`` allineata alle righe di ``test``.

        Raises:
            UnknownClassicMethodError: ``method`` assente nel registry.
        """
        fn = self._methods.get(method)
        if fn is None:
            msg = f"Metodo classic sconosciuto: {method!r}"
            logging.getLogger("slm_baseline.pipeline").error("classic registry: %s", msg)
            raise UnknownClassicMethodError(msg)
        if method == "tfidf_logreg":
            return predict_tfidf_logreg(train, test, seed, macro_scope=macro_scope)
        return fn(train, test, seed)

    def run(
        self,
        spec: ClassicMethodSpec,
        data: ClassicEvalData,
        *,
        artifacts_dir: Path,
        cache_root: Path,
        slug: str,
        logger: logging.Logger,
        rerun: bool = False,
    ) -> ClassicResult:
        """Esegue metodo classic semplice o con tuning Optuna."""
        method = spec.method
        if method == "tfidf_logreg_tuned":
            cache_dir = cache_root / "tfidf" / slug
            tuner = TfidfLogRegTunable(
                macro_scope=data.macro_scope,
                seed=data.seed,
                cache_dir=cache_dir,
                logger=logger,
            )
            out = tune_and_predict(
                tuner,
                data,
                spec_params=spec.params,
                artifacts_dir=artifacts_dir,
                slug=slug,
                logger=logger,
                rerun=rerun,
            )
            return ClassicResult(pred=out.pred, proba_df=out.proba_df, extra_run_info=out.extra_run_info)

        if method not in self._methods:
            msg = f"Metodo classic sconosciuto: {method!r}"
            logger.error("classic registry: %s", msg)
            raise UnknownClassicMethodError(msg)
        pred = self.predict(method, data.train_df, data.test_df, data.seed, macro_scope=data.macro_scope)
        return ClassicResult(pred=pred)