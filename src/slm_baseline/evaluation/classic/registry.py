"""Registry nomi YAML → funzioni previsionali classiche."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from ...pipeline.exceptions import UnknownClassicMethodError
from .predictors import predict_majority, predict_random, predict_tfidf_logreg

ClassicPredictFn = Callable[[pd.DataFrame, pd.DataFrame, int], pd.Series]


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
        """Esegue il predictor registrato.

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
