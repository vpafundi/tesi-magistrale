"""Non-LLM reference baselines: majority per macro, random among allowed, TF-IDF + LogReg."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...domain.labels import MICRO_PER_MACRO, composite_label


def tfidf_docs(df: pd.DataFrame, macro_scope: str | None) -> pd.Series:
    if macro_scope:
        return df["text"]
    return "macro_" + df["macro_category"] + " " + df["text"]


def tfidf_labels(df: pd.DataFrame, macro_scope: str | None) -> list[str]:
    if macro_scope:
        return df["sub_category"].tolist()
    return [composite_label(m, mi) for m, mi in zip(df["macro_category"], df["sub_category"])]


def masked_predictions_from_proba(
    proba: np.ndarray,
    classes: list[str],
    test_df: pd.DataFrame,
    *,
    macro_scope: str | None,
    train_df: pd.DataFrame,
) -> list[str]:
    preds: list[str] = []
    for macro, p in zip(test_df["macro_category"], proba):
        if macro_scope:
            allowed = list(MICRO_PER_MACRO.get(macro, []))
            cols = [classes.index(c) for c in allowed if c in classes]
            if not cols:
                preds.append(allowed[0] if allowed else train_df["sub_category"].iloc[0])
                continue
            preds.append(classes[cols[int(np.argmax(p[cols]))]])
            continue
        allowed = [composite_label(macro, mi) for mi in MICRO_PER_MACRO[macro]]
        cols = [classes.index(c) for c in allowed if c in classes]
        if not cols:
            preds.append(MICRO_PER_MACRO[macro][0])
            continue
        best = classes[cols[int(np.argmax(p[cols]))]]
        preds.append(best.split("::", 1)[1])
    return preds


def predict_majority(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """Micro più frequente nel train, per macro; fallback sulla prima micro ammessa.

    Args:
        train_df: Colonne ``macro_category``, ``sub_category``.
        test_df: Stesso schema; una riga per segmento da predire.

    Returns:
        Serie di ``sub_category`` predette (nome micro), indice allineato a ``test_df``.
    """
    majority = train_df.groupby("macro_category")["sub_category"].agg(lambda s: s.value_counts().index[0])
    return test_df["macro_category"].map(majority).fillna(
        test_df["macro_category"].map(lambda m: MICRO_PER_MACRO[m][0])
    )


def predict_random(test_df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Campiona uniformemente tra le micro ammesse per la macro di ogni riga.

    Args:
        test_df: Colonna ``macro_category``.
        seed: Seme per :class:`numpy.random.Generator`.

    Returns:
        Serie di micro predette.
    """
    rng = np.random.default_rng(seed)
    return test_df["macro_category"].map(lambda m: rng.choice(MICRO_PER_MACRO[m]))


FIXED_TFIDF_LOGREG_PARAMS = {
    "features": "char35",
    "C": 4.0,
    "class_weight": "balanced",
}


def fit_tfidf_logreg(
    train_df: pd.DataFrame,
    seed: int = 42,
    *,
    macro_scope: str | None = None,
):
    """Fit TF-IDF + LogReg con iperparametri fissi del baseline ``tfidf_logreg``."""
    from .tuning.tfidf_logreg import TfidfLogRegTunable

    tuner = TfidfLogRegTunable(macro_scope=macro_scope, seed=seed, cache_dir=None)
    return tuner.refit(train_df, dict(FIXED_TFIDF_LOGREG_PARAMS))


def predict_tfidf_logreg_with_proba(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    macro_scope: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    from .tuning.tfidf_logreg import TfidfLogRegTunable

    tuner = TfidfLogRegTunable(macro_scope=macro_scope, seed=0, cache_dir=None)
    proba, classes = tuner.predict_proba(model, test_df)
    preds = masked_predictions_from_proba(
        proba, classes, test_df, macro_scope=macro_scope, train_df=train_df
    )
    pred_series = pd.Series(preds, index=test_df.index)
    proba_cols = {f"proba__{c}": proba[:, i] for i, c in enumerate(classes)}
    proba_df = pd.DataFrame(proba_cols, index=test_df.index)
    return pred_series, proba_df


def predict_tfidf_logreg(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
    *,
    macro_scope: str | None = None,
) -> pd.Series:
    """TF-IDF caratteri + regressione logistica; probabilità mascherate sul perimetro macro.

    Args:
        train_df: Colonne ``macro_category``, ``sub_category``, ``text``.
        test_df: Stesso schema per inferenza.
        seed: ``random_state`` del classificatore sklearn.

    Returns:
        Serie di micro predette (solo etichette nel perimetro della macro).
    """
    model = fit_tfidf_logreg(train_df, seed, macro_scope=macro_scope)
    pred, _ = predict_tfidf_logreg_with_proba(model, train_df, test_df, macro_scope=macro_scope)
    return pred
