"""Non-LLM reference baselines: majority per macro, random among allowed, TF-IDF + LogReg."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...domain.labels import MICRO_PER_MACRO, composite_label


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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def _docs(df: pd.DataFrame) -> pd.Series:
        if macro_scope:
            return df["text"]
        return "macro_" + df["macro_category"] + " " + df["text"]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=300_000)
    x_train = vec.fit_transform(_docs(train_df))
    if macro_scope:
        y_train = train_df["sub_category"].tolist()
    else:
        y_train = [composite_label(m, mi) for m, mi in zip(train_df["macro_category"], train_df["sub_category"])]
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_train)

    proba = clf.predict_proba(vec.transform(_docs(test_df)))
    classes = list(clf.classes_)
    preds = []
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
    return pd.Series(preds, index=test_df.index)
