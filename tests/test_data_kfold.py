from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slm_baseline.data import apply_kfold_assignments, check_no_group_leakage, make_kfold_splits


def test_kfold_no_document_leakage_per_fold() -> None:
    rows = []
    for i in range(80):
        doc = f"d{i // 4}"
        rows.append(
            {
                "record_key": f"r{i}",
                "document_id": doc,
                "segment_id": i,
                "macro_category": "denuncia",
                "sub_category": "testata",
                "pages_list": [],
            }
        )
    df = pd.DataFrame(rows)
    splits = make_kfold_splits(df, k=4, seed=42)
    for fold in range(4):
        wide = apply_kfold_assignments(df, splits, fold_id=fold, val_frac=0.1, seed=42)
        sub = wide[["record_key", "document_id", "macro_category", "sub_category", "split"]]
        check_no_group_leakage(sub)


def test_holdout_split_checks() -> None:
    df = pd.DataFrame(
        {
            "record_key": ["a", "b"],
            "document_id": ["d1", "d1"],
            "macro_category": ["denuncia", "denuncia"],
            "sub_category": ["testata", "testata"],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(AssertionError):
        check_no_group_leakage(df)
