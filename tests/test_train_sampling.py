from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from slm_baseline.config.experiment import TrainSamplingConfig
from slm_baseline.training.sampling import apply_train_sampling


def test_cap_altro() -> None:
    df = pd.DataFrame(
        {
            "sub_category": ["documentazione_medica_altro"] * 10 + ["prescrizione_medica"] * 3,
            "text": ["x"] * 13,
        }
    )
    out = apply_train_sampling(
        df,
        TrainSamplingConfig(enabled=True, seed=42, max_altro=4, min_non_altro=None),
    )
    assert len(out[out["sub_category"] == "documentazione_medica_altro"]) == 4
    assert len(out[out["sub_category"] == "prescrizione_medica"]) == 3


def test_oversample_non_altro() -> None:
    df = pd.DataFrame(
        {
            "sub_category": ["certificato_avvenuta_guarigione"] * 2,
            "text": ["a", "b"],
        }
    )
    out = apply_train_sampling(
        df,
        TrainSamplingConfig(enabled=True, seed=0, max_altro=None, min_non_altro=5),
    )
    assert len(out) == 5
    assert set(out["text"]) == {"a", "b"}