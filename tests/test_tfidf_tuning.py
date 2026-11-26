from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from slm_baseline.config.experiment import (
    ExperimentConfig,
    canonical_config_hash,
    load_merged_experiment_dict,
    parse_classic_method_entry,
)
from slm_baseline.evaluation.classic.predictors import predict_tfidf_logreg
from slm_baseline.evaluation.classic.tuning import ClassicEvalData, tune_and_predict
from slm_baseline.evaluation.classic.tuning.dummy import DummyTunable
from slm_baseline.evaluation.classic.tuning.tfidf_logreg import (
    TfidfLogRegTunable,
    finalize_trial_params,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_frames(n: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macro = "documentazione_medica"
    subs = [
        "prescrizione_medica",
        "referto_del_pronto_soccorso",
        "certificato_avvenuta_guarigione",
        "lettera_di_dimissioni",
        "documentazione_medica_altro",
    ]
    rows = []
    for i in range(n):
        sub = subs[i % len(subs)]
        text = f"{sub} documento {i} dimissione referto Rp/"
        rows.append(
            {
                "record_key": f"rk_{i}",
                "document_id": f"doc_{i // 3}",
                "macro_category": macro,
                "sub_category": sub,
                "text": text,
            }
        )
    df = pd.DataFrame(rows)
    train = df.iloc[:80].reset_index(drop=True)
    val = df.iloc[80:100].reset_index(drop=True)
    test = df.iloc[100:].reset_index(drop=True)
    return train, val, test


def test_parse_classic_method_entry() -> None:
    assert parse_classic_method_entry("tfidf_logreg").method == "tfidf_logreg"
    assert parse_classic_method_entry("tfidf_logreg").params == {}
    spec = parse_classic_method_entry({"method": "tfidf_logreg_tuned", "tuning": {"n_trials": 3}})
    assert spec.method == "tfidf_logreg_tuned"
    assert spec.params["tuning"]["n_trials"] == 3


def test_holdout_s42_config_hash_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESI_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("TESI_DATA_DIR", str(ROOT / "data"))
    monkeypatch.setenv("TESI_OUTPUT_DIR", str(ROOT / "outputs"))
    path = ROOT / "configs/experiments/holdout_s42_minimal.yaml"
    merged = load_merged_experiment_dict(path)
    h = canonical_config_hash(merged)
    assert len(h) == 16
    cfg = ExperimentConfig.from_dict(merged)
    assert all(isinstance(s.method, str) for s in cfg.runs.classic)


def test_tfidf_logreg_regression_macro_scope() -> None:
    train, _, test = _synthetic_frames()
    macro = "documentazione_medica"
    pred1 = predict_tfidf_logreg(train, test, seed=7, macro_scope=macro)
    pred2 = predict_tfidf_logreg(train, test, seed=7, macro_scope=macro)
    pd.testing.assert_series_equal(pred1, pred2)


def test_dummy_tune_and_predict_val_strategy(tmp_path: Path) -> None:
    train, val, test = _synthetic_frames()
    data = ClassicEvalData(
        train_df=train,
        val_df=val,
        test_df=test,
        seed=42,
        macro_scope="documentazione_medica",
        train_df_for_mask=train,
    )
    method = DummyTunable()
    out = tune_and_predict(
        method,
        data,
        spec_params={
            "tuning": {"strategy": "val", "n_trials": 2, "seed": 42, "refit_on": "train"},
            "search_space": {"strategy": ["most_frequent", "stratified"]},
        },
        artifacts_dir=tmp_path,
        slug="dummy",
        logger=logging.getLogger("test_tfidf_tuning"),
        rerun=True,
    )
    assert len(out.pred) == len(test)
    assert (tmp_path / "tuning__dummy.json").is_file()
    assert (tmp_path / "optuna__dummy.log").is_file()


def test_finalize_trial_params_maps_class_weight_sentinel() -> None:
    assert finalize_trial_params({"class_weight": "__none__", "C": 2.0})["class_weight"] is None
    assert finalize_trial_params({"class_weight": "balanced"})["class_weight"] == "balanced"


def test_tfidf_refit_accepts_optuna_class_weight_sentinel() -> None:
    train, _, _ = _synthetic_frames()
    tuner = TfidfLogRegTunable(macro_scope="documentazione_medica", seed=7)
    model = tuner.refit(
        train,
        {"features": "char35", "C": 1.0, "class_weight": "__none__"},
    )
    assert model.clf is not None


def test_tfidf_tuned_yaml_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESI_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("TESI_DATA_DIR", str(ROOT / "data"))
    monkeypatch.setenv("TESI_OUTPUT_DIR", str(ROOT / "outputs"))
    path = ROOT / "configs/experiments/holdout_curated_v1__documentazione_medica_tfidf_tuned.yaml"
    cfg = ExperimentConfig.from_dict(load_merged_experiment_dict(path))
    assert cfg.training.enabled is False
    assert len(cfg.runs.classic) == 1
    assert cfg.runs.classic[0].method == "tfidf_logreg_tuned"
    assert cfg.runs.slm == []