from __future__ import annotations

import json
from pathlib import Path

from slm_baseline.artifacts.model_card import (
    build_classic_model_card,
    render_classic_model_card_md,
    write_classic_model_card,
)


def test_classic_model_card_roundtrip(tmp_path: Path) -> None:
    card = build_classic_model_card(
        experiment_id="exp_test",
        slug="tfidf_logreg_tuned",
        method="tfidf_logreg_tuned",
        run_info={
            "scope_macro": "denuncia",
            "split": "test",
            "config_hash": "abc",
            "split_hash": "def",
            "best_params": {"features": "word12", "C": 10.0},
            "classic_model": "/tmp/model.joblib",
        },
        metrics={"n": 100, "accuracy": 0.95, "macro_f1": 0.92},
        manifest={"dataset_sha256": "sha"},
        description="Run di prova",
    )
    jp, mp = write_classic_model_card(tmp_path, "tfidf_logreg_tuned", card)
    assert jp.is_file() and mp.is_file()
    loaded = json.loads(jp.read_text(encoding="utf-8"))
    assert loaded["kind"] == "classic_cpu"
    assert loaded["evaluation"]["macro_f1"] == 0.92
    md = mp.read_text(encoding="utf-8")
    assert "denuncia" in md
    assert render_classic_model_card_md(loaded).startswith("# Model card")