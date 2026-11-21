from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from slm_baseline.config.experiment import ExperimentConfig
from slm_baseline.training.dataset import export_sft_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_export_sft_jsonl_line_count(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame(
        [
            {
                "record_key": "k1",
                "document_id": "d1",
                "segment_id": 1,
                "macro_category": "denuncia",
                "sub_category": "cai",
                "pages_list": [1],
                "ocr_surrogates": "testo uno",
            },
            {
                "record_key": "k2",
                "document_id": "d2",
                "segment_id": 2,
                "macro_category": "denuncia",
                "sub_category": "denuncia_altro",
                "pages_list": [1],
                "ocr_surrogates": "testo due",
            },
        ]
    )
    df.to_parquet(data_dir / "tiny.parquet", index=False)

    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    splits = pd.DataFrame(
        {
            "record_key": ["k1", "k2"],
            "document_id": ["d1", "d2"],
            "macro_category": ["denuncia", "denuncia"],
            "sub_category": ["cai", "denuncia_altro"],
            "split": ["train", "val"],
        }
    )
    splits.to_parquet(exp_dir / "splits.parquet", index=False)

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_id": "t",
            "paths": {
                "root": str(ROOT),
                "output_root": str(tmp_path),
                "data_root": str(data_dir),
            },
            "dataset": {"path": "tiny.parquet", "text_column": "ocr_surrogates"},
            "split": {"strategy": "holdout"},
            "prompt": {"version": "minimal"},
            "sequence": {"max_seq_len_mode": "manual", "max_seq_len": 2048},
            "runs": {},
            "sync": {},
            "scope": {"macro_category": "denuncia"},
        }
    )
    train_p, val_p, n_train, n_val = export_sft_jsonl(cfg, exp_dir=exp_dir, prompt_version="minimal")
    assert n_train == 1 and n_val == 1
    train_lines = train_p.read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 1
    rec = json.loads(train_lines[0])
    assert rec["messages"][-1]["role"] == "assistant"
    assert rec["messages"][-1]["content"] == "cai"
    assert val_p.is_file()