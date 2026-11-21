from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from slm_baseline.config.experiment import ExperimentConfig
from slm_baseline.data.core import inherit_split_assignments
from slm_baseline.pipeline import BaselineExperimentOrchestrator

ROOT = Path(__file__).resolve().parents[1]


def _tiny_parquet(path: Path) -> None:
    rows = []
    for i, macro in enumerate(["denuncia", "documentazione_medica"]):
        for j in range(50):
            rows.append(
                {
                    "record_key": f"{macro}::{j}",
                    "document_id": f"doc_{macro}_{j // 10}",
                    "segment_id": j,
                    "macro_category": macro,
                    "sub_category": (
                        "denuncia_altro" if macro == "denuncia" else "documentazione_medica_altro"
                    ),
                    "pages_list": [1],
                    "ocr_surrogates": f"testo {macro} {j}",
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_scope_filter_and_inherit_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    parquet = data_dir / "tiny.parquet"
    _tiny_parquet(parquet)

    out = tmp_path / "out"
    monkeypatch.setenv("TESI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("TESI_OUTPUT_DIR", str(out))
    monkeypatch.setenv("TESI_REPO_ROOT", str(ROOT))

    master_dir = out / "experiments" / "master_scope"
    master_dir.mkdir(parents=True)
    full = pd.read_parquet(parquet)
    splits = full[["record_key", "document_id", "macro_category", "sub_category"]].copy()
    splits["split"] = "train"
    splits.loc[splits.index[:5], "split"] = "test"
    splits.loc[splits.index[5:8], "split"] = "val"
    splits.to_parquet(master_dir / "splits.parquet", index=False)

    child_yaml = tmp_path / "child.yaml"
    child_yaml.write_text(
        f"""
experiment_id: master_scope__denuncia
extends: null
scope:
  macro_category: denuncia
paths:
  output_root: {out}
  data_root: {data_dir}
dataset:
  path: tiny.parquet
split:
  assignment_from: master_scope
prompt:
  version: minimal
sequence:
  max_seq_len_mode: manual
  max_seq_len: 2048
runs:
  classic: []
training:
  enabled: false
sync:
  s3_prefix: ""
""".strip(),
        encoding="utf-8",
    )

    orch = BaselineExperimentOrchestrator()
    orch.prepare(child_yaml, force=True)
    child_dir = out / "experiments" / "master_scope__denuncia"
    splits = pd.read_parquet(child_dir / "splits.parquet")
    assert (splits["macro_category"] == "denuncia").all()
    manifest = json.loads((child_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope_macro"] == "denuncia"
    assert manifest["parent_splits_hash"]


def test_invalid_macro_rejected() -> None:
    merged = {
        "experiment_id": "x",
        "paths": {"root": str(ROOT), "output_root": str(ROOT / "outputs"), "data_root": str(ROOT / "data")},
        "dataset": {"path": "data/x.parquet"},
        "split": {},
        "prompt": {},
        "sequence": {},
        "runs": {},
        "sync": {},
        "scope": {"macro_category": "non_esiste"},
    }
    with pytest.raises(ValueError, match="scope.macro_category"):
        ExperimentConfig.from_dict(merged)


def test_inherit_split_assignments_raises_on_missing() -> None:
    segments = pd.DataFrame(
        {
            "record_key": ["a"],
            "document_id": ["d1"],
            "macro_category": ["denuncia"],
            "sub_category": ["cai"],
        }
    )
    parent = pd.DataFrame({"record_key": ["b"], "split": ["train"]})
    with pytest.raises(ValueError, match="senza assegnazione"):
        inherit_split_assignments(segments, parent)