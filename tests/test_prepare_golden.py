from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_baseline.config.experiment import (
    ExperimentConfig,
    canonical_config_hash,
    load_merged_experiment_dict,
    resolved_dataset_path,
)
from slm_baseline.pipeline import BaselineExperimentOrchestrator

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs/experiments/smoke_sample150.yaml"


def test_prepare_manifest_hashes_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setenv("TESI_DATA_DIR", str(ROOT))
    monkeypatch.setenv("TESI_OUTPUT_DIR", str(out))
    monkeypatch.setenv("TESI_REPO_ROOT", str(ROOT))

    merged = load_merged_experiment_dict(SMOKE)
    ds_path = resolved_dataset_path(ExperimentConfig.from_dict(merged))
    if not ds_path.is_file():
        pytest.skip(f"sample parquet missing: {ds_path}")

    expected_cfg_hash = canonical_config_hash(merged)

    orch = BaselineExperimentOrchestrator()
    orch.prepare(SMOKE, force=True)

    exp_dir = out / "experiments" / "smoke_sample150"
    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_hash"] == expected_cfg_hash
    assert manifest["splits_hash"]
    assert (exp_dir / "splits.parquet").is_file()
    assert (exp_dir / "run_config.json").is_file()
