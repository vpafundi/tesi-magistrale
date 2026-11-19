from __future__ import annotations

from pathlib import Path

import pytest

from slm_baseline.pipeline.context import PrepareStats
from slm_baseline.pipeline.factory import build_context

ROOT = Path(__file__).resolve().parents[1]


def test_build_context_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = ROOT / "configs/experiments/smoke_sample150.yaml"
    monkeypatch.setenv("TESI_DATA_DIR", str(ROOT / "data"))
    monkeypatch.setenv("TESI_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("TESI_REPO_ROOT", str(ROOT))

    ctx = build_context(cfg_path)
    assert ctx.cfg.experiment_id == "smoke_sample150"
    assert ctx.paths.dir.name == "smoke_sample150"
    assert ctx.config_hash


def test_prepare_stats_defaults() -> None:
    s = PrepareStats()
    assert s.n_segments == 0
    assert s.max_seq_len == 0
