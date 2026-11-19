from __future__ import annotations

import os
from pathlib import Path

import pytest

from slm_baseline.experiment import canonical_config_hash, load_merged_experiment_dict

ROOT = Path(__file__).resolve().parents[1]


def test_load_extends_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("a:\n  x: 1\n  y: ${MYVAR}\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text('experiment_id: t\nextends: base.yaml\na:\n  z: 2\n', encoding="utf-8")

    monkeypatch.setenv("MYVAR", "hello")
    monkeypatch.setenv("TESI_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("TESI_DATA_DIR", str(ROOT / "data"))
    monkeypatch.setenv("TESI_OUTPUT_DIR", str(tmp_path / "out"))

    d = load_merged_experiment_dict(child)
    assert d["experiment_id"] == "t"
    assert d["a"]["x"] == 1
    assert d["a"]["y"] == "hello"
    assert d["a"]["z"] == 2


def test_canonical_config_hash_stable() -> None:
    cfg1 = {"b": 2, "a": {"z": 1, "y": 2}}
    cfg2 = {"a": {"y": 2, "z": 1}, "b": 2}
    assert canonical_config_hash(cfg1) == canonical_config_hash(cfg2)
