"""Ensure EDA and baseline notebooks can keep historic import paths."""

from __future__ import annotations


def test_data_public_api() -> None:
    from slm_baseline.data import (
        check_no_group_leakage,
        filter_perimeter,
        load_segments,
        normalize_text,
    )

    assert callable(normalize_text)
    assert callable(check_no_group_leakage)


def test_labels_public_api() -> None:
    from slm_baseline.labels import MICRO_PER_MACRO, is_in_perimeter

    assert "denuncia" in MICRO_PER_MACRO
    assert is_in_perimeter("denuncia", "cai")


def test_inference_and_prompts_shim() -> None:
    from slm_baseline.inference import recommend_max_seq_len
    from slm_baseline.prompts import build_messages, prompt_footprint

    assert callable(recommend_max_seq_len)
    msgs = build_messages("denuncia", "testo", prompt_version="minimal")
    assert msgs[-1]["role"] == "user"
    assert prompt_footprint("minimal")


def test_experiment_shim() -> None:
    from slm_baseline.experiment import canonical_config_hash

    assert canonical_config_hash({"a": 1}) == canonical_config_hash({"a": 1})


def test_orchestrator_import() -> None:
    from slm_baseline.pipeline import BaselineExperimentOrchestrator

    assert BaselineExperimentOrchestrator().compare is not None
