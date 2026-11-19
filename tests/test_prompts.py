from __future__ import annotations

import pytest

from slm_baseline.labels import MICRO_PER_MACRO
from slm_baseline.prompts import build_messages, prompt_footprint, system_prompt_for_macro


@pytest.mark.parametrize("version", ["minimal", "enriched_v1"])
def test_system_prompt_for_every_macro(version: str) -> None:
    for macro in MICRO_PER_MACRO:
        s = system_prompt_for_macro(macro, prompt_version=version)
        assert len(s) > 50


def test_build_messages_user_last(version: str = "minimal") -> None:
    msgs = build_messages("denuncia", "testo", prompt_version=version)
    assert msgs[-1]["role"] == "user"
    assert "denuncia" in msgs[-1]["content"]


def test_prompt_footprint_differs_by_version() -> None:
    assert prompt_footprint("minimal") != prompt_footprint("enriched_v1")
