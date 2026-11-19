from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from slm_baseline.inference import compute_max_seq_len_for_experiment


class _FakeTok:
    """Minimal tokenizer API for overhead test."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": list(range(len(text) // 4 + 1))}

    def apply_chat_template(self, messages: list, tokenize: bool = False, **kwargs: object) -> str:
        return " ".join(m["content"] for m in messages)

    @property
    def name_or_path(self) -> str:
        return "fake"


def test_manual_plus_prompt_uses_overhead() -> None:
    df = pd.DataFrame(
        {
            "macro_category": ["denuncia"],
            "sub_category": ["testata"],
            "text": ["x" * 100],
        }
    )
    with patch("slm_baseline.sequence.core._load_hf_tokenizer", return_value=_FakeTok()):
        val, dbg = compute_max_seq_len_for_experiment(
            mode="manual_plus_prompt",
            base_max_seq_len=1024,
            df=df,
            tokenizer_ids=["fake/model"],
            headroom=0.2,
            multiple=256,
            prompt_version="minimal",
            token_sample_rows=None,
        )
    assert val % 256 == 0
    assert val >= 1024
    assert "max_prompt_overhead_tokens" in dbg
