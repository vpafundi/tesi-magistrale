"""Prompt rendering, token-budget truncation and output parsing (engine-agnostic)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..domain.labels import INVALID_LABEL, MICRO_PER_MACRO
from ..prompt.core import build_messages

# Thinking off for every model: Qwen3 and SmolLM3 read this kwarg, Gemma 3 ignores it.
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

ANSWER_RESERVE_TOKENS = 32
FEW_SHOT_MAX_TEXT_TOKENS = 300

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def adapt_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Model-specific tweaks so the rendered prompt is deterministic. Use it for training too.

    SmolLM3's template otherwise injects metadata with today's date into the system turn.
    """
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "smollm3" in name and messages and messages[0]["role"] == "system":
        messages = [{**messages[0], "content": messages[0]["content"] + "\n/system_override"}, *messages[1:]]
    return messages


def render_prompt(
    tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = True
) -> str:
    return tokenizer.apply_chat_template(
        adapt_messages(tokenizer, messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **CHAT_TEMPLATE_KWARGS,
    )


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """The rendered template already contains BOS where needed (e.g. Gemma): don't add it twice."""
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def token_length_report(
    tokenizer: Any,
    df: pd.DataFrame,
    thresholds: tuple[int, ...] = (1024, 2048, 3072, 4096, 6144, 8192),
    *,
    prompt_version: str = "minimal",
) -> tuple[pd.Series, pd.DataFrame]:
    """Full training-sequence length (system + macro + text + answer) per row, and coverage per threshold.

    This is what `max_seq_length` must cover; `df` needs macro_category, sub_category, text.
    """
    lengths = pd.Series(
        [
            len(
                encode_prompt(
                    tokenizer,
                    render_prompt(
                        tokenizer,
                        build_messages(m, t, mi, prompt_version=prompt_version),
                        add_generation_prompt=False,
                    ),
                )
            )
            for m, t, mi in zip(df["macro_category"], df["text"], df["sub_category"])
        ],
        index=df.index,
        name="seq_tokens",
    )
    coverage = pd.DataFrame(
        {
            "max_seq_len": list(thresholds),
            "pct_covered": [round(float((lengths <= t).mean()) * 100, 2) for t in thresholds],
            "n_truncated": [int((lengths > t).sum()) for t in thresholds],
        }
    )
    return lengths, coverage


def recommend_max_seq_len(lengths: dict[str, pd.Series], headroom: float = 0.20, multiple: int = 1024) -> int:
    """Longest sequence across tokenizers + headroom, rounded up to `multiple`."""
    longest = max(int(s.max()) for s in lengths.values())
    target = longest * (1 + headroom)
    return int(-(-target // multiple) * multiple)


def _load_hf_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _max_prompt_overhead_tokens(
    tokenizer: Any,
    macros: list[str] | None = None,
    *,
    prompt_version: str = "minimal",
) -> int:
    """Tokens for rendered prompt with empty segment text (incl. chat template, no generation marker)."""
    macros = macros or list(MICRO_PER_MACRO.keys())
    return max(
        len(
            encode_prompt(
                tokenizer,
                render_prompt(tokenizer, build_messages(m, "", prompt_version=prompt_version), add_generation_prompt=False),
            )
        )
        for m in macros
    )


def compute_max_seq_len_for_experiment(
    *,
    mode: str,
    base_max_seq_len: int,
    df: pd.DataFrame,
    tokenizer_ids: list[str],
    headroom: float,
    multiple: int,
    prompt_version: str,
    token_sample_rows: int | None,
    macros: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Resolve final max_seq_len from config. Returns (value, debug dict)."""
    debug: dict[str, Any] = {"mode": mode, "base_max_seq_len": base_max_seq_len}
    if mode == "manual":
        return base_max_seq_len, debug

    tok_list = tokenizer_ids
    overhead_by_model: dict[str, int] = {}
    macro_list = macros or list(MICRO_PER_MACRO.keys())
    for tid in tok_list:
        tok = _load_hf_tokenizer(tid)
        overhead_by_model[tid] = _max_prompt_overhead_tokens(tok, macro_list, prompt_version=prompt_version)
    max_over = max(overhead_by_model.values()) if overhead_by_model else 0
    debug["max_prompt_overhead_tokens"] = overhead_by_model

    if mode == "manual_plus_prompt":
        target = base_max_seq_len + max_over
        rounded = int(-(-target // multiple) * multiple)
        debug["unrounded"] = target
        return rounded, debug

    if mode == "computed":
        sample = df
        if token_sample_rows is not None and len(df) > token_sample_rows:
            sample = df.sample(n=token_sample_rows, random_state=42)
        lengths: dict[str, pd.Series] = {}
        for tid in tok_list:
            tok = _load_hf_tokenizer(tid)
            lengths[tid], _ = token_length_report(
                tok, sample, thresholds=(1024, 8192), prompt_version=prompt_version
            )
        rec = recommend_max_seq_len(lengths, headroom=headroom, multiple=multiple)
        debug["longest_per_tokenizer"] = {k: int(v.max()) for k, v in lengths.items()}
        debug["headroom"] = headroom
        return rec, debug

    raise ValueError(f"Unknown max_seq_len mode: {mode!r}")


def truncate_tokens(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, int, bool]:
    """Head truncation. Returns (text, original_token_count, truncated)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, len(ids), False
    return tokenizer.decode(ids[: max(max_tokens, 0)]), len(ids), True


def build_prompts(
    tokenizer: Any,
    df: pd.DataFrame,
    max_seq_len: int,
    few_shot: dict[str, list[dict[str, str]]] | None = None,
    *,
    prompt_version: str = "minimal",
) -> pd.DataFrame:
    """One rendered prompt per row; only the segment text is truncated, never the instructions.

    `df` needs columns record_key, macro_category, text.
    Returns record_key, prompt, prompt_token_ids, prompt_tokens, text_tokens, truncated.
    """
    overhead: dict[str, int] = {}
    for macro in df["macro_category"].unique():
        empty = render_prompt(
            tokenizer,
            build_messages(
                macro, "", few_shot=(few_shot or {}).get(macro), prompt_version=prompt_version
            ),
        )
        overhead[macro] = len(encode_prompt(tokenizer, empty))

    rows = []
    for r in df.itertuples(index=False):
        shots = (few_shot or {}).get(r.macro_category)
        budget = max_seq_len - overhead[r.macro_category] - ANSWER_RESERVE_TOKENS
        if budget <= 0:
            raise ValueError(f"max_seq_len={max_seq_len} troppo piccolo per il prompt di {r.macro_category}")
        text, n_text, truncated = truncate_tokens(tokenizer, r.text, budget)
        prompt = render_prompt(
            tokenizer, build_messages(r.macro_category, text, few_shot=shots, prompt_version=prompt_version)
        )
        ids = encode_prompt(tokenizer, prompt)
        rows.append(
            {
                "record_key": r.record_key,
                "prompt": prompt,
                "prompt_token_ids": ids,
                "prompt_tokens": len(ids),
                "text_tokens": n_text,
                "truncated": truncated,
            }
        )
    return pd.DataFrame(rows)


def select_few_shot(
    train_df: pd.DataFrame,
    tokenizer: Any,
    k: int = 1,
    seed: int = 42,
    *,
    prompt_version: str = "minimal",
) -> dict[str, list[dict[str, str]]]:
    """k examples per micro from the train split, shorter ones preferred, truncated."""
    out: dict[str, list[dict[str, str]]] = {}
    for macro, micros in MICRO_PER_MACRO.items():
        shots: list[dict[str, str]] = []
        for micro in micros:
            pool = train_df[(train_df["macro_category"] == macro) & (train_df["sub_category"] == micro)]
            if pool.empty:
                continue
            short = pool[pool["text"].str.len().between(200, 1500)]
            sample = (short if len(short) >= k else pool).sample(n=min(k, len(pool)), random_state=seed)
            for t in sample["text"]:
                text, _, _ = truncate_tokens(tokenizer, t, FEW_SHOT_MAX_TEXT_TOKENS)
                shots.append({"text": text, "micro": micro})
        out[macro] = shots
    return out


def parse_free_output(raw: str, macro: str) -> str:
    """Map a free-form answer to an allowed micro, else INVALID_LABEL."""
    allowed = MICRO_PER_MACRO[macro]
    s = _THINK_BLOCK.sub("", raw or "").strip().lower()
    s = s.strip("`'\"*.:;, \n\t")
    s_norm = re.sub(r"\s+", "_", s)
    if s_norm in allowed:
        return s_norm
    s_spaced = re.sub(r"[_\s]+", " ", s)
    found = {a for a in allowed if re.search(rf"\b{re.escape(a.replace('_', ' '))}\b", s_spaced)}
    if len(found) == 1:
        return found.pop()
    return INVALID_LABEL
