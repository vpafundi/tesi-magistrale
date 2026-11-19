"""Prompt shared by baseline and fine-tuning: version keys invalidate comparisons."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from ..domain.labels import MICRO_PER_MACRO

# Kept for backwards compatibility: minimal v1 behaviour.
PROMPT_VERSION = "v1"
PROMPT_REGISTRY_VERSION = "r1"

BASE_SYSTEM_PROMPT = (
    "Sei un classificatore di segmenti di documenti assicurativi. "
    "Ti viene indicata la macrocategoria (macro) e il testo del segmento. "
    "Rispondi solo con il nome della microcategoria (micro), "
    "senza spiegazioni, senza JSON e senza altro testo."
)

_VARIANT_DIR = Path(__file__).resolve().parent.parent / "prompt_variants"

MACRO_TO_VARIANT_FILE: dict[str, str] = {
    "denuncia": "denuncia.md",
    "documentazione_medica": "documentazione_medica.md",
    "documento_di_spesa": "documento_di_spesa.md",
    "documento_di_spesa_medica": "documento_di_spesa_medica.md",
}


@lru_cache(maxsize=8)
def _load_enriched_v1_body(macro: str) -> str:
    name = MACRO_TO_VARIANT_FILE[macro]
    path = _VARIANT_DIR / "enriched_v1" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing enriched_v1 prompt for {macro}: {path}")
    return path.read_text(encoding="utf-8").strip()


def prompt_footprint(prompt_version: str = "minimal") -> str:
    """SHA256 of registry + variant files used for `prompt_version` (for manifest)."""
    parts = [PROMPT_REGISTRY_VERSION, prompt_version]
    if prompt_version == "enriched_v1":
        h = hashlib.sha256()
        for macro in sorted(MACRO_TO_VARIANT_FILE):
            p = _VARIANT_DIR / "enriched_v1" / MACRO_TO_VARIANT_FILE[macro]
            h.update(p.read_bytes())
        parts.append(h.hexdigest()[:32])
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def system_prompt_for_macro(macro: str, prompt_version: str = "minimal") -> str:
    allowed = ", ".join(MICRO_PER_MACRO[macro])
    if prompt_version == "minimal":
        return (
            f"{BASE_SYSTEM_PROMPT}\n\n"
            f"Per la macro «{macro}», le uniche micro ammesse sono: {allowed}. "
            f"Rispondi con una sola di queste micro, scritta esattamente come nell'elenco."
        )
    if prompt_version == "enriched_v1":
        body = _load_enriched_v1_body(macro)
        return (
            f"{BASE_SYSTEM_PROMPT}\n\n"
            f"{body}\n\n"
            f"Per la macro «{macro}», le uniche micro ammesse sono: {allowed}. "
            f"Rispondi con una sola di queste micro, scritta esattamente come nell'elenco."
        )
    raise ValueError(f"Unknown prompt_version: {prompt_version!r}")


def user_content(macro: str, text: str) -> str:
    return f"Macro: {macro}\n\n{text}"


def build_messages(
    macro: str,
    text: str,
    micro: str | None = None,
    few_shot: list[dict[str, str]] | None = None,
    prompt_version: str = "minimal",
) -> list[dict[str, str]]:
    """Chat messages; pass `micro` to get the training conversation (assistant turn included).

    `few_shot` items: {"text": ..., "micro": ...}, same macro, inserted as prior turns.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt_for_macro(macro, prompt_version=prompt_version)}
    ]
    for ex in few_shot or []:
        messages.append({"role": "user", "content": user_content(macro, ex["text"])})
        messages.append({"role": "assistant", "content": ex["micro"]})
    messages.append({"role": "user", "content": user_content(macro, text)})
    if micro is not None:
        messages.append({"role": "assistant", "content": micro})
    return messages
