"""Label space in perimeter: macro → allowed micro."""

from __future__ import annotations

MICRO_PER_MACRO: dict[str, list[str]] = {
    "denuncia": [
        "cai",
        "richiesta_danni",
        "denuncia_altro",
    ],
    "documentazione_medica": [
        "prescrizione_medica",
        "referto_del_pronto_soccorso",
        "certificato_avvenuta_guarigione",
        "lettera_di_dimissioni",
        "documentazione_medica_altro",
    ],
    "documento_di_spesa": [
        "fattura",
        "scontrino",
        "preventivo_di_spesa",
        "ricevuta_di_pagamento",
        "documento_di_spesa_altro",
    ],
    "documento_di_spesa_medica": [
        "fattura",
        "ticket_ssn",
        "scontrino",
        "preventivo_di_spesa",
        "ricevuta_di_pagamento",
        "documento_di_spesa_altro",
    ],
}

INVALID_LABEL = "__invalid__"


def composite_label(macro: str, micro: str) -> str:
    """Micro names repeat across macros (e.g. fattura), so metrics use macro::micro."""
    return f"{macro}::{micro}"


def all_composite_labels() -> list[str]:
    return [composite_label(m, mi) for m, micros in MICRO_PER_MACRO.items() for mi in micros]


def is_in_perimeter(macro: str, micro: str) -> bool:
    return micro in MICRO_PER_MACRO.get(macro, [])
