# Capitolo 3 — Dati, task e perimetro

**Stato:** outline + numeri da EDA  
**Estensione target:** ~12–16 pagine

## Obiettivo del capitolo

Descrivere corpus, tassonomia, preprocessing testuale, limiti privacy; definire formalmente input/output e metriche.

## Outline

1. **Origine dati** — segmenti da pipeline ingestion; periodo di raccolta; documenti vs segmenti.
2. **Tassonomia** — 4 macro, 19 micro; stesso nome micro in macro diverse → classe `macro::micro` in valutazione.
3. **Testo `ocr_surrogates`** — scelta rispetto a placeholder; segmenti vuoti (~5,6%); lunghezze (p95, max token).
4. **Etichette silver** — provenienza teacher; implicazioni per training e per EDA (cleanlab).
5. **Split** — StratifiedGroupKFold su `document_id`; train/val/test; rigenerazione su corpus completo.
6. **Metriche** — macro-F1 principale; accuracy; bootstrap; metriche per macro; invalid rate (modalità free SLM).
7. **Perimetro** — `labels.py`; righe escluse (~54 fuori tassonomia).

## Numeri da citare (EDA, verificare date/versione run)

| Grandezza | Valore |
|-----------|--------|
| Segmenti grezzi | 70.637 |
| In perimetro | ~70.584 |
| Documenti | 36.973 |
| Giorni raccolta | 41 (24/09–03/11/2026) |
| Esempio sbilanciamento | `documentazione_medica_altro` ~22,4k; `documento_di_spesa::scontrino` ~104 |

## TODO

- [ ] Tabella completa conteggi per (macro, micro)
- [ ] Aggiornare dopo split definitivo `splits_v1` su full data
