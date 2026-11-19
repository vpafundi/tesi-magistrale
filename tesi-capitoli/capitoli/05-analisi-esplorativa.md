# Capitolo 5 — Analisi esplorativa e fattibilità

**Stato:** parzialmente scrivibile da EDA/O4  
**Estensione target:** ~12–16 pagine

## Obiettivo del capitolo

Presentare EDA (O1–O7): volumi, qualità OCR, token length, segnali nei surrogati, duplicati/leakage, probe TF-IDF/bge, drift temporale; collegare alle decisioni su filtri e training.

## Outline

1. **O1 — Volumi e sbilanciamento** — sub rare; dimensionamento test per documento.
2. **O2 — Qualità testo** — vuoti, corti, confidence OCR, `max_seq_length` (campione tokenizer).
3. **O3 — Segnali discriminanti** — pattern lessicali per sub (limiti senza testo originale).
4. **O6 — Duplicati e split** — hash testo; quasi-duplicati; split per documento.
5. **O4 — Linear probe** — metodologia CV 5-fold per documento; TF-IDF full vs bge campionato (discutere comparabilità).
6. **O5 — Coerenza etichette** — cleanlab (da completare in notebook); set gold proposto.
7. **O7 — Drift** — volume pre/post 15/10; limiti (versione teacher non in parquet).
8. **Sintesi decisioni** — cosa escludere in valutazione; macro più difficili; coppie confuse.

## Risultati O4 già disponibili (run 20261105, TF-IDF full)

| Macro | Macro-F1 TF-IDF | Macro-F1 bge (subsample) |
|--------|-------------------|---------------------------|
| denuncia | 0,953 | 0,931 |
| documentazione_medica | 0,810 | 0,836 |
| documento_di_spesa | 0,796 | 0,734 |
| documento_di_spesa_medica | 0,744 | 0,809 |

**Confusioni da enfatizzare:** certificato vs altro (medica); scontrino raro (spesa); fattura/ricevuta/ticket/preventivo (spesa medica).

## TODO

- [ ] Inserire 1–2 figure UMAP (se pubblicabili senza testo)
- [ ] Completare paragrafo O5 con conteggi cleanlab
- [ ] Non presentare probe EDA come benchmark ufficiale
