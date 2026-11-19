# Capitolo 6 — Baseline e SLM senza fine-tuning

**Stato:** da scrivere dopo notebook `01`  
**Estensione target:** ~14–18 pagine

## Obiettivo del capitolo

Riportare risultati su test fisso: baseline classiche e tre SLM in modalità zero-shot (e opzionale few-shot); analisi errori per macro.

## Outline

1. **Setup** — split, prompt v1, modelli (Qwen3-1.7B, SmolLM3-3B, Gemma-3-4b-it).
2. **Risultati aggregati** — tabella macro-F1 + IC bootstrap; confronto con probe EDA (solo come ordine di grandezza).
3. **Per macro** — matrici di confusione; classi problematiche allineate a cap. 5.
4. **Modalità constrained vs free** — tasso invalid; formato output.
5. **Costi operativi** — throughput, troncamento, memoria GPU (qualitativo).
6. **Sintesi** — pavimento (majority/random) vs TF-IDF vs SLM zero-shot.

## TODO

- [ ] Tabella `outputs/baseline/summary.csv` quando disponibile
- [ ] Esempi errori anonimizzati (senza testo OCR in tesi se policy azienda lo vieta)
