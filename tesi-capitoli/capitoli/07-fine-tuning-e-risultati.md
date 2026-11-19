# Capitolo 7 — Fine-tuning e risultati

**Stato:** da fare  
**Estensione target:** ~16–22 pagine

## Obiettivo del capitolo

Descrivere addestramento LoRA, scelta iperparametri, valutazione su test; confronto zero-shot vs fine-tuned; eventuali curve (dati 5k / 20k / 70k se eseguite).

## Outline

1. **Configurazione training** — LoRA rank/alpha, batch, lr, epoch, `max_seq_length`.
2. **Confronto per modello** — Δ macro-F1 vs zero-shot e vs TF-IDF.
3. **Analisi per macro** — dove il FT recupera più errore (allineare a confusioni cap. 5–6).
4. **Esperimenti opzionali** — few-shot vs FT; lunghezza contesto; ablation prompt v2.
5. **Limiti** — overfitting su classi dominanti; etichette silver.

## TODO

- [ ] Tutti i numeri da `save_run` post-training
- [ ] Figura learning curve (se prevista in PROGETTO.md §8)
