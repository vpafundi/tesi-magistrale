# Capitolo 4 — Metodologia sperimentale

**Stato:** outline  
**Estensione target:** ~14–18 pagine

## Obiettivo del capitolo

Descrivere in modo riproducibile prompt, split, troncamento, baseline, inferenza SLM e fine-tuning — senza duplicare tutto il codice.

## Outline

1. **Principio di comparabilità** — stesso split, `PROMPT_VERSION`, `max_seq_length` da `run_config.json`.
2. **Prompt** — system + macro + testo; solo nome micro in output (v1); vincolo guided choice in inferenza.
3. **Lunghezza contesto** — misura su sequenza training; troncamento solo testo segmento.
4. **Baseline classiche** — majority, random, `tfidf_logreg` (char n-grams, logreg bilanciata, mascheramento micro ammesse).
5. **SLM zero-shot** — vLLM; temperature 0; thinking disattivato; varianti constrained/free/few-shot.
6. **Fine-tuning (pianificato)** — LoRA (Unsloth); iperparametri; early stopping su val; stesso prompt.
7. **Infrastruttura** — sviluppo locale vs SageMaker (GPU, bf16 Gemma).
8. **Riproducibilità** — `save_run`, hash modello, versioni librerie.

## TODO

- [ ] Tabella iperparametri LoRA (quando fissati)
- [ ] Diagramma flusso train/eval
