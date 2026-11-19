# Capitolo 2 — Contesto e stato dell’arte

**Stato:** outline  
**Estensione target:** ~15–20 pagine

## Obiettivo del capitolo

Inquadrare classificazione documentale, NLP su OCR rumoroso, Small Language Models e fine-tuning efficiente (LoRA/QLoRA); posizionare il lavoro rispetto a distillation e gerarchical classification.

## Outline

1. **Classificazione di documenti strutturati e non** — segmenti vs documento intero; tassonomie gerarchiche.
2. **OCR e testo rumoroso** — impact on bag-of-words vs transformers; anonimizzazione e surrogati.
3. **Modelli linguistici piccoli (SLM)** — trade-off dimensione/latency/qualità; modelli open-weights citati in progetto.
4. **Fine-tuning efficiente** — LoRA, instruction tuning, chat templates.
5. **Baseline classiche** — TF-IDF + linear models; encoder (BERT-family) come riferimento supervisionato.
6. **Knowledge distillation / silver labels** — teacher-student, rischi etichette rumorose.
7. **Gap** — poca letteratura pubblica su stesso setup assicurativo; obiettivo replicabile in azienda.

## TODO

- [ ] Bibliografia minima (8–15 paper + doc tecnica HF)
- [ ] Tabella comparativa approcci (feature-based vs SLM zero-shot vs SLM FT)
