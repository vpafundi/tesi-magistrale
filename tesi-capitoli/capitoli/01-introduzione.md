# Capitolo 1 — Introduzione

**Stato:** outline  
**Estensione target:** ~8–12 pagine

## Obiettivo del capitolo

Motivare il problema nel dominio assicurativo, enunciare obiettivo e contributi, anticipare struttura della tesi.

## Outline

1. **Contesto applicativo** — volumi documentali, OCR, classificazione a più livelli (macro/micro), costi di modelli grandi in produzione.
2. **Problema** — dato segmento + macro nota, predire micro; vincolo tassonomico per macro.
3. **Ipotesi di lavoro** — SLM fine-tunati possono avvicinarsi al teacher e battere zero-shot/baseline leggere a parità di protocollo.
4. **Contributi attesi** (da rifinire a fine tesi)
   - Protocollo riproducibile (split, prompt, metriche) condiviso tra baseline e fine-tuning.
   - Studio su corpus reale anonimizzato (~70k segmenti) con EDA e probe di fattibilità.
   - Confronto multi-modello (es. Qwen3, SmolLM3, Gemma) 1,7B–4B.
5. **Struttura del documento** — mappa capitoli 2–9.

## Note / materiali

- Catena produzione: teacher → etichette silver → distillazione SLM (non gold manuali su tutto il corpus).
- Riferimento interno: `docs/PROGETTO.md` §1–2.

## TODO

- [ ] 1–2 figure (pipeline alto livello)
- [ ] Definire “contributi” numerati dopo risultati sperimentali
