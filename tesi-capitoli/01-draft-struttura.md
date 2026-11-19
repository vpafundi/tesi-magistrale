# Draft — struttura generale

## Obiettivo della tesi (1 paragrafo)

Quantificare quanto il **fine-tuning** di Small Language Models migliora la classificazione micro (data la macro) su segmenti OCR di pratiche assicurative, rispetto a (1) gli stessi modelli **zero-shot** con prompt allineato alla produzione e (2) baseline **TF-IDF + regressione logistica** (e, se tempo lo permette, encoder fine-tunato). Il confronto deve essere **rigoroso**: stesso split, stesso prompt versionato, stessa metrica (macro-F1).

## Domande di ricerca (bozza)

1. Con etichette silver e testo surrogato, il segnale testuale basta a distinguere le micro all’interno di ogni macro?
2. Quale guadagno porta il fine-tuning LoRA rispetto allo zero-shot, per modello e per macro?
3. Quando una baseline classica è già vicina al plateau, l’SLM fine-tunato giustifica costo e complessità operativa?
4. Quali limiti impongono anonimizzazione, sbilanciamento, segmenti lunghi/vuoti e possibile rumore nelle etichette?

## Filo narrativo consigliato

Contesto assicurativo → task gerarchico e dati → EDA (fattibilità + rischi) → protocollo sperimentale → baseline → fine-tuning → risultati per macro → discussione limiti → conclusioni.

## Indice capitoli (target)

Vedi file in `capitoli/`; numerazione allineata a consegne tipiche magistrali (adattare al template di ateneo).

| Cap. | Titolo breve | Stato |
|------|----------------|-------|
| 1 | Introduzione | Bozza outline |
| 2 | Contesto e related work | Bozza outline |
| 3 | Dati, task e perimetro | Bozza outline + numeri EDA |
| 4 | Metodologia sperimentale | Bozza outline |
| 5 | Analisi esplorativa e fattibilità | Parzialmente da EDA/O4 |
| 6 | Baseline e modelli senza fine-tuning | Da notebook 01 |
| 7 | Fine-tuning e risultati | Da fare |
| 8 | Discussione, limiti, lavori futuri | Bozza outline |
| 9 | Conclusioni | Da fare |
| A | Appendici (prompt, tassonomia, iperparametri) | Da fare |

## Cosa non va (ancora) in tesi

- Dettagli implementativi di repository interni non pubblici (solo descrizione high-level).
- Numeri presi solo dal campione 150 righe o da probe EDA campionati **senza** dire che la valutazione ufficiale è su split completo.
- Confronti SLM vs TF-IDF su campioni diversi (es. O4 full TF-IDF vs bge subsample): in tesi citare solo protocolli allineati.

## TODO trasversali

- [ ] Allineare indice al template Word/PDF dell’università
- [ ] Inserire numeri definitivi post-`01_baseline` e post-fine-tuning
- [ ] Figura: schema catena macro → micro e posizione SLM
- [ ] Tabella: tassonomia 4×19 micro
- [ ] Bibliografia (SLM, LoRA, document classification, distillation)
