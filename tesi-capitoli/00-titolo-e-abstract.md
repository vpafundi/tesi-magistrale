# Titolo (bozza)

**Titolo provvisorio (IT):**  
Classificazione gerarchica di segmenti di documenti assicurativi con Small Language Models: baseline, fine-tuning e confronto con metodi classici

**Sottotitolo (opzionale):**  
Distillazione da etichette silver e valutazione su OCR anonimizzato a livello di segmento

**Titolo provvisorio (EN):**  
Hierarchical classification of insurance document segments with small language models: baselines, fine-tuning, and comparison with classical methods

---

## Abstract (bozza — IT)

*(Da completare quando baseline SLM e fine-tuning avranno numeri definitivi.)*

Nei flussi assicurativi automatizzati, la classificazione dei documenti avviene spesso in due livelli: una macrocategoria nota o stimata in precedenza, e una microcategoria che specializza il tipo di segmento. Il lavoro presentato valuta se modelli linguistici compatti (1,7B–4B parametri), addestrati con fine-tuning supervisionato, possono avvicinarsi o superare modelli usati senza addestramento e baseline classiche (TF-IDF con regressione logistica), mantenendo lo stesso protocollo di valutazione.

Il task considerato è la predizione della microcategoria **condizionata** alla macrocategoria, su testo OCR anonimizzato (`ocr_surrogates`) a livello di segmento. Le etichette derivano da una catena di produzione basata su modelli più grandi (etichette *silver*). Su un corpus di circa settantamila segmenti, un’analisi esplorativa e probe lineari indicano che il problema è fattibile ma non banale: alcune macro (es. denuncia) sono quasi separabili con metodi lessicali, altre (es. documentazione medica e spesa medica) presentano confusioni tra classi rare o semanticamente vicine.

Metodologia: split stratificato raggruppato per documento, prompt unico versionato, metrica principale macro-F1 con intervalli di confidenza. Si descrivono baseline zero-shot e supervised, fine-tuning (LoRA) su modelli open-weights, e limiti legati ad anonimizzazione, sbilanciamento e qualità OCR.

**Parole chiave:** small language models, fine-tuning, LoRA, classificazione testuale, documenti assicurativi, OCR, distillation, macro-F1.

---

## Abstract (bozza — EN)

*(Placeholder — mirror of Italian abstract once finalized.)*

---

## Metadati tesi

| Campo | Valore |
|--------|--------|
| Studente | Vincenzo Pafundi |
| Anno accademico | 2025–2026 |
| Relatore | *(da compilare)* |
| Correlatore | *(da compilare)* |
| Ateneo / corso | *(da compilare)* |
