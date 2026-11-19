# Strategia di EDA per il fine-tuning dello SLM

## Contesto

Il task è prevedere la `sub_category` di un segmento **data la `macro_category`**, che è già nota in input.
Lo SLM sostituisce lo step micro della catena di produzione (Claude Sonnet, temperature 0, catena macro → micro).

Le `sub_category` sono le previsioni del teacher: etichette **silver**, non verità certa.
Il fine-tuning è distillazione dal teacher allo SLM.

Conseguenza per tutta l'EDA: l'unità di analisi è la **singola macro**. Le sub vanno confrontate solo con le
altre sub della stessa macro.

Il testo usato dal modello è `ocr_surrogates` (valori finti ma realistici al posto delle entità). Nel dataset
completo non compaiono segnaposto tipo `[PIVA_1]`; non è quindi possibile misurare direttamente la perdita
rispetto al testo originale (vedi O3).

L'EDA deve rispondere a tre domande:

1. Il testo in ingresso contiene abbastanza informazione per distinguere le sub all'interno della macro?
2. Le etichette del teacher sono coerenti tra loro?
3. I dati raccolti rappresentano bene la produzione (volume, drift temporale)?

## Obiettivi di esplorazione

| # | Domanda | Metodo | Decisione che abilita |
|---|---|---|---|
| O1 | Quante sub per macro, bilanciamento, sufficienza per train/test? | Conteggi per sub, status rispetto a soglie, volumi giornalieri | Sub da tenere/unire; dimensionamento test per sub |
| O2 | Qualità del testo e lunghezza in token? | Parole/pagine, OCR confidence, handwritten, `token_length_report` (prompt v1, tre tokenizer) | Filtri, `max_seq_length`, troncamento |
| O3 | I surrogati conservano segnali discriminanti? | Presenza di pattern lessicali per sub (IVA, ticket, importi, …) | Limite dell'anonimizzazione da discutere in tesi |
| O4 | Le sub si distinguono nello spazio testo? | TF-IDF + linear probe per macro; `BAAI/bge-m3` su sottocampione; UMAP esplorativo | Fattibilità, coppie confuse, confronto con baseline `01` |
| O5 | Le etichette silver sono coerenti? | cleanlab su probabilità out-of-fold del probe TF-IDF; quasi-duplicati con etichetta diversa (O6) | Set gold, revisione etichette |
| O6 | Leakage e duplicati? | Split per `document_id`; hash testo; quasi-duplicati (cosine TF-IDF > 0.98) | Deduplica in train; rigenerare `splits_v1` sul dataset completo |
| O7 | Drift temporale? | Quota sub per settimana; confronto prima/dopo cambio volume (es. 15/10) | Holdout temporale opzionale; nota su versione teacher (non in parquet) |

O1.4/O1.5 nel notebook servono a **dimensionare il test** (documenti per sub a recall attesa), non a pianificare
raccolta futura: lo snapshot attuale (~70k segmenti) è il corpus di lavoro.

## Analisi nello spazio latente (O4/O5, versione snella)

Tutte le analisi si ripetono **per ogni macro**, sulle sole sub di quella macro.

### Rappresentazioni

1. **TF-IDF su n-grammi di caratteri** (come [`src/slm_baseline/classic.py`](../../src/slm_baseline/classic.py)): riferimento economico, robusto al rumore OCR. Nell'EDA si usa con `StratifiedGroupKFold` per documento; i numeri ufficiali della baseline restano nel notebook `01`.
2. **`BAAI/bge-m3`** su un sottocampione stratificato (max ~1500 segmenti per sub), embedding in cache locale, troncamento token moderato su Mac (MPS).

Non usiamo in questa fase: hidden states dell'SLM, HDBSCAN, BERTopic, kNN/silhouette sistematici, e5-large separato.

### Misure

- **Linear probe per macro**: regressione logistica, macro-F1 e matrice di confusione tra sub.
- **cleanlab** sulle probabilità out-of-fold del TF-IDF: candidati a revisione e campionamento set gold.
- **UMAP 2D per macro** (solo esplorazione visiva sul sottocampione bge-m3).

### Come leggere i risultati

- Probe alto: task fattibile; confrontare con SLM zero-shot e fine-tuned in `01` / training.
- Probe basso su coppie specifiche: sub vicine; unire, più contesto o revisione etichette.
- Probe basso diffuso: OCR, testo vuoto o etichette silver (O2/O5).

## Riferimenti da battere

1. TF-IDF + regressione logistica (baseline `01`, `tfidf_logreg`).
2. SLM zero-shot con lo stesso prompt di [`src/slm_baseline/prompts.py`](../../src/slm_baseline/prompts.py) (`PROMPT_VERSION`), macro in input.
3. SLM con LoRA: deve battere (2) e avvicinarsi al teacher.

Se (1) raggiunge già ~0.95 macro-F1, l'SLM si giustifica soprattutto se servono altri output oltre alla sola micro.

## Strategia di split

### Raggruppamento per documento (obbligatorio)

Tutti i segmenti di uno stesso `document_id` nello stesso split (`StratifiedGroupKFold` in [`src/slm_baseline/data/core.py`](../../src/slm_baseline/data/core.py)).
Implementazione definitiva: notebook `00_prepare_data_and_splits.ipynb` sul parquet completo.

### Holdout temporale (controllo opzionale)

Valutazione principale: split stratificato per `(macro, sub)` raggruppato per documento.
Controllo aggiuntivo: finestra recente come holdout (O7).

## Preparazione dei dati

- **Perimetro etichette**: [`src/slm_baseline/labels.py`](../../src/slm_baseline/labels.py) (`MICRO_PER_MACRO`).
- **Macro in input**: nel prompt v1, elenco micro ammesse per macro; risposta solo con il nome della micro (no rationale).
- **Set gold**: 200–500 segmenti stratificati per `(macro, sub)`, verificati a mano.
- **Sub rare**: soglia indicativa 50–100 segmenti per sub per LoRA; sotto soglia unire o "altro" della macro.
- **Deduplica**: quasi-duplicati (coseno > 0.98 su TF-IDF per macro) rimossi dal train dopo O6.
- **Testo vuoto**: escluso dalla valutazione, contato in O2.

## Ordine di lavoro nel notebook 00b

O1 → O2 → O3 → O6 → O4 → O5 → O7 → sezioni **Decisioni** (input per notebook `00` e [`docs/PROGETTO.md`](../../docs/PROGETTO.md)).

## Artefatti su disco (gitignored sotto `data/`)

| Cartella | Contenuto |
|---|---|
| `data/eda_cache/o4/` | Probe TF-IDF / bge-m3: config, metriche, OOF predictions (senza testo) |
| `data/eda_cache/eda_summary/` | Riepilogo O5–O7 + `decisions.json`, tabelle O1/O3 opzionali; scritto nell’ultima cella **Decisioni finali** |

Per analisi offline: `load_eda_summary()` nella cella di persistenza, oppure leggere CSV/JSON direttamente. Dopo aver aggiornato il notebook, rieseguire almeno le celle **Persistenza riepilogo EDA** → **Decisioni finali** (e O5–O7 se il kernel è stato riavviato).
