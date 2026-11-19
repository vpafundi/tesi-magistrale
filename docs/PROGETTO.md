# Progetto: classificazione di segmenti di documenti assicurativi con Small Language Models

Documento di riferimento per la tesi: obiettivo, scelte fatte (con il perché), stato attuale e prossimi passi. Il "come si lancia" è nel [README](../README.md).

---

## 1. Obiettivo

Misurare quanto il **fine-tuning** di Small Language Models (1.7B–4B parametri) migliora la classificazione di segmenti di documenti assicurativi rispetto agli stessi modelli usati **senza fine-tuning**, e rispetto a baseline classiche.

Per farlo servono prima le **baseline**: numeri di partenza calcolati sullo stesso test, con lo stesso prompt e le stesse metriche che userà la valutazione dopo il fine-tuning. È la fase in cui siamo.

## 2. Il task

Ogni esempio è un **segmento** di documento (una o più pagine OCR) di cui è già nota la **macrocategoria**. Il modello deve scegliere la **microcategoria** tra quelle ammesse per quella macro.

| Macro | Micro ammesse |
|---|---|
| `denuncia` | `cai`, `richiesta_danni`, `denuncia_altro` |
| `documentazione_medica` | `prescrizione_medica`, `referto_del_pronto_soccorso`, `certificato_avvenuta_guarigione`, `lettera_di_dimissioni`, `documentazione_medica_altro` |
| `documento_di_spesa` | `fattura`, `scontrino`, `preventivo_di_spesa`, `ricevuta_di_pagamento`, `documento_di_spesa_altro` |
| `documento_di_spesa_medica` | `fattura`, `ticket_ssn`, `scontrino`, `preventivo_di_spesa`, `ricevuta_di_pagamento`, `documento_di_spesa_altro` |

4 macro, 19 micro. Alcune micro hanno lo stesso nome in macro diverse (`fattura`, `scontrino`, …): nelle metriche le classi sono quindi `macro::micro`. Il perimetro è definito in un solo punto, `src/slm_baseline/labels.py`; le righe fuori perimetro vengono escluse e contate.

## 3. Dati

**Formato.** Parquet già a livello di segmento, prodotto dalla pipeline di ingestion aziendale. Colonne usate:

| Colonna | Uso |
|---|---|
| `record_key` | Chiave univoca della riga, usata dallo split e dalle predizioni |
| `document_id` | Raggruppamento per lo split (niente leakage tra split) |
| `macro_category`, `sub_category` | Macro in input, micro come etichetta |
| `ocr_surrogates` | **Testo usato** (vedi sotto) |
| `pages_list`, `ocr_confidence`, `ocr_handwritten_rate`, … | Disponibili per analisi EDA, non usati dal modello |

**Dimensione (EDA novembre 2025, `data/df_sample_all.parquet`).** 70.637 segmenti grezzi, **~70.583 in perimetro** dopo filtro tassonomia (`src/slm_baseline/labels.py`), 36.973 documenti, 41 giorni di raccolta (24/09–03/11/2026). Campione locale da 150 righe resta in `data/sample/` per test rapidi.

**Quale testo: `ocr_surrogates`.** Valori finti ma realistici al posto delle entità. Nel parquet completo **non** compaiono segnaposto tipo `[PIVA_1]`; la colonna `ocr_anonymized` non è presente. **3.954 segmenti (~5,6%)** hanno `ocr_surrogates` vuoto: vanno esclusi dalla valutazione e contati. I segnali lessicali discriminanti (IVA, ticket, importi) restano analizzabili via EDA O3.

**Limite da discutere nella tesi.** L'anonimizzazione può comunque cancellare informazione utile; nel campione da 150 righe comparivano ancora placeholder errati. Con i surrogati il rischio principale è testo vuoto e segmenti molto lunghi (p95 parole ~993, max >50k parole).

**Distribuzione.** Forte sbilanciamento: `documentazione_medica::documentazione_medica_altro` ~22,4k segmenti; `documento_di_spesa::scontrino` ~104. Macro rara con refuso `document_di_spesa_medica` e poche righe fuori perimetro (~54) vanno escluse. Metrica principale: **macro-F1** (bootstrap in `slm_baseline.metrics`).

**Split.** `data/splits_v1.parquet` attuale è stato generato sul campione da 150 righe: **rigenerarlo** con `notebooks/00_prepare_data_and_splits.ipynb` sul parquet completo prima delle baseline definitive.

## 4. Modelli

| Modello | Parametri | Licenza | Note |
|---|---|---|---|
| `Qwen/Qwen3-1.7B` | 1.7B | Apache 2.0 | Modalità ragionamento disattivata (`enable_thinking=False`) |
| `HuggingFaceTB/SmolLM3-3B` | 3B | Apache 2.0 | Ragionamento disattivato; il chat template inseriva la data di oggi nel prompt, rimossa con `/system_override` |
| `google/gemma-3-4b-it` | 4B | Gemma Terms of Use | Gated su HF (licenza da accettare); richiede bf16 (in fp16 va in overflow) |

**Licenza Gemma.** È "open weights", non open source in senso OSI. L'uso commerciale è consentito, ma con una policy di usi vietati, e i termini valgono anche per i modelli fine-tunati. Va accettata sulla pagina del modello su Hugging Face; su SageMaker si usa un `HF_TOKEN`.

**Candidati da valutare in seguito:** Qwen3-0.6B come limite inferiore, Qwen3-4B-Instruct-2507, Llama-3.2-3B-Instruct, Phi-4-mini-instruct, e un modello nativamente italiano (es. Velvet-2B). Come termine di paragone supervisionato serve anche un **encoder fine-tunato** (es. mmBERT, XLM-RoBERTa): con 70k esempi etichettati è il riferimento con cui confrontare un SLM fine-tunato.

## 5. Metodologia della baseline

Principio: **baseline e valutazione dopo il fine-tuning usano lo stesso split, lo stesso prompt e le stesse metriche**. Altrimenti la differenza misurata non è attribuibile al fine-tuning.

### 5.1 Split fisso

- Train / val / test ≈ 88% / 5% / 7%. Su 70k esempi il test è di circa 5.000, sufficiente per un intervallo di confidenza di circa ±1.3% sull'accuracy.
- Stratificato per `macro::micro`, raggruppato per `document_id` (`StratifiedGroupKFold`), seed 42.
- Calcolato una sola volta e salvato in `data/splits_v1.parquet`. Se il dataset cambia, il notebook si ferma: serve uno split nuovo (`splits_v2`) e vanno rifatte le baseline.

### 5.2 Prompt

Il **registro** è in `src/slm_baseline/prompts.py`: chiave `prompt.version` nel YAML dell’esperimento (`minimal`, `enriched_v1`, …). La variabile `PROMPT_VERSION = "v1"` resta come **tag di compatibilità** con i notebook legacy (equivale al comportamento `minimal` nei messaggi chat).

- **system**: istruzioni + (per `enriched_v1`) corpi letti da `src/slm_baseline/prompt_variants/enriched_v1/*.md` + elenco delle micro ammesse;
- **user**: `Macro: <macro>` + testo del segmento;
- **assistant** (solo nel training): il nome della micro.

Nuove varianti = nuova cartella sotto `prompt_variants/` e nuova chiave nel registro. Il manifest dell’esperimento salva `prompt_sha256` per confrontabilità.

### 5.3 Lunghezza di contesto (`max_seq_length`)

Obiettivo: coprire il **100% dei segmenti con un margine del 20%** sopra il più lungo, misurato con tutti e tre i tokenizer e arrotondato al multiplo di 1024.

- **Legacy:** calcolo in `00_prepare_data_and_splits.ipynb` → `data/run_config.json`.
- **Esperimenti:** in `configs/base.yaml`, sezione `sequence`: modi `manual`, `manual_plus_prompt` (base + overhead prompt vuoto), `computed` (come sopra). Il valore finale è in `outputs/experiments/<id>/run_config.json`.

Sul campione: segmento più lungo 12.550 token (Qwen3), quindi `max_seq_length` = **15.360**. Il controllo per categoria ha mostrato che la media inganna: `referto_del_pronto_soccorso` supera 4096 token nel 25% dei casi.

Se un esempio supera comunque il limite, si tronca **solo il testo del segmento**, mai istruzioni o risposta. Il budget per il testo si calcola sottraendo la lunghezza del prompt di ciascuna macro.

### 5.4 Run di baseline

| Run | Cosa misura |
|---|---|
| `majority` | Micro più frequente per macro (pavimento) |
| `random` | Micro casuale tra quelle ammesse (pavimento) |
| `tfidf_logreg` | TF-IDF su n-grammi di caratteri + regressione logistica, addestrata sul train |
| SLM `constrained` | Zero-shot, output vincolato alle micro ammesse (vLLM guided choice). **Metrica principale** |
| SLM `free` | Zero-shot libero + parsing: misura anche quanto il modello rispetta il formato |
| SLM `fewshot_constrained` | Come `constrained`, con 1 esempio per micro preso dal train (opzionale) |

Generazione deterministica (temperature 0, seed fisso), ragionamento disattivato per tutti i modelli.

### 5.5 Metriche

- **Macro-F1** (principale) e accuracy, con intervallo di confidenza al 95% via bootstrap.
- F1 pesata, precision/recall/F1 per classe.
- Accuracy e macro-F1 per macro, matrice di confusione per macro.
- Tasso di output non validi (solo modalità `free`).
- Throughput (esempi/s), percentuale di esempi troncati, token medi del prompt.

Ogni run salva `predictions__<slug>.parquet` e `metrics__<slug>.json`. In modalità legacy la cartella è `outputs/baseline/`; in modalità esperimenti è `outputs/experiments/<experiment_id>/runs/`. Metadati: revisione HF, `prompt_version`, `split_hash` / `comparable_group`, `config_hash`, eventuale `fold` per k-fold.

### 5.6 Workflow esperimenti (file, CLI, SageMaker)

1. **Config** — `configs/experiments/<nome>.yaml` con `extends: ../base.yaml` e variabili `${TESI_DATA_DIR}`, `${TESI_OUTPUT_DIR}`, `${TESI_S3_PREFIX}`.
2. **Prepare** — `python -m slm_baseline.cli prepare --config ...` crea `outputs/experiments/<id>/` con `config.resolved.yaml`, `manifest.json`, `splits.parquet`, `run_config.json`. Se la cartella esiste già con un `config_hash` diverso, il comando esce (o `--force`).
3. **Run** — `cli run` esegue classiche + SLM (sottoprocessi `run_baseline --experiment-dir ...`). Le run già presenti vengono saltate salvo `--rerun`.
4. **Compare** — `cli compare` aggrega in `outputs/comparison/summary.csv`; `slm_baseline.reporting.load_experiments` include righe aggregate k-fold (`__kfold_agg`).
5. **Sync** — `cli sync` copia la cartella esperimento su S3 con `s3fs` (opzionale).

Su SageMaker Studio: stesso repository, `uv sync --group sagemaker`, `export TESI_DATA_DIR=...` e `HF_TOKEN`, poi i comandi `cli` come in locale (o `uv run python -m slm_baseline.cli ...`). Notebook `00`/`01` possono impostare `CONFIG_PATH` verso lo stesso YAML.

## 6. Infrastruttura

| Dove | Cosa |
|---|---|
| Mac di sviluppo | Codice, test su campione da 150 righe, tokenizer. Niente vLLM (serve CUDA) |
| SageMaker Studio, `ml.g6e.2xlarge` (1× L40S 48 GB) | Dataset completo, baseline con vLLM, fine-tuning (notebook o training job) |

Il PC aziendale ha limitazioni (niente accesso a Hugging Face in locale); SageMaker invece raggiunge HF e PyPI.

Alternative valutate per l'istanza: `ml.g5.2xlarge` (A10G 24 GB) e `ml.g6.2xlarge` (L4 24 GB), più economiche ma più lente e con meno margine di memoria a contesti lunghi. La L40S supporta bf16, necessario per Gemma 3.

## 7. Stato attuale (5 novembre 2026)

**Fatto e verificato in locale (campione 150 righe e EDA su parquet completo):**

- [x] Modulo condiviso `src/slm_baseline/`: perimetro etichette, prompt, caricamento dati, split, troncamento, metriche, baseline classiche, formato dei run
- [x] `notebooks/EDA/00b_EDA_aic_data.ipynb` + `notebooks/EDA/eda_strategy.md` (O1–O7; dataset `data/df_sample_all.parquet`)
- [x] `00_prepare_data_and_splits.ipynb` e `01_baseline.ipynb` pronti; `run_baseline.py` testato con vLLM simulato
- [x] Prompt v1 verificato sui tre chat template

**Non ancora fatto:**

- [ ] Eseguire l'EDA completa end-to-end (probe bge-m3 e tokenizer Gemma richiedono HF/MPS)
- [ ] Rigenerare `splits_v1.parquet` e `run_config.json` sul dataset completo (notebook `00`)
- [ ] Esecuzione reale con vLLM su GPU (SageMaker)
- [ ] Baseline complete sui tre SLM e encoder fine-tunato
- [ ] Fine-tuning degli SLM e valutazione con lo stesso codice

## 8. Prossimi passi

1. **SageMaker, setup**: `uv sync --group sagemaker`, `HF_TOKEN`, licenza Gemma accettata.
2. **`00` sul dataset completo**: controllare la tabella dei segmenti più lunghi. Se il 100% + 20% supera il contesto nativo di Qwen3 (32k token), fissare un tetto e troncare solo quei casi.
3. **`01` con `LIMIT = 50`** come smoke test reale di vLLM sui tre modelli, poi `LIMIT = None`.
4. **Baseline encoder** (mmBERT o XLM-RoBERTa fine-tunato), nello stesso formato di output.
5. **Fine-tuning** (LoRA con Unsloth) usando lo stesso prompt, lo stesso split e lo stesso `max_seq_length`; valutazione con `save_run` per avere tutto nella stessa tabella.
6. **Esperimenti per la tesi**: curva di apprendimento (5k / 20k / 70k esempi), confronto zero-shot / few-shot / fine-tuned, effetto della lunghezza di contesto, errori per categoria.

## 9. Decisioni aperte

- **Troncamento dei segmenti molto lunghi**: solo le prime pagine (attuale) oppure prime e ultime pagine.
- **System prompt con descrizioni delle micro**: possibile `PROMPT_VERSION = "v2"`. Va deciso prima delle baseline definitive, perché cambiarlo dopo obbliga a rifarle.
- **Lista finale dei modelli**: i tre attuali più eventuali candidati dalla sezione 4.
- **Fine-tuning come notebook o come SageMaker Training Job.**
