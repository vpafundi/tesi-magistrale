# tesi-magistrale

Classificazione di segmenti di documenti assicurativi con Small Language Models (Qwen3-1.7B, SmolLM3-3B, Gemma-3-4B): dato il testo OCR di un segmento e la sua macrocategoria, il modello sceglie la microcategoria. La tesi misura quanto il fine-tuning migliora i modelli rispetto al loro uso senza addestramento e rispetto a baseline classiche.

**Obiettivo, scelte fatte, stato e prossimi passi: [docs/PROGETTO.md](docs/PROGETTO.md).**

**Istruzioni complete del framework (dettaglio di ogni file, CLI, YAML, SageMaker): [docs/FRAMEWORK.md](docs/FRAMEWORK.md).**

**Step 1–2 su Studio:** [docs/SAGEMAKER_STEPS_1_2.md](docs/SAGEMAKER_STEPS_1_2.md).

## Stato in breve

| Fase | Stato |
|---|---|
| Codice condiviso (prompt, split, metriche) | Fatto, testato in locale |
| Notebook split e `max_seq_length` (`00`) | Fatto, **da rigenerare** sul dataset completo |
| EDA dataset completo | Spostata nel progetto standalone `EDA/` (repo separata, vedi sotto) |
| Notebook baseline (`01`) | Fatto, da eseguire su SageMaker con vLLM |
| Baseline encoder | Da fare |
| Fine-tuning e valutazione | Da fare |

## Struttura

| Path | Contenuto |
|---|---|
| `docs/PROGETTO.md` | Documento di progetto |
| `src/slm_baseline/labels.py` | Perimetro macro → micro ammesse |
| `src/slm_baseline/prompts.py` | Prompt unico per baseline e fine-tuning (`PROMPT_VERSION`) |
| `src/slm_baseline/data/` | Caricamento parquet, filtro perimetro, split stratificato e raggruppato per documento (`data/core.py`) |
| `src/slm_baseline/inference.py` | Chat template, lunghezze in token, troncamento, parsing dell'output |
| `src/slm_baseline/metrics.py` | Accuracy, macro-F1 con IC bootstrap, metriche per macro, confusion matrix |
| `src/slm_baseline/classic.py` | Baseline majority, random, TF-IDF + regressione logistica |
| `src/slm_baseline/reporting.py` | Formato unico dei run su disco, tabella riassuntiva e `load_experiments` |
| `src/slm_baseline/run_baseline.py` | Un modello × una modalità con vLLM (`--experiment-dir` legge split e config dall’esperimento) |
| `src/slm_baseline/cli.py` | Pipeline da YAML: `prepare`, `classic`, `slm`, `run`, `compare`, `sync` |
| `configs/` | `base.yaml` + esperimenti in `configs/experiments/*.yaml` |
| `notebooks/00_prepare_data_and_splits.ipynb` | Split fisso e scelta di `max_seq_length` |
| `notebooks/01_baseline.ipynb` | Baseline classiche + SLM zero-shot, tabella, grafici, analisi errori |
| `notebooks/02_compare_experiments.ipynb` | Tabella di confronto tra esperimenti (`load_experiments`) |

`data/` (parquet, split, `run_config.json`) e `outputs/` non vanno in git: contengono testo OCR, anche se anonimizzato.

**EDA:** l'analisi esplorativa (O1–O7, curation del dataset, run store) vive nel progetto standalone sotto `EDA/` (gitignored qui, destinato a una repo GitHub propria). Vedi `EDA/README.md`.

## Esperimenti (YAML + CLI)

Ogni esperimento è un file in `configs/experiments/` che **estende** `configs/base.yaml`. Variabili `${TESI_*}` nel YAML si leggono dall’ambiente (stesso repo su Mac e su SageMaker Studio).

| Variabile | Ruolo |
|---|---|
| `TESI_REPO_ROOT` | Root del repo (default: calcolata dal package) |
| `TESI_DATA_DIR` | Directory dati (default: `<repo>/data`) — path relativi nel YAML sono sotto questa cartella |
| `TESI_OUTPUT_DIR` | Output (default: `<repo>/outputs`) — qui vive `experiments/<experiment_id>/` |
| `TESI_S3_PREFIX` | Prefisso S3 per `cli sync` (es. `s3://bucket/tesi`), opzionale |

Comandi tipici:

```bash
# Da repository root, con uv (core + gruppo dev per pytest)
uv sync --group dev

export HF_TOKEN=hf_...   # Gemma gated
export TESI_DATA_DIR=/path/al/dataset    # oppure default
export TESI_OUTPUT_DIR=/path/agli/outputs
export TESI_S3_PREFIX=s3://mio-bucket/tesi   # solo se usi sync

uv run python -m slm_baseline.cli prepare --config configs/experiments/holdout_s42_minimal.yaml
uv run python -m slm_baseline.cli run --config configs/experiments/holdout_s42_minimal.yaml
uv run python -m slm_baseline.cli compare
uv run python -m slm_baseline.cli sync --config configs/experiments/holdout_s42_minimal.yaml
# oppure tutti gli esperimenti con prefix S3 configurato:
uv run python -m slm_baseline.cli sync --all
```

Layout output: `outputs/experiments/<id>/` contiene `config.resolved.yaml`, `manifest.json`, `splits.parquet`, `run_config.json`, cartella `runs/` con `metrics__*` e `predictions__*`. Se rilanci `prepare` con un altro config nella stessa cartella, il comando si ferma (`--force` per sovrascrivere). Le run già presenti in `runs/` vengono saltate salvo `--rerun`.

Test automatici: `uv run --group dev pytest tests/ -q`.

## Esecuzione

### Locale (Mac, senza GPU)

```bash
uv sync
uv run jupyter lab
```

Mettere il campione in `data/sample/df_sample_150.parquet` e il dataset completo in `data/df_sample_all.parquet`. Gira tutto tranne la sezione vLLM di `01_baseline.ipynb`. Per l'EDA usare il progetto standalone in `EDA/` (vedi `EDA/README.md`).

### SageMaker Studio (`ml.g6e.2xlarge`)

```bash
uv sync --group sagemaker   # vLLM + s3fs + pacchetto in editable
export HF_TOKEN=hf_...      # licenza accettata su huggingface.co/google/gemma-3-4b-it
```

1. `notebooks/00_prepare_data_and_splits.ipynb` con `DATA_PATH` sul parquet completo (anche `s3://…`). Produce `data/splits_v1.parquet` e `data/run_config.json`.
2. `notebooks/01_baseline.ipynb` con `LIMIT = 50` come smoke test, poi `LIMIT = None`. Risultati in `outputs/baseline/` (`summary.csv`, predizioni e metriche per run).

## Regole per confronti validi

- Due run sono confrontabili solo se condividono lo stesso **`split_hash`** (stesso test): la colonna `comparable_group` in `load_experiments` coincide con quel hash.
- **Modalità legacy:** stesso `splits_v1.parquet` e `run_config.json` sotto `data/`. **Modalità esperimenti:** split e `run_config` stanno in `outputs/experiments/<id>/`.
- Il prompt è versionato nel YAML (`prompt.version`, es. `minimal` / `enriched_v1`) e nel manifest (`prompt_sha256`). `PROMPT_VERSION` in `prompts.py` resta un tag legacy compatibile con `minimal`.
- `max_seq_length` viene calcolato in `prepare` secondo `sequence.max_seq_len_mode` e scritto in `run_config.json` dell’esperimento.
- Ogni valutazione passa da `slm_baseline.reporting.save_run` (campi esperimento opzionali: `experiment_id`, `config_hash`, `split_hash`, `fold`, `comparable_group`).
