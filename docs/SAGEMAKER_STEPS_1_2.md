# SageMaker Studio — Step 1 e 2 (prepare + classiche)

## Setup una tantum

```bash
cd /home/sagemaker-user/tesi-magistrale   # o path clone
# uv (se manca): curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --no-default-groups --group sagemaker
# LoRA + vLLM:
uv sync --no-default-groups --group train --group sagemaker
export VLLM_USE_FLASHINFER_SAMPLER=0   # Studio senza CUDA toolkit headers
export HF_TOKEN=hf_...                   # Gemma / rate limit Hub
export TESI_REPO_ROOT=$PWD
export TESI_DATA_DIR=$PWD                # path YAML tipo data/df_sample_all.parquet
export TESI_OUTPUT_DIR=$PWD/outputs
export TESI_LOG_LEVEL=INFO    # oppure DEBUG per più dettaglio pipeline/SLM
```

Copia il parquet completo in `data/df_sample_all.parquet` (non è in git).

## Opzione A — Solo CLI (consigliata)

```bash
uv run python -m slm_baseline.cli prepare --config configs/experiments/holdout_s42_minimal.yaml
uv run python -m slm_baseline.cli classic  --config configs/experiments/holdout_s42_minimal.yaml
```

Artefatti: `outputs/experiments/holdout_s42_minimal/`.

## Esperimenti per macro (lavoro quotidiano)

Split canonico **una tantum** sul corpus intero, poi un prepare/classic per macro (dati già filtrati in pipeline):

```bash
uv run python -m slm_baseline.cli prepare --config configs/experiments/holdout_s42_minimal.yaml

uv run python -m slm_baseline.cli prepare --config configs/experiments/holdout_s42__documentazione_medica.yaml
uv run python -m slm_baseline.cli classic  --config configs/experiments/holdout_s42__documentazione_medica.yaml
```

YAML figli: `holdout_s42__denuncia`, `holdout_s42__documentazione_medica`, `holdout_s42__documento_di_spesa`, `holdout_s42__documento_di_spesa_medica`.

Training LoRA (Studio, gruppo `train`; usa `tmux` per sessioni lunghe):

```bash
uv run python -m slm_baseline.cli train --config configs/experiments/holdout_s42__documentazione_medica.yaml --dry-run
```

Esperimento **enriched + bilanciamento train** (prompt `enriched_v1`, cap `*_altro`):

```bash
uv run python -m slm_baseline.cli prepare --config configs/experiments/holdout_s42__documentazione_medica_enriched.yaml
uv run python -m slm_baseline.cli train --config configs/experiments/holdout_s42__documentazione_medica_enriched.yaml
uv run python -m slm_baseline.cli slm --config configs/experiments/holdout_s42__documentazione_medica_enriched.yaml --rerun
```

Metriche globali (dopo SLM su tutte le macro, stesso slug):

```bash
uv run python -m slm_baseline.cli compare-family --slug gemma-3-4b-it__constrained__L15360
```

## Opzione B — Notebook

1. **`notebooks/00_prepare_data_and_splits.ipynb`**: `USE_EXPERIMENT_YAML=True`, `USE_CLI_PREPARE=True` → Step 1 + EDA lunghezze token.
2. **`notebooks/01_baseline.ipynb`**: `USE_CLI_CLASSIC=True` → Step 2 in Config; poi sezione SLM (Step 3).

Smoke locale: `configs/experiments/smoke_sample150.yaml`.

## Validazione

- `manifest.json`, `splits.parquet`, `run_config.json` presenti.
- `runs/metrics__majority.json` (+ random, tfidf_logreg).
- Stesso `splits_hash` in manifest e in `run_info.comparable_group`.

Dettaglio: [FRAMEWORK.md](FRAMEWORK.md).