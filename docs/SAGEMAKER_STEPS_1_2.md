# SageMaker Studio — Step 1 e 2 (prepare + classiche)

## Setup una tantum

```bash
cd /home/sagemaker-user/tesi-magistrale   # o path clone
pip install -r requirements-sagemaker.txt
export HF_TOKEN=hf_...                   # Gemma
export TESI_REPO_ROOT=$PWD
export TESI_DATA_DIR=$PWD                # path YAML tipo data/df_sample_all.parquet
export TESI_OUTPUT_DIR=$PWD/outputs
```

Copia il parquet completo in `data/df_sample_all.parquet` (non è in git).

## Opzione A — Solo CLI (consigliata)

```bash
python -m slm_baseline.cli prepare --config configs/experiments/holdout_s42_minimal.yaml
python -m slm_baseline.cli classic  --config configs/experiments/holdout_s42_minimal.yaml
```

Artefatti: `outputs/experiments/holdout_s42_minimal/`.

## Opzione B — Notebook

1. **`notebooks/00_prepare_data_and_splits.ipynb`**: `USE_EXPERIMENT_YAML=True`, `USE_CLI_PREPARE=True` → Step 1 + EDA lunghezze token.
2. **`notebooks/01_baseline.ipynb`**: `USE_CLI_CLASSIC=True` → Step 2 in Config; poi sezione SLM (Step 3).

Smoke locale: `configs/experiments/smoke_sample150.yaml`.

## Validazione

- `manifest.json`, `splits.parquet`, `run_config.json` presenti.
- `runs/metrics__majority.json` (+ random, tfidf_logreg).
- Stesso `splits_hash` in manifest e in `run_info.comparable_group`.

Dettaglio: [FRAMEWORK.md](FRAMEWORK.md).