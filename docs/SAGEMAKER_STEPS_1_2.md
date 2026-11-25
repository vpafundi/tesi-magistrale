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

Copia il parquet in `data/` (non è in git): legacy `data/df_sample_all.parquet` (YAML: `data/df_sample_all.parquet` è relativo a `data_root`) oppure **post-EDA** `data/curated/df_segments_curated_v1.parquet` (YAML: `curated/df_segments_curated_v1.parquet`).

### tmux su Studio (obbligatorio per job lunghi)

Senza tmux, chiudere il terminale Jupyter interrompe `classic`, `slm` e `train`.

Se `tmux: command not found` (comune). Studio **nuovo** (JupyterLab / Code Editor) usa **apt**, non yum/dnf:

```bash
sudo apt-get update && sudo apt-get install -y tmux
```

Se `sudo` o `apt-get` non ci sono (kernel gateway minimale), prova Conda nel terminale dell’app:

```bash
conda install -y -c conda-forge tmux
# poi: hash -r && which tmux
```

Diagnostica rapida: `cat /etc/os-release` e `command -v sudo apt-get conda`.

**Se non installi tmux** usa sempre `nohup` + log (sotto): lascia aperta l’app Studio (Code Editor / JupyterLab), non solo una tab del browser.

Alternativa senza tmux (terminale bash dell’app, non cella notebook):

```bash
nohup uv run python -m slm_baseline.cli train --config configs/experiments/${FIG}.yaml \
  > outputs/logs/train_${FIG}.log 2>&1 &
tail -f outputs/logs/train_${FIG}.log
```

Per `slm` da **notebook Jupyter**: preferisci un **terminale** (System terminal / bash) con tmux o `nohup`, non una cella — il kernel può restare in timeout mentre vLLM carica i pesi (anche 10–15 min al primo avvio).

```bash
tmux new -s tesi          # nuova sessione
# … lancia i comandi qui dentro …
# Ctrl-b poi d              detach (job continua)
tmux attach -t tesi       # riattacca
tmux ls                     # elenco sessioni
```

Usa una sessione dedicata per figlio, es. `tesi_curated_minimal`, `tesi_curated_enriched`.

## Famiglia curated — documentazione_medica (dataset EDA, nuovo split)

Split master sul curated intero (~112k segmenti), poi due esperimenti scoped (minimal vs enriched), stesso test. Train **senza** resampling; Unsloth **8×2** (`batch_size: 8`, `grad_accum: 2`).

**Lancia tutto dentro tmux** (vedi sopra). Variabili d’ambiente nella stessa shell della sessione.

```bash
tmux new -s tesi_curated_minimal

cd /home/sagemaker-user/tesi-magistrale
export TESI_REPO_ROOT=$PWD TESI_DATA_DIR=$PWD TESI_OUTPUT_DIR=$PWD/outputs
export VLLM_USE_FLASHINFER_SAMPLER=0

# 1) Master (solo split + manifest) — una tantum, anche in tmux
uv run python -m slm_baseline.cli prepare --config configs/experiments/holdout_curated_v1.yaml

# 2) Figlio: prepare → classic → SLM zero-shot → train → SLM LoRA (un comando alla volta)
FIG=holdout_curated_v1__documentazione_medica   # oppure holdout_curated_v1__documentazione_medica_enriched
uv run python -m slm_baseline.cli prepare --config configs/experiments/${FIG}.yaml
uv run python -m slm_baseline.cli classic  --config configs/experiments/${FIG}.yaml
uv run python -m slm_baseline.cli slm --config configs/experiments/${FIG}.yaml --rerun
uv run python -m slm_baseline.cli train --config configs/experiments/${FIG}.yaml
uv run python -m slm_baseline.cli slm --config configs/experiments/${FIG}.yaml --rerun

# compare (dopo minimal + enriched, o quando ti serve)
uv run python -m slm_baseline.cli compare
```

YAML: `holdout_curated_v1.yaml` (master), `holdout_curated_v1__documentazione_medica.yaml`, `holdout_curated_v1__documentazione_medica_enriched.yaml`.

Non confrontabile con `holdout_s42_*` (`splits_hash` diverso). Tracciabilità EDA: `data/curated/curation_manifest.json`.

### TF-IDF + LogReg con Optuna (CPU, train+val CV)

Esperimento dedicato: `holdout_curated_v1__documentazione_medica_tfidf_tuned.yaml` (stesso split del minimal, solo baseline ottimizzata).

```bash
tmux new -s tesi_tfidf_tuned
cd /home/sagemaker-user/tesi-magistrale
export TESI_REPO_ROOT=$PWD TESI_DATA_DIR=$PWD TESI_OUTPUT_DIR=$PWD/outputs
FIG=holdout_curated_v1__documentazione_medica_tfidf_tuned
mkdir -p outputs/logs

uv run python -m slm_baseline.cli prepare --config configs/experiments/${FIG}.yaml
uv run python -m slm_baseline.cli classic --config configs/experiments/${FIG}.yaml 2>&1 | tee outputs/logs/classic_${FIG}.log
uv run python -m slm_baseline.cli compare
```

**Log di avanzamento** (in `classic_*.log`):

```bash
tail -f outputs/logs/classic_${FIG}.log
grep -E 'tuning (start|trial|study done|refit)' outputs/logs/classic_${FIG}.log
```

Righe utili: `tuning trial 12/50 fold 3/5`, `COMPLETE | cv_f1=... | ETA`, `tuning study done`.

**Resume** se interrompi: rilancia lo stesso comando `classic` **senza** `--rerun`. Optuna riprende da `runs/optuna__tfidf_logreg_tuned.log` (journal). Con `--rerun` si cancella lo studio e si riparte da zero.

**Artefatti:** `runs/tuning__tfidf_logreg_tuned.csv/json`, `runs/optuna__tfidf_logreg_tuned.log`, predizioni con colonne `proba__*`. Backup su S3: `uv run python -m slm_baseline.cli sync --config configs/experiments/${FIG}.yaml`.

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

Training LoRA (Studio, gruppo `train`; **tmux o nohup**):

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