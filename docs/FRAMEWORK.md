# Framework `slm_baseline` — istruzioni e riferimento file

Guida operativa al package che gestisce **dati**, **split**, **prompt**, **baseline classiche e SLM**, **esperimenti da YAML** e **confronto tra run**. Il contesto scientifico della tesi resta in [PROGETTO.md](PROGETTO.md); qui si descrive **come si usa il codice**.

---

## Indice

1. [Concetti](#1-concetti)
2. [Installazione](#2-installazione)
3. [Variabili d'ambiente e config YAML](#3-variabili-dambiente-e-config-yaml)
4. [Layout su disco](#4-layout-su-disco)
5. [CLI — comandi](#5-cli--comandi)
6. [Flussi consigliati](#6-flussi-consigliati)
7. [Dettaglio file — package `src/slm_baseline`](#7-dettaglio-file--package-srcslm_baseline)
8. [Config predefiniti](#8-config-predefiniti)
9. [Notebook](#9-notebook)
10. [Test automatici](#10-test-automatici)
11. [Naming delle run (slug)](#11-naming-delle-run-slug)
12. [K-fold](#12-k-fold)
13. [Modalità legacy (senza YAML)](#13-modalità-legacy-senza-yaml)
14. [SageMaker Studio](#14-sagemaker-studio)
15. [Estendere il framework](#15-estendere-il-framework)
16. [FAQ / errori comuni](#16-faq--errori-comuni)

---

## 1. Concetti

### Architettura a layer (pipeline OOP)

Dal refactor 2025 il package è organizzato in layer; la **CLI** delega a `BaselineExperimentOrchestrator` (`slm_baseline.pipeline.orchestrator`), che esegue stage idempotenti su `ExperimentContext`.

```text
configs/experiments/*.yaml
        │
        ▼
  build_context (factory) ──► ExperimentContext
        │
        ├─ PreparePipeline (stages/prepare.py)
        ├─ ClassicEvaluatePipeline (stages/classic.py)
        ├─ SlmEvaluatePipeline + SubprocessVllmRunner → run_baseline / SlmEvaluationJob
        ├─ ComparePipeline / SyncPipeline (stages/compare_sync.py)
        └─ output: outputs/experiments/<id>/
```

| Layer | Package | Ruolo |
|-------|---------|--------|
| Config | `config/` | YAML `extends`, env, dataclass, hash manifest |
| Dati | `data/` | parquet, perimetro, testo, split holdout/kfold |
| Dominio | `domain/` | tassonomia micro (`labels`) |
| Prompt | `prompt/` | varianti chat (`minimal`, `enriched_v1`) |
| Sequenza | `sequence/` | budget token, `build_prompts`, parse output |
| Valutazione | `evaluation/classic`, `evaluation/slm` | baseline CPU e subprocess vLLM |
| Artefatti | `artifacts/` | path esperimento, metriche, `save_run` / `load_experiments` |
| Pipeline | `pipeline/` | context, eccezioni, stage, orchestrator |
| Logging | `logging/` | logger unico (`slm_baseline.pipeline`) passato via orchestrator / ``ctx.logger`` |

**Compatibilità:** molti import storici restano su moduli flat alla root (`experiment.py`, `labels.py`, …) come **re-export** verso i layer. I dati si importano solo come **package** `slm_baseline.data` (cartella `data/`; non esiste più `data.py`, altrimenti Python non risolverebbe il package).

**Aggiungere uno stage prepare:** implementa `PipelineStage` con `name` + `run(ctx)` in `pipeline/stages/`, registralo in `PreparePipeline.stages`.

**Logging:** `setup_pipeline_logger()` in CLI; :class:`BaselineExperimentOrchestrator` accetta un ``logger`` opzionale; ogni :class:`ExperimentContext` espone ``ctx.logger``. Messaggi su stderr, formato `HH:MM:SS | LEVEL | slm_baseline.pipeline | …`. In notebook: ``from slm_baseline.logging import setup_pipeline_logger`` e ``BaselineExperimentOrchestrator(logger=setup_pipeline_logger())``.

**Docstring nel codice:** API pubbliche pipeline (`orchestrator`, stage, registry, runner) usano stile **Google** in italiano: breve descrizione, sezioni `Args`, `Returns`, `Raises` (e `Attributes` sulle dataclass). Per estrarle in IDE: `help(BaselineExperimentOrchestrator.prepare)` o documentazione Sphinx/pdoc se la abiliti in seguito.

| Termine | Significato |
|--------|-------------|
| **Esperimento** | Una configurazione completa definita in `configs/experiments/*.yaml` (dataset, split, prompt, sequenza, modelli). Identificato da `experiment_id` e da `config_hash` nel manifest. |
| **Run** | Una singola valutazione: es. `majority`, oppure `Qwen3-1.7B` × `constrained` × (opzionale) fold k. Ogni run produce `predictions__<slug>.parquet` e `metrics__<slug>.json`. |
| **Prepare** | Fase che crea la cartella `outputs/experiments/<id>/` con split, manifest, `max_seq_len`, senza chiamare vLLM. |
| **Confrontabilità** | Due run sono confrontabili solo se condividono lo stesso **`split_hash`** (stesso insieme di test). In tabella compare compare la colonna `comparable_group` (= `split_hash`). |
| **Tracking** | Solo su file (nessun MLflow). Opzionale copia su S3 con `cli sync`. |

```text
experiment.yaml  →  prepare  →  outputs/experiments/<id>/
                                    ├─ classic (CPU)
                                    └─ slm (sottoprocessi vLLM)
                                          →  runs/*
                                    →  compare → summary.csv
                                    →  sync (S3, opzionale)
```

---

## 2. Installazione

### Locale (Mac, sviluppo e test)

```bash
cd tesi-magistrale
uv sync --group dev    # core + pytest; default-groups include preprocess/eval/eda
```

- **Senza GPU:** `prepare`, `classic`, test pytest; **no** vLLM.
- **Con tokenizer / lunghezze:** gruppo `eval` (`transformers`, `matplotlib`).

### SageMaker (baseline SLM)

```bash
uv sync --group sagemaker   # vllm, s3fs, progetto installato (vedi pyproject.toml)
export HF_TOKEN=...
```

Se `uv` non è sul kernel Studio: `curl -LsSf https://astral.sh/uv/install.sh | sh` oppure `uv pip install -r requirements-sagemaker.txt`.

---

## 3. Variabili d'ambiente e config YAML

### Variabili `${...}` nel YAML

| Variabile | Default se assente | Uso |
|-----------|-------------------|-----|
| `TESI_REPO_ROOT` | root del repo (da `experiment.py`) | Percorsi assoluti interni |
| `TESI_DATA_DIR` | `<repo>/data` | Base per `dataset.path` relativi |
| `TESI_OUTPUT_DIR` | `<repo>/outputs` | `outputs/experiments/<experiment_id>/` |
| `TESI_S3_PREFIX` | vuoto | Destinazione `cli sync` (es. `s3://bucket/tesi`) |

Impostale in shell o nel job Studio; il CLI accetta anche `--output-root` (equivale a impostare `TESI_OUTPUT_DIR` per quella invocazione).

### Struttura YAML

- **`configs/base.yaml`**: default condivisi (`experiment_id` placeholder `replace_me` — va sovrascritto negli esperimenti).
- **`configs/experiments/<nome>.yaml`**: `extends: ../base.yaml` + override.

Sezioni principali:

| Sezione | Campi chiave |
|---------|----------------|
| `experiment_id` | Nome cartella sotto `outputs/experiments/` |
| `scope` | `macro_category`: filtra segmenti subito dopo il perimetro (prepare + classic + train) |
| `dataset` | `path`, `text_column`, `sample: {n, frac, seed}` |
| `split` | `strategy`, `seed`, `test_frac`, `val_frac`, `k`, `assignment_from` (id master per ereditare split) |
| `training` | `enabled`, `base_model`, `unsloth.*` per LoRA (Unsloth) |
| `prompt` | `version: minimal \| enriched_v1` |
| `sequence` | `max_seq_len_mode`, `max_seq_len`, `tokenizers`, `headroom`, `multiple`, `token_sample_rows` |
| `runs` | `classic`, `slm: [{model, modes}]`, `eval_split`, `limit`, `seed`, `gpu_mem` |
| `sync` | `s3_prefix` |

### Modi `max_seq_len_mode`

| Modo | Comportamento |
|------|----------------|
| `manual` | Usa `sequence.max_seq_len` così com'è (nessun download tokenizer in prepare). |
| `manual_plus_prompt` | `max_seq_len` + massimo overhead prompt (system + template, testo vuoto) su tutte le macro e i tokenizer elencati, arrotondato a `multiple`. |
| `computed` | Lunghezza massima sequenza sul campione + `headroom`, arrotondata (come `recommend_max_seq_len` nei notebook). |

Il valore finale è in `run_config.json` → chiave `max_seq_len`.

---

## 4. Layout su disco

Dopo `cli prepare`:

```text
outputs/experiments/<experiment_id>/
  config.resolved.yaml    # YAML finale (extends + env)
  manifest.json           # config_hash, dataset_sha256, splits_hash, prompt_*, git_commit, ...
  splits.parquet          # holdout: colonne split; kfold: colonna fold
  run_config.json         # max_seq_len, prompt_version, dataset_path, limit, ...
  runs/
    metrics__<slug>.json
    predictions__<slug>.parquet

outputs/comparison/
  summary.csv             # da cli compare
```

### Regole di immutabilità e resume

- Se la cartella esperimento esiste e `manifest.json` ha un **`config_hash` diverso**, `prepare` **esce** (usa nuovo `experiment_id` o `prepare --force`).
- Se `runs/metrics__<slug>.json` esiste, `classic` / `slm` / `run_baseline` **saltano** quella run salvo `--rerun`.
- `prepare --force` sovrascrive manifest/split/run_config ma **non cancella** run vecchie in `runs/` (puoi avere slug misti; in genere conviene un `experiment_id` nuovo).

---

## 5. CLI — comandi

Entry point:

```bash
uv run python -m slm_baseline.cli [--output-root PATH] <comando> ...
```

| Comando | Cosa fa | Flag utili |
|---------|---------|------------|
| `prepare` | Carica parquet, perimetro, sample, split, calcola `max_seq_len`, scrive cartella esperimento | `--config`, `--force` |
| `classic` | `majority`, `random`, `tfidf_logreg` su `runs.eval_split` | `--config`, `--rerun` |
| `train` | Export JSONL + subprocess Unsloth se `training.enabled` | `--config`, `--rerun`, `--dry-run` |
| `slm` | Subprocess vLLM (`--lora-adapter` se adapter in manifest) | `--config`, `--rerun`, `--dry-run` |
| `run` | `prepare` + `classic` + `slm` in sequenza | `--force`, `--rerun` |
| `compare` | Scansiona tutti gli esperimenti, stampa tabella, salva `comparison/summary.csv` | (usa `TESI_OUTPUT_DIR` o `--output-root`) |
| `compare-family` | Unisce predizioni `holdout_s42__*` e metriche globali | `--slug`, `--family-prefix` |
| `sync` | Copia ricorsiva cartella esperimento su `sync.s3_prefix/experiments/<id>` (s3fs) | `--config` oppure `--all` |

Esempi:

```bash
uv run python -m slm_baseline.cli prepare --config configs/experiments/smoke_sample150.yaml
uv run python -m slm_baseline.cli classic --config configs/experiments/smoke_sample150.yaml
uv run python -m slm_baseline.cli slm --config configs/experiments/holdout_s42_minimal.yaml --dry-run
uv run python -m slm_baseline.cli run --config configs/experiments/holdout_s42_minimal.yaml
uv run python -m slm_baseline.cli compare
uv run python -m slm_baseline.cli sync --all
```

Implementazione: [`src/slm_baseline/cli.py`](../src/slm_baseline/cli.py).

---

## 6. Flussi consigliati

### Smoke test locale (solo classiche, 150 righe)

1. Parquet in `data/sample/df_sample_150.parquet`.
2. `uv sync --group dev`
3. `uv run python -m slm_baseline.cli run --config configs/experiments/smoke_sample150.yaml`  
   (`runs.slm: []` nel YAML — niente GPU.)

### Baseline SLM complete (Studio)

1. `TESI_DATA_DIR` → parquet completo; `HF_TOKEN` per Gemma.
2. `prepare` per `holdout_s42_minimal` e, se serve confronto prompt, `holdout_s42_enriched_v1`.
3. `run` (o `slm` dopo prepare già fatto).
4. `compare` → tabella unica; opzionale `sync --all`.

### Confrontare prompt minimal vs enriched

- Stesso `split.seed`, stesso `test_frac`/`val_frac`, stesso dataset **senza** `sample` → stesso `splits_hash` solo se anche `experiment_id` diversi ma stessa logica di split su stessi segmenti.  
- In pratica: due YAML con **stessi parametri split e dataset** producono lo stesso `splits_hash`; il `config_hash` differisce per `prompt`/`sequence`. Le run SLM restano confrontabili tramite `comparable_group`.

---

## 7. Dettaglio file — package `src/slm_baseline`

Implementazione nei **layer** sotto (`config/`, `data/`, `pipeline/`, …). Dove esiste un file omonimo alla root (`labels.py`, `inference.py`, …) è solo uno **shim di compatibilità**; l’implementazione sta nel sottopackage indicato.

| Import tipico | Implementazione |
|---------------|-----------------|
| `slm_baseline.data` | Package `data/` (`data/core.py`) — **nessun** `data.py` in root |
| `slm_baseline.labels` | `domain/labels.py` |
| `slm_baseline.experiment` | `config/experiment.py` |
| `slm_baseline.prompts` | `prompt/core.py` |
| `slm_baseline.inference` | `sequence/core.py` |
| `slm_baseline.classic` | `evaluation/classic/predictors.py` |
| `slm_baseline.metrics` | `artifacts/metrics.py` |
| `slm_baseline.reporting` | `artifacts/reporting.py` |

### `__init__.py`

Docstring del package. Nessuna API esportata al root; si importano i sottomoduli (`from slm_baseline.data import ...`).

### `domain/labels.py` (import anche `from slm_baseline import labels`)

- **`MICRO_PER_MACRO`**: dizionario macro → lista micro ammesse (perimetro del task).
- **`INVALID_LABEL`**: etichetta per output SLM non parsabile.
- **`is_in_perimeter(macro, micro)`**, **`composite_label`**, **`all_composite_labels()`**: usati da metriche e TF-IDF.

Tutto ciò che è fuori perimetro viene scartato in `filter_perimeter` prima di split e training.

### Package `data/` (`data/core.py`, `data/splits_strategy.py`)

API pubblica via `from slm_baseline.data import …` (re-export da `data/__init__.py`).

| Funzione | Ruolo |
|----------|--------|
| `load_segments(path)` | Legge parquet (anche `s3://` se pyarrow/s3fs), valida schema (`record_key` univoco, colonne obbligatorie). |
| `filter_perimeter(df)` | `(in_perimetro, fuori_perimetro)`. |
| `prepare_text(df, text_column)` | Aggiunge colonna `text` normalizzata; restituisce righe con testo vuoto escluse. |
| `sample_dataset(df, n, frac, seed)` | Campione stratificato su `macro::micro` (per YAML `dataset.sample`). |
| `make_splits(...)` | Holdout train/val/test, stratificato, **raggruppato per `document_id`**. |
| `make_kfold_splits(df, k, seed)` | Tabella con colonna `fold` (0..k-1); test fold per outer CV. |
| `split_for_kfold_fold` / `apply_kfold_assignments` | Deriva `train`/`val`/`test` per un fold k. |
| `check_no_group_leakage(splits)` | Assert: nessun `document_id` in più split. |
| `attach_splits(df, splits)` | Merge `record_key` → colonna `split`. |
| `label_distribution(splits)` | Conteggi per macro/micro/split (EDA). |

`data/splits_strategy.py`: strategie **holdout** / **kfold** usate da `PreparePipeline` (`split_strategy_for`).

### `prompt/core.py` (import anche `prompts.py`)

- **`PROMPT_VERSION`**: tag legacy `"v1"` (comportamento allineato a **`minimal`** in `run_baseline`).
- **`prompt_footprint(version)`**: hash per manifest (`prompt_sha256`).
- **`system_prompt_for_macro(macro, prompt_version)`**  
  - `minimal`: istruzioni base + elenco micro.  
  - `enriched_v1`: base + markdown in `prompt_variants/enriched_v1/<macro>.md` + elenco micro.
- **`build_messages(macro, text, micro=..., few_shot=..., prompt_version=...)`**: lista messaggi chat (usata da inferenza e calcolo lunghezze).

Aggiungere una variante: nuova cartella sotto `prompt_variants/` + ramo in `system_prompt_for_macro` e in `prompt_footprint`.

### `prompt_variants/enriched_v1/*.md`

Testi lunghi per macro (denuncia, documentazione medica, documenti di spesa). Vanno nel **system**; il segmento OCR resta nel turno **user** (`user_content`).

### `sequence/core.py` (import anche `inference.py`)

| Funzione | Ruolo |
|----------|--------|
| `adapt_messages` | Fix deterministici template (es. SmolLM3 + `/system_override`). |
| `render_prompt` / `encode_prompt` | Chat template Hugging Face, `enable_thinking=False`. |
| `token_length_report(tokenizer, df, ..., prompt_version)` | Lunghezza sequenza training per riga + tabella coverage per soglie. |
| `recommend_max_seq_len` | Max tra tokenizer + headroom, arrotondamento. |
| `compute_max_seq_len_for_experiment(...)` | Tre modi `manual` / `manual_plus_prompt` / `computed`. |
| `truncate_tokens` | Troncamento **solo** testo segmento (head). |
| `build_prompts(tokenizer, df, max_seq_len, few_shot, prompt_version)` | Una riga → prompt tokenizzato + statistiche troncamento. |
| `select_few_shot` | Esempi corti dal train per modalità few-shot. |
| `parse_free_output` | Mappa testo libero → micro ammessa o `INVALID_LABEL`. |

Costante condivisa: `CHAT_TEMPLATE_KWARGS`.

### `evaluation/classic/` (import anche `classic.py`)

- **`predict_majority`**, **`predict_random`**, **`predict_tfidf_logreg`**: baseline CPU; TF-IDF char n-gram + logreg, maschera probabilità sulle micro della macro.

### `artifacts/metrics.py` (import anche `metrics.py`)

- **`compute_metrics(pred_df)`**: accuracy, macro-F1 (bootstrap 95%), weighted-F1, per-macro, `invalid_rate`.
- **`confusion_for_macro`**, **`summary_row`**: reporting notebook e `load_runs`.

Colonne minime predizioni: `macro_category`, `sub_category`, `pred`.

### `config/experiment.py` (import anche `experiment.py`)

- Caricamento YAML: **`load_yaml_merged`**, **`expand_vars`**, **`load_merged_experiment_dict`**, **`load_experiment_config`**.
- Hash: **`canonical_config_hash`**, **`file_sha256`**, **`splits_hash`** (da `record_key` + `split` / `fold`).
- Path: **`experiment_output_dir`**, **`resolved_dataset_path`**, **`repo_root`**.
- **`build_manifest(...)`**: JSON tracciabilità esperimento.
- **Dataclass** `ExperimentConfig` e sotto-config (`DatasetConfig`, `SplitConfig`, …).

### `artifacts/reporting.py` (import anche `reporting.py`)

| Funzione | Ruolo |
|----------|--------|
| `save_run(out_dir, slug, pred_df, run_info)` | Scrive parquet + JSON metriche; `run_info` include metadati esperimento se presenti. |
| `load_runs(out_dir)` | Tabella ordinata per macro-F1 + dict payload per slug. |
| `load_predictions(out_dir, slug)` | Parquet predizioni. |
| `per_macro_table(runs, metric)` | Pivot macro × run. |
| `load_experiments(output_root)` | Tutti gli esperimenti; righe aggregate k-fold `*__kfold_agg` con `macro_f1_std`. |

**`comparable_group`**: di default = `split_hash` (stesso test).

### `pipeline/`

- **`context.py`**: `ExperimentContext`, `PrepareStats`, `EvalFoldContext`.
- **`factory.py`**: `build_context` da path YAML.
- **`orchestrator.py`**: `BaselineExperimentOrchestrator`.
- **`stages/prepare.py`**, **`stages/classic.py`**, **`stages/compare_sync.py`**: macrostep prepare / classic / compare+sync.
- **`evaluation/slm/runner.py`**: `SubprocessVllmRunner`, `SlmEvaluatePipeline`.

### `run_baseline.py`

Un processo vLLM = un modello (libera VRAM tra modelli). Logica in **`SlmEvaluationJob`**; invocato dalla CLI via subprocess.

- **Modalità esperimento:** `--experiment-dir outputs/experiments/<id>` + `--model` + `--mode` + opzionale `--fold`.
- **Modalità legacy:** `--data`, `--splits`, `--out-dir`, …
- Modalità: `constrained`, `free`, `fewshot_constrained`.
- Skip se metrics esiste; `--rerun` per forzare.

Slug: vedi [§11](#11-naming-delle-run-slug).

### `cli.py`

Argparse sottile → `BaselineExperimentOrchestrator`: vedi [§5](#5-cli--comandi) e [architettura a layer](#1-concetti).

---

## 8. Config predefiniti

| File | Scopo |
|------|--------|
| `holdout_s42_minimal.yaml` | Riferimento: prompt `minimal`, `max_seq_len` manual 15360, tre modelli × constrained/free. |
| `holdout_s42_enriched_v1.yaml` | Stesso tipo di split; prompt `enriched_v1`, `manual_plus_prompt`. |
| `smoke_sample150.yaml` | Dataset campione, classiche complete, `slm: []`, `limit: 50` per test Mac. |

Path dataset nel YAML sono relativi a `TESI_DATA_DIR` (es. `data/df_sample_all.parquet`).

---

## 9. Notebook

| Notebook | Ruolo |
|----------|--------|
| `00_prepare_data_and_splits.ipynb` | EDA split e lunghezze token; con `USE_EXPERIMENT_YAML` legge path da `CONFIG_PATH` (allineato al CLI). |
| `01_baseline.ipynb` | Baseline classiche + subprocess SLM; `USE_EXPERIMENT_YAML` usa `--experiment-dir`. |
| `02_compare_experiments.ipynb` | `load_experiments` + export CSV. |

Con YAML: eseguire prima **`cli prepare`** (o cella equivalente) così esistono `splits.parquet` e `run_config.json`.

---

## 10. Test automatici

```bash
uv run --group dev pytest tests/ -q
```

| File | Cosa verifica |
|------|----------------|
| `test_experiment_config.py` | `extends`, espansione env, hash stabile |
| `test_data_kfold.py` | Nessun leakage documento per fold; assert leakage |
| `test_prompts.py` | System prompt per ogni macro/versione; footprint |
| `test_inference_max_seq.py` | `manual_plus_prompt` con tokenizer fittizio |
| `test_pipeline_context.py` | `build_context`, `PrepareStats` |
| `test_legacy_imports.py` | Shim `slm_baseline.data` / `labels` / `inference` |
| `test_prepare_golden.py` | `prepare` su `smoke_sample150` (se dataset presente) |

`conftest.py` aggiunge `src/` al `PYTHONPATH`.

---

## 11. Naming delle run (slug)

| Tipo | Pattern esempio |
|------|-----------------|
| Classica holdout | `majority`, `random`, `tfidf_logreg` |
| Classica k-fold | `majority__fold2` |
| SLM holdout | `Qwen3-1.7B__constrained__L15360` |
| SLM k-fold | `Qwen3-1.7B__constrained__L15360__fold2` |
| Aggregato compare | `...__kfold_agg` (+ colonna `macro_f1_std`) |

I file sono sempre `metrics__<slug>.json` e `predictions__<slug>.parquet`.

---

## 12. K-fold

1. YAML: `split.strategy: kfold`, `split.k: 5` (o altro).
2. `prepare` salva `splits.parquet` con colonna **`fold`** (nessuna colonna `split` fissa).
3. `classic` e `slm` iterano fold `0..k-1`, chiamando `apply_kfold_assignments` (train/val sul resto, test = fold corrente).
4. `compare` / `load_experiments` aggregano media e deviazione standard sulla macro-F1.

---

## 13. Modalità legacy (senza YAML)

Ancora supportata per notebook e script diretti:

- Split: `data/splits_v1.parquet`
- Contesto: `data/run_config.json`
- Output: `outputs/baseline/`
- SLM: `python -m slm_baseline.run_baseline --data ... --splits ... --out-dir ...`

I nuovi esperimenti dovrebbero usare il flusso YAML + cartella `outputs/experiments/<id>/`.

---

## 14. SageMaker Studio

Checklist:

1. Clone repo; `uv sync --group sagemaker`
2. `export HF_TOKEN=...` (Gemma gated)
3. `export TESI_DATA_DIR=/home/sagemaker-user/data` (parquet completo o sync da S3)
4. `export TESI_OUTPUT_DIR=/home/sagemaker-user/outputs` (disco Studio)
5. Opzionale: `export TESI_S3_PREFIX=s3://.../tesi`
6. `uv run python -m slm_baseline.cli run --config configs/experiments/holdout_s42_minimal.yaml`
7. Fine job: `cli compare` e `cli sync --all`

Ogni modello SLM è un **processo separato** (memoria GPU liberata tra un modello e l'altro).

---

## 15. Estendere il framework

### Nuovo esperimento

1. Copia un YAML in `configs/experiments/`.
2. Cambia **`experiment_id`** (nome cartella output).
3. Aggiusta `dataset`, `prompt`, `sequence`, `runs`.
4. `prepare` → verifica `manifest.json` e `splits.parquet`.

### Nuova variante prompt

1. Cartella `src/slm_baseline/prompt_variants/<nome>/` con un `.md` per macro (vedi `MACRO_TO_VARIANT_FILE` in `prompts.py`).
2. Estendi `system_prompt_for_macro` e `prompt_footprint`.
3. YAML: `prompt.version: <nome>`.

### Nuovo modello SLM nell’esperimento

In YAML, sotto `runs.slm`:

```yaml
- model: org/Nome-Modello
  modes: [constrained, free]
```

Aggiungere il tokenizer in `sequence.tokenizers` se usi `manual_plus_prompt` o `computed`.

### Fine-tuning (futuro)

Stesso split (`splits.parquet`), stesso `build_messages` / `prompt_version`, stesso `max_seq_len` da `run_config.json`; valutazione finale tramite **`save_run`** nella stessa cartella `runs/` o esperimento dedicato.

---

## 16. FAQ / errori comuni

| Problema | Causa / soluzione |
|----------|-------------------|
| `Dataset non trovato` | `TESI_DATA_DIR` errato o path YAML sbagliato; verifica `resolved_dataset_path` stampato in prepare. |
| `config_hash diverso` | Cartella esperimento già creata con altro YAML → nuovo `experiment_id` o `prepare --force`. |
| `Manca run_config.json` | Esegui `prepare` prima di `classic`/`slm`. |
| `Per split kfold serve --fold` | `run_baseline` con esperimento kfold richiede `--fold N` (il CLI lo passa automaticamente). |
| `sync.s3_prefix vuoto` | Imposta `TESI_S3_PREFIX` o campo `sync` nel YAML risolto. |
| Gemma non carica | `HF_TOKEN` + licenza accettata su Hugging Face. |
| Confronti “a caso” tra esperimenti | Controlla `comparable_group` / `split_hash`: devono coincidere per confronto sullo stesso test. |
| Import `vllm` fallisce in locale | Normale su Mac; usa solo `classic` o Studio con GPU. |

---

## Riferimenti incrociati

- [README.md](../README.md) — panoramica repo ed esecuzione rapida
- [PROGETTO.md](PROGETTO.md) — obiettivo tesi, dati, metodologia
- Piano implementazione: gestione esperimenti ML (documento interno piano Cursor)
