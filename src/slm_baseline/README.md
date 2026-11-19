# `slm_baseline`

Package Python per classificazione **macro → micro** su segmenti OCR (baseline classiche, zero-shot SLM con vLLM, esperimenti da YAML).

**Manuale completo (tutti i file, CLI, layout, estensioni):**  
[docs/FRAMEWORK.md](../../docs/FRAMEWORK.md)

## Comandi rapidi

```bash
python -m slm_baseline.cli prepare --config configs/experiments/<nome>.yaml
python -m slm_baseline.cli run      --config configs/experiments/<nome>.yaml
python -m slm_baseline.cli compare
```

## Moduli principali

| Modulo | Ruolo |
|--------|--------|
| `experiment.py` | YAML, hash, manifest, path |
| `data/` | Parquet, split, k-fold (`data/core.py`; import `slm_baseline.data`) |
| `prompts.py` | Versioni prompt chat |
| `inference.py` | Template, `max_seq_len`, `build_prompts` |
| `classic.py` | Baseline CPU |
| `metrics.py` / `reporting.py` | Metriche e artefatti run |
| `run_baseline.py` | Un modello vLLM |
| `cli.py` | Pipeline esperimento |

Test: `uv run --group dev pytest tests/ -q` dalla root del repo.
