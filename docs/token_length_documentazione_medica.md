# Lunghezze token — `documentazione_medica` (full curated)

Report da token scan locale su **tutto** il curated (non campione 8k).

| Campo | Valore |
|--------|--------|
| **Curated input** | `data/runs/20261124T120454Z__local-anon-20261122-v2/curation/df_segments_curated_v1.parquet` |
| **Macro** | `documentazione_medica` |
| **Righe** | 55 529 (testo non vuoto dopo `normalize_text`) |
| **Tokenizer** | `Qwen/Qwen3-1.7B` (`trust_remote_code=True`) |
| **Sequenza misurata** | Prompt v1 training: system (macro + elenco micro) + user (`Macro:` + `ocr_surrogates`) + assistant (`sub_category`) — cfr. `src/eda/tokens.py`, `src/eda/prompts.py` |
| **Artefatti numerici** | `data/runs/token-docmed-full/` |
| **Script riproducibile** | `scripts/token_scan_docmed_full.py` |

```bash
uv run python scripts/token_scan_docmed_full.py
```

---

## Distribuzione globale (`seq_tokens`)

| Statistica | Token |
|------------|-------|
| min | 181 |
| **p50** | **740** |
| p75 | 1 232 |
| **p90** | **2 376** |
| p95 | 3 651 |
| p99 | 15 356 |
| p99.9 | 69 267 |
| max | 549 580 |
| media | 1 549,3 |

**Soglia minima per ≥90% righe (full corpus):** `max_seq_len = 3072` (93,50% coperte).

---

## Copertura per finestra (`max_seq_len`)

| max_seq_len | % coperte | n truncati / esclusi | n righe |
|-------------|-----------|----------------------|---------|
| 2 048 | 87,42% | 6 987 | 55 529 |
| **3 072** | **93,50%** | 3 610 | 55 529 |
| **4 096** | **95,77%** | 2 350 | 55 529 |
| 6 144 | 97,54% | 1 368 | 55 529 |
| **8 192** | **98,20%** | **1 001** | 55 529 |
| 12 288 | 98,80% | 669 | 55 529 |
| **16 384** | **99,05%** | 528 | 55 529 |
| 24 576 | 99,35% | 360 | 55 529 |
| 32 768 | 99,58% | 231 | 55 529 |
| 65 536 | 99,89% | 62 | 55 529 |

*Se la policy è **drop** righe oltre soglia, “truncati” = righe non in training. Se **truncate**, le percentuali sotto non cambiano.*

### Confronto con scan EDA (campione ~8k)

| Soglia | Campione `token-docmed` | Full 55k |
|--------|----------------------|----------|
| 4 096 | 95,75% | 95,77% |
| 8 192 | 98,32% | 98,20% |

---

## Percentuali per sub — full vs finestre 8k / 16k

Pool effettivo se si **escludono** righe con `seq_tokens` oltre soglia:

| sub_category | n full | % full | % ≤ 8k | Δ pp | % ≤ 16k | Δ pp |
|--------------|--------|--------|--------|------|---------|------|
| documentazione_medica_altro | 32 470 | 58,474 | 58,537 | +0,06 | 58,495 | +0,02 |
| prescrizione_medica | 12 165 | 21,907 | 22,297 | +0,39 | 22,107 | +0,20 |
| referto_del_pronto_soccorso | 6 914 | 12,451 | 12,456 | +0,01 | 12,547 | +0,10 |
| lettera_di_dimissioni | 2 557 | 4,605 | 4,101 | **−0,50** | 4,264 | −0,34 |
| certificato_avvenuta_guarigione | 1 423 | 2,563 | 2,610 | +0,05 | 2,587 | +0,02 |

Righe nel pool: **54 528** (8k), **55 001** (16k).

### Perdita per sub (% del sub perso oltre soglia)

| sub_category | % righe oltre 8k | % righe oltre 16k |
|--------------|------------------|-------------------|
| lettera_di_dimissioni | **12,55%** | **8,29%** |
| documentazione_medica_altro | 1,70% | 0,91% |
| referto_del_pronto_soccorso | 1,76% | 0,19% |
| prescrizione_medica | ~0,06% | ~0,05% |
| certificato_avvenuta_guarigione | 0% | 0% |

---

## Statistiche per sub (mediana, code)

| sub_category | n | mediana | p90 | p95 | max |
|--------------|---|---------|-----|-----|-----|
| prescrizione_medica | 12 165 | 638 | 862 | 985 | 64 621 |
| documentazione_medica_altro | 32 470 | 738 | 1 864 | 3 070 | 549 580 |
| certificato_avvenuta_guarigione | 1 423 | 481 | 860 | 1 161 | 4 795 |
| referto_del_pronto_soccorso | 6 914 | 1 819 | 4 021 | 5 520 | 31 474 |
| lettera_di_dimissioni | 2 557 | 1 811 | 12 126 | 26 942 | 239 943 |

Code lunghe: `documentazione_medica_altro` (multipagina) e `lettera_di_dimissioni`. **62** segmenti superano anche 65 536 token.

---

## Raccomandazioni training (doc-med)

| Obiettivo | `max_seq_len` suggerito |
|-----------|-------------------------|
| ≥90% righe senza drop | **3 072** (meglio **4 096** ~96%) |
| Bilanciamento quasi invariato + quasi tutto il corpus | **8 192** (~98,2%; −1,8% righe) |
| ~99% righe | **16 384** |
| Outlier estremi | cap pagine, troncamento, o esclusione mirata su record in `o2_seq_lengths_docmed.parquet` |

Allineare sempre tokenizer, `apply_chat_template` e prompt con la repo training (`PROMPT_VERSION` v1).

### Unsloth e `Num examples = 7_280` (non è drop EDA)

Con **48 849** righe in `train.jsonl`, il banner Unsloth può mostrare **~7 280** perché **sample packing / padding-free** accorpa più conversazioni in finestre da `max_seq_len` (8192). Non significa perdere l’85% del corpus per soglia token (EDA full: ~**1 8%** oltre 8192).

Per **1 segmento = 1 esempio** in training: `SFTConfig(packing=False, padding_free=False)`, `UNSLOTH_DISABLE_AUTO_PADDING_FREE=1`, `_unsloth_disable_auto_packing` — vedi `slm_baseline.training.run_unsloth`. Dopo il fix atteso: `Num examples ≈ 48 849`, `Total steps ≈ 9 159` (3 epoche, batch effettivo 16).

---

## File in `data/runs/token-docmed-full/`

| File | Contenuto |
|------|-----------|
| `summary.json` | Percentili globali e soglia ≥90% |
| `o2_token_coverage_full.csv` | Tabella soglie |
| `o2_token_lengths_by_sub.csv` | Mediana / p90 / p95 / max per sub |
| `o2_seq_lengths_docmed.parquet` | `record_key`, `document_id`, `sub_category`, `seq_tokens` per riga |
| `REPORT.md` | Copia breve con link a questo documento |