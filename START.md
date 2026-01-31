# Tesi Magistrale: Benchmark di Modelli Speech-to-Text
**Studente:** Vincenzo Pafundi  
**Anno Accademico:** 2025-2026

## 📋 Obiettivo della Tesi

Confrontare e valutare le performance di diversi modelli di speech recognition (ASR - Automatic Speech Recognition) sul riconoscimento di **dialetti italiani regionali** (Napoletano, Siciliano, Romano), analizzando le sfide specifiche legate alla variazione linguistica e definendo metriche di valutazione e protocolli di test riproducibili.

## 🎯 Scope del Progetto (Tesi Magistrale)

### Cosa farò:
1. **Raccolta Dataset**: Creare o selezionare campioni di audio nei tre dialetti target (Napoletano, Siciliano, Romano)
2. **Protocollo di Ingestion**: Definire pipeline di preprocessing e normalizzazione per dati dialettali
3. **Benchmarking**: Testare 3-5 modelli ASR su ciascun dialetto con metriche standard
4. **Analisi**: Confrontare performance per dialetto, identificare pattern di errore specifici e requisiti computazionali

### Cosa NON farò (per mantenere lo scope ragionevole):
- ❌ Training di nuovi modelli da zero
- ❌ Fine-tuning estensivo (opzionale, solo se il tempo lo permette)
- ❌ Deployment in produzione
- ❌ Dataset di dimensioni industriali

## 🚀 Next Steps

### Phase 1: Setup & Ricerca (Settimane 1-2)
**Obiettivo:** Familiarizzare con il dominio e preparare l'ambiente

- [ ] **1.1 Literature Review**
  - Leggere paper recenti su ASR (Whisper, Wav2Vec2, etc.)
  - Studiare metriche standard: WER (Word Error Rate), CER (Character Error Rate)
  - Documentare stato dell'arte per modelli italiani

- [ ] **1.2 Setup Ambiente Tecnico**
  - Configurare Python environment (Python 3.10+)
  - Installare librerie: `transformers`, `datasets`, `librosa`, `jiwer`
  - Setup Jupyter notebooks per sperimentazione
  - Verificare disponibilità GPU (Colab/locale/cloud)

- [ ] **1.3 Identificare Dataset Candidati per Dialetti**
  - Common Voice (Mozilla) - verificare disponibilità dialetti italiani
  - CLIPS (Corpora e Lessici dell'Italiano Parlato e Scritto)
  - Dataset regionali/universitari specifici per napoletano, siciliano, romano
  - Eventualmente: raccolta dati custom (interviste, media locali, YouTube)
  - Decidere dimensione campione: 3-5 ore per dialetto (~10-15 ore totali)
  - **📄 Vedi dettagli**: [DATA_COLLECTION_STRATEGY.md](DATA_COLLECTION_STRATEGY.md)

### Phase 2: Data Collection & Preparation (Settimane 3-4)
**Obiettivo:** Costruire e preparare il dataset di test

- [ ] **2.1 Download e Selezione Dati**
  - Scaricare dataset per ciascun dialetto (Napoletano, Siciliano, Romano)
  - Selezionare campione stratificato (speaker diversi, generi, età se possibile) per dialetto
  - Documentare statistiche: durata per dialetto, n. speaker, distribuzione lunghezza clip
  - Assicurare bilanciamento tra i tre dialetti per confronto equo

- [ ] **2.2 Definire Protocollo di Ingestion**
  - Standardizzare formato audio (16kHz mono WAV)
  - Normalizzare trascrizioni dialettali (gestire variazioni ortografiche, punteggiatura)
  - Etichettare dati con metadata dialettale (dialetto, regione)
  - Creare script di preprocessing riutilizzabili
  - Split: 70% test, 30% validation per ciascun dialetto

- [ ] **2.3 Data Quality Check**
  - Verificare allineamento audio-trascrizioni
  - Rimuovere clip corrotte o troppo corte/lunghe
  - Creare dataset manifest (CSV/JSON con metadati)

### Phase 3: Model Selection & Testing (Settimane 5-7)
**Obiettivo:** Eseguire benchmark sui modelli selezionati

- [ ] **3.1 Selezionare Modelli da Confrontare** (suggerimenti)
  - OpenAI Whisper (small, medium, large)
  - Meta MMS (Massively Multilingual Speech)
  - Wav2Vec2 Italian (fine-tuned su italiano)
  - SpeechBrain Italian ASR
  - Eventualmente: Google Speech-to-Text API (confronto con soluzione proprietaria)

- [ ] **3.2 Implementare Pipeline di Inferenza**
  - Creare script unificato per inference
  - Gestire batch processing per efficienza
  - Implementare logging di tempi e metriche

- [ ] **3.3 Eseguire Benchmark**
  - Calcolare WER e CER per ogni modello **per ciascun dialetto**
  - Confrontare performance: Napoletano vs Siciliano vs Romano
  - Misurare tempo di inferenza (Real-Time Factor)
  - Registrare uso memoria/GPU
  - Analizzare errori specifici per dialetto (fonemi, costrutti grammaticali)

### Phase 4: Analisi & Documentazione (Settimane 8-10)
**Obiettivo:** Analizzare risultati e scrivere la tesi

- [ ] **4.1 Analisi Comparativa**
  - Creare tabelle comparative per modello e per dialetto
  - Grafici: performance per dialetto, confronto cross-dialettale
  - Analisi errori qualitativi (errori tipici per ciascun dialetto)
  - Identificare quale dialetto è più/meno riconoscibile
  - Identificare trade-off accuracy/velocità e robustezza ai dialetti

- [ ] **4.2 Scrittura Tesi**
  - Introduzione e motivazione
  - Related Work / State of the Art
  - Metodologia (dataset, protocollo, modelli)
  - Risultati sperimentali
  - Discussione e conclusioni
  - Appendici (codice, configurazioni)

- [ ] **4.3 Preparazione Presentazione**
  - Slides per discussione
  - Demo live (opzionale)
  - Repository GitHub con codice e documentazione

## 📊 Deliverable Finali

1. **Tesi scritta** (80-120 pagine)
2. **Codice sorgente** (GitHub repo con README completo)
3. **Dataset manifest** (o link a dataset pubblici)
4. **Report benchmark** (tabelle, grafici, metriche)
5. **Presentazione** (slides per discussione)

## 🛠️ Stack Tecnologico Suggerito

- **Linguaggio:** Python 3.10+
- **ML Framework:** HuggingFace Transformers, PyTorch
- **Audio Processing:** librosa, torchaudio, soundfile
- **Metriche:** jiwer (WER/CER calculation)
- **Data Management:** pandas, datasets (HuggingFace)
- **Visualization:** matplotlib, seaborn, plotly
- **Notebook:** Jupyter Lab
- **Version Control:** Git + GitHub

## ⏱️ Timeline Indicativa

| Fase | Durata | Settimane |
|------|--------|-----------|
| Setup & Ricerca | 2 settimane | 1-2 |
| Data Preparation | 2 settimane | 3-4 |
| Benchmarking | 3 settimane | 5-7 |
| Analisi & Scrittura | 3 settimane | 8-10 |
| **Totale** | **10 settimane** | **(~2.5 mesi)** |

## 📝 Note Importanti

- **Inizia con un subset piccolo** per validare pipeline prima di processare tutto
- **Documenta tutto** durante lo sviluppo, non alla fine
- **Checkpoint regolari** con il PhD advisor (ogni 2 settimane)
- **Usa Git** fin dall'inizio per versioning
- **Considera vincoli computazionali**: alcuni modelli richiedono GPU potenti

## 🎓 Criteri di Successo

✅ Dataset ben documentato e riproducibile  
✅ Almeno 3 modelli confrontati con metriche standard  
✅ Analisi quantitativa E qualitativa dei risultati  
✅ Codice pulito, commentato e riutilizzabile  
✅ Tesi con metodologia chiara e risultati interpretabili

---

**Prossima azione:** Completare gli step della Phase 1 (Setup & Ricerca)