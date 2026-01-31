# Strategia di Raccolta Dati Dialettali

## 📊 Stato Ricerca Fonti Accademiche

### ✅ Fonti Identificate

#### 1. **CLIPS (Università di Napoli Federico II)**
- **URL**: http://www.clips.unina.it/
- **Tipo**: Corpus di parlato italiano stratificato con variazione regionale
- **Potenziale**: ALTO - possibile presenza di audio napoletano
- **Prossimi step**: 
  - Contattare AILab UrbanECO Research Center (ailab@unina.it)
  - Verificare disponibilità sottocorpus napoletano
  - Richiedere accesso dataset se disponibile

#### 2. **Atlante Linguistico Italiano (ALI)**
- **URL**: https://www.atlantelinguistico.it/
- **Tipo**: Trascrizioni dialettali (principalmente testo)
- **Potenziale**: BASSO per ASR (no audio), ALTO per analisi linguistica
- **Utilità**: Riferimento per variazioni lessicali e fonologiche

#### 3. **Common Voice (Mozilla)**
- **Status**: Da verificare presenza dialetti italiani
- **Probabile**: Solo "Italian (it)" generico, senza separazione dialettale
- **Strategia alternativa**: Usare come baseline italiano standard per confronto

---

## 🎥 Strategia B: Raccolta Custom da Fonti Pubbliche

### Fonti Video/Audio Online

#### **NAPOLETANO**

##### YouTube - Canali Consigliati:
1. **TeleNostra** - TV locale napoletana
2. **Canale 21** - Emittente regionale Campania
3. **Made in Sud** - Spettacoli comici in napoletano
4. **Eduardo De Filippo** - Teatro classico napoletano (dominio pubblico)
5. **Notizie RAI Campania** - TG regionali con parlanti napoletani

##### Podcast/Radio:
- Radio CRC (Campania) - trasmissioni in napoletano
- Radio Kiss Kiss Napoli

#### **SICILIANO**

##### YouTube - Canali Consigliati:
1. **Tele One** - TV locale siciliana
2. **Video Mediterraneo** - Emittente regionale Sicilia
3. **Ciuri Ciuri** - Contenuti culturali siciliani
4. **TGR Sicilia** - TG regionali RAI
5. **Film di Peppino De Filippo/Angelo Musco** (dominio pubblico)

##### Podcast/Radio:
- RGS - Radiotelevisione Siciliana
- Radio Time

#### **ROMANESCO**

##### YouTube - Canali Consigliati:
1. **Tele Radio Stereo** - Emittente romana
2. **Roma TV** - TV locale romana
3. **Er Cipolla** - Content creator romanesco
4. **Alberto Sordi** - Film classici (alcuni dominio pubblico)
5. **TGR Lazio** - TG regionali RAI con speaker romani

##### Podcast/Radio:
- Radio Radio Roma
- Centro Suono Sport (radio sportiva romana)

---

## 🛠️ Pipeline di Raccolta Tecnica

### Tool Necessari

```bash
# Installazione tool
pip install yt-dlp          # Download YouTube
pip install pydub           # Audio processing
pip install ffmpeg-python   # Conversione formati
```

### Script di Download Base

```python
# download_youtube_audio.py
import yt_dlp

def download_audio(url, output_path, dialect):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_path}/{dialect}_%(id)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Esempio uso
download_audio('https://youtube.com/watch?v=XXX', './data/raw', 'napoletano')
```

### Considerazioni Legali ⚖️

**IMPORTANTE - Copyright e Fair Use**:
- ✅ Contenuti con licenza Creative Commons
- ✅ Contenuti dominio pubblico (pre-1926 in molti casi)
- ✅ Fair use per ricerca accademica (citare sempre fonte)
- ⚠️ Contenuti RAI: uso educativo, non redistribuzione
- ❌ Non redistribuire dataset finale se contiene contenuti protetti

**Soluzione per tesi**:
- Creare **manifest file** con link alle fonti originali
- Distribuire solo **script di download** e **metadata**
- Non includere audio raw nella repo GitHub
- Documentare processo di raccolta in appendice tesi

---

## 📋 Protocol per Trascrizione

### Opzione 1: Trascrizione Automatica (Bootstrap)
```python
# Usare Whisper per generare trascrizioni iniziali
import whisper
model = whisper.load_model("large")
result = model.transcribe("audio.wav", language="it")
```

### Opzione 2: Sottotitoli Esistenti
- YouTube fornisce sottotitoli automatici (spesso disponibili)
- Scaricare con `yt-dlp --write-auto-sub`
- Validare e correggere manualmente campione piccolo

### Opzione 3: Correzione Manuale
- Usare tool come **Audacity** + trascrizione
- Focus su 3-5 ore per dialetto = ~45-75 ore lavoro totale
- Possibile coinvolgere parlanti nativi (crowdsourcing?)

---

## 📊 Dataset Target Structure

```
tesi-data-science/
├── data/
│   ├── raw/                    # Audio scaricati
│   │   ├── napoletano/
│   │   ├── siciliano/
│   │   └── romanesco/
│   ├── processed/              # Audio normalizzati 16kHz mono
│   │   ├── napoletano/
│   │   ├── siciliano/
│   │   └── romanesco/
│   ├── transcriptions/         # File testo trascrizioni
│   │   ├── napoletano/
│   │   ├── siciliano/
│   │   └── romanesco/
│   └── manifest.csv            # Metadata: path, dialect, duration, speaker_id
└── scripts/
    ├── download_audio.py
    ├── preprocess_audio.py
    └── create_manifest.py
```

### Esempio Manifest CSV:
```csv
file_id,file_path,dialect,duration_sec,source_url,speaker_id,gender,transcript_path
nap_001,processed/napoletano/nap_001.wav,napoletano,8.5,https://youtube.com/...,spk_001,M,transcriptions/napoletano/nap_001.txt
sic_001,processed/siciliano/sic_001.wav,siciliano,6.2,https://youtube.com/...,spk_045,F,transcriptions/siciliano/sic_001.txt
```

---

## 🎯 Milestone Raccolta Dati

### Week 1-2: Identificazione Contenuti
- [ ] Creare playlist YouTube per ciascun dialetto (30-50 video)
- [ ] Documentare sorgenti e licenze
- [ ] Stimare durata totale disponibile

### Week 3: Download e Preprocessing
- [ ] Scaricare audio (target: 5 ore per dialetto raw)
- [ ] Convertire a WAV 16kHz mono
- [ ] Segmentare in clip 5-15 secondi

### Week 4: Trascrizioni
- [ ] Estrarre/generare trascrizioni iniziali
- [ ] Validare campione (10% manuale)
- [ ] Pulire e normalizzare testo

### Week 5: Quality Check
- [ ] Verificare allineamento audio-testo
- [ ] Rimuovere clip problematiche
- [ ] Generare manifest finale

---

## 🚨 Rischi e Mitigazioni

| Rischio | Probabilità | Impatto | Mitigazione |
|---------|-------------|---------|-------------|
| Copyright issues | Media | Alto | Usare solo contenuti pubblici/CC, documentare |
| Qualità audio scarsa | Alta | Medio | Filtrare clip rumorose, applicare noise reduction |
| Trascrizioni imprecise | Alta | Alto | Validazione manuale campione, correzione iterativa |
| Sbilanciamento dialetti | Media | Medio | Monitorare durata per dialetto, bilanciare in fase raccolta |
| Tempo eccessivo | Alta | Alto | Limitare scope: 3-4 ore per dialetto sufficiente per tesi magistrale |

---

## 💡 Raccomandazioni Finali

### Approccio Pragmatico per Tesi Magistrale:

1. **Inizia con CLIPS**: Contatta Università di Napoli per accesso dataset napoletano (se esiste)
2. **Supplementa con raccolta custom**: YouTube/RAI per siciliano e romano
3. **Scala ridotta**: 3-4 ore per dialetto = 10-12 ore totali è **più che sufficiente** per analisi comparativa
4. **Focus su qualità > quantità**: Meglio 3 ore ben trascritte che 10 ore rumorose
5. **Baseline italiano standard**: Usa subset Common Voice IT per confronto

### Prossimo Step Immediato:
**Contattare CLIPS** via email: ailab@unina.it
- Spiegare progetto tesi magistrale
- Chiedere disponibilità corpus napoletano con audio
- Richiedere accesso o partnership accademica

Vuoi che prepari una email template per contattare CLIPS?
