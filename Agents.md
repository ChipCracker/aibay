# Projektüberblick
Dieses Repository zielt darauf ab, mehrere deutschsprachige Speech-to-Text-Experimente zu vereinheitlichen. Der erste Schwerpunkt liegt auf dem BAS RVG1 Korpus; weitere Datensätze sollen später über dieselbe Pipeline angebunden werden.

# Datensatz BAS RVG1
- Speicherort wird über `paths.BAS_RVG1_PATH` erwartet (`DATASETS_PATH/BAS-RVG1/RVG1_CLARIN`).
- Verwendet werden ausschließlich die SP1-Aufnahmen (eine Minute spontane Sprache).
- Audioquelle: Mikrofonkanal `c` → Sennheiser HD/MD 410 (High-Quality).
- Referenztranskription: TRL-Dateien (`sp1XXX.trl`) im Verbmobil-II-Format; nicht relevante Markups (`+/…/+`, `-/…/-`, Tags wie `<aeh>`, `%`, `*`, Dialekt-Marker `<!…>`) müssen entfernt werden.

# Ziel-Funktionalität
1. **Ground-Truth laden**
   Funktion `load_sp1_dataframe` in `datasets/BAS_RVG1.py` soll für alle Sprecher:
   - `speaker_id`, Pfad zur passenden `sp1c*.wav` (oder fallback `.nis`) und bereinigte Transkription liefern.
   - Ein `pandas.DataFrame` mit Spalten `speaker_id`, `audio_path`, `transcription` zurückgeben.
2. **ASR-Inferenz**
   - Für jedes Datensatz-DataFrame eine Transkription erzeugen mittels:
     - **Whisper Large V3** (`whisper_pipeline.py`) → Spalten: `whisper_large_v3_transcription`, `whisper_large_v3_wer`
     - **Parakeet TDT v3** (`parakeet_pipeline.py`) → Spalten: `parakeet_tdt_v3_transcription`, `parakeet_tdt_v3_wer`
   - Batch-Verarbeitung ermöglichen, sodass weitere Datensätze später denselben Ablauf nutzen können.
   - CLI unterstützt `--model whisper|parakeet|both` zur Auswahl des ASR-Systems.
3. **Metriken**
   - Word Error Rate (WER) zwischen Ground Truth und ASR-Transkriptionen berechnen.
   - Beide Modelle verwenden dieselbe WER-Berechnungsfunktion (`add_wer_column`).
4. **Dialektanalyse (Folgeschritt)**
   - Dialektklassen aus `doc/dialects.txt` bzw. `table/sprk_att.txt` lesen und mit den Sprecher-IDs verknüpfen.
   - Korrelationen zwischen Dialektvarianten und WER untersuchen.

# Offene Aufgaben & Nächste Schritte
- ✅ Whisper-Large-V3 Inferenzfunktion implementiert (Batching, GPU/CPU-Optionen)
- ✅ Parakeet TDT v3 Inferenzpipeline hinzugefügt (NeMo-basiert)
- ✅ WER-Berechnung mittels `jiwer` implementiert
- Dialekt-Mapping vorbereiten (Join Speaker-Metadaten ↔︎ WER-Ergebnisse)
- Tests/Notebooks ergänzen, um Pipeline-End-to-End zu validieren
- Audio-Preprocessing (16kHz Resampling für Parakeet) bei Bedarf hinzufügen
- Vergleichsanalysen zwischen Whisper und Parakeet durchführen
