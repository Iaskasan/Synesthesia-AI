## Reuse the existing files on this computer

The current checkout is `/root/RNCP-Synesthesia/Synesthesia-AI`. Activate its
environment before running the commands below:

```bash
cd /root/RNCP-Synesthesia/Synesthesia-AI
source .venv/bin/activate
```

The dataset tools have been restored from the original local checkout at
`/root/Synesthesia-AI/mtg-jamendo-dataset`, at the exact commit recorded by this
repository. The selected classifier has been copied from the original project's
`artifacts/clap_diagnostics/selected_head.joblib` into the same relative location
in this project. The original files were left unchanged.

Existing data stays on the mounted Windows drive; no audio or embeddings need
to be copied or downloaded:

| Files | Path |
| --- | --- |
| Audio | `/mnt/g/AI/Datasets` |
| Log-mel preprocessing | `/mnt/g/AI/Datasets/processed/logmel-v1` |
| CLAP embeddings | `/mnt/g/AI/Datasets/processed/clap-music-v1` |
| Official metadata/splits | `mtg-jamendo-dataset/data` |
| Selected classifier | `artifacts/clap_diagnostics/selected_head.joblib` |

The restored classifier was trained with scikit-learn 1.3.2. The main requirements
pin that version and a compatible NumPy version to avoid a pickle-version mismatch.
Do not install the old dataset repository's entire requirements file over this
environment; it contains unrelated historical package pins.

## Download only if the existing data is unavailable

The restored upstream downloader uses an underscore in the dataset name and a
positional output directory. Its CLI does not implement the newer aliases or
partial-download resumption expected by `tests/test_download_cli.py`.

To download into a different, prepared directory if needed:

```bash
cd mtg-jamendo-dataset
python scripts/download/download.py --dataset autotagging_moodtheme --type audio-low --from mtg-fast /path/to/new/dataset --unpack --remove
cd ..
```

Audit the extracted dataset before preprocessing. This verifies the audio
inventory and official splits, checks for artist leakage, records label
frequencies, and writes the frozen 20-label baseline selection:

```bash
python -m src.data.audit_dataset \
  --dataset-root /mnt/g/AI/Datasets \
  --metadata-root mtg-jamendo-dataset/data \
  --output-dir audit
```

The audit reports are written to `audit/`, including
`dataset_summary.json`, `label_frequencies.csv`, `selected_labels.json`, and
the missing-audio, source-archive, and artist-leakage reports. Archives removed
by `--remove` are recorded as not retained and are not reported as missing
when all expected audio files are available.

Run a small preprocessing smoke test first. This writes log-mel features and
manifests outside the repository:

```bash
python -m src.data.preprocess_dataset \
  --dataset-root /mnt/g/AI/datasets \
  --output-root /mnt/g/AI/datasets/processed/logmel-v1 \
  --max-tracks 5
```

If that succeeds, remove `--max-tracks 5` for the complete dataset. The command
is resumable, so existing feature files are skipped:

```bash
python -m src.data.preprocess_dataset \
  --dataset-root /mnt/g/AI/datasets \
  --output-root /mnt/g/AI/datasets/processed/logmel-v1 \
  --workers 4
```

The output contains one compressed `.npz` per track, `manifest.csv`,
`preprocess_config.json`, `summary.json`, and `errors.csv`.

## Start the Streamlit app

From the repository root, run:

```bash
streamlit run src/app/app.py
```

Streamlit will print the local URL to open in your browser (usually
`http://localhost:8501`).
