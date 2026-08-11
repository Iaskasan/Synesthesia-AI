To download the dataset into the Windows `C:` drive from WSL:

```bash
cd mtg-jamendo-dataset
./scripts/download/download.py --dataset autotagging-moodtheme --type audio-low --from mtg-fast --outputdir /mnt/c/AI/dataset --unpack --remove
cd ..
```

Audit the extracted dataset before preprocessing. This verifies the audio
inventory and official splits, checks for artist leakage, records label
frequencies, and writes the frozen 20-label baseline selection:

```bash
python -m src.data.audit_dataset \
  --dataset-root /mnt/c/AI/dataset \
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
  --dataset-root /mnt/c/AI/dataset \
  --output-root /mnt/c/AI/dataset/processed/logmel-v1 \
  --max-tracks 5
```

If that succeeds, remove `--max-tracks 5` for the complete dataset. The command
is resumable, so existing feature files are skipped:

```bash
python -m src.data.preprocess_dataset \
  --dataset-root /mnt/c/AI/dataset \
  --output-root /mnt/c/AI/dataset/processed/logmel-v1 \
  --workers 4
```

The output contains one compressed `.npz` per track, `manifest.csv`,
`preprocess_config.json`, `summary.json`, and `errors.csv`.
