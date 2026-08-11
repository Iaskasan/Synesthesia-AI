To download the dataset into the Windows `C:` drive from WSL:

```bash
cd mtg-jamendo-dataset
./scripts/download/download.py --dataset autotagging-moodtheme --type audio-low --from mtg-fast --outputdir /mnt/c/AI/dataset --unpack --remove
cd ..
```

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
