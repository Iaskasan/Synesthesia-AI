import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.audit_dataset import Track
from src.data.preprocess_dataset import PreprocessConfig, audio_path, log_mel_spectrogram, preprocess


def _write_split(path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"])
        writer.writerow([f"track_{split}", f"artist_{split}", "album_1", "00/1.wav", "1", "mood/theme---calm"])


def test_log_mel_has_fixed_shape_and_range():
    config = PreprocessConfig(8_000, 1, 16, 256, 128)
    feature = log_mel_spectrogram(np.zeros(100, dtype=np.float32), config)
    assert feature.shape == (16, 61)
    assert feature.dtype == np.float32
    assert np.all((-1 <= feature) & (feature <= 1))


def test_preprocess_writes_manifest_and_resumes(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "00").mkdir(parents=True)
    sf.write(dataset / "00/1.low.wav", np.zeros(8_000, dtype=np.float32), 8_000)
    metadata = tmp_path / "metadata"
    for split in ("train", "validation", "test"):
        _write_split(metadata / "splits/split-0" / f"autotagging_moodtheme-{split}.tsv", split)
    config = PreprocessConfig(8_000, 1, 16, 256, 128)
    output = tmp_path / "processed"

    first = preprocess(dataset, metadata, output, config)
    second = preprocess(dataset, metadata, output, config)

    assert first == {"tracks_requested": 3, "tracks_available": 3, "processed": 3, "already_cached": 0, "errors": 0}
    assert second["already_cached"] == 3
    assert len(list((output / "features").rglob("*.npz"))) == 3
    assert len((output / "manifest.csv").read_text().splitlines()) == 4


def test_audio_path_uses_low_suffix(tmp_path):
    track = Track("track_1", "artist_1", "07/123.mp3", 30, ("calm",))
    assert audio_path(tmp_path, track) == tmp_path / "07/123.low.mp3"
