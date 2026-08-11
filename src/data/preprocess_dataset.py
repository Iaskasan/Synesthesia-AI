"""Preprocess MTG-Jamendo audio into a resumable log-mel cache."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from src.data.audit_dataset import SPLIT_NAMES, Track, read_tracks


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 22_050
    duration: float = 30.0
    n_mels: int = 96
    n_fft: int = 2_048
    hop_length: int = 512
    fmin: float = 20.0
    fmax: float | None = None


def audio_path(dataset_root: Path, track: Track) -> Path:
    relative = Path(track.path)
    return dataset_root / relative.with_name(f"{relative.stem}.low{relative.suffix}")


def cache_path(output_root: Path, split: str, track: Track) -> Path:
    return output_root / "features" / split / Path(track.path).parent / f"{track.track_id}.npz"


def log_mel_spectrogram(audio: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Return a fixed-shape float32 log-mel representation in [-1, 1]."""
    samples = int(round(config.sample_rate * config.duration))
    audio = librosa.util.fix_length(np.asarray(audio, dtype=np.float32), size=samples)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
        fmin=config.fmin, fmax=config.fmax, power=2.0, center=False,
    )
    return np.asarray(
        librosa.power_to_db(mel, ref=np.max, top_db=80.0) / 40.0 + 1.0,
        dtype=np.float32,
    )


def _write_feature(path: Path, feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, log_mel=feature)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_one(job: tuple[str, Track, Path, Path, PreprocessConfig, bool]) -> dict:
    split, track, dataset_root, output_root, config, overwrite = job
    source = audio_path(dataset_root, track)
    destination = cache_path(output_root, split, track)
    status = "cached"
    if overwrite or not destination.is_file():
        audio, _ = librosa.load(source, sr=config.sample_rate, mono=True, duration=config.duration)
        if audio.size == 0:
            raise ValueError("decoded audio is empty")
        _write_feature(destination, log_mel_spectrogram(audio, config))
        status = "processed"
    return {
        "split": split, "track_id": track.track_id, "artist_id": track.artist_id,
        "audio_path": source.relative_to(dataset_root).as_posix(),
        "feature_path": destination.relative_to(output_root).as_posix(),
        "tags": json.dumps(track.tags), "status": status,
    }


def preprocess(
    dataset_root: Path, metadata_root: Path, output_root: Path,
    config: PreprocessConfig, split_index: int = 0, workers: int = 1,
    max_tracks: int | None = None, overwrite: bool = False,
) -> dict:
    """Build the feature cache and return a preprocessing summary."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "preprocess_config.json"
    serialized_config = {**asdict(config), "split_index": split_index}
    if config_path.is_file() and json.loads(config_path.read_text()) != serialized_config:
        raise ValueError(
            f"Existing cache configuration differs: {config_path}. "
            "Choose another --output-root or remove the old cache intentionally."
        )
    config_path.write_text(json.dumps(serialized_config, indent=2) + "\n", encoding="utf-8")

    split_root = metadata_root / "splits" / f"split-{split_index}"
    jobs = []
    for split in SPLIT_NAMES:
        tracks = read_tracks(split_root / f"autotagging_moodtheme-{split}.tsv")
        if max_tracks is not None:
            tracks = tracks[:max_tracks]
        jobs.extend((split, track, dataset_root, output_root, config, overwrite) for track in tracks)

    rows, errors = [], []
    if workers == 1:
        results = []
        for job in jobs:
            try:
                results.append(_process_one(job))
            except (FileNotFoundError, OSError, ValueError) as error:
                results.append(error)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_one, job) for job in jobs]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except (FileNotFoundError, OSError, ValueError) as error:
                    results.append(error)

    for index, (job, result) in enumerate(zip(jobs, results), start=1):
        if isinstance(result, Exception):
            errors.append({"split": job[0], "track_id": job[1].track_id, "error": str(result)})
        else:
            rows.append(result)
        if index % 100 == 0 or index == len(jobs):
            print(f"processed {index}/{len(jobs)} ({len(errors)} errors)", flush=True)

    fields = ("split", "track_id", "artist_id", "audio_path", "feature_path", "tags")
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "track_id", "error"))
        writer.writeheader()
        writer.writerows(errors)
    summary = {
        "tracks_requested": len(jobs), "tracks_available": len(rows),
        "processed": sum(row["status"] == "processed" for row in rows),
        "already_cached": sum(row["status"] == "cached" for row in rows),
        "errors": len(errors),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, default=Path("mtg-jamendo-dataset/data"))
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--n-mels", type=int, default=96)
    parser.add_argument("--n-fft", type=int, default=2_048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tracks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PreprocessConfig(
        args.sample_rate, args.duration, args.n_mels, args.n_fft,
        args.hop_length, args.fmin, args.fmax,
    )
    summary = preprocess(
        args.dataset_root, args.metadata_root, args.output_root, config,
        args.split_index, args.workers, args.max_tracks, args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
