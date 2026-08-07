"""Train a reproducible multilabel baseline on MTG-Jamendo audio."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.audio.analysis import AudioFeatures, analyze_audio
from src.audio.load_audio import load_audio
from src.data.audit_dataset import SPLIT_NAMES, Track, read_tracks


FEATURE_NAMES = (
    "tempo_bpm",
    "rms_energy",
    "spectral_centroid_hz",
    "spectral_contrast",
    "zero_crossing_rate",
    *(f"chroma_{index}" for index in range(12)),
    *(f"mfcc_{index}" for index in range(13)),
)


def feature_vector(features: AudioFeatures) -> np.ndarray:
    """Flatten application audio features in a stable order."""
    return np.asarray(
        [
            features.tempo_bpm,
            features.rms_energy,
            features.spectral_centroid_hz,
            features.spectral_contrast,
            features.zero_crossing_rate,
            *features.chroma,
            *features.mfcc,
        ],
        dtype=np.float32,
    )


def audio_path(dataset_root: Path, track: Track) -> Path:
    """Resolve a metadata path such as 00/7400.mp3 to its low-quality MP3."""
    relative = Path(track.path)
    return dataset_root / relative.with_name(f"{relative.stem}.low{relative.suffix}")


def select_labels(tracks: list[Track], count: int) -> list[str]:
    """Select the most frequent training labels with deterministic tie-breaking."""
    frequencies = Counter(tag for track in tracks for tag in track.tags)
    return [tag for tag, _ in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:count]]


def encode_labels(tracks: list[Track], labels: list[str]) -> np.ndarray:
    return np.asarray(
        [[int(label in track.tags) for label in labels] for track in tracks],
        dtype=np.int8,
    )


def extract_split(
    tracks: list[Track],
    dataset_root: Path,
    labels: list[str],
    duration: float,
    sample_rate: int,
    max_tracks: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load available tracks and return features, targets, and skipped errors."""
    rows: list[np.ndarray] = []
    targets: list[list[int]] = []
    errors: list[str] = []
    if max_tracks and max_tracks < len(tracks):
        # Cover the whole official split instead of taking a biased prefix.
        indices = np.linspace(0, len(tracks) - 1, max_tracks, dtype=int)
        selected = [tracks[index] for index in indices]
    else:
        selected = tracks
    for index, track in enumerate(selected, start=1):
        path = audio_path(dataset_root, track)
        try:
            y, sr = load_audio(path, duration=duration, sample_rate=sample_rate)
            rows.append(feature_vector(analyze_audio(y, sr)))
            targets.append([int(label in track.tags) for label in labels])
        except (FileNotFoundError, OSError, ValueError) as error:
            errors.append(f"{track.track_id}\t{path}\t{error}")
        if index % 100 == 0:
            print(f"  processed {index}/{len(selected)} tracks")
    if not rows:
        raise ValueError(
            f"No audio could be loaded below {dataset_root}. "
            "Expected paths like 00/7400.low.mp3."
        )
    return np.stack(rows), np.asarray(targets, dtype=np.int8), errors


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Choose one F1-maximizing probability threshold per label."""
    candidates = np.arange(0.10, 0.91, 0.05)
    thresholds = []
    for column in range(y_true.shape[1]):
        scores = [
            f1_score(y_true[:, column], probabilities[:, column] >= value, zero_division=0)
            for value in candidates
        ]
        thresholds.append(float(candidates[int(np.argmax(scores))]))
    return np.asarray(thresholds)


def metrics(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict:
    predictions = probabilities >= thresholds
    result = {
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }
    if np.all((y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < len(y_true))):
        result["macro_pr_auc"] = float(average_precision_score(y_true, probabilities, average="macro"))
    else:
        result["macro_pr_auc"] = None
    return result


def train(args: argparse.Namespace) -> dict:
    split_root = args.metadata_root / "splits" / f"split-{args.split_index}"
    tracks = {
        name: read_tracks(split_root / f"autotagging_moodtheme-{name}.tsv")
        for name in SPLIT_NAMES
    }
    labels = args.labels or select_labels(tracks["train"], args.label_count)
    if not labels:
        raise ValueError("At least one label is required.")

    arrays = {}
    skipped: list[str] = []
    for name in SPLIT_NAMES:
        print(f"Extracting {name} features...")
        x, y, errors = extract_split(
            tracks[name], args.dataset_root, labels, args.duration,
            args.sample_rate, args.max_tracks,
        )
        arrays[name] = (x, y)
        skipped.extend(f"{name}\t{error}" for error in errors)

    usable_indices = [
        index for index in range(len(labels))
        if len(np.unique(arrays["train"][1][:, index])) == 2
    ]
    unusable = [label for index, label in enumerate(labels) if index not in usable_indices]
    if args.labels and unusable:
        raise ValueError(f"Training data has no positive/negative examples for: {', '.join(unusable)}")
    if not usable_indices:
        raise ValueError("The selected training sample contains no usable labels; increase --max-tracks.")
    if unusable:
        print("Dropping labels absent from the limited training sample: " + ", ".join(unusable))
        labels = [labels[index] for index in usable_indices]
        arrays = {
            name: (x, y[:, usable_indices])
            for name, (x, y) in arrays.items()
        }

    model = OneVsRestClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )
    )
    model.fit(*arrays["train"])
    validation_probabilities = model.predict_proba(arrays["validation"][0])
    thresholds = tune_thresholds(arrays["validation"][1], validation_probabilities)
    test_probabilities = model.predict_proba(arrays["test"][0])

    report = {
        "labels": labels,
        "labels_dropped": unusable,
        "thresholds": thresholds.tolist(),
        "split_index": args.split_index,
        "duration": args.duration,
        "sample_rate": args.sample_rate,
        "tracks_used": {name: int(len(value[0])) for name, value in arrays.items()},
        "tracks_skipped": len(skipped),
        "validation": metrics(arrays["validation"][1], validation_probabilities, thresholds),
        "test": metrics(arrays["test"][1], test_probabilities, thresholds),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "labels": labels, "thresholds": thresholds,
         "feature_names": FEATURE_NAMES, "sample_rate": args.sample_rate,
         "duration": args.duration},
        args.output_dir / "baseline.joblib",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "skipped_tracks.tsv").write_text("split\ttrack_id\tpath\terror\n" + "\n".join(skipped))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, default=Path("mtg-jamendo-dataset/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--label-count", type=int, default=12)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--max-tracks", type=int, help="Limit each split for a quick smoke test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
