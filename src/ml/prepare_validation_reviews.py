"""Create a reproducible, representative validation review queue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from src.ml.analyze_clap_errors import load_examples, predict_probabilities


def sample_review_rows(
    probabilities: np.ndarray,
    truth: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray,
    metadata: list[dict],
    examples_per_label: int,
    seed: int,
) -> list[dict]:
    """Randomly sample validation tracks independently for each label."""
    if examples_per_label < 1:
        raise ValueError("examples_per_label must be positive")
    if probabilities.shape != truth.shape or probabilities.shape[1] != len(labels):
        raise ValueError("Prediction, truth, and label shapes do not match.")
    if probabilities.shape[0] != len(metadata) or len(thresholds) != len(labels):
        raise ValueError("Metadata or threshold lengths do not match predictions.")
    rng = np.random.default_rng(seed)
    rows = []
    sample_size = min(examples_per_label, len(metadata))
    for column, label in enumerate(labels):
        for index in rng.choice(len(metadata), size=sample_size, replace=False):
            probability = float(probabilities[index, column])
            rows.append({
                "split": "validation",
                "track_id": metadata[index]["track_id"],
                "audio_path": metadata[index]["audio_path"],
                "label": label,
                "probability": probability,
                "threshold": float(thresholds[column]),
                "predicted": str(probability >= thresholds[column]).lower(),
                "dataset_target": str(bool(truth[index, column])).lower(),
                "dataset_tags": json.dumps(metadata[index]["tags"]),
                "verdict": "",
                "notes": "",
            })
    return rows


def run(args: argparse.Namespace) -> list[dict]:
    bundle = joblib.load(args.model)
    labels = list(bundle["labels"])
    features, truth, metadata = load_examples(
        args.embedding_root / "manifest.csv", args.embedding_root, "validation",
        labels, use_crops=bool(bundle.get("crop_aggregation")),
    )
    probabilities = predict_probabilities(bundle, features)
    rows = sample_review_rows(
        probabilities, truth, labels, np.asarray(bundle["thresholds"]), metadata,
        args.examples_per_label, args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path,
        default=Path("artifacts/clap_diagnostics/selected_head.joblib"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/clap_diagnostics/validation_review_queue.csv"),
    )
    parser.add_argument("--examples-per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    rows = run(build_parser().parse_args())
    print(f"Wrote {len(rows)} validation review assignments.")


if __name__ == "__main__":
    main()
