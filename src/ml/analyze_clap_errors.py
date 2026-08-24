"""Produce per-label metrics and inspectable errors for a saved CLAP head."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from src.ml.run_clap_diagnostics import aggregate_crop_probabilities
from src.ml.train_clap_head import parse_tags


def load_examples(
    manifest_path: Path, embedding_root: Path, split: str, labels: list[str],
    use_crops: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    features, targets, metadata = [], [], []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            with np.load(embedding_root / row["embedding_path"]) as archive:
                key = "crop_embeddings" if use_crops else "embedding"
                if key not in archive.files:
                    raise ValueError(f"{key} missing from {row['embedding_path']}")
                features.append(np.asarray(archive[key], dtype=np.float32))
            tags = parse_tags(row["tags"])
            targets.append([int(label in tags) for label in labels])
            metadata.append({
                "track_id": row["track_id"],
                "audio_path": row["audio_path"],
                "tags": list(tags),
            })
    if not features:
        raise ValueError(f"No {split} examples found")
    return np.stack(features), np.asarray(targets, dtype=np.int8), metadata


def predict_probabilities(bundle: dict, features: np.ndarray) -> np.ndarray:
    aggregation = bundle.get("crop_aggregation")
    if aggregation:
        track_count, crop_count, dimension = features.shape
        flat = bundle["model"].predict_proba(features.reshape(-1, dimension))
        return aggregate_crop_probabilities(
            flat.reshape(track_count, crop_count, len(bundle["labels"])), aggregation
        )
    return bundle["model"].predict_proba(features)


def analyze_predictions(
    truth: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray,
    labels: list[str], metadata: list[dict[str, str]], example_count: int,
) -> dict:
    predictions = probabilities >= thresholds
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, average=None, zero_division=0
    )
    report = {}
    for column, label in enumerate(labels):
        false_positive = np.flatnonzero((truth[:, column] == 0) & predictions[:, column])
        false_negative = np.flatnonzero((truth[:, column] == 1) & ~predictions[:, column])

        def examples(indices: np.ndarray, descending: bool) -> list[dict]:
            ordered = sorted(
                indices.tolist(), key=lambda index: probabilities[index, column],
                reverse=descending,
            )[:example_count]
            return [
                {
                    **metadata[index],
                    "probability": float(probabilities[index, column]),
                    "threshold": float(thresholds[column]),
                }
                for index in ordered
            ]

        report[label] = {
            "support": int(support[column]),
            "prevalence": float(truth[:, column].mean()),
            "predicted_positive_count": int(predictions[:, column].sum()),
            "precision": float(precision[column]),
            "recall": float(recall[column]),
            "f1": float(f1[column]),
            "pr_auc": float(average_precision_score(truth[:, column], probabilities[:, column])),
            "false_positive_count": int(len(false_positive)),
            "false_negative_count": int(len(false_negative)),
            "highest_confidence_false_positives": examples(false_positive, True),
            "lowest_confidence_false_negatives": examples(false_negative, False),
        }
    return report


def run(args: argparse.Namespace) -> dict:
    bundle = joblib.load(args.model)
    labels = list(bundle["labels"])
    thresholds = np.asarray(bundle["thresholds"])
    result = {
        "model": str(args.model),
        "experiment": bundle.get("experiment"),
        "note": "Thresholds were selected on validation; test is descriptive only.",
        "splits": {},
    }
    for split in args.splits:
        features, truth, metadata = load_examples(
            args.embedding_root / "manifest.csv", args.embedding_root, split,
            labels, use_crops=bool(bundle.get("crop_aggregation")),
        )
        probabilities = predict_probabilities(bundle, features)
        result["splits"][split] = {
            "track_count": len(metadata),
            "per_label": analyze_predictions(
                truth, probabilities, thresholds, labels, metadata, args.examples
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path,
        default=Path("artifacts/clap_diagnostics/selected_head.joblib"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/clap_diagnostics/error_analysis.json"),
    )
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation", "test"])
    parser.add_argument("--examples", type=int, default=5)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
