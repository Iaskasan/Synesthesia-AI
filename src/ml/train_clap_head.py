"""Train a multilabel linear head on cached frozen CLAP embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.audit_dataset import SELECTED_LABELS, SPLIT_NAMES
from src.ml.train_baseline import tune_thresholds


def parse_tags(value: str) -> tuple[str, ...]:
    tags = json.loads(value)
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("Manifest tags must be a JSON list of strings.")
    return tuple(tags)


def load_split(
    manifest_path: Path, embedding_root: Path, split: str, labels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    rows, targets = [], []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            path = embedding_root / row["embedding_path"]
            with np.load(path) as archive:
                embedding = np.asarray(archive["embedding"], dtype=np.float32)
            if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
                raise ValueError(f"Invalid embedding: {path}")
            tags = parse_tags(row["tags"])
            rows.append(embedding)
            targets.append([int(label in tags) for label in labels])
    if not rows:
        raise ValueError(f"No {split} embeddings found in {manifest_path}")
    return np.stack(rows), np.asarray(targets, dtype=np.int8)


def evaluate(
    truth: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray,
    labels: list[str],
) -> dict:
    predictions = probabilities >= thresholds
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, average=None, zero_division=0
    )
    per_label = {}
    for index, label in enumerate(labels):
        per_label[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "pr_auc": float(average_precision_score(truth[:, index], probabilities[:, index])),
        }
    return {
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "macro_pr_auc": float(average_precision_score(truth, probabilities, average="macro")),
        "per_label": per_label,
    }


def convergence_report(model: OneVsRestClassifier, labels: list[str], max_iterations: int) -> dict:
    """Report optimizer iterations and labels that reached the iteration cap."""
    iterations = {
        label: int(estimator.named_steps["logisticregression"].n_iter_[0])
        for label, estimator in zip(labels, model.estimators_)
    }
    return {
        "max_iterations": max_iterations,
        "iterations_by_label": iterations,
        "labels_at_iteration_limit": [
            label for label, count in iterations.items() if count >= max_iterations
        ],
    }


def train(args: argparse.Namespace) -> dict:
    manifest = args.embedding_root / "manifest.csv"
    labels = list(args.labels or SELECTED_LABELS)
    arrays = {
        split: load_split(manifest, args.embedding_root, split, labels)
        for split in SPLIT_NAMES
    }
    unusable = [
        label for index, label in enumerate(labels)
        if len(np.unique(arrays["train"][1][:, index])) != 2
    ]
    if unusable:
        raise ValueError("Training data lacks positive/negative examples for: " + ", ".join(unusable))

    model = OneVsRestClassifier(make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=args.max_iterations, class_weight="balanced",
            random_state=42, C=args.regularization, solver=args.solver,
        ),
    ), n_jobs=args.jobs)
    model.fit(*arrays["train"])
    validation_probabilities = model.predict_proba(arrays["validation"][0])
    thresholds = tune_thresholds(arrays["validation"][1], validation_probabilities)
    test_probabilities = model.predict_proba(arrays["test"][0])
    config = json.loads((args.embedding_root / "embedding_config.json").read_text())
    report = {
        "model_name": config["model_name"],
        "labels": labels,
        "thresholds": thresholds.tolist(),
        "embedding_dimension": int(arrays["train"][0].shape[1]),
        "tracks_used": {split: int(len(values[0])) for split, values in arrays.items()},
        "convergence": convergence_report(model, labels, args.max_iterations),
        "validation": evaluate(arrays["validation"][1], validation_probabilities, thresholds, labels),
        "test": evaluate(arrays["test"][1], test_probabilities, thresholds, labels),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "labels": labels, "thresholds": thresholds,
        "clap_config": config,
    }, args.output_dir / "clap_head.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    limited = report["convergence"]["labels_at_iteration_limit"]
    if limited:
        print(
            "Warning: classifiers at the iteration limit: " + ", ".join(limited),
            flush=True,
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/clap_head"))
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--regularization", type=float, default=1.0, help="Logistic regression C")
    parser.add_argument(
        "--solver", choices=("liblinear", "lbfgs", "saga"), default="liblinear",
        help="Binary optimizer; liblinear is reliable for the frozen 512-D embeddings",
    )
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
