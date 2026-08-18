"""Compare CLAP reference and supervised heads without touching the test split.

The default experiment uses ten frequent, visually distinct mood labels.  It
compares a prevalence-only reference, CLAP text/audio similarity, regularized
logistic regression heads, and a small nonlinear MLP.  Model selection uses
the validation split; test evaluation is an explicit, final-selection step.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.train_clap_head import load_split


DEFAULT_LABELS = (
    "happy", "energetic", "relaxing", "emotional", "dark",
    "epic", "dream", "inspiring", "sad", "romantic",
)
PROMPT_TEMPLATES = (
    "music that feels {label}",
    "a {label} piece of music",
    "the mood of this music is {label}",
)


def score_predictions(
    truth: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict[str, float]:
    predictions = probabilities >= thresholds
    return {
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "macro_pr_auc": float(average_precision_score(truth, probabilities, average="macro")),
    }


def evaluate_validation(
    truth: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    thresholds = validation_thresholds(truth, probabilities)
    return thresholds, score_predictions(truth, probabilities, thresholds)


def validation_thresholds(truth: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Find exact per-label F1 optima, including scores below 0.10."""
    selected = []
    for column in range(truth.shape[1]):
        precision, recall, thresholds = precision_recall_curve(
            truth[:, column], probabilities[:, column]
        )
        if not len(thresholds):
            selected.append(float(probabilities[0, column]))
            continue
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1],
            precision[:-1] + recall[:-1],
            out=np.zeros_like(precision[:-1]),
            where=(precision[:-1] + recall[:-1]) != 0,
        )
        selected.append(float(thresholds[int(np.argmax(f1))]))
    return np.asarray(selected)


def prevalence_probabilities(train_truth: np.ndarray, row_count: int) -> np.ndarray:
    prevalence = train_truth.mean(axis=0, dtype=np.float64)
    return np.tile(prevalence, (row_count, 1))


def load_clap_text_embeddings(model_name: str, labels: list[str]) -> np.ndarray:
    """Encode prompt ensembles in the same normalized CLAP space as the cache."""
    import torch
    from transformers import AutoTokenizer, ClapModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = ClapModel.from_pretrained(model_name, local_files_only=True).eval()
    prompts = [template.format(label=label) for label in labels for template in PROMPT_TEMPLATES]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        encoded = model.get_text_features(**inputs)
    if hasattr(encoded, "pooler_output"):
        encoded = encoded.pooler_output
    values = encoded.detach().float().cpu().numpy()
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
    values = values.reshape(len(labels), len(PROMPT_TEMPLATES), -1).mean(axis=1)
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
    return values.astype(np.float32)


def zero_shot_probabilities(audio: np.ndarray, text: np.ndarray) -> np.ndarray:
    if audio.shape[1] != text.shape[1]:
        raise ValueError("Audio and text embedding dimensions differ.")
    audio = audio / np.linalg.norm(audio, axis=1, keepdims=True).clip(min=1e-12)
    # A fixed temperature only maps cosine scores onto a threshold-friendly
    # range; it does not alter label ranking or PR-AUC.
    return expit((audio @ text.T) / 0.07)


def load_crop_split(
    manifest_path: Path, embedding_root: Path, split: str, labels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Load arrays shaped (tracks, crops, dimensions) and track targets."""
    rows, targets = [], []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            path = embedding_root / row["embedding_path"]
            with np.load(path) as archive:
                if "crop_embeddings" not in archive.files:
                    raise ValueError(
                        f"Crop embeddings missing from {path}; rerun extraction with "
                        "--store-crop-embeddings"
                    )
                values = np.asarray(archive["crop_embeddings"], dtype=np.float32)
            if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid crop embeddings: {path}")
            tags = json.loads(row["tags"])
            rows.append(values)
            targets.append([int(label in tags) for label in labels])
    if not rows:
        raise ValueError(f"No {split} crop embeddings found in {manifest_path}")
    crop_counts = {len(values) for values in rows}
    if len(crop_counts) != 1:
        raise ValueError(f"Inconsistent crop counts in {split}: {sorted(crop_counts)}")
    return np.stack(rows), np.asarray(targets, dtype=np.int8)


def aggregate_crop_probabilities(
    probabilities: np.ndarray, method: str
) -> np.ndarray:
    if probabilities.ndim != 3:
        raise ValueError("Crop probabilities must have shape (tracks, crops, labels).")
    if method == "mean":
        return probabilities.mean(axis=1)
    if method == "max":
        return probabilities.max(axis=1)
    raise ValueError(f"Unknown crop aggregation method: {method}")


def logistic_model(c: float, jobs: int) -> OneVsRestClassifier:
    return OneVsRestClassifier(make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000,
            random_state=42, solver="liblinear",
        ),
    ), n_jobs=jobs)


def mlp_model(hidden_units: int, max_iterations: int) -> OneVsRestClassifier:
    return OneVsRestClassifier(make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(hidden_units,), activation="relu",
            alpha=1e-3, batch_size=256, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=12,
            max_iter=max_iterations, random_state=42,
        ),
    ))


def _experiment(
    name: str,
    probabilities: np.ndarray,
    validation_truth: np.ndarray,
    report: dict,
) -> tuple[np.ndarray, dict]:
    thresholds, metrics = evaluate_validation(validation_truth, probabilities)
    report["experiments"][name] = {
        "thresholds": thresholds.tolist(), "validation": metrics,
    }
    print(f"{name}: macro PR-AUC={metrics['macro_pr_auc']:.4f}, "
          f"macro F1={metrics['macro_f1']:.4f}", flush=True)
    return thresholds, metrics


def run(args: argparse.Namespace, text_encoder: Callable | None = None) -> dict:
    labels = list(args.labels)
    manifest = args.embedding_root / "manifest.csv"
    train_x, train_y = load_split(manifest, args.embedding_root, "train", labels)
    validation_x, validation_y = load_split(
        manifest, args.embedding_root, "validation", labels
    )
    config = json.loads((args.embedding_root / "embedding_config.json").read_text())
    report: dict = {
        "selection_split": "validation",
        "test_evaluated": bool(args.evaluate_test),
        "labels": labels,
        "tracks_used": {"train": len(train_x), "validation": len(validation_x)},
        "experiments": {},
    }

    _experiment(
        "prevalence", prevalence_probabilities(train_y, len(validation_y)),
        validation_y, report,
    )
    encoder = text_encoder or load_clap_text_embeddings
    text_embeddings = encoder(config["model_name"], labels)
    _experiment(
        "clap_zero_shot", zero_shot_probabilities(validation_x, text_embeddings),
        validation_y, report,
    )

    fitted: dict[str, object] = {}
    for c in args.regularization:
        name = f"logistic_c_{c:g}"
        model = logistic_model(c, args.jobs).fit(train_x, train_y)
        _experiment(name, model.predict_proba(validation_x), validation_y, report)
        fitted[name] = model

    name = f"mlp_{args.hidden_units}"
    model = mlp_model(args.hidden_units, args.mlp_max_iterations).fit(train_x, train_y)
    _experiment(name, model.predict_proba(validation_x), validation_y, report)
    fitted[name] = model

    if args.include_crop_head:
        train_crops, crop_train_y = load_crop_split(
            manifest, args.embedding_root, "train", labels
        )
        validation_crops, crop_validation_y = load_crop_split(
            manifest, args.embedding_root, "validation", labels
        )
        crop_count = train_crops.shape[1]
        if validation_crops.shape[1] != crop_count:
            raise ValueError("Train and validation crop counts differ.")
        crop_model = logistic_model(args.crop_regularization, args.jobs).fit(
            train_crops.reshape(-1, train_crops.shape[-1]),
            np.repeat(crop_train_y, crop_count, axis=0),
        )
        flat_probabilities = crop_model.predict_proba(
            validation_crops.reshape(-1, validation_crops.shape[-1])
        )
        crop_probabilities = flat_probabilities.reshape(
            len(validation_crops), crop_count, len(labels)
        )
        for aggregation in ("mean", "max"):
            crop_name = f"crop_logistic_{aggregation}"
            _experiment(
                crop_name,
                aggregate_crop_probabilities(crop_probabilities, aggregation),
                crop_validation_y, report,
            )
            fitted[crop_name] = crop_model

    supervised = [name for name in report["experiments"] if name in fitted]
    winner = max(
        supervised,
        key=lambda item: (
            report["experiments"][item]["validation"]["macro_pr_auc"],
            report["experiments"][item]["validation"]["macro_f1"],
        ),
    )
    report["selected_supervised_model"] = winner
    selected = report["experiments"][winner]

    if args.evaluate_test:
        if winner.startswith("crop_logistic_"):
            test_crops, test_y = load_crop_split(
                manifest, args.embedding_root, "test", labels
            )
            report["tracks_used"]["test"] = len(test_crops)
            flat = fitted[winner].predict_proba(
                test_crops.reshape(-1, test_crops.shape[-1])
            )
            crop_probabilities = flat.reshape(
                len(test_crops), test_crops.shape[1], len(labels)
            )
            test_probabilities = aggregate_crop_probabilities(
                crop_probabilities, winner.rsplit("_", 1)[-1]
            )
        else:
            test_x, test_y = load_split(manifest, args.embedding_root, "test", labels)
            report["tracks_used"]["test"] = len(test_x)
            test_probabilities = fitted[winner].predict_proba(test_x)
        selected["test"] = score_predictions(
            test_y, test_probabilities, np.asarray(selected["thresholds"])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": fitted[winner], "labels": labels,
        "thresholds": np.asarray(selected["thresholds"]),
        "clap_config": config, "experiment": winner,
        "crop_aggregation": (
            winner.rsplit("_", 1)[-1] if winner.startswith("crop_logistic_") else None
        ),
    }, args.output_dir / "selected_head.joblib")
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/clap_diagnostics"))
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--mlp-max-iterations", type=int, default=150)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--include-crop-head", action="store_true")
    parser.add_argument("--crop-regularization", type=float, default=0.01)
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Evaluate only the validation-selected model on the sealed test split",
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
