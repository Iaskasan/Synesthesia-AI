"""Train a separate crop head with space, evaluating reviews without fitting them.

Run from the repository root with python -m src.ml.train_space_candidate.
The existing checkpoint and test split are never modified or evaluated.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.ml.analyze_clap_errors import predict_probabilities
from src.ml.run_clap_diagnostics import load_crop_split, logistic_model, validation_thresholds
from src.ml.train_clap_head import convergence_report, evaluate


def binary_metrics(truth, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return dict(precision=float(precision), recall=float(recall), f1=float(f1),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def main():
    root = Path("artifacts/clap_diagnostics")
    output = Path("artifacts/clap_space_candidate")
    output.mkdir(parents=True, exist_ok=False)
    embedding_root = Path("/mnt/g/AI/datasets/processed/clap-music-v1")
    source = root / "selected_head.joblib"
    baseline = joblib.load(source)
    assert baseline["crop_aggregation"] == "mean"
    labels = [label for label in baseline["labels"] if label != "inspiring"] + ["space"]
    review_path = root / "validation_review_queue_calibration_check.csv"
    with review_path.open() as handle:
        reviews = list(csv.DictReader(handle))
    assert len(reviews) == 200
    assert all(r["split"] == "validation" and r["verdict"] in
               ("correct", "incorrect", "ambiguous") for r in reviews)
    excluded = set()
    for path in (root / "validation_review_queue.csv",
                 root / "validation_review_queue_followup.csv", review_path):
        with path.open() as handle:
            excluded.update(r["track_id"] for r in csv.DictReader(handle))
    with (embedding_root / "manifest.csv").open() as handle:
        manifest = list(csv.DictReader(handle))
    train_ids = {r["track_id"] for r in manifest if r["split"] == "train"}
    assert not train_ids.intersection(excluded)
    val_rows = [r for r in manifest if r["split"] == "validation"]
    val_index = {r["track_id"]: i for i, r in enumerate(val_rows)}
    calibration = np.array([r["track_id"] not in excluded for r in val_rows])
    print("Loading cached training crops...", flush=True)
    train_x, train_y = load_crop_split(embedding_root / "manifest.csv", embedding_root,
                                      "train", labels)
    print("Loading cached validation crops...", flush=True)
    val_x, val_y = load_crop_split(embedding_root / "manifest.csv", embedding_root,
                                  "validation", labels)
    c = baseline["model"].estimators_[0].named_steps["logisticregression"].C
    print(f"Fitting {len(labels)} labels on {len(train_x)} training tracks (C={c})...", flush=True)
    model = logistic_model(c, 1).fit(
        train_x.reshape(-1, train_x.shape[-1]),
        np.repeat(train_y, train_x.shape[1], axis=0),
    )
    thresholds = np.array([baseline["thresholds"][baseline["labels"].index(label)]
                           for label in labels[:-1]] + [0.5])
    bundle = dict(model=model, labels=labels, thresholds=thresholds,
                  clap_config=baseline["clap_config"], crop_aggregation="mean",
                  experiment="crop_logistic_mean_space_candidate")
    scores = predict_probabilities(bundle, val_x)
    thresholds[-1] = validation_thresholds(val_y[calibration, -1:], scores[calibration, -1:])[0]
    # This sad candidate was selected on the earlier followup review and then
    # checked on the completed calibration-check queue; retain other thresholds.
    earlier = json.loads((root / "review_threshold_comparison.json").read_text())
    thresholds[labels.index("sad")] = earlier["labels"]["sad"]["threshold"]
    old_scores = predict_probabilities(baseline, val_x)
    report = dict(
        labels=labels, thresholds=dict(zip(labels, thresholds.tolist())),
        train_tracks=len(train_x), validation_tracks=len(val_x),
        space_calibration_tracks=int(calibration.sum()),
        space_train_positives=int(train_y[:, -1].sum()),
        space_calibration_positives=int(val_y[calibration, -1].sum()),
        test_evaluated=False, deployed=False,
        threshold_policy="Keep baseline shared-label thresholds except sad (previously reviewed candidate). Space threshold maximizes dataset F1 on validation tracks excluding all three review queues.",
        limitations="Manual reviews are validation evidence, not a sealed test. Sad was selected using these reviews. Space has only five decisive human positives. Dataset tags are incomplete.",
        convergence=convergence_report(model, labels, 2000),
        source_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reviews_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
        dataset_validation=evaluate(val_y, scores, thresholds, labels),
        manual_reviews={}, shared_label_max_score_difference={},
    )
    for label in labels[:-1]:
        j, k = labels.index(label), baseline["labels"].index(label)
        report["shared_label_max_score_difference"][label] = float(np.max(np.abs(scores[:, j] - old_scores[:, k])))
    predictions = []
    for label in dict.fromkeys(r["label"] for r in reviews):
        rr = [r for r in reviews if r["label"] == label and r["verdict"] != "ambiguous"]
        truth = [int((r["predicted"] == "true") == (r["verdict"] == "correct")) for r in rr]
        indices = [val_index[r["track_id"]] for r in rr]
        j = labels.index(label)
        pred = scores[indices, j] >= thresholds[j]
        result = dict(decisive=len(rr), positives=sum(truth), ambiguous=sum(
            r["label"] == label and r["verdict"] == "ambiguous" for r in reviews),
            candidate=binary_metrics(truth, pred))
        if label in baseline["labels"]:
            k = baseline["labels"].index(label)
            result["baseline"] = binary_metrics(truth, old_scores[indices, k] >= baseline["thresholds"][k])
        report["manual_reviews"][label] = result
        for r, target, i in zip(rr, truth, indices):
            predictions.append(dict(track_id=r["track_id"], label=label, human_target=target,
                                    probability=float(scores[i, j]), threshold=float(thresholds[j]),
                                    predicted=bool(scores[i, j] >= thresholds[j])))
    joblib.dump(bundle, output / "selected_head.joblib")
    np.savez_compressed(output / "validation_predictions.npz", probabilities=scores,
                        targets=val_y, labels=np.array(labels),
                        track_ids=np.array([r["track_id"] for r in val_rows]),
                        space_calibration_mask=calibration)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "review_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(report["manual_reviews"], indent=2), flush=True)
    print(f"Candidate saved to {output}; live checkpoint unchanged.", flush=True)


if __name__ == "__main__":
    main()
