"""Structured human-review records and validation-only threshold suggestions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score

VERDICTS = ("correct", "incorrect", "ambiguous")


@dataclass(frozen=True)
class PredictionReview:
    label: str
    probability: float
    threshold: float
    predicted: bool
    verdict: str
    split: str = "unassigned"
    track_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        if self.verdict not in VERDICTS:
            raise ValueError(f"Unknown verdict: {self.verdict}")
        if not 0 <= self.probability <= 1 or not 0 <= self.threshold <= 1:
            raise ValueError("Probability and threshold must be between zero and one.")
        return asdict(self)


def reviewed_target(review: PredictionReview) -> int | None:
    """Convert agreement with a model decision into a human binary target."""
    if review.verdict == "ambiguous":
        return None
    if review.verdict not in VERDICTS:
        raise ValueError(f"Unknown verdict: {review.verdict}")
    return int(review.predicted == (review.verdict == "correct"))


def suggest_validation_thresholds(
    reviews: list[PredictionReview], *, min_decisive: int = 10
) -> dict[str, dict[str, float | int]]:
    """Find per-label F1 thresholds from human-reviewed validation examples.

    Reviews from test or unassigned/user-uploaded audio are rejected to prevent
    leakage into model selection. Ambiguous judgments are retained as evidence
    but excluded from the numeric recommendation.
    """
    invalid = sorted({review.split for review in reviews if review.split != "validation"})
    if invalid:
        raise ValueError(
            "Threshold tuning only accepts validation reviews; found: "
            + ", ".join(invalid)
        )
    grouped: dict[str, list[PredictionReview]] = defaultdict(list)
    for review in reviews:
        grouped[review.label].append(review)
    suggestions = {}
    for label, label_reviews in grouped.items():
        decisive = [(item.probability, reviewed_target(item)) for item in label_reviews]
        decisive = [(score, target) for score, target in decisive if target is not None]
        if len(decisive) < min_decisive:
            continue
        scores = np.asarray([item[0] for item in decisive], dtype=float)
        truth = np.asarray([item[1] for item in decisive], dtype=np.int8)
        if len(np.unique(truth)) < 2:
            continue
        candidates = np.unique(np.r_[0.0, scores, 1.0])
        metrics = [f1_score(truth, scores >= value, zero_division=0) for value in candidates]
        best = int(np.argmax(metrics))
        suggestions[label] = {
            "threshold": float(candidates[best]),
            "reviewed_f1": float(metrics[best]),
            "decisive_reviews": len(decisive),
            "ambiguous_reviews": len(label_reviews) - len(decisive),
        }
    return suggestions

