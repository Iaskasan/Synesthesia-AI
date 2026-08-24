import pytest

from src.ml.review_feedback import PredictionReview, reviewed_target, suggest_validation_thresholds


def review(score, predicted, verdict, split="validation"):
    return PredictionReview("dream", score, 0.5, predicted, verdict, split)


def test_reviewed_target_handles_agreement_disagreement_and_ambiguity():
    assert reviewed_target(review(0.8, True, "correct")) == 1
    assert reviewed_target(review(0.8, True, "incorrect")) == 0
    assert reviewed_target(review(0.2, False, "incorrect")) == 1
    assert reviewed_target(review(0.2, False, "ambiguous")) is None


def test_threshold_suggestions_only_use_validation_and_skip_ambiguous():
    reviews = [
        review(0.1, False, "correct"), review(0.2, False, "correct"),
        review(0.7, True, "correct"), review(0.8, True, "correct"),
        review(0.6, True, "ambiguous"),
    ]
    result = suggest_validation_thresholds(reviews, min_decisive=4)["dream"]
    assert result["threshold"] == pytest.approx(0.7)
    assert result["reviewed_f1"] == 1.0
    assert result["ambiguous_reviews"] == 1


def test_threshold_suggestions_reject_test_reviews():
    with pytest.raises(ValueError, match="validation"):
        suggest_validation_thresholds([review(0.8, True, "correct", "test")])
