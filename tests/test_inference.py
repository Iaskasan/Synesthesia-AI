import numpy as np
import pytest

from src.ml.inference import rank_predictions


def test_rank_predictions_applies_thresholds_and_sorts_scores():
    ranked = rank_predictions(
        ["calm", "dark", "happy"], np.array([0.4, 0.8, 0.6]),
        np.array([0.3, 0.9, 0.6]),
    )
    assert [item.label for item in ranked] == ["dark", "happy", "calm"]
    assert [item.detected for item in ranked] == [False, True, True]


def test_rank_predictions_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal length"):
        rank_predictions(["calm"], np.array([0.2, 0.3]), np.array([0.5]))
