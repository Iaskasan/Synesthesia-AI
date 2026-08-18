import numpy as np

from src.ml.run_clap_diagnostics import (
    DEFAULT_LABELS,
    aggregate_crop_probabilities,
    evaluate_validation,
    prevalence_probabilities,
    zero_shot_probabilities,
)


def test_default_vocabulary_is_reduced_and_unique():
    assert len(DEFAULT_LABELS) == 10
    assert len(set(DEFAULT_LABELS)) == len(DEFAULT_LABELS)


def test_prevalence_reference_repeats_training_rates():
    truth = np.array([[1, 0], [0, 0], [1, 1]], dtype=np.int8)
    result = prevalence_probabilities(truth, 2)
    np.testing.assert_allclose(result, [[2 / 3, 1 / 3], [2 / 3, 1 / 3]])


def test_prevalence_reference_can_be_thresholded_below_point_one():
    truth = np.zeros((20, 1), dtype=np.int8)
    truth[0, 0] = 1
    probabilities = np.full((20, 1), 0.05)
    thresholds, metrics = evaluate_validation(truth, probabilities)
    assert thresholds[0] == 0.05
    assert metrics["macro_f1"] > 0


def test_zero_shot_scores_favor_matching_unit_vectors():
    audio = np.eye(2, dtype=np.float32)
    text = np.eye(2, dtype=np.float32)
    probabilities = zero_shot_probabilities(audio, text)
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]


def test_validation_evaluation_returns_one_threshold_per_label():
    truth = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int8)
    probabilities = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]
    ])
    thresholds, metrics = evaluate_validation(truth, probabilities)
    assert thresholds.shape == (2,)
    assert metrics["macro_f1"] == 1.0


def test_crop_probability_aggregation():
    values = np.array([[[0.2, 0.8], [0.6, 0.4]]])
    np.testing.assert_allclose(aggregate_crop_probabilities(values, "mean"), [[0.4, 0.6]])
    np.testing.assert_allclose(aggregate_crop_probabilities(values, "max"), [[0.6, 0.8]])
