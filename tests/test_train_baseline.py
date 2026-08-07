import numpy as np

from src.audio.analysis import AudioFeatures
from src.data.audit_dataset import Track
from src.ml.train_baseline import audio_path, feature_vector, select_labels, tune_thresholds


def test_audio_path_converts_jamendo_metadata_path(tmp_path):
    track = Track("track_7400", "artist_1", "00/7400.mp3", 146.1, ("drama",))
    assert audio_path(tmp_path, track) == tmp_path / "00/7400.low.mp3"


def test_feature_vector_has_stable_fixed_size():
    features = AudioFeatures(120, 0.2, 2000, 10, 0.1, tuple(range(12)), tuple(range(13)))
    vector = feature_vector(features)
    assert vector.shape == (30,)
    assert vector.dtype == np.float32


def test_select_labels_uses_training_frequency():
    tracks = [
        Track("1", "a", "1.mp3", 1, ("calm", "dark")),
        Track("2", "b", "2.mp3", 1, ("calm", "happy")),
    ]
    assert select_labels(tracks, 2) == ["calm", "dark"]


def test_tune_thresholds_returns_one_threshold_per_label():
    truth = np.array([[0, 1], [1, 0], [1, 1]])
    probabilities = np.array([[0.1, 0.8], [0.7, 0.2], [0.6, 0.9]])
    thresholds = tune_thresholds(truth, probabilities)
    assert thresholds.shape == (2,)
    assert np.all((thresholds >= 0.1) & (thresholds <= 0.9))
