import numpy as np

from src.ml.prepare_validation_reviews import sample_review_rows


def test_review_queue_is_validation_only_and_reproducible():
    probabilities = np.array([[0.1], [0.7], [0.9]])
    truth = np.array([[0], [1], [0]], dtype=np.int8)
    metadata = [
        {"track_id": str(i), "audio_path": f"{i}.mp3", "tags": []}
        for i in range(3)
    ]
    arguments = (probabilities, truth, ["dream"], np.array([0.5]), metadata, 2, 7)
    first = sample_review_rows(*arguments)
    second = sample_review_rows(*arguments)
    assert first == second
    assert len(first) == 2
    assert all(row["split"] == "validation" for row in first)
    assert all(row["verdict"] == "" for row in first)
