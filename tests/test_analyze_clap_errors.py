import numpy as np

from src.ml.analyze_clap_errors import analyze_predictions


def test_analysis_reports_metrics_and_ranks_errors():
    truth = np.array([[1], [1], [0], [0]], dtype=np.int8)
    probabilities = np.array([[0.9], [0.2], [0.8], [0.1]])
    metadata = [
        {"track_id": str(i), "audio_path": f"{i}.mp3", "tags": []}
        for i in range(4)
    ]
    report = analyze_predictions(
        truth, probabilities, np.array([0.5]), ["happy"], metadata, 2
    )["happy"]
    assert report["support"] == 2
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert report["false_positive_count"] == 1
    assert report["false_negative_count"] == 1
    assert report["highest_confidence_false_positives"][0]["track_id"] == "2"
    assert report["lowest_confidence_false_negatives"][0]["track_id"] == "1"
