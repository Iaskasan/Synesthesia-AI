import csv

import pytest

from src.app.validation_review_app import first_unreviewed, load_queue, save_queue


FIELDS = [
    "split", "track_id", "audio_path", "label", "probability", "threshold",
    "predicted", "dataset_target", "dataset_tags", "verdict", "notes",
]


def row(verdict="", notes=""):
    return {
        "split": "validation", "track_id": "track_1", "audio_path": "1.mp3",
        "label": "happy", "probability": "0.8", "threshold": "0.5",
        "predicted": "true", "dataset_target": "false", "dataset_tags": "[]",
        "verdict": verdict, "notes": notes,
    }


def test_queue_round_trip_and_resume(tmp_path):
    path = tmp_path / "queue.csv"
    rows = [row("correct", "clear"), row()]
    save_queue(path, rows, FIELDS)

    loaded, fields = load_queue(path)

    assert fields == FIELDS
    assert loaded == rows
    assert first_unreviewed(loaded) == 1


def test_load_queue_rejects_invalid_verdict(tmp_path):
    path = tmp_path / "queue.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row("yes"))

    with pytest.raises(ValueError, match="invalid verdicts"):
        load_queue(path)
