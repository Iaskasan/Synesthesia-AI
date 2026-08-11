import csv
import json
import tarfile
from io import BytesIO

from src.data.audit_dataset import SELECTED_LABELS, read_tracks, run_audit


def _write_split(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]
        )
        writer.writerows(rows)


def test_read_tracks_supports_multiple_tag_columns(tmp_path):
    split = tmp_path / "split.tsv"
    _write_split(
        split,
        [["track_1", "artist_1", "album_1", "01/1.mp3", "30", "mood/theme---calm", "mood/theme---happy"]],
    )

    track = read_tracks(split)[0]

    assert track.tags == ("calm", "happy")


def test_audit_reports_frequencies_inventory_and_leakage(tmp_path):
    metadata = tmp_path / "metadata"
    split_root = metadata / "splits" / "split-0"
    split_root.mkdir(parents=True)
    _write_split(
        split_root / "autotagging_moodtheme-train.tsv",
        [["track_1", "artist_1", "album_1", "01/1.mp3", "30", "mood/theme---calm"]],
    )
    _write_split(
        split_root / "autotagging_moodtheme-validation.tsv",
        [["track_2", "artist_2", "album_2", "02/2.mp3", "30", "mood/theme---dark"]],
    )
    _write_split(
        split_root / "autotagging_moodtheme-test.tsv",
        [["track_3", "artist_1", "album_3", "03/3.mp3", "30", "mood/theme---calm"]],
    )

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with tarfile.open(dataset / "audio.tar", "w") as archive:
        for name in ("01/1.low.mp3", "02/2.low.mp3"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, BytesIO(b"x"))

    output = tmp_path / "output"
    summary = run_audit(dataset, metadata, output)

    assert summary["unique_tracks"] == 3
    assert summary["missing_audio_files"] == 1
    assert summary["expected_archives"] is None
    assert summary["artist_leakage_counts"]["train_test"] == 1
    selected = json.loads((output / "selected_labels.json").read_text())
    assert selected["status"] == "frozen_for_baseline"
    assert [row["tag"] for row in selected["labels"]] == list(SELECTED_LABELS)


def test_audit_treats_removed_archives_as_cleanup_when_audio_is_complete(tmp_path):
    metadata = tmp_path / "metadata"
    split_root = metadata / "splits" / "split-0"
    split_root.mkdir(parents=True)
    rows = {
        "train": ["track_1", "artist_1", "01/1.mp3"],
        "validation": ["track_2", "artist_2", "02/2.mp3"],
        "test": ["track_3", "artist_3", "03/3.mp3"],
    }
    for split, (track, artist, path) in rows.items():
        _write_split(
            split_root / f"autotagging_moodtheme-{split}.tsv",
            [[track, artist, "album_1", path, "30", "mood/theme---calm"]],
        )

    download_metadata = metadata / "download"
    download_metadata.mkdir()
    (download_metadata / "autotagging_moodtheme_audio-low_sha256_tars.txt").write_text(
        "checksum removed-after-extraction.tar\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    for path in ("01/1.low.mp3", "02/2.low.mp3", "03/3.low.mp3"):
        destination = dataset / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x")

    output = tmp_path / "output"
    summary = run_audit(dataset, metadata, output)

    assert summary["missing_audio_files"] == 0
    assert summary["missing_archives"] == 0
    assert summary["source_archives_not_retained"] == 1
    assert (output / "missing_archives.csv").read_text() == "archive\n"
