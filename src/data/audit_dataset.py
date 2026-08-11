"""Audit the MTG-Jamendo mood/theme dataset before model training."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPLIT_NAMES = ("train", "validation", "test")
SELECTED_LABELS = (
    "happy",
    "energetic",
    "relaxing",
    "emotional",
    "dark",
    "epic",
    "dream",
    "inspiring",
    "sad",
    "meditative",
    "uplifting",
    "motivational",
    "romantic",
    "fun",
    "calm",
    "adventure",
    "melancholic",
    "dramatic",
    "powerful",
    "hopeful",
)


@dataclass(frozen=True)
class Track:
    track_id: str
    artist_id: str
    path: str
    duration: float
    tags: tuple[str, ...]


def _clean_tag(tag: str) -> str:
    return tag.removeprefix("mood/theme---")


def read_tracks(path: Path) -> list[Track]:
    """Read an official MTG-Jamendo split TSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"TRACK_ID", "ARTIST_ID", "PATH", "DURATION", "TAGS"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} does not have the expected columns")

        tracks = []
        for row in reader:
            tags = tuple(
                _clean_tag(value)
                for key, value in row.items()
                if key == "TAGS" or key is None
                for value in ([value] if isinstance(value, str) else value or [])
                if value
            )
            tracks.append(
                Track(
                    track_id=row["TRACK_ID"],
                    artist_id=row["ARTIST_ID"],
                    path=row["PATH"],
                    duration=float(row["DURATION"]),
                    tags=tags,
                )
            )
        return tracks


def _audio_member(metadata_path: str) -> str:
    path = Path(metadata_path)
    return str(path.with_name(f"{path.stem}.low{path.suffix}"))


def inventory_audio(dataset_root: Path) -> set[str]:
    """Return relative audio paths found extracted or inside local tar files."""
    found = {
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("*.low.mp3")
    }
    for archive in sorted(dataset_root.glob("*.tar")):
        try:
            with tarfile.open(archive, mode="r:") as tar:
                found.update(
                    member.name.removeprefix("./")
                    for member in tar
                    if member.isfile() and member.name.endswith(".low.mp3")
                )
        except tarfile.TarError as error:
            raise ValueError(f"Unreadable archive {archive}: {error}") from error
    return found


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    dataset_root: Path,
    metadata_root: Path,
    output_dir: Path,
    split_index: int = 0,
) -> dict:
    """Run the audit and write machine-readable reports."""
    split_root = metadata_root / "splits" / f"split-{split_index}"
    split_files = {
        name: split_root / f"autotagging_moodtheme-{name}.tsv"
        for name in SPLIT_NAMES
    }
    missing_metadata = [str(path) for path in split_files.values() if not path.is_file()]
    if missing_metadata:
        raise FileNotFoundError(
            "Missing official split files: " + ", ".join(missing_metadata)
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    splits = {name: read_tracks(path) for name, path in split_files.items()}
    audio_paths = inventory_audio(dataset_root)

    frequency_rows = []
    all_tags = sorted({tag for tracks in splits.values() for row in tracks for tag in row.tags})
    for tag in all_tags:
        counts = {
            name: sum(tag in track.tags for track in tracks)
            for name, tracks in splits.items()
        }
        frequency_rows.append(
            {
                "tag": tag,
                **counts,
                "total": sum(counts.values()),
            }
        )
    frequency_rows.sort(key=lambda row: (-row["total"], row["tag"]))

    missing_rows = []
    for split_name, tracks in splits.items():
        for track in tracks:
            expected = _audio_member(track.path)
            if expected not in audio_paths:
                missing_rows.append(
                    {
                        "split": split_name,
                        "track_id": track.track_id,
                        "expected_path": expected,
                    }
                )

    artists = {
        name: {track.artist_id for track in tracks}
        for name, tracks in splits.items()
    }
    leakage = {
        "train_validation": sorted(artists["train"] & artists["validation"]),
        "train_test": sorted(artists["train"] & artists["test"]),
        "validation_test": sorted(artists["validation"] & artists["test"]),
    }
    archive_paths = sorted(dataset_root.glob("*.tar"))
    archive_manifest = (
        metadata_root
        / "download"
        / "autotagging_moodtheme_audio-low_sha256_tars.txt"
    )
    expected_archives: list[str] = []
    if archive_manifest.is_file():
        expected_archives = [
            line.split(maxsplit=1)[1].strip()
            for line in archive_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and len(line.split(maxsplit=1)) == 2
        ]
    present_archive_names = {path.name for path in archive_paths}
    absent_archives = [
        name for name in expected_archives if name not in present_archive_names
    ]
    # With download.py --unpack --remove, source archives are intentionally
    # deleted after their tracks pass checksum validation. If every expected
    # audio file is available, absent archives are therefore not missing data.
    extraction_complete = not missing_rows
    missing_archives = [] if extraction_complete else absent_archives
    source_archives_not_retained = absent_archives if extraction_complete else []
    summary = {
        "split_index": split_index,
        "tracks": {name: len(tracks) for name, tracks in splits.items()},
        "unique_tracks": len(
            {track.track_id for tracks in splits.values() for track in tracks}
        ),
        "unique_labels": len(all_tags),
        "audio_files_in_inventory": len(audio_paths),
        "missing_audio_files": len(missing_rows),
        "archives": len(archive_paths),
        "expected_archives": len(expected_archives) or None,
        "missing_archives": len(missing_archives),
        "source_archives_not_retained": len(source_archives_not_retained),
        "archive_bytes": sum(path.stat().st_size for path in archive_paths),
        "artist_leakage_counts": {
            pair: len(values) for pair, values in leakage.items()
        },
    }
    frequencies_by_tag = {row["tag"]: row for row in frequency_rows}
    label_proposal = {
        "status": "frozen_for_baseline",
        "selection_rule": (
            "Visually meaningful mood/theme labels with support across the official "
            "splits and broad emotional coverage. Semantically related labels remain "
            "separate for the baseline and will be reviewed using validation metrics."
        ),
        "labels": [
            {
                "tag": tag,
                "train": frequencies_by_tag.get(tag, {}).get("train", 0),
                "validation": frequencies_by_tag.get(tag, {}).get("validation", 0),
                "test": frequencies_by_tag.get(tag, {}).get("test", 0),
                "total": frequencies_by_tag.get(tag, {}).get("total", 0),
            }
            for tag in SELECTED_LABELS
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "selected_labels.json").write_text(
        json.dumps(label_proposal, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "artist_leakage.json").write_text(
        json.dumps(leakage, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "label_frequencies.csv",
        ("tag", "train", "validation", "test", "total"),
        frequency_rows,
    )
    _write_csv(
        output_dir / "missing_audio.csv",
        ("split", "track_id", "expected_path"),
        missing_rows,
    )
    _write_csv(
        output_dir / "missing_archives.csv",
        ("archive",),
        ({"archive": name} for name in missing_archives),
    )
    _write_csv(
        output_dir / "source_archives_not_retained.csv",
        ("archive",),
        ({"archive": name} for name in source_archives_not_retained),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit MTG-Jamendo mood/theme metadata and local audio."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_audit(
        dataset_root=args.dataset_root,
        metadata_root=args.metadata_root,
        output_dir=args.output_dir,
        split_index=args.split_index,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
