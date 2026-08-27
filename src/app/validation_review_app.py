"""Lightweight UI for reviewing the validation mood-label queue."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE = PROJECT_ROOT / "artifacts/clap_diagnostics/validation_review_queue.csv"
DEFAULT_AUDIO_ROOT = Path("/mnt/g/AI/datasets")
REQUIRED_COLUMNS = {
    "track_id", "audio_path", "label", "probability", "threshold",
    "predicted", "dataset_target", "dataset_tags", "verdict", "notes",
}
VERDICTS = ("correct", "incorrect", "ambiguous")


def load_queue(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load a review queue while retaining its original column order."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError("Queue is missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("The review queue is empty.")
    invalid = sorted({row["verdict"] for row in rows if row["verdict"] not in ("", *VERDICTS)})
    if invalid:
        raise ValueError("Queue contains invalid verdicts: " + ", ".join(invalid))
    return rows, fieldnames


def save_queue(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Atomically replace the CSV so an interrupted save cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", delete=False,
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def first_unreviewed(rows: list[dict[str, str]]) -> int:
    return next((index for index, row in enumerate(rows) if not row["verdict"]), 0)


def _go_to(index: int) -> None:
    st.session_state.review_index = index
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Validation music review", page_icon="🎧", layout="centered")
    st.title("🎧 Validation music review")

    with st.sidebar:
        st.header("Files")
        queue_text = st.text_input("Review CSV", str(DEFAULT_QUEUE))
        audio_root_text = st.text_input("Audio root", str(DEFAULT_AUDIO_ROOT))
        st.caption("Each choice is saved immediately to the CSV.")

    queue_path = Path(queue_text).expanduser().resolve()
    audio_root = Path(audio_root_text).expanduser()
    try:
        rows, fieldnames = load_queue(queue_path)
    except (OSError, ValueError) as error:
        st.error(f"Could not open the review queue: {error}")
        st.stop()

    queue_identity = str(queue_path)
    if st.session_state.get("queue_identity") != queue_identity:
        st.session_state.queue_identity = queue_identity
        st.session_state.review_index = first_unreviewed(rows)

    index = min(max(int(st.session_state.get("review_index", 0)), 0), len(rows) - 1)
    row = rows[index]
    reviewed_count = sum(bool(item["verdict"]) for item in rows)

    st.progress(reviewed_count / len(rows), text=f"{reviewed_count} of {len(rows)} reviewed")
    heading, position = st.columns([3, 1])
    heading.subheader(f"Does this sound **{row['label']}**?")
    position.write(f"Track {index + 1} / {len(rows)}")

    audio_path = audio_root / row["audio_path"]
    if audio_path.is_file():
        st.audio(str(audio_path), format="audio/mpeg")
        st.caption(f"{row['track_id']} · {audio_path.name}")
    else:
        st.error(f"Audio file not found: {audio_path}")

    predicted = row["predicted"].lower() == "true"
    st.info(
        f"The model says **{'YES' if predicted else 'NO'}** for “{row['label']}”. "
        "Choose whether that yes/no decision is correct."
    )

    current = row["verdict"] or "not reviewed"
    st.caption(f"Current verdict: **{current}**")
    choices = st.columns(3)
    selected: str | None = None
    if choices[0].button("✅ Correct", use_container_width=True, type="primary"):
        selected = "correct"
    if choices[1].button("❌ Incorrect", use_container_width=True):
        selected = "incorrect"
    if choices[2].button("❓ Ambiguous", use_container_width=True):
        selected = "ambiguous"

    notes_key = f"notes_{queue_identity}_{index}"
    if notes_key not in st.session_state:
        st.session_state[notes_key] = row["notes"]
    notes = st.text_area(
        "Notes (optional)", key=notes_key,
        placeholder="Why was this clear, incorrect, or ambiguous?",
    )

    if selected is not None:
        row["verdict"] = selected
        row["notes"] = notes
        save_queue(queue_path, rows, fieldnames)
        if index < len(rows) - 1:
            _go_to(index + 1)
        st.success("Saved. You have reached the final track.")

    navigation = st.columns([1, 1, 1])
    if navigation[0].button("← Previous", disabled=index == 0, use_container_width=True):
        row["notes"] = notes
        save_queue(queue_path, rows, fieldnames)
        _go_to(index - 1)
    if navigation[1].button("Save notes", use_container_width=True):
        row["notes"] = notes
        save_queue(queue_path, rows, fieldnames)
        st.toast("Notes saved")
    if navigation[2].button("Next →", disabled=index == len(rows) - 1, use_container_width=True):
        row["notes"] = notes
        save_queue(queue_path, rows, fieldnames)
        _go_to(index + 1)

    with st.expander("Details (optional — review by ear first)"):
        st.write({
            "probability": round(float(row["probability"]), 3),
            "threshold": round(float(row["threshold"]), 3),
            "dataset_target": row["dataset_target"],
            "dataset_tags": row["dataset_tags"],
            "audio_path": row["audio_path"],
        })


if __name__ == "__main__":
    main()
