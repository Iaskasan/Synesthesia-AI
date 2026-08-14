"""Extract deterministic frozen CLAP embeddings from a preprocessing manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import librosa
import numpy as np


DEFAULT_MODEL = "laion/larger_clap_music"


@dataclass(frozen=True)
class ClapEmbeddingConfig:
    model_name: str = DEFAULT_MODEL
    sample_rate: int = 48_000
    excerpt_seconds: float = 30.0
    crop_seconds: float = 10.0


def fixed_crops(
    audio: np.ndarray,
    sample_rate: int = 48_000,
    excerpt_seconds: float = 30.0,
    crop_seconds: float = 10.0,
) -> np.ndarray:
    """Return contiguous, fixed-length crops covering one padded excerpt."""
    if sample_rate <= 0 or excerpt_seconds <= 0 or crop_seconds <= 0:
        raise ValueError("Sample rate and durations must be positive.")
    crop_samples = int(round(sample_rate * crop_seconds))
    excerpt_samples = int(round(sample_rate * excerpt_seconds))
    if excerpt_samples % crop_samples:
        raise ValueError("Excerpt duration must be divisible by crop duration.")
    padded = librosa.util.fix_length(
        np.asarray(audio, dtype=np.float32), size=excerpt_samples
    )
    return padded.reshape(excerpt_samples // crop_samples, crop_samples)


def pool_crop_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool crop embeddings and return one unit-length float32 vector."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("Crop embeddings must be a non-empty 2D array.")
    pooled = values.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Pooled CLAP embedding has invalid norm.")
    return np.asarray(pooled / norm, dtype=np.float32)


def embedding_path(output_root: Path, row: dict[str, str]) -> Path:
    parent = Path(row["feature_path"]).parent
    if parent.parts and parent.parts[0] == "features":
        parent = Path(*parent.parts[1:])
    return output_root / "embeddings" / parent / f"{row['track_id']}.npz"


def _write_embedding(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, embedding=embedding)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "track_id", "audio_path", "feature_path", "tags"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest does not contain the required columns: {path}")
    return rows


def _batches(values: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def extract_embeddings(
    manifest_path: Path,
    dataset_root: Path,
    output_root: Path,
    config: ClapEmbeddingConfig = ClapEmbeddingConfig(),
    batch_size: int = 4,
    overwrite: bool = False,
    max_tracks: int | None = None,
    device: str | None = None,
    encoder: Callable[[list[np.ndarray]], np.ndarray] | None = None,
) -> dict:
    """Extract embeddings, preserving manifest metadata and resumability."""
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    rows = read_manifest(manifest_path)
    if max_tracks is not None:
        rows = rows[:max_tracks]
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "embedding_config.json"
    serialized = asdict(config)
    if config_path.is_file() and json.loads(config_path.read_text()) != serialized:
        raise ValueError(
            f"Existing embedding configuration differs: {config_path}. "
            "Choose another --output-root or remove the old cache intentionally."
        )
    config_path.write_text(json.dumps(serialized, indent=2) + "\n", encoding="utf-8")

    if encoder is None:
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as error:
            raise RuntimeError(
                "CLAP extraction requires torch and transformers. Install requirements.txt "
                "inside the project environment."
            ) from error
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = ClapProcessor.from_pretrained(config.model_name)
        model = ClapModel.from_pretrained(config.model_name).to(selected_device).eval()

        def encoder(crops: list[np.ndarray]) -> np.ndarray:
            inputs = processor(
                audio=crops,
                sampling_rate=config.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(selected_device) for key, value in inputs.items()}
            with torch.inference_mode():
                result = model.get_audio_features(**inputs)
            # Transformers 4.x returns a tensor here; 5.x returns a
            # BaseModelOutputWithPooling containing the projected tensor.
            if hasattr(result, "pooler_output"):
                result = result.pooler_output
            return result.detach().float().cpu().numpy()

    pending = [
        row for row in rows
        if overwrite or not embedding_path(output_root, row).is_file()
    ]
    errors: list[dict[str, str]] = []
    processed = 0
    for batch_index, batch in enumerate(_batches(pending, batch_size), start=1):
        loaded: list[tuple[dict[str, str], np.ndarray]] = []
        for row in batch:
            source = dataset_root / row["audio_path"]
            try:
                audio, _ = librosa.load(
                    source, sr=config.sample_rate, mono=True,
                    duration=config.excerpt_seconds,
                )
                if not audio.size:
                    raise ValueError("decoded audio is empty")
                loaded.append((row, fixed_crops(
                    audio, config.sample_rate, config.excerpt_seconds,
                    config.crop_seconds,
                )))
            except (FileNotFoundError, OSError, ValueError) as error:
                errors.append({"track_id": row["track_id"], "error": str(error)})
        if loaded:
            crops_per_track = loaded[0][1].shape[0]
            flat_crops = [crop for _, crops in loaded for crop in crops]
            try:
                encoded = np.asarray(encoder(flat_crops), dtype=np.float32)
                expected = len(loaded) * crops_per_track
                if encoded.ndim != 2 or len(encoded) != expected:
                    raise ValueError(
                        f"Encoder returned {encoded.shape}; expected ({expected}, embedding_dim)."
                    )
                for index, (row, _) in enumerate(loaded):
                    start = index * crops_per_track
                    pooled = pool_crop_embeddings(encoded[start : start + crops_per_track])
                    _write_embedding(embedding_path(output_root, row), pooled)
                    processed += 1
            except (OSError, RuntimeError, ValueError) as error:
                for row, _ in loaded:
                    errors.append({"track_id": row["track_id"], "error": str(error)})
        done = min(batch_index * batch_size, len(pending))
        if done % 100 < batch_size or done == len(pending):
            print(f"processed {done}/{len(pending)} pending tracks ({len(errors)} errors)", flush=True)

    output_rows = []
    failed = {row["track_id"] for row in errors}
    for row in rows:
        destination = embedding_path(output_root, row)
        if destination.is_file() and row["track_id"] not in failed:
            output_rows.append({
                **row,
                "embedding_path": destination.relative_to(output_root).as_posix(),
            })
    fields = list(rows[0]) + ["embedding_path"]
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    with (output_root / "errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("track_id", "error"))
        writer.writeheader()
        writer.writerows(errors)
    summary = {
        "tracks_requested": len(rows),
        "tracks_available": len(output_rows),
        "processed": processed,
        "already_cached": len(rows) - len(pending),
        "errors": len(errors),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tracks", type=int)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = extract_embeddings(
        args.manifest, args.dataset_root, args.output_root,
        ClapEmbeddingConfig(model_name=args.model_name), args.batch_size,
        args.overwrite, args.max_tracks, args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
