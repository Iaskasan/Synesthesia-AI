"""Inference helpers for the trained CLAP multilabel mood classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ml.extract_clap_embeddings import ClapEmbeddingConfig, fixed_crops


@dataclass(frozen=True)
class MoodPrediction:
    """One mood score and the validation-tuned decision made from it."""

    label: str
    confidence: float
    threshold: float
    detected: bool


def rank_predictions(
    labels: list[str], probabilities: np.ndarray, thresholds: np.ndarray
) -> list[MoodPrediction]:
    """Validate and rank one row of multilabel probabilities."""
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    cutoffs = np.asarray(thresholds, dtype=float).reshape(-1)
    if len(labels) != len(scores) or len(scores) != len(cutoffs):
        raise ValueError("Labels, probabilities, and thresholds must have equal length.")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(cutoffs)):
        raise ValueError("Prediction values must be finite.")
    predictions = [
        MoodPrediction(label, float(score), float(threshold), bool(score >= threshold))
        for label, score, threshold in zip(labels, scores, cutoffs)
    ]
    return sorted(predictions, key=lambda item: item.confidence, reverse=True)


class ClapMoodClassifier:
    """Load the frozen CLAP encoder and project an audio excerpt into mood scores."""

    def __init__(
        self,
        artifact_path: str | Path = "artifacts/clap_diagnostics/selected_head.joblib",
        *,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            import joblib
            import torch
            from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel, ClapProcessor
        except ImportError as error:
            raise RuntimeError(
                "Mood inference requires joblib, torch, and transformers. "
                "Install the project requirements first."
            ) from error

        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Classifier artifact not found: {path}. Train or copy the selected CLAP head first."
            )
        bundle = joblib.load(path)
        required = {"model", "labels", "thresholds", "clap_config"}
        missing = required.difference(bundle)
        if missing:
            raise ValueError(f"Classifier artifact is missing: {', '.join(sorted(missing))}")

        self.model = bundle["model"]
        self.labels = list(bundle["labels"])
        self.thresholds = np.asarray(bundle["thresholds"], dtype=float)
        self.experiment = str(bundle.get("experiment", "track_head"))
        self.crop_aggregation = bundle.get("crop_aggregation")
        self.config = ClapEmbeddingConfig(**bundle["clap_config"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        feature_extractor = ClapFeatureExtractor.from_pretrained(
            self.config.model_name, local_files_only=local_files_only
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, local_files_only=local_files_only
        )
        self.processor = ClapProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.encoder = ClapModel.from_pretrained(
            self.config.model_name, local_files_only=local_files_only
        ).to(self.device).eval()

    def predict(self, audio: np.ndarray, sample_rate: int) -> list[MoodPrediction]:
        """Predict ranked moods from a mono audio waveform."""
        import librosa

        values = np.asarray(audio, dtype=np.float32)
        if values.ndim != 1 or not values.size:
            raise ValueError("Audio must be a non-empty mono waveform.")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        if sample_rate != self.config.sample_rate:
            values = librosa.resample(
                values, orig_sr=sample_rate, target_sr=self.config.sample_rate
            )
        crops = fixed_crops(
            values,
            self.config.sample_rate,
            self.config.excerpt_seconds,
            self.config.crop_seconds,
        )
        inputs = self.processor(
            audio=list(crops),
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            encoded = self.encoder.get_audio_features(**inputs)
        if hasattr(encoded, "pooler_output"):
            encoded = encoded.pooler_output
        embeddings = encoded.detach().float().cpu().numpy()
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)

        if self.crop_aggregation:
            crop_scores = np.asarray(self.model.predict_proba(embeddings), dtype=float)
            if self.crop_aggregation == "mean":
                scores = crop_scores.mean(axis=0)
            elif self.crop_aggregation == "max":
                scores = crop_scores.max(axis=0)
            else:
                raise ValueError(f"Unsupported crop aggregation: {self.crop_aggregation}")
        else:
            pooled = embeddings.mean(axis=0)
            pooled /= np.linalg.norm(pooled).clip(min=1e-12)
            scores = np.asarray(self.model.predict_proba(pooled[None, :]))[0]
        return rank_predictions(self.labels, scores, self.thresholds)

