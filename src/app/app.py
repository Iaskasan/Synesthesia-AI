"""Streamlit application for the SynesthesiaAI portfolio MVP."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st

# Streamlit executes this file with ``src/app`` on sys.path. Add the repository
# root so absolute ``src.*`` imports work without requiring an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.analysis import analyze_audio
from src.audio.load_audio import load_audio
from src.generation.generate_image import generate_image
from src.generation.prompt_builder import build_prompt
from src.ml.inference import ClapMoodClassifier

CHECKPOINT = PROJECT_ROOT / "artifacts/clap_diagnostics/selected_head.joblib"
IMAGE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ASPECT_RATIOS = {
    "Square (1:1)": (512, 512),
    "Landscape (4:3)": (640, 480),
    "Portrait (3:4)": (480, 640),
}


@st.cache_resource(show_spinner="Loading the CLAP mood model…")
def load_classifier(path: str) -> ClapMoodClassifier:
    return ClapMoodClassifier(path)


@st.cache_data(show_spinner="Analyzing the excerpt…")
def analyze_upload(data: bytes) -> tuple[np.ndarray, int, dict, list]:
    audio, sample_rate = load_audio(io.BytesIO(data))
    features = analyze_audio(audio, sample_rate)
    predictions = load_classifier(str(CHECKPOINT)).predict(audio, sample_rate)
    return audio, sample_rate, features.to_dict(), predictions


def _default_tags(predictions: list, count: int) -> list[str]:
    detected = [item.label for item in predictions if item.detected]
    ranked = detected or [item.label for item in predictions]
    return ranked[:count]


def _image_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _metadata(
    *, data: bytes, features: dict, predictions: list, tags: list[str],
    prompt: str, style: str, subject: str, abstraction: float,
    influence: float, aspect_ratio: str, seed: int,
) -> dict:
    classifier = load_classifier(str(CHECKPOINT))
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio_sha256": hashlib.sha256(data).hexdigest(),
        "excerpt_seconds": 30,
        "audio_features": features,
        "mood_predictions": [
            {
                "label": item.label, "confidence": item.confidence,
                "threshold": item.threshold, "detected": item.detected,
            }
            for item in predictions
        ],
        "selected_tags": tags,
        "visual_settings": {
            "style": style, "subject": subject, "abstraction": abstraction,
            "music_influence": influence, "aspect_ratio": aspect_ratio,
        },
        "prompt": prompt,
        "seed": seed,
        "mood_model": {
            "artifact": str(CHECKPOINT), "encoder": classifier.config.model_name,
            "experiment": classifier.experiment,
        },
        "image_model": IMAGE_MODEL,
    }


def main() -> None:
    st.set_page_config(page_title="SynesthesiaAI", page_icon="🎨", layout="wide")
    st.title("🎨 SynesthesiaAI")
    st.caption("Turn a 30-second musical excerpt into an editable visual interpretation.")
    uploaded = st.file_uploader(
        "Choose an audio file", type=["mp3", "wav", "flac", "ogg", "m4a"]
    )
    if uploaded is None:
        st.info("Upload a track to begin. Only the first 30 seconds are analyzed.")
        return

    data = uploaded.getvalue()
    st.audio(data)
    try:
        _, _, features, predictions = analyze_upload(data)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        st.error(str(error))
        st.stop()

    loudness_db = 20 * np.log10(max(float(features["rms_energy"]), 1e-12))
    metrics = st.columns(4)
    metrics[0].metric("Tempo", f"{features['tempo_bpm']:.1f} BPM")
    metrics[1].metric("Energy (RMS)", f"{features['rms_energy']:.3f}")
    metrics[2].metric("Loudness", f"{loudness_db:.1f} dBFS")
    metrics[3].metric("Brightness", f"{features['spectral_centroid_hz']:.0f} Hz")

    st.subheader("Mood interpretation")
    st.caption(
        "Confidence is the model score, not objective certainty. Decisions use "
        "thresholds tuned on validation data."
    )
    st.dataframe(
        [{
            "Mood": item.label, "Confidence": round(item.confidence, 3),
            "Threshold": round(item.threshold, 3),
            "Detected": "Yes" if item.detected else "No",
        } for item in predictions],
        hide_index=True, use_container_width=True,
    )

    controls, preview = st.columns([2, 3])
    with controls:
        tag_count = st.slider("Suggested tag count", 1, 5, 3)
        available = [item.label for item in predictions]
        selected_tags = st.multiselect(
            "Mood tags (review and edit)", available,
            default=_default_tags(predictions, tag_count),
        )
        emphasized = st.selectbox("Emphasize", selected_tags) if selected_tags else None
        ordered_tags = (
            [emphasized] + [tag for tag in selected_tags if tag != emphasized]
            if emphasized else []
        )
        style = st.selectbox(
            "Visual style",
            ["digital art", "watercolor", "cinematic photography", "surrealism"],
        )
        subject = st.text_input("Subject", "an abstract landscape")
        abstraction = st.slider("Abstraction", 0.0, 1.0, 0.7, 0.05)
        influence = st.slider("Music influence", 0.0, 1.0, 0.8, 0.05)
        aspect_ratio = st.selectbox("Aspect ratio", list(ASPECT_RATIOS))
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42)

    suggested_prompt = build_prompt(
        ordered_tags, tempo_bpm=float(features["tempo_bpm"]), style=style,
        subject=subject, abstraction=abstraction, music_influence=influence,
    )
    with preview:
        prompt_key = f"prompt_{hashlib.sha256(data).hexdigest()[:12]}"
        if prompt_key not in st.session_state:
            st.session_state[prompt_key] = suggested_prompt
        if st.button("Recompose prompt from controls"):
            st.session_state[prompt_key] = suggested_prompt
        prompt = st.text_area("Image prompt (editable)", height=180, key=prompt_key)
        if not ordered_tags:
            st.warning("Select at least one mood tag before generating an image.")
        if st.button("Generate artwork", type="primary", disabled=not ordered_tags):
            width, height = ASPECT_RATIOS[aspect_ratio]
            try:
                with st.spinner("Generating artwork…"):
                    image = generate_image(
                        prompt, IMAGE_MODEL, seed=int(seed), width=width, height=height
                    )
                metadata = _metadata(
                    data=data, features=features, predictions=predictions,
                    tags=ordered_tags, prompt=prompt, style=style, subject=subject,
                    abstraction=abstraction, influence=influence,
                    aspect_ratio=aspect_ratio, seed=int(seed),
                )
                st.session_state["result"] = {
                    "image": _image_bytes(image),
                    "metadata": json.dumps(metadata, indent=2) + "\n",
                }
            except (RuntimeError, ValueError, OSError) as error:
                st.error(f"Image generation failed: {error}")

        result = st.session_state.get("result")
        if result:
            st.image(result["image"], caption="Generated visual interpretation")
            downloads = st.columns(2)
            downloads[0].download_button(
                "Download image", result["image"], "synesthesia.png", "image/png"
            )
            downloads[1].download_button(
                "Download settings", result["metadata"],
                "synesthesia-settings.json", "application/json",
            )


if __name__ == "__main__":
    main()
