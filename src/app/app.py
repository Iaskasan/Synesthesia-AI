#!/usr/bin/env python3
"""
Streamlit app for Synesthesia AI.
"""

import streamlit as st
from src.audio.load_audio import load_audio
from src.audio.extract_features import extract_mfcc
from src.audio.analysis import analyze_audio
from src.audio.visualize import plot_waveform, plot_mfcc
from src.generation.prompt_builder import build_prompt
from src.generation.generate_image import generate_image


def main():
    st.title("🎨 Synesthesia AI")
    st.write("Upload a song and get AI-generated artwork based on its mood!")

    uploaded_file = st.file_uploader(
        "Choose an audio excerpt", type=["mp3", "wav", "flac", "ogg"]
    )
    if uploaded_file:
        try:
            y, sr = load_audio(uploaded_file)
        except (ValueError, RuntimeError) as error:
            st.error(f"Could not read this audio file: {error}")
            return
        st.write("Audio loaded successfully!")
        st.audio(uploaded_file)

        # Show waveform
        st.pyplot(plot_waveform(y, sr))

        # Extract features
        features = analyze_audio(y, sr)
        st.metric("Estimated tempo", f"{features.tempo_bpm:.1f} BPM")
        st.metric("Average energy", f"{features.rms_energy:.3f}")

        mfccs = extract_mfcc(y, sr)
        st.pyplot(plot_mfcc(mfccs))

        # TODO: mood classification (stub)
        mood = st.selectbox(
            "Mood (manual until the first classifier is trained)",
            ["calm", "energetic", "dark", "happy"],
        )
        style = st.selectbox(
            "Visual style",
            ["digital art", "watercolor", "cinematic photography", "surrealism"],
        )
        prompt = build_prompt(
            mood, tempo_bpm=features.tempo_bpm, style=style
        )
        st.text_area("Generated prompt", prompt)

        if st.button("Generate artwork", type="primary"):
            with st.spinner("Generating artwork…"):
                image = generate_image(prompt)
            st.image(image, caption=f"Artwork for mood: {mood}")


if __name__ == "__main__":
    main()
