#!/usr/bin/env python3
"""
Streamlit app for Synesthesia AI.
"""

import streamlit as st
from src.audio.load_audio import load_audio
from src.audio.extract_features import extract_tempo, extract_mfcc
from src.audio.visualize import plot_waveform, plot_mfcc
from src.generation.prompt_builder import build_prompt
from src.generation.generate_image import generate_image


def main():
    st.title("🎨 Synesthesia AI")
    st.write("Upload a song and get AI-generated artwork based on its mood!")

    uploaded_file = st.file_uploader("Choose a song", type=["mp3", "wav"])
    if uploaded_file:
        y, sr = load_audio(uploaded_file)
        st.write("Audio loaded successfully!")

        # Show waveform
        st.pyplot(plot_waveform(y, sr))

        # Extract features
        tempo = extract_tempo(y, sr)
        st.write(f"Estimated tempo: {tempo:.2f} BPM")

        mfccs = extract_mfcc(y, sr)
        st.pyplot(plot_mfcc(mfccs))

        # TODO: mood classification (stub)
        mood = "calm"  # placeholder
        prompt = build_prompt(mood)
        st.write(f"Generated prompt: {prompt}")

        # Generate image
        image = generate_image(prompt)
        st.image(image, caption=f"Artwork for mood: {mood}")


if __name__ == "__main__":
    main()
