#!/usr/bin/env python3
"""
Build text prompts for image generation based on audio mood.
"""


def build_prompt(
    mood: str,
    *,
    tempo_bpm: float | None = None,
    style: str = "digital art",
    subject: str = "an abstract landscape",
) -> str:
    """
    Return a Stable Diffusion prompt for a given mood.

    Args:
        mood (str): Predicted mood (e.g., 'calm', 'energetic').

    Returns:
        str: Text prompt.
    """
    mood_details = {
        "calm": "tranquil atmosphere, soft light, flowing composition",
        "energetic": "vibrant colors, dynamic composition, dramatic movement",
        "dark": "deep shadows, mysterious atmosphere, cinematic contrast",
        "happy": "warm light, joyful atmosphere, bright harmonious colors",
    }

    details = mood_details.get(
        mood.lower(), f"{mood.lower()} atmosphere, expressive composition"
    )
    pace = ""
    if tempo_bpm is not None:
        if tempo_bpm < 80:
            pace = "slow visual rhythm, spacious composition"
        elif tempo_bpm < 125:
            pace = "balanced visual rhythm"
        else:
            pace = "fast visual rhythm, energetic motion"

    parts = [subject, style, details, pace, "highly detailed"]
    return ", ".join(part for part in parts if part)


# TODO: allow user to customize prompts (style, theme)
