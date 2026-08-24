#!/usr/bin/env python3
"""
Build text prompts for image generation based on audio mood.
"""


def build_prompt(
    mood: str | list[str] | tuple[str, ...],
    *,
    tempo_bpm: float | None = None,
    style: str = "digital art",
    subject: str = "an abstract landscape",
    abstraction: float = 0.7,
    music_influence: float = 0.8,
) -> str:
    """
    Return a Stable Diffusion prompt for a given mood.

    Args:
        mood (str): Predicted mood (e.g., 'calm', 'energetic').

    Returns:
        str: Text prompt.
    """
    if not 0 <= abstraction <= 1 or not 0 <= music_influence <= 1:
        raise ValueError("Abstraction and music influence must be between 0 and 1.")
    moods = [mood] if isinstance(mood, str) else list(mood)
    moods = [value.strip().lower() for value in moods if value.strip()]
    if not moods:
        moods = ["expressive"]
    mood_details = {
        "calm": "tranquil atmosphere, soft light, flowing composition",
        "energetic": "vibrant colors, dynamic composition, dramatic movement",
        "dark": "deep shadows, mysterious atmosphere, cinematic contrast",
        "happy": "warm light, joyful atmosphere, bright harmonious colors",
    }

    details = [
        mood_details.get(value, f"{value} atmosphere, expressive composition")
        for value in moods
    ]
    pace = ""
    if tempo_bpm is not None:
        if tempo_bpm < 80:
            pace = "slow visual rhythm, spacious composition"
        elif tempo_bpm < 125:
            pace = "balanced visual rhythm"
        else:
            pace = "fast visual rhythm, energetic motion"

    abstraction_text = (
        "abstract forms and symbolic imagery" if abstraction >= 0.67 else
        "a balance of recognizable and abstract forms" if abstraction >= 0.34 else
        "recognizable subjects and realistic structure"
    )
    influence_text = (
        "composition strongly shaped by the musical mood" if music_influence >= 0.67 else
        "composition gently informed by the musical mood" if music_influence >= 0.34 else
        "subtle musical influence"
    )
    parts = [
        subject, style, "; ".join(details), pace, abstraction_text,
        influence_text, "highly detailed",
    ]
    return ", ".join(part for part in parts if part)


# TODO: allow user to customize prompts (style, theme)
