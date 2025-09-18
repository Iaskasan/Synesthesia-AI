#!/usr/bin/env python3
"""
Build text prompts for image generation based on audio mood.
"""


def build_prompt(mood: str) -> str:
    """
    Return a Stable Diffusion prompt for a given mood.

    Args:
        mood (str): Predicted mood (e.g., 'calm', 'energetic').

    Returns:
        str: Text prompt.
    """
    prompts = {
        "calm": "dreamy watercolor landscape, soft pastel colors, tranquil",
        "energetic": "abstract neon cyberpunk city, vibrant, dynamic, sharp strokes",
        "dark": "misty gothic forest, monochrome, dramatic shadows",
        "happy": "cartoonish sunny meadow, bright cheerful colors",
    }

    return prompts.get(mood, "abstract painting, surreal, artistic")


# TODO: allow user to customize prompts (style, theme)
