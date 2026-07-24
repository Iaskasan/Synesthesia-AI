#!/usr/bin/env python3
"""
Generate images using Stable Diffusion (via HuggingFace diffusers).
"""

from functools import lru_cache

import torch
from diffusers import StableDiffusionPipeline


@lru_cache(maxsize=2)
def _load_pipeline(model_id: str, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    return pipe.to(device)


def generate_image(
    prompt: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
):
    """
    Generate an image from a text prompt.

    Args:
        prompt (str): Text description for the image.
        model_id (str): HuggingFace model ID.

    Returns:
        PIL.Image: Generated image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = _load_pipeline(model_id, device)
    image = pipe(prompt).images[0]
    return image


# TODO: add support for multiple images, seeds, and styles
