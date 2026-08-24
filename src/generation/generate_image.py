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
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    *,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
):
    """
    Generate an image from a text prompt.

    Args:
        prompt (str): Text description for the image.
        model_id (str): HuggingFace model ID.
        seed: Reproducible diffusion seed.
        width: Output width in pixels, divisible by eight.
        height: Output height in pixels, divisible by eight.

    Returns:
        PIL.Image: Generated image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = _load_pipeline(model_id, device)
    if width % 8 or height % 8:
        raise ValueError("Image width and height must be divisible by 8.")
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt, generator=generator, width=width, height=height
    ).images[0]
    return image


# TODO: add support for multiple images, seeds, and styles
