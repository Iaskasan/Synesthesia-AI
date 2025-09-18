#!/usr/bin/env python3
"""
Generate images using Stable Diffusion (via HuggingFace diffusers).
"""

from diffusers import StableDiffusionPipeline
import torch


def generate_image(prompt: str, model_id="runwayml/stable-diffusion-v1-5"):
    """
    Generate an image from a text prompt.

    Args:
        prompt (str): Text description for the image.
        model_id (str): HuggingFace model ID.

    Returns:
        PIL.Image: Generated image.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image = pipe(prompt).images[0]
    return image


# TODO: add support for multiple images, seeds, and styles
