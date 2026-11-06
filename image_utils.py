"""Helpers for loading and saving grayscale images."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """Open an image and return its grayscale pixels as a floating-point matrix."""
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"The image path is not a file: {path}")

    with Image.open(path) as image:
        grayscale_image = image.convert("L")
        return np.array(grayscale_image, dtype=np.float64)


def save_grayscale_image(image_matrix: np.ndarray, output_path: str | Path) -> None:
    """Clip pixel values to grayscale range and save them as an 8-bit image."""
    pixels = np.rint(np.clip(image_matrix, 0, 255)).astype(np.uint8)
    Image.fromarray(pixels).save(output_path)
