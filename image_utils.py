"""Helpers for loading and saving grayscale and RGB images.

This module wraps Pillow (PIL) to convert image files into NumPy arrays
suitable for numerical processing and to convert arrays back into images.

Key design decision
-------------------
Pixel values are stored as ``float64`` so that SVD computations (which
produce floating-point results) do not lose precision to integer
truncation.  Values are only clipped and rounded back to 0–255 at
save time.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(image_path: str | Path, mode: str = "L") -> np.ndarray:
    """Open an image and return its pixels as a floating-point matrix.

    Args:
        image_path: Path to the image file.
        mode: ``"L"`` for grayscale, ``"RGB"`` for colour.

    Returns:
        A NumPy array of shape ``(H, W)`` for grayscale or ``(H, W, 3)``
        for RGB, with ``dtype=float64``.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a regular file.
    """
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"The image path is not a file: {path}")

    with Image.open(path) as image:
        converted_image = image.convert(mode)
        # Returns (H, W) for "L" and (H, W, 3) for "RGB"
        return np.array(converted_image, dtype=np.float64)


def save_image(image_matrix: np.ndarray, output_path: str | Path) -> None:
    """Clip pixel values and save them as an 8-bit image.

    Values below 0 are set to 0 and values above 255 are set to 255.
    The array is then rounded to the nearest integer and cast to
    ``uint8`` before saving.

    Args:
        image_matrix: Array of shape ``(H, W)`` or ``(H, W, 3)``.
        output_path: Destination file path (format inferred from extension).
    """
    pixels = np.rint(np.clip(image_matrix, 0, 255)).astype(np.uint8)
    Image.fromarray(pixels).save(output_path)


# ── Convenience aliases ─────────────────────────────────────────────
# These thin wrappers keep the public API readable and let callers
# avoid remembering Pillow mode strings.


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """Load an image as a single-channel grayscale matrix.

    This is a convenience wrapper around :func:`load_image` with
    ``mode="L"``.
    """
    return load_image(image_path, mode="L")


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image as a three-channel RGB matrix.

    This is a convenience wrapper around :func:`load_image` with
    ``mode="RGB"``.
    """
    return load_image(image_path, mode="RGB")


def save_grayscale_image(
    image_matrix: np.ndarray, output_path: str | Path
) -> None:
    """Save a grayscale matrix as an image file.

    This is a convenience alias for :func:`save_image`; the underlying
    logic is identical because ``Image.fromarray`` already infers the
    colour mode from the array shape.
    """
    save_image(image_matrix, output_path)
