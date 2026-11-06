"""Numerical quality and storage estimates for SVD reconstructions."""

import numpy as np


def relative_reconstruction_error(
    original_matrix: np.ndarray, reconstructed_matrix: np.ndarray
) -> float:
    """Return the Frobenius reconstruction error relative to the input magnitude."""
    original_norm = np.linalg.norm(original_matrix, ord="fro")
    difference_norm = np.linalg.norm(
        original_matrix - reconstructed_matrix, ord="fro"
    )

    if original_norm == 0:
        return 0.0 if difference_norm == 0 else float("inf")

    return float(difference_norm / original_norm)


def estimate_storage(
    image_shape: tuple[int, ...], k: int
) -> tuple[int, int, float]:
    """Return original entries, truncated-SVD entries, and percent reduction."""
    if len(image_shape) != 2:
        raise ValueError("The image must be a two-dimensional matrix.")

    height, width = image_shape
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive.")
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k < 1
        or k > min(height, width)
    ):
        raise ValueError(
            f"k must be an integer between 1 and {min(height, width)}."
        )

    original_entries = height * width
    svd_entries = k * (height + width + 1)
    percentage_reduction = (
        (original_entries - svd_entries) / original_entries
    ) * 100
    return original_entries, svd_entries, percentage_reduction
