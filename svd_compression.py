"""Low-rank image reconstruction using singular value decomposition."""

import numpy as np


def validate_k(k: int, image_shape: tuple[int, ...]) -> None:
    """Ensure k is an integer rank that this image can support."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer.")
    if len(image_shape) != 2:
        raise ValueError("The image must be a two-dimensional grayscale matrix.")

    maximum_k = min(image_shape)
    if k < 1 or k > maximum_k:
        raise ValueError(f"k must be between 1 and {maximum_k}.")


def compress_image(image_matrix: np.ndarray, k: int) -> np.ndarray:
    """Return a rank-k approximation of a grayscale image matrix."""
    decomposition = compute_svd(image_matrix)
    return reconstruct_from_svd(decomposition, k)


def compute_svd(
    image_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the reduced SVD once for reuse with different k values."""
    return np.linalg.svd(image_matrix, full_matrices=False)


def reconstruct_from_svd(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray], k: int
) -> np.ndarray:
    """Return a rank-k approximation using precomputed SVD components."""
    u, singular_values, v_transpose = decomposition
    validate_k(k, (u.shape[0], v_transpose.shape[1]))

    u_k = u[:, :k]
    singular_values_k = singular_values[:k]
    v_transpose_k = v_transpose[:k, :]

    # Broadcasting scales each retained column of U by its singular value.
    return (u_k * singular_values_k) @ v_transpose_k
  