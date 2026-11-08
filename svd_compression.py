"""Low-rank image reconstruction using singular value decomposition.

This module is the mathematical heart of the project.  It wraps
NumPy's ``linalg.svd`` and provides helpers that:

* Validate the chosen rank *k*.
* Compute the SVD once so it can be reused with many *k* values.
* Reconstruct an approximation from truncated SVD factors.
* Handle both single-channel (grayscale) and three-channel (RGB) images.

Linear Algebra connection
-------------------------
Every real m × n matrix A can be factored as  A = U Σ Vᵀ  where U and V
are orthogonal and Σ is diagonal with non-negative entries (the singular
values).  Keeping only the *k* largest singular values gives the best
rank-k approximation in the Frobenius norm (the Eckart–Young theorem).
"""

import numpy as np


# ── Validation ──────────────────────────────────────────────────────


def validate_k(k: int, image_shape: tuple[int, ...]) -> None:
    """Ensure *k* is a valid rank for the given image dimensions.

    Args:
        k: Desired number of singular values to retain.
        image_shape: ``(H, W)`` or ``(H, W, 3)``.

    Raises:
        ValueError: If *k* is not an integer or falls outside [1, min(H, W)].
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer.")

    # image_shape could be (H, W) or (H, W, 3)
    height, width = image_shape[:2]
    maximum_k = min(height, width)

    if k < 1 or k > maximum_k:
        raise ValueError(f"k must be between 1 and {maximum_k}.")


# ── Core SVD operations ────────────────────────────────────────────


def compute_svd(
    image_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the reduced SVD once for reuse with different *k* values.

    Uses ``full_matrices=False`` so that U is m × r and Vᵀ is r × n
    where r = min(m, n), which is faster and uses less memory than the
    full decomposition.

    Args:
        image_matrix: A 2-D array (single channel).

    Returns:
        ``(U, singular_values, Vt)`` — the three SVD factors.
    """
    return np.linalg.svd(image_matrix, full_matrices=False)


def reconstruct_from_svd(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray], k: int
) -> np.ndarray:
    """Return a rank-*k* approximation using precomputed SVD components.

    The reconstruction is  U[:, :k] · diag(σ[:k]) · Vᵀ[:k, :].
    Broadcasting ``u_k * singular_values_k`` is equivalent to
    multiplying by the diagonal matrix but avoids creating a full
    k × k diagonal.

    Args:
        decomposition: ``(U, singular_values, Vt)`` from :func:`compute_svd`.
        k: Number of singular values to keep.

    Returns:
        The approximated matrix with the same shape as the original.
    """
    u, singular_values, v_transpose = decomposition
    validate_k(k, (u.shape[0], v_transpose.shape[1]))

    u_k = u[:, :k]
    singular_values_k = singular_values[:k]
    v_transpose_k = v_transpose[:k, :]

    # Broadcasting scales each retained column of U by its singular value.
    return (u_k * singular_values_k) @ v_transpose_k


# ── RGB helpers ─────────────────────────────────────────────────────


def compute_svd_channels(
    image_matrix: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the SVD separately for each colour channel.

    Performing SVD independently on the red, green, and blue channels
    works because each channel is a standalone H × W matrix of
    intensity values.  The channels are mathematically independent,
    so there is no cross-channel SVD to exploit.

    This function lets you compute the (expensive) SVD once and then
    try many *k* values cheaply via :func:`reconstruct_rgb_from_svd`.

    Args:
        image_matrix: A 3-D array of shape ``(H, W, 3)``.

    Returns:
        A list of three ``(U, singular_values, Vt)`` tuples, one per
        channel (R, G, B).
    """
    if image_matrix.ndim != 3 or image_matrix.shape[2] != 3:
        raise ValueError("Expected an (H, W, 3) RGB image array.")
    return [compute_svd(image_matrix[:, :, c]) for c in range(3)]


def reconstruct_rgb_from_svd(
    channel_decompositions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    k: int,
) -> np.ndarray:
    """Reconstruct an RGB image from precomputed per-channel SVDs.

    Args:
        channel_decompositions: Three ``(U, S, Vt)`` tuples from
            :func:`compute_svd_channels`.
        k: Number of singular values to keep per channel.

    Returns:
        An ``(H, W, 3)`` array of the reconstructed colour image.
    """
    channels = [
        reconstruct_from_svd(decomposition, k)
        for decomposition in channel_decompositions
    ]
    return np.stack(channels, axis=2)


# ── One-shot convenience function ──────────────────────────────────


def compress_image(image_matrix: np.ndarray, k: int) -> np.ndarray:
    """Return a rank-*k* approximation of a grayscale or RGB image matrix.

    This is the simplest entry point: pass an image array and a *k*
    value and get the compressed result.  For exploring multiple *k*
    values, prefer :func:`compute_svd` / :func:`compute_svd_channels`
    to avoid recomputing the decomposition each time.

    Args:
        image_matrix: ``(H, W)`` grayscale or ``(H, W, 3)`` RGB array.
        k: Number of singular values to retain.

    Returns:
        The approximated image array with the same shape as the input.
    """
    validate_k(k, image_matrix.shape)

    if image_matrix.ndim == 2:
        # Grayscale case
        decomposition = compute_svd(image_matrix)
        return reconstruct_from_svd(decomposition, k)
    elif image_matrix.ndim == 3:
        # RGB case
        reconstructed_channels = []
        for i in range(3):
            channel = image_matrix[:, :, i]
            decomposition = compute_svd(channel)
            reconstructed_channels.append(reconstruct_from_svd(decomposition, k))
        return np.stack(reconstructed_channels, axis=2)
    else:
        raise ValueError("Image matrix must be 2D or 3D.")
