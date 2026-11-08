"""Numerical quality and storage estimates for SVD reconstructions.

This module provides two concerns:

1. **Reconstruction quality** — measured by the Frobenius-norm relative
   error between the original and approximated matrices.
2. **Storage cost** — estimated by counting the floating-point entries
   that must be kept to represent the truncated SVD factors, compared
   with the entries in the original pixel matrix.

Both functions work for grayscale (2-D) and RGB (3-D) images.

Linear Algebra connection
-------------------------
The relative reconstruction error uses the *Frobenius norm*, which for a
matrix A is defined as ``||A||_F = sqrt(sum of all a_ij^2)``.  This is
the matrix analogue of the Euclidean vector norm and tells us how
"close" two matrices are element-by-element.
"""

import numpy as np


def relative_reconstruction_error(
    original_matrix: np.ndarray, reconstructed_matrix: np.ndarray
) -> float:
    """Return the Frobenius reconstruction error relative to the input magnitude.

    The formula is::

        ||A - A_k||_F / ||A||_F

    where *A* is the original matrix and *A_k* is the rank-k
    approximation.  A return value of 0.0 means a perfect
    reconstruction; values closer to 1.0 mean most information was lost.

    For 3-D (RGB) arrays the matrices are reshaped to 2-D before
    computing the Frobenius norm, because ``np.linalg.norm(..., ord="fro")``
    requires a 2-D input.
    """
    # Flatten to 2-D so the Frobenius norm works for both (H,W) and (H,W,3).
    orig_2d = original_matrix.reshape(original_matrix.shape[0], -1)
    recon_2d = reconstructed_matrix.reshape(reconstructed_matrix.shape[0], -1)

    original_norm = np.linalg.norm(orig_2d, ord="fro")
    difference_norm = np.linalg.norm(orig_2d - recon_2d, ord="fro")

    if original_norm == 0:
        return 0.0 if difference_norm == 0 else float("inf")

    return float(difference_norm / original_norm)


def estimate_storage(
    image_shape: tuple[int, ...], k: int
) -> tuple[int, int, float]:
    """Return original entries, truncated-SVD entries, and percentage reduction.

    Supports both 2-D shapes ``(H, W)`` and 3-D shapes ``(H, W, 3)``.

    Storage model
    ~~~~~~~~~~~~~
    For **one channel** of size H × W, the truncated SVD at rank *k*
    stores:

    * U_k  — H × k entries  (the first k columns of U)
    * σ_k  — k entries       (the k largest singular values)
    * V^T_k — k × W entries  (the first k rows of V^T)

    Total per channel: ``k * (H + W + 1)``

    For an **RGB image** this is repeated for each of the three channels,
    so the total is ``3 * k * (H + W + 1)``, while the original uses
    ``3 * H * W`` entries.

    Assumptions
    ~~~~~~~~~~~
    * Every stored number occupies the same amount of memory (one "entry").
      In practice all values are 64-bit floats during computation.
    * This is the *representation* size of the SVD factors, **not** the
      size of the final saved image file, which depends on the image
      format and its own compression algorithm.

    Raises:
        ValueError: If dimensions are non-positive, k is out of range,
            or the shape has an unexpected number of dimensions.
    """
    if len(image_shape) == 2:
        height, width = image_shape
        num_channels = 1
    elif len(image_shape) == 3:
        height, width, channels = image_shape
        if channels != 3:
            raise ValueError(
                f"Expected 3 colour channels, got {channels}."
            )
        num_channels = 3
    else:
        raise ValueError(
            "image_shape must be 2-D (H, W) or 3-D (H, W, 3)."
        )

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

    original_entries = height * width * num_channels
    svd_entries = num_channels * k * (height + width + 1)
    percentage_reduction = (
        (original_entries - svd_entries) / original_entries
    ) * 100
    return original_entries, svd_entries, percentage_reduction
