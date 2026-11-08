"""Matplotlib visualisations for the SVD Image Compression Explorer.

This module generates three kinds of plots:

1. **Error vs k** — shows how reconstruction quality improves as more
   singular values are kept.
2. **Storage vs k** — shows the trade-off between compression and the
   number of retained singular values.
3. **Side-by-side comparison** — displays the original image next to
   reconstructions at selected *k* values.

All plots are saved to disk (no interactive windows), making the module
safe to use in headless / CI environments.

Design note
-----------
``matplotlib.use("Agg")`` is called before importing ``pyplot`` so that
no GUI back-end is required.  This means ``plt.show()`` is never called;
every figure is saved with ``savefig`` and then closed.
"""

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from metrics import relative_reconstruction_error, estimate_storage
from svd_compression import (
    compute_svd,
    compute_svd_channels,
    compress_image,
    reconstruct_from_svd,
    reconstruct_rgb_from_svd,
)


def plot_error_vs_k(
    image_matrix: np.ndarray,
    k_values: list[int],
    output_path: str,
) -> None:
    """Plot relative reconstruction error against *k* and save the figure.

    The SVD is computed **once** and reused for every *k*, so this
    function is efficient even with many *k* values.

    **How to interpret the graph:**
    The curve starts high (large error when very few singular values are
    kept) and falls towards zero as *k* grows.  A steep initial drop
    means most of the image energy is concentrated in the first few
    singular values — the image compresses well.  The "elbow" of the
    curve is a good practical choice for *k*: beyond it, adding more
    singular values yields diminishing returns.

    Args:
        image_matrix: A 2-D (grayscale) or 3-D (RGB) image array.
        k_values: Sorted list of *k* values to evaluate.
        output_path: File path for the saved plot (e.g. ``"error.png"``).
    """
    errors = []

    if image_matrix.ndim == 2:
        decomposition = compute_svd(image_matrix)
        for k in k_values:
            reconstructed = reconstruct_from_svd(decomposition, k)
            errors.append(relative_reconstruction_error(image_matrix, reconstructed))
    elif image_matrix.ndim == 3:
        channel_decomps = compute_svd_channels(image_matrix)
        for k in k_values:
            reconstructed = reconstruct_rgb_from_svd(channel_decomps, k)
            errors.append(relative_reconstruction_error(image_matrix, reconstructed))
    else:
        raise ValueError("image_matrix must be 2-D or 3-D.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, errors, marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Number of singular values kept (k)")
    ax.set_ylabel("Relative reconstruction error")
    ax.set_title("Relative Reconstruction Error vs k")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Error-vs-k plot saved to {output_path}")


def plot_storage_vs_k(
    image_shape: tuple[int, ...],
    k_values: list[int],
    output_path: str,
) -> None:
    """Plot estimated SVD representation size against *k* and save.

    A horizontal line marks the number of entries in the original image
    so you can see when the SVD representation stops being smaller.

    **How to interpret the graph:**
    Below the horizontal line the SVD representation is more compact
    than the raw pixel data.  The intersection point is the break-even
    rank: choosing *k* above that value means the SVD factors actually
    use *more* numbers than the original image.

    Args:
        image_shape: ``(H, W)`` or ``(H, W, 3)``.
        k_values: Sorted list of *k* values to evaluate.
        output_path: Destination file path.
    """
    original_list = []
    svd_list = []

    for k in k_values:
        original_entries, svd_entries, _ = estimate_storage(image_shape, k)
        original_list.append(original_entries)
        svd_list.append(svd_entries)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        k_values, svd_list, marker="o", markersize=4, linewidth=1.5,
        label="SVD representation entries",
    )
    ax.axhline(
        y=original_list[0], color="r", linestyle="--", linewidth=1.2,
        label="Original image entries",
    )
    ax.set_xlabel("Number of singular values kept (k)")
    ax.set_ylabel("Number of stored entries")
    ax.set_title("Estimated SVD Representation Size vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Storage-vs-k plot saved to {output_path}")


def plot_comparison(
    original_matrix: np.ndarray,
    k_values: list[int],
    output_path: str,
) -> None:
    """Show the original image beside reconstructions at selected *k* values.

    **How to interpret the figure:**
    The leftmost panel is the original.  Each subsequent panel uses
    fewer singular values (smaller *k*), so fine detail and texture are
    progressively lost while large-scale structure is preserved.
    Comparing panels lets you judge visually how much quality a given
    *k* sacrifices.

    Args:
        original_matrix: ``(H, W)`` or ``(H, W, 3)`` image array.
        k_values: List of *k* values to display.
        output_path: Destination file path.
    """
    num_panels = 1 + len(k_values)
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))

    if num_panels == 1:
        axes = [axes]

    is_grayscale = original_matrix.ndim == 2

    def _display(ax, matrix, title):
        if is_grayscale:
            ax.imshow(
                np.clip(matrix, 0, 255).astype(np.uint8), cmap="gray",
                vmin=0, vmax=255,
            )
        else:
            ax.imshow(np.clip(matrix, 0, 255).astype(np.uint8))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    _display(axes[0], original_matrix, "Original")

    for idx, k in enumerate(k_values, start=1):
        reconstructed = compress_image(original_matrix, k)
        _display(axes[idx], reconstructed, f"k = {k}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to {output_path}")

