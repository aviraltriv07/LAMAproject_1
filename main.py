"""Command-line entry point for the SVD Image Compression Explorer.

This module uses Python's ``argparse`` library to parse command-line
arguments.  ``argparse`` lets you define expected flags (like ``--k``
and ``--mode``), automatically generates help text (``python main.py -h``),
and produces clear error messages when the user provides invalid input.

Usage examples
--------------
Compress a grayscale image with k = 50::

    python main.py assets/sample_image.jpg -k 50

Compress an RGB image and specify the output path::

    python main.py photo.png -k 100 --mode rgb --output compressed.png

Generate visualisation plots::

    python main.py photo.png --visualise --k-values 5 10 20 50 100

Interactive mode (legacy, no arguments)::

    python main.py
"""

import argparse
import sys
from pathlib import Path

from image_utils import load_image, save_image
from metrics import estimate_storage, relative_reconstruction_error
from svd_compression import (
    compress_image,
    compute_svd,
    compute_svd_channels,
    reconstruct_from_svd,
    reconstruct_rgb_from_svd,
    validate_k,
)


def build_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser.

    ``argparse.ArgumentParser`` works by registering expected arguments
    (positional or optional) with ``add_argument``.  When ``parse_args()``
    is called it reads ``sys.argv``, matches tokens to the registered
    arguments, converts types, and returns a ``Namespace`` object whose
    attributes hold the parsed values.
    """
    parser = argparse.ArgumentParser(
        description="SVD Image Compression Explorer — compress images "
        "using low-rank SVD approximation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py image.jpg -k 50\n"
            "  python main.py image.jpg -k 100 --mode rgb -o out.png\n"
            "  python main.py image.jpg --visualise --k-values 5 10 50\n"
            "  python main.py                         (interactive mode)\n"
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the input image.  Omit to enter interactive mode.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        help="Number of singular values to keep (required unless --visualise).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output image path.  Defaults to <input>_compressed_k<K>.png.",
    )
    parser.add_argument(
        "--mode",
        choices=["grayscale", "rgb"],
        default=None,
        help='Colour mode: "grayscale" or "rgb". If omitted, you will be prompted.',
    )
    parser.add_argument(
        "--visualise", "--visualize",
        action="store_true",
        default=False,
        help="Generate error, storage, and comparison plots instead of "
        "compressing a single image.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help="List of k values for --visualise (e.g. --k-values 5 10 50).",
    )
    return parser


# ── CLI-mode functions ──────────────────────────────────────────────


def run_compress(args: argparse.Namespace) -> None:
    """Compress a single image at the requested rank *k*."""
    if args.k is None:
        print("Error: -k is required when compressing an image.", file=sys.stderr)
        sys.exit(1)

    args.mode = choose_mode(args.mode)
    pil_mode = "RGB" if args.mode == "rgb" else "L"

    try:
        image_matrix = load_image(args.input, mode=pil_mode)
    except (OSError, ValueError) as error:
        print(f"Error: Could not load the image: {error}", file=sys.stderr)
        sys.exit(1)

    try:
        validate_k(args.k, image_matrix.shape)
    except ValueError as error:
        print(f"Error: Invalid k: {error}", file=sys.stderr)
        sys.exit(1)

    compressed = compress_image(image_matrix, args.k)

    input_path = Path(args.input).expanduser()
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}_compressed_k{args.k}.png"
        )

    try:
        save_image(compressed, output_path)
    except OSError as error:
        print(f"Error: Could not save the image: {error}", file=sys.stderr)
        sys.exit(1)

    # ── Report ──
    height, width = image_matrix.shape[:2]
    original_entries, svd_entries, percentage_reduction = estimate_storage(
        image_matrix.shape, args.k
    )
    error = relative_reconstruction_error(image_matrix, compressed)

    print(f"Input image dimensions: {height} x {width} pixels")
    print(f"Colour mode: {args.mode}")
    print(f"Chosen k: {args.k}")
    print(f"Relative reconstruction error: {error:.6f}")
    print(f"Original matrix entries: {original_entries}")
    print(f"Estimated SVD representation entries: {svd_entries}")
    print(f"Estimated reduction in numerical values: {percentage_reduction:.2f}%")
    if percentage_reduction < 0:
        print(
            "The SVD representation uses more numerical values "
            "than the original matrix."
        )
    print(f"Output image saved to: {output_path}")


def run_visualise(args: argparse.Namespace) -> None:
    """Generate error, storage, and comparison plots."""
    # Import here so matplotlib is only loaded when needed.
    from visualization import plot_comparison, plot_error_vs_k, plot_storage_vs_k

    args.mode = choose_mode(args.mode)
    pil_mode = "RGB" if args.mode == "rgb" else "L"

    try:
        image_matrix = load_image(args.input, mode=pil_mode)
    except (OSError, ValueError) as error:
        print(f"Error: Could not load the image: {error}", file=sys.stderr)
        sys.exit(1)

    height, width = image_matrix.shape[:2]
    max_k = min(height, width)

    if args.k_values:
        k_values = sorted(set(args.k_values))
    else:
        # Sensible default: ~10 evenly spaced k values up to half the max.
        step = max(1, max_k // 20)
        k_values = list(range(1, min(max_k, max_k // 2 + 1), step))
        if k_values[-1] != min(max_k, max_k // 2):
            k_values.append(min(max_k, max_k // 2))

    # Validate every k.
    for k in k_values:
        try:
            validate_k(k, image_matrix.shape)
        except ValueError as error:
            print(f"Error: Invalid k value {k}: {error}", file=sys.stderr)
            sys.exit(1)

    input_path = Path(args.input).expanduser()
    stem = input_path.stem

    plot_error_vs_k(
        image_matrix, k_values,
        str(input_path.with_name(f"{stem}_error_vs_k.png")),
    )
    plot_storage_vs_k(
        image_matrix.shape, k_values,
        str(input_path.with_name(f"{stem}_storage_vs_k.png")),
    )
    # For comparison, pick a manageable subset (at most 5 values).
    comparison_ks = k_values[:5] if len(k_values) > 5 else k_values
    plot_comparison(
        image_matrix, comparison_ks,
        str(input_path.with_name(f"{stem}_comparison.png")),
    )


# ── Interactive (legacy) mode ───────────────────────────────────────


def choose_mode(mode: str | None = None) -> str:
    """Return the selected image mode, prompting when it was not supplied."""
    if mode is not None:
        return mode

    while True:
        choice = input("Choose output mode: [1] Grayscale or [2] RGB color: ").strip().lower()
        if choice in {"1", "g", "gray", "grey", "grayscale"}:
            return "grayscale"
        if choice in {"2", "c", "color", "colour", "rgb"}:
            return "rgb"
        print("Please enter 1 for grayscale or 2 for RGB color.")


def run_interactive() -> None:
    """Run the original interactive loop for backward compatibility.

    This mode activates when the user runs ``python main.py`` with no
    arguments.  It mirrors the original Stage-1/Stage-2 behaviour.
    """
    image_path_text = input("Enter the input image path: ").strip().strip('"')

    mode = choose_mode()
    pil_mode = "RGB" if mode == "rgb" else "L"
    try:
        image_matrix = load_image(image_path_text, mode=pil_mode)
    except (OSError, ValueError) as error:
        print(f"Could not load the image: {error}")
        return

    height, width = image_matrix.shape[:2]
    decomposition = (
        compute_svd_channels(image_matrix) if mode == "rgb" else compute_svd(image_matrix)
    )
    input_path = Path(image_path_text).expanduser()

    while True:
        k_text = input(
            f"Enter k (an integer from 1 to {min(height, width)}): "
        )

        try:
            k = int(k_text)
            validate_k(k, image_matrix.shape)
        except ValueError as error:
            print(f"Invalid k: {error}")
            continue

        compressed_matrix = (
            reconstruct_rgb_from_svd(decomposition, k)
            if mode == "rgb"
            else reconstruct_from_svd(decomposition, k)
        )
        output_path = input_path.with_name(
            f"{input_path.stem}_compressed_k{k}.png"
        )

        try:
            save_image(compressed_matrix, output_path)
        except OSError as error:
            print(f"Could not save the compressed image: {error}")
            return

        original_entries, svd_entries, percentage_reduction = estimate_storage(
            image_matrix.shape, k
        )
        reconstruction_error = relative_reconstruction_error(
            image_matrix, compressed_matrix
        )

        print(f"Input image dimensions: {height} x {width} pixels")
        print(f"Colour mode: {mode}")
        print(f"Chosen k: {k}")
        print(f"Relative reconstruction error: {reconstruction_error:.6f}")
        print(f"Original matrix entries: {original_entries}")
        print(f"Estimated SVD representation entries: {svd_entries}")
        print(
            "Estimated reduction in numerical values: "
            f"{percentage_reduction:.2f}%"
        )
        if percentage_reduction < 0:
            print(
                "The SVD representation uses more numerical values "
                "than the original matrix."
            )
        print(f"Output image path: {output_path}")

        compare_again = input("Compare another k for this image? (y/n): ")
        if compare_again.strip().lower() != "y":
            break


# ── Entry point ─────────────────────────────────────────────────────


def main() -> None:
    """Dispatch to CLI or interactive mode based on command-line arguments."""
    parser = build_parser()

    # No arguments at all → interactive mode (preserving legacy behaviour).
    if len(sys.argv) == 1:
        run_interactive()
        return

    args = parser.parse_args()

    if args.input is None:
        parser.error("An input image path is required.")

    if args.visualise:
        run_visualise(args)
    else:
        run_compress(args)


if __name__ == "__main__":
    main()
