"""Interactive command-line entry point for the SVD image explorer."""

from pathlib import Path

from image_utils import load_grayscale_image, save_grayscale_image
from metrics import estimate_storage, relative_reconstruction_error
from svd_compression import compute_svd, reconstruct_from_svd, validate_k


def main() -> None:
    image_path_text = input("Enter the input image path: ").strip().strip('"')

    try:
        image_matrix = load_grayscale_image(image_path_text)
    except (OSError, ValueError) as error:
        print(f"Could not load the image: {error}")
        return

    height, width = image_matrix.shape
    decomposition = compute_svd(image_matrix)
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

        compressed_matrix = reconstruct_from_svd(decomposition, k)
        output_path = input_path.with_name(
            f"{input_path.stem}_compressed_k{k}.png"
        )

        try:
            save_grayscale_image(compressed_matrix, output_path)
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


if __name__ == "__main__":
    main()
