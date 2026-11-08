# SVD Image Compression Explorer

A Python command-line tool that compresses images using **Singular Value Decomposition (SVD)** — one of the most important techniques in Linear Algebra and Matrix Analysis. The project is designed for learning: every module is documented, every design decision is explained, and the maths is kept visible.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Results](#example-results)
- [How to Interpret the Graphs](#how-to-interpret-the-graphs)
- [SVD Storage vs Actual File Size](#svd-storage-vs-actual-file-size)
- [Limitations](#limitations)
- [Running the Tests](#running-the-tests)

---

## Project Overview

Every digital image is a matrix of numbers (pixel intensities). SVD lets us factor that matrix into three components and then throw away the least important parts, keeping a good-looking approximation that requires fewer numbers to describe.

This tool lets you:

- **Compress** grayscale or RGB images at any rank *k* you choose.
- **Visualise** how reconstruction error and storage cost change as *k* varies.
- **Compare** the original and compressed images side by side.
- **Learn** the underlying Linear Algebra by reading the documented source code.

---

## Mathematical Foundation

### Singular Value Decomposition

Any real *m × n* matrix **A** can be factored as:

$$A = U \Sigma V^T$$

where:

| Factor | Shape | Property |
|--------|-------|----------|
| **U** | *m × m* (full) or *m × r* (reduced) | Columns are *left singular vectors* (orthonormal) |
| **Σ** | *r × r* diagonal | Diagonal entries σ₁ ≥ σ₂ ≥ … ≥ σᵣ ≥ 0 are the *singular values* |
| **Vᵀ** | *r × n* (reduced) or *n × n* (full) | Rows are *right singular vectors* (orthonormal) |

Here *r = min(m, n)* is the rank of the decomposition. The singular values tell you how much "energy" each component carries.

### Low-Rank Approximation

The **rank-k approximation** keeps only the *k* largest singular values and their corresponding vectors:

$$A_k = U_k \, \Sigma_k \, V_k^T$$

The **Eckart–Young theorem** guarantees that this is the *best* rank-k approximation in the Frobenius norm:

$$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}$$

In plain English: the error equals the energy in the singular values you discarded. For typical photographs, the singular values decay quickly, so a relatively small *k* captures most of the image.

### Why It Works Channel by Channel (RGB)

A colour image has three channels — Red, Green, Blue — each of which is an independent *m × n* intensity matrix. SVD is applied to each channel separately because:

1. Each channel is a valid real matrix on its own.
2. There is no mathematical coupling between channels that a joint decomposition could exploit for image-pixel data.
3. The three compressed channels are simply stacked back into an *m × n × 3* array.

The trade-off: an RGB image needs **three SVDs** (one per channel), so both computation time and storage are roughly tripled compared with a single grayscale image.

---

## Project Structure

```
LAMAproject_1/
├── main.py                  # CLI entry point (argparse + interactive mode)
├── image_utils.py           # Load/save images via Pillow → NumPy
├── svd_compression.py       # SVD computation and low-rank reconstruction
├── metrics.py               # Reconstruction error and storage estimation
├── visualization.py         # Matplotlib plots (error, storage, comparison)
├── requirements.txt         # External dependencies
├── README.md                # This file
├── assets/
│   └── sample_image.jpg     # Sample image for testing
└── tests/
    └── test_compression.py  # 43 automated tests (pytest)
```

### How the modules connect

```
main.py
  ├── image_utils.py      load / save images
  ├── svd_compression.py   compute SVD, reconstruct at rank k
  ├── metrics.py           measure error and estimate storage
  └── visualization.py     generate plots
        ├── svd_compression.py   (reuses SVD decompositions)
        └── metrics.py           (reuses error / storage functions)
```

| Module | Purpose | Linear Algebra used |
|--------|---------|---------------------|
| `svd_compression.py` | Factor a matrix with `np.linalg.svd`, then truncate U, Σ, Vᵀ to rank k | SVD, matrix multiplication, low-rank approximation |
| `metrics.py` | Compute ‖A − Aₖ‖_F / ‖A‖_F and count stored entries | Frobenius norm |
| `image_utils.py` | Convert between image files and float64 NumPy arrays | — (data pipeline) |
| `visualization.py` | Plot error curves, storage curves, and side-by-side comparisons | — (uses the above) |
| `main.py` | Parse CLI arguments or run the interactive loop | — (orchestration) |

---

## Installation

**Prerequisites:** Python 3.10 or later.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/svd-image-compression.git
cd svd-image-compression

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

The four dependencies are:

| Package | Why |
|---------|-----|
| `numpy` | Matrix operations and SVD |
| `pillow` | Reading and writing image files |
| `matplotlib` | Generating plots |
| `pytest` | Running the test suite |

---

## Usage

### Choose the output mode when prompted

```bash
python main.py assets/sample_image.jpg -k 50
```

When `--mode` is omitted, the program asks whether to save the result as grayscale or RGB color.

### Compress an RGB image

```bash
python main.py assets/sample_image.jpg -k 50 --mode rgb
```

To skip the prompt and choose grayscale directly, use `--mode grayscale`.

### Specify an output path

```bash
python main.py photo.png -k 100 --mode rgb -o compressed_photo.png
```

### Generate visualisation plots

```bash
python main.py assets/sample_image.jpg --visualise --k-values 5 10 20 50 100
```

This creates three plot files next to the input image:

- `sample_image_error_vs_k.png`
- `sample_image_storage_vs_k.png`
- `sample_image_comparison.png`

### Interactive mode (legacy)

```bash
python main.py
```

Running with no arguments asks for the image path and output mode, then lets you try different *k* values in a loop.

### Full help

```bash
python main.py -h
```

---

## Example Results

> **Note:** This section describes how to obtain results — it does not contain fabricated data. Run the commands below on your own images to see real numbers and plots.

### Compression

```bash
python main.py assets/sample_image.jpg -k 20
```

The program will print:

- **Input image dimensions** (height × width in pixels).
- **Chosen k** — the number of singular values retained.
- **Relative reconstruction error** — a number between 0 and 1 (lower is better).
- **Original matrix entries** — H × W (or 3 × H × W for RGB).
- **Estimated SVD representation entries** — k × (H + W + 1) per channel.
- **Estimated reduction** — percentage fewer numbers in the SVD representation.
- **Output path** of the saved compressed image.

### Plots

```bash
python main.py assets/sample_image.jpg --visualise --k-values 1 5 10 20 50 100
```

This generates three PNG files described in the next section.

---

## How to Interpret the Graphs

### 1. Relative Reconstruction Error vs k

This plot shows how the quality of the compressed image improves as you keep more singular values.

- **x-axis:** number of singular values kept (*k*).
- **y-axis:** relative Frobenius error ‖A − Aₖ‖_F / ‖A‖_F.
- The curve **always decreases** (or stays flat) as *k* grows.
- A **steep initial drop** means the image compresses well — most energy is in the first few singular values.
- The **"elbow"** of the curve is a practical sweet spot: below it, quality degrades noticeably; above it, each extra singular value yields diminishing returns.

### 2. Estimated SVD Representation Size vs k

This plot shows the storage trade-off.

- **x-axis:** *k*.
- **y-axis:** number of stored numerical values.
- The **blue line** rises linearly — the SVD representation grows with *k*.
- The **red dashed line** is the original image size (H × W entries, or 3 × H × W for RGB).
- Where the blue line **crosses** the red line is the break-even point: for larger *k* values the SVD representation is actually *bigger* than the original.

### 3. Side-by-Side Comparison

- The **leftmost panel** is the original image.
- Each subsequent panel shows the reconstruction at a given *k*.
- You can visually judge how much detail is preserved or lost at each rank.

---

## SVD Storage vs Actual File Size

The "estimated SVD representation entries" reported by this program count the **floating-point numbers** needed to store the truncated SVD factors:

$$\text{SVD entries} = k \times (H + W + 1)$$

per channel. This is the *mathematical* representation size.

The **actual output file size** (in bytes on disk) is different because:

1. The reconstructed image is saved as a PNG (or JPEG, etc.), which applies its **own compression** (lossless for PNG, lossy for JPEG).
2. The saved file stores 8-bit integers (0–255), not 64-bit floats.
3. Image format headers add a small fixed overhead.

Therefore the file on disk may be **smaller or larger** than the SVD entry count would suggest. The SVD entry count is useful for understanding the *mathematical* efficiency of the approximation, not for predicting the final file size.

---

## Limitations

1. **Not a practical image codec.** Real-world formats (JPEG, WebP, AVIF) use transforms tailored to human vision (DCT, wavelets) and are far more efficient than naive SVD truncation.

2. **Speed.** Computing the full SVD of a large matrix is O(m · n · min(m, n)). For very high-resolution images this can take several seconds.

3. **Memory.** The full U, Σ, Vᵀ factors are kept in memory. For a 4000 × 3000 grayscale image, U alone is 4000 × 3000 × 8 bytes ≈ 91 MB of float64 data.

4. **No perceptual quality metric.** The Frobenius norm treats every pixel equally. It does not account for human visual sensitivity (e.g., we notice changes in smooth regions more than in textures). Metrics like SSIM or LPIPS would be more perceptually meaningful.

5. **Channel independence.** The three RGB channels are compressed independently. A colour-space transform (e.g., to YCbCr) could concentrate energy better before applying SVD, but that is outside the current scope.

6. **No alpha channel support.** Transparent PNG images are converted to opaque grayscale or RGB before processing.

---

## Running the Tests

The project uses [pytest](https://docs.pytest.org/). There are 43 tests covering validation, SVD correctness, reconstruction error, storage estimation, image I/O, and visualisation smoke tests.

```bash
# Run all tests with verbose output
pytest tests/test_compression.py -v

# Run a specific test class
pytest tests/test_compression.py::TestEstimateStorage -v

# Run a single test
pytest tests/test_compression.py::TestReconstructionError::test_error_decreases_with_k -v
```

All tests use small synthetic matrices (no external image files required) and run in about 10–20 seconds.

