"""Tests for the SVD Image Compression Explorer.

Run with::

    pytest tests/test_compression.py -v

These tests use ``pytest`` — a popular Python testing framework that
discovers functions whose names start with ``test_``, runs them, and
reports which passed or failed.  No boilerplate test-class is needed
(though you can use classes if you prefer grouping).

We create small synthetic images (NumPy arrays) rather than relying on
external files, so the tests are self-contained and fast.
"""

import os
import tempfile

import numpy as np
import pytest

# ── Imports from the project under test ─────────────────────────────
from image_utils import load_image, save_image, load_grayscale_image, load_rgb_image
from metrics import estimate_storage, relative_reconstruction_error
from svd_compression import (
    compress_image,
    compute_svd,
    compute_svd_channels,
    reconstruct_from_svd,
    reconstruct_rgb_from_svd,
    validate_k,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _random_grayscale(height: int = 40, width: int = 30, seed: int = 42):
    """Return a deterministic random grayscale image matrix."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 255, size=(height, width))


def _random_rgb(height: int = 40, width: int = 30, seed: int = 42):
    """Return a deterministic random RGB image matrix."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 255, size=(height, width, 3))


# ═══════════════════════════════════════════════════════════════════
# 1. validate_k
# ═══════════════════════════════════════════════════════════════════

class TestValidateK:
    """Tests for svd_compression.validate_k."""

    def test_valid_k_minimum(self):
        validate_k(1, (10, 20))

    def test_valid_k_maximum(self):
        validate_k(10, (10, 20))  # min(10,20) = 10

    def test_valid_k_rgb_shape(self):
        validate_k(5, (10, 20, 3))

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="between 1"):
            validate_k(0, (10, 20))

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="between 1"):
            validate_k(-1, (10, 20))

    def test_k_too_large_raises(self):
        with pytest.raises(ValueError, match="between 1"):
            validate_k(11, (10, 20))

    def test_k_float_raises(self):
        with pytest.raises(ValueError, match="integer"):
            validate_k(5.0, (10, 20))

    def test_k_bool_raises(self):
        with pytest.raises(ValueError, match="integer"):
            validate_k(True, (10, 20))

    def test_k_string_raises(self):
        with pytest.raises(ValueError, match="integer"):
            validate_k("5", (10, 20))


# ═══════════════════════════════════════════════════════════════════
# 2. SVD computation and reconstruction
# ═══════════════════════════════════════════════════════════════════

class TestSVDComputation:
    """Tests for compute_svd and reconstruct_from_svd."""

    def test_svd_returns_three_arrays(self):
        matrix = _random_grayscale(20, 15)
        u, s, vt = compute_svd(matrix)
        assert u.shape == (20, 15)
        assert s.shape == (15,)
        assert vt.shape == (15, 15)

    def test_full_rank_reconstruction_is_exact(self):
        matrix = _random_grayscale(20, 15)
        decomp = compute_svd(matrix)
        reconstructed = reconstruct_from_svd(decomp, k=15)
        np.testing.assert_allclose(reconstructed, matrix, atol=1e-10)

    def test_reconstruction_dimensions_match(self):
        matrix = _random_grayscale(30, 25)
        decomp = compute_svd(matrix)
        for k in [1, 5, 10, 25]:
            result = reconstruct_from_svd(decomp, k)
            assert result.shape == (30, 25)


class TestCompressImage:
    """Tests for compress_image (the one-shot API)."""

    def test_grayscale_output_shape(self):
        matrix = _random_grayscale(20, 15)
        result = compress_image(matrix, k=5)
        assert result.shape == (20, 15)

    def test_rgb_output_shape(self):
        matrix = _random_rgb(20, 15)
        result = compress_image(matrix, k=5)
        assert result.shape == (20, 15, 3)

    def test_grayscale_full_rank_exact(self):
        matrix = _random_grayscale(20, 15)
        result = compress_image(matrix, k=15)
        np.testing.assert_allclose(result, matrix, atol=1e-10)

    def test_rgb_full_rank_exact(self):
        matrix = _random_rgb(20, 15)
        result = compress_image(matrix, k=15)
        np.testing.assert_allclose(result, matrix, atol=1e-10)

    def test_invalid_k_raises(self):
        matrix = _random_grayscale(20, 15)
        with pytest.raises(ValueError):
            compress_image(matrix, k=0)
        with pytest.raises(ValueError):
            compress_image(matrix, k=16)


# ═══════════════════════════════════════════════════════════════════
# 3. RGB channel helpers
# ═══════════════════════════════════════════════════════════════════

class TestRGBHelpers:
    """Tests for compute_svd_channels and reconstruct_rgb_from_svd."""

    def test_compute_svd_channels_returns_three(self):
        matrix = _random_rgb(20, 15)
        decomps = compute_svd_channels(matrix)
        assert len(decomps) == 3

    def test_reconstruct_rgb_shape(self):
        matrix = _random_rgb(20, 15)
        decomps = compute_svd_channels(matrix)
        result = reconstruct_rgb_from_svd(decomps, k=5)
        assert result.shape == (20, 15, 3)

    def test_reconstruct_rgb_full_rank_exact(self):
        matrix = _random_rgb(20, 15)
        decomps = compute_svd_channels(matrix)
        result = reconstruct_rgb_from_svd(decomps, k=15)
        np.testing.assert_allclose(result, matrix, atol=1e-10)

    def test_rejects_grayscale_input(self):
        matrix = _random_grayscale(20, 15)
        with pytest.raises(ValueError, match="RGB"):
            compute_svd_channels(matrix)


# ═══════════════════════════════════════════════════════════════════
# 4. Reconstruction error
# ═══════════════════════════════════════════════════════════════════

class TestReconstructionError:
    """Tests for metrics.relative_reconstruction_error."""

    def test_identical_matrices_give_zero(self):
        matrix = _random_grayscale()
        assert relative_reconstruction_error(matrix, matrix) == 0.0

    def test_full_rank_error_near_zero(self):
        matrix = _random_grayscale(20, 15)
        reconstructed = compress_image(matrix, k=15)
        error = relative_reconstruction_error(matrix, reconstructed)
        assert error < 1e-10

    def test_low_rank_error_positive(self):
        matrix = _random_grayscale(20, 15)
        reconstructed = compress_image(matrix, k=1)
        error = relative_reconstruction_error(matrix, reconstructed)
        assert 0 < error < 1

    def test_zero_matrix(self):
        zeros = np.zeros((10, 10))
        assert relative_reconstruction_error(zeros, zeros) == 0.0

    def test_zero_original_nonzero_reconstructed(self):
        zeros = np.zeros((10, 10))
        nonzero = np.ones((10, 10))
        assert relative_reconstruction_error(zeros, nonzero) == float("inf")

    def test_error_decreases_with_k(self):
        matrix = _random_grayscale(30, 20)
        decomp = compute_svd(matrix)
        errors = []
        for k in [1, 5, 10, 20]:
            rec = reconstruct_from_svd(decomp, k)
            errors.append(relative_reconstruction_error(matrix, rec))
        # Error should be non-increasing as k grows.
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-12


# ═══════════════════════════════════════════════════════════════════
# 5. Storage estimation
# ═══════════════════════════════════════════════════════════════════

class TestEstimateStorage:
    """Tests for metrics.estimate_storage."""

    def test_2d_known_values(self):
        # 100x80, k=10 → original=8000, svd=10*(100+80+1)=1810
        original, svd, reduction = estimate_storage((100, 80), 10)
        assert original == 8000
        assert svd == 1810
        expected_reduction = (8000 - 1810) / 8000 * 100
        assert abs(reduction - expected_reduction) < 1e-10

    def test_3d_known_values(self):
        # 100x80x3, k=10 → original=24000, svd=3*1810=5430
        original, svd, reduction = estimate_storage((100, 80, 3), 10)
        assert original == 24000
        assert svd == 5430
        expected_reduction = (24000 - 5430) / 24000 * 100
        assert abs(reduction - expected_reduction) < 1e-10

    def test_negative_reduction_when_k_large(self):
        # 10x10, k=10 → svd=10*(10+10+1)=210 > 100 → negative reduction
        _, svd, reduction = estimate_storage((10, 10), 10)
        assert svd > 100
        assert reduction < 0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            estimate_storage((100, 80), 0)
        with pytest.raises(ValueError):
            estimate_storage((100, 80), 81)

    def test_bool_k_raises(self):
        with pytest.raises(ValueError):
            estimate_storage((100, 80), True)

    def test_wrong_dimensions_raises(self):
        with pytest.raises(ValueError):
            estimate_storage((100,), 5)
        with pytest.raises(ValueError):
            estimate_storage((100, 80, 4), 5)  # 4 channels not valid

    def test_non_positive_dimensions_raises(self):
        with pytest.raises(ValueError):
            estimate_storage((0, 80), 1)


# ═══════════════════════════════════════════════════════════════════
# 6. Image loading and saving
# ═══════════════════════════════════════════════════════════════════

class TestImageIO:
    """Tests for image_utils load/save functions."""

    def test_save_and_load_grayscale_roundtrip(self):
        matrix = np.random.default_rng(0).uniform(0, 255, (50, 40))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_image(matrix, path)
            loaded = load_image(path, mode="L")
            assert loaded.shape == (50, 40)
            # After clipping/rounding, values should be close to uint8.
            np.testing.assert_allclose(loaded, np.rint(matrix), atol=1)
        finally:
            os.remove(path)

    def test_save_and_load_rgb_roundtrip(self):
        matrix = np.random.default_rng(1).uniform(0, 255, (50, 40, 3))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_image(matrix, path)
            loaded = load_image(path, mode="RGB")
            assert loaded.shape == (50, 40, 3)
        finally:
            os.remove(path)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_file_12345.png")

    def test_load_grayscale_alias(self):
        """load_grayscale_image should behave identically to load_image(…, 'L')."""
        matrix = np.random.default_rng(2).uniform(0, 255, (30, 20))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_image(matrix, path)
            a = load_image(path, mode="L")
            b = load_grayscale_image(path)
            np.testing.assert_array_equal(a, b)
        finally:
            os.remove(path)

    def test_load_rgb_alias(self):
        """load_rgb_image should behave identically to load_image(…, 'RGB')."""
        matrix = np.random.default_rng(3).uniform(0, 255, (30, 20, 3))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_image(matrix, path)
            a = load_image(path, mode="RGB")
            b = load_rgb_image(path)
            np.testing.assert_array_equal(a, b)
        finally:
            os.remove(path)


# ═══════════════════════════════════════════════════════════════════
# 7. Visualisation (smoke test)
# ═══════════════════════════════════════════════════════════════════

class TestVisualization:
    """Smoke tests — verify that plots are created without errors."""

    def test_plot_error_vs_k_creates_file(self):
        from visualization import plot_error_vs_k
        matrix = _random_grayscale(30, 20)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_error_vs_k(matrix, [1, 5, 10, 20], path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        finally:
            os.remove(path)

    def test_plot_storage_vs_k_creates_file(self):
        from visualization import plot_storage_vs_k
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_storage_vs_k((100, 80), [1, 10, 20, 40], path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        finally:
            os.remove(path)

    def test_plot_comparison_grayscale_creates_file(self):
        from visualization import plot_comparison
        matrix = _random_grayscale(30, 20)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_comparison(matrix, [1, 5, 10], path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        finally:
            os.remove(path)

    def test_plot_comparison_rgb_creates_file(self):
        from visualization import plot_comparison
        matrix = _random_rgb(30, 20)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_comparison(matrix, [1, 5, 10], path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        finally:
            os.remove(path)

