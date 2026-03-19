"""Tests for feature extraction and embedding computation.

Tests use synthetic ZetaZero objects -- does NOT require a real ZeroCatalog
or pre-computed zeros.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from mpmath import mpc, mpf

from riemann.types import ZetaZero


# ---------------------------------------------------------------------------
# Synthetic zeros fixture
# ---------------------------------------------------------------------------

def _make_zeros(n: int = 50) -> list[ZetaZero]:
    """Create n synthetic ZetaZero objects with realistic spacing.

    Imaginary parts start near 14.13 and increase by approximate mean spacing
    at that height: 2*pi / log(t/(2*pi)).
    """
    zeros = []
    t = 14.134725
    for i in range(1, n + 1):
        mean_spacing = 2 * np.pi / np.log(t / (2 * np.pi))
        value = mpc("0.5", str(t))
        zeros.append(ZetaZero(
            index=i,
            value=value,
            precision_digits=50,
            validated=True,
        ))
        # Next zero roughly mean_spacing away (add small jitter for realism)
        rng = np.random.default_rng(42 + i)
        t += mean_spacing * (0.8 + 0.4 * rng.random())
    return zeros


@pytest.fixture
def zeros():
    return _make_zeros(50)


@pytest.fixture
def small_zeros():
    """Small set for quick tests."""
    return _make_zeros(10)


# ---------------------------------------------------------------------------
# extract_imaginary_part
# ---------------------------------------------------------------------------

class TestExtractImaginaryPart:
    def test_returns_ndarray_float64(self, zeros):
        from riemann.embedding.coordinates import extract_imaginary_part
        result = extract_imaginary_part(zeros)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_correct_length(self, zeros):
        from riemann.embedding.coordinates import extract_imaginary_part
        result = extract_imaginary_part(zeros)
        assert len(result) == len(zeros)

    def test_values_match(self, zeros):
        from riemann.embedding.coordinates import extract_imaginary_part
        result = extract_imaginary_part(zeros)
        expected = np.array([float(z.value.imag) for z in zeros], dtype=np.float64)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# extract_left_spacing / extract_right_spacing
# ---------------------------------------------------------------------------

class TestExtractLeftSpacing:
    def test_returns_ndarray_correct_length(self, zeros):
        from riemann.embedding.coordinates import extract_left_spacing
        result = extract_left_spacing(zeros)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(zeros)

    def test_first_element_is_zero(self, zeros):
        from riemann.embedding.coordinates import extract_left_spacing
        result = extract_left_spacing(zeros)
        assert result[0] == 0.0


class TestExtractRightSpacing:
    def test_returns_ndarray_correct_length(self, zeros):
        from riemann.embedding.coordinates import extract_right_spacing
        result = extract_right_spacing(zeros)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(zeros)

    def test_last_element_is_zero(self, zeros):
        from riemann.embedding.coordinates import extract_right_spacing
        result = extract_right_spacing(zeros)
        assert result[-1] == 0.0


# ---------------------------------------------------------------------------
# extract_local_density_deviation
# ---------------------------------------------------------------------------

class TestExtractLocalDensityDeviation:
    def test_returns_ndarray_correct_length(self, zeros):
        from riemann.embedding.coordinates import extract_local_density_deviation
        result = extract_local_density_deviation(zeros)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(zeros)


# ---------------------------------------------------------------------------
# extract_zeta_derivative_magnitude (mocked)
# ---------------------------------------------------------------------------

class TestExtractZetaDerivativeMagnitude:
    def test_returns_ndarray_correct_length(self, small_zeros, tmp_path):
        from riemann.embedding.coordinates import extract_zeta_derivative_magnitude
        # Mock mpmath.zeta to avoid expensive computation
        with patch("riemann.embedding.coordinates.mpmath") as mock_mp:
            mock_mp.workdps = __import__("mpmath").workdps
            mock_mp.zeta.return_value = mpc("1.5", "0.3")
            mock_mp.mpf = mpf
            mock_mp.mpc = mpc
            result = extract_zeta_derivative_magnitude(small_zeros, cache_dir=tmp_path)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(small_zeros)


# ---------------------------------------------------------------------------
# compute_embedding
# ---------------------------------------------------------------------------

class TestComputeEmbedding:
    def test_spectral_basic_shape(self, zeros):
        from riemann.embedding.coordinates import compute_embedding
        from riemann.embedding.registry import get_preset
        config = get_preset("spectral_basic")
        emb = compute_embedding(config, zeros)
        assert emb.shape == (len(zeros), 3)  # 3 features in spectral_basic

    def test_standard_scaling_zero_mean(self, zeros):
        from riemann.embedding.coordinates import compute_embedding
        from riemann.embedding.registry import EmbeddingConfig
        config = EmbeddingConfig(
            name="test",
            description="test",
            feature_names=("imag_part", "spacing_left", "spacing_right"),
            scaling="standard",
        )
        emb = compute_embedding(config, zeros)
        # Standard-scaled columns should have ~zero mean
        for col in range(emb.shape[1]):
            assert abs(np.mean(emb[:, col])) < 1e-10

    def test_none_scaling_preserves_raw(self, zeros):
        from riemann.embedding.coordinates import compute_embedding
        from riemann.embedding.registry import EmbeddingConfig
        config = EmbeddingConfig(
            name="test",
            description="test",
            feature_names=("imag_part",),
            scaling="none",
        )
        emb = compute_embedding(config, zeros)
        expected = np.array([float(z.value.imag) for z in zeros], dtype=np.float64)
        np.testing.assert_allclose(emb[:, 0], expected)

    def test_unknown_feature_raises(self, zeros):
        from riemann.embedding.coordinates import compute_embedding
        from riemann.embedding.registry import EmbeddingConfig
        config = EmbeddingConfig(
            name="bad",
            description="bad",
            feature_names=("nonexistent_feature",),
        )
        with pytest.raises(ValueError, match="Unknown feature"):
            compute_embedding(config, zeros)


# ---------------------------------------------------------------------------
# FEATURE_EXTRACTORS registry updated (stubs replaced)
# ---------------------------------------------------------------------------

class TestRegistryUpdated:
    def test_imag_part_no_longer_stub(self, small_zeros):
        """After importing coordinates, FEATURE_EXTRACTORS['imag_part'] should work."""
        import riemann.embedding.coordinates  # noqa: F401 - triggers registration
        from riemann.embedding.registry import FEATURE_EXTRACTORS
        # Should NOT raise NotImplementedError
        result = FEATURE_EXTRACTORS["imag_part"](small_zeros)
        assert isinstance(result, np.ndarray)

    def test_spacing_left_no_longer_stub(self, small_zeros):
        import riemann.embedding.coordinates  # noqa: F401
        from riemann.embedding.registry import FEATURE_EXTRACTORS
        result = FEATURE_EXTRACTORS["spacing_left"](small_zeros)
        assert isinstance(result, np.ndarray)
