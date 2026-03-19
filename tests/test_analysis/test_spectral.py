"""Tests for spectral operator analysis module.

Tests cover:
- SpectralResult dataclass structure
- construct_berry_keating_box: real symmetric matrix with real eigenvalues
- construct_berry_keating_smooth: Hermitian matrix with smooth potential
- compute_spectrum: sorted eigenvalues in SpectralResult
- spectral_comparison: chi-squared and KS statistics
- Edge cases: small matrices, identical/divergent distributions
"""
import numpy as np
import pytest


class TestSpectralResult:
    def test_dataclass_fields(self):
        """SpectralResult has eigenvalues, operator_name, matrix_size, chi_squared_fit, metadata."""
        from riemann.analysis.spectral import SpectralResult

        result = SpectralResult(
            eigenvalues=np.array([1.0, 2.0, 3.0]),
            operator_name="test",
            matrix_size=3,
            chi_squared_fit=0.0,
            metadata={},
        )
        assert isinstance(result.eigenvalues, np.ndarray)
        assert result.operator_name == "test"
        assert result.matrix_size == 3
        assert result.chi_squared_fit == 0.0
        assert result.metadata == {}


class TestConstructBerryKeatingBox:
    def test_returns_square_matrix(self):
        """construct_berry_keating_box(n=100) returns (100, 100) array."""
        from riemann.analysis.spectral import construct_berry_keating_box

        H = construct_berry_keating_box(n=100)
        assert H.shape == (100, 100)

    def test_real_symmetric(self):
        """Matrix is real and symmetric."""
        from riemann.analysis.spectral import construct_berry_keating_box

        H = construct_berry_keating_box(n=100)
        assert H.dtype in (np.float64, np.float32)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_eigenvalues_are_real(self):
        """All eigenvalues have imaginary parts < 1e-10."""
        from riemann.analysis.spectral import construct_berry_keating_box

        H = construct_berry_keating_box(n=100)
        eigenvalues = np.linalg.eigvals(H)
        assert np.all(np.abs(eigenvalues.imag) < 1e-10)

    def test_different_sizes(self):
        """Works for various matrix sizes."""
        from riemann.analysis.spectral import construct_berry_keating_box

        for n in [10, 50, 200]:
            H = construct_berry_keating_box(n=n)
            assert H.shape == (n, n)


class TestConstructBerryKeatingSmooth:
    def test_returns_square_matrix(self):
        """construct_berry_keating_smooth(n=100) returns (100, 100) array."""
        from riemann.analysis.spectral import construct_berry_keating_smooth

        H = construct_berry_keating_smooth(n=100)
        assert H.shape == (100, 100)

    def test_real_symmetric(self):
        """Smooth regularization preserves real symmetric structure."""
        from riemann.analysis.spectral import construct_berry_keating_smooth

        H = construct_berry_keating_smooth(n=100)
        assert H.dtype in (np.float64, np.float32)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_differs_from_box(self):
        """Smooth version differs from box version (potential adds to diagonal)."""
        from riemann.analysis.spectral import (
            construct_berry_keating_box,
            construct_berry_keating_smooth,
        )

        H_box = construct_berry_keating_box(n=50)
        H_smooth = construct_berry_keating_smooth(n=50)
        assert not np.allclose(H_box, H_smooth)


class TestComputeSpectrum:
    def test_returns_spectral_result(self):
        """compute_spectrum returns a SpectralResult dataclass."""
        from riemann.analysis.spectral import SpectralResult, compute_spectrum

        matrix = np.eye(10)
        result = compute_spectrum(matrix, operator_name="identity")
        assert isinstance(result, SpectralResult)

    def test_eigenvalues_sorted_ascending(self):
        """Eigenvalues in the result are sorted ascending."""
        from riemann.analysis.spectral import compute_spectrum

        matrix = np.diag([3.0, 1.0, 2.0])
        result = compute_spectrum(matrix)
        assert np.all(np.diff(result.eigenvalues) >= 0)

    def test_correct_fields(self):
        """Result has correct operator_name and matrix_size."""
        from riemann.analysis.spectral import compute_spectrum

        matrix = np.eye(5)
        result = compute_spectrum(matrix, operator_name="test_op")
        assert result.operator_name == "test_op"
        assert result.matrix_size == 5
        assert len(result.eigenvalues) == 5


class TestSpectralComparison:
    def test_returns_dict_with_required_keys(self):
        """spectral_comparison returns dict with chi_squared, ks_statistic, ks_pvalue, n_eigenvalues, n_zeros."""
        from riemann.analysis.spectral import spectral_comparison

        eigs = np.sort(np.random.default_rng(42).standard_normal(100))
        zeros = np.sort(np.random.default_rng(43).standard_normal(100))
        result = spectral_comparison(eigs, zeros)
        assert "chi_squared" in result
        assert "ks_statistic" in result
        assert "ks_pvalue" in result
        assert "n_eigenvalues" in result
        assert "n_zeros" in result

    def test_identical_distributions_low_chi_squared(self):
        """Same distribution should give chi_squared < 1.0."""
        from riemann.analysis.spectral import spectral_comparison

        rng = np.random.default_rng(42)
        data = np.abs(rng.standard_normal(500))
        # Use same data for both -- identical distributions
        result = spectral_comparison(data, data.copy())
        assert result["chi_squared"] < 1.0, (
            f"Expected chi_squared < 1.0 for identical distributions, got {result['chi_squared']}"
        )

    def test_different_distributions_high_chi_squared(self):
        """Very different distributions should give chi_squared > 5.0."""
        from riemann.analysis.spectral import spectral_comparison

        rng = np.random.default_rng(42)
        # Uniform spacings vs exponential spacings -- very different shapes
        eigs = np.sort(rng.uniform(0, 4, 500))
        zeros = np.sort(rng.exponential(0.2, 500))
        result = spectral_comparison(eigs, zeros)
        assert result["chi_squared"] > 5.0, (
            f"Expected chi_squared > 5.0 for very different distributions, got {result['chi_squared']}"
        )

    def test_counts_match_input_sizes(self):
        """n_eigenvalues and n_zeros match input array lengths."""
        from riemann.analysis.spectral import spectral_comparison

        eigs = np.sort(np.random.default_rng(42).standard_normal(80))
        zeros = np.sort(np.random.default_rng(43).standard_normal(120))
        result = spectral_comparison(eigs, zeros)
        assert result["n_eigenvalues"] == 80
        assert result["n_zeros"] == 120
