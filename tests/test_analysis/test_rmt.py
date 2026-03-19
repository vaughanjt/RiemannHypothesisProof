"""Tests for random matrix theory ensemble generation and eigenvalue statistics."""

import numpy as np
import pytest

from riemann.analysis.rmt import (
    eigenvalue_spacings,
    fit_effective_n,
    generate_goe,
    generate_gse,
    generate_gue,
    wigner_surmise,
)


class TestGenerateGUE:
    """Tests for GUE ensemble generation."""

    def test_returns_correct_number_of_matrices(self):
        result = generate_gue(n=10, num_matrices=5, seed=42)
        assert len(result) == 5

    def test_each_eigenvalue_array_has_correct_shape(self):
        result = generate_gue(n=10, num_matrices=5, seed=42)
        for eigs in result:
            assert isinstance(eigs, np.ndarray)
            assert eigs.shape == (10,)

    def test_eigenvalues_are_real(self):
        result = generate_gue(n=10, num_matrices=5, seed=42)
        for eigs in result:
            assert np.all(np.abs(eigs.imag) < 1e-10) if np.iscomplexobj(eigs) else True

    def test_reproducibility_with_same_seed(self):
        result1 = generate_gue(n=10, num_matrices=5, seed=42)
        result2 = generate_gue(n=10, num_matrices=5, seed=42)
        for e1, e2 in zip(result1, result2):
            np.testing.assert_array_equal(e1, e2)


class TestGenerateGOE:
    """Tests for GOE ensemble generation."""

    def test_returns_correct_number_and_shape(self):
        result = generate_goe(n=10, num_matrices=5, seed=42)
        assert len(result) == 5
        for eigs in result:
            assert isinstance(eigs, np.ndarray)
            assert eigs.shape == (10,)


class TestGenerateGSE:
    """Tests for GSE ensemble generation."""

    def test_returns_correct_number_and_shape(self):
        result = generate_gse(n=10, num_matrices=5, seed=42)
        assert len(result) == 5
        for eigs in result:
            assert isinstance(eigs, np.ndarray)
            assert eigs.shape == (10,)


class TestEigenvalueSpacings:
    """Tests for eigenvalue spacing computation with unfolding."""

    def test_mean_spacing_close_to_one(self):
        """GUE(100) with 200 matrices should have mean unfolded spacing near 1.0."""
        eigs = generate_gue(n=100, num_matrices=200, seed=42)
        spacings = eigenvalue_spacings(eigs)
        assert abs(spacings.mean() - 1.0) < 0.05

    def test_convergence_with_matrix_size(self):
        """GUE(N=200) spacings should match Wigner surmise better than GUE(N=10).

        As N increases, the spacing distribution converges to the universal
        Wigner surmise. We measure this via chi-squared distance from the
        theoretical GUE Wigner surmise distribution.
        """
        eigs_small = generate_gue(n=10, num_matrices=300, seed=42)
        eigs_large = generate_gue(n=200, num_matrices=300, seed=42)
        spacings_small = eigenvalue_spacings(eigs_small)
        spacings_large = eigenvalue_spacings(eigs_large)

        # Measure distance from Wigner surmise for each
        bins = np.linspace(0, 4, 41)
        centers = (bins[:-1] + bins[1:]) / 2
        wigner = wigner_surmise(centers, beta=2)

        hist_small, _ = np.histogram(spacings_small, bins=bins, density=True)
        hist_large, _ = np.histogram(spacings_large, bins=bins, density=True)

        mask = wigner > 0.01
        chi_sq_small = np.sum((hist_small[mask] - wigner[mask]) ** 2 / wigner[mask])
        chi_sq_large = np.sum((hist_large[mask] - wigner[mask]) ** 2 / wigner[mask])

        # Larger N should be closer to Wigner surmise (smaller chi-squared)
        assert chi_sq_large < chi_sq_small, (
            f"GUE(N=200) chi-sq={chi_sq_large:.4f} should be less than "
            f"GUE(N=10) chi-sq={chi_sq_small:.4f}"
        )


class TestWignerSurmise:
    """Tests for Wigner surmise probability density."""

    def test_gue_beta2_at_s1(self):
        """wigner_surmise(s=1.0, beta=2) matches (32/pi^2) * 1^2 * exp(-4/pi)."""
        s = np.array([1.0])
        expected = (32.0 / np.pi**2) * 1.0**2 * np.exp(-4.0 / np.pi)
        result = wigner_surmise(s, beta=2)
        np.testing.assert_allclose(result[0], expected, atol=1e-10)

    def test_goe_beta1_at_s1(self):
        """wigner_surmise(s=1.0, beta=1) matches (pi/2) * 1 * exp(-pi/4)."""
        s = np.array([1.0])
        expected = (np.pi / 2.0) * 1.0 * np.exp(-np.pi / 4.0)
        result = wigner_surmise(s, beta=1)
        np.testing.assert_allclose(result[0], expected, atol=1e-10)

    def test_gse_beta4_at_s1_positive(self):
        """wigner_surmise(s=1.0, beta=4) returns a positive float."""
        s = np.array([1.0])
        result = wigner_surmise(s, beta=4)
        assert result[0] > 0

    def test_invalid_beta_raises_valueerror(self):
        """wigner_surmise raises ValueError for beta=3."""
        s = np.array([1.0])
        with pytest.raises(ValueError, match="beta"):
            wigner_surmise(s, beta=3)


class TestGUESpacingMatchesWigner:
    """Tests that GUE spacing distribution matches Wigner surmise statistically."""

    def test_gue_histogram_matches_wigner_surmise(self):
        """GUE(N=100) with 500 matrices spacing histogram matches wigner_surmise(beta=2)."""
        eigs = generate_gue(n=100, num_matrices=500, seed=42)
        spacings = eigenvalue_spacings(eigs)

        # Create histogram
        bins = np.linspace(0, 4, 41)
        observed, _ = np.histogram(spacings, bins=bins, density=True)

        # Wigner surmise prediction at bin centers
        centers = (bins[:-1] + bins[1:]) / 2
        expected = wigner_surmise(centers, beta=2)

        # Chi-squared-like test: sum of squared residuals weighted by expected
        # Use a tolerance that allows for finite-N deviations
        mask = expected > 0.01  # Ignore near-zero bins
        residuals = (observed[mask] - expected[mask]) ** 2 / expected[mask]
        chi_sq = np.sum(residuals)

        # With 500 matrices of N=100, chi-squared should be moderate
        # Allow generous tolerance for statistical fluctuation
        assert chi_sq < 50, f"Chi-squared {chi_sq} too large -- GUE spacings don't match Wigner surmise"


class TestFitEffectiveN:
    """Tests for the fit_effective_n residual analysis tool."""

    def test_returns_positive_n_and_chi_squared(self):
        """fit_effective_n returns an integer N > 0 and chi-squared metric."""
        # Use synthetic GUE(N=50) data as target
        eigs = generate_gue(n=50, num_matrices=100, seed=123)
        spacings = eigenvalue_spacings(eigs)

        result = fit_effective_n(spacings, n_range=(10, 60), num_matrices=50, seed=42)

        assert isinstance(result["best_n"], int)
        assert result["best_n"] > 0
        assert isinstance(result["chi_squared"], float)
        assert result["chi_squared"] >= 0
        assert "all_fits" in result
        assert len(result["all_fits"]) > 0
