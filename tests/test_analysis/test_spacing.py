"""Tests for zero distribution statistics engine.

Tests cover:
- normalized_spacings: mean ~1.0 for properly scaled zeros
- pair_correlation: shape, range, GUE comparison
- gue_pair_correlation: analytic sine kernel formula
- n_level_density: correct output shape
- number_variance: correct output length
- All functions: reject empty input with ValueError
"""
import numpy as np
import pytest
from mpmath import mpc, mpf

from riemann.types import ZetaZero


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_zero(index: int, imag_part: float) -> ZetaZero:
    """Create a synthetic ZetaZero with given imaginary part on the critical line."""
    return ZetaZero(
        index=index,
        value=mpc(0.5, imag_part),
        precision_digits=15,
        validated=False,
        on_critical_line=True,
    )


@pytest.fixture
def synthetic_zeros():
    """200 synthetic zeros with imaginary parts spaced so that
    normalized spacings have mean approximately 1.0.

    Mean spacing at height t is 2*pi / log(t / (2*pi)).
    We generate zeros starting at t=100 (where mean spacing ~ 0.92)
    and accumulate using the local mean spacing formula.
    """
    zeros = []
    t = 100.0  # Start high enough for asymptotic formula to be reasonable
    for i in range(200):
        zeros.append(_make_zero(i + 1, t))
        # Local mean spacing at height t
        mean_spacing = 2 * np.pi / np.log(t / (2 * np.pi))
        t += mean_spacing  # Next zero at exactly one mean spacing
    return zeros


@pytest.fixture
def small_zeros():
    """A small set of 10 synthetic zeros for basic tests."""
    t_values = [14.13, 21.02, 25.01, 30.42, 32.94,
                37.59, 40.92, 43.33, 48.01, 49.77]
    return [_make_zero(i + 1, t) for i, t in enumerate(t_values)]


# ---------------------------------------------------------------------------
# normalized_spacings
# ---------------------------------------------------------------------------

class TestNormalizedSpacings:
    def test_returns_ndarray(self, small_zeros):
        from riemann.analysis.spacing import normalized_spacings
        result = normalized_spacings(small_zeros)
        assert isinstance(result, np.ndarray)

    def test_length_is_n_minus_one(self, small_zeros):
        from riemann.analysis.spacing import normalized_spacings
        result = normalized_spacings(small_zeros)
        assert len(result) == len(small_zeros) - 1

    def test_mean_approx_one_for_synthetic(self, synthetic_zeros):
        """Zeros constructed to have mean normalized spacing of 1.0."""
        from riemann.analysis.spacing import normalized_spacings
        result = normalized_spacings(synthetic_zeros)
        assert abs(np.mean(result) - 1.0) < 0.1, (
            f"Mean normalized spacing {np.mean(result):.4f} not within 0.1 of 1.0"
        )

    def test_rejects_empty(self):
        from riemann.analysis.spacing import normalized_spacings
        with pytest.raises(ValueError):
            normalized_spacings([])

    def test_rejects_single_zero(self):
        from riemann.analysis.spacing import normalized_spacings
        with pytest.raises(ValueError):
            normalized_spacings([_make_zero(1, 14.13)])


# ---------------------------------------------------------------------------
# pair_correlation
# ---------------------------------------------------------------------------

class TestPairCorrelation:
    def test_returns_tuple_of_two_arrays(self, synthetic_zeros):
        from riemann.analysis.spacing import normalized_spacings, pair_correlation
        spacings = normalized_spacings(synthetic_zeros)
        x_centers, r2 = pair_correlation(spacings)
        assert isinstance(x_centers, np.ndarray)
        assert isinstance(r2, np.ndarray)

    def test_shapes_match_bins(self, synthetic_zeros):
        from riemann.analysis.spacing import normalized_spacings, pair_correlation
        spacings = normalized_spacings(synthetic_zeros)
        bins = 100
        x_centers, r2 = pair_correlation(spacings, bins=bins)
        assert len(x_centers) == bins
        assert len(r2) == bins

    def test_x_range_respected(self, synthetic_zeros):
        from riemann.analysis.spacing import normalized_spacings, pair_correlation
        spacings = normalized_spacings(synthetic_zeros)
        x_centers, r2 = pair_correlation(spacings, x_range=(0.0, 3.0))
        assert x_centers[0] > 0.0
        assert x_centers[-1] < 3.0

    def test_rejects_empty(self):
        from riemann.analysis.spacing import pair_correlation
        with pytest.raises(ValueError):
            pair_correlation(np.array([]))


# ---------------------------------------------------------------------------
# gue_pair_correlation
# ---------------------------------------------------------------------------

class TestGuePairCorrelation:
    def test_analytic_values(self):
        """Verify against known analytic values of 1 - (sin(pi*x)/(pi*x))^2."""
        from riemann.analysis.spacing import gue_pair_correlation
        x = np.array([0.5, 1.0, 2.0])
        result = gue_pair_correlation(x)

        # At x=0.5: 1 - (sin(pi*0.5)/(pi*0.5))^2 = 1 - (1/(pi/2))^2 = 1 - 4/pi^2
        expected_05 = 1.0 - (np.sin(np.pi * 0.5) / (np.pi * 0.5)) ** 2
        assert abs(result[0] - expected_05) < 1e-10

        # At x=1.0: sin(pi) = 0, so gue = 1 - 0 = 1.0
        assert abs(result[1] - 1.0) < 1e-10

        # At x=2.0: sin(2*pi) = 0, so gue = 1 - 0 = 1.0
        assert abs(result[2] - 1.0) < 1e-10

    def test_at_zero(self):
        """At x=0, sinc(0)=1, so gue = 1 - 1 = 0."""
        from riemann.analysis.spacing import gue_pair_correlation
        result = gue_pair_correlation(np.array([0.0]))
        assert abs(result[0] - 0.0) < 1e-10

    def test_approaches_one_at_large_x(self):
        """For large x, sin(pi*x)/(pi*x) -> 0, so gue -> 1."""
        from riemann.analysis.spacing import gue_pair_correlation
        result = gue_pair_correlation(np.array([100.0]))
        assert abs(result[0] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# pair_correlation vs GUE (statistical test with Wigner surmise sample)
# ---------------------------------------------------------------------------

class TestPairCorrelationVsGUE:
    def test_gue_like_spacings_match_gue_prediction(self):
        """Generate spacings from Wigner surmise p(s) = (pi/2)*s*exp(-pi*s^2/4),
        verify pair_correlation at x=1.0 is within 0.2 of gue_pair_correlation(1.0).
        """
        from riemann.analysis.spacing import pair_correlation, gue_pair_correlation

        rng = np.random.default_rng(42)
        # Inverse CDF sampling from Wigner surmise:
        # CDF(s) = 1 - exp(-pi*s^2/4)
        # s = sqrt(-4*ln(1-u)/pi)
        u = rng.uniform(0, 1, size=10_000)
        spacings = np.sqrt(-4.0 * np.log(1.0 - u) / np.pi)

        x_centers, r2 = pair_correlation(spacings, bins=200, x_range=(0.0, 4.0))
        gue = gue_pair_correlation(x_centers)

        # Find bin closest to x=1.0
        idx_1 = np.argmin(np.abs(x_centers - 1.0))
        assert abs(r2[idx_1] - gue[idx_1]) < 0.2, (
            f"pair_correlation at x~1.0: got {r2[idx_1]:.3f}, "
            f"expected ~{gue[idx_1]:.3f} (tolerance 0.2)"
        )


# ---------------------------------------------------------------------------
# n_level_density
# ---------------------------------------------------------------------------

class TestNLevelDensity:
    def test_returns_ndarray_for_n2(self, synthetic_zeros):
        from riemann.analysis.spacing import normalized_spacings, n_level_density
        spacings = normalized_spacings(synthetic_zeros)
        result = n_level_density(spacings, n=2)
        assert isinstance(result, np.ndarray)

    def test_correct_length(self, synthetic_zeros):
        from riemann.analysis.spacing import normalized_spacings, n_level_density
        spacings = normalized_spacings(synthetic_zeros)
        bins = 80
        result = n_level_density(spacings, n=2, bins=bins)
        assert len(result) == bins

    def test_rejects_empty(self):
        from riemann.analysis.spacing import n_level_density
        with pytest.raises(ValueError):
            n_level_density(np.array([]), n=2)


# ---------------------------------------------------------------------------
# number_variance
# ---------------------------------------------------------------------------

class TestNumberVariance:
    def test_returns_ndarray(self, synthetic_zeros):
        from riemann.analysis.spacing import normalized_spacings, number_variance
        spacings = normalized_spacings(synthetic_zeros)
        L_values = np.linspace(0.1, 3.0, 20)
        result = number_variance(spacings, L_values)
        assert isinstance(result, np.ndarray)

    def test_length_matches_L_values(self, synthetic_zeros):
        from riemann.analysis.spacing import normalized_spacings, number_variance
        spacings = normalized_spacings(synthetic_zeros)
        L_values = np.linspace(0.1, 5.0, 50)
        result = number_variance(spacings, L_values)
        assert len(result) == len(L_values)

    def test_default_L_values(self, synthetic_zeros):
        from riemann.analysis.spacing import normalized_spacings, number_variance
        spacings = normalized_spacings(synthetic_zeros)
        result = number_variance(spacings)
        assert len(result) == 50  # default is 50 points

    def test_rejects_empty(self):
        from riemann.analysis.spacing import number_variance
        with pytest.raises(ValueError):
            number_variance(np.array([]))
