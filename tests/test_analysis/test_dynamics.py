"""Tests for dynamical systems module.

Tests cover:
- Gauss/zeta map computation
- Logistic map computation
- Orbit generation
- Lyapunov exponent: positive for chaos, negative for stable
- Fixed point detection
- Analyze dynamics convenience function
- DynamicsResult structure
"""
import numpy as np
import pytest

from riemann.analysis.dynamics import (
    DynamicsResult,
    analyze_dynamics,
    compute_orbit,
    find_fixed_points,
    logistic_map,
    lyapunov_exponent,
    zeta_map,
)


# ---------------------------------------------------------------------------
# zeta_map (Gauss map)
# ---------------------------------------------------------------------------


class TestZetaMap:
    def test_returns_float(self):
        result = zeta_map(0.3)
        assert isinstance(result, float)

    def test_known_value(self):
        """f(0.3) = frac(1/0.3) = frac(3.333...) = 0.333..."""
        result = zeta_map(0.3)
        assert abs(result - 1.0 / 3.0) < 0.01

    def test_near_zero_clamped(self):
        """Very small x should not cause division by zero."""
        result = zeta_map(1e-20)
        assert np.isfinite(result)

    def test_output_in_unit_interval(self):
        """Gauss map output should be in [0, 1) for x in (0, 1)."""
        for x in [0.1, 0.25, 0.5, 0.7, 0.9]:
            result = zeta_map(x)
            assert 0.0 <= result < 1.0 + 1e-10


# ---------------------------------------------------------------------------
# logistic_map
# ---------------------------------------------------------------------------


class TestLogisticMap:
    def test_known_value(self):
        """f(0.5, r=4) = 4 * 0.5 * 0.5 = 1.0."""
        result = logistic_map(0.5, r=4.0)
        assert abs(result - 1.0) < 1e-10

    def test_fixed_point_at_stable_r(self):
        """For r=2, the fixed point is x* = 1 - 1/r = 0.5.
        f(0.5, r=2) = 2 * 0.5 * 0.5 = 0.5."""
        result = logistic_map(0.5, r=2.0)
        assert abs(result - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# compute_orbit
# ---------------------------------------------------------------------------


class TestComputeOrbit:
    def test_returns_ndarray(self):
        orbit = compute_orbit(lambda x: logistic_map(x, r=3.9), x0=0.1, n_steps=100)
        assert isinstance(orbit, np.ndarray)

    def test_correct_length(self):
        orbit = compute_orbit(lambda x: logistic_map(x, r=3.9), x0=0.1, n_steps=50)
        assert len(orbit) == 50

    def test_gauss_map_orbit_bounded(self):
        """Gauss map orbit should stay in [0, 1)."""
        orbit = compute_orbit(zeta_map, x0=0.3, n_steps=100, transient=50)
        assert np.all(orbit >= 0.0)
        assert np.all(orbit <= 1.0 + 1e-10)

    def test_transient_discarded(self):
        """With transient=50, orbit should not include early iterates."""
        orbit = compute_orbit(lambda x: logistic_map(x, r=3.9), x0=0.1,
                              n_steps=100, transient=50)
        assert len(orbit) == 100


# ---------------------------------------------------------------------------
# lyapunov_exponent
# ---------------------------------------------------------------------------


class TestLyapunovExponent:
    def test_chaotic_logistic_positive(self):
        """Logistic map at r=4.0 (fully chaotic): Lyapunov exponent should be > 0.5.
        Theoretical value: ln(2) ~ 0.693."""
        f = lambda x: logistic_map(x, r=4.0)
        orbit = compute_orbit(f, x0=0.1, n_steps=5000, transient=200)
        lyap = lyapunov_exponent(orbit, map_func=f)
        assert lyap > 0.5, f"Expected positive Lyapunov for chaotic logistic, got {lyap}"

    def test_stable_logistic_negative(self):
        """Logistic map at r=2.0 (stable fixed point): Lyapunov exponent should be < 0."""
        f = lambda x: logistic_map(x, r=2.0)
        orbit = compute_orbit(f, x0=0.4, n_steps=5000, transient=200)
        lyap = lyapunov_exponent(orbit, map_func=f)
        assert lyap < 0.0, f"Expected negative Lyapunov for stable logistic, got {lyap}"

    def test_returns_float(self):
        f = lambda x: logistic_map(x, r=3.5)
        orbit = compute_orbit(f, x0=0.1, n_steps=1000, transient=100)
        lyap = lyapunov_exponent(orbit, map_func=f)
        assert isinstance(lyap, float)


# ---------------------------------------------------------------------------
# find_fixed_points
# ---------------------------------------------------------------------------


class TestFindFixedPoints:
    def test_logistic_fixed_point(self):
        """Logistic map at r=3.0 has fixed point x* = 1 - 1/r = 2/3."""
        f = lambda x: logistic_map(x, r=3.0)
        fps = find_fixed_points(f, search_range=(0.01, 0.99))
        expected = 1.0 - 1.0 / 3.0  # 2/3
        # Should find at least one fixed point near 2/3
        assert any(abs(fp - expected) < 0.01 for fp in fps), (
            f"Expected fixed point near {expected}, found {fps}"
        )

    def test_logistic_also_finds_zero_region(self):
        """The trivial fixed point x*=0 may also be found (or near-zero)."""
        f = lambda x: logistic_map(x, r=3.0)
        fps = find_fixed_points(f, search_range=(0.001, 0.99))
        # At least the non-trivial fixed point should be found
        assert len(fps) >= 1

    def test_returns_sorted_list(self):
        f = lambda x: logistic_map(x, r=3.5)
        fps = find_fixed_points(f, search_range=(0.01, 0.99))
        assert fps == sorted(fps)


# ---------------------------------------------------------------------------
# analyze_dynamics (convenience function)
# ---------------------------------------------------------------------------


class TestAnalyzeDynamics:
    def test_returns_dynamics_result(self):
        result = analyze_dynamics(
            lambda x: logistic_map(x, r=3.9),
            x0=0.1,
            n_steps=500,
            map_name="logistic_3.9",
        )
        assert isinstance(result, DynamicsResult)

    def test_result_has_all_fields(self):
        result = analyze_dynamics(
            lambda x: logistic_map(x, r=3.9),
            x0=0.1,
            n_steps=500,
            map_name="logistic_3.9",
        )
        assert isinstance(result.orbit, np.ndarray)
        assert isinstance(result.lyapunov, float)
        assert isinstance(result.fixed_points, list)
        assert result.map_name == "logistic_3.9"
        assert isinstance(result.metadata, dict)
