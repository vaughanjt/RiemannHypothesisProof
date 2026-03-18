"""Tests for COMP-03: Related functions (Hardy Z, Dirichlet L, xi, Selberg).

Covers Hardy Z-function, Dirichlet L-functions, xi function symmetry,
and Selberg zeta stub.
"""
import mpmath
import pytest

from riemann.types import ComputationResult


class TestHardyZ:
    """Hardy Z-function tests: real-valued for real t, |Z(t)| = |zeta(1/2+it)|."""

    def test_hardy_z_real_valued_first_zero(self, default_precision):
        """siegelz(14.134725) returns real-valued ComputationResult."""
        from riemann.engine.lfunctions import hardy_z

        result = hardy_z(14.134725, dps=50)
        assert isinstance(result, ComputationResult)
        # The value should be real (mpf) or have negligible imaginary part
        if isinstance(result.value, mpmath.mpc):
            assert abs(result.value.imag) < mpmath.power(10, -45), (
                f"Hardy Z(14.134725) imaginary part too large: {result.value.imag}"
            )

    def test_hardy_z_real_valued_multiple_t(self, default_precision):
        """Z(t) is real-valued for multiple real t values."""
        from riemann.engine.lfunctions import hardy_z

        for t in [10, 20, 30, 50]:
            result = hardy_z(t, dps=50)
            if isinstance(result.value, mpmath.mpc):
                assert abs(result.value.imag) < mpmath.power(10, -45), (
                    f"Hardy Z({t}) not real-valued: imag = {result.value.imag}"
                )

    def test_hardy_z_magnitude_equals_zeta(self, default_precision):
        """|Z(t)| = |zeta(1/2+it)| for test points."""
        from riemann.engine.lfunctions import hardy_z

        for t in [10.0, 20.0, 30.0]:
            result = hardy_z(t, dps=50)
            z_mag = abs(result.value)
            with mpmath.workdps(60):
                zeta_val = mpmath.zeta(mpmath.mpc(0.5, t))
                zeta_mag = abs(zeta_val)
            # Should agree to at least 30 digits
            if zeta_mag > 0:
                rel_err = abs(z_mag - zeta_mag) / zeta_mag
                assert rel_err < mpmath.power(10, -30), (
                    f"|Z({t})| != |zeta(1/2+{t}i)|: rel_err = {rel_err}"
                )


class TestDirichletL:
    """Dirichlet L-function tests: trivial character gives zeta, non-trivial works."""

    def test_dirichlet_trivial_equals_zeta(self, default_precision):
        """dirichlet_l(2, [1]) equals zeta(2) = pi^2/6 to 45 digits."""
        from riemann.engine.lfunctions import dirichlet_l

        result = dirichlet_l(2, [1], dps=50)
        assert isinstance(result, ComputationResult)
        with mpmath.workdps(60):
            expected = mpmath.pi ** 2 / 6
            diff = abs(result.value - expected)
            assert diff < mpmath.power(10, -45), (
                f"dirichlet(2, [1]) != pi^2/6: diff = {diff}"
            )

    def test_dirichlet_nontrivial_character(self, default_precision):
        """dirichlet_l with non-trivial character mod 4 returns valid complex number."""
        from riemann.engine.lfunctions import dirichlet_l

        s = mpmath.mpc(0.5, 10)
        result = dirichlet_l(s, [0, 1, 0, -1])
        assert isinstance(result, ComputationResult)
        # Should be a finite complex number
        assert mpmath.isfinite(result.value), (
            f"dirichlet(0.5+10i, chi_mod4) not finite: {result.value}"
        )


class TestXiFunction:
    """Xi function tests: symmetry xi(s) = xi(1-s), zeros match zeta zeros."""

    def test_xi_symmetry(self, default_precision):
        """xi(s) = xi(1-s) for several test points."""
        from riemann.engine.lfunctions import xi_function

        # Use string-based construction and compute 1-s via mpmath arithmetic
        # to avoid float64 truncation (e.g., Python float 0.7 != 7/10 exactly).
        with mpmath.workdps(60):
            s = mpmath.mpc("0.3", "5")
            one_minus_s = 1 - s  # exact: (0.7 - 5j)

        xi_s = xi_function(s, dps=50)
        xi_1ms = xi_function(one_minus_s, dps=50)
        with mpmath.workdps(60):
            diff = abs(xi_s.value - xi_1ms.value)
            scale = max(abs(xi_s.value), abs(xi_1ms.value), mpmath.mpf(1))
            relative_err = diff / scale
            assert relative_err < mpmath.power(10, -40), (
                f"xi(0.3+5i) != xi(0.7-5i): relative error = {relative_err}"
            )

    def test_xi_symmetry_additional(self, default_precision):
        """xi(s) = xi(1-s) for additional test points."""
        from riemann.engine.lfunctions import xi_function

        # Use exact-precision inputs and compute 1-s via mpmath arithmetic
        with mpmath.workdps(60):
            test_s_values = [
                mpmath.mpc("0.5", "10"),
                mpmath.mpc("2", "0"),
            ]
        for s in test_s_values:
            with mpmath.workdps(60):
                one_minus_s = 1 - s
            xi_s = xi_function(s, dps=50)
            xi_1ms = xi_function(one_minus_s, dps=50)
            with mpmath.workdps(60):
                diff = abs(xi_s.value - xi_1ms.value)
                scale = max(abs(xi_s.value), abs(xi_1ms.value), mpmath.mpf(1))
                relative_err = diff / scale
                assert relative_err < mpmath.power(10, -35), (
                    f"xi({s}) != xi({one_minus_s}): relative error = {relative_err}"
                )

    def test_xi_near_zero_at_zeta_zero(self, default_precision):
        """xi function at a known zeta zero location is near zero."""
        from riemann.engine.lfunctions import xi_function

        # First zeta zero: 0.5 + 14.134725...i
        s = mpmath.mpc(0.5, 14.134725)
        xi_val = xi_function(s, dps=50)
        assert abs(xi_val.value) < mpmath.power(10, -5), (
            f"xi(0.5 + 14.134725i) should be near zero: got {abs(xi_val.value)}"
        )


class TestSelbergZetaStub:
    """Selberg zeta stub: callable, raises NotImplementedError."""

    def test_selberg_zeta_stub_callable(self):
        """selberg_zeta_stub is callable and raises NotImplementedError."""
        from riemann.engine.lfunctions import selberg_zeta_stub

        assert callable(selberg_zeta_stub)

    def test_selberg_zeta_stub_raises(self):
        """selberg_zeta_stub raises NotImplementedError with informative message."""
        from riemann.engine.lfunctions import selberg_zeta_stub

        with pytest.raises(NotImplementedError, match="Selberg zeta"):
            selberg_zeta_stub()

    def test_selberg_zeta_stub_with_args(self):
        """selberg_zeta_stub accepts arguments but still raises."""
        from riemann.engine.lfunctions import selberg_zeta_stub

        with pytest.raises(NotImplementedError):
            selberg_zeta_stub(spectral_data=[1.0, 2.0])
