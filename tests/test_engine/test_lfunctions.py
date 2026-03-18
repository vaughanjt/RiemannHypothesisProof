"""Test scaffolds for COMP-03: Related functions (Hardy Z, Dirichlet L, xi, Selberg).

Tests are marked xfail pending implementation in Plan 01-03.
"""
import mpmath
import pytest

from riemann.types import ComputationResult


@pytest.mark.xfail(reason="Implementation pending in Plan 01-03")
def test_hardy_z_real_valued(default_precision):
    """siegelz(t) returns real value for real t."""
    from riemann.engine.lfunctions import hardy_z

    test_t = [10.0, 14.134, 25.0, 40.0]
    for t in test_t:
        result = hardy_z(t, dps=50)
        assert isinstance(result.value, mpmath.mpf) or abs(result.value.imag) < mpmath.power(10, -45), (
            f"Hardy Z({t}) not real-valued: {result.value}"
        )


@pytest.mark.xfail(reason="Implementation pending in Plan 01-03")
def test_dirichlet_trivial(default_precision):
    """dirichlet(s, [1]) equals zeta(s) for the trivial character."""
    from riemann.engine.lfunctions import dirichlet_l

    test_points = [2, 3, mpmath.mpc(0.5, 10)]
    for s in test_points:
        l_result = dirichlet_l(s, [1], dps=50)
        zeta_val = mpmath.zeta(s)
        diff = abs(l_result.value - zeta_val)
        assert diff < mpmath.power(10, -40), (
            f"dirichlet(s={s}, [1]) != zeta(s): diff={diff}"
        )


@pytest.mark.xfail(reason="Implementation pending in Plan 01-03")
def test_xi_symmetry(default_precision):
    """xi(s) = xi(1-s) for several test points."""
    from riemann.engine.lfunctions import xi_function

    test_points = [
        mpmath.mpc(0.5, 10),
        mpmath.mpc(0.3, 20),
        mpmath.mpc(0.7, 5),
        mpmath.mpc(2, 0),
    ]

    for s in test_points:
        xi_s = xi_function(s, dps=50)
        xi_1ms = xi_function(1 - s, dps=50)
        diff = abs(xi_s.value - xi_1ms.value)
        scale = max(abs(xi_s.value), abs(xi_1ms.value), mpmath.mpf(1))
        relative_err = diff / scale
        assert relative_err < mpmath.power(10, -40), (
            f"xi({s}) != xi(1-{s}): relative error = {relative_err}"
        )


@pytest.mark.xfail(reason="Implementation pending in Plan 01-03")
def test_xi_zeros_match_zeta_zeros(default_precision):
    """xi function has zeros at same locations as zeta non-trivial zeros."""
    from riemann.engine.lfunctions import xi_function

    # First few zeta zeros
    test_zeros_t = [14.134725, 21.022040, 25.010858]
    for t in test_zeros_t:
        s = mpmath.mpc(0.5, t)
        xi_val = xi_function(s, dps=50)
        assert abs(xi_val.value) < mpmath.power(10, -5), (
            f"xi(0.5 + {t}i) should be near zero: got {abs(xi_val.value)}"
        )


@pytest.mark.xfail(reason="Implementation pending in Plan 01-03")
def test_selberg_zeta_stub(default_precision):
    """Selberg zeta stub exists and is callable."""
    from riemann.engine.lfunctions import selberg_zeta

    # Just verify the stub exists and returns something
    result = selberg_zeta(mpmath.mpc(2, 0))
    assert result is not None
