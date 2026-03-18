"""Test scaffolds for COMP-01: Zeta function evaluation.

Tests are marked xfail pending implementation in Plan 01-02.
"""
import mpmath
import pytest

from riemann.types import ComputationResult


@pytest.mark.xfail(reason="Implementation pending in Plan 01-02")
def test_zeta_known_values(default_precision):
    """zeta(2) = pi^2/6, zeta(4) = pi^4/90 to 50 digits."""
    # TODO: Use validated zeta wrapper when implemented
    from riemann.engine.zeta import validated_zeta

    result_2 = validated_zeta(2, dps=50)
    expected_2 = mpmath.pi ** 2 / 6
    assert abs(result_2.value - expected_2) < mpmath.power(10, -45)

    result_4 = validated_zeta(4, dps=50)
    expected_4 = mpmath.pi ** 4 / 90
    assert abs(result_4.value - expected_4) < mpmath.power(10, -45)


@pytest.mark.xfail(reason="Implementation pending in Plan 01-02")
def test_zeta_critical_line_near_zero(default_precision):
    """zeta(0.5 + 14.134725i) should be near zero (first non-trivial zero)."""
    from riemann.engine.zeta import validated_zeta

    s = mpmath.mpc(0.5, mpmath.mpf("14.134725141734693790"))
    result = validated_zeta(s, dps=50)
    assert abs(result.value) < mpmath.power(10, -10)


@pytest.mark.xfail(reason="Implementation pending in Plan 01-02")
def test_functional_equation(default_precision):
    """zeta(s) = chi(s) * zeta(1-s) for several test points."""
    from riemann.engine.zeta import validated_zeta

    test_points = [
        mpmath.mpc(0.5, 10),
        mpmath.mpc(0.3, 20),
        mpmath.mpc(0.7, 5),
    ]

    for s in test_points:
        zeta_s = validated_zeta(s, dps=50)
        zeta_1ms = validated_zeta(1 - s, dps=50)

        # chi(s) = 2^s * pi^(s-1) * sin(pi*s/2) * gamma(1-s)
        chi = (mpmath.power(2, s) * mpmath.power(mpmath.pi, s - 1)
               * mpmath.sin(mpmath.pi * s / 2) * mpmath.gamma(1 - s))

        lhs = zeta_s.value
        rhs = chi * zeta_1ms.value
        assert abs(lhs - rhs) < mpmath.power(10, -40), (
            f"Functional equation failed at s={s}: |lhs - rhs| = {abs(lhs - rhs)}"
        )


@pytest.mark.xfail(reason="Implementation pending in Plan 01-02")
def test_zeta_negative_integers(default_precision):
    """zeta(-1) = -1/12, trivial zeros at -2, -4, -6."""
    from riemann.engine.zeta import validated_zeta

    result = validated_zeta(-1, dps=50)
    assert abs(result.value - mpmath.mpf("-1") / 12) < mpmath.power(10, -45)

    # Trivial zeros
    for n in [-2, -4, -6]:
        result = validated_zeta(n, dps=50)
        assert abs(result.value) < mpmath.power(10, -45), f"zeta({n}) should be 0"
