"""Tests for COMP-01: Zeta function evaluation with always-validate pattern.

Tests verify zeta_eval and zeta_on_critical_line against known values,
the functional equation, and metadata correctness.
"""
import mpmath
import pytest

from riemann.types import ComputationResult


def test_zeta_known_values(default_precision):
    """zeta(2) = pi^2/6, zeta(4) = pi^4/90 to 45 digits."""
    from riemann.engine.zeta import zeta_eval

    with mpmath.workdps(60):
        result_2 = zeta_eval(mpmath.mpf(2))
        expected_2 = mpmath.pi ** 2 / 6
        assert abs(result_2.value - expected_2) < mpmath.power(10, -45), (
            f"zeta(2) mismatch: diff={abs(result_2.value - expected_2)}"
        )

        result_4 = zeta_eval(mpmath.mpf(4))
        expected_4 = mpmath.pi ** 4 / 90
        assert abs(result_4.value - expected_4) < mpmath.power(10, -45), (
            f"zeta(4) mismatch: diff={abs(result_4.value - expected_4)}"
        )


def test_zeta_critical_line_near_zero(default_precision):
    """zeta(0.5 + 14.134725i) should be near zero (first non-trivial zero)."""
    from riemann.engine.zeta import zeta_eval

    s = mpmath.mpc(mpmath.mpf("0.5"), mpmath.mpf("14.134725141734693790"))
    result = zeta_eval(s)
    assert abs(result.value) < mpmath.power(10, -6), (
        f"zeta at first zero not near zero: |zeta(s)| = {abs(result.value)}"
    )


def test_functional_equation(default_precision):
    """zeta(s) = chi(s) * zeta(1-s) for several test points."""
    from riemann.engine.zeta import zeta_eval

    test_points = [
        mpmath.mpc(mpmath.mpf("0.3"), mpmath.mpf("10.5")),
    ]

    for s in test_points:
        with mpmath.workdps(60):
            zeta_s = zeta_eval(s, dps=50)
            zeta_1ms = zeta_eval(1 - s, dps=50)

            # chi(s) = 2^s * pi^(s-1) * sin(pi*s/2) * gamma(1-s)
            chi = (
                mpmath.power(2, s)
                * mpmath.power(mpmath.pi, s - 1)
                * mpmath.sin(mpmath.pi * s / 2)
                * mpmath.gamma(1 - s)
            )

            lhs = zeta_s.value
            rhs = chi * zeta_1ms.value
            assert abs(lhs - rhs) < mpmath.power(10, -40), (
                f"Functional equation failed at s={s}: |lhs - rhs| = {abs(lhs - rhs)}"
            )


def test_zeta_negative_integers(default_precision):
    """zeta(-1) = -1/12."""
    from riemann.engine.zeta import zeta_eval

    with mpmath.workdps(60):
        result = zeta_eval(mpmath.mpf(-1))
        expected = mpmath.mpf("-1") / 12
        assert abs(result.value - expected) < mpmath.power(10, -45), (
            f"zeta(-1) mismatch: got {result.value}, expected {expected}"
        )


def test_zeta_eval_returns_computation_result(default_precision):
    """zeta_eval returns ComputationResult with validated=True and algorithm metadata."""
    from riemann.engine.zeta import zeta_eval

    result = zeta_eval(mpmath.mpf(2))
    assert isinstance(result, ComputationResult)
    assert result.validated is True
    assert result.algorithm == "mpmath.zeta"
    assert result.precision_digits == 50


def test_zeta_on_critical_line(default_precision):
    """zeta_on_critical_line(14.134725) returns ComputationResult near zero."""
    from riemann.engine.zeta import zeta_on_critical_line

    result = zeta_on_critical_line(mpmath.mpf("14.134725141734693790"))
    assert isinstance(result, ComputationResult)
    assert abs(result.value) < mpmath.power(10, -6), (
        f"zeta_on_critical_line at first zero not near zero: |value| = {abs(result.value)}"
    )
    assert result.validated is True
    assert result.algorithm == "mpmath.zeta(0.5+it)"


def test_zeta_eval_high_precision(default_precision):
    """zeta_eval with dps=100 produces result validated at 200 digits."""
    from riemann.engine.zeta import zeta_eval

    result = zeta_eval(mpmath.mpf(2), dps=100)
    assert result.precision_digits == 100
    assert result.validation_precision == 200
    assert result.validated is True

    with mpmath.workdps(110):
        expected = mpmath.pi ** 2 / 6
        assert abs(result.value - expected) < mpmath.power(10, -95)
