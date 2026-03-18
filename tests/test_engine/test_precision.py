"""Tests for precision management layer: precision_scope and validated_computation.

TDD RED phase: these tests define the expected behavior of the precision module.
"""
import mpmath
import pytest

from riemann.types import ComputationResult, PrecisionError


def test_precision_scope_sets_dps():
    """precision_scope(100) should set mpmath.mp.dps to 105 (100 + 5 guard digits)."""
    from riemann.engine.precision import precision_scope

    original_dps = mpmath.mp.dps
    with precision_scope(100) as dps:
        assert mpmath.mp.dps == 105, f"Expected 105, got {mpmath.mp.dps}"
        assert dps == 100

    # Verify dps restored
    assert mpmath.mp.dps == original_dps


def test_precision_scope_exception_safe():
    """precision_scope must restore dps even if an exception occurs inside."""
    from riemann.engine.precision import precision_scope

    original_dps = mpmath.mp.dps
    with pytest.raises(ValueError):
        with precision_scope(200):
            assert mpmath.mp.dps == 205
            raise ValueError("Intentional error")

    assert mpmath.mp.dps == original_dps, f"DPS not restored: got {mpmath.mp.dps}"


def test_validated_computation_agrees():
    """validated_computation should return validated=True when P and 2P agree."""
    from riemann.engine.precision import validated_computation

    result = validated_computation(lambda: mpmath.zeta(2), dps=30)

    assert isinstance(result, ComputationResult)
    assert result.validated is True
    assert result.precision_digits == 30
    assert result.validation_precision == 60

    # zeta(2) = pi^2/6
    with mpmath.workdps(60):
        expected = mpmath.pi ** 2 / 6
        diff = abs(result.value - expected)
        assert diff < mpmath.power(10, -25), f"zeta(2) result too far from pi^2/6: diff={diff}"


def test_validated_computation_catches_disagreement():
    """validated_computation should raise PrecisionError when P and 2P disagree."""
    from riemann.engine.precision import validated_computation

    def bad_func():
        """Returns different values depending on precision -- simulates collapse."""
        if mpmath.mp.dps < 60:
            return mpmath.mpf("1.0")
        else:
            return mpmath.mpf("2.0")

    with pytest.raises(PrecisionError):
        validated_computation(bad_func, dps=30)


def test_validated_computation_skip_validation():
    """validated_computation with validate=False should skip double-computation."""
    from riemann.engine.precision import validated_computation

    result = validated_computation(lambda: mpmath.zeta(2), dps=30, validate=False)

    assert isinstance(result, ComputationResult)
    assert result.validated is False
    assert result.validation_precision is None
    assert result.precision_digits == 30


def test_validated_computation_returns_2p_result():
    """validated_computation should return the 2P (higher precision) result."""
    from riemann.engine.precision import validated_computation

    result = validated_computation(lambda: mpmath.zeta(2), dps=30)

    # The returned value should be the 2P computation
    # Compute at 2P ourselves and compare
    with mpmath.workdps(65):  # 2*30 + 5 guard digits
        expected_2p = mpmath.zeta(2)

    with mpmath.workdps(65):
        diff = abs(result.value - expected_2p)
        # Should agree to very high precision
        assert diff < mpmath.power(10, -55), f"Result doesn't match 2P computation: diff={diff}"


def test_default_dps_from_config():
    """validated_computation without specifying dps should use DEFAULT_DPS=50."""
    from riemann.engine.precision import validated_computation

    result = validated_computation(lambda: mpmath.zeta(2))

    assert result.precision_digits == 50
    assert result.validation_precision == 100


def test_computation_time_recorded():
    """computation_time_ms should be > 0 in the result."""
    from riemann.engine.precision import validated_computation

    result = validated_computation(lambda: mpmath.zeta(2), dps=30)

    assert result.computation_time_ms > 0
