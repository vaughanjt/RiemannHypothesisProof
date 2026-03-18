"""Tests for COMP-04: Stress-test and validation framework.

Tests the stress_test function which re-runs computations at escalating
precision levels to distinguish genuine patterns from numerical artifacts.
"""
import time

import mpmath
import pytest

from riemann.types import ComputationResult, PrecisionError


class TestStressTestBasic:
    """Basic stress_test functionality: runs at multiple precisions, returns StressTestResult."""

    def test_stress_test_default_levels(self, default_precision):
        """stress_test runs at dps=50, 100, 200 and returns StressTestResult."""
        from riemann.engine.validation import stress_test, StressTestResult

        result = stress_test(lambda: mpmath.zeta(2))
        assert isinstance(result, StressTestResult)
        assert len(result.results) == 3
        assert result.dps_levels == [50, 100, 200]

    def test_stress_test_custom_levels(self, default_precision):
        """stress_test with custom dps_levels=[30, 50, 80] uses those levels."""
        from riemann.engine.validation import stress_test

        result = stress_test(
            lambda: mpmath.zeta(2),
            dps_levels=[30, 50, 80],
        )
        assert len(result.results) == 3
        assert result.dps_levels == [30, 50, 80]

    def test_stress_test_result_fields(self, default_precision):
        """StressTestResult contains results list, consistent flag, max_deviation, metadata."""
        from riemann.engine.validation import stress_test, StressTestResult

        result = stress_test(
            lambda: mpmath.zeta(2),
            pattern_description="zeta(2) = pi^2/6",
        )
        assert hasattr(result, 'results')
        assert hasattr(result, 'consistent')
        assert hasattr(result, 'max_deviation')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'dps_levels')
        assert hasattr(result, 'total_time_ms')
        assert result.pattern_description == "zeta(2) = pi^2/6"
        assert isinstance(result.consistent, bool)
        assert isinstance(result.max_deviation, float)


class TestStressTestConsistency:
    """Consistency detection: genuine vs fake patterns."""

    def test_genuine_pattern_consistent(self, default_precision):
        """stress_test with genuine pattern (zeta(2) = pi^2/6) returns consistent=True."""
        from riemann.engine.validation import stress_test

        result = stress_test(
            lambda: mpmath.zeta(2),
            dps_levels=[30, 50, 100],
            pattern_description="zeta(2) = pi^2/6",
        )
        assert result.consistent is True
        # All results should be validated
        for r in result.results:
            assert isinstance(r, ComputationResult)
            assert r.validated is True

    def test_fake_pattern_inconsistent(self, default_precision):
        """stress_test with fake pattern (changes at higher precision) returns consistent=False."""
        from riemann.engine.validation import stress_test

        call_count = [0]

        def unstable_func():
            """Returns different values based on call count.

            validated_computation calls func twice per level (P and 2P).
            Calls 1-4 (levels 1 & 2) return 3.14159.
            Calls 5+ (level 3) return 2.71828.

            This passes P-vs-2P within each level but the cross-level
            comparison in stress_test detects the value changed.
            """
            call_count[0] += 1
            if call_count[0] > 4:
                return mpmath.mpf("2.71828")
            return mpmath.mpf("3.14159")

        result = stress_test(
            unstable_func,
            dps_levels=[30, 50, 100],
        )
        assert result.consistent is False
        assert result.max_deviation > 0


class TestStressTestPredicate:
    """Predicate-based consistency checking."""

    def test_predicate_all_true(self, default_precision):
        """stress_test with predicate that always returns True -> consistent."""
        from riemann.engine.validation import stress_test

        result = stress_test(
            lambda: mpmath.zeta(2),
            dps_levels=[30, 50],
            predicate=lambda r: abs(r.value - mpmath.pi ** 2 / 6) < mpmath.power(10, -10),
        )
        assert result.consistent is True
        assert result.predicate_results == [True, True]

    def test_predicate_fails_at_some_level(self, default_precision):
        """stress_test with predicate that fails makes result inconsistent."""
        from riemann.engine.validation import stress_test

        # Predicate that fails for the last result: checks if value < 1 (zeta(2) ~ 1.64)
        result = stress_test(
            lambda: mpmath.zeta(2),
            dps_levels=[30, 50],
            predicate=lambda r: float(r.value) < 1.0,  # Always fails since zeta(2) > 1
        )
        assert result.consistent is False
        assert result.predicate_results == [False, False]


class TestStressTestTiming:
    """Timing metadata in stress test results."""

    def test_records_computation_time(self, default_precision):
        """stress_test records total computation time."""
        from riemann.engine.validation import stress_test

        result = stress_test(
            lambda: mpmath.zeta(2),
            dps_levels=[30, 50],
        )
        assert result.total_time_ms > 0


class TestStressTestAlwaysValidate:
    """Integration with always-validate (P-vs-2P) pattern."""

    def test_always_validate_catches_precision_error(self, default_precision):
        """validated_computation inside stress_test catches precision collapse."""
        from riemann.engine.validation import stress_test

        def precision_collapsing_func():
            """Function that has genuine precision collapse (P vs 2P disagree).

            Returns wildly different values at different precisions.
            This should trigger PrecisionError inside validated_computation.
            """
            dps = mpmath.mp.dps
            # Return a value based on precision that doesn't converge
            return mpmath.mpf(1) / (dps * dps)

        with pytest.raises(PrecisionError):
            stress_test(
                precision_collapsing_func,
                dps_levels=[50, 100, 200],
            )
