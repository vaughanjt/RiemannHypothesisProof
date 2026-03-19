"""Tests for the trace formula module (Weil explicit formula and Chebyshev psi).

Tests cover:
- TraceFormulaResult dataclass structure
- chebyshev_psi_exact: known values of the Chebyshev psi function
- weil_explicit_psi: convergence toward exact value with more zeros
- explicit_formula_terms: returns list of (n_terms, psi_approx) pairs
- compute_trace_formula: convenience wrapper returning TraceFormulaResult
"""
import math

import numpy as np
import pytest


# First 20 non-trivial zeta zero imaginary parts (positive)
KNOWN_ZEROS = [
    14.134725141734693,
    21.022039638771555,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
    48.005150881167159,
    49.773832477672302,
    52.970321477714460,
    56.446247697063394,
    59.347044002602353,
    60.831778524609809,
    65.112544048081606,
    67.079810529494173,
    69.546401711173979,
    72.067157674481907,
    75.704690699083933,
    77.144840068874805,
]


class TestTraceFormulaResult:
    def test_dataclass_fields(self):
        """TraceFormulaResult has x, psi_exact, psi_approx, n_terms, relative_error, metadata."""
        from riemann.analysis.trace_formula import TraceFormulaResult

        result = TraceFormulaResult(
            x=100.0,
            psi_exact=94.0,
            psi_approx=95.0,
            n_terms=10,
            relative_error=0.01,
            metadata={},
        )
        assert result.x == 100.0
        assert result.psi_exact == 94.0
        assert result.psi_approx == 95.0
        assert result.n_terms == 10
        assert result.relative_error == 0.01
        assert result.metadata == {}


class TestChebyshevPsiExact:
    def test_psi_10(self):
        """psi(10) = sum Lambda(n) for n=2..10.

        Prime powers up to 10: 2,3,4=2^2,5,7,8=2^3,9=3^2
        Lambda(2)=log(2), Lambda(3)=log(3), Lambda(4)=log(2), Lambda(5)=log(5),
        Lambda(7)=log(7), Lambda(8)=log(2), Lambda(9)=log(3)
        psi(10) = 3*log(2) + 2*log(3) + log(5) + log(7) approx 7.832
        """
        from riemann.analysis.trace_formula import chebyshev_psi_exact

        expected = 3 * math.log(2) + 2 * math.log(3) + math.log(5) + math.log(7)
        result = chebyshev_psi_exact(10.0)
        assert abs(result - expected) < 0.01, (
            f"psi(10) = {result}, expected {expected}"
        )

    def test_psi_1(self):
        """psi(1) = 0 (no prime powers <= 1)."""
        from riemann.analysis.trace_formula import chebyshev_psi_exact

        assert chebyshev_psi_exact(1.0) == 0.0

    def test_psi_2(self):
        """psi(2) = log(2)."""
        from riemann.analysis.trace_formula import chebyshev_psi_exact

        assert abs(chebyshev_psi_exact(2.0) - math.log(2)) < 1e-10

    def test_psi_100(self):
        """psi(100) should be close to 100 (prime number theorem: psi(x) ~ x)."""
        from riemann.analysis.trace_formula import chebyshev_psi_exact

        result = chebyshev_psi_exact(100.0)
        # psi(100) is approximately 94.01 (actual sum of von Mangoldt)
        assert 90.0 < result < 100.0, f"psi(100) = {result}, expected ~94"


class TestWeilExplicitPsi:
    def test_basic_approximation(self):
        """weil_explicit_psi with 20 zeros should give a finite float for x=100."""
        from riemann.analysis.trace_formula import weil_explicit_psi

        result = weil_explicit_psi(100.0, KNOWN_ZEROS, n_terms=20)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_convergence_with_more_terms(self):
        """More terms should give a closer approximation to psi_exact.

        The explicit formula converges slowly and can oscillate with few zeros.
        We test that using 1 term vs 20 terms shows meaningful improvement,
        since single-term error is always large.
        """
        from riemann.analysis.trace_formula import chebyshev_psi_exact, weil_explicit_psi

        x = 50.0
        psi_exact = chebyshev_psi_exact(x)

        approx_1 = weil_explicit_psi(x, KNOWN_ZEROS, n_terms=1)
        approx_20 = weil_explicit_psi(x, KNOWN_ZEROS, n_terms=20)

        error_1 = abs(approx_1 - psi_exact)
        error_20 = abs(approx_20 - psi_exact)

        # 20 zeros should outperform 1 zero significantly
        assert error_20 < error_1, (
            f"Expected 20 terms to beat 1 term. "
            f"error_1={error_1:.4f}, error_20={error_20:.4f}"
        )


class TestExplicitFormulaTerms:
    def test_returns_list_of_tuples(self):
        """explicit_formula_terms returns a list of (n_terms, psi_approx) tuples."""
        from riemann.analysis.trace_formula import explicit_formula_terms

        result = explicit_formula_terms(50.0, KNOWN_ZEROS, max_terms=20)
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            n_terms, psi_approx = item
            assert isinstance(n_terms, int)
            assert isinstance(psi_approx, float)

    def test_powers_of_two_progression(self):
        """Terms list should include powers of 2 up to max_terms."""
        from riemann.analysis.trace_formula import explicit_formula_terms

        result = explicit_formula_terms(50.0, KNOWN_ZEROS, max_terms=16)
        n_terms_list = [t[0] for t in result]
        # Should include 1, 2, 4, 8, 16
        for k in [1, 2, 4, 8, 16]:
            assert k in n_terms_list, f"Expected {k} in n_terms_list: {n_terms_list}"


class TestComputeTraceFormula:
    def test_returns_trace_formula_result(self):
        """compute_trace_formula returns a TraceFormulaResult."""
        from riemann.analysis.trace_formula import TraceFormulaResult, compute_trace_formula

        result = compute_trace_formula(100.0, KNOWN_ZEROS, n_terms=10)
        assert isinstance(result, TraceFormulaResult)
        assert result.x == 100.0
        assert result.n_terms == 10
        assert isinstance(result.psi_exact, float)
        assert isinstance(result.psi_approx, float)
        assert isinstance(result.relative_error, float)

    def test_relative_error_computed(self):
        """relative_error = |psi_approx - psi_exact| / |psi_exact|."""
        from riemann.analysis.trace_formula import compute_trace_formula

        result = compute_trace_formula(100.0, KNOWN_ZEROS, n_terms=20)
        expected_error = abs(result.psi_approx - result.psi_exact) / abs(result.psi_exact)
        assert abs(result.relative_error - expected_error) < 1e-10
