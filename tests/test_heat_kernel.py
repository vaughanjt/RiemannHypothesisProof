"""Tests for Phase 5: Heat Kernel Feasibility Gate.

Test scaffolds for HEAT-01 through HEAT-04 requirements.
- test_dual_compute_basic: dual_compute on exp(-1) gives agreement > 10 digits
- test_dual_compute_flags_disagreement: intentionally bad flint func triggers flag
- Stubs for Plans 02 and 03 (Maass spectral sum, heat kernel trace, Eisenstein,
  parameter mapping, barrier comparison, dual precision on all computations)
"""
from __future__ import annotations

import pytest

from riemann.types import DualResult, BarrierComparison, ConvergenceDiagnostic


# ---------------------------------------------------------------------------
# Dual-precision backend tests (Plan 01)
# ---------------------------------------------------------------------------

class TestDualComputeBasic:
    def test_dual_compute_basic(self):
        """dual_compute on exp(-1) returns DualResult with agreement_digits > 10."""
        import mpmath
        from riemann.engine.dual_precision import dual_compute

        result = dual_compute(
            func_mpmath=lambda: mpmath.exp(-1),
            func_flint=lambda prec: __import__('flint').arb(-1).exp(),
            dps=50,
            label="exp(-1)",
        )
        assert isinstance(result, DualResult)
        assert result.agreement_digits > 10
        assert result.label == "exp(-1)"
        assert not result.flagged

    def test_dual_compute_flags_disagreement(self):
        """dual_compute with intentionally wrong flint func flags disagreement."""
        import mpmath
        from riemann.engine.dual_precision import dual_compute

        # Use low dps=15 so catastrophic threshold (dps-20=-5) never triggers,
        # but flag threshold (dps-10=5) does trigger for a completely wrong value.
        result = dual_compute(
            func_mpmath=lambda: mpmath.exp(-1),
            func_flint=lambda prec: __import__('flint').arb(0),  # wrong!
            dps=15,
            label="bad_flint",
        )
        assert isinstance(result, DualResult)
        assert result.flagged is True
        assert result.agreement_digits < 1


# ---------------------------------------------------------------------------
# Plan 02 stubs: Maass spectral sum, heat kernel trace, Eisenstein
# ---------------------------------------------------------------------------

class TestMaassSpectralSum:
    def test_maass_spectral_sum_convergence(self):
        """Spectral sum over Maass eigenvalues converges with diagnostics."""
        pytest.skip("Plan 02")

    def test_heat_kernel_trace_includes_all_terms(self):
        """Heat kernel trace = discrete sum + Eisenstein + constant."""
        pytest.skip("Plan 02")

    def test_eisenstein_continuous_spectrum(self):
        """Eisenstein integral computes continuous spectrum contribution."""
        pytest.skip("Plan 02")


# ---------------------------------------------------------------------------
# Plan 02-03 stubs: parameter mapping, barrier comparison, dual precision
# ---------------------------------------------------------------------------

class TestParameterMapping:
    def test_parameter_mapping_cross_validation(self):
        """Analytic t(L) formula matches numerical fit."""
        pytest.skip("Plan 02-03")


class TestBarrierComparison:
    def test_barrier_comparison_100_values(self):
        """K(t(L)) agrees with B(L) at 100+ L values to 6+ digits."""
        pytest.skip("Plan 03")


class TestDualPrecisionAll:
    def test_dual_precision_all_computations(self):
        """Every computation runs in both mpmath and python-flint."""
        pytest.skip("Plan 03")
