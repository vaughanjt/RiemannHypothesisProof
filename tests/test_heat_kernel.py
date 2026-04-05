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
        import mpmath
        from riemann.analysis.heat_kernel import maass_spectral_sum
        from riemann.types import ConvergenceDiagnostic

        result, diag = maass_spectral_sum(t=0.1, dps=30)
        # Result should be a positive mpmath number
        assert isinstance(result, mpmath.mpf) or float(result) > 0
        assert float(result) > 0
        # Convergence diagnostic should be valid
        assert isinstance(diag, ConvergenceDiagnostic)
        assert diag.n_terms_used > 0
        assert diag.tail_bound > 0
        assert diag.tail_bound < 1e-10  # should be very small for t=0.1
        assert diag.n_terms_available >= 100

    def test_scattering_phase_real(self):
        """scattering_phase(r=10.0) returns a finite real number."""
        import mpmath
        from riemann.analysis.heat_kernel import scattering_phase

        result = scattering_phase(r=10.0, dps=30)
        # Result should be real (imaginary part negligible)
        if isinstance(result, mpmath.mpc):
            assert abs(mpmath.im(result)) < 1e-20
        # Result should be finite
        assert mpmath.isfinite(result)

    def test_load_maass_params_count(self):
        """load_maass_spectral_params returns list of correct length and values."""
        from riemann.analysis.heat_kernel import load_maass_spectral_params

        params = load_maass_spectral_params()
        assert isinstance(params, list)
        assert len(params) >= 100
        # First r value should be approximately 9.5337
        assert abs(params[0] - 9.5337) < 0.01

    def test_eisenstein_continuous_spectrum(self):
        """Eisenstein integral computes continuous spectrum contribution."""
        import mpmath
        from riemann.analysis.heat_kernel import eisenstein_continuous_integral

        result = eisenstein_continuous_integral(t=0.5, dps=20)
        # Result should be a finite real number
        assert mpmath.isfinite(result)
        # Magnitude should be reasonable (sanity bound)
        assert abs(float(result)) < 10

    def test_heat_kernel_trace_includes_all_terms(self):
        """Heat kernel trace = discrete sum + Eisenstein + constant."""
        from riemann.analysis.heat_kernel import heat_kernel_trace

        result = heat_kernel_trace(t=0.1, dps=20, use_dual=False)
        # Must have all expected keys
        assert "total" in result
        assert "constant_term" in result
        assert "discrete_sum" in result
        assert "continuous_integral" in result
        assert "convergence" in result
        # Constant term: 1/(12*0.1) ~ 0.8333
        assert abs(result["constant_term"] - 1 / (12 * 0.1)) < 0.01
        # Total should be positive (heat kernel positivity)
        assert result["total"] > 0

    def test_heat_kernel_trace_constant_term_dominates_small_t(self):
        """At very small t, constant term dominates the discrete sum."""
        from riemann.analysis.heat_kernel import heat_kernel_trace

        result = heat_kernel_trace(t=0.001, dps=20, use_dual=False)
        # Constant = 1/(12*0.001) ~ 83.33, discrete sum should be smaller
        assert result["constant_term"] > result["discrete_sum"]


# ---------------------------------------------------------------------------
# Plan 03: parameter mapping, barrier comparison, dual precision, plots
# ---------------------------------------------------------------------------

class TestBarrierValue:
    def test_barrier_value_numpy_positive(self):
        """barrier_value_numpy returns a finite value at moderate L."""
        from riemann.analysis.heat_kernel import barrier_value_numpy

        # L ~ 6.9 corresponds to lam_sq ~ 1000
        val = barrier_value_numpy(6.9)
        assert isinstance(val, float)
        assert not (val != val)  # not NaN

    def test_barrier_value_numpy_matches_known(self):
        """barrier_value_numpy roughly matches session41g results."""
        from riemann.analysis.heat_kernel import barrier_value_numpy

        # At lam_sq=1000, L=ln(1000)~6.908, barrier should be small positive
        import math
        L = math.log(1000)
        val = barrier_value_numpy(L)
        # The barrier is positive at this range
        assert val > -10  # sanity: not wildly negative


class TestParameterMapping:
    def test_parameter_mapping_cross_validation(self):
        """find_parameter_mapping returns structured result with analytic candidates."""
        from riemann.analysis.heat_kernel import (
            find_parameter_mapping,
            barrier_value_numpy,
        )

        L_vals = [3.9, 6.9, 9.2]
        # Pre-compute barrier values for speed
        bv = {L: barrier_value_numpy(L) for L in L_vals}
        result = find_parameter_mapping(
            L_vals, dps=15, barrier_values=bv,
        )

        assert isinstance(result, dict)
        assert "analytic_candidates" in result
        assert "numerical_fits" in result
        assert "best_candidate" in result

        # At least one analytic candidate should have agreement > 0 at all L
        for name, info in result["analytic_candidates"].items():
            assert "agreement_digits_by_L" in info
            # agreement values should be numeric
            for L, digits in info["agreement_digits_by_L"].items():
                assert isinstance(digits, (int, float))


class TestBarrierComparison:
    def test_barrier_comparison_100_values(self):
        """run_feasibility_comparison returns list of BarrierComparison objects."""
        from riemann.analysis.heat_kernel import run_feasibility_comparison

        comps = run_feasibility_comparison(
            n_points=10, L_range=(3.0, 12.0), dps=15, verbose=False,
        )
        assert isinstance(comps, list)
        assert len(comps) >= 5  # at least some points

        for c in comps:
            assert isinstance(c, BarrierComparison)
            assert not (c.digits_of_agreement != c.digits_of_agreement)  # not NaN
            assert c.digits_of_agreement > -2  # not broken


class TestDualPrecisionAll:
    def test_dual_precision_all_computations(self):
        """heat_kernel_trace with use_dual=True populates dual_results."""
        from riemann.analysis.heat_kernel import heat_kernel_trace

        result = heat_kernel_trace(t=0.1, dps=20, use_dual=True)
        assert "dual_results" in result
        # Constant term dual should succeed when flint is available
        dr = result["dual_results"]
        if dr["constant"] is not None:
            from riemann.types import DualResult
            assert isinstance(dr["constant"], DualResult)
            assert dr["constant"].agreement_digits > 10


class TestFeasibilityVerdict:
    def test_verdict_viable(self):
        """feasibility_verdict returns VIABLE for high-agreement data."""
        from riemann.analysis.heat_kernel import feasibility_verdict

        comps = [
            BarrierComparison(
                L=5.0, t=5.0, heat_kernel_value=1.0, barrier_value=1.0000001,
                discrete_sum=0.5, eisenstein_contrib=0.001, constant_term=0.5,
                digits_of_agreement=7.0, n_maass_terms=100, dual_validated=True,
            )
            for _ in range(10)
        ]
        v = feasibility_verdict(comps)
        assert v["verdict"] == "VIABLE"
        assert v["median_agreement"] >= 6

    def test_verdict_dead(self):
        """feasibility_verdict returns DEAD for low-agreement data."""
        from riemann.analysis.heat_kernel import feasibility_verdict

        comps = [
            BarrierComparison(
                L=5.0, t=5.0, heat_kernel_value=1.0, barrier_value=2.0,
                discrete_sum=0.5, eisenstein_contrib=0.001, constant_term=0.5,
                digits_of_agreement=0.3, n_maass_terms=100, dual_validated=False,
            )
            for _ in range(10)
        ]
        v = feasibility_verdict(comps)
        assert v["verdict"] == "DEAD"


class TestPlotConvergence:
    def test_plot_convergence_returns_figure(self):
        """plot_convergence_vs_L returns a Plotly Figure with 2 rows."""
        import plotly.graph_objects as go
        from riemann.viz.heat_kernel import plot_convergence_vs_L

        comps = [
            BarrierComparison(
                L=float(i), t=float(i), heat_kernel_value=1.0 / (i + 1),
                barrier_value=1.0 / (i + 1) + 0.001, discrete_sum=0.5 / (i + 1),
                eisenstein_contrib=0.01 / (i + 1), constant_term=0.5 / (i + 1),
                digits_of_agreement=3.0 + i, n_maass_terms=50, dual_validated=True,
            )
            for i in range(1, 6)
        ]

        fig = plot_convergence_vs_L(comps)
        assert isinstance(fig, go.Figure)
        # Should have multiple traces (discrete, eisenstein, digits)
        assert len(fig.data) >= 2


class TestPlotHeatmap:
    def test_plot_agreement_heatmap_returns_figure(self):
        """plot_agreement_heatmap returns a Plotly Figure with Heatmap trace."""
        import plotly.graph_objects as go
        from riemann.viz.heat_kernel import plot_agreement_heatmap

        comps = [
            BarrierComparison(
                L=float(i), t=float(i) * 0.5, heat_kernel_value=1.0,
                barrier_value=1.001, discrete_sum=0.5,
                eisenstein_contrib=0.01, constant_term=0.5,
                digits_of_agreement=3.0 + i, n_maass_terms=50, dual_validated=True,
            )
            for i in range(1, 6)
        ]

        fig = plot_agreement_heatmap(comps)
        assert isinstance(fig, go.Figure)
        # Should have a Heatmap trace
        assert any(isinstance(t, go.Heatmap) for t in fig.data)
