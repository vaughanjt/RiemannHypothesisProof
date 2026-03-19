"""Tests for modular forms computation module.

Tests cover:
- eisenstein_series: Fourier coefficients of E_k for known weights
- compute_q_expansion: Ramanujan Delta function q-expansion
- hecke_eigenvalues: extraction of Hecke eigenvalues from eigenforms
- ModularFormResult: dataclass fields and construction
- Input validation: rejection of invalid weight/level
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# ModularFormResult dataclass
# ---------------------------------------------------------------------------

class TestModularFormResult:
    def test_has_required_fields(self):
        from riemann.analysis.modular_forms import ModularFormResult

        result = ModularFormResult(
            weight=12,
            level=1,
            coefficients=[0.0, 1.0, -24.0],
            n_terms=3,
            metadata={"source": "test"},
        )
        assert result.weight == 12
        assert result.level == 1
        assert result.coefficients == [0.0, 1.0, -24.0]
        assert result.n_terms == 3
        assert result.metadata == {"source": "test"}

    def test_metadata_default_empty_dict(self):
        """ModularFormResult should be constructible with all required fields."""
        from riemann.analysis.modular_forms import ModularFormResult

        result = ModularFormResult(
            weight=4, level=1, coefficients=[1.0], n_terms=1, metadata={},
        )
        assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# eisenstein_series
# ---------------------------------------------------------------------------

class TestEisensteinSeries:
    def test_e4_constant_term_is_one(self):
        """The normalized Eisenstein series E_4 has constant term 1."""
        from riemann.analysis.modular_forms import eisenstein_series

        coeffs = eisenstein_series(k=4, n_terms=1)
        assert abs(coeffs[0] - 1.0) < 1e-10

    def test_e4_second_coeff_is_240(self):
        """E_4 = 1 + 240*q + 2160*q^2 + ..."""
        from riemann.analysis.modular_forms import eisenstein_series

        coeffs = eisenstein_series(k=4, n_terms=2)
        assert abs(coeffs[1] - 240.0) < 1e-10

    def test_e4_third_coeff_is_2160(self):
        """E_4 third Fourier coefficient is 2160."""
        from riemann.analysis.modular_forms import eisenstein_series

        coeffs = eisenstein_series(k=4, n_terms=3)
        assert abs(coeffs[2] - 2160.0) < 1e-10

    def test_e6_second_coeff_is_minus_504(self):
        """E_6 = 1 - 504*q - 16632*q^2 + ..."""
        from riemann.analysis.modular_forms import eisenstein_series

        coeffs = eisenstein_series(k=6, n_terms=2)
        assert abs(coeffs[1] - (-504.0)) < 1e-10

    def test_e6_third_coeff_is_minus_16632(self):
        """E_6 third coefficient."""
        from riemann.analysis.modular_forms import eisenstein_series

        coeffs = eisenstein_series(k=6, n_terms=3)
        assert abs(coeffs[2] - (-16632.0)) < 1e-10

    def test_returns_correct_number_of_terms(self):
        from riemann.analysis.modular_forms import eisenstein_series

        coeffs = eisenstein_series(k=4, n_terms=10)
        assert len(coeffs) == 10

    def test_rejects_odd_weight(self):
        from riemann.analysis.modular_forms import eisenstein_series

        with pytest.raises(ValueError, match="even.*>= 4"):
            eisenstein_series(k=3)

    def test_rejects_weight_less_than_4(self):
        from riemann.analysis.modular_forms import eisenstein_series

        with pytest.raises(ValueError, match="even.*>= 4"):
            eisenstein_series(k=2)


# ---------------------------------------------------------------------------
# compute_q_expansion (Ramanujan Delta function)
# ---------------------------------------------------------------------------

class TestComputeQExpansion:
    def test_delta_first_coeff_is_zero(self):
        """Delta function has a_0 = 0 (cusp form vanishes at infinity)."""
        from riemann.analysis.modular_forms import compute_q_expansion

        result = compute_q_expansion(weight=12, level=1, n_terms=5)
        assert abs(result.coefficients[0]) < 1e-8

    def test_delta_second_coeff_is_one(self):
        """The normalized Delta function has a_1 = 1 (tau(1) = 1)."""
        from riemann.analysis.modular_forms import compute_q_expansion

        result = compute_q_expansion(weight=12, level=1, n_terms=5)
        assert abs(result.coefficients[1] - 1.0) < 1e-6

    def test_delta_third_coeff_is_minus_24(self):
        """Ramanujan's tau function: tau(2) = -24."""
        from riemann.analysis.modular_forms import compute_q_expansion

        result = compute_q_expansion(weight=12, level=1, n_terms=5)
        assert abs(result.coefficients[2] - (-24.0)) < 1e-4

    def test_returns_modular_form_result(self):
        from riemann.analysis.modular_forms import (
            compute_q_expansion,
            ModularFormResult,
        )

        result = compute_q_expansion(weight=12, level=1, n_terms=5)
        assert isinstance(result, ModularFormResult)
        assert result.weight == 12
        assert result.level == 1
        assert result.n_terms == 5

    def test_low_weight_returns_eisenstein(self):
        """Weight < 12 at level 1 has no cusp form; returns Eisenstein series."""
        from riemann.analysis.modular_forms import compute_q_expansion

        result = compute_q_expansion(weight=4, level=1, n_terms=3)
        # For weight 4 at level 1, the space of cusp forms is trivial,
        # so we return E_4
        assert abs(result.coefficients[0] - 1.0) < 1e-10
        assert abs(result.coefficients[1] - 240.0) < 1e-10


# ---------------------------------------------------------------------------
# hecke_eigenvalues
# ---------------------------------------------------------------------------

class TestHeckeEigenvalues:
    def test_delta_tau_2_is_minus_24(self):
        """For the Delta function, the Hecke eigenvalue at p=2 is tau(2)=-24."""
        from riemann.analysis.modular_forms import hecke_eigenvalues

        evals = hecke_eigenvalues(weight=12, level=1, primes=[2])
        assert abs(evals[2] - (-24.0)) < 1e-4

    def test_delta_tau_3_is_252(self):
        """tau(3) = 252."""
        from riemann.analysis.modular_forms import hecke_eigenvalues

        evals = hecke_eigenvalues(weight=12, level=1, primes=[3])
        assert abs(evals[3] - 252.0) < 1e-2

    def test_delta_tau_5_is_4830(self):
        """tau(5) = 4830 (OEIS A000594: verified via product formula)."""
        from riemann.analysis.modular_forms import hecke_eigenvalues

        evals = hecke_eigenvalues(weight=12, level=1, primes=[5])
        assert abs(evals[5] - 4830.0) < 1e-1

    def test_returns_dict(self):
        from riemann.analysis.modular_forms import hecke_eigenvalues

        evals = hecke_eigenvalues(weight=12, level=1, primes=[2, 3, 5])
        assert isinstance(evals, dict)
        assert set(evals.keys()) == {2, 3, 5}

    def test_default_primes(self):
        """Default primes list should be [2, 3, 5, 7, 11, 13]."""
        from riemann.analysis.modular_forms import hecke_eigenvalues

        evals = hecke_eigenvalues(weight=12, level=1)
        assert set(evals.keys()) == {2, 3, 5, 7, 11, 13}

    def test_delta_tau_7_is_minus_16744(self):
        """tau(7) = -16744 (OEIS A000594: verified via product formula)."""
        from riemann.analysis.modular_forms import hecke_eigenvalues

        evals = hecke_eigenvalues(weight=12, level=1, primes=[7])
        assert abs(evals[7] - (-16744.0)) < 1.0
