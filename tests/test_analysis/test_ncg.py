"""Tests for noncommutative geometry module (Bost-Connes system).

Tests cover:
- bost_connes_partition: convergence to zeta(beta) for beta > 1
- bost_connes_partition: divergence guard for beta <= 1
- bost_connes_kms_values: probability distribution summing to 1.0
- bost_connes_kms_values: ordering (first element is largest)
- phase_transition_scan: entropy decrease with increasing beta
- phase_transition_scan: dict structure with expected keys
- compute_bost_connes: BostConnesResult fields and zeta comparison
- BostConnesResult: dataclass fields present
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# BostConnesResult dataclass
# ---------------------------------------------------------------------------

class TestBostConnesResult:
    def test_has_required_fields(self):
        from riemann.analysis.ncg import BostConnesResult
        result = BostConnesResult(
            beta=2.0,
            partition_value=1.6449,
            kms_values=np.array([0.5, 0.3, 0.2]),
            zeta_comparison=0.001,
            metadata={"n_max": 1000},
        )
        assert result.beta == 2.0
        assert result.partition_value == 1.6449
        assert isinstance(result.kms_values, np.ndarray)
        assert result.zeta_comparison == 0.001
        assert result.metadata == {"n_max": 1000}


# ---------------------------------------------------------------------------
# bost_connes_partition
# ---------------------------------------------------------------------------

class TestBostConnesPartition:
    def test_matches_zeta_2(self):
        """Z(2) = sum n^{-2} should match zeta(2) = pi^2/6 to 4+ decimal places."""
        from riemann.analysis.ncg import bost_connes_partition
        import mpmath
        result = bost_connes_partition(beta=2.0, n_max=1000)
        expected = float(mpmath.zeta(2))
        assert abs(result - expected) < 1e-4, (
            f"Partition function {result} not within 1e-4 of zeta(2)={expected}"
        )

    def test_matches_zeta_3(self):
        """Z(3) should match zeta(3) to 4+ decimal places."""
        from riemann.analysis.ncg import bost_connes_partition
        import mpmath
        result = bost_connes_partition(beta=3.0, n_max=1000)
        expected = float(mpmath.zeta(3))
        assert abs(result - expected) < 1e-4, (
            f"Partition function {result} not within 1e-4 of zeta(3)={expected}"
        )

    def test_higher_n_max_improves_accuracy(self):
        """More terms should give better approximation."""
        from riemann.analysis.ncg import bost_connes_partition
        import mpmath
        expected = float(mpmath.zeta(2))
        result_100 = bost_connes_partition(beta=2.0, n_max=100)
        result_10000 = bost_connes_partition(beta=2.0, n_max=10000)
        err_100 = abs(result_100 - expected)
        err_10000 = abs(result_10000 - expected)
        assert err_10000 < err_100, (
            f"n_max=10000 error ({err_10000}) not less than n_max=100 error ({err_100})"
        )

    def test_raises_for_beta_le_1(self):
        """beta <= 1 causes divergence; should raise ValueError."""
        from riemann.analysis.ncg import bost_connes_partition
        with pytest.raises(ValueError, match="beta.*>.*1"):
            bost_connes_partition(beta=0.5)

    def test_raises_for_beta_equal_1(self):
        """beta = 1 is the harmonic series; should raise ValueError."""
        from riemann.analysis.ncg import bost_connes_partition
        with pytest.raises(ValueError, match="beta.*>.*1"):
            bost_connes_partition(beta=1.0)

    def test_returns_float(self):
        from riemann.analysis.ncg import bost_connes_partition
        result = bost_connes_partition(beta=2.0, n_max=100)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# bost_connes_kms_values
# ---------------------------------------------------------------------------

class TestBostConnesKMSValues:
    def test_sums_to_one(self):
        """KMS state probabilities must sum to 1.0."""
        from riemann.analysis.ncg import bost_connes_kms_values
        kms = bost_connes_kms_values(beta=2.0, n_max=50)
        assert abs(np.sum(kms) - 1.0) < 1e-10, (
            f"KMS values sum to {np.sum(kms)}, expected 1.0"
        )

    def test_shape(self):
        """Should return array of shape (n_max,)."""
        from riemann.analysis.ncg import bost_connes_kms_values
        kms = bost_connes_kms_values(beta=2.0, n_max=50)
        assert kms.shape == (50,)

    def test_first_element_is_largest(self):
        """1^{-beta}/Z is the largest term since n^{-beta} is decreasing."""
        from riemann.analysis.ncg import bost_connes_kms_values
        kms = bost_connes_kms_values(beta=2.0, n_max=50)
        assert kms[0] == np.max(kms), (
            f"First element {kms[0]} is not the largest (max={np.max(kms)})"
        )

    def test_all_positive(self):
        """All KMS values should be positive."""
        from riemann.analysis.ncg import bost_connes_kms_values
        kms = bost_connes_kms_values(beta=2.0, n_max=50)
        assert np.all(kms > 0), "All KMS values should be positive"


# ---------------------------------------------------------------------------
# phase_transition_scan
# ---------------------------------------------------------------------------

class TestPhaseTransitionScan:
    def test_returns_dict_with_expected_keys(self):
        from riemann.analysis.ncg import phase_transition_scan
        result = phase_transition_scan(beta_range=(1.5, 3.0), n_points=10)
        assert isinstance(result, dict)
        for key in ("betas", "partition_values", "kms_entropy", "d_entropy_d_beta"):
            assert key in result, f"Missing key: {key}"

    def test_array_lengths_match_n_points(self):
        from riemann.analysis.ncg import phase_transition_scan
        n_points = 15
        result = phase_transition_scan(beta_range=(1.5, 3.0), n_points=n_points)
        assert len(result["betas"]) == n_points
        assert len(result["partition_values"]) == n_points
        assert len(result["kms_entropy"]) == n_points

    def test_entropy_decreases_with_beta(self):
        """Higher beta = lower temperature = more concentrated distribution = lower entropy."""
        from riemann.analysis.ncg import phase_transition_scan
        result = phase_transition_scan(beta_range=(1.5, 5.0), n_points=20, n_max=100)
        entropy = result["kms_entropy"]
        # Overall trend: entropy should decrease
        # Compare first quarter to last quarter
        q1_mean = np.mean(entropy[:5])
        q4_mean = np.mean(entropy[-5:])
        assert q4_mean < q1_mean, (
            f"Entropy should decrease: first quarter mean {q1_mean:.4f} "
            f"should be > last quarter mean {q4_mean:.4f}"
        )

    def test_partition_values_decrease_with_beta(self):
        """zeta(beta) is decreasing for beta > 1."""
        from riemann.analysis.ncg import phase_transition_scan
        result = phase_transition_scan(beta_range=(1.5, 5.0), n_points=10)
        pv = result["partition_values"]
        # Each value should be >= next value
        assert np.all(np.diff(pv) <= 0), "Partition values should decrease with beta"


# ---------------------------------------------------------------------------
# compute_bost_connes
# ---------------------------------------------------------------------------

class TestComputeBostConnes:
    def test_returns_bost_connes_result(self):
        from riemann.analysis.ncg import compute_bost_connes, BostConnesResult
        result = compute_bost_connes(beta=2.0, n_max=500)
        assert isinstance(result, BostConnesResult)

    def test_zeta_comparison_is_small(self):
        """Comparison with mpmath zeta should be small for large n_max."""
        from riemann.analysis.ncg import compute_bost_connes
        result = compute_bost_connes(beta=2.0, n_max=1000)
        assert result.zeta_comparison < 0.01, (
            f"zeta_comparison {result.zeta_comparison} should be < 0.01"
        )

    def test_kms_values_sum_to_one(self):
        from riemann.analysis.ncg import compute_bost_connes
        result = compute_bost_connes(beta=3.0, n_max=100)
        assert abs(np.sum(result.kms_values) - 1.0) < 1e-10
