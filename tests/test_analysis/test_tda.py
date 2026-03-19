"""Tests for topological data analysis module.

Tests cover:
- Persistent homology computation via ripser
- PersistenceResult structure validation
- Circle detection (H_1 loop)
- Random noise (no persistent features)
- Persistence summary analysis
- Cross-embedding diagram comparison (bottleneck distance)
- Input validation
"""
import numpy as np
import pytest

from riemann.analysis.tda import (
    PersistenceResult,
    compare_persistence_diagrams,
    compute_persistence,
    persistence_summary,
)


# ---------------------------------------------------------------------------
# Fixtures: known point clouds
# ---------------------------------------------------------------------------


@pytest.fixture
def circle_points():
    """100 points sampled uniformly from a circle (has 1 persistent H_1 loop)."""
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    # Add small noise
    x += rng.normal(0, 0.05, 100)
    y += rng.normal(0, 0.05, 100)
    return np.column_stack([x, y])


@pytest.fixture
def random_points():
    """100 random points in [0,1]^2 (no persistent topological features)."""
    rng = np.random.default_rng(123)
    return rng.uniform(0, 1, (100, 2))


@pytest.fixture
def two_clusters():
    """Two well-separated clusters (2 components = 2 H_0 features)."""
    rng = np.random.default_rng(7)
    c1 = rng.normal(loc=[0, 0], scale=0.1, size=(50, 2))
    c2 = rng.normal(loc=[5, 5], scale=0.1, size=(50, 2))
    return np.vstack([c1, c2])


# ---------------------------------------------------------------------------
# compute_persistence
# ---------------------------------------------------------------------------


class TestComputePersistence:
    def test_returns_persistence_result(self, circle_points):
        result = compute_persistence(circle_points, max_dim=1)
        assert isinstance(result, PersistenceResult)

    def test_circle_has_persistent_h1(self, circle_points):
        """A circle should have exactly 1 persistent H_1 feature."""
        result = compute_persistence(circle_points, max_dim=1)
        # There should be H_1 features
        assert 1 in result.num_features
        assert result.num_features[1] >= 1
        # The most persistent H_1 feature should have large lifetime
        h1_diagram = result.diagrams[1]
        if len(h1_diagram) > 0:
            lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
            # Filter to finite lifetimes
            finite_mask = np.isfinite(lifetimes)
            finite_lifetimes = lifetimes[finite_mask]
            if len(finite_lifetimes) > 0:
                assert np.max(finite_lifetimes) > 0.5  # The loop should be significant

    def test_random_no_persistent_h1(self, random_points):
        """Random uniform points should not have persistent H_1 features."""
        result = compute_persistence(random_points, max_dim=1)
        if 1 in result.num_features and result.num_features[1] > 0:
            h1_diagram = result.diagrams[1]
            lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
            finite_mask = np.isfinite(lifetimes)
            finite_lifetimes = lifetimes[finite_mask]
            # All H_1 features should be short-lived
            if len(finite_lifetimes) > 0:
                assert np.max(finite_lifetimes) < 0.5

    def test_metadata_populated(self, circle_points):
        result = compute_persistence(circle_points, max_dim=1)
        assert "n_points" in result.metadata
        assert "max_dim" in result.metadata
        assert result.metadata["n_points"] == 100

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            compute_persistence(np.array([]))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            compute_persistence(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# persistence_summary
# ---------------------------------------------------------------------------


class TestPersistenceSummary:
    def test_returns_dict(self, circle_points):
        result = compute_persistence(circle_points, max_dim=1)
        summary = persistence_summary(result)
        assert isinstance(summary, dict)

    def test_summary_has_expected_keys(self, circle_points):
        result = compute_persistence(circle_points, max_dim=1)
        summary = persistence_summary(result)
        assert "dominant_dimension" in summary
        assert "max_lifetime" in summary
        assert "n_significant" in summary
        assert "feature_counts" in summary

    def test_circle_dominant_dimension_is_1(self, circle_points):
        """For a circle, the dominant dimension (excluding H_0) should be 1."""
        result = compute_persistence(circle_points, max_dim=1)
        summary = persistence_summary(result)
        assert summary["dominant_dimension"] == 1

    def test_max_lifetime_positive(self, circle_points):
        result = compute_persistence(circle_points, max_dim=1)
        summary = persistence_summary(result)
        assert summary["max_lifetime"] > 0


# ---------------------------------------------------------------------------
# compare_persistence_diagrams
# ---------------------------------------------------------------------------


class TestComparePersistenceDiagrams:
    def test_same_cloud_distance_near_zero(self, circle_points):
        """Comparing a point cloud with itself should give distance near 0."""
        result1 = compute_persistence(circle_points, max_dim=1)
        result2 = compute_persistence(circle_points, max_dim=1)
        comparison = compare_persistence_diagrams(result1, result2, dimension=1)
        assert "bottleneck_distance" in comparison
        assert comparison["bottleneck_distance"] < 0.01

    def test_different_clouds_positive_distance(self, circle_points, random_points):
        """Circle vs random should have positive bottleneck distance."""
        result1 = compute_persistence(circle_points, max_dim=1)
        result2 = compute_persistence(random_points, max_dim=1)
        comparison = compare_persistence_diagrams(result1, result2, dimension=1)
        assert comparison["bottleneck_distance"] > 0.1

    def test_comparison_has_feature_counts(self, circle_points, random_points):
        result1 = compute_persistence(circle_points, max_dim=1)
        result2 = compute_persistence(random_points, max_dim=1)
        comparison = compare_persistence_diagrams(result1, result2, dimension=1)
        assert "n_features_1" in comparison
        assert "n_features_2" in comparison
        assert "dimension" in comparison
