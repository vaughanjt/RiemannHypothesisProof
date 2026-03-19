"""Tests for all projection methods including Hopf fibration."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_5d():
    """100 x 5 random data for standard projections."""
    return np.random.default_rng(42).standard_normal((100, 5))


@pytest.fixture
def data_4d():
    """100 x 4 random data for Hopf fibration."""
    return np.random.default_rng(42).standard_normal((100, 4))


# ---------------------------------------------------------------------------
# ProjectionResult dataclass
# ---------------------------------------------------------------------------

class TestProjectionResult:
    def test_is_dataclass(self):
        from dataclasses import fields
        from riemann.viz.projection import ProjectionResult
        f_names = [f.name for f in fields(ProjectionResult)]
        assert "coordinates" in f_names
        assert "method" in f_names
        assert "source_dim" in f_names
        assert "target_dim" in f_names
        assert "metadata" in f_names


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class TestProjectPCA:
    def test_correct_target_dim(self, data_5d):
        from riemann.viz.projection import project_pca
        result = project_pca(data_5d, n_components=3)
        assert result.coordinates.shape == (100, 3)
        assert result.target_dim == 3

    def test_variance_explained_in_metadata(self, data_5d):
        from riemann.viz.projection import project_pca
        result = project_pca(data_5d, n_components=3)
        assert "variance_explained" in result.metadata
        ve = result.metadata["variance_explained"]
        assert len(ve) == 3

    def test_variance_explained_sums_to_lte_1(self, data_5d):
        from riemann.viz.projection import project_pca
        result = project_pca(data_5d, n_components=3)
        assert sum(result.metadata["variance_explained"]) <= 1.0 + 1e-10

    def test_method_name(self, data_5d):
        from riemann.viz.projection import project_pca
        result = project_pca(data_5d)
        assert result.method == "pca"


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

class TestProjectTSNE:
    def test_correct_shape(self, data_5d):
        from riemann.viz.projection import project_tsne
        result = project_tsne(data_5d, n_components=2)
        assert result.coordinates.shape == (100, 2)

    def test_perplexity_in_metadata(self, data_5d):
        from riemann.viz.projection import project_tsne
        result = project_tsne(data_5d, n_components=2, perplexity=20.0)
        assert "perplexity" in result.metadata
        assert result.metadata["perplexity"] == 20.0


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

class TestProjectUMAP:
    def test_correct_shape(self, data_5d):
        from riemann.viz.projection import project_umap
        result = project_umap(data_5d, n_components=3)
        assert result.coordinates.shape == (100, 3)

    def test_n_neighbors_in_metadata(self, data_5d):
        from riemann.viz.projection import project_umap
        result = project_umap(data_5d, n_components=3, n_neighbors=10)
        assert "n_neighbors" in result.metadata
        assert result.metadata["n_neighbors"] == 10


# ---------------------------------------------------------------------------
# Stereographic
# ---------------------------------------------------------------------------

class TestProjectStereographic:
    def test_reduces_dimension_by_one(self, data_5d):
        from riemann.viz.projection import project_stereographic
        result = project_stereographic(data_5d)
        assert result.coordinates.shape == (100, 4)  # 5D -> 4D
        assert result.source_dim == 5
        assert result.target_dim == 4

    def test_unit_sphere_points_finite(self):
        """Stereographic projection of unit-sphere points produces finite coords."""
        from riemann.viz.projection import project_stereographic
        rng = np.random.default_rng(42)
        # Points on unit sphere (not at north pole)
        raw = rng.standard_normal((50, 4))
        sphere = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        # Ensure no point is exactly at north pole
        sphere[:, -1] = np.clip(sphere[:, -1], -0.99, 0.99)
        result = project_stereographic(sphere)
        assert np.all(np.isfinite(result.coordinates))


# ---------------------------------------------------------------------------
# Hopf fibration
# ---------------------------------------------------------------------------

class TestProjectHopfFibration:
    def test_basic_4d_to_3d(self, data_4d):
        from riemann.viz.projection import project_hopf_fibration
        result = project_hopf_fibration(data_4d)
        assert result.coordinates.shape == (100, 3)
        assert result.target_dim == 3
        assert result.method == "hopf_fibration"

    def test_fiber_structure_preserved(self):
        """Two points on the same S^1 fiber should map to the same S^2 point."""
        from riemann.viz.projection import project_hopf_fibration
        # A point on S^3
        p = np.array([1.0, 0.0, 0.0, 0.0])
        # Rotate by phase angle theta in S^1 fiber:
        # (z1, z2) -> (z1 * e^(i*theta), z2 * e^(i*theta))
        theta = 0.7
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # z1 = x0 + i*x1 -> (x0*cos - x1*sin) + i*(x0*sin + x1*cos)
        # z2 = x2 + i*x3 -> (x2*cos - x3*sin) + i*(x2*sin + x3*cos)
        q = np.array([
            p[0] * cos_t - p[1] * sin_t,
            p[0] * sin_t + p[1] * cos_t,
            p[2] * cos_t - p[3] * sin_t,
            p[2] * sin_t + p[3] * cos_t,
        ])
        data = np.vstack([p, q])
        result = project_hopf_fibration(data)
        # Both points should map to the same S^2 point
        np.testing.assert_allclose(
            result.coordinates[0], result.coordinates[1], atol=1e-10
        )

    def test_wrong_dim_raises(self):
        from riemann.viz.projection import project_hopf_fibration
        data_5d = np.random.default_rng(42).standard_normal((50, 5))
        with pytest.raises(ValueError, match="4"):
            project_hopf_fibration(data_5d)

    def test_fiber_phase_in_metadata(self, data_4d):
        from riemann.viz.projection import project_hopf_fibration
        result = project_hopf_fibration(data_4d)
        assert "fiber_phase" in result.metadata
        assert len(result.metadata["fiber_phase"]) == 100
