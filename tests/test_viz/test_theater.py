"""Smoke tests for projection theater visualization.

Each test verifies that figures are created and contain data --
not visual correctness (that is a human judgment).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from riemann.viz.projection import ProjectionResult, project_pca
from riemann.viz.theater import (
    create_dimension_slice_view,
    create_projection_path_animation,
    create_side_by_side,
    create_theater_figure,
)


@pytest.fixture
def sample_projection_3d() -> ProjectionResult:
    """3D PCA projection from random data."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((80, 6))
    return project_pca(data, n_components=3)


@pytest.fixture
def sample_projection_2d() -> ProjectionResult:
    """2D PCA projection from random data."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((80, 6))
    return project_pca(data, n_components=2)


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Random 80x6 embedding matrix."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((80, 6))


# ----- create_theater_figure -----

def test_create_theater_figure_returns_figure(sample_projection_3d: ProjectionResult):
    fig = create_theater_figure(sample_projection_3d)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_theater_figure_with_color_by(sample_projection_3d: ProjectionResult):
    color = np.arange(sample_projection_3d.coordinates.shape[0], dtype=float)
    fig = create_theater_figure(sample_projection_3d, color_by=color)
    assert isinstance(fig, go.Figure)
    # Verify colorscale is set on the marker
    assert fig.data[0].marker.showscale is True


def test_create_theater_figure_2d(sample_projection_2d: ProjectionResult):
    fig = create_theater_figure(sample_projection_2d)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # 2D projection should use Scatter, not Scatter3d
    assert isinstance(fig.data[0], go.Scatter)


# ----- create_projection_path_animation -----

def test_create_projection_path_animation(sample_embedding: np.ndarray):
    fig = create_projection_path_animation(
        sample_embedding, methods=["pca"], n_frames=3,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # With only one method, there are no transitions so frames may be empty.
    # But figure itself must exist with initial data.


def test_create_projection_path_animation_two_methods(sample_embedding: np.ndarray):
    fig = create_projection_path_animation(
        sample_embedding, methods=["pca", "pca"], n_frames=5,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) > 0


# ----- create_dimension_slice_view -----

def test_create_dimension_slice_view(sample_embedding: np.ndarray):
    fig = create_dimension_slice_view(
        sample_embedding, fix_dims={0: 0.0}, project_remaining="pca",
    )
    assert isinstance(fig, go.Figure)


# ----- create_side_by_side -----

def test_create_side_by_side_three_projections(sample_embedding: np.ndarray):
    rng = np.random.default_rng(42)
    projs = {}
    for name in ["A", "B", "C"]:
        coords = rng.standard_normal((80, 3))
        projs[name] = ProjectionResult(
            coordinates=coords, method=name, source_dim=6, target_dim=3,
        )
    fig = create_side_by_side(projs)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
