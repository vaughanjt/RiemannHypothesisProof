"""Smoke tests for comparison visualization functions.

Each test creates minimal synthetic input, calls the function,
and asserts the return is a non-empty ``go.Figure``.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from riemann.viz.comparison import (
    create_info_comparison_heatmap,
    create_number_variance_comparison,
    create_pair_correlation_comparison,
    create_rmt_slider_figure,
    create_spacing_comparison,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def zero_spacings(rng):
    """Synthetic normalized spacings mimicking zeta zero distribution."""
    return rng.exponential(1.0, 100)


@pytest.fixture
def gue_spacings(rng):
    """Synthetic normalized spacings mimicking GUE distribution."""
    return rng.exponential(1.0, 100)


# ----- create_spacing_comparison -----

def test_create_spacing_comparison(zero_spacings, gue_spacings):
    fig = create_spacing_comparison(zero_spacings, gue_spacings)
    assert isinstance(fig, go.Figure)
    # At least 2 traces: two histograms (plus optional Wigner curve)
    assert len(fig.data) >= 2


# ----- create_pair_correlation_comparison -----

def test_create_pair_correlation_comparison(zero_spacings):
    fig = create_pair_correlation_comparison(zero_spacings)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


# ----- create_rmt_slider_figure -----

def test_create_rmt_slider_figure(zero_spacings):
    fig = create_rmt_slider_figure(
        zero_spacings, n_values=[10, 25], num_matrices=10,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Should have slider or multiple frames
    assert len(fig.frames) > 0 or fig.layout.sliders is not None


# ----- create_info_comparison_heatmap -----

def test_create_info_comparison_heatmap():
    comparison = {
        "zeros": {"entropy": 0.5, "complexity": 0.3},
        "gue": {"entropy": 0.4, "complexity": 0.6},
    }
    fig = create_info_comparison_heatmap(comparison)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Check it's a heatmap trace
    assert isinstance(fig.data[0], go.Heatmap)


# ----- create_number_variance_comparison -----

def test_create_number_variance_comparison(zero_spacings):
    fig = create_number_variance_comparison(zero_spacings)
    assert isinstance(fig, go.Figure)
    # At least 2 traces: empirical + theoretical
    assert len(fig.data) >= 2
