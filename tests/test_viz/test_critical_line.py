"""Tests for VIZ-01: Critical line visualization.

Tests Hardy Z-function data generation and both static (matplotlib) and
interactive (Plotly) plotting functions.
"""
import time

import numpy as np
import pytest


def test_critical_line_data_generation():
    """Generating t vs Z(t) data produces arrays of correct shape."""
    from riemann.viz.critical_line import critical_line_data

    t_values, z_values = critical_line_data(0, 50, num_points=100, dps=15)

    assert isinstance(t_values, np.ndarray)
    assert isinstance(z_values, np.ndarray)
    assert t_values.shape == (100,)
    assert z_values.shape == (100,)
    assert t_values[0] == 0.0
    assert t_values[-1] == 50.0


def test_critical_line_z_values_are_real():
    """Z(t) values should be real-valued (Hardy Z-function is real on real axis)."""
    from riemann.viz.critical_line import critical_line_data

    _, z_values = critical_line_data(10, 20, num_points=50, dps=15)

    # z_values should be real floats (numpy float64), not complex
    assert z_values.dtype in (np.float64, np.float32), (
        f"Expected real dtype, got {z_values.dtype}"
    )


def test_critical_line_zero_crossing():
    """Z(t) near t=14.134 should be close to zero (first non-trivial zero)."""
    from riemann.viz.critical_line import critical_line_data

    # Narrow range around first zero
    t_values, z_values = critical_line_data(14.0, 14.3, num_points=100, dps=15)

    # The minimum absolute value should be small (zero crossing)
    min_abs_z = np.min(np.abs(z_values))
    assert min_abs_z < 0.5, (
        f"Expected Z(t) near zero around t=14.134, min |Z(t)| = {min_abs_z}"
    )


def test_critical_line_plot_static_no_error():
    """Static matplotlib plot generation completes without error."""
    from riemann.viz.critical_line import plot_critical_line_static
    import matplotlib.figure

    fig = plot_critical_line_static(t_start=0, t_end=30, num_points=50, dps=15)
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)


def test_critical_line_plot_interactive_no_error():
    """Interactive Plotly plot generation completes without error."""
    from riemann.viz.critical_line import plot_critical_line_interactive
    import plotly.graph_objects as go

    fig = plot_critical_line_interactive(t_start=0, t_end=30, num_points=50, dps=15)
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_critical_line_lower_precision_faster():
    """Lower precision (dps=15) should run faster than higher precision (dps=50)."""
    from riemann.viz.critical_line import critical_line_data

    # Time at dps=15
    start = time.time()
    critical_line_data(0, 30, num_points=50, dps=15)
    time_low = time.time() - start

    # Time at dps=50
    start = time.time()
    critical_line_data(0, 30, num_points=50, dps=50)
    time_high = time.time() - start

    # dps=15 should be faster (or at least not significantly slower)
    # Allow 2x tolerance for measurement noise
    assert time_low < time_high * 2.0, (
        f"dps=15 took {time_low:.3f}s, dps=50 took {time_high:.3f}s"
    )
