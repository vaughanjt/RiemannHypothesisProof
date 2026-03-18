"""Test scaffolds for VIZ-01: Critical line visualization.

Tests are marked xfail pending implementation in Plan 01-04.
"""
import numpy as np
import pytest


@pytest.mark.xfail(reason="Implementation pending in Plan 01-04")
def test_critical_line_data_generation():
    """Generating t vs |Z(t)| data produces arrays of correct shape."""
    from riemann.viz.critical_line import critical_line_data

    t_values, z_values = critical_line_data(0, 50, num_points=100, dps=30)

    assert isinstance(t_values, np.ndarray)
    assert isinstance(z_values, np.ndarray)
    assert t_values.shape == (100,)
    assert z_values.shape == (100,)
    assert t_values[0] == 0.0
    assert t_values[-1] == 50.0


@pytest.mark.xfail(reason="Implementation pending in Plan 01-04")
def test_critical_line_plot_no_error():
    """Plot generation completes without error."""
    from riemann.viz.critical_line import plot_critical_line

    # Should not raise
    fig = plot_critical_line(t_start=0, t_end=30, num_points=50, dps=15)
    assert fig is not None
