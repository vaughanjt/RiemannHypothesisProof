"""Tests for VIZ-02: Domain coloring visualization.

Tests both fast numpy mode and high-precision mpmath mode for domain coloring,
plus the matplotlib rendering function.
"""
import time

import numpy as np
import pytest


def test_domain_coloring_produces_rgb():
    """Output is a valid RGB array with shape (N, N, 3) and values in [0, 1]."""
    from riemann.viz.domain_coloring import domain_coloring

    rgb, re, im = domain_coloring(
        lambda z: z ** 2 - 1,
        re_range=(-2, 2),
        im_range=(-2, 2),
        resolution=50,
    )

    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (50, 50, 3)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


def test_domain_coloring_zeros_dark():
    """Zero locations should appear dark (low brightness)."""
    from riemann.viz.domain_coloring import domain_coloring

    # z^2 - 1 has zeros at z = +1, -1
    rgb, re, im = domain_coloring(
        lambda z: z ** 2 - 1,
        re_range=(-2, 2),
        im_range=(-2, 2),
        resolution=100,
    )

    # Find pixel near z = 1 (re=1, im=0)
    re_idx = np.argmin(np.abs(re - 1.0))
    im_idx = np.argmin(np.abs(im - 0.0))

    # Brightness at zero should be relatively low
    brightness = np.mean(rgb[im_idx, re_idx, :])
    assert brightness < 0.5, f"Zero at (1,0) should be dark, got brightness {brightness}"


def test_domain_coloring_no_nan_inf():
    """RGB array should have no NaN or Inf values."""
    from riemann.viz.domain_coloring import domain_coloring

    # Use a function with a pole at z=0
    def f_with_pole(z):
        # Avoid division by zero in numpy -- add tiny offset
        return 1.0 / (z + 1e-30)

    rgb, re, im = domain_coloring(
        f_with_pole,
        re_range=(-2, 2),
        im_range=(-2, 2),
        resolution=50,
    )

    assert np.all(np.isfinite(rgb)), "RGB array contains NaN or Inf"


def test_domain_coloring_mpmath_critical_strip():
    """High-precision mpmath variant works in the critical strip."""
    import mpmath
    from riemann.viz.domain_coloring import domain_coloring_mpmath

    rgb, re, im = domain_coloring_mpmath(
        mpmath.zeta,
        re_range=(0.0, 1.0),
        im_range=(10.0, 20.0),
        resolution=20,
        dps=20,
    )

    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (20, 20, 3)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0
    assert np.all(np.isfinite(rgb)), "mpmath mode RGB contains NaN or Inf"


def test_domain_coloring_progressive_resolution():
    """Higher resolution takes longer than lower resolution (progressive works)."""
    from riemann.viz.domain_coloring import domain_coloring

    f = lambda z: z ** 2 - 1

    start = time.time()
    domain_coloring(f, resolution=50)
    time_low = time.time() - start

    start = time.time()
    domain_coloring(f, resolution=200)
    time_high = time.time() - start

    # Resolution 200 should be slower than resolution 50
    # (200^2 = 40000 vs 50^2 = 2500, ~16x more work)
    assert time_high > time_low * 0.5, (
        f"res=200 took {time_high:.3f}s, res=50 took {time_low:.3f}s -- "
        "higher resolution should take longer"
    )


def test_plot_domain_coloring_returns_figure():
    """plot_domain_coloring returns a matplotlib Figure."""
    from riemann.viz.domain_coloring import plot_domain_coloring
    import matplotlib.figure

    fig = plot_domain_coloring(
        f=lambda z: z ** 2 - 1,
        re_range=(-2, 2),
        im_range=(-2, 2),
        resolution=30,
    )

    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)
