"""Test scaffolds for VIZ-02: Domain coloring visualization.

Tests are marked xfail pending implementation in Plan 01-04.
"""
import numpy as np
import pytest


@pytest.mark.xfail(reason="Implementation pending in Plan 01-04")
def test_domain_coloring_produces_rgb():
    """Output is a valid RGB array with shape (N, M, 3) and values in [0, 1]."""
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


@pytest.mark.xfail(reason="Implementation pending in Plan 01-04")
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
