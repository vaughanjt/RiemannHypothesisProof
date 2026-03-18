"""Domain coloring: phase -> hue, log-magnitude -> brightness.

DESIGN RULES (from CONTEXT.md):
- Start coarse (200x200), refine on demand (progressive resolution)
- Claude picks resolution levels per zoom stage
- NEVER compute inside plot callbacks

Two modes:
- domain_coloring(): Uses numpy complex128 for speed. Good for overview.
- domain_coloring_mpmath(): Uses mpmath per-point evaluation. Required for
  critical strip where float64 is insufficient (Pitfall 1).
"""
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpmath


def domain_coloring(
    f,
    re_range: tuple[float, float] = (-2, 2),
    im_range: tuple[float, float] = (-2, 2),
    resolution: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized domain coloring for a complex function.

    Uses numpy complex128 for speed. NOT suitable for high-precision
    critical strip evaluation -- use domain_coloring_mpmath for that.

    Args:
        f: Complex function accepting numpy array of complex128.
        re_range: (re_min, re_max) for real axis.
        im_range: (im_min, im_max) for imaginary axis.
        resolution: Grid size (resolution x resolution pixels).

    Returns:
        (RGB, re_values, im_values): RGB is shape (resolution, resolution, 3)
        with values in [0, 1]. re_values and im_values are 1D arrays.
    """
    re = np.linspace(*re_range, resolution)
    im = np.linspace(*im_range, resolution)
    Re, Im = np.meshgrid(re, im)
    Z = Re + 1j * Im

    W = f(Z)

    # Phase -> Hue (wrapping 0 to 1)
    H = (np.angle(W) / (2 * np.pi)) % 1.0

    # Log-magnitude -> Brightness (zeros are dark, poles are bright)
    mag = np.abs(W)
    # Avoid log(0): use log1p which handles mag=0 gracefully
    V = 1.0 - 1.0 / (1.0 + 0.3 * np.log1p(mag))

    # Saturation: high for clarity
    S = 0.9 * np.ones_like(H)

    # Handle NaN/Inf from poles or branch cuts
    nan_mask = ~np.isfinite(W)
    H[nan_mask] = 0.0
    S[nan_mask] = 0.0
    V[nan_mask] = 1.0  # Poles appear white

    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)

    return RGB, re, im


def domain_coloring_mpmath(
    f,
    re_range: tuple[float, float] = (-1, 2),
    im_range: tuple[float, float] = (0, 30),
    resolution: int = 100,
    dps: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-point domain coloring using mpmath for critical strip precision.

    Slower than vectorized version but safe near the critical line.
    Use this when re_range includes the critical strip (Re ~ 0 to 1).

    Args:
        f: Function accepting mpmath.mpc, returning mpmath.mpc.
            Typically lambda s: mpmath.zeta(s).
        re_range: Real axis range.
        im_range: Imaginary axis range.
        resolution: Grid size.
        dps: mpmath precision digits for each evaluation.

    Returns:
        (RGB, re_values, im_values): Same format as domain_coloring.
    """
    re = np.linspace(*re_range, resolution)
    im = np.linspace(*im_range, resolution)

    W = np.empty((resolution, resolution), dtype=complex)

    with mpmath.workdps(dps + 5):
        for i, im_val in enumerate(im):
            for j, re_val in enumerate(re):
                s = mpmath.mpc(re_val, im_val)
                w = f(s)
                W[i, j] = complex(w)

    # Phase -> Hue
    H = (np.angle(W) / (2 * np.pi)) % 1.0
    mag = np.abs(W)
    V = 1.0 - 1.0 / (1.0 + 0.3 * np.log1p(mag))
    S = 0.9 * np.ones_like(H)

    nan_mask = ~np.isfinite(W)
    H[nan_mask] = 0.0
    S[nan_mask] = 0.0
    V[nan_mask] = 1.0

    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)

    return RGB, re, im


def plot_domain_coloring(
    f=None,
    re_range: tuple[float, float] = (-2, 2),
    im_range: tuple[float, float] = (-2, 2),
    resolution: int = 200,
    use_mpmath: bool = False,
    dps: int = 20,
    title: str = "Domain Coloring",
    ax=None,
):
    """Render domain coloring as a matplotlib image.

    Args:
        f: Complex function. Default: Dirichlet series approximation for overview,
            or mpmath.zeta if use_mpmath=True.
        re_range, im_range: Plot region.
        resolution: Grid size.
        use_mpmath: If True, use per-point mpmath evaluation (slower, accurate).
        dps: Precision for mpmath mode.
        title: Plot title.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib.figure.Figure
    """
    if f is None:
        if use_mpmath:
            f = mpmath.zeta
        else:
            def f(z):
                # Simple Dirichlet series approximation for overview
                # NOT suitable for critical strip -- use mpmath mode
                result = np.zeros_like(z, dtype=complex)
                for n in range(1, 100):
                    result += 1.0 / np.power(n + 0j, z)
                return result

    if use_mpmath:
        RGB, re_vals, im_vals = domain_coloring_mpmath(
            f, re_range, im_range, resolution, dps
        )
    else:
        RGB, re_vals, im_vals = domain_coloring(
            f, re_range, im_range, resolution
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    ax.imshow(
        RGB,
        extent=[re_range[0], re_range[1], im_range[0], im_range[1]],
        origin='lower',
        aspect='auto',
    )
    ax.set_xlabel('Re(s)')
    ax.set_ylabel('Im(s)')
    ax.set_title(title)

    return fig
