"""Critical line visualization: |zeta(1/2+it)| via Hardy's Z-function.

DESIGN RULES (from CONTEXT.md):
- Optimize for speed over visual polish
- Start coarse, zoom to refine on demand
- Claude picks matplotlib vs Plotly per use case
- NEVER compute zeta values inside plot callbacks (Anti-Pattern #3)

Computation and visualization are strictly separated:
1. critical_line_data() computes and returns arrays
2. plot_*() functions render pre-computed data
"""
import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from riemann.config import DEFAULT_DPS
from riemann.viz.styles import ANALYTICAL_PALETTE, MATPLOTLIB_DEFAULTS


def critical_line_data(
    t_start: float,
    t_end: float,
    num_points: int = 500,
    dps: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute |zeta(1/2+it)| along the critical line via Hardy Z-function.

    Uses mpmath.siegelz(t) which is real-valued and satisfies |Z(t)| = |zeta(1/2+it)|.
    Z(t) itself (not |Z(t)|) is returned so sign changes (zero crossings) are visible.

    Args:
        t_start: Start of t range.
        t_end: End of t range.
        num_points: Number of evaluation points.
        dps: Decimal digits precision for evaluation. Default 15 is sufficient
            for visualization (no need for 50-digit precision in plot data).

    Returns:
        (t_values, z_values): numpy arrays of shape (num_points,).
        z_values are real (Hardy Z-function values).
    """
    t_values = np.linspace(t_start, t_end, num_points)
    z_values = np.empty(num_points)

    with mpmath.workdps(dps + 5):
        for i, t in enumerate(t_values):
            z_values[i] = float(mpmath.siegelz(mpmath.mpf(t)))

    return t_values, z_values


def plot_critical_line_static(
    t_start: float = 0,
    t_end: float = 50,
    num_points: int = 1000,
    dps: int = 15,
    ax=None,
):
    """Static matplotlib plot of Z(t) along the critical line.

    Shows zero crossings as sign changes. Good for Claude's analysis.

    Returns:
        matplotlib.figure.Figure
    """
    t_values, z_values = critical_line_data(t_start, t_end, num_points, dps)

    with plt.rc_context(MATPLOTLIB_DEFAULTS):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        else:
            fig = ax.get_figure()

        ax.plot(t_values, z_values,
                color=ANALYTICAL_PALETTE["primary"],
                linewidth=0.8, label="Z(t)")
        ax.axhline(y=0, color=ANALYTICAL_PALETTE["zero_line"],
                   linewidth=0.5, linestyle='--', alpha=0.7)
        ax.set_xlabel("t")
        ax.set_ylabel("Z(t)")
        ax.set_title(f"|zeta(1/2+it)| via Hardy Z-function, t in [{t_start}, {t_end}]")
        ax.legend(loc="upper right")

    return fig


def plot_critical_line_interactive(
    t_start: float = 0,
    t_end: float = 50,
    num_points: int = 1000,
    dps: int = 15,
):
    """Interactive Plotly plot of Z(t) with zoom/pan/hover.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    t_values, z_values = critical_line_data(t_start, t_end, num_points, dps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_values, y=z_values,
        mode='lines',
        name='Z(t)',
        line=dict(color=ANALYTICAL_PALETTE["primary"], width=1),
        hovertemplate="t=%{x:.6f}<br>Z(t)=%{y:.6f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash",
                  line_color=ANALYTICAL_PALETTE["zero_line"],
                  opacity=0.5)
    fig.update_layout(
        title="|zeta(1/2+it)| via Hardy Z-function",
        xaxis_title="t",
        yaxis_title="Z(t)",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig
