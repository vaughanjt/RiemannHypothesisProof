"""Diagnostic Plotly plots for heat kernel feasibility comparison (D-04).

Visualization functions for the Phase 5 feasibility gate: convergence
analysis and agreement heatmaps for K(t(L)) vs B(L) comparison.

All functions return ``plotly.graph_objects.Figure`` objects ready for
JupyterLab display. Follows project Plotly convention from viz/comparison.py.
"""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from riemann.types import BarrierComparison
from riemann.viz.styles import ANALYTICAL_PALETTE


def plot_convergence_vs_L(comparisons: list[BarrierComparison]) -> go.Figure:
    """Create a two-panel subplot showing spectral convergence and agreement vs L.

    Top panel: discrete sum and Eisenstein contribution magnitudes vs L,
    with 0.036 budget reference line.
    Bottom panel: digits of agreement vs L, color-coded by quality.

    Args:
        comparisons: List of BarrierComparison objects from run_feasibility_comparison.

    Returns:
        Plotly Figure with 2 rows.
    """
    Ls = [c.L for c in comparisons]
    discrete = [c.discrete_sum for c in comparisons]
    eisenstein = [abs(c.eisenstein_contrib) for c in comparisons]
    digits = [c.digits_of_agreement for c in comparisons]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Spectral Contributions", "Digits of Agreement"),
    )

    # --- Top panel: spectral contributions ---
    fig.add_trace(
        go.Scatter(
            x=Ls,
            y=discrete,
            mode="lines+markers",
            name="Discrete (Maass) sum",
            marker=dict(size=4, color=ANALYTICAL_PALETTE["primary"]),
            line=dict(color=ANALYTICAL_PALETTE["primary"], width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=Ls,
            y=eisenstein,
            mode="lines+markers",
            name="Eisenstein |contrib|",
            marker=dict(size=4, color=ANALYTICAL_PALETTE["secondary"]),
            line=dict(color=ANALYTICAL_PALETTE["secondary"], width=1.5),
        ),
        row=1,
        col=1,
    )

    # Budget reference line at 0.036
    fig.add_hline(
        y=0.036,
        line_dash="dash",
        line_color=ANALYTICAL_PALETTE["zero_line"],
        annotation_text="0.036 budget",
        annotation_position="top left",
        row=1,
        col=1,
    )

    # --- Bottom panel: digits of agreement ---
    # Color by quality: green >= 6, orange 3-6, red < 3
    colors = []
    for d in digits:
        if d >= 6:
            colors.append(ANALYTICAL_PALETTE["annotation"])  # green
        elif d >= 3:
            colors.append(ANALYTICAL_PALETTE["secondary"])   # orange
        else:
            colors.append(ANALYTICAL_PALETTE["zero_line"])   # red

    fig.add_trace(
        go.Scatter(
            x=Ls,
            y=digits,
            mode="lines+markers",
            name="Digits of agreement",
            marker=dict(size=6, color=colors),
            line=dict(color=ANALYTICAL_PALETTE["primary"], width=1),
        ),
        row=2,
        col=1,
    )

    # Reference lines at 3 and 6 digits
    fig.add_hline(y=6, line_dash="dot", line_color="#999999", row=2, col=1)
    fig.add_hline(y=3, line_dash="dot", line_color="#cccccc", row=2, col=1)

    fig.update_layout(
        title="Heat Kernel Feasibility: Convergence vs L",
        plot_bgcolor=ANALYTICAL_PALETTE["background"],
        paper_bgcolor=ANALYTICAL_PALETTE["background"],
        showlegend=True,
        height=600,
    )
    fig.update_xaxes(title_text="L", row=2, col=1, gridcolor=ANALYTICAL_PALETTE["grid"])
    fig.update_yaxes(title_text="Spectral contribution", row=1, col=1, gridcolor=ANALYTICAL_PALETTE["grid"])
    fig.update_yaxes(title_text="Digits of agreement", row=2, col=1, gridcolor=ANALYTICAL_PALETTE["grid"])

    return fig


def plot_agreement_heatmap(
    comparisons: list[BarrierComparison],
    L_bins: int = 20,
    t_bins: int = 20,
) -> go.Figure:
    """Create a 2D heatmap of digits_of_agreement in (L, t) space.

    If comparisons lie on a 1D curve t(L), this shows agreement along
    that curve with the parameter mapping path annotated.

    Args:
        comparisons: List of BarrierComparison objects.
        L_bins: Number of bins for the L axis.
        t_bins: Number of bins for the t axis.

    Returns:
        Plotly Figure with Heatmap trace.
    """
    Ls = np.array([c.L for c in comparisons])
    ts = np.array([c.t for c in comparisons])
    digits = np.array([c.digits_of_agreement for c in comparisons])

    # Create binned grid
    L_edges = np.linspace(Ls.min(), Ls.max(), L_bins + 1)
    t_edges = np.linspace(ts.min(), ts.max(), t_bins + 1)

    # Bin centers
    L_centers = 0.5 * (L_edges[:-1] + L_edges[1:])
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    # Assign each comparison to the nearest bin
    grid = np.full((t_bins, L_bins), np.nan)
    for i in range(len(comparisons)):
        li = np.searchsorted(L_edges, Ls[i]) - 1
        ti = np.searchsorted(t_edges, ts[i]) - 1
        li = max(0, min(li, L_bins - 1))
        ti = max(0, min(ti, t_bins - 1))
        # Take max agreement in each bin (or replace with latest)
        if np.isnan(grid[ti, li]):
            grid[ti, li] = digits[i]
        else:
            grid[ti, li] = max(grid[ti, li], digits[i])

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=L_centers,
            y=t_centers,
            colorscale="RdYlGn",
            colorbar=dict(title="Digits of agreement"),
            zmin=0,
            zmax=max(15, float(np.nanmax(digits)) if len(digits) > 0 else 15),
        )
    )

    # Overlay the parameter mapping curve as a line trace
    sort_idx = np.argsort(Ls)
    fig.add_trace(
        go.Scatter(
            x=Ls[sort_idx],
            y=ts[sort_idx],
            mode="lines",
            name="t(L) mapping",
            line=dict(color="black", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Agreement Heatmap: K(t) vs B(L)",
        xaxis_title="L",
        yaxis_title="t",
        plot_bgcolor=ANALYTICAL_PALETTE["background"],
        paper_bgcolor=ANALYTICAL_PALETTE["background"],
        height=500,
    )

    return fig


def save_feasibility_plots(
    comparisons: list[BarrierComparison],
    output_dir: str = "data",
) -> list[str]:
    """Create and save both diagnostic plots as interactive HTML files.

    Args:
        comparisons: List of BarrierComparison objects.
        output_dir: Directory to write HTML files.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    conv_fig = plot_convergence_vs_L(comparisons)
    conv_path = os.path.join(output_dir, "feasibility_convergence.html")
    conv_fig.write_html(conv_path)
    saved.append(conv_path)

    heat_fig = plot_agreement_heatmap(comparisons)
    heat_path = os.path.join(output_dir, "feasibility_heatmap.html")
    heat_fig.write_html(heat_path)
    saved.append(heat_path)

    return saved
