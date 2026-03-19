"""Side-by-side comparison views: RMT overlay, info theory heatmap.

Visualization functions for comparing zero spacing statistics against
random matrix theory predictions and information-theoretic signatures
across mathematical objects.

All functions return ``plotly.graph_objects.Figure`` objects ready for
JupyterLab display. Computation is delegated to ``riemann.analysis.*``.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from riemann.analysis.rmt import eigenvalue_spacings, generate_gue, wigner_surmise
from riemann.analysis.spacing import gue_pair_correlation, number_variance, pair_correlation
from riemann.viz.styles import ANALYTICAL_PALETTE


# ---------------------------------------------------------------------------
# Spacing distribution comparison (histogram overlay)
# ---------------------------------------------------------------------------

def create_spacing_comparison(
    zero_spacings: np.ndarray,
    gue_spacings: np.ndarray,
    title: str = "Zero Spacing vs GUE",
) -> go.Figure:
    """Overlay histograms of zero spacings and GUE eigenvalue spacings.

    Also plots the Wigner surmise (GUE, beta=2) theoretical curve.

    Args:
        zero_spacings: Normalized spacings from zeta zeros.
        gue_spacings: Normalized spacings from GUE ensemble.
        title: Figure title.

    Returns:
        Plotly Figure with two overlaid histograms and theoretical curve.
    """
    fig = go.Figure()

    # Zero spacings histogram
    fig.add_trace(go.Histogram(
        x=zero_spacings,
        histnorm="probability density",
        nbinsx=80,
        xbins=dict(start=0, end=4),
        name="Zeta zeros",
        marker_color=ANALYTICAL_PALETTE["primary"],
        opacity=0.6,
    ))

    # GUE spacings histogram
    fig.add_trace(go.Histogram(
        x=gue_spacings,
        histnorm="probability density",
        nbinsx=80,
        xbins=dict(start=0, end=4),
        name="GUE",
        marker_color=ANALYTICAL_PALETTE["secondary"],
        opacity=0.6,
    ))

    # Wigner surmise theoretical curve
    s_vals = np.linspace(0, 4, 200)
    ws_vals = wigner_surmise(s_vals, beta=2)
    fig.add_trace(go.Scatter(
        x=s_vals,
        y=ws_vals,
        mode="lines",
        name="Wigner surmise (GUE)",
        line=dict(color=ANALYTICAL_PALETTE["zero_line"], dash="dash", width=2),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Normalized spacing s",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        legend=dict(x=0.7, y=0.95),
    )

    return fig


# ---------------------------------------------------------------------------
# Pair correlation comparison with residual subplot
# ---------------------------------------------------------------------------

def create_pair_correlation_comparison(
    zero_spacings: np.ndarray,
    gue_spacings: np.ndarray | None = None,
    title: str = "Pair Correlation",
) -> go.Figure:
    """Pair correlation R_2(x) comparison with residual plot.

    Top panel: zero R_2, GUE theory, and optionally empirical GUE R_2.
    Bottom panel: residual (zero R_2 - GUE theory).

    Args:
        zero_spacings: Normalized spacings from zeta zeros.
        gue_spacings: Optional normalized spacings from GUE ensemble.
        title: Figure title.

    Returns:
        Plotly Figure with two vertically stacked subplots.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Pair Correlation R_2(x)", "Residual"],
    )

    # Compute zero pair correlation
    x_centers, zero_r2 = pair_correlation(zero_spacings, bins=200, x_range=(0.0, 4.0))

    # GUE theoretical curve
    gue_r2_theory = gue_pair_correlation(x_centers)

    # Zero R_2
    fig.add_trace(go.Scatter(
        x=x_centers, y=zero_r2,
        mode="lines", name="Zeta zeros",
        line=dict(color=ANALYTICAL_PALETTE["primary"], width=1.5),
    ), row=1, col=1)

    # GUE theory
    fig.add_trace(go.Scatter(
        x=x_centers, y=gue_r2_theory,
        mode="lines", name="GUE theory",
        line=dict(color=ANALYTICAL_PALETTE["zero_line"], dash="dash", width=2),
    ), row=1, col=1)

    # Empirical GUE (if provided)
    if gue_spacings is not None and len(gue_spacings) > 0:
        x_gue, gue_r2_emp = pair_correlation(gue_spacings, bins=200, x_range=(0.0, 4.0))
        fig.add_trace(go.Scatter(
            x=x_gue, y=gue_r2_emp,
            mode="lines", name="GUE empirical",
            line=dict(color=ANALYTICAL_PALETTE["secondary"], width=1),
        ), row=1, col=1)

    # Residual: zero_r2 - gue_theory
    residual = zero_r2 - gue_r2_theory
    fig.add_trace(go.Scatter(
        x=x_centers, y=residual,
        mode="lines", name="Residual",
        line=dict(color=ANALYTICAL_PALETTE["annotation"], width=1),
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        legend=dict(x=0.7, y=0.95),
    )
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_yaxes(title_text="R_2(x)", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# RMT N-slider: vary matrix size and see spacing update
# ---------------------------------------------------------------------------

def create_rmt_slider_figure(
    zero_spacings: np.ndarray,
    n_values: list[int] | None = None,
    num_matrices: int = 200,
    seed: int = 42,
) -> go.Figure:
    """Interactive slider to compare zero spacings with GUE(N) for different N.

    Precomputes GUE spacing histograms for each N. Uses Plotly sliders
    (works in static HTML, no ipywidgets needed).

    Args:
        zero_spacings: Normalized spacings from zeta zeros.
        n_values: List of matrix dimensions to try (default [10,25,50,100,200,500]).
        num_matrices: Number of matrices per N value.
        seed: Random seed.

    Returns:
        Plotly Figure with slider controlling which GUE(N) overlay is shown.
    """
    if n_values is None:
        n_values = [10, 25, 50, 100, 200, 500]

    # Precompute GUE histograms
    bins_edges = np.linspace(0, 4, 81)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2.0

    # Zero spacings histogram (constant across slider positions)
    zero_hist, _ = np.histogram(zero_spacings, bins=bins_edges, density=True)

    # Wigner surmise curve (constant)
    s_fine = np.linspace(0, 4, 200)
    ws_fine = wigner_surmise(s_fine, beta=2)

    # Build traces and frames for each N
    frames = []
    slider_steps = []

    for idx, n in enumerate(n_values):
        gue_eigs = generate_gue(n=n, num_matrices=num_matrices, seed=seed + idx)
        gue_spac = eigenvalue_spacings(gue_eigs)
        gue_hist, _ = np.histogram(gue_spac, bins=bins_edges, density=True)

        # Chi-squared goodness-of-fit
        mask = zero_hist > 0.01
        if mask.sum() > 0:
            chi_sq = float(np.sum((gue_hist[mask] - zero_hist[mask])**2 / zero_hist[mask]))
        else:
            chi_sq = float("nan")

        frame_data = [
            go.Bar(
                x=bin_centers, y=zero_hist,
                name="Zeta zeros", marker_color=ANALYTICAL_PALETTE["primary"],
                opacity=0.6, width=0.05,
            ),
            go.Bar(
                x=bin_centers, y=gue_hist,
                name=f"GUE(N={n})", marker_color=ANALYTICAL_PALETTE["secondary"],
                opacity=0.6, width=0.05,
            ),
            go.Scatter(
                x=s_fine, y=ws_fine,
                mode="lines", name="Wigner surmise",
                line=dict(color=ANALYTICAL_PALETTE["zero_line"], dash="dash", width=2),
            ),
        ]

        frames.append(go.Frame(data=frame_data, name=str(n)))

        slider_steps.append(dict(
            method="animate",
            args=[[str(n)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
            label=f"N={n}",
        ))

    # Initial state: first N value
    first_frame_data = frames[0].data if frames else []
    first_n = n_values[0] if n_values else 0

    fig = go.Figure(
        data=list(first_frame_data),
        frames=frames,
        layout=go.Layout(
            title=f"Zero Spacings vs GUE(N) -- N={first_n}",
            xaxis_title="Normalized spacing s",
            yaxis_title="Density",
            barmode="overlay",
            template="plotly_white",
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Matrix size: "),
                pad=dict(t=50),
                steps=slider_steps,
            )],
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Information-theoretic signature heatmap
# ---------------------------------------------------------------------------

def create_info_comparison_heatmap(comparison_result: dict) -> go.Figure:
    """Heatmap of information-theoretic signatures across mathematical objects.

    Args:
        comparison_result: Output of ``cross_object_comparison``:
            ``{object_name: {metric_name: value}}``.

    Returns:
        Plotly Figure with annotated heatmap.
    """
    objects = list(comparison_result.keys())
    if not objects:
        fig = go.Figure()
        fig.update_layout(title="No data for heatmap")
        return fig

    metrics = list(comparison_result[objects[0]].keys())

    # Build value matrix: rows = objects, columns = metrics
    z_raw = np.array([
        [float(comparison_result[obj].get(m, 0)) for m in metrics]
        for obj in objects
    ])

    # Normalize each column to [0, 1] for color scale
    z_norm = z_raw.copy()
    for col_idx in range(z_norm.shape[1]):
        col = z_norm[:, col_idx]
        col_min, col_max = col.min(), col.max()
        span = col_max - col_min
        if span > 1e-12:
            z_norm[:, col_idx] = (col - col_min) / span
        else:
            z_norm[:, col_idx] = 0.5

    # Text annotations show raw values
    text_matrix = [
        [f"{z_raw[r, c]:.3f}" for c in range(len(metrics))]
        for r in range(len(objects))
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_norm,
        x=metrics,
        y=objects,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmid=0.5,
        showscale=True,
        colorbar=dict(title="Normalized"),
    ))

    fig.update_layout(
        title="Information-Theoretic Signature Comparison",
        xaxis_title="Metric",
        yaxis_title="Object",
        template="plotly_white",
    )

    return fig


# ---------------------------------------------------------------------------
# Number variance comparison
# ---------------------------------------------------------------------------

def create_number_variance_comparison(
    zero_spacings: np.ndarray,
    gue_spacings: np.ndarray | None = None,
    L_values: np.ndarray | None = None,
) -> go.Figure:
    """Compare number variance Sigma_2(L) across sources.

    Plots empirical zero number variance, theoretical GUE prediction,
    optionally empirical GUE, and Poisson reference (Sigma_2 = L).

    Args:
        zero_spacings: Normalized spacings from zeta zeros.
        gue_spacings: Optional normalized GUE spacings for empirical comparison.
        L_values: Interval lengths to evaluate (default: 50 values in [0.1, 5]).

    Returns:
        Plotly Figure with number variance curves.
    """
    if L_values is None:
        L_values = np.linspace(0.1, 5.0, 50)

    fig = go.Figure()

    # Empirical zero number variance
    zero_nv = number_variance(zero_spacings, L_values=L_values)
    fig.add_trace(go.Scatter(
        x=L_values, y=zero_nv,
        mode="lines+markers", name="Zeta zeros",
        marker=dict(size=4),
        line=dict(color=ANALYTICAL_PALETTE["primary"], width=1.5),
    ))

    # Theoretical GUE number variance:
    # Sigma_2(L) ~ (2/pi^2) * (ln(2*pi*L) + gamma_euler + 1 - pi^2/8)
    gamma_euler = 0.5772156649015329  # Euler-Mascheroni constant
    L_positive = L_values[L_values > 0]
    gue_theory = (2.0 / np.pi**2) * (
        np.log(2.0 * np.pi * L_positive) + gamma_euler + 1.0 - np.pi**2 / 8.0
    )
    # Clamp negative values (formula is approximate for small L)
    gue_theory = np.maximum(gue_theory, 0.0)

    fig.add_trace(go.Scatter(
        x=L_positive, y=gue_theory,
        mode="lines", name="GUE theory",
        line=dict(color=ANALYTICAL_PALETTE["zero_line"], dash="dash", width=2),
    ))

    # Empirical GUE number variance (if provided)
    if gue_spacings is not None and len(gue_spacings) > 0:
        gue_nv = number_variance(gue_spacings, L_values=L_values)
        fig.add_trace(go.Scatter(
            x=L_values, y=gue_nv,
            mode="lines", name="GUE empirical",
            line=dict(color=ANALYTICAL_PALETTE["secondary"], width=1),
        ))

    # Poisson reference: Sigma_2(L) = L
    fig.add_trace(go.Scatter(
        x=L_values, y=L_values,
        mode="lines", name="Poisson",
        line=dict(color="gray", dash="dot", width=1),
    ))

    fig.update_layout(
        title="Number Variance Sigma_2(L)",
        xaxis_title="L (mean spacings)",
        yaxis_title="Sigma_2(L)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    return fig
