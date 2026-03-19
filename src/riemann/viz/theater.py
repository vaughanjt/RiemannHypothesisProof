"""Projection theater: 3D Plotly visualization, animation, dimension slicing, side-by-side comparison.

Interactive exploration of higher-dimensional embedding data through multiple
projection methods. All functions return ``plotly.graph_objects.Figure`` objects
compatible with JupyterLab display (``fig.show()``).

DESIGN RULES:
- Optimize for speed over visual polish
- Computation and visualization strictly separated
- Functions accept pre-computed ProjectionResult objects
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from riemann.viz.projection import ProjectionResult, project_pca
from riemann.viz.styles import ANALYTICAL_PALETTE

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# 3D scatter / 2D scatter from ProjectionResult
# ---------------------------------------------------------------------------

def create_theater_figure(
    projection_result: ProjectionResult,
    color_by: np.ndarray | None = None,
    title: str = "",
    point_labels: list[str] | None = None,
) -> go.Figure:
    """Create an interactive scatter figure from a ProjectionResult.

    Args:
        projection_result: Result from any projection method (2D or 3D).
        color_by: Optional array for coloring points (e.g., zero index, spacing).
        title: Figure title.
        point_labels: Optional per-point hover labels.

    Returns:
        Plotly Figure with 3D scatter (target_dim==3) or 2D scatter.
    """
    coords = projection_result.coordinates
    method = projection_result.method
    meta = projection_result.metadata

    marker_kwargs: dict = dict(size=3, opacity=0.7)
    if color_by is not None:
        marker_kwargs.update(
            color=color_by,
            colorscale="Viridis",
            showscale=True,
        )
    else:
        marker_kwargs["color"] = ANALYTICAL_PALETTE["primary"]

    hover_template = None
    custom_data = None
    if point_labels is not None:
        custom_data = np.array(point_labels).reshape(-1, 1)
        hover_template = "%{customdata[0]}<extra></extra>"

    # Build axis labels from method and metadata
    axis_labels = _axis_labels(method, meta, projection_result.target_dim)

    if projection_result.target_dim >= 3:
        trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=marker_kwargs,
            customdata=custom_data,
            hovertemplate=hover_template,
        )
        layout = go.Layout(
            title=title or f"Projection Theater ({method})",
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
            ),
            template="plotly_white",
        )
    else:
        trace = go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords)),
            mode="markers",
            marker=marker_kwargs,
            customdata=custom_data,
            hovertemplate=hover_template,
        )
        layout = go.Layout(
            title=title or f"Projection Theater ({method})",
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1] if len(axis_labels) > 1 else "Dim 2",
            template="plotly_white",
        )

    # Annotation with method info
    method_info = f"Method: {method}"
    if "variance_explained" in meta:
        total_var = meta.get("total_variance_explained", 0)
        method_info += f" | Total variance: {total_var:.1%}"

    fig = go.Figure(data=[trace], layout=layout)
    fig.add_annotation(
        text=method_info,
        xref="paper", yref="paper",
        x=0.01, y=-0.05,
        showarrow=False,
        font=dict(size=9, color="gray"),
    )

    return fig


# ---------------------------------------------------------------------------
# Animation: interpolation between projection methods
# ---------------------------------------------------------------------------

def create_projection_path_animation(
    embedding: np.ndarray,
    methods: list[str] | None = None,
    n_frames: int = 20,
) -> go.Figure:
    """Animate smooth transitions between different projection methods.

    Linearly interpolates 3D coordinates between consecutive projections
    to reveal how structural patterns morph under different views.

    Args:
        embedding: ndarray of shape (n_points, n_features) with n_features >= 3.
        methods: List of projection method names (default: ["pca", "tsne", "umap"]).
        n_frames: Number of interpolation frames between each method pair.

    Returns:
        Plotly Figure with animation frames and play/pause controls.
    """
    if methods is None:
        methods = ["pca", "tsne", "umap"]

    # Import projection functions by name
    from riemann.viz.projection import project_pca, project_tsne, project_umap

    projectors = {
        "pca": project_pca,
        "tsne": project_tsne,
        "umap": project_umap,
    }

    # Compute all projections to 3D
    projections: list[tuple[str, np.ndarray]] = []
    for m in methods:
        if m not in projectors:
            raise ValueError(f"Unknown projection method '{m}'. Available: {list(projectors.keys())}")
        result = projectors[m](embedding, n_components=3)
        projections.append((m, result.coordinates))

    # Normalize coordinate ranges for smooth interpolation
    normalized = []
    for name, coords in projections:
        c = coords.copy()
        for dim in range(c.shape[1]):
            col = c[:, dim]
            span = col.max() - col.min()
            if span > 1e-12:
                c[:, dim] = (col - col.min()) / span
        normalized.append((name, c))

    # Build frames: for each pair of methods, interpolate
    frames = []
    all_coords = []

    for pair_idx in range(len(normalized) - 1):
        name_a, coords_a = normalized[pair_idx]
        name_b, coords_b = normalized[pair_idx + 1]
        for fi in range(n_frames):
            alpha = fi / max(n_frames - 1, 1)
            interpolated = (1.0 - alpha) * coords_a + alpha * coords_b
            label = f"{name_a} -> {name_b} ({alpha:.0%})"
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=interpolated[:, 0],
                    y=interpolated[:, 1],
                    z=interpolated[:, 2],
                    mode="markers",
                    marker=dict(size=3, opacity=0.7, color=ANALYTICAL_PALETTE["primary"]),
                )],
                name=label,
            ))
            if not all_coords:
                all_coords.append(interpolated)

    # Initial frame
    init_coords = normalized[0][1]
    init_trace = go.Scatter3d(
        x=init_coords[:, 0],
        y=init_coords[:, 1],
        z=init_coords[:, 2],
        mode="markers",
        marker=dict(size=3, opacity=0.7, color=ANALYTICAL_PALETTE["primary"]),
    )

    fig = go.Figure(
        data=[init_trace],
        frames=frames,
        layout=go.Layout(
            title="Projection Path Animation",
            scene=dict(
                xaxis_title="Dim 1",
                yaxis_title="Dim 2",
                zaxis_title="Dim 3",
            ),
            template="plotly_white",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, dict(
                                frame=dict(duration=100, redraw=True),
                                transition=dict(duration=50),
                                fromcurrent=True,
                                mode="immediate",
                            )],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            )],
                        ),
                    ],
                )
            ],
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Dimension slicing: fix some dims, project remaining
# ---------------------------------------------------------------------------

def create_dimension_slice_view(
    embedding: np.ndarray,
    fix_dims: dict[int, float],
    project_remaining: str = "pca",
) -> go.Figure:
    """Fix some dimensions and project the remaining.

    Selects points whose fixed dimensions fall within a tolerance of
    the target values, then projects the remaining dimensions.

    Args:
        embedding: ndarray of shape (n_points, n_features).
        fix_dims: Mapping ``{dim_index: target_value}`` for dimensions to fix.
        project_remaining: Projection method for remaining dims (default "pca").

    Returns:
        Plotly Figure showing the sliced, projected data.
    """
    from riemann.viz.projection import project_pca, project_tsne, project_umap

    projectors = {"pca": project_pca, "tsne": project_tsne, "umap": project_umap}
    if project_remaining not in projectors:
        raise ValueError(f"Unknown method '{project_remaining}'. Available: {list(projectors.keys())}")

    # Identify points within tolerance for all fixed dimensions
    mask = np.ones(embedding.shape[0], dtype=bool)
    for dim_idx, target_val in fix_dims.items():
        col = embedding[:, dim_idx]
        tol = 0.5 * np.std(col) if np.std(col) > 1e-12 else 0.5
        mask &= np.abs(col - target_val) <= tol

    # Extract remaining dimensions
    all_dims = list(range(embedding.shape[1]))
    remaining_dims = [d for d in all_dims if d not in fix_dims]

    if len(remaining_dims) == 0:
        remaining_dims = all_dims  # fallback: keep all if all are fixed

    sliced_data = embedding[mask][:, remaining_dims]

    if sliced_data.shape[0] < 3:
        # Not enough points -- return an empty figure with explanation
        fig = go.Figure()
        fig.update_layout(
            title="Dimension Slice: too few points matched",
            annotations=[dict(
                text=f"Only {sliced_data.shape[0]} points within tolerance. "
                     f"Try adjusting fix_dims values.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14),
            )],
            template="plotly_white",
        )
        return fig

    # Determine target dim for projection
    target_dim = min(3, sliced_data.shape[1], sliced_data.shape[0])
    if target_dim < 2:
        target_dim = 2

    proj_result = projectors[project_remaining](sliced_data, n_components=target_dim)

    fig = create_theater_figure(
        proj_result,
        title=f"Dimension Slice ({project_remaining})",
    )

    # Annotate which dims are fixed
    fix_desc = ", ".join(f"dim{k}={v:.2f}" for k, v in fix_dims.items())
    fig.add_annotation(
        text=f"Fixed: {fix_desc} | {mask.sum()} / {len(mask)} points",
        xref="paper", yref="paper",
        x=0.01, y=1.05,
        showarrow=False,
        font=dict(size=10),
    )

    return fig


# ---------------------------------------------------------------------------
# Side-by-side comparison of multiple projections
# ---------------------------------------------------------------------------

def create_side_by_side(
    projections: dict[str, ProjectionResult],
    color_by: np.ndarray | None = None,
    title: str = "",
) -> go.Figure:
    """Show multiple projections side-by-side for comparison.

    Creates a 1-row subplot layout. Uses first 2 components for each
    projection so all panels are consistent 2D views.

    Args:
        projections: Mapping ``{name: ProjectionResult}``.
        color_by: Optional coloring array (same for all panels).
        title: Overall figure title.

    Returns:
        Plotly Figure with linked subplots.
    """
    names = list(projections.keys())
    n_cols = len(names)

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=names,
    )

    marker_kwargs: dict = dict(size=3, opacity=0.7)
    if color_by is not None:
        marker_kwargs.update(
            color=color_by,
            colorscale="Viridis",
            showscale=False,
        )
    else:
        marker_kwargs["color"] = ANALYTICAL_PALETTE["primary"]

    for col_idx, name in enumerate(names, start=1):
        pr = projections[name]
        coords = pr.coordinates
        x = coords[:, 0]
        y = coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords))

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=marker_kwargs,
                name=name,
                customdata=np.arange(len(x)).reshape(-1, 1),
                hovertemplate="idx=%{customdata[0]}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        title=title or "Side-by-Side Projection Comparison",
        template="plotly_white",
        showlegend=False,
        hovermode="closest",
    )

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axis_labels(method: str, metadata: dict, ndim: int) -> list[str]:
    """Generate axis labels based on projection method and metadata."""
    if method == "pca" and "variance_explained" in metadata:
        var = metadata["variance_explained"]
        labels = []
        for i in range(ndim):
            if i < len(var):
                labels.append(f"PC{i+1} ({var[i]:.1%})")
            else:
                labels.append(f"PC{i+1}")
        return labels

    # Generic labels for t-SNE, UMAP, stereographic, etc.
    return [f"Dim {i+1}" for i in range(ndim)]
