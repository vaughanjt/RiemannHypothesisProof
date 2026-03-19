"""Projection pipeline: PCA, t-SNE, UMAP, stereographic, Hopf fibration.

All projection methods accept an ndarray and return a ProjectionResult
dataclass with coordinates, method name, dimensions, and metadata dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProjectionResult:
    """Result of a dimensionality reduction / projection.

    Attributes:
        coordinates: ndarray of shape (n_points, target_dim).
        method: Name of the projection method.
        source_dim: Dimensionality of the input data.
        target_dim: Dimensionality of the output coordinates.
        metadata: Method-specific metadata (variance explained, KL divergence, etc.).
    """
    coordinates: np.ndarray
    method: str
    source_dim: int
    target_dim: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def project_pca(
    data: np.ndarray,
    n_components: int = 3,
) -> ProjectionResult:
    """Project data using Principal Component Analysis.

    Args:
        data: ndarray of shape (n_points, n_features).
        n_components: Number of output dimensions.

    Returns:
        ProjectionResult with PCA coordinates and variance_explained metadata.
    """
    from sklearn.decomposition import PCA

    # Clamp n_components to valid range
    n_components = min(n_components, data.shape[1], data.shape[0])

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(data)

    return ProjectionResult(
        coordinates=coords,
        method="pca",
        source_dim=data.shape[1],
        target_dim=n_components,
        metadata={
            "variance_explained": pca.explained_variance_ratio_.tolist(),
            "total_variance_explained": float(sum(pca.explained_variance_ratio_)),
            "components": pca.components_.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def project_tsne(
    data: np.ndarray,
    n_components: int = 3,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> ProjectionResult:
    """Project data using t-SNE.

    Auto-reduces perplexity if n_samples < 3 * perplexity.
    Pre-reduces with PCA(50) if input has > 50 features (sklearn recommendation).

    Args:
        data: ndarray of shape (n_points, n_features).
        n_components: Number of output dimensions (2 or 3).
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for reproducibility.

    Returns:
        ProjectionResult with t-SNE coordinates and KL divergence metadata.
    """
    from sklearn.manifold import TSNE

    pre_reduced = False

    # Pre-reduce with PCA if high-dimensional
    if data.shape[1] > 50:
        from sklearn.decomposition import PCA
        data = PCA(n_components=50, random_state=random_state).fit_transform(data)
        pre_reduced = True

    # Auto-reduce perplexity for small datasets
    effective_perplexity = perplexity
    n_samples = data.shape[0]
    if n_samples < 3 * perplexity:
        effective_perplexity = max(1.0, (n_samples - 1) / 3.0)

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        random_state=random_state,
    )
    coords = tsne.fit_transform(data)

    return ProjectionResult(
        coordinates=coords,
        method="tsne",
        source_dim=data.shape[1],
        target_dim=n_components,
        metadata={
            "perplexity": perplexity,
            "effective_perplexity": effective_perplexity,
            "kl_divergence": float(tsne.kl_divergence_),
            "pre_reduced": pre_reduced,
        },
    )


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def project_umap(
    data: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> ProjectionResult:
    """Project data using UMAP.

    Auto-reduces n_neighbors if n_samples < n_neighbors.

    Args:
        data: ndarray of shape (n_points, n_features).
        n_components: Number of output dimensions.
        n_neighbors: Number of nearest neighbors for UMAP.
        min_dist: Minimum distance parameter.
        random_state: Random seed for reproducibility.

    Returns:
        ProjectionResult with UMAP coordinates.
    """
    import umap

    # Auto-reduce n_neighbors for small datasets
    effective_neighbors = min(n_neighbors, data.shape[0] - 1)
    effective_neighbors = max(effective_neighbors, 2)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(data)

    return ProjectionResult(
        coordinates=coords,
        method="umap",
        source_dim=data.shape[1],
        target_dim=n_components,
        metadata={
            "n_neighbors": n_neighbors,
            "effective_n_neighbors": effective_neighbors,
            "min_dist": min_dist,
        },
    )


# ---------------------------------------------------------------------------
# Stereographic projection
# ---------------------------------------------------------------------------

def project_stereographic(data: np.ndarray) -> ProjectionResult:
    """Stereographic projection from the north pole.

    Normalizes rows to the unit sphere, then projects from the north pole:
    x_i / (1 - x_n) for i < n.

    Reduces dimension by 1: R^n -> R^(n-1).

    Args:
        data: ndarray of shape (n_points, n_features) with n_features >= 2.

    Returns:
        ProjectionResult with (n_features - 1) dimensional coordinates.
    """
    source_dim = data.shape[1]

    # Normalize to unit sphere
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)  # avoid division by zero
    sphere = data / norms

    # Stereographic from north pole: x_i / (1 - x_n) for i < n
    x_n = sphere[:, -1:]
    denom = 1.0 - x_n
    # Clip denominator to avoid division by zero (points near north pole)
    denom = np.clip(denom, 1e-10, None)

    coords = sphere[:, :-1] / denom

    return ProjectionResult(
        coordinates=coords,
        method="stereographic",
        source_dim=source_dim,
        target_dim=source_dim - 1,
        metadata={
            "projection_pole": "north",
        },
    )


# ---------------------------------------------------------------------------
# Hopf fibration: S^3 -> S^2
# ---------------------------------------------------------------------------

def project_hopf_fibration(data: np.ndarray) -> ProjectionResult:
    """Hopf fibration: map 4D data (S^3) to 3D (S^2).

    Custom mathematical projection per user decision. The Hopf map
    sends S^3 to S^2 by treating S^3 as pairs (z1, z2) in C^2 and
    computing [z1:z2] in CP^1 ~ S^2.

    Given (z1, z2) = (x0+ix1, x2+ix3) with |z1|^2+|z2|^2=1:
        eta_1 = 2 * Re(z1 * conj(z2)) = 2*(x0*x2 + x1*x3)
        eta_2 = 2 * Im(z1 * conj(z2)) = 2*(x1*x2 - x0*x3)
        eta_3 = |z1|^2 - |z2|^2 = x0^2 + x1^2 - x2^2 - x3^2

    The result lies on S^2. The S^1 fiber phase (lost in projection)
    is stored in metadata for downstream coloring.

    Args:
        data: ndarray of shape (n_points, 4). Must have exactly 4 columns.

    Returns:
        ProjectionResult with 3D coordinates on S^2 and fiber_phase metadata.

    Raises:
        ValueError: If data does not have exactly 4 columns.
    """
    if data.shape[1] != 4:
        raise ValueError(
            f"Hopf fibration requires 4D input, got {data.shape[1]}D. "
            f"Data shape: {data.shape}"
        )

    # Normalize each row to unit S^3
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    s3 = data / norms

    x0, x1, x2, x3 = s3[:, 0], s3[:, 1], s3[:, 2], s3[:, 3]

    # Hopf map S^3 -> S^2
    eta_1 = 2.0 * (x0 * x2 + x1 * x3)
    eta_2 = 2.0 * (x1 * x2 - x0 * x3)
    eta_3 = x0**2 + x1**2 - x2**2 - x3**2

    coords = np.column_stack([eta_1, eta_2, eta_3])

    # Fiber phase: relative phase angle between z1 and z2
    # angle = atan2(x1, x0) - atan2(x3, x2)
    phase_z1 = np.arctan2(x1, x0)
    phase_z2 = np.arctan2(x3, x2)
    fiber_phase = phase_z1 - phase_z2

    return ProjectionResult(
        coordinates=coords,
        method="hopf_fibration",
        source_dim=4,
        target_dim=3,
        metadata={
            "projection_type": "hopf_fibration",
            "fiber_phase": fiber_phase.tolist(),
        },
    )
