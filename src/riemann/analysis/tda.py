"""Topological data analysis via persistent homology.

Provides persistent homology computation using ripser, persistence diagram
analysis, and cross-embedding diagram comparison using bottleneck distance.

Function-based API with PersistenceResult dataclass for structured output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ripser import ripser


@dataclass
class PersistenceResult:
    """Result from persistent homology computation.

    Attributes:
        diagrams: Persistence diagrams per dimension. Each is an ndarray
            of shape (n_features, 2) with columns (birth, death).
        num_features: Count of finite-lifetime features per dimension.
        total_persistence: Sum of finite lifetimes per dimension.
        metadata: Computation parameters (max_dim, max_edge, n_points, etc.).
    """

    diagrams: list[np.ndarray]
    num_features: dict[int, int]
    total_persistence: dict[int, float]
    metadata: dict


def compute_persistence(
    points: np.ndarray,
    max_dim: int = 2,
    max_edge: float | None = None,
) -> PersistenceResult:
    """Compute persistent homology of a point cloud.

    Args:
        points: 2D array of shape (N, D) where N >= 2 points in D dimensions.
        max_dim: Maximum homology dimension to compute.
        max_edge: Maximum edge length (threshold). None means np.inf.

    Returns:
        PersistenceResult with diagrams, feature counts, and total persistence.

    Raises:
        ValueError: If points is empty, 1D, or has fewer than 2 points.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim == 0 or points.size == 0:
        raise ValueError("Input points array is empty")
    if points.ndim != 2:
        raise ValueError(
            f"Input must be a 2D array of shape (N, D), got shape {points.shape}"
        )
    if points.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 points for persistent homology, got {points.shape[0]}"
        )

    thresh = max_edge if max_edge is not None else np.inf
    rips_result = ripser(points, maxdim=max_dim, thresh=thresh)

    diagrams = rips_result["dgms"]

    # Compute per-dimension statistics
    num_features: dict[int, int] = {}
    total_persistence: dict[int, float] = {}

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            num_features[dim] = 0
            total_persistence[dim] = 0.0
            continue

        lifetimes = dgm[:, 1] - dgm[:, 0]
        finite_mask = np.isfinite(lifetimes)
        finite_lifetimes = lifetimes[finite_mask]

        num_features[dim] = int(len(finite_lifetimes))
        total_persistence[dim] = float(np.sum(finite_lifetimes))

    metadata = {
        "n_points": int(points.shape[0]),
        "n_dims": int(points.shape[1]),
        "max_dim": max_dim,
        "max_edge": max_edge,
    }

    return PersistenceResult(
        diagrams=diagrams,
        num_features=num_features,
        total_persistence=total_persistence,
        metadata=metadata,
    )


def persistence_summary(
    result: PersistenceResult,
    significance_threshold: float = 0.1,
) -> dict:
    """Summarize persistent homology results.

    Args:
        result: PersistenceResult from compute_persistence.
        significance_threshold: Fraction of max_lifetime below which features
            are considered insignificant.

    Returns:
        Dict with keys:
            - "dominant_dimension": dimension with largest total persistence
              (excluding H_0 which always dominates).
            - "max_lifetime": largest finite birth-death gap across all dims.
            - "n_significant": count of features with lifetime > threshold * max_lifetime.
            - "feature_counts": num_features dict from result.
    """
    max_lifetime = 0.0
    all_lifetimes: list[float] = []

    for dim, dgm in enumerate(result.diagrams):
        if len(dgm) == 0:
            continue
        lifetimes = dgm[:, 1] - dgm[:, 0]
        finite_mask = np.isfinite(lifetimes)
        finite_lifetimes = lifetimes[finite_mask]
        if len(finite_lifetimes) > 0:
            dim_max = float(np.max(finite_lifetimes))
            if dim_max > max_lifetime:
                max_lifetime = dim_max
            all_lifetimes.extend(finite_lifetimes.tolist())

    # Find dominant dimension (excluding H_0)
    dominant_dimension = 0
    best_persistence = 0.0
    for dim, total_pers in result.total_persistence.items():
        if dim == 0:
            continue  # Skip H_0 (always dominates due to connected components)
        if total_pers > best_persistence:
            best_persistence = total_pers
            dominant_dimension = dim

    # If no higher dimensions have persistence, fall back to 0
    if best_persistence == 0.0 and 0 in result.total_persistence:
        dominant_dimension = 0

    # Count significant features
    threshold = significance_threshold * max_lifetime
    n_significant = sum(1 for lt in all_lifetimes if lt > threshold)

    return {
        "dominant_dimension": dominant_dimension,
        "max_lifetime": max_lifetime,
        "n_significant": n_significant,
        "feature_counts": dict(result.num_features),
    }


def compare_persistence_diagrams(
    result1: PersistenceResult,
    result2: PersistenceResult,
    dimension: int = 1,
) -> dict:
    """Compare persistence diagrams using bottleneck distance.

    Args:
        result1: First PersistenceResult.
        result2: Second PersistenceResult.
        dimension: Homology dimension to compare.

    Returns:
        Dict with:
            - "bottleneck_distance": float distance between diagrams.
            - "dimension": int dimension compared.
            - "n_features_1": int feature count in result1.
            - "n_features_2": int feature count in result2.
    """
    from persim import bottleneck

    # Extract diagrams for the given dimension
    dgm1 = result1.diagrams[dimension] if dimension < len(result1.diagrams) else np.empty((0, 2))
    dgm2 = result2.diagrams[dimension] if dimension < len(result2.diagrams) else np.empty((0, 2))

    # Filter to finite features only
    if len(dgm1) > 0:
        finite1 = np.isfinite(dgm1[:, 1])
        dgm1 = dgm1[finite1]
    if len(dgm2) > 0:
        finite2 = np.isfinite(dgm2[:, 1])
        dgm2 = dgm2[finite2]

    n1 = len(dgm1)
    n2 = len(dgm2)

    # Handle edge case: if either diagram is empty
    if n1 == 0 and n2 == 0:
        dist = 0.0
    elif n1 == 0 or n2 == 0:
        # Distance from empty diagram to non-empty:
        # each point matched to diagonal, distance = max lifetime / 2
        non_empty = dgm1 if n1 > 0 else dgm2
        lifetimes = non_empty[:, 1] - non_empty[:, 0]
        dist = float(np.max(lifetimes) / 2.0) if len(lifetimes) > 0 else 0.0
    else:
        dist = float(bottleneck(dgm1, dgm2))

    return {
        "bottleneck_distance": dist,
        "dimension": dimension,
        "n_features_1": n1,
        "n_features_2": n2,
    }
