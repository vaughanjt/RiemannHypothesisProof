"""Random matrix theory ensemble generation and eigenvalue statistics.

Provides GUE/GOE/GSE ensemble generation, eigenvalue spacing statistics
with semicircle-law unfolding, Wigner surmise distributions, and
residual analysis tools for comparing zero spacings against RMT predictions.

The eigenvalue_spacings output format (np.ndarray of normalized spacings)
matches the convention used by spacing.py for direct comparison.
"""

from __future__ import annotations

import numpy as np


def generate_gue(
    n: int, num_matrices: int = 100, seed: int | None = None
) -> list[np.ndarray]:
    """Generate GUE (Gaussian Unitary Ensemble) eigenvalues.

    GUE(N): H = (A + A*) / (2*sqrt(N)) where A has i.i.d. complex Gaussian entries
    with variance 1/2 per real/imaginary component.

    Args:
        n: Matrix dimension.
        num_matrices: Number of random matrices to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of eigenvalue arrays, each of shape (n,), sorted ascending.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(num_matrices):
        # Complex Gaussian entries: real and imaginary parts each ~ N(0, 1/2)
        a = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(
            2
        )
        # Hermitianize and normalize
        h = (a + a.conj().T) / (2 * np.sqrt(n))
        # eigvalsh returns real eigenvalues sorted ascending for Hermitian matrices
        eigs = np.linalg.eigvalsh(h)
        results.append(eigs)
    return results


def generate_goe(
    n: int, num_matrices: int = 100, seed: int | None = None
) -> list[np.ndarray]:
    """Generate GOE (Gaussian Orthogonal Ensemble) eigenvalues.

    GOE(N): H = (A + A^T) / (2*sqrt(N)) where A has i.i.d. real Gaussian entries.

    Args:
        n: Matrix dimension.
        num_matrices: Number of random matrices to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of eigenvalue arrays, each of shape (n,), sorted ascending.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(num_matrices):
        a = rng.standard_normal((n, n))
        h = (a + a.T) / (2 * np.sqrt(n))
        eigs = np.linalg.eigvalsh(h)
        results.append(eigs)
    return results


def generate_gse(
    n: int, num_matrices: int = 100, seed: int | None = None
) -> list[np.ndarray]:
    """Generate GSE (Gaussian Symplectic Ensemble) eigenvalues.

    GSE(N): Uses quaternion structure via 2N x 2N block representation.
    H_q = [[A, B], [-B*, A*]] made Hermitian, eigenvalues come in degenerate pairs.

    Args:
        n: Matrix dimension (returns n eigenvalues from 2n x 2n matrix).
        num_matrices: Number of random matrices to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of eigenvalue arrays, each of shape (n,), sorted ascending.
        Only unique eigenvalues returned (degenerate pairs collapsed).
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(num_matrices):
        # Build quaternion-structured random matrix
        a = (
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        ) / np.sqrt(2)
        b = (
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        ) / np.sqrt(2)

        # Assemble 2n x 2n block matrix
        h_q = np.block([[a, b], [-b.conj(), a.conj()]])

        # Hermitianize and normalize
        h = (h_q + h_q.conj().T) / (2 * np.sqrt(2 * n))

        # Eigenvalues come in degenerate pairs; take every other one
        all_eigs = np.linalg.eigvalsh(h)
        eigs = all_eigs[::2]  # Take unique eigenvalues
        results.append(eigs)
    return results


def _unfold_semicircle(eigenvalues: np.ndarray, n: int) -> np.ndarray:
    """Unfold eigenvalues using Wigner semicircle law.

    Maps eigenvalues to have uniform density via the semicircle CDF,
    so that spacings become comparable across different regions of the spectrum.

    For the semicircle distribution on [-2, 2]:
        CDF(x) = 1/2 + x*sqrt(4 - x^2)/(4*pi) + arcsin(x/2)/pi

    We use only the bulk of the spectrum (drop edges where the semicircle
    density vanishes and unfolding becomes unreliable). The fraction of
    eigenvalues kept is chosen to balance statistical power against edge effects.

    Args:
        eigenvalues: Sorted eigenvalue array from a single matrix.
        n: Matrix dimension (for scaling).

    Returns:
        Unfolded eigenvalues with approximately uniform density.
    """
    # The eigenvalues are already on approximate semicircle support [-2, 2]
    # due to the 1/(2*sqrt(n)) normalization in ensemble generation.
    scaled = eigenvalues.copy()

    # Use central fraction of eigenvalues to avoid edge effects.
    # The semicircle density vanishes at +/-2, causing unfolding instability
    # at the spectral edges. Keep the central ~80% of eigenvalues (by index).
    total = len(scaled)
    trim = max(1, int(total * 0.1))  # Trim 10% from each edge
    if total - 2 * trim < 3:
        # For very small matrices, keep everything
        trim = 0
    bulk = scaled[trim : total - trim] if trim > 0 else scaled

    # Semicircle CDF: F(x) = 1/2 + x*sqrt(4-x^2)/(4*pi) + arcsin(x/2)/pi
    x_clipped = np.clip(bulk, -2.0 + 1e-10, 2.0 - 1e-10)
    unfolded = (
        0.5
        + x_clipped * np.sqrt(4.0 - x_clipped**2) / (4.0 * np.pi)
        + np.arcsin(x_clipped / 2.0) / np.pi
    )

    # Scale to [0, N_bulk] so mean spacing is approximately 1
    n_bulk = len(unfolded)
    unfolded = unfolded * n_bulk

    return unfolded


def eigenvalue_spacings(eigenvalues_list: list[np.ndarray]) -> np.ndarray:
    """Compute normalized eigenvalue spacings from ensemble of matrices.

    For each eigenvalue array: unfold using semicircle law, compute consecutive
    spacings, normalize by mean spacing. Concatenate all normalized spacings
    from all matrices.

    Output format matches spacing.normalized_spacings convention: a single
    np.ndarray of normalized spacings with mean approximately 1.0.

    Args:
        eigenvalues_list: List of eigenvalue arrays (one per matrix).

    Returns:
        Single np.ndarray of normalized spacings.
    """
    all_spacings = []
    for eigs in eigenvalues_list:
        n = len(eigs)
        unfolded = _unfold_semicircle(eigs, n)
        if len(unfolded) < 2:
            continue
        spacings = np.diff(unfolded)
        # Remove any non-positive spacings (numerical artifacts)
        spacings = spacings[spacings > 0]
        if len(spacings) > 0:
            all_spacings.append(spacings)

    if not all_spacings:
        return np.array([])

    combined = np.concatenate(all_spacings)

    # Normalize by mean spacing so mean is 1.0
    mean_spacing = combined.mean()
    if mean_spacing > 0:
        combined = combined / mean_spacing

    return combined


def wigner_surmise(s: np.ndarray, beta: int = 2) -> np.ndarray:
    """Wigner surmise probability density for nearest-neighbor spacing.

    Theoretical prediction for the spacing distribution of random matrix
    ensembles in the large-N limit.

    Args:
        s: Spacing values (non-negative).
        beta: Dyson index (1=GOE, 2=GUE, 4=GSE).

    Returns:
        Probability density values at each spacing.

    Raises:
        ValueError: If beta is not 1, 2, or 4.
    """
    s = np.asarray(s, dtype=float)

    if beta == 1:
        # GOE: p(s) = (pi/2) * s * exp(-pi*s^2/4)
        return (np.pi / 2.0) * s * np.exp(-np.pi * s**2 / 4.0)
    elif beta == 2:
        # GUE: p(s) = (32/pi^2) * s^2 * exp(-4*s^2/pi)
        return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)
    elif beta == 4:
        # GSE: p(s) = (2^18 / (3^6 * pi^3)) * s^4 * exp(-64*s^2/(9*pi))
        return (
            (2**18 / (3**6 * np.pi**3)) * s**4 * np.exp(-64.0 * s**2 / (9.0 * np.pi))
        )
    else:
        raise ValueError(
            f"beta must be 1 (GOE), 2 (GUE), or 4 (GSE), got beta={beta}"
        )


def fit_effective_n(
    target_spacings: np.ndarray,
    n_range: tuple[int, int] = (10, 500),
    num_matrices: int = 200,
    seed: int = 42,
) -> dict:
    """Find the GUE matrix size N whose spacing distribution best matches target.

    For each candidate N in range, generates a GUE ensemble, computes
    spacing distribution, and measures chi-squared distance against the
    target spacing histogram. This is the residual analysis tool:
    "at what effective N do zeros best match GUE?"

    Args:
        target_spacings: Array of normalized spacings to fit against.
        n_range: (min_n, max_n) range of matrix sizes to try.
        num_matrices: Number of matrices per candidate N.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            - "best_n": int, optimal matrix size
            - "chi_squared": float, best chi-squared value
            - "all_fits": dict mapping N -> chi-squared value
    """
    bins = np.linspace(0, 4, 41)
    target_hist, _ = np.histogram(target_spacings, bins=bins, density=True)

    all_fits: dict[int, float] = {}
    best_n = n_range[0]
    best_chi_sq = float("inf")

    rng_seed = seed
    for n in range(n_range[0], n_range[1], 10):
        eigs = generate_gue(n=n, num_matrices=num_matrices, seed=rng_seed)
        spacings = eigenvalue_spacings(eigs)
        rng_seed += 1  # Different seed per N to avoid correlation

        if len(spacings) == 0:
            continue

        candidate_hist, _ = np.histogram(spacings, bins=bins, density=True)

        # Chi-squared statistic: sum((observed - expected)^2 / expected)
        # Use target as "expected", candidate as "observed"
        mask = target_hist > 0.01  # Avoid division by near-zero
        if mask.sum() == 0:
            continue

        chi_sq = float(
            np.sum((candidate_hist[mask] - target_hist[mask]) ** 2 / target_hist[mask])
        )
        all_fits[n] = chi_sq

        if chi_sq < best_chi_sq:
            best_chi_sq = chi_sq
            best_n = n

    return {
        "best_n": int(best_n),
        "chi_squared": float(best_chi_sq),
        "all_fits": all_fits,
    }
