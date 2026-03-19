"""Spectral operator analysis for the Berry-Keating Hamiltonian.

Constructs candidate operators whose eigenvalues might match zeta zeros
(the Hilbert-Polya conjecture pathway). Provides:
- Berry-Keating Hamiltonian with box and smooth regularizations
- Eigenvalue spectrum computation
- Spectral comparison against zeta zero spacings (chi-squared, KS)

The Berry-Keating operator H = (xp + px)/2 is a symmetrized version of
the operator xp, whose classical eigenvalues on a suitable domain would
be the Riemann zeros if the Hilbert-Polya conjecture holds.

Function-based API. Returns numpy arrays and dataclasses, never plots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.stats import ks_2samp


@dataclass
class SpectralResult:
    """Result of a spectral computation on an operator matrix.

    Attributes:
        eigenvalues: Sorted eigenvalue array (ascending).
        operator_name: Human-readable name of the operator.
        matrix_size: Dimension of the matrix.
        chi_squared_fit: Chi-squared fit against a reference distribution (0.0 if not compared).
        metadata: Additional information about the computation.
    """

    eigenvalues: np.ndarray
    operator_name: str
    matrix_size: int
    chi_squared_fit: float = 0.0
    metadata: dict = field(default_factory=dict)


def construct_berry_keating_box(n: int, L: float = 10.0) -> np.ndarray:
    """Construct the Berry-Keating Hamiltonian with box boundary conditions.

    Discretizes H = (xp + px)/2 on the interval (0, L] using central
    finite differences. The operator xp is the generator of dilations;
    its symmetrization yields a self-adjoint operator on L^2.

    The grid avoids x=0 (where the operator is singular) by starting
    at x = L/n.

    Args:
        n: Matrix dimension (number of grid points).
        L: Length of the spatial domain.

    Returns:
        Real symmetric (n, n) numpy array representing the discretized Hamiltonian.
    """
    dx = L / n
    # Grid points: x_i = (i+1) * dx for i = 0, ..., n-1
    # This places points at dx, 2*dx, ..., L, avoiding x=0
    x = np.linspace(dx, L, n)

    # Discretize p = -i * d/dx using central finite differences
    # H = (xp + px)/2: symmetrized momentum operator
    # H_{i,j} involves x_i * (finite diff at i) terms
    # Central difference: df/dx at i ~ (f_{i+1} - f_{i-1}) / (2*dx)
    # For H = (xp + px)/2 discretized:
    #   H_{i,i-1} = -x_i / (2*dx)   (from xp part)
    #   H_{i,i+1} =  x_i / (2*dx)   (from xp part)
    # Then symmetrize: H_sym = (H + H^T) / 2

    H = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        if i > 0:
            H[i, i - 1] = -x[i] / (2.0 * dx)
        if i < n - 1:
            H[i, i + 1] = x[i] / (2.0 * dx)

    # Symmetrize to ensure real symmetric matrix
    H_sym = (H + H.T) / 2.0

    return H_sym


def construct_berry_keating_smooth(
    n: int, L: float = 10.0, V_strength: float = 0.1
) -> np.ndarray:
    """Construct the Berry-Keating Hamiltonian with smooth confining potential.

    Same as the box version but adds a quadratic confining potential
    V(x) = V_strength * x^2 along the diagonal. This smooth regularization
    replaces the hard-wall boundary condition and produces a discrete spectrum
    without boundary artifacts.

    Args:
        n: Matrix dimension (number of grid points).
        L: Length of the spatial domain.
        V_strength: Strength of the confining potential V(x) = V_strength * x^2.

    Returns:
        Real symmetric (n, n) numpy array representing the regularized Hamiltonian.
    """
    H_sym = construct_berry_keating_box(n, L)

    dx = L / n
    x = np.linspace(dx, L, n)

    # Add smooth confining potential along diagonal
    H_smooth = H_sym + np.diag(V_strength * x**2)

    return H_smooth


def compute_spectrum(
    matrix: np.ndarray, operator_name: str = "unknown"
) -> SpectralResult:
    """Compute the eigenvalue spectrum of a symmetric/Hermitian matrix.

    Uses scipy.linalg.eigh for dense real symmetric or complex Hermitian
    matrices. Returns eigenvalues sorted in ascending order.

    Args:
        matrix: Square symmetric/Hermitian matrix.
        operator_name: Human-readable name for the operator.

    Returns:
        SpectralResult with sorted eigenvalues and metadata.
    """
    eigenvalues = eigh(matrix, eigvals_only=True)
    # eigh returns eigenvalues sorted ascending by default
    eigenvalues = np.sort(eigenvalues)

    return SpectralResult(
        eigenvalues=eigenvalues,
        operator_name=operator_name,
        matrix_size=matrix.shape[0],
        chi_squared_fit=0.0,
        metadata={},
    )


def spectral_comparison(
    eigenvalues: np.ndarray, zero_heights: np.ndarray
) -> dict:
    """Compare eigenvalue spacing distribution against zeta zero spacing distribution.

    Normalizes both spacing distributions to mean 1, computes chi-squared
    distance on 40-bin histograms over [0, 4], and the Kolmogorov-Smirnov
    test statistic.

    Args:
        eigenvalues: Array of eigenvalues or eigenvalue spacings.
        zero_heights: Array of zeta zero heights or zero spacings.

    Returns:
        Dict with keys:
            - "chi_squared": Chi-squared distance between spacing histograms.
            - "ks_statistic": KS test statistic.
            - "ks_pvalue": KS test p-value.
            - "n_eigenvalues": Number of eigenvalues provided.
            - "n_zeros": Number of zeros provided.
    """
    n_eig = len(eigenvalues)
    n_zeros = len(zero_heights)

    # Compute spacings if not already spacings (if sorted, diff gives spacings)
    eig_spacings = np.diff(np.sort(eigenvalues))
    zero_spacings = np.diff(np.sort(zero_heights))

    # Normalize spacings to mean 1
    eig_mean = eig_spacings.mean() if len(eig_spacings) > 0 else 1.0
    zero_mean = zero_spacings.mean() if len(zero_spacings) > 0 else 1.0

    if eig_mean > 0:
        eig_norm = eig_spacings / eig_mean
    else:
        eig_norm = eig_spacings

    if zero_mean > 0:
        zero_norm = zero_spacings / zero_mean
    else:
        zero_norm = zero_spacings

    # Chi-squared on 40-bin histograms over [0, 4]
    bins = np.linspace(0, 4, 41)
    eig_hist, _ = np.histogram(eig_norm, bins=bins, density=True)
    zero_hist, _ = np.histogram(zero_norm, bins=bins, density=True)

    # Chi-squared: sum((obs - exp)^2 / exp) where exp > threshold
    # Use zero_hist as expected, eig_hist as observed
    threshold = 0.01
    mask = zero_hist > threshold
    if mask.sum() > 0:
        chi_squared = float(
            np.sum((eig_hist[mask] - zero_hist[mask]) ** 2 / zero_hist[mask])
        )
    else:
        # Fallback: use eig_hist as expected
        mask2 = eig_hist > threshold
        if mask2.sum() > 0:
            chi_squared = float(
                np.sum((eig_hist[mask2] - zero_hist[mask2]) ** 2 / eig_hist[mask2])
            )
        else:
            chi_squared = 0.0

    # KS test on normalized spacings
    ks_stat, ks_pvalue = ks_2samp(eig_norm, zero_norm)

    return {
        "chi_squared": chi_squared,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "n_eigenvalues": n_eig,
        "n_zeros": n_zeros,
    }
