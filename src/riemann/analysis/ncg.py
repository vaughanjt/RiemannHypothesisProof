"""Noncommutative geometry module: Bost-Connes quantum statistical mechanical system.

Implements Connes' approach to the Riemann Hypothesis through the Bost-Connes system,
where the partition function equals zeta(beta) and a phase transition at beta=1
connects to the distribution of primes.

Key objects:
- Partition function Z(beta) = sum_{n=1}^{N} n^{-beta}, converging to zeta(beta)
- KMS (Kubo-Martin-Schwinger) equilibrium states: phi_beta(e_n) = n^{-beta}/Z(beta)
- Phase transition at beta=1: discontinuity in KMS state entropy

Function-based API. Returns numpy arrays and dataclass results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mpmath
import numpy as np


@dataclass
class BostConnesResult:
    """Result from Bost-Connes system computation.

    Attributes:
        beta: Inverse temperature parameter.
        partition_value: Z(beta) = sum n^{-beta}.
        kms_values: KMS state probabilities p_n = n^{-beta}/Z(beta).
        zeta_comparison: |Z(beta) - zeta(beta)| error measure.
        metadata: Additional computation metadata.
    """

    beta: float
    partition_value: float
    kms_values: np.ndarray
    zeta_comparison: float
    metadata: dict


def _validate_beta(beta: float) -> None:
    """Validate that beta > 1 for convergence."""
    if beta <= 1.0:
        raise ValueError(
            f"beta must be > 1 for convergence of the Bost-Connes partition function, "
            f"got beta={beta}. The sum sum(n^{{-beta}}) diverges for beta <= 1."
        )


def _partial_sum(beta: float, n_max: int) -> float:
    """Raw partial sum sum_{n=1}^{n_max} n^{-beta} without tail correction."""
    ns = np.arange(1, n_max + 1, dtype=np.float64)
    return float(np.sum(ns ** (-beta)))


def bost_connes_partition(beta: float, n_max: int = 1000) -> float:
    """Compute Bost-Connes partition function Z(beta) = sum_{n=1}^{n_max} n^{-beta}.

    For beta > 1 this converges to the Riemann zeta function zeta(beta).
    For beta <= 1 the sum diverges (harmonic series or worse).

    Includes Euler-Maclaurin tail correction for better convergence to zeta(beta).

    Args:
        beta: Inverse temperature. Must be > 1 for convergence.
        n_max: Number of terms in the partial sum.

    Returns:
        Float value of the tail-corrected sum (approximation to zeta(beta)).

    Raises:
        ValueError: If beta <= 1 (divergent regime).
    """
    _validate_beta(beta)

    partial = _partial_sum(beta, n_max)

    # Euler-Maclaurin tail correction: approximate sum_{n=n_max+1}^{inf} n^{-beta}
    # by integral from n_max + 0.5 to infinity of x^{-beta} dx
    # = (n_max + 0.5)^{1-beta} / (beta - 1)
    tail = (n_max + 0.5) ** (1.0 - beta) / (beta - 1.0)
    return partial + tail


def bost_connes_kms_values(beta: float, n_max: int = 50) -> np.ndarray:
    """Compute KMS state expectation values at inverse temperature beta.

    The KMS state is the equilibrium state of the Bost-Connes system:
        phi_beta(e_n) = n^{-beta} / Z(beta)

    These form a probability distribution (sum to 1.0) over n = 1, ..., n_max.

    Args:
        beta: Inverse temperature. Must be > 1.
        n_max: Number of terms (also determines partition function truncation).

    Returns:
        Array of shape (n_max,) with KMS probabilities summing to 1.0.

    Raises:
        ValueError: If beta <= 1.
    """
    _validate_beta(beta)
    ns = np.arange(1, n_max + 1, dtype=np.float64)
    weights = ns ** (-beta)
    # Normalize by the raw partial sum (not tail-corrected) so KMS values
    # sum to exactly 1.0 as a proper probability distribution.
    z_raw = float(np.sum(weights))
    kms = weights / z_raw
    return kms


def phase_transition_scan(
    beta_range: tuple[float, float] = (1.01, 5.0),
    n_points: int = 50,
    n_max: int = 500,
) -> dict:
    """Scan the Bost-Connes system across beta values to detect phase transition.

    At each beta, computes the partition function, KMS state, and Shannon
    entropy of the KMS distribution. The derivative d(entropy)/d(beta) shows
    a sharp change near beta=1 (the phase transition).

    Args:
        beta_range: (min_beta, max_beta) range to scan. min_beta must be > 1.
        n_points: Number of beta values to sample.
        n_max: Truncation for partition function at each point.

    Returns:
        Dict with keys:
            - "betas": ndarray of beta values
            - "partition_values": ndarray of Z(beta) values
            - "kms_entropy": ndarray of Shannon entropy values
            - "d_entropy_d_beta": ndarray of numerical derivative of entropy
    """
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    partition_values = np.zeros(n_points)
    kms_entropy = np.zeros(n_points)

    for i, beta in enumerate(betas):
        z = bost_connes_partition(float(beta), n_max)
        partition_values[i] = z

        kms = bost_connes_kms_values(float(beta), n_max)
        # Shannon entropy: -sum(p * log(p)), with convention 0*log(0) = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_kms = np.where(kms > 0, np.log(kms), 0.0)
        entropy = -np.sum(kms * log_kms)
        kms_entropy[i] = entropy

    # Numerical derivative of entropy w.r.t. beta
    d_entropy = np.gradient(kms_entropy, betas)

    return {
        "betas": betas,
        "partition_values": partition_values,
        "kms_entropy": kms_entropy,
        "d_entropy_d_beta": d_entropy,
    }


def compute_bost_connes(beta: float, n_max: int = 500) -> BostConnesResult:
    """Convenience function: compute full Bost-Connes analysis at given beta.

    Combines partition function, KMS values, and comparison to mpmath.zeta.

    Args:
        beta: Inverse temperature. Must be > 1.
        n_max: Truncation parameter.

    Returns:
        BostConnesResult with all computed fields.

    Raises:
        ValueError: If beta <= 1.
    """
    z = bost_connes_partition(beta, n_max)
    kms = bost_connes_kms_values(beta, n_max)
    zeta_exact = float(mpmath.zeta(beta))
    zeta_comparison = abs(z - zeta_exact)

    return BostConnesResult(
        beta=beta,
        partition_value=z,
        kms_values=kms,
        zeta_comparison=zeta_comparison,
        metadata={"n_max": n_max, "zeta_exact": zeta_exact},
    )
