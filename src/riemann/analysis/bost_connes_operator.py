"""Bost-Connes-derived spectral operators.

Constructs operators from arithmetic/prime structure rather than
phase-space quantization (Berry-Keating). Tests whether primes
encoded ab initio produce eigenvalue statistics matching zeta zeros
in both distribution AND sequential correlations.
"""

import numpy as np
from scipy import linalg
from sympy import primerange


def construct_hecke_prime_adjacency(n: int) -> np.ndarray:
    """Adjacency matrix on {1,...,n} with prime multiplication edges.

    A[i,j] = log(p) if j = i*p or i = j*p for some prime p <= n.
    Encodes the multiplicative structure of integers through their
    prime factorization graph. Real symmetric.
    """
    A = np.zeros((n, n), dtype=np.float64)
    for p in primerange(2, n + 1):
        log_p = np.log(p)
        for i in range(1, n + 1):
            j = i * p
            if j <= n:
                A[i - 1, j - 1] = log_p
                A[j - 1, i - 1] = log_p
    return A


def construct_bc_hamiltonian(n: int, alpha: float = 1.0) -> np.ndarray:
    """Bost-Connes Hamiltonian with Hecke mixing.

    Diagonal: H[i,i] = log(i) (the BC number operator).
    Off-diagonal: H[i, ip] = alpha * log(p) / sqrt(i * ip) for prime p.
    The sqrt normalization keeps matrix elements bounded.
    """
    H = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n + 1):
        H[i - 1, i - 1] = np.log(i)
    for p in primerange(2, n + 1):
        log_p = np.log(p)
        for i in range(1, n + 1):
            j = i * p
            if j <= n:
                w = alpha * log_p / np.sqrt(i * j)
                H[i - 1, j - 1] = w
                H[j - 1, i - 1] = w
    return H


def construct_divisor_operator(n: int) -> np.ndarray:
    """Operator weighted by full divisor structure.

    H[i,j] = d(gcd(i,j)) / sqrt(i*j) for i != j, where d(k) = number
    of divisors of k. Diagonal: log(i). Captures arithmetic relationships
    beyond just prime multiplication.
    """
    from math import gcd

    d = np.zeros(n + 1, dtype=np.float64)
    for k in range(1, n + 1):
        for j in range(k, n + 1, k):
            d[j] += 1
    H = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n + 1):
        H[i - 1, i - 1] = np.log(i)
        for j in range(i + 1, n + 1):
            w = d[gcd(i, j)] / np.sqrt(i * j)
            H[i - 1, j - 1] = w
            H[j - 1, i - 1] = w
    return H


def polynomial_unfold(eigenvalues: np.ndarray, degree: int = 5,
                      trim_fraction: float = 0.1) -> np.ndarray:
    """Unfold eigenvalue spectrum using polynomial fit to staircase.

    Fits a degree-d polynomial to the integrated density of states,
    then maps eigenvalues through it. Trims edge fractions where
    the fit is unreliable. Resulting spacings have mean ~1.0.
    """
    eigs = np.sort(eigenvalues)
    n = len(eigs)
    staircase = np.arange(1, n + 1, dtype=np.float64)
    coeffs = np.polyfit(eigs, staircase, degree)
    unfolded = np.polyval(coeffs, eigs)
    spacings = np.diff(unfolded)
    # Trim edges where polynomial fit is unreliable
    trim = int(n * trim_fraction)
    if trim > 0 and len(spacings) > 2 * trim:
        spacings = spacings[trim:-trim]
    # Normalize to mean 1
    mean_s = np.mean(spacings)
    if mean_s > 1e-15:
        spacings = spacings / mean_s
    return spacings


def spacing_autocorrelation(spacings: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Normalized autocorrelation of spacing sequence.

    C(k) = E[(s_i - mu)(s_{i+k} - mu)] / var(s).
    C[0] = 1.0 by definition.
    """
    s = spacings - np.mean(spacings)
    var = np.var(spacings)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    n = len(s)
    acf = np.zeros(max_lag + 1)
    for k in range(min(max_lag + 1, n)):
        acf[k] = np.sum(s[:n - k] * s[k:]) / ((n - k) * var)
    return acf


def gue_reference_autocorrelation(n_matrix: int = 200, n_matrices: int = 500,
                                  max_lag: int = 20, seed: int = 42) -> np.ndarray:
    """Compute mean autocorrelation from GUE ensemble for reference.

    Generates many GUE matrices, unfolds each, computes autocorrelation,
    and averages. This gives the expected autocorrelation under pure
    random matrix universality (no arithmetic modulation).
    """
    rng = np.random.default_rng(seed)
    acfs = []
    for _ in range(n_matrices):
        # Generate GUE matrix
        A = rng.standard_normal((n_matrix, n_matrix)) + 1j * rng.standard_normal((n_matrix, n_matrix))
        H = (A + A.conj().T) / (2 * np.sqrt(2 * n_matrix))
        eigs = np.linalg.eigvalsh(H)
        spacings = polynomial_unfold(eigs, degree=5, trim_fraction=0.1)
        if len(spacings) > max_lag:
            acf = spacing_autocorrelation(spacings, max_lag)
            acfs.append(acf)
    return np.mean(acfs, axis=0)
