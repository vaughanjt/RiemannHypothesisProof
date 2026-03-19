"""Information-theoretic analysis of zero spacing sequences.

Quantifies the information content of spacing patterns using entropy,
mutual information, and complexity measures. Enables cross-object comparison
(zeros vs GUE vs Poisson vs primes) to surface hidden structural similarities.

Function-based API. Returns floats and dicts, never plots.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression


def spacing_entropy(spacings: np.ndarray, method: str = "kde", bins: int = 50) -> float:
    """Compute Shannon entropy of a spacing sequence.

    Args:
        spacings: Array of (normalized) spacings.
        method: "binned" for histogram-based, "kde" for kernel density estimate.
        bins: Number of bins (binned) or evaluation points (kde).

    Returns:
        Shannon entropy as a float (nats for kde, bits-like for binned).
    """
    if len(spacings) < 2:
        raise ValueError("Need at least 2 spacings for entropy computation")
    if method not in ("binned", "kde"):
        raise ValueError(f"Unknown method '{method}': use 'binned' or 'kde'")

    if method == "binned":
        # Use fixed range [0, max(5, max_val)] so entropy reflects spread, not just uniformity
        bin_range = (0, max(5.0, float(spacings.max()) + 0.1))
        counts, _ = np.histogram(spacings, bins=bins, range=bin_range, density=False)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return float(scipy_entropy(probs))
    else:
        kde = gaussian_kde(spacings)
        x = np.linspace(spacings.min(), spacings.max(), bins)
        density = kde(x)
        density = density[density > 0]
        dx = (spacings.max() - spacings.min()) / bins
        return float(-np.sum(density * np.log(density) * dx))


def mutual_information_spacings(spacings: np.ndarray, lag: int = 1, k: int = 5) -> float:
    """Compute mutual information between spacings at a given lag.

    Uses k-nearest-neighbor estimator via sklearn.

    Args:
        spacings: Array of spacings.
        lag: Lag between pairs (default 1 = consecutive).
        k: Number of neighbors for MI estimation.

    Returns:
        Estimated mutual information as a float.
    """
    if len(spacings) <= lag + 1:
        raise ValueError(f"Need at least {lag + 2} spacings for lag={lag}")

    x = spacings[:-lag].reshape(-1, 1)
    y = spacings[lag:]
    mi = mutual_info_regression(x, y, n_neighbors=k, random_state=42)
    return float(mi[0])


def lempel_ziv_complexity(sequence: np.ndarray, threshold: float | None = None) -> int:
    """Compute Lempel-Ziv complexity of a sequence (LZ76 algorithm).

    Binarizes the sequence around a threshold, then counts distinct subsequences.

    Args:
        sequence: Numerical sequence to analyze.
        threshold: Binarization threshold (default: median).

    Returns:
        LZ76 complexity count (int).
    """
    if threshold is None:
        threshold = float(np.median(sequence))

    binary = "".join("1" if x > threshold else "0" for x in sequence)

    n = len(binary)
    if n == 0:
        return 0

    # LZ76: count distinct phrases in the parsing
    complexity = 1
    i = 0  # start of current new phrase
    k = 1  # length being tested
    while i + k <= n:
        # Check if binary[i:i+k] appeared as a substring in binary[0:i+k-1]
        current = binary[i : i + k]
        search_space = binary[: i + k - 1]
        if current in search_space:
            k += 1
        else:
            complexity += 1
            i = i + k
            k = 1

    return complexity


def cross_object_comparison(
    zero_spacings: np.ndarray,
    gue_spacings: np.ndarray,
    poisson_spacings: np.ndarray | None = None,
    prime_gaps: np.ndarray | None = None,
) -> dict:
    """Compare information-theoretic signatures across mathematical objects.

    Args:
        zero_spacings: Normalized spacings of zeta zeros.
        gue_spacings: Normalized spacings from GUE ensemble.
        poisson_spacings: Optional Poisson spacings (generated if None).
        prime_gaps: Optional array of prime gaps.

    Returns:
        Nested dict: {object_name: {metric_name: value}}.
    """
    rng = np.random.default_rng(42)
    if poisson_spacings is None:
        poisson_spacings = rng.exponential(1.0, size=len(zero_spacings))

    objects = {
        "zeta_zeros": zero_spacings,
        "gue_eigenvalues": gue_spacings,
        "poisson": poisson_spacings,
    }
    if prime_gaps is not None:
        objects["primes"] = prime_gaps

    result = {}
    for name, spacings in objects.items():
        s = np.asarray(spacings, dtype=float)
        if len(s) < 10:
            result[name] = {
                "entropy_binned": 0.0, "entropy_kde": 0.0,
                "mi_lag1": 0.0, "mi_lag2": 0.0, "lz_complexity": 0,
            }
            continue
        result[name] = {
            "entropy_binned": spacing_entropy(s, method="binned"),
            "entropy_kde": spacing_entropy(s, method="kde"),
            "mi_lag1": mutual_information_spacings(s, lag=1),
            "mi_lag2": mutual_information_spacings(s, lag=2) if len(s) > 3 else 0.0,
            "lz_complexity": lempel_ziv_complexity(s),
        }

    return result
