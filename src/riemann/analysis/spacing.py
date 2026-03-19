"""Zero distribution statistics engine.

Computes spacing statistics for non-trivial zeros of the Riemann zeta function.
All functions accept either ZetaZero objects or pre-computed numpy arrays.
Function-based API -- no classes. Returns numpy arrays, never plots.

Statistics provided:
- normalized_spacings: raw spacings divided by local mean spacing
- pair_correlation: two-point correlation function R_2(x)
- gue_pair_correlation: GUE sine kernel prediction 1 - (sin(pi*x)/(pi*x))^2
- n_level_density: density of n-tuple spacing sums
- number_variance: Sigma_2(L) = variance of zero count in intervals of length L
"""
from __future__ import annotations

import numpy as np

from riemann.types import ZetaZero


def normalized_spacings(zeros: list[ZetaZero]) -> np.ndarray:
    """Compute normalized spacings from a list of ZetaZero objects.

    Extracts imaginary parts, sorts them, computes raw spacings via np.diff,
    then divides by the local mean spacing at each midpoint using the
    asymptotic formula: mean_spacing(t) = 2*pi / log(t / (2*pi)).

    Args:
        zeros: List of ZetaZero objects (must have len >= 2).

    Returns:
        1D array of normalized spacings (length = len(zeros) - 1).
        Mean should be approximately 1.0 for zeros in asymptotic regime.

    Raises:
        ValueError: If fewer than 2 zeros provided.
    """
    if len(zeros) < 2:
        raise ValueError(
            f"Need at least 2 zeros to compute spacings, got {len(zeros)}"
        )

    # Extract imaginary parts as float64 and sort
    imag_parts = np.array(
        [float(z.value.imag) for z in zeros], dtype=np.float64
    )
    imag_parts.sort()

    # Raw spacings
    raw_spacings = np.diff(imag_parts)

    # Midpoints for local mean spacing computation
    midpoints = (imag_parts[:-1] + imag_parts[1:]) / 2.0

    # Local mean spacing: 2*pi / log(t / (2*pi))
    # Guard against log domain issues for very small t
    two_pi = 2.0 * np.pi
    log_arg = midpoints / two_pi
    # Clamp to avoid log(0) or log(negative)
    log_arg = np.maximum(log_arg, np.e)  # ensures log >= 1
    mean_spacings = two_pi / np.log(log_arg)

    return raw_spacings / mean_spacings


def pair_correlation(
    spacings: np.ndarray,
    bins: int = 200,
    x_range: tuple[float, float] = (0.0, 4.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the two-point correlation function R_2(x) from normalized spacings.

    For N normalized spacings, accumulates all pairwise normalized gaps:
    gap(i, j) = sum of spacings from i to j-1, for all i < j within range.
    Histograms these gaps and normalizes to get a density estimate.

    Args:
        spacings: 1D array of normalized spacings.
        bins: Number of histogram bins.
        x_range: (min, max) range for the correlation function.

    Returns:
        Tuple of (x_centers, r2_values), each 1D array of length `bins`.

    Raises:
        ValueError: If spacings is empty.
    """
    if len(spacings) == 0:
        raise ValueError("Cannot compute pair correlation from empty spacings")

    n = len(spacings)
    x_min, x_max = x_range

    # Compute cumulative sum for efficient gap computation
    cumsum = np.concatenate([[0.0], np.cumsum(spacings)])

    # Collect all pairwise gaps within x_range
    gaps = []
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            gap = cumsum[j] - cumsum[i]
            if gap > x_max:
                break  # cumsum is monotonic, no point continuing
            if gap >= x_min:
                gaps.append(gap)

    gaps = np.array(gaps) if gaps else np.array([])

    # Histogram and normalize
    counts, bin_edges = np.histogram(gaps, bins=bins, range=x_range)
    bin_width = (x_max - x_min) / bins
    x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Normalize: density = counts / (total_pairs * bin_width)
    # where total_pairs accounts for the expected uniform density
    # For normalized spacings with mean 1, expected density of gaps at
    # distance x is proportional to N (number of zero pairs at that scale)
    total = len(gaps) if len(gaps) > 0 else 1
    r2_values = counts.astype(np.float64) / (total * bin_width)

    # Scale so that far-field approaches 1.0 (the Poisson limit)
    # For a proper pair correlation, normalize by the mean density
    if np.max(r2_values) > 0:
        # Use the average of the upper quarter of bins as reference level
        upper_quarter = r2_values[3 * bins // 4:]
        if len(upper_quarter) > 0 and np.mean(upper_quarter) > 0:
            r2_values = r2_values / np.mean(upper_quarter)

    return x_centers, r2_values


def gue_pair_correlation(x: np.ndarray) -> np.ndarray:
    """GUE pair correlation function (sine kernel prediction).

    R_2(x) = 1 - (sin(pi*x) / (pi*x))^2

    This is the two-point correlation function for eigenvalues of random
    matrices from the Gaussian Unitary Ensemble in the bulk scaling limit.

    Args:
        x: 1D array of positive real values.

    Returns:
        1D array of R_2 values. At x=0 returns 0 (since sinc(0)=1).
        Approaches 1.0 as x -> infinity.
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.ones_like(x)

    # Handle x=0 separately (sinc(0) = 1, so R_2(0) = 1 - 1 = 0)
    zero_mask = (x == 0.0)
    nonzero_mask = ~zero_mask

    if np.any(nonzero_mask):
        pi_x = np.pi * x[nonzero_mask]
        sinc_val = np.sin(pi_x) / pi_x
        result[nonzero_mask] = 1.0 - sinc_val ** 2

    result[zero_mask] = 0.0

    return result


def n_level_density(
    spacings: np.ndarray,
    n: int = 2,
    bins: int = 100,
) -> np.ndarray:
    """Compute n-level density from normalized spacings.

    The n-level density captures correlations among n consecutive spacings.
    For n=2, this is the density of sums of 2 consecutive normalized spacings.
    For n=3, sums of 3 consecutive spacings, etc.

    Args:
        spacings: 1D array of normalized spacings.
        n: Number of consecutive spacings to sum (2, 3, or 4).
        bins: Number of histogram bins.

    Returns:
        1D array of density values (length = bins).

    Raises:
        ValueError: If spacings is empty or has fewer than n elements.
    """
    if len(spacings) == 0:
        raise ValueError("Cannot compute n-level density from empty spacings")
    if len(spacings) < n:
        raise ValueError(
            f"Need at least {n} spacings for {n}-level density, got {len(spacings)}"
        )

    # Sum of n consecutive spacings using a sliding window
    # For n=2: s[0]+s[1], s[1]+s[2], ...
    # For n=3: s[0]+s[1]+s[2], s[1]+s[2]+s[3], ...
    cumsum = np.concatenate([[0.0], np.cumsum(spacings)])
    n_sums = cumsum[n:] - cumsum[:-n]

    # Histogram and normalize to density
    counts, bin_edges = np.histogram(n_sums, bins=bins)
    bin_width = bin_edges[1] - bin_edges[0]
    total = len(n_sums)
    density = counts.astype(np.float64) / (total * bin_width) if total > 0 else counts.astype(np.float64)

    return density


def number_variance(
    spacings: np.ndarray,
    L_values: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the number variance Sigma_2(L) from normalized spacings.

    Sigma_2(L) is the variance of the number of zeros in an interval of
    length L (in units of mean spacing). Computed by sliding a window of
    length L over the cumulative spacing sequence and measuring the variance
    of the zero count within each window position.

    Args:
        spacings: 1D array of normalized spacings.
        L_values: 1D array of interval lengths to evaluate. Default: 50
            values linearly spaced from 0.1 to 5.0.

    Returns:
        1D array of variance values, same length as L_values.

    Raises:
        ValueError: If spacings is empty.
    """
    if len(spacings) == 0:
        raise ValueError("Cannot compute number variance from empty spacings")

    if L_values is None:
        L_values = np.linspace(0.1, 5.0, 50)

    L_values = np.asarray(L_values, dtype=np.float64)

    # Cumulative spacings give positions of zeros (in normalized units)
    positions = np.concatenate([[0.0], np.cumsum(spacings)])
    total_length = positions[-1]

    result = np.zeros(len(L_values), dtype=np.float64)

    for i, L in enumerate(L_values):
        if L <= 0 or L > total_length:
            result[i] = 0.0
            continue

        # Slide window of length L across the sequence
        # Count zeros in [start, start + L) for each window position
        counts = []
        # Use positions of zeros as window start points
        n_windows = 0
        for start_idx in range(len(positions)):
            start = positions[start_idx]
            end = start + L
            if end > total_length:
                break
            # Count zeros in [start, start + L)
            count = np.searchsorted(positions, end, side='left') - start_idx
            counts.append(count)
            n_windows += 1

        if n_windows > 1:
            counts_arr = np.array(counts, dtype=np.float64)
            result[i] = np.var(counts_arr)
        else:
            result[i] = 0.0

    return result
