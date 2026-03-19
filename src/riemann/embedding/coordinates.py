"""Feature extraction functions for zero embeddings.

Implements 9 feature extractors that turn ZetaZero lists into N-dimensional
float64 arrays. All extractors share the signature:

    (zeros: list[ZetaZero], *, dps: int = 50) -> np.ndarray

The compute_embedding() function stacks selected features into a matrix
and optionally scales columns.

On import, this module replaces stubs in FEATURE_EXTRACTORS with real
implementations (registration pattern -- avoids circular imports).
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import mpmath
import numpy as np

from riemann.config import CACHE_DIR, DEFAULT_DPS
from riemann.engine.precision import precision_scope

if TYPE_CHECKING:
    from riemann.embedding.registry import EmbeddingConfig
    from riemann.types import ZetaZero

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: extract sorted imaginary parts as float64
# ---------------------------------------------------------------------------

def _imag_parts(zeros: list[ZetaZero]) -> np.ndarray:
    """Extract imaginary parts as sorted float64 array."""
    return np.array(
        sorted(float(z.value.imag) for z in zeros), dtype=np.float64
    )


def _local_mean_spacing(t: np.ndarray) -> np.ndarray:
    """Local mean spacing at height t: 2*pi / log(t/(2*pi)).

    Clamps argument to avoid log domain issues.
    """
    two_pi = 2.0 * np.pi
    log_arg = np.maximum(t / two_pi, np.e)
    return two_pi / np.log(log_arg)


# ---------------------------------------------------------------------------
# Feature extractors (cheap / fast)
# ---------------------------------------------------------------------------

def extract_imaginary_part(
    zeros: list[ZetaZero], *, dps: int = DEFAULT_DPS
) -> np.ndarray:
    """Extract the imaginary part of each zero (height on critical line)."""
    return np.array([float(z.value.imag) for z in zeros], dtype=np.float64)


def extract_left_spacing(
    zeros: list[ZetaZero], *, dps: int = DEFAULT_DPS
) -> np.ndarray:
    """Normalized gap to the previous zero.

    First element is 0.0 (no left neighbor). Spacings normalized by
    local mean spacing: 2*pi / log(t/(2*pi)).
    """
    t_sorted = _imag_parts(zeros)
    raw = np.diff(t_sorted)
    midpoints = (t_sorted[:-1] + t_sorted[1:]) / 2.0
    mean_sp = _local_mean_spacing(midpoints)
    normalized = raw / mean_sp
    return np.concatenate([[0.0], normalized])


def extract_right_spacing(
    zeros: list[ZetaZero], *, dps: int = DEFAULT_DPS
) -> np.ndarray:
    """Normalized gap to the next zero.

    Last element is 0.0 (no right neighbor). Spacings normalized by
    local mean spacing.
    """
    t_sorted = _imag_parts(zeros)
    raw = np.diff(t_sorted)
    midpoints = (t_sorted[:-1] + t_sorted[1:]) / 2.0
    mean_sp = _local_mean_spacing(midpoints)
    normalized = raw / mean_sp
    return np.concatenate([normalized, [0.0]])


def extract_local_density_deviation(
    zeros: list[ZetaZero], *, dps: int = DEFAULT_DPS
) -> np.ndarray:
    """Deviation of local zero density from Riemann-von Mangoldt prediction.

    Expected density at height t: log(t/(2*pi)) / (2*pi).
    Actual density: count of zeros in window of size w around each zero / w.
    Window size w = 10 * mean_spacing(t).
    Returns (actual - expected) / expected.
    """
    t_sorted = _imag_parts(zeros)
    n = len(t_sorted)
    result = np.zeros(n, dtype=np.float64)

    two_pi = 2.0 * np.pi

    for i in range(n):
        t_i = t_sorted[i]
        mean_sp = _local_mean_spacing(np.array([t_i]))[0]
        w = 10.0 * mean_sp

        # Count zeros in [t_i - w/2, t_i + w/2]
        lo = t_i - w / 2.0
        hi = t_i + w / 2.0
        count = np.searchsorted(t_sorted, hi, side="right") - np.searchsorted(
            t_sorted, lo, side="left"
        )

        actual_density = count / w

        # Expected density: log(t/(2*pi)) / (2*pi)
        log_arg = max(t_i / two_pi, np.e)
        expected_density = np.log(log_arg) / two_pi

        if expected_density > 0:
            result[i] = (actual_density - expected_density) / expected_density
        else:
            result[i] = 0.0

    return result


def extract_pair_correlation_local(
    zeros: list[ZetaZero], *, dps: int = DEFAULT_DPS
) -> np.ndarray:
    """Local pair correlation r_2(1.0) using a window of 50 surrounding zeros.

    For each zero, compute the pair correlation at distance x=1.0 (in mean-spacing
    units) using a local window.
    """
    t_sorted = _imag_parts(zeros)
    n = len(t_sorted)
    half_window = 25
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window)
        local_t = t_sorted[lo:hi]

        if len(local_t) < 3:
            result[i] = 0.0
            continue

        # Normalize spacings locally
        local_spacings = np.diff(local_t)
        midpoints = (local_t[:-1] + local_t[1:]) / 2.0
        mean_sp = _local_mean_spacing(midpoints)
        norm_spacings = local_spacings / mean_sp

        if len(norm_spacings) == 0:
            result[i] = 0.0
            continue

        # Cumulative normalized gaps
        cumsum = np.concatenate([[0.0], np.cumsum(norm_spacings)])

        # Count pairs at distance ~1.0 (within [0.8, 1.2])
        count = 0
        total_pairs = 0
        for j in range(len(cumsum)):
            for k in range(j + 1, len(cumsum)):
                gap = cumsum[k] - cumsum[j]
                if gap > 1.5:
                    break
                if 0.8 <= gap <= 1.2:
                    count += 1
                total_pairs += 1

        result[i] = count / max(total_pairs, 1)

    return result


def extract_hardy_z_sign_changes(
    zeros: list[ZetaZero], *, dps: int = DEFAULT_DPS
) -> np.ndarray:
    """Count Hardy Z-function sign changes in a local neighborhood.

    Uses mpmath.siegelz for evaluation. Returns count of sign changes
    in a window of 10 mean spacings around each zero.
    """
    t_sorted = _imag_parts(zeros)
    n = len(t_sorted)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        t_i = t_sorted[i]
        mean_sp = _local_mean_spacing(np.array([t_i]))[0]
        w = 10.0 * mean_sp

        lo = t_i - w / 2.0
        hi = t_i + w / 2.0

        # Sample Z(t) at 50 points in the window
        sample_t = np.linspace(max(lo, 1.0), hi, 50)
        with mpmath.workdps(15):  # Low precision is fine for sign detection
            z_vals = [float(mpmath.siegelz(mpmath.mpf(str(t)))) for t in sample_t]

        # Count sign changes
        signs = np.sign(z_vals)
        changes = np.sum(np.abs(np.diff(signs)) > 0)
        result[i] = float(changes)

    return result


def extract_local_entropy(
    zeros: list[ZetaZero], *, dps: int = DEFAULT_DPS
) -> np.ndarray:
    """Shannon entropy of spacings in a local window of 50 zeros.

    Uses binned method with 10 bins (fast approximation).
    """
    t_sorted = _imag_parts(zeros)
    n = len(t_sorted)
    half_window = 25
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window)
        local_t = t_sorted[lo:hi]

        if len(local_t) < 3:
            result[i] = 0.0
            continue

        # Normalized spacings in window
        local_spacings = np.diff(local_t)
        midpoints = (local_t[:-1] + local_t[1:]) / 2.0
        mean_sp = _local_mean_spacing(midpoints)
        norm_spacings = local_spacings / mean_sp

        if len(norm_spacings) < 2:
            result[i] = 0.0
            continue

        # Binned Shannon entropy
        counts, _ = np.histogram(norm_spacings, bins=10, range=(0.0, 4.0))
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # drop zeros for log
        entropy = -np.sum(probs * np.log2(probs))
        result[i] = entropy

    return result


def extract_compression_distance(
    zeros: list[ZetaZero], *, dps: int = DEFAULT_DPS
) -> np.ndarray:
    """Lempel-Ziv complexity of binarized spacings in local window.

    Binarizes normalized spacings (>1 = 1, <=1 = 0) and computes
    LZ complexity, normalized by theoretical maximum for the window size.
    """
    t_sorted = _imag_parts(zeros)
    n = len(t_sorted)
    half_window = 25
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window)
        local_t = t_sorted[lo:hi]

        if len(local_t) < 3:
            result[i] = 0.0
            continue

        # Normalized spacings
        local_spacings = np.diff(local_t)
        midpoints = (local_t[:-1] + local_t[1:]) / 2.0
        mean_sp = _local_mean_spacing(midpoints)
        norm_spacings = local_spacings / mean_sp

        # Binarize: above mean = 1, below = 0
        binary_seq = (norm_spacings > 1.0).astype(int)

        # Lempel-Ziv complexity
        lz_c = _lempel_ziv_complexity(binary_seq)

        # Theoretical maximum for random binary sequence of length n:
        # ~ n / log2(n) for large n
        seq_len = len(binary_seq)
        if seq_len > 1:
            max_c = seq_len / np.log2(seq_len)
            result[i] = lz_c / max_c
        else:
            result[i] = 0.0

    return result


def _lempel_ziv_complexity(seq: np.ndarray) -> int:
    """Compute Lempel-Ziv complexity of a binary sequence.

    Counts the number of distinct subsequences encountered during
    a left-to-right parse of the sequence.
    """
    n = len(seq)
    if n == 0:
        return 0

    complexity = 1
    i = 0
    k = 1
    kmax = 1

    while i + k <= n:
        # Check if seq[i+1 : i+k+1] appears in seq[0 : i+k]
        substring = tuple(seq[i + 1 : i + k + 1]) if i + k + 1 <= n else tuple(seq[i + 1 :])
        prefix = tuple(seq[: i + k])

        found = False
        for j in range(len(prefix) - len(substring) + 1):
            if prefix[j : j + len(substring)] == substring:
                found = True
                break

        if found:
            k += 1
            if i + k > n:
                break
        else:
            complexity += 1
            i += kmax if kmax > k else k
            k = 1
            kmax = 1
            if i >= n:
                break
            continue

        kmax = max(kmax, k)

    return complexity


# ---------------------------------------------------------------------------
# Expensive feature extractor (with caching)
# ---------------------------------------------------------------------------

def extract_zeta_derivative_magnitude(
    zeros: list[ZetaZero],
    *,
    dps: int = DEFAULT_DPS,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Magnitude |zeta'(rho)| at each zero.

    Computes mpmath.zeta(zero.value, derivative=1) and takes magnitude.
    Results are cached to disk (expensive computation).

    Args:
        zeros: List of ZetaZero objects.
        dps: Precision digits for computation.
        cache_dir: Directory for cache files. Default: DATA_DIR/cache.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache key from zero indices + dps
    key_data = f"{[z.index for z in zeros]}_{dps}".encode()
    cache_hash = hashlib.sha256(key_data).hexdigest()[:16]
    cache_path = cache_dir / f"zeta_derivatives_{cache_hash}.npy"

    if cache_path.exists():
        logger.debug("Loading cached zeta derivative magnitudes from %s", cache_path)
        return np.load(str(cache_path))

    n = len(zeros)
    if n > 100:
        warnings.warn(
            f"Computing |zeta'(rho)| for {n} zeros -- this may take a while",
            stacklevel=2,
        )

    result = np.zeros(n, dtype=np.float64)
    with mpmath.workdps(dps + 5):
        for i, z in enumerate(zeros):
            deriv = mpmath.zeta(z.value, derivative=1)
            result[i] = float(abs(deriv))

    # Cache result
    np.save(str(cache_path), result)
    logger.debug("Cached zeta derivative magnitudes to %s", cache_path)

    return result


# ---------------------------------------------------------------------------
# compute_embedding: the main pipeline
# ---------------------------------------------------------------------------

def compute_embedding(
    config: EmbeddingConfig,
    zeros: list[ZetaZero],
) -> np.ndarray:
    """Compute N-dimensional embedding from zero list using config features.

    Args:
        config: EmbeddingConfig specifying features and scaling.
        zeros: List of ZetaZero objects.

    Returns:
        ndarray of shape (len(zeros), len(config.feature_names)).

    Raises:
        ValueError: If a feature name is not in FEATURE_EXTRACTORS.
    """
    from riemann.embedding.registry import FEATURE_EXTRACTORS

    columns = []
    for name in config.feature_names:
        if name not in FEATURE_EXTRACTORS:
            raise ValueError(
                f"Unknown feature '{name}'. "
                f"Available: {sorted(FEATURE_EXTRACTORS.keys())}"
            )
        extractor = FEATURE_EXTRACTORS[name]
        col = extractor(zeros, dps=config.dps)
        columns.append(col)

    embedding = np.column_stack(columns)

    # Apply scaling
    if config.scaling == "standard":
        from sklearn.preprocessing import StandardScaler
        embedding = StandardScaler().fit_transform(embedding)
    elif config.scaling == "robust":
        from sklearn.preprocessing import RobustScaler
        embedding = RobustScaler().fit_transform(embedding)
    elif config.scaling == "none":
        pass  # raw values
    else:
        raise ValueError(f"Unknown scaling method: {config.scaling}")

    return embedding


# ---------------------------------------------------------------------------
# Registration: replace stubs in FEATURE_EXTRACTORS
# ---------------------------------------------------------------------------
# This runs at import time. registry.py defines stubs; we replace them.
# The __init__.py imports registry first, then coordinates, ensuring
# FEATURE_EXTRACTORS dict exists before we update it.

def _register_extractors() -> None:
    """Replace stub extractors with real implementations."""
    from riemann.embedding.registry import FEATURE_EXTRACTORS

    FEATURE_EXTRACTORS.update({
        "imag_part": extract_imaginary_part,
        "spacing_left": extract_left_spacing,
        "spacing_right": extract_right_spacing,
        "zeta_derivative_magnitude": extract_zeta_derivative_magnitude,
        "local_density_deviation": extract_local_density_deviation,
        "pair_correlation_local": extract_pair_correlation_local,
        "hardy_z_sign_changes": extract_hardy_z_sign_changes,
        "local_entropy": extract_local_entropy,
        "compression_distance": extract_compression_distance,
    })


_register_extractors()
