"""Dynamical systems tools: maps, orbits, Lyapunov exponents, fixed points.

Provides the Gauss/zeta map, logistic map, orbit computation, Lyapunov
exponent calculation, and fixed point detection for studying the dynamical
properties of zeta-related systems.

Function-based API with DynamicsResult dataclass for structured output.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq


@dataclass
class DynamicsResult:
    """Result from dynamical systems analysis.

    Attributes:
        orbit: Array of orbit points after transient.
        lyapunov: Lyapunov exponent of the orbit.
        fixed_points: Detected fixed points of the map.
        map_name: Name of the map analyzed.
        metadata: Additional info (n_steps, transient, search_range, etc.).
    """

    orbit: np.ndarray
    lyapunov: float
    fixed_points: list[float]
    map_name: str
    metadata: dict


def zeta_map(x: float, modulus: float = 1.0) -> float:
    """Gauss map / continued fraction map, related to Riemann zeta function.

    f(x) = frac(1/x) = (1/x) mod 1

    This is the shift map on the continued fraction expansion, connected
    to the zeta function via the Gauss-Kuzmin-Wirsing operator.

    Args:
        x: Input value.
        modulus: Period for the fractional part (default 1.0).

    Returns:
        The fractional part of 1/x, in [0, modulus).
    """
    if abs(x) < 1e-15:
        return 0.0
    inv = 1.0 / x
    result = inv % modulus
    return result


def logistic_map(x: float, r: float = 3.9) -> float:
    """Classic logistic map: f(x) = r * x * (1 - x).

    The logistic map exhibits a full range of dynamical behaviors:
    - r < 3: stable fixed point at x* = 1 - 1/r
    - 3 < r < 3.57: period-doubling cascade
    - r = 4: fully chaotic on [0, 1]

    Args:
        x: Input value in [0, 1].
        r: Growth parameter.

    Returns:
        r * x * (1 - x).
    """
    return r * x * (1.0 - x)


def compute_orbit(
    map_func: Callable[[float], float],
    x0: float,
    n_steps: int = 1000,
    transient: int = 100,
) -> np.ndarray:
    """Compute orbit of a 1D iterated map.

    Iterates x_{n+1} = map_func(x_n) starting from x0, discarding the
    first `transient` steps to allow convergence to the attractor.

    Args:
        map_func: The iterated map f: R -> R.
        x0: Initial condition.
        n_steps: Number of orbit points to return (after transient).
        transient: Number of initial iterations to discard.

    Returns:
        1D array of shape (n_steps,) containing the orbit after transient.
    """
    total = n_steps + transient
    orbit = np.empty(total, dtype=np.float64)
    x = x0
    for i in range(total):
        orbit[i] = x
        x = map_func(x)

    return orbit[transient:]


def lyapunov_exponent(
    orbit: np.ndarray,
    map_func: Callable[[float], float] | None = None,
    dt: float = 1e-8,
) -> float:
    """Compute the maximal Lyapunov exponent of an orbit.

    If map_func is provided, uses the numerical derivative method:
        lambda = (1/N) * sum(log|f'(x_i)|)
    where f'(x) is approximated by finite difference (f(x+dt) - f(x))/dt.

    If map_func is not provided, falls back to nolds.lyap_r as an estimator.

    Args:
        orbit: 1D array of orbit points.
        map_func: The iterated map (optional but recommended for accuracy).
        dt: Step size for finite difference derivative approximation.

    Returns:
        Float Lyapunov exponent. Positive = chaos, negative = stable,
        zero = marginally stable / bifurcation point.
    """
    if map_func is not None:
        # Numerical derivative method: lambda = mean(log|f'(x_i)|)
        log_derivs = []
        for x in orbit:
            # Finite difference derivative
            fx_plus = map_func(x + dt)
            fx = map_func(x)
            deriv = (fx_plus - fx) / dt
            abs_deriv = abs(deriv)
            if abs_deriv > 1e-30:
                log_derivs.append(math.log(abs_deriv))
            # Skip points where derivative is effectively zero

        if len(log_derivs) == 0:
            return 0.0
        return float(np.mean(log_derivs))
    else:
        # Fallback: use nolds
        import nolds

        return float(nolds.lyap_r(orbit))


def find_fixed_points(
    map_func: Callable[[float], float],
    search_range: tuple[float, float] = (0.01, 0.99),
    n_samples: int = 200,
    tolerance: float = 1e-8,
) -> list[float]:
    """Find fixed points of a 1D map via root-finding on g(x) = f(x) - x.

    Samples g(x) at n_samples points, identifies sign-change intervals,
    and refines roots using Brent's method.

    Args:
        map_func: The iterated map f: R -> R.
        search_range: (min, max) interval to search.
        n_samples: Number of sample points for sign change detection.
        tolerance: Convergence tolerance for root-finding.

    Returns:
        Sorted list of fixed points, deduplicated within tolerance.
    """

    def g(x: float) -> float:
        return map_func(x) - x

    lo, hi = search_range
    x_samples = np.linspace(lo, hi, n_samples)
    g_values = np.array([g(x) for x in x_samples])

    fixed_points: list[float] = []

    # Find sign-change intervals
    for i in range(len(g_values) - 1):
        if g_values[i] * g_values[i + 1] < 0:
            try:
                root = brentq(g, x_samples[i], x_samples[i + 1], xtol=tolerance)
                fixed_points.append(root)
            except ValueError:
                pass  # brentq failed, skip this interval

    # Also check for exact zeros at sample points
    for i, gv in enumerate(g_values):
        if abs(gv) < tolerance:
            fixed_points.append(float(x_samples[i]))

    # Deduplicate within tolerance
    if not fixed_points:
        return []

    fixed_points.sort()
    deduped = [fixed_points[0]]
    for fp in fixed_points[1:]:
        if abs(fp - deduped[-1]) > tolerance * 10:
            deduped.append(fp)

    return deduped


def analyze_dynamics(
    map_func: Callable[[float], float],
    x0: float = 0.1,
    n_steps: int = 1000,
    map_name: str = "unknown",
    transient: int = 100,
    search_range: tuple[float, float] = (0.01, 0.99),
) -> DynamicsResult:
    """Convenience function: compute orbit, Lyapunov exponent, and fixed points.

    Args:
        map_func: The iterated map.
        x0: Initial condition.
        n_steps: Orbit length after transient.
        map_name: Name for the map (stored in result).
        transient: Transient iterations to discard.
        search_range: Range for fixed point search.

    Returns:
        DynamicsResult with orbit, Lyapunov exponent, fixed points, and metadata.
    """
    orbit = compute_orbit(map_func, x0=x0, n_steps=n_steps, transient=transient)
    lyap = lyapunov_exponent(orbit, map_func=map_func)
    fps = find_fixed_points(map_func, search_range=search_range)

    metadata = {
        "x0": x0,
        "n_steps": n_steps,
        "transient": transient,
        "search_range": search_range,
    }

    return DynamicsResult(
        orbit=orbit,
        lyapunov=lyap,
        fixed_points=fps,
        map_name=map_name,
        metadata=metadata,
    )
