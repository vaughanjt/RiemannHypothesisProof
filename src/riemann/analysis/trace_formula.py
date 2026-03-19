"""Trace formula workbench: Weil explicit formula and Chebyshev psi.

Implements the explicit formula connecting non-trivial zeros of the
Riemann zeta function to the distribution of primes:

    psi(x) = x - sum_{rho} x^rho / rho - log(2*pi) - 0.5*log(1 - x^{-2})

where the sum is over non-trivial zeros rho = 0.5 + i*t (assuming RH),
taken in conjugate pairs.

This module lets the user:
- Compute exact Chebyshev psi(x) via prime power enumeration
- Approximate psi(x) using the explicit formula with N zero terms
- Visualize convergence as more zeros are included

Function-based API. Returns floats and dataclasses, never plots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sympy import factorint, primerange


@dataclass
class TraceFormulaResult:
    """Result of a trace formula computation.

    Attributes:
        x: Point at which psi is evaluated.
        psi_exact: Exact Chebyshev psi(x) via prime power enumeration.
        psi_approx: Approximation via Weil explicit formula.
        n_terms: Number of zero pairs used in the approximation.
        relative_error: |psi_approx - psi_exact| / |psi_exact|.
        metadata: Additional computation metadata.
    """

    x: float
    psi_exact: float
    psi_approx: float
    n_terms: int
    relative_error: float
    metadata: dict = field(default_factory=dict)


def chebyshev_psi_exact(x: float) -> float:
    """Compute the exact Chebyshev psi function via prime power enumeration.

    psi(x) = sum_{p^k <= x} log(p)

    where p ranges over primes and k >= 1. Equivalently, psi(x) = sum_{n<=x} Lambda(n),
    where Lambda(n) = log(p) if n = p^k for some prime p and integer k >= 1, else 0.

    Uses sympy.factorint for prime power detection.

    Args:
        x: Upper bound (inclusive). Must be >= 1.

    Returns:
        Float value of psi(x).
    """
    if x < 2.0:
        return 0.0

    total = 0.0
    n_max = int(x)

    # For each prime p <= x, add log(p) for each power p^k <= x
    for p in primerange(2, n_max + 1):
        pk = p
        while pk <= x:
            total += math.log(p)
            # Check overflow before multiplying
            if pk > x / p:
                break
            pk *= p

    return total


def weil_explicit_psi(
    x: float,
    zeros: list[float],
    n_terms: int | None = None,
) -> float:
    """Approximate Chebyshev psi(x) using the Weil explicit formula.

    Implements:
        psi(x) ~ x - sum_{k=1}^{N} [x^{rho_k}/rho_k + x^{conj(rho_k)}/conj(rho_k)] - log(2*pi) - 0.5*log(1 - x^{-2})

    where rho_k = 0.5 + i*t_k are the non-trivial zeros (assuming RH),
    and t_k are the positive imaginary parts provided in the zeros list.

    Args:
        x: Point at which to evaluate (must be > 1).
        zeros: Positive imaginary parts of non-trivial zeros (e.g., [14.134..., 21.022..., ...]).
        n_terms: Number of zero pairs to include. Default: all available zeros.

    Returns:
        Float approximation of psi(x).
    """
    if x <= 1.0:
        return 0.0

    if n_terms is None:
        n_terms = len(zeros)
    n_terms = min(n_terms, len(zeros))

    # Main term
    result = x

    # Sum over conjugate zero pairs
    log_x = math.log(x)
    for k in range(n_terms):
        t = zeros[k]
        # rho = 0.5 + i*t, conj(rho) = 0.5 - i*t
        # x^rho = x^(0.5 + i*t) = x^0.5 * x^(i*t) = sqrt(x) * exp(i*t*log(x))
        # x^rho / rho + x^{conj(rho)} / conj(rho) = 2 * Re(x^rho / rho)
        sqrt_x = math.sqrt(x)
        theta = t * log_x
        # x^rho = sqrt(x) * (cos(theta) + i*sin(theta))
        xrho_real = sqrt_x * math.cos(theta)
        xrho_imag = sqrt_x * math.sin(theta)

        # rho = 0.5 + i*t
        rho_real = 0.5
        rho_imag = t

        # x^rho / rho = (xrho_real + i*xrho_imag) / (rho_real + i*rho_imag)
        # = [(xrho_real*rho_real + xrho_imag*rho_imag) + i*(xrho_imag*rho_real - xrho_real*rho_imag)]
        #   / (rho_real^2 + rho_imag^2)
        denom = rho_real**2 + rho_imag**2
        quotient_real = (xrho_real * rho_real + xrho_imag * rho_imag) / denom

        # The conjugate pair contributes 2 * Re(x^rho / rho)
        result -= 2.0 * quotient_real

    # Constant term: -log(2*pi)
    result -= math.log(2.0 * math.pi)

    # Trivial zero contribution: -0.5 * log(1 - x^{-2})
    # For x > 1: 1 - x^{-2} > 0
    if x > 1.0:
        result -= 0.5 * math.log(1.0 - x ** (-2))

    return result


def explicit_formula_terms(
    x: float,
    zeros: list[float],
    max_terms: int = 50,
) -> list[tuple[int, float]]:
    """Compute explicit formula approximations for increasing numbers of zero terms.

    Returns approximations at powers of 2 (1, 2, 4, 8, ...) up to max_terms,
    plus max_terms itself if not already a power of 2. This shows how the
    approximation converges as more zeros are included.

    Args:
        x: Point at which to evaluate psi.
        zeros: Positive imaginary parts of non-trivial zeros.
        max_terms: Maximum number of zero pairs to use.

    Returns:
        List of (n_terms, psi_approx) tuples showing convergence.
    """
    max_terms = min(max_terms, len(zeros))

    # Build list of evaluation points: powers of 2 up to max_terms
    eval_points = []
    k = 1
    while k <= max_terms:
        eval_points.append(k)
        k *= 2

    # Ensure max_terms itself is included
    if max_terms not in eval_points:
        eval_points.append(max_terms)

    eval_points.sort()

    results = []
    for n in eval_points:
        approx = weil_explicit_psi(x, zeros, n_terms=n)
        results.append((n, approx))

    return results


def compute_trace_formula(
    x: float,
    zeros: list[float],
    n_terms: int | None = None,
) -> TraceFormulaResult:
    """Compute both exact and approximate Chebyshev psi and return structured result.

    Convenience function combining chebyshev_psi_exact and weil_explicit_psi.

    Args:
        x: Point at which to evaluate.
        zeros: Positive imaginary parts of non-trivial zeros.
        n_terms: Number of zero pairs to use. Default: all available.

    Returns:
        TraceFormulaResult with exact value, approximation, and relative error.
    """
    psi_exact = chebyshev_psi_exact(x)
    psi_approx = weil_explicit_psi(x, zeros, n_terms=n_terms)

    if n_terms is None:
        n_terms_used = len(zeros)
    else:
        n_terms_used = min(n_terms, len(zeros))

    if abs(psi_exact) > 0:
        relative_error = abs(psi_approx - psi_exact) / abs(psi_exact)
    else:
        relative_error = abs(psi_approx)

    return TraceFormulaResult(
        x=x,
        psi_exact=psi_exact,
        psi_approx=psi_approx,
        n_terms=n_terms_used,
        relative_error=relative_error,
        metadata={
            "n_zeros_available": len(zeros),
        },
    )
