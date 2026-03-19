"""Modular forms computation module.

Computes Fourier coefficients of modular forms via q-series expansion,
including Eisenstein series E_k, the Ramanujan Delta function (weight 12
cusp form), and Hecke eigenvalue extraction from eigenforms.

Central to modern approaches to the Riemann Hypothesis via the Langlands
program: modular forms connect L-functions to automorphic representations.

Function-based API with ModularFormResult dataclass for structured output.
Uses mpmath for arbitrary-precision Bernoulli numbers and arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mpmath


@dataclass
class ModularFormResult:
    """Result of a modular form q-expansion computation.

    Attributes:
        weight: Weight of the modular form.
        level: Level (conductor) of the modular form.
        coefficients: List of Fourier coefficients [a_0, a_1, ..., a_{n_terms-1}].
        n_terms: Number of terms computed.
        metadata: Additional info (source, computation parameters, etc.).
    """

    weight: int
    level: int
    coefficients: list[complex]
    n_terms: int
    metadata: dict = field(default_factory=dict)


def _divisor_sigma(n: int, k: int) -> int:
    """Sum of k-th powers of divisors of n.

    sigma_k(n) = sum(d^k for d in divisors(n))

    Args:
        n: Positive integer.
        k: Non-negative integer exponent.

    Returns:
        Integer sum of k-th powers of all positive divisors of n.
    """
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d ** k
    return total


def eisenstein_series(k: int, n_terms: int = 50, dps: int = 15) -> list[float]:
    """Compute Fourier coefficients of the normalized Eisenstein series E_k.

    E_k(q) = 1 - (2k / B_k) * sum_{n=1}^{N} sigma_{k-1}(n) * q^n

    where B_k is the k-th Bernoulli number and sigma_{k-1}(n) is the sum
    of (k-1)-th powers of divisors of n.

    Known values:
        E_4 = 1 + 240*q + 2160*q^2 + 6720*q^3 + ...
        E_6 = 1 - 504*q - 16632*q^2 - 122976*q^3 + ...

    Args:
        k: Weight of the Eisenstein series. Must be even and >= 4.
        n_terms: Number of Fourier coefficients to compute (including a_0).
        dps: Decimal precision for mpmath computation.

    Returns:
        List of float Fourier coefficients [a_0, a_1, ..., a_{n_terms-1}].

    Raises:
        ValueError: If k is odd or less than 4.
    """
    if k < 4 or k % 2 != 0:
        raise ValueError(
            f"Eisenstein series weight must be even and >= 4, got k={k}"
        )

    with mpmath.workdps(dps):
        # B_k = k-th Bernoulli number
        bk = mpmath.bernoulli(k)

        # Normalization constant: -2k / B_k
        # The standard convention is E_k = 1 - (2k/B_k) * sum(sigma_{k-1}(n) q^n)
        # which equals E_k = 1 + (-2k/B_k) * sum(...)
        norm = mpmath.mpf(-2 * k) / bk

        coefficients: list[float] = []

        # a_0 = 1 (constant term of normalized Eisenstein series)
        coefficients.append(1.0)

        # a_n = norm * sigma_{k-1}(n) for n >= 1
        for n in range(1, n_terms):
            sigma = _divisor_sigma(n, k - 1)
            an = float(norm * sigma)
            coefficients.append(an)

    return coefficients


def compute_q_expansion(
    weight: int,
    level: int = 1,
    n_terms: int = 50,
    dps: int = 15,
) -> ModularFormResult:
    """Compute the q-expansion of a modular form.

    For level 1:
    - Weight 12: The unique normalized cusp form is the Ramanujan Delta
      function, computed as Delta = (E_4^3 - E_6^2) / 1728.
    - Weight < 12: No cusp forms exist; returns the Eisenstein series E_k.
    - Weight > 12: Returns the Eisenstein series (cusp form dimension grows,
      full decomposition is future work).

    The Ramanujan Delta function has q-expansion:
        Delta(q) = q - 24*q^2 + 252*q^3 - 1472*q^4 + 4830*q^5 - ...
    so the Fourier coefficients are the Ramanujan tau function values.

    Args:
        weight: Weight of the modular form.
        level: Level (conductor). Currently only level 1 supported.
        n_terms: Number of Fourier coefficients to return.
        dps: Decimal precision for computation.

    Returns:
        ModularFormResult with computed coefficients.
    """
    if level == 1 and weight == 12:
        # Delta = (E_4^3 - E_6^2) / 1728
        # We need enough terms from E_4 and E_6 to compute the product
        # E_4^3 needs convolution of 3 series, so we need n_terms in each
        e4 = eisenstein_series(4, n_terms=n_terms, dps=dps)
        e6 = eisenstein_series(6, n_terms=n_terms, dps=dps)

        # Compute E_4^3 via successive convolution
        e4_cubed = _multiply_q_series(
            _multiply_q_series(e4, e4, n_terms), e4, n_terms
        )

        # Compute E_6^2
        e6_squared = _multiply_q_series(e6, e6, n_terms)

        # Delta = (E_4^3 - E_6^2) / 1728
        coefficients = []
        for i in range(n_terms):
            delta_i = (e4_cubed[i] - e6_squared[i]) / 1728.0
            coefficients.append(complex(delta_i))

        return ModularFormResult(
            weight=weight,
            level=level,
            coefficients=coefficients,
            n_terms=n_terms,
            metadata={"form": "delta", "method": "eisenstein_product"},
        )

    elif level == 1 and weight < 12 and weight >= 4 and weight % 2 == 0:
        # No cusp forms at level 1 for weight < 12; return Eisenstein series
        e_coeffs = eisenstein_series(weight, n_terms=n_terms, dps=dps)
        return ModularFormResult(
            weight=weight,
            level=level,
            coefficients=[complex(c) for c in e_coeffs],
            n_terms=n_terms,
            metadata={"form": "eisenstein", "series": f"E_{weight}"},
        )

    else:
        # General case: return Eisenstein series if weight is valid
        if weight >= 4 and weight % 2 == 0:
            e_coeffs = eisenstein_series(weight, n_terms=n_terms, dps=dps)
            return ModularFormResult(
                weight=weight,
                level=level,
                coefficients=[complex(c) for c in e_coeffs],
                n_terms=n_terms,
                metadata={
                    "form": "eisenstein",
                    "series": f"E_{weight}",
                    "note": "full cusp form decomposition not yet implemented for this weight/level",
                },
            )
        else:
            raise ValueError(
                f"Cannot compute q-expansion for weight={weight}, level={level}"
            )


def _multiply_q_series(
    a: list[float], b: list[float], n_terms: int
) -> list[float]:
    """Multiply two q-series (Cauchy product / polynomial multiplication).

    Given a = [a_0, a_1, ...] and b = [b_0, b_1, ...], returns
    c where c_n = sum_{k=0}^{n} a_k * b_{n-k}.

    Args:
        a: First q-series coefficients.
        b: Second q-series coefficients.
        n_terms: Number of output terms.

    Returns:
        List of product coefficients [c_0, c_1, ..., c_{n_terms-1}].
    """
    result = [0.0] * n_terms
    len_a = min(len(a), n_terms)
    len_b = min(len(b), n_terms)

    for i in range(len_a):
        for j in range(len_b):
            if i + j < n_terms:
                result[i + j] += a[i] * b[j]

    return result


def hecke_eigenvalues(
    weight: int,
    level: int = 1,
    primes: list[int] | None = None,
    n_terms: int = 100,
) -> dict[int, complex]:
    """Extract Hecke eigenvalues from a cusp form eigenform.

    For eigenforms (like the Ramanujan Delta function), the Fourier
    coefficients a_p at primes p are exactly the Hecke eigenvalues:
    T_p(f) = a_p * f.

    Known eigenvalues for Delta (weight 12, level 1):
        tau(2) = -24, tau(3) = 252, tau(5) = -4830, tau(7) = 16744

    Args:
        weight: Weight of the modular form.
        level: Level (conductor).
        primes: List of primes at which to compute eigenvalues.
            Default: [2, 3, 5, 7, 11, 13].
        n_terms: Number of q-expansion terms to compute (must be > max prime).

    Returns:
        Dict mapping prime p -> Hecke eigenvalue a_p.
    """
    if primes is None:
        primes = [2, 3, 5, 7, 11, 13]

    # Ensure enough terms to cover all requested primes
    max_prime = max(primes)
    if n_terms <= max_prime:
        n_terms = max_prime + 1

    # Compute q-expansion
    result = compute_q_expansion(weight=weight, level=level, n_terms=n_terms)

    # Extract a_p for each prime
    eigenvalues: dict[int, complex] = {}
    for p in primes:
        if p < len(result.coefficients):
            eigenvalues[p] = result.coefficients[p]

    return eigenvalues
