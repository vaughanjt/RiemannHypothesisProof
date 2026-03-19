"""p-adic arithmetic, Kubota-Leopoldt zeta, and fractal tree data.

Provides p-adic number representation with arithmetic operations (add, mul,
neg, sub), conversion from rationals, the Kubota-Leopoldt p-adic zeta function,
and fractal tree data generation for visualization.

Function-based API with PadicNumber dataclass for structured output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import mpmath
import numpy as np


@dataclass
class PadicNumber:
    """A p-adic number with finite precision.

    Represents sum_{i=0}^{precision-1} digits[i] * p^{i + valuation}.
    digits are in [0, p-1], least significant first.
    """

    p: int
    digits: list[int]
    valuation: int
    precision: int

    def norm(self) -> float:
        """p-adic absolute value: |x|_p = p^{-v_p(x)} for nonzero x, 0 for zero."""
        if all(d == 0 for d in self.digits):
            return 0.0
        # Find actual valuation (first nonzero digit)
        for i, d in enumerate(self.digits):
            if d != 0:
                actual_val = self.valuation + i
                return float(self.p ** (-actual_val))
        return 0.0

    def __repr__(self) -> str:
        parts = []
        for i, d in enumerate(self.digits):
            if d != 0:
                exp = self.valuation + i
                if exp == 0:
                    parts.append(f"{d}")
                elif exp == 1:
                    parts.append(f"{d}*{self.p}")
                else:
                    parts.append(f"{d}*{self.p}^{exp}")
        if not parts:
            parts.append("0")
        total_order = self.valuation + self.precision
        return " + ".join(parts) + f" + O({self.p}^{total_order})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PadicNumber):
            return NotImplemented
        if self.p != other.p:
            return False
        # Compare digit-by-digit up to common precision
        min_prec = min(self.precision, other.precision)
        for i in range(min_prec):
            d1 = self._digit_at(self.valuation + i)
            d2 = other._digit_at(other.valuation + i)
            if d1 != d2:
                return False
        return True

    def _digit_at(self, position: int) -> int:
        """Get the digit coefficient of p^position."""
        idx = position - self.valuation
        if 0 <= idx < len(self.digits):
            return self.digits[idx]
        return 0

    def __add__(self, other: PadicNumber) -> PadicNumber:
        if self.p != other.p:
            raise ValueError(
                f"Cannot add p-adic numbers with different primes: {self.p} vs {other.p}"
            )
        p = self.p
        # Determine output range
        min_val = min(self.valuation, other.valuation)
        max_pos = max(
            self.valuation + self.precision,
            other.valuation + other.precision,
        )
        out_prec = max_pos - min_val

        result_digits = []
        carry = 0
        for i in range(out_prec):
            pos = min_val + i
            d1 = self._digit_at(pos)
            d2 = other._digit_at(pos)
            total = d1 + d2 + carry
            carry = total // p
            result_digits.append(total % p)

        return PadicNumber(p=p, digits=result_digits, valuation=min_val, precision=out_prec)

    def __neg__(self) -> PadicNumber:
        """Negate: compute (p^precision - self) in digit representation.

        In p-adic arithmetic, -x has digits that are the "p's complement".
        """
        p = self.p
        if all(d == 0 for d in self.digits):
            return PadicNumber(p=p, digits=list(self.digits), valuation=self.valuation,
                               precision=self.precision)

        result_digits = []
        borrow = 0
        for i in range(self.precision):
            d = self.digits[i] if i < len(self.digits) else 0
            val = -d - borrow
            if val < 0:
                val += p
                borrow = 1
            else:
                borrow = 0
            result_digits.append(val % p)

        return PadicNumber(p=p, digits=result_digits, valuation=self.valuation,
                           precision=self.precision)

    def __sub__(self, other: PadicNumber) -> PadicNumber:
        return self + (-other)

    def __mul__(self, other: PadicNumber) -> PadicNumber:
        if self.p != other.p:
            raise ValueError(
                f"Cannot multiply p-adic numbers with different primes: "
                f"{self.p} vs {other.p}"
            )
        p = self.p
        new_val = self.valuation + other.valuation
        # Output precision is min of the two input precisions
        out_prec = min(self.precision, other.precision)

        # Polynomial convolution of digit sequences
        n1 = len(self.digits)
        n2 = len(other.digits)
        # We only need out_prec digits of the product
        raw = [0] * out_prec
        for i in range(min(n1, out_prec)):
            for j in range(min(n2, out_prec - i)):
                raw[i + j] += self.digits[i] * other.digits[j]

        # Reduce with carry
        result_digits = []
        carry = 0
        for i in range(out_prec):
            total = raw[i] + carry
            carry = total // p
            result_digits.append(total % p)

        return PadicNumber(p=p, digits=result_digits, valuation=new_val, precision=out_prec)


def _p_valuation(n: int, p: int) -> int:
    """Compute the p-adic valuation of integer n (how many times p divides n)."""
    if n == 0:
        return float("inf")  # type: ignore[return-value]
    v = 0
    n = abs(n)
    while n % p == 0:
        n //= p
        v += 1
    return v


def _mod_inverse(a: int, m: int) -> int:
    """Compute modular inverse of a mod m using extended Euclidean algorithm."""
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse mod {m}")
    return x % m


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm: returns (gcd, x, y) such that a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def padic_from_rational(
    a: int, b: int, p: int, precision: int = 20
) -> PadicNumber:
    """Convert the rational number a/b to a p-adic number in Q_p.

    Args:
        a: Numerator.
        b: Denominator (must be nonzero).
        p: Prime for the p-adic field.
        precision: Number of p-adic digits to compute.

    Returns:
        PadicNumber representing a/b in Q_p.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Denominator cannot be zero")

    if a == 0:
        return PadicNumber(p=p, digits=[0] * precision, valuation=0, precision=precision)

    # Compute valuation: v_p(a/b) = v_p(a) - v_p(b)
    va = _p_valuation(a, p)
    vb = _p_valuation(b, p)
    valuation = va - vb

    # Remove p-factors
    a_unit = a // (p ** va) if va > 0 else a
    b_unit = b // (p ** vb) if vb > 0 else b

    # Handle sign
    if b_unit < 0:
        a_unit = -a_unit
        b_unit = -b_unit

    # Now compute digits of a_unit / b_unit mod p^precision
    # b_unit is coprime to p, so it has an inverse mod p^k
    modulus = p ** precision
    b_inv = _mod_inverse(b_unit % modulus, modulus)
    value = (a_unit * b_inv) % modulus

    # Handle negative values (p-adic representation of negative rationals)
    if value < 0:
        value = value % modulus

    # Extract digits
    digits = []
    for _ in range(precision):
        digits.append(value % p)
        value //= p

    return PadicNumber(p=p, digits=digits, valuation=valuation, precision=precision)


def kubota_leopoldt_zeta(
    s: int, p: int, precision: int = 20
) -> PadicNumber:
    """Compute the Kubota-Leopoldt p-adic zeta function at negative odd integer s.

    For s = 1-n where n >= 2 is even:
        zeta_p(1-n) = -(1 - p^{n-1}) * B_n / n

    where B_n is the n-th Bernoulli number.

    Args:
        s: A negative odd integer.
        p: Prime for the p-adic field.
        precision: Number of p-adic digits.

    Returns:
        PadicNumber representing zeta_p(s).

    Raises:
        ValueError: If s is not a negative odd integer.
    """
    if s >= 0:
        raise ValueError(f"s must be a negative odd integer, got s={s}")
    if s % 2 == 0:
        raise ValueError(f"s must be odd, got s={s}")

    # s = 1 - n => n = 1 - s
    n = 1 - s  # n >= 2 and even

    if n < 2 or n % 2 != 0:
        raise ValueError(
            f"s={s} gives n={n} which is not >= 2 and even"
        )

    # Compute -(1 - p^{n-1}) * B_n / n using exact rational arithmetic
    # Use mpmath rational Bernoulli number extraction: bernfrac gives exact (p, q)
    bp, bq = mpmath.bernfrac(n)
    bn_frac = Fraction(int(bp), int(bq))

    euler_factor = 1 - p ** (n - 1)
    result_frac = Fraction(-euler_factor, 1) * bn_frac / Fraction(n, 1)

    # Convert to PadicNumber
    num = int(result_frac.numerator)
    den = int(result_frac.denominator)

    return padic_from_rational(num, den, p=p, precision=precision)


def padic_fractal_tree_data(p: int, depth: int = 4) -> dict:
    """Generate p-adic fractal tree data for visualization.

    Creates a p-ary tree where each node at depth d has p children.
    Nodes are positioned using p-adic digit decomposition to create
    a fractal branching pattern.

    Args:
        p: Prime (branching factor).
        depth: Maximum depth of the tree (0-indexed).

    Returns:
        Dict with:
            - "nodes": list of dicts with "x", "y", "label", "depth"
            - "edges": list of dicts with "from_idx", "to_idx"
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    # BFS to build the tree
    # Each node identified by (depth, index_at_depth)
    # Total nodes: (p^{depth+1} - 1) / (p - 1)
    queue = [(0, 0, 0.0, 0.0)]  # (depth_level, node_global_idx, x, y)

    # Root node
    nodes.append({"x": 0.0, "y": 0.0, "label": "root", "depth": 0})
    node_idx = 1

    # Build level by level
    current_level_nodes = [(0, 0.0)]  # (global_idx, x_position)

    for d in range(1, depth + 1):
        next_level_nodes = []
        spread = 1.0 / (p ** (d - 1))  # Narrowing spread at each level

        for parent_idx, parent_x in current_level_nodes:
            for k in range(p):
                # Position children symmetrically around parent
                offset = (k - (p - 1) / 2.0) * spread
                child_x = parent_x + offset
                child_y = -float(d)  # Depth goes downward

                label = f"d{d}_k{k}"
                nodes.append({
                    "x": child_x,
                    "y": child_y,
                    "label": label,
                    "depth": d,
                })
                edges.append({
                    "from_idx": parent_idx,
                    "to_idx": node_idx,
                })
                next_level_nodes.append((node_idx, child_x))
                node_idx += 1

        current_level_nodes = next_level_nodes

    return {"nodes": nodes, "edges": edges}
