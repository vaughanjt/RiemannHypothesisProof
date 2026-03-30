"""
Session 28e: Arithmetic structure of G_{jk} for the novelty proof.

G_{jk} = integral_0^1 {1/(jx)} {1/(kx)} dx

By substitution u = 1/x: G_{jk} = integral_1^inf {ju}{ku} / u^2 du

The key: {ju}{ku} depends on gcd(j,k) through the Chinese Remainder Theorem
for fractional parts.

Step 1: Compute G_{jk} exactly (high precision) and find the formula
        in terms of gcd(j,k) and lcm(j,k).

Step 2: Use the formula to compute the Schur complement analytically.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, nstr, fabs, log, pi, zeta, euler
import sympy
from math import gcd

mp.dps = 50


def gram_entry_hp(j, k):
    """High-precision Gram matrix entry via mpmath quadrature."""
    j_mp = mpf(j); k_mp = mpf(k)
    def integrand(x):
        if x < mpf(10)**(-14): return mpf(0)
        fj = 1/(j_mp*x); fj = fj - mpmath.floor(fj)
        fk = 1/(k_mp*x); fk = fk - mpmath.floor(fk)
        return fj * fk
    return mpmath.quad(integrand, [mpf(10)**(-14), mpf(1)], method='tanh-sinh')


if __name__ == "__main__":
    print("ARITHMETIC STRUCTURE OF G_{jk}")
    print("=" * 70)

    # Compute G_{jk} for small j,k and look for patterns with gcd/lcm
    print("\nG_{jk} vs arithmetic functions of (j,k):")
    print(f"{'j':>3} {'k':>3} {'G_{jk}':>14} {'gcd':>4} {'lcm':>6} "
          f"{'G*jk':>10} {'G*lcm':>10} {'1/gcd':>8}")
    print("-" * 70)

    for j in range(1, 13):
        for k in range(j, 13):
            g = float(gram_entry_hp(j, k))
            d = gcd(j, k)
            l = j * k // d
            print(f"{j:>3} {k:>3} {g:>14.8f} {d:>4} {l:>6} "
                  f"{g*j*k:>10.4f} {g*l:>10.4f} {1/d:>8.4f}")

    # Key test: does G_{jk} * lcm(j,k) depend only on gcd(j,k)?
    print("\n\nG_{jk} * lcm(j,k) grouped by gcd:")
    from collections import defaultdict
    gcd_groups = defaultdict(list)
    for j in range(1, 20):
        for k in range(j, 20):
            g = float(gram_entry_hp(j, k))
            d = gcd(j, k)
            l = j * k // d
            gcd_groups[d].append((j, k, g * l))

    for d in sorted(gcd_groups.keys())[:6]:
        vals = [v[2] for v in gcd_groups[d]]
        pairs = [(v[0], v[1]) for v in gcd_groups[d]]
        print(f"  gcd={d}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
              f"n={len(vals)}")
        if len(vals) <= 5:
            for (j,k), v in zip(pairs, vals):
                print(f"    ({j},{k}): G*lcm = {v:.6f}")

    # Diagonal formula: G_{jj} = integral_0^1 {1/(jx)}^2 dx
    print("\n\nDiagonal: G_{jj} and its relationship to 1/j")
    print(f"{'j':>3} {'G_{jj}':>14} {'j*G_{jj}':>12} {'j*G - (ln j + gamma)/2':>22}")
    for j in range(1, 21):
        g = float(gram_entry_hp(j, j))
        expected = float(log(mpf(j)) + euler) / 2  # hypothesis
        print(f"{j:>3} {g:>14.8f} {j*g:>12.6f} {j*g - expected:>22.8f}")

    # Off-diagonal for coprime pairs
    print("\n\nCoprime pairs (gcd=1): G_{jk} * j * k")
    for j in range(2, 10):
        for k in range(j+1, 15):
            if gcd(j, k) == 1:
                g = float(gram_entry_hp(j, k))
                print(f"  ({j},{k}): G*jk = {g*j*k:.6f}")
