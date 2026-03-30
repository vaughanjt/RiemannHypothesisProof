"""
Session 29f: PROVE the Euler product novelty formula.

Strategy:
1. Derive exact formula for G_{jj} (diagonal)
2. Derive exact formula for G_{jk} (off-diagonal)
3. Compute C(j) = j^2 * dist_j^2 / EP(j) to 30+ digits
4. Identify C(j) as a known function
5. Prove the Euler product via Schur complement

Key derivation:
  G_{jk} = integral_0^1 {1/(jx)} {1/(kx)} dx

  Substituting y = 1/x:
  G_{jk} = integral_1^inf {jy}{ky}/y^2 dy = (1/j) integral_{1/j}^inf {t}{kt/j}/t^2 dt

  Actually better: use the EXACT piecewise formula.

  On each interval where both floor functions are constant, compute exactly.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, zeta, pi, log, euler, nstr, fabs,
                    floor as mpfloor, quad, psi, harmonic)
from math import gcd
from functools import reduce
import sympy
import time

mp.dps = 50


def gram_exact(j, k):
    """Compute G_{jk} = integral_0^1 {1/(jx)} {1/(kx)} dx using mpmath quadrature.

    High precision (50 digits) with tanh-sinh.
    """
    j_mp, k_mp = mpf(j), mpf(k)
    def integrand(x):
        if x < mpf(10)**(-20):
            return mpf(0)
        fj = 1/(j_mp * x); fj = fj - mpfloor(fj)
        fk = 1/(k_mp * x); fk = fk - mpfloor(fk)
        return fj * fk
    return quad(integrand, [mpf(10)**(-20), mpf(1)], method='tanh-sinh')


def gram_diagonal_exact(j):
    """Exact formula for G_{jj}.

    G_{jj} = (1/j) * [(1 - 1/j) + integral_1^inf {t}^2 / t^2 dt]

    Wait, that's not right for general j. Let me derive it properly.

    G_{jj} = (1/j) integral_{1/j}^inf {u}^2 / u^2 du

    For u in [1/j, 1): {u} = u (since 0 < u < 1)
      integral_{1/j}^1 u^2/u^2 du = 1 - 1/j

    For u in [n, n+1) with n >= 1: {u} = u - n
      integral_n^{n+1} (u-n)^2/u^2 du = I(n)

    So G_{jj} = (1/j)[(1 - 1/j) + sum_{n=1}^inf I(n)]
    """
    j_mp = mpf(j)
    # I(n) = integral_n^{n+1} (u-n)^2/u^2 du
    # = 1 - 2n*ln(1 + 1/n) + n/(n+1)
    # Sum this series to high precision

    S = mpf(0)
    for n in range(1, 10000):
        n_mp = mpf(n)
        I_n = 1 - 2*n_mp*log(1 + 1/n_mp) + n_mp/(n_mp + 1)
        S += I_n
        if n > 100 and abs(I_n) < mpf(10)**(-40):
            break

    # Tail correction: for large n, I(n) ~ 1/(3n^2) + 1/(4n^3) + ...
    # sum_{n=N+1}^inf 1/(3n^2) ~ 1/(3N) - 1/(6N^2) + ...
    N = n
    tail = 1/(3*mpf(N)) - 1/(6*mpf(N)**2) + 1/(6*mpf(N)**3)
    S += tail

    return (1 - 1/j_mp + S) / j_mp


def I_n_exact(n):
    """I(n) = integral_n^{n+1} (u-n)^2/u^2 du = 1 - 2n*ln(1+1/n) + n/(n+1)"""
    n_mp = mpf(n)
    return 1 - 2*n_mp*log(1 + 1/n_mp) + n_mp/(n_mp + 1)


def S_infinity():
    """Sum_{n=1}^inf I(n) = integral_1^inf {u}^2/u^2 du"""
    S = mpf(0)
    for n in range(1, 50000):
        S += I_n_exact(n)
        if n > 1000 and abs(I_n_exact(n)) < mpf(10)**(-45):
            break
    # Tail: sum_{n>N} I(n) ~ sum 1/(3n^2) + 1/(4n^3) + ...
    N = mpf(n)
    S += 1/(3*N) - 1/(6*N**2)
    return S


if __name__ == "__main__":
    print("PROVING THE NOVELTY FORMULA")
    print("=" * 70)

    # ================================================================
    # STEP 1: Exact diagonal formula
    # ================================================================
    print("\nSTEP 1: EXACT DIAGONAL FORMULA G_{jj}")
    print("-" * 70)

    # The key constant: S_inf = integral_1^inf {u}^2/u^2 du
    S_inf = S_infinity()
    print(f"S_inf = integral_1^inf {{u}}^2/u^2 du = {nstr(S_inf, 40)}")

    # Known: integral_0^1 {u}^2 du = integral_0^1 u^2 du = 1/3
    # But we need integral_1^inf...

    # Actually, the full integral integral_0^inf {u}^2/u^2 du diverges at 0.
    # S_inf is the part from 1 to inf.

    # Check if S_inf = known constant:
    # S_inf = 1 - ln(2pi)/2 + gamma/2?
    candidate1 = 1 - log(2*pi)/2 + euler/2
    # S_inf = gamma + 1 - ln(2)?
    candidate2 = euler + 1 - log(2)
    # S_inf = 1/3?
    candidate3 = mpf(1)/3
    # S_inf = 1 - gamma?
    candidate4 = 1 - euler
    # S_inf ~ 0.2639...
    # gamma = 0.5772...
    # ln(2) = 0.6931...
    # ln(2pi) = 1.8379...

    print(f"\nS_inf = {nstr(S_inf, 30)}")
    print(f"Candidates:")
    print(f"  1 - ln(2pi)/2 + gamma/2 = {nstr(candidate1, 30)}")
    print(f"  gamma + 1 - ln(2) = {nstr(candidate2, 30)}")
    print(f"  1/3 = {nstr(candidate3, 30)}")
    print(f"  1 - gamma = {nstr(candidate4, 30)}")

    # Try more:
    # S_inf = 1 - ln(2pi)/2?
    c5 = 1 - log(2*pi)/2
    # S_inf = gamma - 1/2 + ln(2)?
    c6 = euler - mpf(1)/2 + log(2)
    # 3/2 - gamma - ln(2pi)/2?
    c7 = mpf(3)/2 - euler - log(2*pi)/2

    print(f"  1 - ln(2pi)/2 = {nstr(c5, 30)}")
    print(f"  gamma - 1/2 + ln(2) = {nstr(c6, 30)}")
    print(f"  3/2 - gamma - ln(2pi)/2 = {nstr(c7, 30)}")

    # Compute the value to very high precision
    mp.dps = 100
    S_inf_hp = S_infinity()
    mp.dps = 50
    print(f"\nS_inf (100 digits) = {nstr(S_inf_hp, 60)}")

    # Try: 2*gamma - 1 + log(2pi) - 2?
    # Actually, let me use ISC (inverse symbolic calculator) approach:
    # Compute S_inf and try linear combinations of known constants

    val = float(S_inf)
    print(f"\nS_inf as float = {val}")
    print(f"  1/2 - ln(2)/2 = {0.5 - np.log(2)/2:.10f}")
    print(f"  (3 - 2*ln(2*pi))/6 = {(3 - 2*np.log(2*np.pi))/6:.10f}")

    # Brute force: try a*gamma + b*ln(2) + c*ln(pi) + d for rational a,b,c,d
    from fractions import Fraction
    gamma_f = float(euler)
    ln2 = float(log(2))
    lnpi = float(log(pi))
    target = float(S_inf)

    best_err = 1
    best_combo = None
    for a_num in range(-4, 5):
        for a_den in range(1, 7):
            for b_num in range(-4, 5):
                for b_den in range(1, 7):
                    for c_num in range(-4, 5):
                        for c_den in range(1, 7):
                            for d_num in range(-6, 7):
                                for d_den in range(1, 7):
                                    val_try = (a_num/a_den * gamma_f +
                                              b_num/b_den * ln2 +
                                              c_num/c_den * lnpi +
                                              d_num/d_den)
                                    err = abs(val_try - target)
                                    if err < best_err and err < 1e-8:
                                        best_err = err
                                        best_combo = (Fraction(a_num, a_den),
                                                     Fraction(b_num, b_den),
                                                     Fraction(c_num, c_den),
                                                     Fraction(d_num, d_den))

    if best_combo:
        a, b, c, d = best_combo
        print(f"\nBest match: S_inf = {a}*gamma + {b}*ln(2) + {c}*ln(pi) + {d}")
        print(f"  Value: {float(a)*gamma_f + float(b)*ln2 + float(c)*lnpi + float(d):.15f}")
        print(f"  Target: {target:.15f}")
        print(f"  Error: {best_err:.2e}")
    else:
        print(f"\nNo simple combination found")

    # Verify diagonal formula
    print(f"\nDiagonal verification: G_{{jj}} = (1/j)(1 - 1/j + S_inf)")
    print(f"{'j':>4} {'G_quad':>20} {'G_formula':>20} {'diff':>12}")
    for j in [1, 2, 3, 5, 10, 20]:
        g_q = gram_exact(j, j)
        g_f = (1 - 1/mpf(j) + S_inf) / mpf(j)
        print(f"{j:>4} {nstr(g_q, 15):>20} {nstr(g_f, 15):>20} {nstr(abs(g_q - g_f), 8):>12}")

    # ================================================================
    # STEP 2: Off-diagonal formula
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 2: OFF-DIAGONAL FORMULA G_{jk}")
    print("-" * 70)

    # G_{jk} = integral_0^1 {1/(jx)} {1/(kx)} dx
    # = (1/j) integral_{1/j}^inf {u} {ku/j} / u^2 du  (substituting u = 1/(jx))
    #
    # Hmm, {ku/j} depends on whether k/j is rational.
    # For integer j,k: ku/j has specific structure.

    # Let d = gcd(j,k), a = j/d, b = k/d (so gcd(a,b) = 1).
    # Then {ku/j} = {bu/a} which has period a.

    # KEY FORMULA ATTEMPT:
    # G_{jk} = (1/(jk)) * [something involving gcd and log]

    # Let's compute G_{jk} for many (j,k) and look for patterns
    print(f"\nG_{{jk}} for small j,k:")
    print(f"{'j':>3} {'k':>3} {'gcd':>4} {'G_{jk}':>20} {'G*jk':>12} {'G*lcm':>12}")

    gcd_data = {}
    for j in range(1, 16):
        for k in range(j, 16):
            g = gram_exact(j, k)
            d = gcd(j, k)
            lcm = j * k // d
            gjk_jk = float(g * j * k)
            gjk_lcm = float(g * lcm)

            key = (j//d, k//d)  # reduced pair
            if key not in gcd_data:
                gcd_data[key] = []
            gcd_data[key].append((j, k, float(g)))

            if k <= 8:
                print(f"{j:>3} {k:>3} {d:>4} {nstr(g, 15):>20} {gjk_jk:>12.6f} {gjk_lcm:>12.6f}")

    # Check: is G_{jk} * jk a function of (j/gcd, k/gcd)?
    print(f"\nG_{{jk}} * jk grouped by reduced pair (a,b) = (j/gcd, k/gcd):")
    for key in sorted(gcd_data.keys()):
        entries = gcd_data[key]
        vals = [e[2] * e[0] * e[1] for e in entries]
        if len(vals) > 1:
            print(f"  ({key[0]},{key[1]}): values = {[f'{v:.6f}' for v in vals[:5]]}, "
                  f"spread = {max(vals)-min(vals):.6e}")

    # ================================================================
    # STEP 3: The key — G_{jk} * jk as function of gcd(j,k)
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 3: G_{{jk}} * jk vs ARITHMETIC FUNCTIONS")
    print("-" * 70)

    # Hypothesis: G_{jk} = phi(j,k)/(jk) where phi depends on gcd structure
    # From the Mellin formula:
    #   G_{jk} = 1/(jk) + (1/2pi(jk)^{1/2}) integral (k/j)^{it} |zeta|^2/(1/4+t^2) dt + cross
    #
    # The pole-pole part: 1/(jk) — independent of gcd
    # The zeta-zeta part: depends on (k/j)^{it} = exp(it*ln(k/j))
    #   This oscillates, and the integral depends on ln(k/j)

    # KEY INSIGHT: G_{jk} should depend on ln(k/j) = ln(k) - ln(j)
    # But also on the ARITHMETIC relationship (via the cross terms)

    # Let me check: G_{jk} vs (log(gcd^2/(jk)) + gamma + 1) / (2jk) or similar

    # From Vasyunin's formula (if I recall correctly):
    # For the full Beurling space on (0, inf):
    #   <{theta/x}, {phi/x}> = -(1/2)[ln(theta/phi)]^2 / (4*pi^2*theta*phi) + ...
    # But our space is on (0,1), which is different.

    # Let me just look at what G_{jk}*jk - 1 depends on
    # (subtracting the rank-1 pole contribution)

    print(f"\n(G_{{jk}} * jk - 1) for coprime (j,k):")
    print(f"{'j':>3} {'k':>3} {'G*jk-1':>14} {'ln(k/j)':>10} {'ln(k)*ln(j)':>12}")
    for j in range(1, 12):
        for k in range(j, 12):
            if gcd(j, k) == 1:
                g = float(gram_exact(j, k))
                gjk = g * j * k - 1
                print(f"{j:>3} {k:>3} {gjk:>14.8f} {np.log(k/j):>10.4f} "
                      f"{np.log(k)*np.log(j):>12.6f}")

    # ================================================================
    # STEP 4: Compute C(j) = j^2 * dist_j^2 / EP(j) to high precision
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 4: HIGH-PRECISION C(j)")
    print("-" * 70)

    # Build small Gram matrix at high precision
    N = 30
    print(f"Computing {N}x{N} Gram matrix at 50-digit precision...")
    t0 = time.time()
    G_hp = [[None]*N for _ in range(N)]
    for j in range(1, N+1):
        for k in range(j, N+1):
            g = gram_exact(j, k)
            G_hp[j-1][k-1] = g
            G_hp[k-1][j-1] = g
        if j % 5 == 0:
            print(f"  Row {j}/{N} done ({time.time()-t0:.1f}s)")

    # Convert to numpy for Schur complement
    G_np = np.array([[float(G_hp[i][j]) for j in range(N)] for i in range(N)])

    # Compute dist_j^2 and C(j)
    print(f"\n{'j':>4} {'dist_j^2':>14} {'j^2*d^2':>10} {'EP':>8} {'C(j)':>12} "
          f"{'0.75*ln(j)+0.30':>16}")
    print("-" * 70)

    for j in range(1, N+1):
        if j == 1:
            dist_sq = G_np[0, 0]
        else:
            G_sub = G_np[:j-1, :j-1]
            g_cross = G_np[j-1, :j-1]
            coeffs = np.linalg.solve(G_sub, g_cross)
            dist_sq = G_np[j-1, j-1] - np.dot(g_cross, coeffs)

        ep = reduce(lambda a, p: a * (1 - 1.0/p**2),
                    sympy.factorint(j).keys(), 1.0) if j > 1 else 1.0
        C_j = j**2 * dist_sq / ep if ep > 0 else 0
        pred = 0.75 * np.log(j) + 0.30 if j > 1 else 0.26

        if j <= 15 or sympy.isprime(j) or j == N:
            print(f"{j:>4} {dist_sq:>14.8e} {j**2*dist_sq:>10.4f} {ep:>8.4f} "
                  f"{C_j:>12.6f} {pred:>16.4f}")

    # ================================================================
    # STEP 5: Analytical structure of C(j)
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 5: ANALYTICAL STRUCTURE OF C(j)")
    print("-" * 70)

    # C(j) = j^2 * dist_j^2 / EP(j)
    # dist_j^2 = G_{jj} - projection
    #
    # G_{jj} = (1/j)(1 - 1/j + S_inf) = (j-1)/j^2 + S_inf/j
    # j^2 * G_{jj} = j - 1 + j*S_inf
    #
    # j^2 * projection = j^2 * G_{j,<j} G_{<j}^{-1} G_{<j,j}
    # This is the hard part.
    #
    # For the FIRST Schur complement (j=2):
    # dist_2^2 = G_{22} - G_{21}^2 / G_{11}

    G_11 = float(gram_exact(1, 1))
    G_12 = float(gram_exact(1, 2))
    G_22 = float(gram_exact(2, 2))

    dist_2_sq = G_22 - G_12**2 / G_11

    print(f"\nj=2 Schur complement:")
    print(f"  G_11 = {G_11:.12f}")
    print(f"  G_12 = {G_12:.12f}")
    print(f"  G_22 = {G_22:.12f}")
    print(f"  dist_2^2 = G_22 - G_12^2/G_11 = {dist_2_sq:.12f}")
    print(f"  4*dist_2^2 = {4*dist_2_sq:.12f}")
    print(f"  EP(2) = 3/4 = {0.75:.4f}")
    print(f"  C(2) = 4*dist_2^2 / 0.75 = {4*dist_2_sq/0.75:.12f}")

    # For primes p:
    # dist_p^2 = G_{pp} - G_{p,<p} G_{<p}^{-1} G_{<p,p}
    # EP(p) = 1 - 1/p^2
    # C(p) = p^2 * dist_p^2 / (1 - 1/p^2)

    # For prime p, the main correlation is with f_1 (since gcd(p, k) = 1 for all k < p)
    # So dist_p^2 ≈ G_{pp} - sum of correlations

    # What's the "pure" part? G_{pp} = (p-1)/p^2 + S_inf/p
    # The projection onto f_1,...,f_{p-1} removes some of this.

    print(f"\nDiagonal formula check: G_{{jj}} = (j-1+j*S_inf)/j^2")
    S_val = float(S_inf)
    for j in [1, 2, 3, 5, 10, 20, 30]:
        g_formula = (j - 1 + j*S_val) / j**2
        g_actual = G_np[j-1, j-1]
        print(f"  j={j:>3}: formula={g_formula:.10f}, actual={g_actual:.10f}, "
              f"diff={abs(g_formula-g_actual):.2e}")

    print(f"\n{'='*70}")
    print("PROOF STATUS")
    print("=" * 70)
