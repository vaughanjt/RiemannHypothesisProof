"""
Session 26k: Nyman-Beurling criterion — build the Gram matrix and analyze.

RH <=> chi_{(0,1]} in closure of span{rho_{1/k} : k=1,2,...} in L^2(0,1)
where rho_{1/k}(x) = {1/(kx)} (fractional part).

The Gram matrix G_{jk} = <rho_{1/j}, rho_{1/k}>_{L^2(0,1)}
                        = integral_0^1 {1/(jx)} {1/(kx)} dx

The distance: d_N^2 = 1 - b^T G_N^{-1} b
where b_j = <rho_{1/j}, 1>_{L^2} = integral_0^1 {1/(jx)} dx

Key questions:
1. What is the displacement rank of G_N?
2. How does sigma_min(G_N) behave as N grows?
3. Does G_N have the same structure as our Weil matrix tau?
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, quad, floor, nstr, fabs, log
import time

mp.dps = 50


def fractional_part(x):
    """Compute {x} = x - floor(x)."""
    return x - floor(x)


def gram_entry(j, k, n_quad=5000):
    """Compute G_{jk} = integral_0^1 {1/(jx)} {1/(kx)} dx.

    Split integral at x = 1/max(j,k) to handle the oscillating part.
    """
    j_mp = mpf(j); k_mp = mpf(k)

    def integrand(x):
        if x < mpf(10)**(-15):
            return mpf(0)
        fj = fractional_part(1 / (j_mp * x))
        fk = fractional_part(1 / (k_mp * x))
        return fj * fk

    # Integrate — use subdivision for better accuracy
    # The integrand has jump discontinuities at x = 1/(j*m) and x = 1/(k*m)
    # For moderate j,k, integrate by parts
    result = quad(integrand, [mpf(10)**(-10), mpf(1)],
                  method='gauss-legendre', maxdegree=7)
    return result


def gram_entry_fast(j, k, M=500):
    """Compute G_{jk} by summing over integer-part intervals.

    For x in (1/(j*(a+1)), 1/(j*a)): {1/(jx)} = 1/(jx) - a
    For x in (1/(k*(b+1)), 1/(k*b)): {1/(kx)} = 1/(kx) - b

    We sum over all (a,b) pairs where the intervals overlap.
    """
    j_mp = mpf(j); k_mp = mpf(k)
    total = mpf(0)

    for a in range(1, M + 1):
        # Interval for j: (1/(j*(a+1)), 1/(j*a))
        xj_lo = 1 / (j_mp * (a + 1))
        xj_hi = 1 / (j_mp * a)

        for b in range(1, M + 1):
            # Interval for k: (1/(k*(b+1)), 1/(k*b))
            xk_lo = 1 / (k_mp * (b + 1))
            xk_hi = 1 / (k_mp * b)

            # Intersection
            lo = max(xj_lo, xk_lo)
            hi = min(xj_hi, xk_hi)
            if lo >= hi:
                continue
            if hi > 1:
                hi = mpf(1)
            if lo >= 1:
                continue

            # On [lo, hi]: {1/(jx)} = 1/(jx) - a, {1/(kx)} = 1/(kx) - b
            # Integral of (1/(jx)-a)(1/(kx)-b) dx from lo to hi
            # = integral [1/(jkx^2) - b/(jx) - a/(kx) + ab] dx
            # = [-1/(jkx) - (b/j)ln(x) - (a/k)ln(x) + ab*x] from lo to hi
            val = (-1/(j_mp*k_mp*hi) + 1/(j_mp*k_mp*lo)
                   - (b/j_mp + a/k_mp) * (mpmath.log(hi) - mpmath.log(lo))
                   + a * b * (hi - lo))
            total += val

    return total


def b_entry(k, M=500):
    """Compute b_k = <rho_{1/k}, 1>_{L^2(0,1)} = integral_0^1 {1/(kx)} dx."""
    k_mp = mpf(k)
    total = mpf(0)

    for a in range(1, M + 1):
        lo = 1 / (k_mp * (a + 1))
        hi = min(1 / (k_mp * a), mpf(1))
        if lo >= hi or lo >= 1:
            continue

        # Integral of (1/(kx) - a) dx from lo to hi
        # = [ln(x)/(-k) ... wait:
        # integral (1/(kx) - a) dx = (1/k)ln(x) - ax
        val = (mpmath.log(hi) - mpmath.log(lo)) / k_mp - a * (hi - lo)
        total += val

    return total


if __name__ == "__main__":
    N = 20  # Size of Gram matrix

    print(f"NYMAN-BEURLING GRAM MATRIX (N={N})")
    print("=" * 70)

    # Build Gram matrix
    print("Building Gram matrix (interval summation)...", flush=True)
    t0 = time.time()
    G = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(j, N + 1):
            val = float(gram_entry_fast(j, k, M=1000))
            G[j-1, k-1] = val
            G[k-1, j-1] = val
        if j % 5 == 0:
            print(f"  row {j}/{N} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  Built in {time.time()-t0:.0f}s")

    # Build b vector
    b = np.array([float(b_entry(k, M=1000)) for k in range(1, N + 1)])

    # Eigenvalues
    evals = np.linalg.eigvalsh(G)
    print(f"\n  Eigenvalues of G (sorted):")
    for i in range(min(10, N)):
        print(f"    sigma_{i+1} = {evals[i]:.10e}")
    print(f"    ...")
    print(f"    sigma_{N} = {evals[-1]:.10e}")
    print(f"  sigma_min = {evals[0]:.10e}")
    print(f"  sigma_max = {evals[-1]:.10e}")
    print(f"  condition number = {evals[-1]/evals[0]:.4e}")

    # Distance
    try:
        c_opt = np.linalg.solve(G, b)
        d_sq = 1.0 - np.dot(b, c_opt)
        print(f"\n  d_N^2 = {d_sq:.10e}")
        print(f"  d_N = {np.sqrt(max(0, d_sq)):.10e}")
    except:
        print(f"\n  G is singular, cannot compute distance")

    # DISPLACEMENT RANK CHECK
    print(f"\n  DISPLACEMENT RANK CHECK:")
    D = np.diag(np.arange(1, N + 1, dtype=float))
    disp = D @ G - G @ D
    U, S, Vt = np.linalg.svd(disp)
    print(f"  Top 8 singular values of D*G - G*D:")
    for i in range(min(8, N)):
        print(f"    s_{i+1} = {S[i]:.6e}")
    eff_rank = np.sum(S > S[0] * 1e-10)
    print(f"  Effective displacement rank: {eff_rank}")

    # Structure of G
    print(f"\n  GRAM MATRIX STRUCTURE (first 5x5):")
    for i in range(min(5, N)):
        row = "  ".join(f"{G[i,j]:10.6f}" for j in range(min(5, N)))
        print(f"    {row}")

    print(f"\n  Diagonal: {', '.join(f'{G[i,i]:.6f}' for i in range(min(10,N)))}")
    print(f"  Off-diag ratios G[1,k]/G[1,1]: {', '.join(f'{G[0,k]/G[0,0]:.4f}' for k in range(min(10,N)))}")
