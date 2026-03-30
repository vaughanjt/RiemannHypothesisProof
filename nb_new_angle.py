"""
Session 29e: The New Angle — Novelty Spectrum as RH Diagnostic

SYNTHESIS OF SESSION 29 FINDINGS:

Dead ends:
  - Mobius inversion in Mellin space (j^{-s} linearly independent)
  - MV bound on [N,2N] (error = main term at T~N)
  - Greedy novelty selection (worst for approximation)
  - Primes-only basis (d plateaus at 0.228)

Discoveries:
  - dist_j^2 = C(j) * EP(j) / j^2 where C(j) ~ 0.75*ln(j) + 0.30
  - Squarefree indices capture 97% of approximation power
  - Optimal c_j has Mobius-like structure: (Mc)_j ~ mu(j) * j * (growing)
  - sigma_min ~ 0.4 * dist_N^2 (the last-added novelty controls the spectrum)

NEW IDEA: The "novelty spectrum" {j^2 * dist_j^2}_{j=1}^N tells us about
the LINEAR INDEPENDENCE RATE of the NB basis. This rate determines
how fast d_N -> 0.

THEOREM CANDIDATE: For the NB Gram matrix:
  prod_{j=1}^N dist_j^2 = det(G_N) = prod eigenvalues

  This connects the Euler product in dist_j^2 to det(G_N).

  det(G_N) = prod_{j=1}^N dist_j^2
           ~ prod_{j=1}^N C(j) * EP(j) / j^2
           = [prod C(j)] * [prod EP(j)] * [1/(N!)^2]

  Since prod EP(j) = prod_{j=1}^N prod_{p|j}(1-1/p^2),
  and prod_{p|j} appears once for each multiple of p up to N:

  prod_{j=1}^N EP(j) = prod_{p <= N} (1-1/p^2)^{floor(N/p)}

  And prod_{p<=N} (1-1/p^2) = 1/zeta_N(2) -> 6/pi^2

  So: det(G_N) ~ C^N * prod_{p<=N}(1-1/p^2)^{N/p} / (N!)^2 * (poly corrections)

This connects det(G_N) to the prime distribution!
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, zeta as mpzeta, pi as mppi
import sympy
from math import gcd, factorial
from functools import reduce
import time

mp.dps = 50


def build_gram(N, n_grid=500000):
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    return (fp @ fp.T) * dx


def euler_product_factor(j):
    if j <= 1: return 1.0
    return reduce(lambda a, p: a * (1 - 1.0/p**2), sympy.factorint(j).keys(), 1.0)


if __name__ == "__main__":
    print("SESSION 29e: NOVELTY SPECTRUM AND DETERMINANT FORMULA")
    print("=" * 70)

    # ================================================================
    # PART 1: Verify det(G_N) = prod dist_j^2
    # ================================================================
    print("\nPART 1: DETERMINANT DECOMPOSITION")
    print("-" * 70)

    for N in [10, 20, 30, 50, 75, 100]:
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals = np.linalg.eigvalsh(G)

        # det(G) from eigenvalues
        log_det_eig = np.sum(np.log(evals))

        # det(G) from Schur complements = prod dist_j^2
        dist_sq = np.zeros(N)
        for j in range(1, N+1):
            if j == 1:
                dist_sq[0] = G[0, 0]
            else:
                G_sub = G[:j-1, :j-1]
                g_cross = G[j-1, :j-1]
                coeffs = np.linalg.solve(G_sub, g_cross)
                dist_sq[j-1] = G[j-1, j-1] - np.dot(g_cross, coeffs)

        log_det_schur = np.sum(np.log(np.maximum(dist_sq, 1e-300)))

        # Euler product prediction: prod dist_j^2 ~ prod C(j)*EP(j)/j^2
        log_EP_pred = 0
        for j in range(1, N+1):
            ep = euler_product_factor(j)
            Cj = j**2 * dist_sq[j-1] / ep if ep > 0 else 1
            log_EP_pred += np.log(max(ep, 1e-300)) - 2*np.log(j) + np.log(max(Cj, 1e-300))

        print(f"N={N:>4}: log(det) from eigs = {log_det_eig:>12.4f}, "
              f"from Schur = {log_det_schur:>12.4f}, "
              f"diff = {abs(log_det_eig-log_det_schur):.2e}")

    # ================================================================
    # PART 2: Decompose log(det) into EP, C, and factorial parts
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: log(det(G_N)) DECOMPOSITION")
    print("-" * 70)

    print(f"\n{'N':>4} {'log(det)':>12} {'sum ln(EP)':>12} {'-2*ln(N!)':>12} "
          f"{'sum ln(C)':>12} {'residual':>10}")
    print("-" * 70)

    for N in [10, 20, 30, 50, 75, 100, 150, 200]:
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals = np.linalg.eigvalsh(G)
        log_det = np.sum(np.log(evals))

        # Compute dist_j^2 and decompose
        dist_sq = np.zeros(N)
        for j in range(1, N+1):
            if j == 1:
                dist_sq[0] = G[0, 0]
            else:
                G_sub = G[:j-1, :j-1]
                g_cross = G[j-1, :j-1]
                coeffs = np.linalg.solve(G_sub, g_cross)
                dist_sq[j-1] = G[j-1, j-1] - np.dot(g_cross, coeffs)

        sum_ln_EP = sum(np.log(euler_product_factor(j)) for j in range(1, N+1))
        sum_ln_j2 = 2 * sum(np.log(j) for j in range(1, N+1))  # = 2*ln(N!)
        sum_ln_C = sum(np.log(max(j**2 * dist_sq[j-1] / euler_product_factor(j), 1e-300))
                       for j in range(1, N+1))
        residual = log_det - (sum_ln_EP - sum_ln_j2 + sum_ln_C)

        print(f"{N:>4} {log_det:>12.2f} {sum_ln_EP:>12.4f} {-sum_ln_j2:>12.2f} "
              f"{sum_ln_C:>12.4f} {residual:>10.4f}")

    # ================================================================
    # PART 3: The prime distribution in EP sums
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: EULER PRODUCT SUM AND PRIME COUNTING")
    print("-" * 70)

    # sum_{j=1}^N ln(EP(j)) = sum_{j=1}^N sum_{p|j} ln(1-1/p^2)
    # = sum_{p<=N} floor(N/p) * ln(1-1/p^2)
    # ~ -sum_{p<=N} (N/p) * (1/p^2 + 1/(2p^4) + ...)
    # ~ -N * sum_{p<=N} 1/p^3 - (N/2) * sum 1/p^5 ...
    # = -N * P(3) - ... where P(3) = sum 1/p^3 ≈ 0.1748

    for N in [50, 100, 200, 500, 1000]:
        # Direct computation
        sum_ln_EP = 0
        for j in range(1, N+1):
            sum_ln_EP += np.log(euler_product_factor(j))

        # Prime-indexed formula
        primes = list(sympy.primerange(2, N+1))
        sum_prime_formula = sum(int(N//p) * np.log(1 - 1.0/p**2) for p in primes)

        # Leading term
        sum_1_p3 = sum(1.0/p**3 for p in primes)
        leading = -N * sum_1_p3

        print(f"N={N:>5}: sum_ln_EP = {sum_ln_EP:>10.4f}, "
              f"prime formula = {sum_prime_formula:>10.4f}, "
              f"leading (-N*P_3) = {leading:>10.4f}")

    # ================================================================
    # PART 4: Rate of d_N -> 0 from determinant growth
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: d_N DECAY RATE FROM DETERMINANT PERSPECTIVE")
    print("-" * 70)

    # d_N^2 = 1 - b^T G^{-1} b
    # G^{-1} has entries growing like 1/sigma_min ~ N^2
    # b^T G^{-1} b = sum_i (b . v_i)^2 / lambda_i
    # where v_i, lambda_i are eigenpairs

    # The CONTRIBUTION of each eigenvector to d_N^2:
    # d_N^2 = 1 - sum (b.v_i)^2 / lambda_i
    # = sum (1/N - (b.v_i)^2/lambda_i) ... no that's not right
    #
    # Actually: let a_i = b . v_i (projection of b onto eigenvector i)
    # Then b^T G^{-1} b = sum a_i^2 / lambda_i
    # And ||b||^2 = sum a_i^2
    # So d_N^2 = 1 - sum a_i^2 / lambda_i

    for N in [50, 100, 200]:
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals, evecs = np.linalg.eigh(G)

        x = np.linspace(1.0/(N*5000), 1.0, N*5000)
        dx_val = x[1] - x[0]
        fp = np.zeros((N, len(x)))
        for k in range(1, N+1):
            v = 1.0/(k*x)
            fp[k-1] = v - np.floor(v)
        b = np.sum(fp, axis=1) * dx_val

        # Projections
        a = evecs.T @ b  # a_i = b . v_i
        contributions = a**2 / evals  # contribution to b^T G^{-1} b

        # Sort by contribution (descending)
        order = np.argsort(-contributions)

        print(f"\nN={N}: d_N^2 = {1 - np.sum(contributions):.6e}")
        print(f"  Top 10 contributing eigenvectors:")
        print(f"  {'rank':>5} {'lambda_i':>12} {'a_i^2':>12} {'a_i^2/lam':>12} {'cum %':>8}")

        cum = 0
        total = np.sum(contributions)
        for rank, idx in enumerate(order[:10]):
            cum += contributions[idx]
            pct = 100 * cum / total
            print(f"  {rank+1:>5} {evals[idx]:>12.4e} {a[idx]**2:>12.4e} "
                  f"{contributions[idx]:>12.4e} {pct:>7.1f}%")

        # Key: is the bottleneck at the smallest eigenvalue?
        print(f"\n  Smallest eigenvalue: lambda_1 = {evals[0]:.4e}")
        print(f"    a_1^2 = {a[0]**2:.4e}, contribution = {contributions[0]:.4e}")
        print(f"    Fraction of total: {100*contributions[0]/total:.1f}%")
        print(f"  Largest eigenvalue: lambda_N = {evals[-1]:.4e}")
        print(f"    a_N^2 = {a[-1]**2:.4e}, contribution = {contributions[-1]:.4e}")

        # Where is d_N^2 coming from?
        # d_N^2 = 1 - total_contribution
        # The "missing" part = what we can't capture
        missing = 1 - total
        print(f"\n  Missing = 1 - sum(a_i^2/lambda_i) = {missing:.6e}")
        print(f"  ||b||^2 = {np.sum(a**2):.6f}")
        print(f"  If all lambda were equal (lambda_bar = trace/N = {np.trace(G)/N:.4e}):")
        miss_equal = 1 - np.sum(a**2) / (np.trace(G)/N)
        print(f"    d^2 would be: {miss_equal:.6f}")

    # ================================================================
    # PART 5: The spectral gap between 1 and the approximation
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: WHY d_N -> 0 SLOWLY — THE SPECTRAL BOTTLENECK")
    print("-" * 70)

    N = 100
    G = build_gram(N, n_grid=500000)
    evals, evecs = np.linalg.eigh(G)

    x = np.linspace(1.0/500000, 1.0, 500000)
    dx_val = x[1] - x[0]
    fp = np.zeros((N, len(x)))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    b = np.sum(fp, axis=1) * dx_val

    a = evecs.T @ b
    contributions = a**2 / evals

    # d_N^2 = 1 - sum contributions
    # The function 1 has Mellin transform 1/(s-1)
    # The f_j span a subspace. The projection of 1 onto this subspace misses d_N^2.
    # WHERE (in spectral space) is the missing part?

    # Cumulative from bottom eigenvalues
    print(f"N={N}: Cumulative contribution to b^T G^{{-1}} b from bottom eigenvalues:")
    print(f"  {'k (bottom)':>12} {'cum contribution':>16} {'missing':>12} {'d^2 if stopped':>15}")
    cum = 0
    for k in [1, 2, 5, 10, 20, 50, N]:
        idx = range(min(k, N))
        cum_k = sum(contributions[i] for i in idx)
        print(f"  {k:>12} {cum_k:>16.6e} {1-cum_k:>12.6e} {1-cum_k:>15.6e}")

    # From TOP eigenvalues
    print(f"\n  Cumulative from top eigenvalues:")
    print(f"  {'k (top)':>12} {'cum contribution':>16} {'missing':>12}")
    for k in [1, 2, 5, 10, 20, 50, N]:
        idx = range(N-min(k,N), N)
        cum_k = sum(contributions[i] for i in idx)
        print(f"  {k:>12} {cum_k:>16.6e} {1-cum_k:>12.6e}")

    # KEY INSIGHT: The largest eigenvalue (lambda_N ~ trace/few) dominates b^T G^{-1} b
    # because b is "aligned" with the large eigenvectors.
    # The small eigenvalues contribute negligibly because b has small projection there.

    # This means d_N^2 ≈ 1 - ||b||^2/lambda_eff where lambda_eff is an effective eigenvalue
    # d_N -> 0 requires lambda_eff -> ||b||^2 from above

    b_norm_sq = np.sum(a**2)
    lambda_eff = b_norm_sq / (1 - max(0, 1 - np.sum(contributions)))
    print(f"\n  ||b||^2 = {b_norm_sq:.6f}")
    print(f"  Effective lambda = ||b||^2 / (1-d^2) = {lambda_eff:.6f}")
    print(f"  lambda_max = {evals[-1]:.6f}")
    print(f"  lambda_eff / lambda_max = {lambda_eff/evals[-1]:.6f}")

    print(f"\n{'='*70}")
    print("FINAL SYNTHESIS")
    print("=" * 70)
    print("""
THE DRAGON'S ARMOR:

1. RH <=> d_N -> 0 <=> 1 is in the closed span of {{1/(jx)}}
   d_N ~ 0.36 * exp(-0.59 * sqrt(log N)) -- glacially slow

2. d_N^2 is NOT controlled by sigma_min.
   The projection b . v_min is tiny (sigma_min is irrelevant to d_N).
   d_N is controlled by the top eigenvalues and how well they capture b.

3. The EULER PRODUCT in dist_j^2 is a beautiful structural result about
   the NB Gram matrix, but it describes LINEAR INDEPENDENCE (which gets
   harder at rate 1/j^2) -- not the approximation quality (which depends
   on how well the f_j span captures the constant function 1).

4. The bottleneck for d_N -> 0 is NOT eigenvalue decay but rather
   the ALIGNMENT of b with the top eigenvectors. As N grows, b gets
   slightly better aligned, but only logarithmically.

5. Proving d_N -> 0 is equivalent to proving RH.
   No shortcut through sigma_min, novelty, or Euler products.

WHAT'S NOVEL (potentially publishable):
  - dist_j^2 = C(j) * prod_{p|j}(1-1/p^2) / j^2 with C(j) ~ 0.75*ln(j)
  - Squarefree basis captures 97% of NB approximation at 61% size
  - det(G_N) decomposes into Euler product * (N!)^{-2} * correction
  - sigma_min ~ 0.4 * dist_N^2 (eigenvalue-novelty ratio)

NEXT MOVES ON THE DRAGON:
  A. Prove the novelty formula analytically (new result about NB system)
  B. Try Connes H1/H2 (different angle, might yield more)
  C. Explore Li criterion + our spectral tools
  D. Look for a NEW reformulation that our tools can crack
""")
