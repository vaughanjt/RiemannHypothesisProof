"""
Session 25h: Verify Cauchy-like displacement structure of tau.

From Grok / Lemma 5.1, tau has the form:
  tau_{ij} = a_i             if i = j
  tau_{ij} = (b_i - b_j)/(i-j)  if i != j

Displacement equation: D*tau - tau*D = v*u^T - u*v^T
where D = diag(-N,...,N), u = (1,...,1)^T, v = (b_{-N},...,b_N)^T

This script:
1. Extracts a_n (diagonal) and b_n (from off-diagonal) from computed tau
2. Verifies consistency: (b_i-b_j)/(i-j) = tau_{ij} for ALL off-diagonal pairs
3. Computes displacement D*tau - tau*D, verifies rank = 2
4. Studies b_n as function of L: decomposes into limit + O(e^{-2L}) correction
5. Checks analyticity properties needed for Beckermann bound
"""

import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, sinh, hyp2f1, digamma, eig, quad,
                    sqrt, nstr, fabs, svd)
import sympy
import numpy as np
import time


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def build_tau(lam_sq, N=30):
    """Build tau matrix, return it with L."""
    L = log(mpf(lam_sq)); eL = exp(L)
    vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq):
            vM.append((pk, lp, log(pk))); pk *= p
    dim = 2 * N + 1
    al = {}
    for n in range(-N, N + 1):
        nn = abs(n)
        if nn == 0:
            al[n] = mpf(0); continue
        z = exp(-2 * L); a = pi * mpc(0, nn) / L + mpf(1) / 4
        h = hyp2f1(1, a, a + 1, z)
        al[n] = (exp(-L / 2) * (2 * L / (L + 4 * pi * mpc(0, nn)) * h).imag
                 + digamma(a).imag / 2) / pi
        if n < 0:
            al[n] = -al[n]
    wr_d = {}
    for nv in range(N + 1):
        w_c = euler + log(4 * pi * (eL - 1) / (eL + 1))
        def ig(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        wr_d[nv] = w_c + quad(ig, [mpf(0), L])
        wr_d[-nv] = wr_d[nv]
    tau = mpmatrix(dim, dim)
    L2 = L * L; p2 = 16 * pi * pi; pf = 32 * L * sinh(L / 4) ** 2
    def q_mp(n, m, y):
        if n != m:
            return (sin(2 * pi * m * y / L) - sin(2 * pi * n * y / L)) / (pi * (n - m))
        else:
            return 2 * (L - y) / L * cos(2 * pi * n * y / L)
    for i in range(dim):
        n = i - N
        for j in range(i, dim):
            m = j - N
            w02 = pf * (L2 - p2 * m * n) / ((L2 + p2 * m ** 2) * (L2 + p2 * n ** 2))
            wp = sum(lk * pkv ** (-mpf(1) / 2) * q_mp(n, m, logk) for pkv, lk, logk in vM)
            wr = wr_d[n] if n == m else (al[m] - al[n]) / (n - m)
            tau[i, j] = w02 - wr - wp
            tau[j, i] = tau[i, j]
    return tau, L, N


if __name__ == "__main__":
    DPS = 80
    mp.dps = DPS

    print(f"DISPLACEMENT STRUCTURE VERIFICATION (dps={DPS})")
    print("=" * 90)

    for lam_sq in [14, 50, 100]:
        t0 = time.time()
        print(f"\nlam^2 = {lam_sq}...", end="", flush=True)
        tau, L, N = build_tau(lam_sq, N=30)
        dim = 2 * N + 1
        print(f" ({time.time()-t0:.0f}s)")
        print(f"  L = {nstr(L, 8)}, dim = {dim}")

        # 1. EXTRACT a_n and b_n
        # a_n = tau[n+N, n+N] (diagonal)
        a_n = [tau[n + N, n + N] for n in range(-N, N + 1)]

        # b_n: from off-diagonal, b_n - b_0 = n * tau[n+N, N] for n != 0
        # Set b_0 = 0 (gauge choice)
        b_n = [mpf(0)] * dim
        for n in range(-N, N + 1):
            if n == 0:
                b_n[n + N] = mpf(0)
            else:
                b_n[n + N] = n * tau[n + N, N]

        # 2. VERIFY CONSISTENCY: (b_i - b_j)/(i-j) = tau_{ij} for i != j
        max_err = mpf(0)
        n_checks = 0
        for i in range(-N, N + 1):
            for j in range(-N, N + 1):
                if i == j:
                    continue
                predicted = (b_n[i + N] - b_n[j + N]) / (i - j)
                actual = tau[i + N, j + N]
                err = fabs(predicted - actual)
                if err > max_err:
                    max_err = err
                n_checks += 1

        print(f"\n  CAUCHY-LIKE VERIFICATION:")
        print(f"  Checked {n_checks} off-diagonal pairs")
        print(f"  Max |predicted - actual| = {nstr(max_err, 8)}")
        print(f"  Relative to max|tau_ij|:   {nstr(max_err / max(fabs(tau[i+N,j+N]) for i in range(-N,N+1) for j in range(-N,N+1) if i!=j), 4)}")

        # 3. DISPLACEMENT RANK: compute D*tau - tau*D
        # (D*tau - tau*D)_{ij} = (i-j) * tau_{ij}  (using n,m indexing)
        # For i != j: this should equal b_i - b_j
        # For i = j: this is 0
        # So displacement = v*u^T - u*v^T where u=(1,...,1), v=(b_{-N},...,b_N)
        # This has rank 2 (antisymmetric rank-1 pair)

        # Build displacement matrix as numpy for SVD
        disp = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                n = i - N; m = j - N
                disp[i, j] = float((n - m) * tau[i, j])

        # SVD to find rank
        U, S, Vt = np.linalg.svd(disp)
        print(f"\n  DISPLACEMENT D*tau - tau*D:")
        print(f"  Top 6 singular values: {', '.join(f'{s:.6e}' for s in S[:6])}")
        print(f"  Ratio S[2]/S[0] = {S[2]/S[0]:.2e}  (should be ~0 for rank 2)")
        effective_rank = np.sum(S > S[0] * 1e-10)
        print(f"  Effective rank (tol 1e-10): {effective_rank}")

        # 4. b_n STRUCTURE
        print(f"\n  b_n SEQUENCE (n=0..10):")
        print(f"  {'n':>4} {'b_n':>25} {'b_n/n':>20} {'b_{-n}':>25}")
        for n in range(11):
            bn = b_n[n + N]
            bn_neg = b_n[-n + N] if n > 0 else mpf(0)
            bn_over_n = bn / n if n > 0 else mpf(0)
            print(f"  {n:>4} {nstr(bn, 15):>25} {nstr(bn_over_n, 15):>20} {nstr(bn_neg, 15):>25}")

        # Check antisymmetry: b_{-n} = -b_n
        max_antisym_err = max(fabs(b_n[n + N] + b_n[-n + N]) for n in range(1, N + 1))
        print(f"  Antisymmetry check: max|b_n + b_{{-n}}| = {nstr(max_antisym_err, 4)}")

        # 5. ANALYTICITY: study b_n as function of n
        # For Beckermann bound, we need b_n to be analytic in n
        # (extendable to a strip around the real axis)
        # Key: b_n involves alpha_n which contains 2F1(1, pi*i*n/L+1/4; ...; e^{-2L})
        # The small parameter e^{-2L} controls the analyticity radius

        e_minus_2L = float(exp(-2 * L))
        print(f"\n  ANALYTICITY PARAMETER:")
        print(f"  e^{{-2L}} = {e_minus_2L:.6e}  (the small parameter in 2F1)")
        print(f"  log10(e^{{-2L}}) = {float(-2 * L / mpmath.log(10)):.2f}")

        # Compute b_n / n for large n to see if it has a limit
        if N >= 20:
            ratios = [float(b_n[n + N] / n) for n in range(10, N + 1)]
            print(f"  b_n/n for n=10..{N}: min={min(ratios):.6f}, max={max(ratios):.6f}")
            print(f"  b_n/n appears to {'converge' if max(ratios)-min(ratios) < 0.01 else 'vary'}")

        print(f"\n  Time: {time.time()-t0:.0f}s")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("If displacement rank = 2 and b_n is analytic in n:")
    print("  -> Beckermann/Tyrtyshnikov theorem gives sigma_k <= C * rho^k")
    print("  -> With rho ~ e^{-2L}, we get |eps_0| <= C * exp(-c*N*L)")
    print("  -> This is the rigorous bound needed for the proof")
