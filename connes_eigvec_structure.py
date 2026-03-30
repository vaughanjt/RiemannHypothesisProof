"""
Session 25g: Eigenvector structure analysis.

a₀ = τ[0,0] ~ 0.025 (constant) — Rayleigh on constant mode fails.
The super-exponential ε₀ decay comes from the FULL MATRIX structure.

This script investigates:
1. Eigenvector components of ξ — is there a pattern we can exploit?
2. det(τ) — if computable, gives ε₀ via det = prod(eigenvalues)
3. Condition number and eigenvalue distribution of τ
4. Better test vectors for tighter Rayleigh bounds
"""

import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, sinh, hyp2f1, digamma, eig, quad,
                    sqrt, nstr, fabs, det as mpdet)
import sympy
import numpy as np
import time


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def build_tau_and_eigvec(lam_sq, N=30):
    """Build τ matrix, return eigenvalues, eigenvector, and matrix."""
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

    E, ER = eig(tau, left=False, right=True)
    evals_sorted = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    eps_0 = evals_sorted[0][0]; idx_0 = evals_sorted[0][1]
    eps_1 = evals_sorted[1][0]

    xi_mp = [ER[j, idx_0].real for j in range(dim)]
    xs = sum(xi_mp); sqL = sqrt(L)
    if fabs(xs) > mpf(10) ** (-20):
        xi_mp = [x * sqL / xs for x in xi_mp]

    all_evals = [evals_sorted[i][0] for i in range(dim)]

    return tau, xi_mp, all_evals, eps_0, eps_1, L, N


if __name__ == "__main__":
    DPS = 120
    mp.dps = DPS

    print(f"EIGENVECTOR & MATRIX STRUCTURE ANALYSIS (dps={DPS})")
    print("=" * 90)

    for lam_sq in [14, 50]:
        t0 = time.time()
        print(f"\nlam^2 = {lam_sq}...", end="", flush=True)
        tau, xi_mp, all_evals, eps_0, eps_1, L, N = build_tau_and_eigvec(lam_sq, N=30)
        dim = 2 * N + 1
        print(f" ({time.time()-t0:.0f}s)")

        print(f"  L = {nstr(L, 8)}, eps_0 = {nstr(eps_0, 15)}")
        print()

        # 1. EIGENVECTOR COMPONENTS
        print("  EIGENVECTOR xi (n=-5..5, center at N=30):")
        print(f"  {'n':>4} {'xi_n':>25} {'|xi_n|':>15}")
        for n in range(-10, 11):
            j = n + N
            print(f"  {n:>4} {nstr(xi_mp[j], 18):>25} {nstr(fabs(xi_mp[j]), 10):>15}")

        # Overall structure
        xi_abs = [float(fabs(x)) for x in xi_mp]
        max_idx = np.argmax(xi_abs)
        max_n = max_idx - N
        print(f"\n  Max |xi_n| at n={max_n}: {nstr(fabs(xi_mp[max_idx]), 15)}")
        print(f"  ||xi||_2 = {nstr(sqrt(sum(x**2 for x in xi_mp)), 10)}")
        print(f"  sum(xi_n) = {nstr(sum(xi_mp), 10)}")

        # Even/odd decomposition
        even_energy = sum(float(xi_mp[n+N]**2) for n in range(-N, N+1) if n % 2 == 0)
        odd_energy = sum(float(xi_mp[n+N]**2) for n in range(-N, N+1) if n % 2 != 0)
        print(f"  Even mode energy: {even_energy:.6f}")
        print(f"  Odd mode energy:  {odd_energy:.6f}")

        # 2. EIGENVALUE DISTRIBUTION
        print(f"\n  EIGENVALUE DISTRIBUTION (all {dim} eigenvalues):")
        print(f"  {'i':>4} {'eval':>25} {'log10|eval|':>15}")
        for i in range(min(10, dim)):
            ev = all_evals[i]
            log_ev = float(mpmath.log10(fabs(ev))) if fabs(ev) > 0 else float('-inf')
            print(f"  {i:>4} {nstr(ev, 18):>25} {log_ev:>15.2f}")
        print(f"  ...")
        for i in range(max(0, dim-3), dim):
            ev = all_evals[i]
            log_ev = float(mpmath.log10(fabs(ev))) if fabs(ev) > 0 else float('-inf')
            print(f"  {i:>4} {nstr(ev, 18):>25} {log_ev:>15.2f}")

        # 3. DETERMINANT
        print(f"\n  DETERMINANT:")
        tau_det = mpdet(tau)
        log_det = float(mpmath.log10(fabs(tau_det))) if fabs(tau_det) > 0 else float('-inf')
        # Product of eigenvalues for comparison
        log_prod = sum(float(mpmath.log10(fabs(ev))) for ev in all_evals if fabs(ev) > 0)
        print(f"  det(tau)      = {nstr(tau_det, 15)}")
        print(f"  log10|det|    = {log_det:.2f}")
        print(f"  sum log10|ev| = {log_prod:.2f}  (should match)")

        # 4. BETTER RAYLEIGH BOUNDS: Try eigenvector-informed test vectors
        print(f"\n  RAYLEIGH QUOTIENTS for various test vectors:")

        # v1: constant mode e_0
        v1 = [mpf(0)] * dim; v1[N] = mpf(1)
        rq1 = sum(v1[i] * sum(tau[i, j] * v1[j] for j in range(dim)) for i in range(dim))
        print(f"  e_0 (constant):     {nstr(rq1, 15)}")

        # v2: uniform vector
        v2 = [mpf(1) / sqrt(mpf(dim))] * dim
        rq2 = sum(v2[i] * sum(tau[i, j] * v2[j] for j in range(dim)) for i in range(dim))
        print(f"  uniform (1/sqrt):   {nstr(rq2, 15)}")

        # v3: the actual eigenvector (should give eps_0)
        xi_norm = sqrt(sum(x**2 for x in xi_mp))
        v3 = [x / xi_norm for x in xi_mp]
        rq3 = sum(v3[i] * sum(tau[i, j] * v3[j] for j in range(dim)) for i in range(dim))
        print(f"  eigenvector xi:     {nstr(rq3, 15)}  (should = eps_0)")

        # v4: "defect" vector = xi with small perturbation
        # Use xi but zero out the first component -> see how much RQ changes
        v4 = list(xi_mp)
        v4[N] = mpf(0)  # zero out n=0
        v4_norm = sqrt(sum(x**2 for x in v4))
        if v4_norm > 0:
            v4 = [x / v4_norm for x in v4]
            rq4 = sum(v4[i] * sum(tau[i, j] * v4[j] for j in range(dim)) for i in range(dim))
            print(f"  xi with xi_0=0:     {nstr(rq4, 15)}")

        print(f"\n  Time: {time.time()-t0:.0f}s")
        print()
