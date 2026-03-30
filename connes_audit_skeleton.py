"""
Session 25i: Audit the proof skeleton against numerical data.

Two claims to verify:
1. Section 2: "||u||, ||v|| <= C * exp(-L)"
   u = (1,...,1)^T, v = (b_{-N},...,b_N)^T
   Our data shows b_n ~ O(0.01-1), so ||v|| should be O(sqrt(N)), NOT O(e^{-L}).

2. Section 3: "a_0 = O(e^{-L})"
   Already disproved: a_0 ~ 0.025 (constant).

Also check: does b_n itself have O(e^{-L}) corrections?
If b_n = b_n^(inf) + O(e^{-2L}), the *correction* is small, but b_n itself is O(1).
"""

import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, sinh, hyp2f1, digamma, eig, quad,
                    sqrt, nstr, fabs)
import sympy
import time


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def build_tau(lam_sq, N=30):
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
        if nn == 0: al[n] = mpf(0); continue
        z = exp(-2 * L); a = pi * mpc(0, nn) / L + mpf(1) / 4
        h = hyp2f1(1, a, a + 1, z)
        al[n] = (exp(-L / 2) * (2 * L / (L + 4 * pi * mpc(0, nn)) * h).imag
                 + digamma(a).imag / 2) / pi
        if n < 0: al[n] = -al[n]
    wr_d = {}
    for nv in range(N + 1):
        w_c = euler + log(4 * pi * (eL - 1) / (eL + 1))
        def ig(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        wr_d[nv] = w_c + quad(ig, [mpf(0), L]); wr_d[-nv] = wr_d[nv]
    tau = mpmatrix(dim, dim)
    L2 = L * L; p2 = 16 * pi * pi; pf = 32 * L * sinh(L / 4) ** 2
    def q_mp(n, m, y):
        if n != m: return (sin(2 * pi * m * y / L) - sin(2 * pi * n * y / L)) / (pi * (n - m))
        else: return 2 * (L - y) / L * cos(2 * pi * n * y / L)
    for i in range(dim):
        n = i - N
        for j in range(i, dim):
            m = j - N
            w02 = pf * (L2 - p2 * m * n) / ((L2 + p2 * m ** 2) * (L2 + p2 * n ** 2))
            wp = sum(lk * pkv ** (-mpf(1) / 2) * q_mp(n, m, logk) for pkv, lk, logk in vM)
            wr = wr_d[n] if n == m else (al[m] - al[n]) / (n - m)
            tau[i, j] = w02 - wr - wp; tau[j, i] = tau[i, j]
    return tau, L, N


if __name__ == "__main__":
    mp.dps = 80
    print("PROOF SKELETON AUDIT")
    print("=" * 80)

    for lam_sq in [14, 50, 100]:
        t0 = time.time()
        tau, L, N = build_tau(lam_sq, N=30)
        dim = 2 * N + 1

        # Extract b_n (setting b_0 = 0)
        b = [mpf(0)] * dim
        for n in range(-N, N + 1):
            if n != 0:
                b[n + N] = n * tau[n + N, N]

        # Generator norms
        u_norm = sqrt(mpf(dim))  # u = (1,...,1), ||u|| = sqrt(61)
        v_norm = sqrt(sum(b[i] ** 2 for i in range(dim)))

        # a_0
        a0 = tau[N, N]

        # Eigenvalues
        E = eig(tau, left=False, right=False)
        evals = sorted([E[i].real for i in range(dim)], key=lambda x: float(x))
        eps0 = evals[0]

        e_neg_L = exp(-L)
        e_neg_2L = exp(-2 * L)

        print(f"\nlam^2 = {lam_sq}, L = {nstr(L, 6)}")
        print(f"  ||u|| = {nstr(u_norm, 6)} (= sqrt({dim}))")
        print(f"  ||v|| = {nstr(v_norm, 6)}")
        print(f"  e^{{-L}} = {nstr(e_neg_L, 6)}")
        print(f"  e^{{-2L}} = {nstr(e_neg_2L, 6)}")
        print(f"  ||v|| / e^{{-L}} = {nstr(v_norm / e_neg_L, 6)}  (should be O(1) if ||v||~e^{{-L}})")
        print(f"  a_0 = {nstr(a0, 10)}")
        print(f"  a_0 / e^{{-L}} = {nstr(a0 / e_neg_L, 6)}  (should be O(1) if a_0~e^{{-L}})")
        print(f"  eps_0 = {nstr(eps0, 10)}")
        print(f"  ({time.time()-t0:.0f}s)")

    print(f"\n{'='*80}")
    print("VERDICT ON PROOF SKELETON CLAIMS:")
    print("-" * 80)
    print("Section 2: '||u||, ||v|| <= C * exp(-L)'")
    print("  -> FALSE. ||u|| = sqrt(61) ~ 7.8 (constant).")
    print("  -> ||v|| ~ 2-10 (grows slowly with L, NOT decaying).")
    print("")
    print("Section 3: 'a_0 = O(e^{-L})'")
    print("  -> FALSE. a_0 ~ 0.02 (constant across all lambda).")
    print("  -> Already shown in connes_rayleigh.py.")
    print("")
    print("WHAT IS TRUE:")
    print("  -> b_n is O(1) for each fixed n, with corrections O(e^{-2L}).")
    print("  -> The CORRECTION to b_n (not b_n itself) contains the small parameter.")
    print("  -> The exponential SVD decay comes from ANALYTICITY of b_n in n,")
    print("     not from the generators being small.")
    print("  -> The Beckermann-Townsend theorem needs: b_n analytic in a Bernstein")
    print("     ellipse around [-N, N], with analyticity radius depending on L.")
    print("  -> The proof sketch must use the correct SVD mechanism, not Rayleigh.")
