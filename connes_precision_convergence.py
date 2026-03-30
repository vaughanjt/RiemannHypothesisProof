"""
Session 25d: Precision convergence test for eps_0.

Critical question: Is eps_0 a genuine eigenvalue or numerical noise?

Evidence so far:
  lam_sq=14:  eps stable at ~5.86e-50 across dps=50 and dps=80 -> GENUINE
  lam_sq=100: eps at ~10^{-dps} for both dps=50 and dps=80 -> NOISE FLOOR

This script tests at dps=50, 80, 120 for lam_sq=14, 50, 100, 200 to map
where the genuine/noise boundary lies.

If eps tracks 10^{-dps}: true eigenvalue is zero (or much smaller)
If eps stabilizes: genuine eigenvalue, measurable for Fourier bump test
"""

import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, sinh, hyp2f1, digamma, eig, quad, nstr, fabs)
import sympy
import time


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def compute_eps(lam_sq, N=30):
    """Compute the 3 smallest eigenvalues of tau at current dps."""
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
            wp = sum(lk * pk ** (-mpf(1) / 2) * q_mp(n, m, logk) for pk, lk, logk in vM)
            wr = wr_d[n] if n == m else (al[m] - al[n]) / (n - m)
            tau[i, j] = w02 - wr - wp
            tau[j, i] = tau[i, j]
    E = eig(tau, left=False, right=False)
    evals = sorted([E[i].real for i in range(dim)], key=lambda x: float(x))
    return evals[:5], float(L)


if __name__ == "__main__":
    print("PRECISION CONVERGENCE TEST")
    print("Does eps_0 stabilize (genuine) or track 10^{-dps} (noise)?")
    print("=" * 80)
    print()

    for lam_sq in [14, 30, 50, 100, 200]:
        print(f"lam^2 = {lam_sq}")
        print(f"  {'dps':>5} {'eps_0':>25} {'eps_1':>25} {'eps_2':>25} {'gap(0-1)':>20} {'time':>6}")
        print(f"  {'-'*5} {'-'*25} {'-'*25} {'-'*25} {'-'*20} {'-'*6}")

        for dps in [50, 80, 120]:
            mp.dps = dps
            t0 = time.time()
            evals, L = compute_eps(lam_sq, N=30)
            dt = time.time() - t0

            e0 = evals[0]
            e1 = evals[1]
            e2 = evals[2]
            gap = e1 - e0

            print(f"  {dps:>5} {nstr(e0, 15):>25} {nstr(e1, 15):>25} "
                  f"{nstr(e2, 15):>25} {nstr(gap, 12):>20} {dt:5.0f}s")

        # Verdict
        print()

    print()
    print("INTERPRETATION:")
    print("  If eps_0 is same at dps=50,80,120: GENUINE eigenvalue")
    print("  If eps_0 ~ 10^{-dps} at each step: true value is 0 (or << 10^{-120})")
    print("  The boundary between genuine/noise tells us where RH 'kicks in'")
    print("  for this truncation of the Weil distribution")
