"""
Session 25e: Reliable Fourier bump test at dps=120.

Only test lam_sq values where eps_0 is CONFIRMED genuine (stable across precisions):
  lam_sq=14: eps=5.859e-50, gap=1.518e-46, eigvec ~73 reliable digits
  lam_sq=30: eps=1.578e-66, gap=3.506e-63, eigvec ~57 reliable digits
  lam_sq=50: eps=3.687e-74, gap=9.093e-71, eigvec ~49 reliable digits

KEY QUESTION: Does |F_T[xi](gamma_k)| decrease with lambda?
If F_T ~ O(1) for all lambda: gap is intrinsic
If F_T decreases: rate tells us if proof path exists
"""

import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, sinh, hyp2f1, digamma, eig, quad, sqrt, nstr, fabs)
import sympy
import time

ZETA_ZEROS = [
    "14.134725141734693790457251983562",
    "21.022039638771554992628479593897",
    "25.010857580145688763213790992563",
    "30.424876125859513210311897530584",
    "32.935061587739189690662368964075",
]


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def build_and_test(lam_sq, N=30):
    """Build xi at full precision, return eps, eigvec, and bump results."""
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
    E, ER = eig(tau, left=False, right=True)
    evals_sorted = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    eps = evals_sorted[0][0]; idx = evals_sorted[0][1]
    eps_1 = evals_sorted[1][0]
    gap = eps_1 - eps

    xi_mp = [ER[j, idx].real for j in range(dim)]
    xs = sum(xi_mp)
    sqL = sqrt(L)
    if fabs(xs) > mpf(10) ** (-20):
        xi_mp = [x * sqL / xs for x in xi_mp]

    # Verify eigenvector: residual ||tau*xi - eps*xi|| / ||xi||
    tau_xi = [sum(tau[i, j] * xi_mp[j] for j in range(dim)) for i in range(dim)]
    resid = sqrt(sum((tau_xi[i] - eps * xi_mp[i])**2 for i in range(dim)))
    xi_norm = sqrt(sum(x**2 for x in xi_mp))
    rel_resid = resid / (fabs(eps) * xi_norm)

    # Fourier bump tests
    results = []
    max_freq = float(2 * pi * N / L)

    for gamma_str in ZETA_ZEROS:
        gamma_k = mpf(gamma_str)
        gamma_f = float(gamma_str)
        in_band = gamma_f * float(L) / (2 * 3.14159265) < N - 1

        center = gamma_k * L / (2 * pi)

        # F_T = L * sum xi_n sinc(n - center)
        FT = mpf(0)
        for j in range(dim):
            n = j - N
            x = mpf(n) - center
            if fabs(x) < mpf(10) ** (-(mp.dps - 10)):
                s = mpf(1)
            else:
                s = sin(pi * x) / (pi * x)
            FT += xi_mp[j] * s
        FT *= L

        # Mellin M[xi](1/2+i*gamma_k)
        M_val = mpc(0, 0)
        for j in range(dim):
            n = j - N
            alpha = mpc(-mpf(1) / 2, 2 * pi * n / L + gamma_k)
            M_val += xi_mp[j] * 2 * sinh(alpha * L / 2) / alpha

        results.append({
            'gamma': gamma_f, 'in_band': in_band,
            'FT': FT, 'M': M_val,
            'abs_FT': fabs(FT), 'abs_M': fabs(M_val)
        })

    return eps, eps_1, gap, L, xi_norm, rel_resid, results


if __name__ == "__main__":
    DPS = 120
    mp.dps = DPS

    print(f"RELIABLE FOURIER BUMP TEST (dps={DPS})")
    print(f"Only confirmed-genuine eigenvalues")
    print("=" * 90)
    print()

    summary = []

    for lam_sq in [14, 30, 50]:
        t0 = time.time()
        print(f"lam^2 = {lam_sq}  (building at dps={DPS})...", end="", flush=True)
        eps, eps_1, gap, L, xi_norm, rel_resid, results = build_and_test(lam_sq, N=30)
        dt = time.time() - t0
        print(f" ({dt:.0f}s)", flush=True)

        print(f"  eps_0     = {nstr(eps, 20)}")
        print(f"  eps_1     = {nstr(eps_1, 20)}")
        print(f"  gap       = {nstr(gap, 12)}")
        print(f"  ||xi||    = {nstr(xi_norm, 12)}")
        print(f"  residual  = {nstr(rel_resid, 6)} (relative)")
        eigvec_digits = int(-float(mpmath.log10(rel_resid))) if rel_resid > 0 else DPS
        print(f"  eigvec reliable digits: ~{eigvec_digits}")
        print()

        print(f"  {'gamma_k':>10} {'band':>5} {'|F_T|':>25} {'|M|':>25} {'F_T/eps':>18} {'log10|F_T|':>12}")
        print(f"  {'-'*10} {'-'*5} {'-'*25} {'-'*25} {'-'*18} {'-'*12}")

        for r in results:
            ratio = r['abs_FT'] / fabs(eps) if fabs(eps) > 0 else mpf('inf')
            log_FT = float(mpmath.log10(r['abs_FT'])) if r['abs_FT'] > 0 else float('-inf')
            band = "YES" if r['in_band'] else "no"
            print(f"  {r['gamma']:10.4f} {band:>5} {nstr(r['abs_FT'], 15):>25} "
                  f"{nstr(r['abs_M'], 15):>25} {nstr(ratio, 10):>18} {log_FT:>12.2f}")

            if r['gamma'] < 15:  # gamma_1
                summary.append({
                    'lam_sq': lam_sq, 'L': float(L),
                    'eps': float(eps), 'log_eps': float(mpmath.log10(fabs(eps))),
                    'FT': float(r['abs_FT']), 'log_FT': log_FT,
                    'ratio': float(ratio)
                })

        print()

    # Summary: trend of F_T vs eps
    print("=" * 90)
    print("TREND at gamma_1 = 14.1347:")
    print("-" * 90)
    print(f"{'lam^2':>8} {'L':>8} {'log10|eps|':>12} {'log10|F_T|':>12} {'log10(F_T/eps)':>16} {'verdict':>10}")
    print(f"{'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*16} {'-'*10}")
    for s in summary:
        log_ratio = s['log_FT'] - s['log_eps']
        print(f"{s['lam_sq']:>8} {s['L']:>8.3f} {s['log_eps']:>12.1f} {s['log_FT']:>12.2f} {log_ratio:>16.1f}")
    print()
    print("If log10(F_T/eps) grows with L -> gap is intrinsic, F_T does NOT scale with eps")
    print("If log10(F_T/eps) is constant -> F_T ~ C*eps, proof path open")
    print("If log10|F_T| decreases (even if not as fast as eps) -> partial control")
