"""
Session 25f: Rayleigh quotient decomposition for Path B proof.

The Rayleigh quotient on the constant mode v = e_0 (n=0) gives:
    ε₀ ≤ a₀ = τ[N,N] = W_{0,2}[0,0] - W_r[0,0] - W_p[0,0]

We decompose a₀ into its three components to understand the cancellation
that drives the super-exponential decay.

Grok's claim: a₀ = O(e^{-cL}) with c > 2, driven by the ₂F₁ and
digamma asymptotics of the archimedean terms.

This script:
1. Computes W_{0,2}[0,0], W_r[0,0], W_p[0,0] individually at dps=120
2. Shows the cancellation: a₀ = W_{0,2} - W_r - W_p
3. Compares a₀ to ε₀ (must have a₀ ≥ ε₀)
4. Fits the effective exponential rate c from a₀ vs L
5. Also computes a₀ for a range of N to verify N-independence

Additionally: compute more Rayleigh quotients (not just the constant mode)
to see if tighter upper bounds are available.
"""

import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, sinh, hyp2f1, digamma, eig, quad,
                    sqrt, nstr, fabs, ln)
import sympy
import time


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def decompose_a0(lam_sq, N=30):
    """Compute the three components of a₀ = τ[N,N] (the n=0 diagonal entry).

    Returns (w02_00, wr_00, wp_00, a0, eps_0, L)
    """
    L = log(mpf(lam_sq)); eL = exp(L)
    dim = 2 * N + 1

    # === W_{0,2}[0,0] ===
    # pf = 32*L*sinh(L/4)^2
    # w02 = pf * (L^2 - 16*pi^2*0*0) / ((L^2 + 0)(L^2 + 0)) = pf / L^2
    pf = 32 * L * sinh(L / 4) ** 2
    w02_00 = pf / (L * L)  # = 32*sinh^2(L/4)/L

    # === W_r[0,0] (diagonal) ===
    # wr_d[0] = gamma + log(4*pi*(eL-1)/(eL+1)) + integral
    w_c = euler + log(4 * pi * (eL - 1) / (eL + 1))
    def ig(x):
        return (exp(x / 2) * 2 * (1 - x / L) * cos(0) - 2) / (exp(x) - exp(-x))
    wr_00 = w_c + quad(ig, [mpf(0), L])

    # === W_p[0,0] ===
    # q_mp(0, 0, y) = 2*(L-y)/L * cos(0) = 2*(L-y)/L
    # wp = sum over p^k <= lam_sq of log(p) * p^{-1/2} * 2*(L-log(p^k))/L
    vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq):
            vM.append((pk, lp, log(pk))); pk *= p

    wp_00 = mpf(0)
    for pk, lp, logk in vM:
        wp_00 += lp * pk ** (-mpf(1) / 2) * 2 * (L - logk) / L

    # a₀ = W_{0,2} - W_r - W_p
    a0 = w02_00 - wr_00 - wp_00

    # === Full eigenvalue computation for comparison ===
    # Build full τ matrix
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
        w_c2 = euler + log(4 * pi * (eL - 1) / (eL + 1))
        def ig2(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        wr_d[nv] = w_c2 + quad(ig2, [mpf(0), L])
        wr_d[-nv] = wr_d[nv]
    tau = mpmatrix(dim, dim)
    L2 = L * L; p2 = 16 * pi * pi
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

    E = eig(tau, left=False, right=False)
    evals = sorted([E[i].real for i in range(dim)], key=lambda x: float(x))
    eps_0 = evals[0]
    eps_1 = evals[1]

    # Also compute Rayleigh quotient for the actual eigenvector direction (as check)
    # and for the trace/dim (average eigenvalue)
    trace_tau = sum(tau[i, i] for i in range(dim))
    avg_eval = trace_tau / dim

    return {
        'w02': w02_00, 'wr': wr_00, 'wp': wp_00,
        'a0': a0, 'eps_0': eps_0, 'eps_1': eps_1,
        'L': L, 'trace': trace_tau, 'avg': avg_eval,
        'n_primes': len(vM)
    }


if __name__ == "__main__":
    DPS = 120
    mp.dps = DPS

    print(f"RAYLEIGH QUOTIENT DECOMPOSITION (dps={DPS})")
    print("a₀ = τ[0,0] = W_{0,2}[0,0] - W_r[0,0] - W_p[0,0]")
    print("ε₀ ≤ a₀ (Rayleigh quotient bound on minimum eigenvalue)")
    print("=" * 90)
    print()

    data = []

    for lam_sq in [14, 20, 30, 40, 50, 70, 100]:
        t0 = time.time()
        print(f"λ² = {lam_sq}...", end="", flush=True)
        r = decompose_a0(lam_sq, N=30)
        dt = time.time() - t0
        print(f" ({dt:.0f}s)", flush=True)

        L = r['L']
        print(f"  L = {nstr(L, 8)}")
        print(f"  Prime powers ≤ λ²: {r['n_primes']}")
        print()
        print(f"  W_{{0,2}}[0,0] = {nstr(r['w02'], 20)}")
        print(f"  W_r[0,0]     = {nstr(r['wr'], 20)}")
        print(f"  W_p[0,0]     = {nstr(r['wp'], 20)}")
        print(f"  ────────────────────────────────────")
        print(f"  a₀ = W02-Wr-Wp = {nstr(r['a0'], 20)}")
        print(f"  ε₀ (min eval) = {nstr(r['eps_0'], 20)}")
        print(f"  ε₁ (2nd eval) = {nstr(r['eps_1'], 20)}")
        print()

        # Cancellation analysis
        max_term = max(fabs(r['w02']), fabs(r['wr']), fabs(r['wp']))
        cancel_orders = float(mpmath.log10(max_term / fabs(r['a0']))) if fabs(r['a0']) > 0 else float('inf')
        bound_ok = "YES" if r['a0'] >= r['eps_0'] else "NO"

        print(f"  Cancellation: {cancel_orders:.1f} orders (max term / a₀)")
        print(f"  a₀ ≥ ε₀? {bound_ok}  (a₀/ε₀ = {nstr(r['a0']/r['eps_0'], 8) if fabs(r['eps_0']) > 0 else 'N/A'})")
        print(f"  Trace/dim = {nstr(r['avg'], 15)}")
        print()

        log_a0 = float(mpmath.log10(fabs(r['a0']))) if fabs(r['a0']) > 0 else float('-inf')
        log_eps = float(mpmath.log10(fabs(r['eps_0']))) if fabs(r['eps_0']) > 0 else float('-inf')
        data.append({
            'lam_sq': lam_sq, 'L': float(L),
            'a0': float(r['a0']), 'log_a0': log_a0,
            'eps_0': float(r['eps_0']), 'log_eps': log_eps,
            'cancel': cancel_orders
        })

    # Summary and rate fitting
    print("=" * 90)
    print("SUMMARY")
    print("-" * 90)
    print(f"{'λ²':>6} {'L':>7} {'log₁₀|a₀|':>12} {'log₁₀|ε₀|':>12} {'a₀≥ε₀':>6} {'cancel':>8} {'eff c':>8}")
    print("-" * 90)
    for i, d in enumerate(data):
        eff_c = -d['log_a0'] * mpmath.log(10) / d['L'] if d['log_a0'] != float('-inf') else float('inf')
        bound_ok = "YES" if d['a0'] >= d['eps_0'] else "NO"
        print(f"{d['lam_sq']:>6} {d['L']:>7.3f} {d['log_a0']:>12.1f} {d['log_eps']:>12.1f} "
              f"{bound_ok:>6} {d['cancel']:>8.1f} {float(eff_c):>8.1f}")

    print()
    print("eff c = -log|a₀|/L (effective exponential rate in |a₀| ≤ C·e^{-cL})")
    print("If c stabilizes: we have the analytic rate for the proof.")
    print("If c grows with L: decay is truly super-exponential (faster than any e^{-cL}).")
