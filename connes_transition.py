"""
Session 30c: Investigate the eps_0 transition at lam^2 ~ 3000-5000.

eps_0 drops 20x between lam^2=2000 and 5000. Gap ratio jumps 12x.
WHY? Is this a smooth acceleration or a phase transition?

Also: dps=200 run at lam^2=10000 to confirm positivity.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time


def build_QW_hp(lam_sq, N_val, dps_val=80, n_quad=30000):
    """Build Q_W at specified precision."""
    mp.dps = dps_val
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    vM = []
    limit = min(lam_sq, 10000)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5)+2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    for p in range(2, limit+1):
        if sieve[p] and p <= lam_sq:
            pk = p
            while pk <= lam_sq:
                vM.append((pk, np.log(p), np.log(pk)))
                pk *= p

    def q_func(n, m, y):
        if n != m:
            return (np.sin(2*np.pi*m*y/L_f) - np.sin(2*np.pi*n*y/L_f)) / (np.pi*(n-m))
        else:
            return 2*(L_f - y)/L_f * np.cos(2*np.pi*n*y/L_f)

    L2_f = L_f**2; p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    alpha = {}
    for n in range(-N_val, N_val+1):
        if n == 0:
            alpha[n] = 0.0
        else:
            z = exp(-2*L)
            a = pi*mpc(0,abs(n))/L + mpf(1)/4
            h = hyp2f1(1,a,a+1,z)
            f1 = exp(-L/2) * (2*L/(L+4*pi*mpc(0,abs(n)))*h).imag
            d = digamma(a).imag/2
            val = float((f1+d)/pi)
            alpha[n] = val if n>0 else -val

    wr_diag = {}
    omega_0 = mpf(2)
    for nv in range(N_val+1):
        def omega(x, nv=nv):
            return 2*(1-x/L)*cos(2*pi*nv*x/L)
        w_const = (omega_0/2)*(euler+log(4*pi*(eL-1)/(eL+1)))
        dx = L/n_quad; integral = mpf(0)
        for k in range(n_quad):
            x = dx*(k+mpf(1)/2)
            numer = exp(x/2)*omega(x)-omega_0
            denom = exp(x)-exp(-x)
            if abs(denom) > mpf(10)**(-dps_val+10): integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const+integral)
        wr_diag[-nv] = wr_diag[nv]

    QW = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(i, dim):
            m = j - N_val
            w02 = pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))
            wp = sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
            wr = wr_diag[n] if n==m else (alpha[m]-alpha[n])/(n-m)
            QW[i,j] = w02 - wr - wp
            QW[j,i] = QW[i,j]
    QW = (QW + QW.T)/2
    return QW


if __name__ == "__main__":
    print("TRANSITION ANALYSIS + dps=200 PUSH")
    print("=" * 70)

    # ================================================================
    # PART 1: Fine scan of the transition region lam^2 = 1000..5000
    # ================================================================
    print("\nPART 1: FINE SCAN lam^2 = 1000..5000 (dps=120)")
    print("-" * 70)

    print(f"{'lam^2':>7} {'N':>4} {'eps_0':>14} {'eps_1':>14} {'gap_r':>8} {'pos':>5} {'time':>5}")
    print("-" * 65)

    for lam_sq in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N + 1
        t0 = time.time()

        QW = build_QW_hp(lam_sq, N, dps_val=120, n_quad=30000)
        evals = np.linalg.eigvalsh(QW)
        eps_0 = evals[0]
        eps_1 = evals[1]
        n_pos = np.sum(evals > -1e-14)
        gap_r = eps_1/eps_0 if abs(eps_0) > 1e-20 else float('inf')
        status = "ALL+" if n_pos == dim else f"{dim-n_pos}neg"
        dt = time.time() - t0

        print(f"{lam_sq:>7} {N:>4} {eps_0:>14.6e} {eps_1:>14.6e} {gap_r:>8.1f} {status:>5} {dt:>4.0f}s")

    # ================================================================
    # PART 2: dps=200 at lam^2=10000
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: dps=200 AT lam^2=10000")
    print("-" * 70)

    lam_sq = 10000
    L_f = np.log(lam_sq)
    N = round(8 * L_f)
    dim = 2*N + 1

    print(f"lam^2={lam_sq}, N={N}, dim={dim}, dps=200, n_quad=40000", flush=True)
    t0 = time.time()
    QW = build_QW_hp(lam_sq, N, dps_val=200, n_quad=40000)
    evals = np.linalg.eigvalsh(QW)
    eps_0 = evals[0]
    eps_1 = evals[1]
    n_pos = np.sum(evals > -1e-14)
    gap_r = eps_1/eps_0 if abs(eps_0) > 1e-20 else float('inf')
    status = "ALL+" if n_pos == dim else f"{dim-n_pos}neg"
    dt = time.time() - t0

    print(f"\n  eps_0 = {eps_0:.6e}")
    print(f"  eps_1 = {eps_1:.6e}")
    print(f"  gap ratio = {gap_r:.1f}")
    print(f"  positive: {n_pos}/{dim} ({status})")
    print(f"  time: {dt:.0f}s")

    print(f"\n{'='*70}")
    print("COMPLETE")
