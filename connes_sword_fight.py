"""
Session 29s: SWORD FIGHT — high-precision double limit.

dps=50, n_quad=20000. Resolve |xi_hat(gamma_k)| -> 0 or not.
This is the question that decides everything.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time

mp.dps = 50


def build_QW_hp(lam_sq, N_val, n_quad=20000):
    """Build Q_W at high precision."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    vM = []
    primes_list = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
                   101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,
                   197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293]
    for p in primes_list:
        if p > lam_sq: break
        lp_f = np.log(p)
        pk = p
        while pk <= lam_sq:
            vM.append((pk, lp_f, np.log(pk)))
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
            if abs(denom) > mpf(10)**(-40): integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const+integral)
        wr_diag[-nv] = wr_diag[nv]
        if nv % 10 == 0 and nv > 0:
            print(f"    WR_diag[{nv}] done", flush=True)

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


def xi_hat_hp(z, xi, N, L_f):
    """xi_hat at high precision using mpmath."""
    z_mp = mpf(z)
    L_mp = mpf(L_f)
    s = mpmath.sin(z_mp * L_mp / 2)
    total = mpf(0)
    for j in range(-N, N+1):
        dj = 2*pi*j/L_mp
        diff = z_mp - dj
        if abs(diff) > mpf(10)**(-20):
            total += mpf(xi[j+N]) / diff
    return float(abs(2 * L_mp**(-mpf(1)/2) * s * total))


if __name__ == "__main__":
    print("SWORD FIGHT: HIGH-PRECISION DOUBLE LIMIT")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")
    C_scale = 8

    targets = [
        (14, 21),
        (50, 31),
        (100, 37),
        (200, 42),
        (300, 46),
    ]

    print(f"\ndps=50, n_quad=20000, N = 8*L")
    print(f"\n{'lam^2':>6} {'N':>4} {'eps_0':>14} {'eps_1':>14} {'gap_r':>7} "
          f"{'pos':>4} {'|xh(g1)|':>12} {'|xh(g2)|':>12} {'|xh(g3)|':>12}")
    print("-" * 100)

    for lam_sq, N in targets:
        L_f = np.log(lam_sq)
        dim = 2*N + 1
        bandwidth = np.pi * N / L_f

        print(f"\n  Building lam^2={lam_sq}, N={N}, dim={dim}, BW={bandwidth:.1f}...", flush=True)
        t0 = time.time()

        try:
            QW = build_QW_hp(lam_sq, N, n_quad=20000)
            dt_build = time.time() - t0
            print(f"  Built in {dt_build:.0f}s", flush=True)

            evals, evecs = np.linalg.eigh(QW)
            xi = evecs[:, 0]
            eps_0 = evals[0]
            eps_1 = evals[1]
            eps_2 = evals[2]
            n_pos = np.sum(evals > -1e-12)
            gap_r = eps_1 / eps_0 if abs(eps_0) > 1e-20 else float('inf')

            # Normalize
            xs = np.sum(xi)
            if abs(xs) > 1e-30:
                xi_n = xi * np.sqrt(L_f) / xs
            else:
                xi_n = xi

            # xi_hat at first 3 zeta zeros (high precision)
            xh = []
            for k in range(min(3, len(gammas))):
                if gammas[k] < bandwidth:
                    xh.append(xi_hat_hp(gammas[k], xi_n, N, L_f))
                else:
                    xh.append(float('nan'))

            dt_total = time.time() - t0

            xh_strs = [f"{v:.6e}" if not np.isnan(v) else "---" for v in xh]
            while len(xh_strs) < 3:
                xh_strs.append("---")

            print(f"{lam_sq:>6} {N:>4} {eps_0:>14.6e} {eps_1:>14.6e} {gap_r:>7.2f} "
                  f"{n_pos:>4} {xh_strs[0]:>12} {xh_strs[1]:>12} {xh_strs[2]:>12}")

            # Extra diagnostics
            print(f"  eps_2={eps_2:.4e}, parity={'EVEN' if sum(abs(xi[N+k]-xi[N-k]) for k in range(1,N+1)) < sum(abs(xi[N+k]+xi[N-k]) for k in range(1,N+1)) else 'ODD'}")
            print(f"  |xi_hat/eps_0| ratios: {', '.join(f'{v/abs(eps_0):.1f}' for v in xh if not np.isnan(v) and abs(eps_0)>1e-20)}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print("=" * 70)
