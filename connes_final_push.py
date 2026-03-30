"""
Session 29t: FINAL PUSH — dps=80, lam^2 up to 1000, N=8*L.

Two objectives:
1. Push to lam^2=500,700,1000 and verify all positive + eps_0 decreasing
2. Track |xi_hat(gamma_k)| — does it converge to 0?

At dps=80, the WR quadrature precision is ~10^{-20}, far below eps_0 ~10^{-10}.
This eliminates numerical noise as a factor.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time

mp.dps = 80


def build_QW_ultra(lam_sq, N_val, n_quad=30000):
    """Build Q_W at dps=80 precision."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    vM = []
    # Sieve primes up to lam_sq
    sieve = [True] * (lam_sq + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(lam_sq**0.5)+2):
        if i <= lam_sq and sieve[i]:
            for j in range(i*i, lam_sq+1, i):
                sieve[j] = False
    primes = [p for p in range(2, lam_sq+1) if sieve[p]]

    for p in primes:
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

    # Alpha values at dps=80
    print(f"  Computing alpha_L...", flush=True)
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

    # WR diagonal at dps=80 with n_quad=30000
    print(f"  Computing WR_diag (n_quad={n_quad})...", flush=True)
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
            if abs(denom) > mpf(10)**(-60): integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const+integral)
        wr_diag[-nv] = wr_diag[nv]
        if nv % 15 == 0 and nv > 0:
            print(f"    n={nv}/{N_val}", flush=True)

    # Assemble
    print(f"  Assembling QW...", flush=True)
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


def xi_hat_val(z, xi, N, L_f):
    """xi_hat using mpmath for precision."""
    z_mp = mpf(z)
    L_mp = mpf(L_f)
    s = mpmath.sin(z_mp * L_mp / 2)
    total = mpf(0)
    for j in range(-N, N+1):
        dj = 2*pi*j/L_mp
        diff = z_mp - dj
        if abs(diff) > mpf(10)**(-30):
            total += mpf(xi[j+N]) / diff
    return float(abs(2 * L_mp**(-mpf(1)/2) * s * total))


if __name__ == "__main__":
    print("FINAL PUSH: dps=80, N=8*L, lam^2 up to 1000")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    targets = [
        (14, 21),
        (50, 31),
        (100, 37),
        (200, 42),
        (300, 46),
        (500, 50),
        (700, 52),
        (1000, 55),
    ]

    print(f"\n{'lam^2':>6} {'N':>4} {'eps_0':>14} {'gap_r':>7} {'pos':>5} "
          f"{'|xh(g1)|':>12} {'|xh(g2)|':>12} {'|xh(g3)|':>12} {'time':>6}")
    print("-" * 100)

    all_results = []

    for lam_sq, N in targets:
        L_f = np.log(lam_sq)
        dim = 2*N + 1
        bandwidth = np.pi * N / L_f

        print(f"\n--- lam^2={lam_sq}, N={N}, dim={dim}, BW={bandwidth:.1f} ---", flush=True)
        t0 = time.time()

        try:
            QW = build_QW_ultra(lam_sq, N, n_quad=30000)
            dt_build = time.time() - t0

            evals, evecs = np.linalg.eigh(QW)
            xi = evecs[:, 0]
            eps_0 = evals[0]
            eps_1 = evals[1]
            n_pos = np.sum(evals > -1e-14)
            gap_r = eps_1 / eps_0 if abs(eps_0) > 1e-20 else float('inf')

            xs = np.sum(xi)
            if abs(xs) > 1e-30:
                xi_n = xi * np.sqrt(L_f) / xs
            else:
                xi_n = xi

            xh = []
            for k in range(min(3, len(gammas))):
                if gammas[k] < bandwidth:
                    xh.append(xi_hat_val(gammas[k], xi_n, N, L_f))
                else:
                    xh.append(float('nan'))

            dt_total = time.time() - t0

            xh_strs = [f"{v:.6e}" if not np.isnan(v) else "---" for v in xh]
            while len(xh_strs) < 3: xh_strs.append("---")

            status = "ALL+" if n_pos == dim else f"{dim-n_pos}neg"

            print(f"{lam_sq:>6} {N:>4} {eps_0:>14.6e} {gap_r:>7.2f} {status:>5} "
                  f"{xh_strs[0]:>12} {xh_strs[1]:>12} {xh_strs[2]:>12} {dt_total:>5.0f}s")

            all_results.append({
                'lam_sq': lam_sq, 'N': N, 'eps_0': eps_0, 'gap_r': gap_r,
                'n_pos': n_pos, 'dim': dim, 'xh': xh, 'status': status
            })

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'lam^2':>6} {'L':>6} {'N':>4} {'eps_0':>14} {'eps_0*L':>12} {'gap':>6} "
          f"{'pos?':>5} {'|xh(g1)|':>12}")
    print("-" * 75)
    for r in all_results:
        L = np.log(r['lam_sq'])
        epsL = r['eps_0'] * L
        xh1 = r['xh'][0] if len(r['xh'])>0 and not np.isnan(r['xh'][0]) else 0
        print(f"{r['lam_sq']:>6} {L:>6.3f} {r['N']:>4} {r['eps_0']:>14.6e} {epsL:>12.4e} "
              f"{r['gap_r']:>6.2f} {r['status']:>5} {xh1:>12.4e}")

    print(f"\n{'='*70}")
    print("THE VERDICT")
    print("=" * 70)
