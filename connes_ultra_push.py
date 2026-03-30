"""
Session 30: ULTRA PUSH — lam^2 up to 10000 at dps=120.

Also: compute sup|H-Xi| on a compact set (not just pointwise at zeros).
This is what the proof actually needs.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time

mp.dps = 120


def build_QW_ultra(lam_sq, N_val, n_quad=30000):
    """Build Q_W at ultra precision (dps=120)."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    # Sieve primes
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

    print(f"  alpha_L...", end="", flush=True)
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
    print(f" done.", flush=True)

    print(f"  WR_diag (n_quad={n_quad})...", flush=True)
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
            if abs(denom) > mpf(10)**(-100): integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const+integral)
        wr_diag[-nv] = wr_diag[nv]
        if nv > 0 and nv % 20 == 0:
            print(f"    n={nv}/{N_val}", flush=True)

    print(f"  assemble...", end="", flush=True)
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
    print(f" done.", flush=True)
    return QW


def xi_hat_val(z, xi, N, L_f):
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
    print("ULTRA PUSH: dps=120, lam^2 up to 10000")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # Targets: each gets N = round(8*L)
    targets = []
    for lam_sq in [14, 100, 500, 1000, 2000, 5000, 10000]:
        L = np.log(lam_sq)
        N = max(21, round(8 * L))
        targets.append((lam_sq, N))

    print(f"\n{'lam^2':>7} {'L':>6} {'N':>4} {'dim':>5} {'eps_0':>14} {'gap_r':>7} "
          f"{'pos':>5} {'|xh(g1)|':>12} {'|xh(g2)|':>12} {'time':>6}")
    print("-" * 95)

    for lam_sq, N in targets:
        L_f = np.log(lam_sq)
        dim = 2*N + 1
        bw = np.pi * N / L_f

        print(f"\n--- lam^2={lam_sq}, N={N}, dim={dim}, BW={bw:.1f} ---", flush=True)
        t0 = time.time()

        try:
            QW = build_QW_ultra(lam_sq, N, n_quad=30000)
            evals, evecs = np.linalg.eigh(QW)
            xi = evecs[:, 0]
            eps_0 = evals[0]
            eps_1 = evals[1]
            n_pos = np.sum(evals > -1e-14)
            gap_r = eps_1/eps_0 if abs(eps_0) > 1e-20 else float('inf')

            xs = np.sum(xi)
            xi_n = xi * np.sqrt(L_f)/xs if abs(xs) > 1e-30 else xi

            xh = []
            for k in range(min(2, len(gammas))):
                if gammas[k] < bw:
                    xh.append(xi_hat_val(gammas[k], xi_n, N, L_f))
                else:
                    xh.append(float('nan'))

            dt = time.time() - t0
            status = "ALL+" if n_pos == dim else f"{dim-n_pos}neg"
            xh_s = [f"{v:.4e}" if not np.isnan(v) else "---" for v in xh]
            while len(xh_s) < 2: xh_s.append("---")

            print(f"{lam_sq:>7} {L_f:>6.3f} {N:>4} {dim:>5} {eps_0:>14.6e} {gap_r:>7.2f} "
                  f"{status:>5} {xh_s[0]:>12} {xh_s[1]:>12} {dt:>5.0f}s")

        except Exception as e:
            dt = time.time() - t0
            print(f"{lam_sq:>7} ERROR: {str(e)[:60]} ({dt:.0f}s)")

    print(f"\n{'='*70}")
    print("ULTRA PUSH COMPLETE")
