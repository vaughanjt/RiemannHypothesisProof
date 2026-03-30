"""
Session 29l: Quick N(lambda) scaling test.
For each lam^2, find the minimum N such that Q_W is positive-definite.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, log, pi, euler, exp, cos, sin, hyp2f1, digamma, sinh
import time

mp.dps = 30  # lower precision for speed


def build_QW_fast(lam_sq, N_val):
    """Build Q_W with reduced precision for speed."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    vM = []
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p > lam_sq:
            break
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

    L2_f = L_f**2
    p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    alpha = {}
    for n in range(-N_val, N_val+1):
        if n == 0:
            alpha[n] = 0.0
        else:
            z = exp(-2*L)
            a = pi*mpmath.mpc(0, abs(n))/L + mpf(1)/4
            h = hyp2f1(1, a, a+1, z)
            f1 = exp(-L/2) * (2*L/(L + 4*pi*mpmath.mpc(0, abs(n))) * h).imag
            d = digamma(a).imag / 2
            val = float((f1 + d) / pi)
            alpha[n] = val if n > 0 else -val

    wr_diag = {}
    omega_0 = mpf(2)
    n_quad = 5000  # reduced for speed
    for nv in range(N_val+1):
        def omega(x):
            return 2*(1 - x/L)*cos(2*pi*nv*x/L)
        w_const = (omega_0/2)*(euler + log(4*pi*(eL-1)/(eL+1)))
        dx = L/n_quad
        integral = mpf(0)
        for k in range(n_quad):
            x = dx*(k + mpf(1)/2)
            numer = exp(x/2)*omega(x) - omega_0
            denom = exp(x) - exp(-x)
            if abs(denom) > mpf(10)**(-30):
                integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const + integral)
        wr_diag[-nv] = wr_diag[nv]

    QW = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(i, dim):
            m = j - N_val
            w02 = pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))
            wp = sum(lk * k**(-0.5) * q_func(n, m, logk) for k, lk, logk in vM)
            wr = wr_diag[n] if n == m else (alpha[m] - alpha[n]) / (n - m)
            QW[i, j] = w02 - wr - wp
            QW[j, i] = QW[i, j]
    QW = (QW + QW.T) / 2
    return QW


if __name__ == "__main__":
    print("N(lambda) SCALING FOR POSITIVE-DEFINITENESS")
    print("=" * 60)

    print(f"\n{'lam^2':>6} {'N':>4} {'dim':>5} {'eps_0':>14} {'eps_1':>14} {'gap_ratio':>10} {'time':>6}")
    print("-" * 65)

    for lam_sq in [14, 30, 50, 60, 70, 75, 80, 90, 100]:
        for N in [15, 20, 25, 30, 35, 40, 50]:
            t0 = time.time()
            try:
                QW = build_QW_fast(lam_sq, N)
                evals = np.linalg.eigvalsh(QW)
                eps_0 = evals[0]
                eps_1 = evals[1]
                dt = time.time() - t0

                if eps_0 > -1e-6:
                    gap = eps_1 - eps_0
                    gr = gap / abs(eps_0) if abs(eps_0) > 1e-20 else float('inf')
                    print(f"{lam_sq:>6} {N:>4} {2*N+1:>5} {eps_0:>14.4e} {eps_1:>14.4e} "
                          f"{gr:>10.1f} {dt:>5.0f}s")
                    break
                elif N == 50:
                    print(f"{lam_sq:>6} {'>50':>4} {'---':>5} {eps_0:>14.4e} {'---':>14} "
                          f"{'---':>10} {dt:>5.0f}s")
            except Exception as e:
                print(f"{lam_sq:>6} {N:>4} ERROR: {e}")
                break

    print(f"\nDone.")
