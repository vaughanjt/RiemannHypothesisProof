"""
SESSION 42a — M_diag + M_alpha AT LARGE LAMBDA

Compute M_diag and M_alpha Rayleigh quotients at lam^2 = 20000, 50000, 100000.
These do NOT depend on the prime sieve — only on the Weil explicit formula
integral (wr_diag) and alpha coefficients (hypergeometric).

Uses reduced precision (dps=30) for speed where possible.
Uses the analytic w vector (no eigendecomposition needed).

Priority 1 from Session 41: determine if M_diag+M_alpha converges below 3.01.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time
import sys

mp.dps = 30  # reduced from 50 for speed


def compute_mdiag_malpha_rayleigh(lam_sq, N=None, n_quad=6000):
    """
    Compute <w_hat, M_diag, w_hat> and <w_hat, M_alpha, w_hat>.
    Returns (mdiag_rq, malpha_rq, wr_diag_values, alpha_values).
    """
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    if N is None:
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    # Build w_hat
    ns = np.arange(-N, N + 1, dtype=float)
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N] = 0.0
    w_norm = np.linalg.norm(w_vec)
    w_hat = w_vec / w_norm

    # ── Compute alpha coefficients ──
    alpha = {}
    for n in range(-N, N + 1):
        if n == 0:
            alpha[n] = 0.0
        else:
            z = exp(-2 * L)
            a = pi * mpc(0, abs(n)) / L + mpf(1) / 4
            h = hyp2f1(1, a, a + 1, z)
            f1 = exp(-L / 2) * (2 * L / (L + 4 * pi * mpc(0, abs(n))) * h).imag
            d = digamma(a).imag / 2
            val = float((f1 + d) / pi)
            alpha[n] = val if n > 0 else -val

    # ── Compute wr_diag ──
    omega_0 = mpf(2)
    wr_diag = {}
    for nv in range(N + 1):
        def omega(x, nv=nv):
            return 2 * (1 - x / L) * cos(2 * pi * nv * x / L)
        w_const = (omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1)))
        dx = L / n_quad
        integral = mpf(0)
        for k in range(n_quad):
            x = dx * (k + mpf(1) / 2)
            numer = exp(x / 2) * omega(x) - omega_0
            denom = exp(x) - exp(-x)
            if abs(denom) > mpf(10)**(-30):
                integral += numer / denom
        integral *= dx
        wr_diag[nv] = float(w_const + integral)
        wr_diag[-nv] = wr_diag[nv]

    # ── Build M_diag and M_alpha matrices (as Rayleigh quotients) ──
    # M_diag: diagonal, so <w, M_diag, w> = sum |w[n]|^2 * wr_diag[n]
    diag_vals = np.array([wr_diag[int(n)] for n in ns])
    mdiag_rq = float(np.sum(w_hat**2 * diag_vals))

    # M_alpha: off-diagonal (alpha[m]-alpha[n])/(n-m)
    # <w, M_alpha, w> = sum_{n!=m} w[n] * (alpha[m]-alpha[n])/(n-m) * w[m]
    alpha_arr = np.array([alpha[int(n)] for n in ns])
    malpha_rq = 0.0
    for i in range(dim):
        for j in range(dim):
            if i != j:
                n, m = int(ns[i]), int(ns[j])
                malpha_rq += w_hat[i] * (alpha[m] - alpha[n]) / (n - m) * w_hat[j]
    malpha_rq = float(malpha_rq)

    return {
        'lam_sq': lam_sq,
        'L': L_f,
        'N': N,
        'mdiag': mdiag_rq,
        'malpha': malpha_rq,
        'sum': mdiag_rq + malpha_rq,
        'wr_diag_0': wr_diag[0],
        'wr_diag_1': wr_diag[1],
        'alpha_1': alpha[1],
    }


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42a -- M_diag + M_alpha AT LARGE LAMBDA')
    print('#' * 70)

    # Verify against known values first
    print('\n  Verification against Session 41 data:')
    print('  ' + '=' * 50)

    for lam_sq in [1000, 5000, 10000]:
        t0 = time.time()
        r = compute_mdiag_malpha_rayleigh(lam_sq)
        dt = time.time() - t0
        print(f'  lam^2={lam_sq:>6d}  M_diag={r["mdiag"]:+.6f}  '
              f'M_alpha={r["malpha"]:+.6f}  sum={r["sum"]:+.6f}  ({dt:.0f}s)')

    # Push to large lambda
    print('\n\n  Large lambda computation:')
    print('  ' + '=' * 50)

    results = []
    for lam_sq in [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        t0 = time.time()
        r = compute_mdiag_malpha_rayleigh(lam_sq)
        dt = time.time() - t0
        results.append(r)
        print(f'  lam^2={lam_sq:>6d}  L={r["L"]:.3f}  N={r["N"]:>3d}  '
              f'M_diag={r["mdiag"]:+.6f}  M_alpha={r["malpha"]:+.6f}  '
              f'sum={r["sum"]:+.6f}  wr[0]={r["wr_diag_0"]:.4f}  ({dt:.0f}s)')
        sys.stdout.flush()

    # Growth analysis
    print('\n\n  Growth analysis:')
    print('  ' + '=' * 50)

    Ls = np.array([r['L'] for r in results])
    sums = np.array([r['sum'] for r in results])
    mdiags = np.array([r['mdiag'] for r in results])
    malphas = np.array([r['malpha'] for r in results])

    # Rate of change
    for i in range(1, len(results)):
        dL = Ls[i] - Ls[i-1]
        ds = sums[i] - sums[i-1]
        rate = ds / dL if dL > 0 else 0
        print(f'  L={Ls[i]:.2f}: delta_sum={ds:+.6f}  rate={rate:.4f}/L')

    # Fit: sum = a + b/L (convergence to a)
    if len(Ls) >= 4:
        X = np.column_stack([np.ones_like(Ls[-4:]), 1/Ls[-4:]])
        c = np.linalg.lstsq(X, sums[-4:], rcond=None)[0]
        print(f'\n  Fit (last 4): sum = {c[0]:.6f} + {c[1]:.4f}/L')
        print(f'  Asymptotic limit: {c[0]:.6f}')

        # Compare to W02-Mp limit of 3.007
        print(f'\n  W02-Mp limit:       ~3.007')
        print(f'  M_diag+M_alpha limit: {c[0]:.4f}')
        print(f'  Predicted barrier:  {3.007 - c[0]:.4f}')

    print('\n' + '#' * 70)
    print('  SESSION 42a COMPLETE')
    print('#' * 70)
