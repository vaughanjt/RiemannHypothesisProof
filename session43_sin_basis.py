"""
SESSION 43 — SIN-BASIS SQUARE ROOT: THE ODD DIRECTION

The cos-basis Mellin transforms vanish on the odd direction (parity).
Fix: use sin-basis for odd modes, cos-basis for even modes.

For the ODD vector w, the relevant transform is:
  G_n^{sin}(rho) = integral_0^L 2(1-x/L)*sin(2*pi*n*x/L) * x^{rho-1} dx

Then the spectral piece on w:
  <w, Q_spectral, w> = sum_rho |sum_n w[n] * G_n^{sin}(rho)|^2

This should be NONZERO and capture part of the barrier.

The remainder (corrections) should be smaller and potentially provable.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, exp, sin, cos, quad,
                    zetazero, power, re, im, sqrt, fabs)
import time
import sys
import os

mp.dps = 20

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all


def mellin_sin_basis(n_val, gamma, L):
    """G_n^{sin}(rho) = integral_0^L 2(1-x/L)*sin(2*pi*n*x/L) * x^{rho-1} dx"""
    n_val = abs(int(n_val))
    if n_val == 0:
        return 0.0 + 0.0j
    s = mpf(1)/2 + mpc(0, mpf(gamma))

    def integrand(x):
        return 2 * (1 - x/L) * sin(2 * pi * n_val * x / L) * power(x, s - 1)

    result = quad(integrand, [mpf(0), L], maxdegree=6)
    return complex(result)


def mellin_cos_basis(n_val, gamma, L):
    """G_n^{cos}(rho) = integral_0^L 2(1-x/L)*cos(2*pi*n*x/L) * x^{rho-1} dx"""
    n_val = abs(int(n_val))
    s = mpf(1)/2 + mpc(0, mpf(gamma))

    def integrand(x):
        return 2 * (1 - x/L) * cos(2 * pi * n_val * x / L) * power(x, s - 1)

    result = quad(integrand, [mpf(0), L], maxdegree=6)
    return complex(result)


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  SIN-BASIS SQUARE ROOT: CAPTURING THE ODD DIRECTION')
    print('#' * 72)

    lam_sq = 50
    L_mp = log(mpf(lam_sq))
    L_f = float(L_mp)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    K = 30

    # Build Q_W
    print(f'\n  lam^2={lam_sq}, N={N}, dim={dim}, K={K} zeros')
    print('\n  Building Q_W...', flush=True)
    W02, M, QW = build_all(lam_sq, N, n_quad=4000)
    ns = np.arange(-N, N + 1, dtype=float)

    # w_hat (odd) and u_hat (even)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    u = 1.0 / (L_f**2 + (4*np.pi)**2 * ns**2)
    u_hat = u / np.linalg.norm(u)

    barrier_w = float(w_hat @ QW @ w_hat)
    barrier_u = float(u_hat @ QW @ u_hat)
    print(f'  Barrier (odd): {barrier_w:.8f}')
    print(f'  Barrier (even): {barrier_u:.8f}')

    # Get zeros
    print('\n  Loading zeros...', flush=True)
    zeros = [float(zetazero(k).imag) for k in range(1, K + 1)]

    # Build H_w(rho) = sum_n w_hat[n] * G_n^{sin}(rho) for each rho
    # Since w is odd: w[-n] = -w[n], and sin is odd in n
    # H_w(rho) = sum_{n=1}^{N} 2*w_hat[n] * G_n^{sin}(rho)  [factor 2 from ±n pairing]
    print(f'\n  Computing H_w(rho) using SIN basis at {K} zeros...', flush=True)

    H_w = np.zeros(K, dtype=complex)
    H_u = np.zeros(K, dtype=complex)

    t0 = time.time()
    for k_idx, gamma in enumerate(zeros):
        # Odd direction: H_w = sum_{n=1}^N 2*w_hat[N+n] * G_n^{sin}
        hw = 0.0 + 0.0j
        hu = 0.0 + 0.0j
        for n in range(1, N + 1):
            g_sin = mellin_sin_basis(n, gamma, L_mp)
            g_cos = mellin_cos_basis(n, gamma, L_mp)

            # Odd: w_hat[N+n] = -w_hat[N-n], contribution = 2*w_hat[N+n]*G_sin
            hw += 2 * w_hat[N + n] * g_sin

            # Even: u_hat[N+n] = u_hat[N-n], contribution = 2*u_hat[N+n]*G_cos
            # Plus the n=0 term
            hu += 2 * u_hat[N + n] * g_cos

        # Add n=0 term for even direction
        g_cos_0 = mellin_cos_basis(0, gamma, L_mp)
        hu += u_hat[N] * g_cos_0

        H_w[k_idx] = hw
        H_u[k_idx] = hu

        if (k_idx + 1) % 5 == 0 or k_idx == 0:
            print(f'    zero {k_idx+1:>2d} (gamma={gamma:>7.3f}): '
                  f'|H_w|={abs(hw):.6f}  |H_u|={abs(hu):.6f}', flush=True)

    dt = time.time() - t0
    print(f'  Done in {dt:.0f}s')

    # Spectral barrier = sum |H(rho)|^2
    spectral_w = np.sum(np.abs(H_w)**2)
    spectral_u = np.sum(np.abs(H_u)**2)

    print(f'\n  SPECTRAL BARRIERS (sum of |H(rho)|^2):')
    print(f'    Odd (sin-basis):  {spectral_w:.8f}')
    print(f'    Even (cos-basis): {spectral_u:.8f}')

    print(f'\n  ACTUAL BARRIERS:')
    print(f'    Odd:  {barrier_w:.8f}')
    print(f'    Even: {barrier_u:.8f}')

    print(f'\n  CORRECTIONS (actual - spectral):')
    corr_w = barrier_w - spectral_w
    corr_u = barrier_u - spectral_u
    print(f'    Odd:  {corr_w:+.8f}')
    print(f'    Even: {corr_u:+.8f}')

    print(f'\n  DECOMPOSITION:')
    print(f'    Odd barrier  = {spectral_w:.6f} (spectral, >=0) + {corr_w:+.6f} (correction)')
    print(f'    Even barrier = {spectral_u:.6f} (spectral, >=0) + {corr_u:+.6f} (correction)')

    # Is spectral alone enough?
    print(f'\n  Is spectral piece alone sufficient?')
    print(f'    Odd:  spectral ({spectral_w:.6f}) > |correction| ({abs(corr_w):.6f})? '
          f'{"YES" if spectral_w > abs(corr_w) else "NO"}')
    print(f'    Even: spectral ({spectral_u:.6f}) > |correction| ({abs(corr_u):.6f})? '
          f'{"YES" if spectral_u > abs(corr_u) else "NO"}')

    # Convergence: how does spectral sum build up?
    print(f'\n  CONVERGENCE (cumulative |H_w|^2):')
    cum_w = np.cumsum(np.abs(H_w)**2)
    cum_u = np.cumsum(np.abs(H_u)**2)
    for k in [1, 5, 10, 20, 30]:
        if k <= K:
            print(f'    {k:>2d} zeros: odd={cum_w[k-1]:.6f}  even={cum_u[k-1]:.6f}')

    # Per-zero contributions
    print(f'\n  TOP 5 ZEROS by |H_w|^2 contribution:')
    top_w = np.argsort(np.abs(H_w)**2)[::-1][:5]
    for idx in top_w:
        pct = np.abs(H_w[idx])**2 / spectral_w * 100 if spectral_w > 0 else 0
        print(f'    gamma_{idx+1} = {zeros[idx]:>7.3f}: |H_w|^2 = {np.abs(H_w[idx])**2:.6f} ({pct:.1f}%)')

    # The big question
    print(f'\n' + '=' * 72)
    print(f'  THE BIG QUESTION')
    print(f'=' * 72)
    print(f'\n  If correction is POSITIVE:')
    print(f'    barrier = (sum of squares) + (positive correction) > 0')
    print(f'    PROOF COMPLETE on this direction.')
    print(f'\n  If correction is NEGATIVE but small:')
    print(f'    Need: spectral > |correction|')
    print(f'    Spectral is a sum of squares (provably >= 0)')
    print(f'    Correction is a finite analytic expression (no primes)')
    print(f'    If we can bound |correction| < spectral, DONE.')

    print(f'\n  ODD DIRECTION:  correction = {corr_w:+.6f}  '
          f'({"POSITIVE!" if corr_w > 0 else "negative"})')
    print(f'  EVEN DIRECTION: correction = {corr_u:+.6f}  '
          f'({"POSITIVE!" if corr_u > 0 else "negative"})')

    print('\n' + '#' * 72)
