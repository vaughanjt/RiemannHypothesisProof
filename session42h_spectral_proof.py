"""
SESSION 42h — SPECTRAL BARRIER: THE PROOF ENGINE

Compute barrier = sum_rho |G_w(rho)|^2 using mpmath adaptive quadrature.

The test function for odd vector w_hat is:
    g_w(x) = 2 * sum_{n=1}^{N} w_hat[n] * sin(2*pi*n*x/L)

Its Mellin transform at the zero rho = 1/2 + i*gamma:
    G_w(s) = integral_0^L g_w(x) * x^{s-1} dx
    G_w(1/2 + i*gamma) = integral_0^L g_w(x) * x^{-1/2 + i*gamma} dx

The Weil explicit formula (assuming RH) gives:
    <w, QW, w> = sum_rho |G_w(rho)|^2 + [analytic corrections]

We compute G_w(rho) for each zero and sum |G_w|^2 to verify.

Uses mpmath quad for proper handling of:
- x^{-1/2} singularity at x=0
- x^{i*gamma} oscillation (period 2*pi/gamma in log x)
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, exp, sin, cos, quad,
                    zetazero, gamma as mpgamma, loggamma, power, sqrt,
                    fabs, im, re, conj, nstr)
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mp.dps = 25


# ═══════════════════════════════════════════════════════════════
# TEST FUNCTION AND ITS MELLIN TRANSFORM
# ═══════════════════════════════════════════════════════════════

def build_w_hat_mp(lam_sq, N=None):
    """Build w_hat in mpmath precision."""
    L = log(mpf(lam_sq))
    L_f = float(L)
    if N is None:
        N = max(15, round(6 * L_f))

    # w[n] = n / (L^2 + 16*pi^2*n^2), odd, w[0]=0
    coeffs = []
    norm_sq = mpf(0)
    for n in range(-N, N + 1):
        val = mpf(n) / (L**2 + 16 * pi**2 * mpf(n)**2)
        coeffs.append(val)
        norm_sq += val**2

    norm = sqrt(norm_sq)
    w_hat = [c / norm for c in coeffs]
    return w_hat, L, N


def mellin_transform_gw(w_hat_positive, L, gamma_val):
    """
    Compute G_w(1/2 + i*gamma) = integral_0^L g_w(x) * x^{-1/2+i*gamma} dx

    where g_w(x) = 2 * sum_{n=1}^{N} w_hat[n] * sin(2*pi*n*x/L)

    We compute each n's integral separately and sum:
    G_w = sum_n 2*w_hat[n] * I_n(gamma)
    where I_n(gamma) = integral_0^L sin(2*pi*n*x/L) * x^{-1/2+i*gamma} dx

    Substitution u = x/L:
    I_n = L^{1/2+i*gamma} * integral_0^1 sin(2*pi*n*u) * u^{-1/2+i*gamma} du
    """
    s = mpf(1)/2 + mpc(0, mpf(gamma_val))
    L_s = power(L, s)  # L^s = L^{1/2+i*gamma}

    G = mpc(0, 0)
    N = len(w_hat_positive)

    for n_idx in range(N):
        n = n_idx + 1  # n = 1, 2, ..., N
        wn = w_hat_positive[n_idx]
        if fabs(wn) < mpf(10)**(-20):
            continue

        # I_n = L^s * integral_0^1 sin(2*pi*n*u) * u^{s-1} du
        freq = 2 * pi * n

        def integrand_real(u):
            return sin(freq * u) * power(u, re(s) - 1) * cos(im(s) * log(u))

        def integrand_imag(u):
            return sin(freq * u) * power(u, re(s) - 1) * sin(im(s) * log(u))

        # mpmath quad handles the u^{-1/2} singularity and oscillation
        I_real = quad(integrand_real, [mpf(0), mpf(1)], error=True, maxdegree=8)[0]
        I_imag = quad(integrand_imag, [mpf(0), mpf(1)], error=True, maxdegree=8)[0]

        I_n = L_s * mpc(I_real, I_imag)
        G += 2 * wn * I_n

    return G


def matrix_barrier(lam_sq, N=None):
    """Compute barrier from matrix for comparison."""
    from connes_crossterm import build_all
    L_f = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L_f))

    W02, M, QW = build_all(lam_sq, N)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    return float(w_hat @ QW @ w_hat)


# ═══════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42h -- SPECTRAL BARRIER: THE PROOF ENGINE')
    print('#' * 70)

    # Load zeros
    print('\n  Loading zeta zeros...', flush=True)
    t0 = time.time()
    n_zeros = 100
    zeros = []
    for k in range(1, n_zeros + 1):
        z = zetazero(k)
        zeros.append(float(z.imag))
    zeros = np.array(zeros)
    print(f'  Loaded {n_zeros} zeros in {time.time()-t0:.1f}s')

    # ── Test at lam^2 = 200 (small, fast) ──
    print('\n\n  TEST: lam^2 = 200')
    print('  ' + '=' * 60)

    lam_sq = 200
    w_hat_mp, L_mp, N = build_w_hat_mp(lam_sq)

    # Extract positive coefficients (n = 1, ..., N)
    w_hat_pos = [w_hat_mp[N + n] for n in range(1, N + 1)]

    # Matrix barrier for reference
    t0 = time.time()
    mat_bar = matrix_barrier(lam_sq, N)
    print(f'  Matrix barrier: {mat_bar:.8f}  ({time.time()-t0:.1f}s)')

    # Spectral barrier: sum over zeros
    print(f'  Computing Mellin transforms at {n_zeros} zeros...', flush=True)

    contributions = []
    running_sum = 0.0

    for z_idx, gamma in enumerate(zeros):
        t0 = time.time()
        G = mellin_transform_gw(w_hat_pos, L_mp, gamma)
        G_sq = float(re(G * conj(G)))
        contributions.append(G_sq)
        running_sum += G_sq
        dt = time.time() - t0

        if (z_idx + 1) <= 10 or (z_idx + 1) % 20 == 0:
            print(f'    zero {z_idx+1:>3d}: gamma={gamma:>8.3f}  '
                  f'|G|^2={G_sq:.8f}  running={running_sum:.8f}  '
                  f'({dt:.1f}s)', flush=True)

    spectral_bar = running_sum
    print(f'\n  Spectral barrier (sum of {n_zeros} terms): {spectral_bar:.8f}')
    print(f'  Matrix barrier:                          {mat_bar:.8f}')
    print(f'  Ratio spectral/matrix:                   {spectral_bar/mat_bar:.6f}')
    print(f'  Difference:                              {spectral_bar - mat_bar:+.8f}')

    # Per-zero analysis
    contribs = np.array(contributions)
    top_idx = np.argsort(contribs)[::-1][:10]

    print(f'\n  Top 10 contributing zeros:')
    for idx in top_idx:
        pct = contribs[idx] / spectral_bar * 100 if spectral_bar > 0 else 0
        print(f'    gamma_{idx+1:>3d} = {zeros[idx]:>8.3f}  |G|^2 = {contribs[idx]:.8f}  ({pct:.1f}%)')

    # Cumulative convergence
    cum = np.cumsum(contribs)
    print(f'\n  Cumulative convergence:')
    for n in [5, 10, 20, 50, 100]:
        if n <= len(cum):
            print(f'    First {n:>3d} zeros: {cum[n-1]:.8f}  '
                  f'({cum[n-1]/mat_bar*100:.1f}% of matrix barrier)')

    # Decay rate
    print(f'\n  |G(gamma)|^2 decay:')
    if len(contribs) > 20:
        log_g = np.log(zeros[10:])
        log_c = np.log(contribs[10:] + 1e-30)
        valid = contribs[10:] > 1e-20
        if np.sum(valid) > 3:
            c = np.polyfit(log_g[valid], log_c[valid], 1)
            print(f'    |G|^2 ~ gamma^{{{c[0]:.3f}}} (power law fit, zeros 11-{n_zeros})')

    print('\n' + '#' * 70)
    print('  SESSION 42h COMPLETE')
    print('#' * 70)
