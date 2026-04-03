"""
SESSION 42c — ANALYTICAL ASYMPTOTICS OF wr_diag[n]

Derive the limiting value of wr_diag[n] as L -> infinity for fixed n,
and as |n| -> infinity for fixed L.

wr_diag[n] = C(L) + I(n, L)

where:
    C(L) = (gamma + log(4*pi*(e^L-1)/(e^L+1)))  ->  gamma + log(4*pi) as L->inf
    I(n,L) = int_0^L [e^{x/2} * 2(1-x/L)cos(2*pi*n*x/L) - 2] / (e^x - e^{-x}) dx

Key questions:
1. Does wr_diag[n] converge for each fixed n as L -> infinity?
2. What is the limiting function wr_diag_inf(n)?
3. What is <w_hat, M_diag, w_hat> asymptotically?

The answer determines whether M_diag + M_alpha converges below 3.01.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    quad, sinh, cosh, gamma as mpgamma, loggamma)
import time

mp.dps = 30


def wr_diag_single(n_val, L_val, n_quad=8000):
    """Compute wr_diag[n] at given L."""
    L = mpf(L_val)
    eL = exp(L)
    omega_0 = mpf(2)
    n_val = int(n_val)

    def omega(x):
        return 2 * (1 - x / L) * cos(2 * pi * abs(n_val) * x / L)

    w_const = (omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1)))
    dx = L / n_quad
    integral = mpf(0)
    for k in range(n_quad):
        x = dx * (k + mpf(1) / 2)
        numer = exp(x / 2) * omega(x) - omega_0
        denom = exp(x) - exp(-x)
        if abs(denom) > mpf(10)**(-25):
            integral += numer / denom
    integral *= dx
    return float(w_const + integral)


def wr_diag_limit(n_val, n_quad=8000):
    """
    Compute lim_{L->inf} wr_diag[n] analytically.

    As L -> inf:
    C(L) -> gamma + log(4*pi)
    I(n, L) -> int_0^inf [e^{x/2} * 2*cos(0) - 2] / (e^x - e^{-x}) dx
                                                      ^^ because 2*pi*n*x/L -> 0

    Wait, this depends on how n scales with L. For FIXED n:
    cos(2*pi*n*x/L) -> cos(0) = 1 for fixed x as L -> inf
    and (1-x/L) -> 1 for fixed x

    So omega_n(x) -> 2*1*1 = 2 = omega_0 for fixed x.

    This means the integrand [e^{x/2}*omega - omega_0]/(e^x-e^{-x})
    -> [e^{x/2}*2 - 2]/(e^x-e^{-x}) = 2(e^{x/2}-1)/(e^x-e^{-x})

    And the integral int_0^inf 2(e^{x/2}-1)/(e^x-e^{-x}) dx converges.

    But for n ~ a = L/(4*pi), the cos oscillates at O(1) frequency.
    The weight |w_hat[n]|^2 peaks at n ~ a, so we need wr_diag at n ~ L/(4*pi).
    """
    # For fixed n, the limit as L -> inf:
    n_val = abs(int(n_val))

    # Constant part: gamma + log(4*pi)
    const = float(euler + log(4 * pi))

    # Integral part: int_0^inf [e^{x/2} * 2 - 2] / (e^x - e^{-x}) dx
    # = 2 * int_0^inf (e^{x/2} - 1) / (e^x - e^{-x}) dx
    # = 2 * int_0^inf (e^{x/2} - 1) / (2*sinh(x)) dx

    def integrand(x):
        return 2 * (exp(x/2) - 1) / (exp(x) - exp(-x))

    integral = float(quad(integrand, [mpf(0), mpf('inf')]))

    return const + integral


def wr_diag_at_scaled_n(t, L_val, n_quad=8000):
    """
    Compute wr_diag[n] where n = round(t * L / (4*pi)).
    This evaluates wr_diag at the Lorentzian peak scale.
    """
    L_f = float(L_val)
    n_val = max(1, round(t * L_f / (4 * np.pi)))
    return wr_diag_single(n_val, L_f, n_quad), n_val


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42c -- wr_diag[n] ASYMPTOTICS')
    print('#' * 70)

    # Part 1: wr_diag[n] for fixed n as L -> infinity
    print('\n  PART 1: wr_diag[n] for fixed n, varying L')
    print('  ' + '=' * 60)

    for n_val in [0, 1, 2, 5]:
        print(f'\n  n = {n_val}:')
        print(f'  {"L":>8s} {"wr_diag":>12s}')
        print('  ' + '-' * 24)
        for L_val in [5.0, 7.0, 9.0, 11.0, 15.0, 20.0, 30.0]:
            t0 = time.time()
            val = wr_diag_single(n_val, L_val, n_quad=6000)
            dt = time.time() - t0
            print(f'  {L_val:>8.1f} {val:>+12.6f}  ({dt:.1f}s)')

    # Part 2: Limiting value
    print('\n\n  PART 2: Limiting value as L -> inf (fixed n)')
    print('  ' + '=' * 60)

    limit_0 = wr_diag_limit(0)
    print(f'  wr_diag_inf(0) = {limit_0:.8f}')
    print(f'  gamma + log(4*pi) = {float(euler + log(4*pi)):.8f}')
    print(f'  Integral part = {limit_0 - float(euler + log(4*pi)):.8f}')

    # Part 3: wr_diag at the Lorentzian peak scale n ~ L/(4*pi)
    print('\n\n  PART 3: wr_diag at n ~ t*L/(4*pi) (Lorentzian peak)')
    print('  ' + '=' * 60)

    for t in [0.5, 1.0, 1.5, 2.0]:
        print(f'\n  t = {t} (n ~ {t}*L/(4*pi)):')
        print(f'  {"L":>8s} {"n":>4s} {"wr_diag":>12s}')
        print('  ' + '-' * 30)
        for L_val in [5.0, 7.0, 9.0, 11.0, 15.0]:
            val, n = wr_diag_at_scaled_n(t, L_val, n_quad=6000)
            print(f'  {L_val:>8.1f} {n:>4d} {val:>+12.6f}')

    # Part 4: Predict M_diag asymptote
    print('\n\n  PART 4: Predict <w_hat, M_diag, w_hat> limit')
    print('  ' + '=' * 60)

    # The Rayleigh quotient is sum |w_hat[n]|^2 * wr_diag[n]
    # As L -> inf, |w_hat[n]|^2 ~ Lorentzian peaked at n ~ a = L/(4*pi)
    # If wr_diag at the peak converges, then M_diag converges.

    # Compute M_diag at several L values using explicit summation
    for L_val in [7.0, 9.0, 11.0, 15.0]:
        N = max(15, round(6 * L_val))
        ns = np.arange(-N, N + 1, dtype=float)
        w_vec = ns / (L_val**2 + (4*np.pi)**2 * ns**2)
        w_vec[N] = 0.0
        w_hat = w_vec / np.linalg.norm(w_vec)

        # Compute wr_diag for all n
        t0 = time.time()
        wr_vals = np.array([wr_diag_single(int(n), L_val, n_quad=4000) for n in ns])
        dt = time.time() - t0

        mdiag = np.sum(w_hat**2 * wr_vals)
        # Weighted average n at peak
        avg_n = np.sum(w_hat**2 * np.abs(ns))

        print(f'  L={L_val:.1f}  N={N}  <w,Md,w>={mdiag:+.6f}  '
              f'avg|n|={avg_n:.2f}  wr[0]={wr_vals[N]:.4f}  ({dt:.0f}s)')

    print('\n' + '#' * 70)
    print('  SESSION 42c COMPLETE')
    print('#' * 70)
