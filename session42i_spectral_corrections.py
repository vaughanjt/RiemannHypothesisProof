"""
SESSION 42i — ANALYTIC CORRECTIONS FOR SPECTRAL BARRIER

The Weil explicit formula gives:
    sum_rho h_hat(gamma_rho) = h_hat(i/2) + h_hat(-i/2) + correction_integral - prime_sum

For the quadratic form h_w = |g_w|^2 (positive definite test function):
    sum_rho |G_w(rho)|^2 = barrier + analytic_corrections

So: barrier = sum_rho |G_w(rho)|^2 - analytic_corrections

Compute the corrections to reconcile the spectral sum with the matrix barrier.

The corrections come from:
1. The "conductor" term (proportional to the test function at 0)
2. The Gamma-function integral
3. The continuous spectrum contribution
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, sin, cos, quad,
                    power, sqrt, fabs, im, re, conj, loggamma, digamma,
                    gamma as mpgamma)
import time
import sys

mp.dps = 25


def compute_gw_at_s(w_hat_positive, L, s_val):
    """
    Compute G_w(s) = integral_0^L g_w(x) * x^{s-1} dx
    for complex s.

    g_w(x) = 2 * sum_n w_hat[n] * sin(2*pi*n*x/L)
    """
    L_s = power(L, s_val)
    G = mpc(0, 0)

    for n_idx in range(len(w_hat_positive)):
        n = n_idx + 1
        wn = w_hat_positive[n_idx]
        if fabs(wn) < mpf(10)**(-20):
            continue

        freq = 2 * pi * n

        def integrand_real(u):
            return sin(freq * u) * power(u, re(s_val) - 1) * cos(im(s_val) * log(u))

        def integrand_imag(u):
            return sin(freq * u) * power(u, re(s_val) - 1) * sin(im(s_val) * log(u))

        I_r = quad(integrand_real, [mpf(0), mpf(1)], maxdegree=8)
        I_i = quad(integrand_imag, [mpf(0), mpf(1)], maxdegree=8)

        G += 2 * wn * L_s * mpc(I_r, I_i)

    return G


def weil_correction_terms(w_hat_positive, L, lam_sq):
    """
    Compute the analytic correction terms in the Weil explicit formula.

    For a test function h(x) on [0, inf), the explicit formula gives:
    sum_rho h_hat(gamma_rho) = h_hat(0) * (log(conductor) + C_0)
                              + integral involving Gamma'/Gamma
                              - sum_{p^k} log(p) * h(log p^k) / p^{k/2}

    For the QUADRATIC form with h = g_w * g_w (convolution):
    h_hat(gamma) = |G_w(1/2 + i*gamma)|^2

    The corrections are:
    A1 = h_hat(0) * log_conductor_term
    A2 = integral of h * Gamma'/Gamma kernel
    A3 = prime sum contribution (already in our matrix)

    Actually, the explicit formula for the Weil quadratic form is:
    <v, QW, v> = sum_rho |G(rho)|^2 - <v, (W02 + M_diag_correction), v>

    Let me just compute G(s) at key points and see what corrections
    reconcile the spectral sum with the matrix barrier.
    """

    # G_w at s = 1/2 (the central point)
    G_half = compute_gw_at_s(w_hat_positive, L, mpf(1)/2)

    # G_w at s = 0 (related to conductor term)
    G_zero = compute_gw_at_s(w_hat_positive, L, mpf(0))

    # G_w at s = 1 (related to residue term)
    G_one = compute_gw_at_s(w_hat_positive, L, mpf(1))

    # |G(1/2)|^2 — this is h_hat(0), related to the leading correction
    G_half_sq = float(re(G_half * conj(G_half)))

    # Gamma'/Gamma correction:
    # The explicit formula includes a term involving:
    # integral_0^inf h(x) * [Omega(x)] dx
    # where Omega(x) = Gamma'/Gamma related kernel
    # For h(x) = |g_w(x)|^2, this is a double integral.

    return {
        'G_half': complex(G_half),
        'G_zero': complex(G_zero),
        'G_one': complex(G_one),
        'G_half_sq': G_half_sq,
        '|G_zero|^2': float(re(G_zero * conj(G_zero))),
        '|G_one|^2': float(re(G_one * conj(G_one))),
    }


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42i -- SPECTRAL CORRECTIONS')
    print('#' * 70)

    lam_sq = 200
    L_mp = log(mpf(lam_sq))
    N = max(15, round(6 * float(L_mp)))

    # Build w_hat
    norm_sq = mpf(0)
    coeffs = []
    for n in range(-N, N + 1):
        val = mpf(n) / (L_mp**2 + 16 * pi**2 * mpf(n)**2)
        coeffs.append(val)
        norm_sq += val**2
    norm = sqrt(norm_sq)
    w_hat_pos = [coeffs[N + n] / norm for n in range(1, N + 1)]

    print(f'\n  lam^2 = {lam_sq}, L = {float(L_mp):.3f}, N = {N}')

    # Compute corrections
    print('\n  Computing G_w at special points...')
    t0 = time.time()
    corr = weil_correction_terms(w_hat_pos, L_mp, lam_sq)
    print(f'  Done in {time.time()-t0:.1f}s')

    print(f'\n  G_w(1/2) = {corr["G_half"]}')
    print(f'  |G_w(1/2)|^2 = {corr["G_half_sq"]:.8f}')
    print(f'  G_w(0) = {corr["G_zero"]}')
    print(f'  |G_w(0)|^2 = {corr["|G_zero|^2"]:.8f}')
    print(f'  G_w(1) = {corr["G_one"]}')
    print(f'  |G_w(1)|^2 = {corr["|G_one|^2"]:.8f}')

    # Attempt to reconcile
    # From session 42h, the spectral sum (100 zeros) will be some value S.
    # The matrix barrier is 0.04801.
    # The correction should be S - 0.04801.

    # Key relationship from explicit formula:
    # Sum_rho |G(rho)|^2 = barrier + C
    # where C involves |G(1/2)|^2 * log-conductor + Gamma integral + ...

    # The log-conductor for Riemann zeta at level lambda^2:
    # This is just log(lambda^2 / (2*pi*e)) per the Connes framework
    log_conductor = float(log(mpf(lam_sq) / (2 * pi * exp(1))))

    print(f'\n  Log conductor term: {log_conductor:.6f}')
    print(f'  |G(1/2)|^2 * log_cond = {corr["G_half_sq"] * log_conductor:.6f}')

    # The explicit formula correction for the completed zeta function is:
    # C = integral_0^L |g_w(x)|^2 * [sum of kernel terms] dx
    # This includes the Gamma'/Gamma term and the W02 term.

    # Actually, from our matrix decomposition:
    # barrier = W02_term - M_prime_term - M_diag_term - M_alpha_term
    # spectral = sum |G(rho)|^2

    # The explicit formula says:
    # spectral = W02_term - M_prime_term + [zero sum analytic correction]
    # barrier = spectral - M_diag_term - M_alpha_term - [correction]

    # So: spectral - barrier = M_diag_term + M_alpha_term + correction

    # From session 42a at lam^2=200:
    # M_diag = 1.178, M_alpha = 0.853, sum = 2.031
    # W02-Mp = 2.079

    # So: spectral (inf zeros) should be approximately:
    # W02_term + M_diag_term + M_alpha_term + correction
    # = barrier + (M_diag + M_alpha) + correction
    # = 0.048 + 2.031 + correction

    mat_barrier = 0.04801
    m_diag_alpha = 2.031

    print(f'\n  Matrix barrier: {mat_barrier:.6f}')
    print(f'  M_diag + M_alpha: {m_diag_alpha:.6f}')
    print(f'  Expected spectral total: {mat_barrier + m_diag_alpha:.6f} + correction')
    print(f'  (i.e., the spectral sum should be ~2.08 + correction)')

    print('\n' + '#' * 70)
    print('  SESSION 42i COMPLETE')
    print('#' * 70)
