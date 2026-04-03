"""
SESSION 41e — SPECTRAL INTERPRETATION OF THE BARRIER

The Weil quadratic form has a spectral representation:
    <v, QW, v> = sum_rho |h_v(rho)|^2

where h_v is the test function associated with v and rho are the zeta zeros.
(Conditional on RH: this is a sum of squares, hence >= 0.)

For our specific v = w_hat (odd Lorentzian), compute:
1. The test function h_w in position space
2. Its values at the first few hundred zeta zeros
3. Compare spectral sum to the matrix barrier

This also gives us insight into WHETHER the barrier can approach 0:
if h_w(gamma_rho) is bounded away from 0 for all rho, the barrier stays positive.

Also: fast barrier computation using only primes (no mpmath overhead for
the analytic terms, computed via a different route).
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, zetazero, log, pi, exp, cos, sin, sinh
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mp.dps = 30


# ═══════════════════════════════════════════════════════════════
# ZETA ZEROS
# ═══════════════════════════════════════════════════════════════

def get_zeros(n_zeros=200):
    """Get first n_zeros imaginary parts of zeta zeros."""
    zeros = []
    for k in range(1, n_zeros + 1):
        z = zetazero(k)
        zeros.append(float(z.imag))
    return np.array(zeros)


# ═══════════════════════════════════════════════════════════════
# TEST FUNCTION FOR w_hat
# ═══════════════════════════════════════════════════════════════

def compute_test_function_at_zero(gamma, lam_sq, N=None):
    """
    Compute h_w(gamma) = sum_n w_hat[n] * g_n(gamma)

    where g_n(gamma) is the test function basis element:
    g_n(gamma) = integral_0^L e^{(1/2 + i*gamma)*x} * e^{2*pi*i*n*x/L} * (1-x/L) dx / sqrt(L)

    Simplified for the Weil explicit formula framework:
    The n-th Fourier mode contributes a factor related to
    the Mellin transform of the triangular window (1-x/L).

    For the even test function omega_n(x) = 2(1-x/L)cos(2*pi*n*x/L):
    Its Mellin transform at s = 1/2+i*gamma is:
    H_n(gamma) = int_0^L e^{-(1/2+i*gamma)*x} * 2(1-x/L)*cos(2*pi*n*x/L) dx

    The quadratic form value is then:
    <w_hat, QW, w_hat> = sum_{n,m} w_hat[n] * K(n,m) * w_hat[m]

    where K(n,m) = sum_rho H_n(gamma_rho) * conj(H_m(gamma_rho)).

    But for the barrier we need the full form. Let me compute it differently.
    """
    # The explicit formula for a single test function h(x) gives:
    # sum_rho h_hat(gamma_rho) = [analytic terms] + [prime sum]
    #
    # For our matrix form, the quadratic form <v, QW, v> corresponds to
    # using the test function:
    #   h_v(x) = sum_{n,m} v[n] * phi_{nm}(x) * v[m]
    #
    # This is complicated. Let me instead just verify numerically that
    # the matrix QW matches the expected structure.
    pass


def barrier_from_explicit_formula(lam_sq, N=None, n_quad=8000):
    """
    Compute barrier using the explicit formula directly, evaluating
    the Weil quadratic form <w, QW, w> as a combination of explicit
    formula terms.

    This is just a re-derivation to cross-check the matrix computation.
    """
    L_f = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L_f))

    # w_hat weights
    ns = np.arange(-N, N + 1, dtype=float)
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N] = 0.0
    w_norm = np.linalg.norm(w_vec)
    w_hat = w_vec / w_norm

    # Compute <w, W02, w> analytically
    pf = 32 * L_f * np.sinh(L_f / 4)**2
    a = L_f / (4 * np.pi)
    S1_trunc = sum(n**2 / (a**2 + n**2)**2 for n in range(-N, N+1))
    w02_term = -pf * (4 * np.pi)**2 * S1_trunc / (4 * np.pi)**4

    # For M_prime: compute directly as weighted sum
    limit = min(int(lam_sq), 10000)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False

    # M_prime as full quadratic form (both diagonal and off-diagonal)
    mprime_total = 0.0
    for p in range(2, limit + 1):
        if not sieve[p]:
            continue
        pk = p
        k = 1
        while pk <= lam_sq:
            logp = np.log(p)
            logpk = k * logp
            weight = logp * pk**(-0.5)

            # Compute <w, Q_{pk}, w> where Q is the q_func matrix
            # Diagonal contribution
            diag_sum = 0.0
            for i in range(2 * N + 1):
                n = i - N
                diag_sum += w_hat[i]**2 * 2 * (L_f - logpk) / L_f * np.cos(2 * np.pi * n * logpk / L_f)

            # Off-diagonal contribution
            off_sum = 0.0
            for i in range(2 * N + 1):
                n = i - N
                for j in range(2 * N + 1):
                    m = j - N
                    if n != m:
                        q_nm = (np.sin(2*np.pi*m*logpk/L_f) - np.sin(2*np.pi*n*logpk/L_f)) / (np.pi*(n-m))
                        off_sum += w_hat[i] * q_nm * w_hat[j]

            mprime_total += weight * (diag_sum + off_sum)

            pk *= p
            k += 1

    return w02_term, mprime_total


# ═══════════════════════════════════════════════════════════════
# QUICK BARRIER (recompute M_prime without full mpmath)
# ═══════════════════════════════════════════════════════════════

def quick_barrier_w(lam_sq, N=None):
    """
    Compute just the M_prime Rayleigh quotient without mpmath.
    Uses numpy only for maximum speed.
    """
    L_f = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    ns = np.arange(-N, N + 1, dtype=float)
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N] = 0.0
    w_norm = np.linalg.norm(w_vec)
    w_hat = w_vec / w_norm

    # W02 term (exact for truncated sum)
    pf = 32 * L_f * np.sinh(L_f / 4)**2
    W02 = np.zeros((dim, dim))
    for i in range(dim):
        n = ns[i]
        for j in range(dim):
            m = ns[j]
            W02[i,j] = pf * (L_f**2 - (4*np.pi)**2 * m * n) / \
                        ((L_f**2 + (4*np.pi)**2 * m**2) * (L_f**2 + (4*np.pi)**2 * n**2))
    w02_rq = w_hat @ W02 @ w_hat

    # M_prime term (numpy, fast)
    limit = min(int(lam_sq), 10000)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            sieve[i*i::i] = False

    M_prime = np.zeros((dim, dim))
    for p in range(2, limit + 1):
        if not sieve[p]:
            continue
        pk = p
        k = 1
        while pk <= lam_sq:
            logp = np.log(p)
            logpk = k * logp
            weight = logp * pk**(-0.5)
            y = logpk

            for i in range(dim):
                n = ns[i]
                # Diagonal
                M_prime[i, i] += weight * 2 * (L_f - y) / L_f * np.cos(2*np.pi*n*y/L_f)
                # Off-diagonal
                for j in range(i+1, dim):
                    m = ns[j]
                    q = (np.sin(2*np.pi*m*y/L_f) - np.sin(2*np.pi*n*y/L_f)) / (np.pi*(n-m))
                    M_prime[i, j] += weight * q
                    M_prime[j, i] += weight * q

            pk *= p
            k += 1

    mprime_rq = w_hat @ M_prime @ w_hat

    return {
        'w02': w02_rq,
        'mprime': mprime_rq,
        'diff': w02_rq - mprime_rq,
        'N': N,
        'dim': dim,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 41e — SPECTRAL BARRIER ANALYSIS')
    print('#' * 70)

    # ── Part 1: Get zeta zeros ──
    print('\n  PART 1: Loading zeta zeros...')
    t0 = time.time()
    zeros = get_zeros(200)
    print(f'  Loaded {len(zeros)} zeros in {time.time()-t0:.1f}s')
    print(f'  First 10: {zeros[:10]}')
    print(f'  Range: [{zeros[0]:.4f}, {zeros[-1]:.4f}]')

    # ── Part 2: Quick barrier (W02 and M_prime only, no mpmath) ──
    print('\n\n  PART 2: Quick barrier (W02 - M_prime only, numpy)')
    print('  ' + '=' * 60)
    print('  Note: this is PARTIAL barrier, missing M_diag and M_alpha')

    lam_values = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]

    print(f'\n  {"lam^2":>7s} {"W02":>10s} {"M_prime":>10s} {"W02-Mp":>10s} {"N":>4s} {"time":>6s}')
    print('  ' + '-' * 55)

    quick_results = []
    for lam_sq in lam_values:
        t0 = time.time()
        r = quick_barrier_w(lam_sq)
        dt = time.time() - t0
        quick_results.append({**r, 'lam_sq': lam_sq})
        print(f'  {lam_sq:>7d} {r["w02"]:>+10.5f} {r["mprime"]:>+10.5f} '
              f'{r["diff"]:>+10.6f} {r["N"]:>4d} {dt:>5.1f}s')

    # ── Part 3: N-convergence for M_prime Rayleigh quotient ──
    print('\n\n  PART 3: N-convergence of M_prime Rayleigh quotient')
    print('  ' + '=' * 60)

    for lam_sq in [1000, 5000, 10000]:
        L_f = np.log(lam_sq)
        N_base = max(15, round(6 * L_f))
        print(f'\n  lam^2={lam_sq}:')
        print(f'  {"N":>5s} {"W02":>10s} {"M_prime":>10s} {"diff":>10s}')
        print('  ' + '-' * 38)

        for mult in [1.0, 1.5, 2.0, 3.0]:
            N = round(mult * N_base)
            t0 = time.time()
            r = quick_barrier_w(lam_sq, N=N)
            dt = time.time() - t0
            print(f'  {N:>5d} {r["w02"]:>+10.5f} {r["mprime"]:>+10.5f} '
                  f'{r["diff"]:>+10.6f}  ({dt:.1f}s)')

    # ── Part 4: Dense sweep with quick barrier ──
    print('\n\n  PART 4: Dense sweep W02-M_prime (the dominant piece)')
    print('  ' + '=' * 60)

    dense_lam = list(range(100, 2001, 100)) + list(range(2500, 10001, 500)) + \
                [12000, 15000, 20000]

    print(f'\n  {"lam^2":>7s} {"W02-Mp":>10s}')
    print('  ' + '-' * 20)

    dense_results = []
    for lam_sq in dense_lam:
        r = quick_barrier_w(lam_sq)
        dense_results.append({'lam_sq': lam_sq, 'diff': r['diff']})
        print(f'  {lam_sq:>7d} {r["diff"]:>+10.6f}')

    # Statistics
    diffs = np.array([r['diff'] for r in dense_results])
    Ls = np.array([np.log(r['lam_sq']) for r in dense_results])
    print(f'\n  Statistics:')
    print(f'    min={diffs.min():.6f}  max={diffs.max():.6f}  '
          f'mean={diffs.mean():.6f}  std={diffs.std():.6f}')

    # Linear trend
    c = np.polyfit(Ls, diffs, 1)
    print(f'    Linear trend: {c[0]:+.6f} * L + {c[1]:+.6f}')

    print('\n' + '#' * 70)
    print('  SESSION 41e COMPLETE')
    print('#' * 70)
