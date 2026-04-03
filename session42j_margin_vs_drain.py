"""
SESSION 42j — THE KILL SHOT: ANALYTIC MARGIN vs SMALL PRIME DRAIN

Decompose the barrier as:
    barrier = smooth_margin - small_prime_drain + PNT_error

Where:
    smooth_margin = W02_exact - M_prime_PNT_integral - M_diag - M_alpha
        (purely analytic — no primes, no zeros, just special functions)

    small_prime_drain = M_prime_actual - M_prime_PNT_integral
        (finite sum: how much the actual first ~20 primes deviate from PNT)

    PNT_error = O(exp(-c*sqrt(L)))
        (residual from large primes, bounded unconditionally)

If smooth_margin > small_prime_drain + PNT_error_bound for all L >= L_0,
then barrier > 0 WITHOUT ASSUMING RH.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, quad)
import time
import sys
import os

mp.dps = 25

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import compute_barrier_partial, sieve_primes


# ═══════════════════════════════════════════════════════════════
# PNT INTEGRAL OF M_PRIME
# ═══════════════════════════════════════════════════════════════

def mprime_pnt_integral(lam_sq, N=None, n_pts=5000):
    """
    Compute the PNT smooth approximation:
    integral_2^{lam^2} F(log(t)/L) / sqrt(t) dt

    where F(u) = <w_hat, Q_{uL}, w_hat> is the filter function.

    By PNT: sum_{p<=x} log(p) * h(p) ~ integral_2^x h(t) dt
    """
    L = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    nm_diff = ns[:, None] - ns[None, :]

    # Integration over y = log(t), t = e^y, dt = e^y dy
    # integral = integral_{log(2)}^{L} e^{-y/2} * F(y/L) * e^y dy
    #          = integral_{log(2)}^{L} e^{y/2} * F(y/L) dy
    y_pts = np.linspace(np.log(2), L, n_pts)
    dy = y_pts[1] - y_pts[0]

    integral = 0.0
    for y in y_pts:
        sin_arr = np.sin(2 * np.pi * ns * y / L)
        cos_arr = np.cos(2 * np.pi * ns * y / L)

        # Diagonal: 2(L-y)/L * cos(2*pi*n*y/L)
        diag_val = 2 * (L - y) / L * np.sum(w_hat**2 * cos_arr)

        # Off-diagonal
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off_diag = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off_diag, 0.0)
        off_val = w_hat @ off_diag @ w_hat

        F_val = diag_val + off_val
        integral += np.exp(y / 2) * F_val * dy

    return integral


# ═══════════════════════════════════════════════════════════════
# M_DIAG + M_ALPHA (mpmath)
# ═══════════════════════════════════════════════════════════════

def mdiag_malpha(lam_sq, N=None, n_quad=4000):
    """Compute M_diag + M_alpha on w direction."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    if N is None:
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    w = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    # Alpha
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

    # wr_diag
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
            if abs(denom) > mpf(10)**(-20):
                integral += numer / denom
        integral *= dx
        wr_diag[nv] = float(w_const + integral)
        wr_diag[-nv] = wr_diag[nv]

    diag_vals = np.array([wr_diag[int(n)] for n in ns])
    mdiag = float(np.sum(w_hat**2 * diag_vals))

    malpha = 0.0
    for i in range(dim):
        for j in range(dim):
            if i != j:
                n, m = int(ns[i]), int(ns[j])
                malpha += w_hat[i] * (alpha[m] - alpha[n]) / (n - m) * w_hat[j]

    return mdiag, float(malpha)


# ═══════════════════════════════════════════════════════════════
# SMALL PRIME DRAIN
# ═══════════════════════════════════════════════════════════════

def small_prime_drain(lam_sq, P_cutoff=None, N=None):
    """
    Compute the drain = M_prime_actual - M_prime_PNT_integral.

    Split by prime size: drain from p <= P, and drain from p > P.
    The drain from p > P should be tiny (PNT works well for large primes).
    """
    L = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    nm_diff = ns[:, None] - ns[None, :]
    primes = sieve_primes(int(lam_sq))

    if P_cutoff is None:
        P_cutoff = int(lam_sq)

    # Per-prime contributions
    per_prime = {}
    for p in primes:
        pk = int(p)
        k = 1
        logp = np.log(p)
        total = 0.0
        while pk <= lam_sq:
            logpk = k * logp
            weight = logp * pk**(-0.5)
            y = logpk

            sin_arr = np.sin(2 * np.pi * ns * y / L)
            cos_arr = np.cos(2 * np.pi * ns * y / L)
            diag = 2 * (L - y) / L * cos_arr
            diag_c = weight * np.sum(w_hat**2 * diag)

            sin_diff = sin_arr[None, :] - sin_arr[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                off = sin_diff / (np.pi * nm_diff)
            np.fill_diagonal(off, 0.0)
            off_c = weight * (w_hat @ off @ w_hat)

            total += diag_c + off_c
            pk *= int(p)
            k += 1
        per_prime[int(p)] = total

    # PNT integral
    pnt = mprime_pnt_integral(lam_sq, N)

    # Total actual M_prime
    actual = sum(per_prime.values())

    # Drain = actual - PNT
    total_drain = actual - pnt

    # Split by prime size
    small_drain = sum(v for p, v in per_prime.items() if p <= P_cutoff) - \
                  pnt * (P_cutoff / lam_sq)  # rough allocation
    # Actually, let's compute more carefully:
    # The PNT integral contribution from [2, P] vs [P, lam^2]
    # For now, just report per-prime contributions

    # Top contributing primes to the drain
    sorted_primes = sorted(per_prime.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        'actual_mp': actual,
        'pnt_mp': pnt,
        'total_drain': total_drain,
        'per_prime': per_prime,
        'sorted_primes': sorted_primes,
        'n_primes': len(primes),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN: THE KILL SHOT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42j -- THE KILL SHOT')
    print('  ANALYTIC MARGIN vs SMALL PRIME DRAIN')
    print('#' * 70)

    # ── Part 1: Compute all three pieces at multiple lambda ──
    print('\n  PART 1: Three-way decomposition')
    print('  ' + '=' * 60)

    lam_values = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    print(f'\n  {"lam^2":>7s} {"L":>6s} {"W02_exact":>10s} {"PNT_Mp":>10s} '
          f'{"Md+Ma":>10s} {"MARGIN":>10s} {"DRAIN":>10s} {"BARRIER":>10s} {"safe":>6s}')
    print('  ' + '-' * 85)

    results = []
    for lam_sq in lam_values:
        t0 = time.time()

        # W02 and actual M_prime (fast, numpy)
        r = compute_barrier_partial(lam_sq)
        w02 = r['w02']
        actual_mp = r['mprime']

        # PNT integral
        pnt_mp = mprime_pnt_integral(lam_sq)

        # M_diag + M_alpha (slow, mpmath)
        md, ma = mdiag_malpha(lam_sq)

        # Three pieces
        margin = w02 - pnt_mp - md - ma    # smooth analytic margin
        drain = actual_mp - pnt_mp           # small prime drain
        barrier = w02 - actual_mp - md - ma  # = margin - drain (actual barrier)
        pnt_error_bound = 0.01  # conservative

        safe = 'YES' if margin > abs(drain) + pnt_error_bound else 'no'

        dt = time.time() - t0
        results.append({
            'lam_sq': lam_sq, 'L': np.log(lam_sq),
            'w02': w02, 'pnt_mp': pnt_mp, 'md': md, 'ma': ma,
            'margin': margin, 'drain': drain, 'barrier': barrier,
        })

        print(f'  {lam_sq:>7d} {np.log(lam_sq):>6.2f} {w02:>+10.4f} {pnt_mp:>+10.4f} '
              f'{md+ma:>+10.4f} {margin:>+10.6f} {drain:>+10.6f} '
              f'{barrier:>+10.6f} {safe:>6s}  ({dt:.0f}s)')
        sys.stdout.flush()

    # ── Part 2: Margin and drain trends ──
    print('\n\n  PART 2: Trend analysis')
    print('  ' + '=' * 60)

    Ls = np.array([r['L'] for r in results])
    margins = np.array([r['margin'] for r in results])
    drains = np.array([r['drain'] for r in results])
    barriers = np.array([r['barrier'] for r in results])

    print(f'  Margin range:  [{margins.min():.6f}, {margins.max():.6f}]')
    print(f'  Drain range:   [{drains.min():.6f}, {drains.max():.6f}]')
    print(f'  Barrier range: [{barriers.min():.6f}, {barriers.max():.6f}]')

    # Margin trend: fit a + b/L
    if len(Ls) >= 4:
        X = np.column_stack([np.ones_like(Ls), 1/Ls])
        c_m = np.linalg.lstsq(X, margins, rcond=None)[0]
        c_d = np.linalg.lstsq(X, drains, rcond=None)[0]

        print(f'\n  Margin fit: {c_m[0]:.6f} + {c_m[1]:.4f}/L')
        print(f'  Margin limit (L->inf): {c_m[0]:.6f}')
        print(f'  Drain fit:  {c_d[0]:.6f} + {c_d[1]:.4f}/L')
        print(f'  Drain limit (L->inf): {c_d[0]:.6f}')
        print(f'  Predicted barrier limit: {c_m[0] - c_d[0]:.6f}')

    # ── Part 3: Anatomy of the drain ──
    print('\n\n  PART 3: Drain anatomy (per-prime)')
    print('  ' + '=' * 60)

    for lam_sq in [1000, 10000]:
        print(f'\n  lam^2 = {lam_sq}:')
        d = small_prime_drain(lam_sq)
        print(f'    actual_Mp = {d["actual_mp"]:+.6f}')
        print(f'    PNT_Mp    = {d["pnt_mp"]:+.6f}')
        print(f'    drain     = {d["total_drain"]:+.6f}')

        # Cumulative drain by prime size
        cum_drain_vs_pnt = 0.0
        primes_sorted = sorted(d['per_prime'].items())
        print(f'\n    Cumulative drain by prime cutoff P:')
        print(f'    {"P":>6s} {"cum_Mp":>12s} {"primes":>7s}')
        print('    ' + '-' * 30)

        cum = 0.0
        for p, c in primes_sorted:
            cum += c
            if p in [2, 3, 5, 7, 11, 13, 19, 29, 50, 100, 500, 1000] or p == primes_sorted[-1][0]:
                print(f'    {p:>6d} {cum:>+12.6f} {sum(1 for pp,_ in primes_sorted if pp<=p):>7d}')

    # ── Part 4: THE INEQUALITY ──
    print('\n\n  PART 4: THE INEQUALITY')
    print('  ' + '=' * 60)
    print('  Question: margin(L) > |drain(L)| for all L?')
    print()

    for r in results:
        gap = r['margin'] - abs(r['drain'])
        status = 'POSITIVE' if gap > 0 else 'NEGATIVE'
        print(f'  L={r["L"]:.2f}: margin={r["margin"]:+.6f}  '
              f'|drain|={abs(r["drain"]):.6f}  gap={gap:+.6f}  [{status}]')

    # Overall
    gaps = margins - np.abs(drains)
    print(f'\n  All gaps positive? {np.all(gaps > 0)}')
    print(f'  Min gap: {gaps.min():.6f} at lam^2={results[np.argmin(gaps)]["lam_sq"]}')

    if np.all(gaps > 0):
        print(f'\n  *** MARGIN EXCEEDS DRAIN AT ALL COMPUTED POINTS ***')
        print(f'  If this holds asymptotically, barrier > 0 without RH.')
    else:
        print(f'\n  Margin does NOT always exceed drain.')
        print(f'  The drain is too large at some points.')

    print('\n' + '#' * 70)
    print('  SESSION 42j COMPLETE')
    print('#' * 70)
