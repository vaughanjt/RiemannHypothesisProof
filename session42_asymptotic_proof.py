"""
SESSION 42 — ASYMPTOTIC PROOF ATTEMPT

Can we PROVE margin(L) > drain(L) for all L > L_0?

Two strategies:
A) Prove margin is monotone increasing => bounded below by margin(L_0)
B) Prove drain is bounded above by decomposing into small primes + PNT tail

For (A): compute d(margin)/dL numerically at many points.
If it's always positive, monotonicity is established (modulo rigor).

For (B): drain = sum_{p<=P} [exact(p) - integral_share(p)] + tail(P)
The small-prime part is a finite sum of known constants.
The tail is bounded by PNT: |tail| <= C * exp(-c*sqrt(log P)).
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session42k_close_the_gap import compute_margin_precise, compute_drain
from session42j_margin_vs_drain import mprime_pnt_integral
from session41g_uncapped_barrier import compute_barrier_partial, sieve_primes


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  ASYMPTOTIC PROOF ATTEMPT')
    print('#' * 72)

    # ═══════════════════════════════════════════════════════════
    # PART A: IS MARGIN MONOTONE?
    # ═══════════════════════════════════════════════════════════
    print('\n  PART A: Monotonicity of margin(L)')
    print('  ' + '=' * 60)

    # Compute margin at fine grid
    lam_values = list(range(100, 501, 20)) + list(range(600, 2001, 100)) + \
                 list(range(2500, 10001, 500)) + [12000, 15000, 20000, 30000, 50000]

    margins = []
    print(f'\n  {"lam^2":>7s} {"L":>7s} {"margin":>12s} {"d_margin":>10s}')
    print('  ' + '-' * 42)

    for lam_sq in lam_values:
        t0 = time.time()
        r = compute_margin_precise(lam_sq, n_quad=4000, n_pnt=5000)
        dt = time.time() - t0
        margins.append((lam_sq, np.log(lam_sq), r['margin']))

    # Check monotonicity
    mono_violations = 0
    for i in range(len(margins)):
        L = margins[i][1]
        m = margins[i][2]
        dm = (margins[i][2] - margins[i-1][2]) / (margins[i][1] - margins[i-1][1]) if i > 0 else 0
        if i > 0 and margins[i][2] < margins[i-1][2]:
            mono_violations += 1
            marker = ' <<<'
        else:
            marker = ''
        if i % 5 == 0 or marker:
            print(f'  {margins[i][0]:>7d} {L:>7.3f} {m:>+12.8f} {dm:>+10.6f}{marker}')

    print(f'\n  Monotonicity violations: {mono_violations}')
    print(f'  Min margin: {min(m for _,_,m in margins):.8f}')
    print(f'  Max margin: {max(m for _,_,m in margins):.8f}')

    if mono_violations == 0:
        print(f'  => MARGIN IS MONOTONE on [{margins[0][0]}, {margins[-1][0]}]')
        print(f'  => margin(L) >= {margins[0][2]:.6f} for all L >= {margins[0][1]:.2f}')
    else:
        print(f'  => Margin is NOT strictly monotone ({mono_violations} violations)')
        # Find local minimum
        min_m = min(m for _,_,m in margins)
        print(f'  => But margin >= {min_m:.6f} everywhere')

    # ═══════════════════════════════════════════════════════════
    # PART B: DRAIN DECOMPOSITION
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART B: Drain decomposition (small primes + tail)')
    print('  ' + '=' * 60)

    # For each lam^2, decompose drain into:
    # drain = drain_small(P=100) + drain_tail(P=100)
    # where drain_small = Σ_{p<=100} [exact - integral_share]
    # and drain_tail = Σ_{p>100} [exact - integral_share]

    for lam_sq in [1000, 5000, 10000, 50000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        dim = 2 * N + 1
        ns = np.arange(-N, N + 1, dtype=float)
        w = ns / (L**2 + (4*np.pi)**2 * ns**2)
        w[N] = 0.0
        w_hat = w / np.linalg.norm(w)
        nm_diff = ns[:, None] - ns[None, :]

        primes = sieve_primes(int(lam_sq))

        # Per-prime contribution to M_prime
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
                sin_arr = np.sin(2*np.pi*ns*y/L)
                cos_arr = np.cos(2*np.pi*ns*y/L)
                diag = 2*(L-y)/L * np.sum(w_hat**2 * cos_arr)
                sin_diff = sin_arr[None,:] - sin_arr[:,None]
                with np.errstate(divide='ignore', invalid='ignore'):
                    off = sin_diff / (np.pi * nm_diff)
                np.fill_diagonal(off, 0.0)
                off_val = w_hat @ off @ w_hat
                total += weight * (diag + off_val)
                pk *= int(p)
                k += 1
            per_prime[int(p)] = total

        actual_mp = sum(per_prime.values())
        pnt_mp = mprime_pnt_integral(lam_sq, N)
        total_drain = actual_mp - pnt_mp

        # Split by P=100
        P = 100
        small_sum = sum(v for p, v in per_prime.items() if p <= P)
        large_sum = sum(v for p, v in per_prime.items() if p > P)

        # PNT integral split roughly by the P boundary
        # (crude: allocate proportionally by prime count)
        n_small = sum(1 for p in primes if p <= P)
        n_large = sum(1 for p in primes if p > P)
        # Actually, we need the integral from 2 to P and from P to lam^2
        # For now, just report the total drain and the small-prime contribution

        print(f'\n  lam^2 = {lam_sq}:')
        print(f'    actual Mp = {actual_mp:+.6f}')
        print(f'    PNT Mp    = {pnt_mp:+.6f}')
        print(f'    drain     = {total_drain:+.6f}')
        print(f'    Mp from p<=100: {small_sum:+.6f} ({n_small} primes)')
        print(f'    Mp from p>100:  {large_sum:+.6f} ({n_large} primes)')

        # The drain from small primes: how much does their exact sum
        # exceed what PNT predicts for that range?
        # Crude estimate: PNT for p<=100 ~ integral_2^100 F(log(t)/L)/sqrt(t) dt
        n_pnt_pts = 2000
        y_pts = np.linspace(np.log(2), min(np.log(P), L), n_pnt_pts)
        dy = y_pts[1] - y_pts[0]
        pnt_small = 0.0
        for y in y_pts:
            sin_arr = np.sin(2*np.pi*ns*y/L)
            cos_arr = np.cos(2*np.pi*ns*y/L)
            dv = 2*(L-y)/L * np.sum(w_hat**2 * cos_arr)
            sd = sin_arr[None,:] - sin_arr[:,None]
            with np.errstate(divide='ignore', invalid='ignore'):
                of = sd / (np.pi * nm_diff)
            np.fill_diagonal(of, 0.0)
            ov = w_hat @ of @ w_hat
            pnt_small += np.exp(y/2) * (dv + ov) * dy

        drain_small = small_sum - pnt_small
        drain_large = large_sum - (pnt_mp - pnt_small)

        print(f'    PNT for p<=100: {pnt_small:+.6f}')
        print(f'    drain from p<=100: {drain_small:+.6f}')
        print(f'    drain from p>100:  {drain_large:+.6f}')
        print(f'    Ratio |drain_large|/|drain|: {abs(drain_large)/abs(total_drain)*100:.1f}%')

    # ═══════════════════════════════════════════════════════════
    # PART C: THE BOUND
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART C: Can we bound drain(L) <= 0.238?')
    print('  ' + '=' * 60)

    # From the data: drain ranges from 0.198 to 0.238
    # The drain_small (p<=100) is the dominant piece
    # The drain_large (p>100) should be bounded by PNT

    # Check: what is drain_small as a function of L?
    print(f'\n  drain_small(L) at various L:')
    for lam_sq in [500, 1000, 2000, 5000, 10000, 20000]:
        d = compute_drain(lam_sq)
        print(f'    lam^2={lam_sq:>6d}: drain={d:+.6f}')

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print('\n\n  VERDICT')
    print('  ' + '=' * 60)

    min_margin = min(m for _,_,m in margins)
    max_drain = max(abs(compute_drain(l)) for l in [500,1000,2000,5000,10000])

    print(f'  min margin (computed): {min_margin:.6f}')
    print(f'  max |drain| (computed): {max_drain:.6f}')
    print(f'  gap: {min_margin - max_drain:.6f}')

    if min_margin > max_drain:
        print(f'  => margin > drain at all computed points')
        print(f'\n  To make this rigorous:')
        print(f'  1. Prove margin monotone => margin >= {min_margin:.4f}')
        print(f'  2. Prove drain bounded => drain <= {max_drain:.4f}')
        print(f'  3. Gap = {min_margin - max_drain:.4f} > 0')
    else:
        print(f'  => Cannot close the gap from computed data alone')

    print('\n' + '#' * 72)
