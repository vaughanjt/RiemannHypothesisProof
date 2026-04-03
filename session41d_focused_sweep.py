"""
SESSION 41d — FOCUSED BARRIER SWEEP + N-CONVERGENCE

Streamlined version: skip the huge lambda^2 values, focus on:
1. N-convergence at lambda^2 = 5000 and 10000
2. Dense sweep from 100 to 20000
3. Analytical formula for the full M_prime Rayleigh quotient
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, log, pi, exp, sinh
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition

mp.dps = 50


def barrier_components(lam_sq, N=None, n_quad=8000):
    """Compute barrier on w direction. Returns dict."""
    L_f = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N, n_quad=n_quad)
    M_diag, M_alpha, M_prime, M_full, vM = compute_M_decomposition(lam_sq, N, n_quad=n_quad)

    ns = np.arange(-N, N + 1, dtype=float)
    # Analytic odd eigenvector
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N] = 0.0
    w_norm = np.linalg.norm(w_vec)
    w_hat = w_vec / w_norm

    # Even eigenvector
    u_vec = 1.0 / (L_f**2 + (4 * np.pi)**2 * ns**2)
    u_hat = u_vec / np.linalg.norm(u_vec)

    return {
        'lam_sq': lam_sq, 'L': L_f, 'N': N,
        'w_barrier': float(w_hat @ QW @ w_hat),
        'u_barrier': float(u_hat @ QW @ u_hat),
        'w02': float(w_hat @ W02 @ w_hat),
        'mprime': float(w_hat @ M_prime @ w_hat),
        'mdiag': float(w_hat @ M_diag @ w_hat),
        'malpha': float(w_hat @ M_alpha @ w_hat),
        'eps_0': float(np.linalg.eigvalsh(QW)[0]),
        # Cross-direction
        'cross': float(w_hat @ QW @ u_hat),
        # 2x2 block determinant
        'u_w02': float(u_hat @ W02 @ u_hat),
        'u_mprime': float(u_hat @ M_prime @ u_hat),
    }


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 41d — FOCUSED BARRIER SWEEP')
    print('#' * 70)

    # ── Part 1: N-convergence ──
    print('\n  PART 1: N-convergence study')
    print('  ' + '=' * 60)

    for lam_sq in [1000, 5000, 10000]:
        L_f = np.log(lam_sq)
        N_base = max(15, round(6 * L_f))
        print(f'\n  lam^2={lam_sq}, L={L_f:.3f}')
        print(f'  {"N":>5s} {"w_barrier":>12s} {"u_barrier":>12s} {"eps_0":>12s}')
        print('  ' + '-' * 46)

        for mult in [1.0, 1.25, 1.5, 2.0]:
            N = max(15, round(mult * 6 * L_f))
            t0 = time.time()
            r = barrier_components(lam_sq, N=N)
            dt = time.time() - t0
            print(f'  {N:>5d} {r["w_barrier"]:>+12.8f} {r["u_barrier"]:>+12.8f} '
                  f'{r["eps_0"]:>12.4e}  ({dt:.0f}s)')

    # ── Part 2: Dense barrier sweep ──
    print('\n\n  PART 2: Dense barrier sweep (lam^2 = 100 to 20000)')
    print('  ' + '=' * 60)

    # Include some non-round numbers to test oscillation hypothesis
    lam_values = list(range(100, 1001, 100)) + list(range(1500, 5001, 500)) + \
                 list(range(6000, 10001, 1000)) + [12000, 15000, 20000]

    results = []
    print(f'\n  {"lam^2":>7s} {"L":>6s} {"w_bar":>10s} {"u_bar":>10s} '
          f'{"cross":>10s} {"det_2x2":>10s} {"eps_0":>12s}')
    print('  ' + '-' * 75)

    for lam_sq in lam_values:
        t0 = time.time()
        r = barrier_components(lam_sq)
        dt = time.time() - t0

        # 2x2 determinant on range(W02)
        det_2x2 = r['w_barrier'] * r['u_barrier'] - r['cross']**2

        r['det_2x2'] = det_2x2
        results.append(r)

        print(f'  {lam_sq:>7d} {r["L"]:>6.2f} {r["w_barrier"]:>+10.6f} '
              f'{r["u_barrier"]:>+10.6f} {r["cross"]:>+10.6f} '
              f'{det_2x2:>+10.6f} {r["eps_0"]:>12.4e}  ({dt:.0f}s)')

    # ── Part 3: Statistical analysis ──
    print('\n\n  PART 3: Statistical analysis')
    print('  ' + '=' * 60)

    w_bars = np.array([r['w_barrier'] for r in results])
    u_bars = np.array([r['u_barrier'] for r in results])
    Ls = np.array([r['L'] for r in results])

    print(f'\n  w_barrier: min={w_bars.min():.6f}  max={w_bars.max():.6f}  '
          f'mean={w_bars.mean():.6f}  std={w_bars.std():.6f}')
    print(f'  u_barrier: min={u_bars.min():.6f}  max={u_bars.max():.6f}  '
          f'mean={u_bars.mean():.6f}  std={u_bars.std():.6f}')

    # Is there a trend in w_barrier?
    # Linear fit: barrier = a * L + b
    c = np.polyfit(Ls, w_bars, 1)
    print(f'\n  Linear fit w_barrier = {c[0]:.6f} * L + {c[1]:.6f}')
    if c[0] < 0:
        L_zero = -c[1] / c[0]
        print(f'  => DECREASING. Projected zero at L={L_zero:.1f} (lam^2={np.exp(L_zero):.0f})')
    else:
        print(f'  => INCREASING or stable')

    # Better fit: barrier = a/L + b + c*L
    X = np.column_stack([1/Ls, np.ones_like(Ls), Ls])
    coeffs = np.linalg.lstsq(X, w_bars, rcond=None)[0]
    print(f'  Refined fit: barrier = {coeffs[0]:.4f}/L + {coeffs[1]:.6f} + {coeffs[2]:.6f}*L')

    residuals = w_bars - X @ coeffs
    print(f'  Residual std: {residuals.std():.6f}')

    # ── Part 4: Prime-jump analysis ──
    print('\n\n  PART 4: Barrier jumps at new primes')
    print('  ' + '=' * 60)
    print('  Testing: does barrier jump when lam^2 crosses a prime?')

    # Find primes near our sampled lambda^2 values
    from sympy import isprime, nextprime, prevprime

    for i in range(1, len(results)):
        lam_prev = results[i-1]['lam_sq']
        lam_curr = results[i]['lam_sq']
        bar_prev = results[i-1]['w_barrier']
        bar_curr = results[i]['w_barrier']
        delta = bar_curr - bar_prev

        # Count new primes in (lam_prev, lam_curr]
        new_primes = 0
        p = nextprime(lam_prev)
        while p <= lam_curr:
            new_primes += 1
            p = nextprime(p)

        if abs(delta) > 0.005 and new_primes > 0:
            print(f'  [{lam_prev:>7d} -> {lam_curr:>7d}] '
                  f'delta={delta:+.6f}  new_primes={new_primes}')

    # ── Part 5: Component decomposition at key points ──
    print('\n\n  PART 5: Component decomposition at key points')
    print('  ' + '=' * 60)

    for lam_sq in [200, 500, 1000, 2000, 5000, 10000, 20000]:
        r_list = [r for r in results if r['lam_sq'] == lam_sq]
        if not r_list:
            continue
        r = r_list[0]
        delta1 = abs(r['mprime']) - abs(r['w02'])
        delta2 = r['mdiag'] + r['malpha']
        print(f'  lam^2={lam_sq:>6d}: |Mp|-|W|={delta1:+.4f}  '
              f'Md+Ma={delta2:+.4f}  barrier={r["w_barrier"]:+.6f}')

    print('\n' + '#' * 70)
    print('  SESSION 41d COMPLETE')
    print('#' * 70)
