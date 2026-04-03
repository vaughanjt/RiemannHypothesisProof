"""
SESSION 42f — PRECISE SMOOTH BARRIER COMPUTATION

Compute smooth_barrier = smooth(W02-Mp) - M_diag - M_alpha at many L values.
This is the purely analytic quantity that must stay positive.

Uses:
- Vectorized W02-Mp from session41g (fast, includes all primes)
- Moving average for smoothing
- mpmath for M_diag and M_alpha at each point

This is the KEY computation: if smooth_barrier > 0 everywhere and
fluctuation amplitude < smooth_barrier, we have a proof.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, exp, cos, sin, hyp2f1, digamma, sinh
import time
import sys
import os

mp.dps = 25  # lower precision for speed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import compute_barrier_partial


def mdiag_malpha_fast(lam_sq, N=None, n_quad=4000):
    """Fast M_diag + M_alpha on w direction. Reduced precision."""
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

    # Alpha coefficients
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

    # Rayleigh quotients
    diag_vals = np.array([wr_diag[int(n)] for n in ns])
    mdiag = float(np.sum(w_hat**2 * diag_vals))

    alpha_arr = np.array([alpha[int(n)] for n in ns])
    malpha = 0.0
    for i in range(dim):
        for j in range(dim):
            if i != j:
                n, m = int(ns[i]), int(ns[j])
                malpha += w_hat[i] * (alpha[m] - alpha[n]) / (n - m) * w_hat[j]

    return mdiag, float(malpha)


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42f -- PRECISE SMOOTH BARRIER')
    print('#' * 70)

    # ── Part 1: Compute M_diag+M_alpha at many lambda values ──
    print('\n  PART 1: M_diag + M_alpha sweep')
    print('  ' + '=' * 60)

    # Key lambda values spanning the range
    lam_values = [200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000,
                  5000, 7000, 10000, 15000, 20000, 30000, 50000]

    md_ma_results = []
    print(f'\n  {"lam^2":>7s} {"L":>6s} {"M_diag":>10s} {"M_alpha":>10s} '
          f'{"sum":>10s} {"time":>6s}')
    print('  ' + '-' * 55)

    for lam_sq in lam_values:
        t0 = time.time()
        md, ma = mdiag_malpha_fast(lam_sq)
        dt = time.time() - t0
        L_f = np.log(lam_sq)
        md_ma_results.append({
            'lam_sq': lam_sq, 'L': L_f,
            'mdiag': md, 'malpha': ma, 'sum': md + ma,
        })
        print(f'  {lam_sq:>7d} {L_f:>6.2f} {md:>+10.6f} {ma:>+10.6f} '
              f'{md+ma:>+10.6f} {dt:>5.0f}s')
        sys.stdout.flush()

    # ── Part 2: W02-Mp smooth envelope ──
    print('\n\n  PART 2: Smooth W02-Mp envelope')
    print('  ' + '=' * 60)

    # Dense W02-Mp computation
    dense_lam = list(range(100, 10001, 100))
    dense_results = []
    for lam_sq in dense_lam:
        r = compute_barrier_partial(lam_sq)
        dense_results.append({'lam_sq': lam_sq, 'L': r['L'], 'w02_mp': r['partial_barrier']})

    # Also get W02-Mp at larger lambda (from 41g data, recomputed)
    for lam_sq in [12000, 15000, 20000, 30000, 50000]:
        r = compute_barrier_partial(lam_sq)
        dense_results.append({'lam_sq': lam_sq, 'L': r['L'], 'w02_mp': r['partial_barrier']})

    dense_results.sort(key=lambda x: x['lam_sq'])
    dense_Ls = np.array([r['L'] for r in dense_results])
    dense_w02mp = np.array([r['w02_mp'] for r in dense_results])

    # Smooth with window of 20 (in the dense part)
    window = 20
    smooth = np.convolve(dense_w02mp[:len(dense_lam)],
                         np.ones(window)/window, mode='valid')
    smooth_L = dense_Ls[window//2:window//2+len(smooth)]

    # Extend smooth to larger L by fitting
    # Fit: smooth = a + b*L + c/L
    if len(smooth) >= 10:
        X = np.column_stack([np.ones_like(smooth_L), smooth_L, 1/smooth_L])
        c_fit = np.linalg.lstsq(X, smooth, rcond=None)[0]

        # Evaluate at all L values including large ones
        all_Ls = np.array([r['L'] for r in md_ma_results])
        smooth_at_md = c_fit[0] + c_fit[1]*all_Ls + c_fit[2]/all_Ls

        print(f'  Smooth fit: W02-Mp ~ {c_fit[0]:.4f} + {c_fit[1]:.4f}*L + {c_fit[2]:.4f}/L')

    # ── Part 3: SMOOTH BARRIER = smooth(W02-Mp) - M_diag - M_alpha ──
    print('\n\n  PART 3: SMOOTH BARRIER')
    print('  ' + '=' * 60)

    print(f'\n  {"lam^2":>7s} {"L":>6s} {"smooth_W-M":>12s} {"Md+Ma":>10s} '
          f'{"SMOOTH_BAR":>12s}')
    print('  ' + '-' * 55)

    smooth_barriers = []
    for i, r in enumerate(md_ma_results):
        L = r['L']
        md_ma = r['sum']

        # Get smooth W02-Mp (from fit or direct interpolation)
        if L <= smooth_L[-1]:
            sw = float(np.interp(L, smooth_L, smooth))
        else:
            sw = float(c_fit[0] + c_fit[1]*L + c_fit[2]/L)

        sb = sw - md_ma
        smooth_barriers.append({**r, 'smooth_w02mp': sw, 'smooth_barrier': sb})
        marker = ' <<<' if sb < 0 else ''
        print(f'  {r["lam_sq"]:>7d} {L:>6.2f} {sw:>+12.6f} {md_ma:>+10.6f} '
              f'{sb:>+12.6f}{marker}')

    # ── Part 4: Smooth barrier trend ──
    print('\n\n  PART 4: Smooth barrier trend analysis')
    print('  ' + '=' * 60)

    sb_Ls = np.array([r['L'] for r in smooth_barriers])
    sb_vals = np.array([r['smooth_barrier'] for r in smooth_barriers])

    print(f'  Range: [{sb_vals.min():.6f}, {sb_vals.max():.6f}]')
    print(f'  Mean:  {sb_vals.mean():.6f}')
    print(f'  All positive? {np.all(sb_vals > 0)}')

    # Fit: sb = a + b/L
    if len(sb_Ls) >= 4:
        X = np.column_stack([np.ones_like(sb_Ls), 1/sb_Ls])
        c_sb = np.linalg.lstsq(X, sb_vals, rcond=None)[0]
        print(f'  Fit: smooth_barrier = {c_sb[0]:.6f} + {c_sb[1]:.4f}/L')
        print(f'  Limit (L->inf): {c_sb[0]:.6f}')

        # Predict
        for L_val in [12, 15, 20, 50]:
            pred = c_sb[0] + c_sb[1]/L_val
            print(f'    L={L_val:>2d}: predicted smooth barrier = {pred:.6f}')

    # ── Part 5: Full barrier decomposition ──
    print('\n\n  PART 5: Complete decomposition at all lambda')
    print('  ' + '=' * 60)

    # For each lambda where we have M_diag+M_alpha, also compute actual barrier
    print(f'\n  {"lam^2":>7s} {"L":>6s} {"actual_bar":>12s} {"smooth_bar":>12s} '
          f'{"fluctuation":>12s} {"ratio":>8s}')
    print('  ' + '-' * 65)

    for r in smooth_barriers:
        lam_sq = r['lam_sq']
        rp = compute_barrier_partial(lam_sq)
        actual_barrier = rp['partial_barrier'] - r['sum']  # W02-Mp - Md-Ma
        fluct = actual_barrier - r['smooth_barrier']
        ratio = abs(fluct) / abs(r['smooth_barrier']) if abs(r['smooth_barrier']) > 1e-10 else float('inf')
        print(f'  {lam_sq:>7d} {r["L"]:>6.2f} {actual_barrier:>+12.6f} '
              f'{r["smooth_barrier"]:>+12.6f} {fluct:>+12.6f} {ratio:>8.4f}')

    print('\n' + '#' * 70)
    print('  SESSION 42f COMPLETE')
    print('#' * 70)
