"""
SESSION 41b — LARGE-SCALE BARRIER TRACKING

Push the barrier computation to larger lambda^2 to determine:
1. Does the barrier converge to a positive constant?
2. Does it approach zero?
3. What are the precise growth rates of |M_prime|-|W02| vs M_diag+M_alpha?

Also: investigate N-convergence (are we using enough basis functions?)
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, exp, cos, sin, hyp2f1, digamma, sinh
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def barrier_fast(lam_sq, N=None, n_quad=8000):
    """
    Compute barrier on odd eigenvector of W02. Optimized version.
    Returns (barrier, components_dict).
    """
    L_f = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N, n_quad=n_quad)
    M_diag, M_alpha, M_prime, M_full, vM = compute_M_decomposition(lam_sq, N, n_quad=n_quad)

    # Build analytic odd eigenvector of W02
    ns = np.arange(-N, N + 1, dtype=float)
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N] = 0.0  # n=0
    w_norm = np.linalg.norm(w_vec)
    w_hat = w_vec / w_norm

    # Rayleigh quotients
    w02_term = w_hat @ W02 @ w_hat
    mprime_term = w_hat @ M_prime @ w_hat
    mdiag_term = w_hat @ M_diag @ w_hat
    malpha_term = w_hat @ M_alpha @ w_hat
    barrier = w_hat @ QW @ w_hat

    # Also check: use eigenvector from W02 directly (may differ at small N)
    ew, ev = np.linalg.eigh(W02)
    nz_idx = np.where(np.abs(ew) > np.max(np.abs(ew)) * 1e-10)[0]
    for idx in nz_idx:
        v = ev[:, idx]
        odd_err = sum(abs(v[N + k] + v[N - k]) for k in range(1, N + 1))
        even_err = sum(abs(v[N + k] - v[N - k]) for k in range(1, N + 1))
        if odd_err < even_err:
            v_hat = v / np.linalg.norm(v)
            barrier_eig = v_hat @ QW @ v_hat
            alignment = abs(np.dot(w_hat, v_hat))
            break
    else:
        barrier_eig = barrier
        alignment = 1.0

    # Full eps_0
    eps_0 = np.linalg.eigvalsh(QW)[0]

    return {
        'lam_sq': lam_sq,
        'L': L_f,
        'N': N,
        'w02': w02_term,
        'mprime': mprime_term,
        'mdiag': mdiag_term,
        'malpha': malpha_term,
        'barrier': barrier,
        'barrier_eig': barrier_eig,
        'alignment': alignment,
        'delta1': abs(mprime_term) - abs(w02_term),  # |M_prime| - |W02|
        'delta2': mdiag_term + malpha_term,           # M_diag + M_alpha
        'eps_0': eps_0,
    }


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 41b — LARGE-SCALE BARRIER TRACKING')
    print('#' * 70)

    # ── Part 1: N-convergence study ──
    print('\n  PART 1: N-convergence at fixed lambda^2')
    print('  ' + '=' * 60)

    for lam_sq in [200, 1000, 5000]:
        L_f = np.log(lam_sq)
        N_default = max(15, round(6 * L_f))
        print(f'\n  lam^2 = {lam_sq}, L = {L_f:.3f}, default N = {N_default}')
        print(f'  {"N":>5s} {"dim":>5s} {"barrier":>12s} {"barrier_eig":>12s} '
              f'{"alignment":>10s} {"delta1":>10s} {"delta2":>10s}')
        print('  ' + '-' * 70)

        for N_mult in [1.0, 1.5, 2.0]:
            N = max(15, round(N_mult * 6 * L_f))
            t0 = time.time()
            r = barrier_fast(lam_sq, N=N, n_quad=8000)
            dt = time.time() - t0
            print(f'  {r["N"]:>5d} {2*r["N"]+1:>5d} {r["barrier"]:>+12.8f} '
                  f'{r["barrier_eig"]:>+12.8f} {r["alignment"]:>10.8f} '
                  f'{r["delta1"]:>+10.6f} {r["delta2"]:>+10.6f}  ({dt:.0f}s)')

    # ── Part 2: Large lambda^2 sweep ──
    print('\n\n  PART 2: Barrier at large lambda^2')
    print('  ' + '=' * 60)

    lam_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    results = []

    print(f'\n  {"lam^2":>7s} {"L":>6s} {"N":>4s} {"barrier":>12s} '
          f'{"delta1":>10s} {"delta2":>10s} {"delta1-2":>10s} {"eps_0":>12s} {"time":>6s}')
    print('  ' + '-' * 85)

    for lam_sq in lam_values:
        t0 = time.time()
        r = barrier_fast(lam_sq)
        dt = time.time() - t0
        results.append(r)
        print(f'  {lam_sq:>7d} {r["L"]:>6.2f} {r["N"]:>4d} {r["barrier"]:>+12.8f} '
              f'{r["delta1"]:>+10.6f} {r["delta2"]:>+10.6f} '
              f'{r["delta1"]-r["delta2"]:>+10.6f} {r["eps_0"]:>12.4e} {dt:>5.0f}s')

    # ── Part 3: Analyze delta1 and delta2 growth ──
    print('\n\n  PART 3: Growth analysis of delta1 = |M_prime|-|W02| and delta2 = M_diag+M_alpha')
    print('  ' + '=' * 60)

    Ls = np.array([r['L'] for r in results])
    d1s = np.array([r['delta1'] for r in results])
    d2s = np.array([r['delta2'] for r in results])
    bars = np.array([r['barrier'] for r in results])

    # Fit delta1 = a*L + b
    c1 = np.polyfit(Ls, d1s, 1)
    # Fit delta2 = a*L + b
    c2 = np.polyfit(Ls, d2s, 1)

    print(f'\n  Linear fit delta1 = {c1[0]:.6f} * L + {c1[1]:.6f}')
    print(f'  Linear fit delta2 = {c2[0]:.6f} * L + {c2[1]:.6f}')
    print(f'  Slope difference: {c1[0] - c2[0]:.6f}')
    print(f'  Intercept difference: {c1[1] - c2[1]:.6f}')

    if c1[0] > c2[0]:
        print(f'  => delta1 grows FASTER: barrier increases with L')
    elif c1[0] < c2[0]:
        # Projected crossing
        if (c1[1] - c2[1]) > 0:
            L_cross = (c2[1] - c1[1]) / (c1[0] - c2[0])
            print(f'  => delta2 grows FASTER: barrier DECREASES')
            print(f'  => Projected crossing at L = {L_cross:.2f} (lam^2 = {np.exp(L_cross):.0f})')
        else:
            print(f'  => delta2 grows FASTER and already leads: barrier negative')

    # Also try: delta1 = a*L + b*log(L) + c (better fit?)
    # Use basis [L, log(L), 1]
    X = np.column_stack([Ls, np.log(Ls), np.ones_like(Ls)])
    c1_ext = np.linalg.lstsq(X, d1s, rcond=None)[0]
    c2_ext = np.linalg.lstsq(X, d2s, rcond=None)[0]

    print(f'\n  Extended fit delta1 = {c1_ext[0]:.6f}*L + {c1_ext[1]:.6f}*log(L) + {c1_ext[2]:.6f}')
    print(f'  Extended fit delta2 = {c2_ext[0]:.6f}*L + {c2_ext[1]:.6f}*log(L) + {c2_ext[2]:.6f}')
    print(f'  Barrier ≈ {c1_ext[0]-c2_ext[0]:.6f}*L + {c1_ext[1]-c2_ext[1]:.6f}*log(L) + {c1_ext[2]-c2_ext[2]:.6f}')

    # Residuals
    d1_pred = X @ c1_ext
    d2_pred = X @ c2_ext
    bar_pred = d1_pred - d2_pred

    print(f'\n  Fit quality (residuals):')
    print(f'  {"lam^2":>7s} {"barrier":>10s} {"predicted":>10s} {"residual":>10s}')
    print('  ' + '-' * 42)
    for i, r in enumerate(results):
        print(f'  {r["lam_sq"]:>7d} {r["barrier"]:>+10.6f} {bar_pred[i]:>+10.6f} '
              f'{r["barrier"]-bar_pred[i]:>+10.6f}')

    # ── Part 4: Component-by-component analytics ──
    print('\n\n  PART 4: Individual component vs L')
    print('  ' + '=' * 60)

    # M_diag grows like what?
    mdiags = np.array([r['mdiag'] for r in results])
    malphas = np.array([r['malpha'] for r in results])

    # Fit M_diag = a*L + b
    c_md = np.polyfit(Ls, mdiags, 1)
    c_ma = np.polyfit(Ls, malphas, 1)

    print(f'  M_diag  ≈ {c_md[0]:.6f} * L + {c_md[1]:.6f}')
    print(f'  M_alpha ≈ {c_ma[0]:.6f} * L + {c_ma[1]:.6f}')

    # W02 and M_prime individually
    w02s = np.array([r['w02'] for r in results])
    mps = np.array([r['mprime'] for r in results])

    # These grow roughly as e^{L/2}, so fit log|x| = aL + b
    c_w02 = np.polyfit(Ls, np.log(np.abs(w02s)), 1)
    c_mp = np.polyfit(Ls, np.log(np.abs(mps)), 1)

    print(f'\n  |W02 term|  ≈ exp({c_w02[0]:.6f} * L + {c_w02[1]:.6f})')
    print(f'  |M_prime|   ≈ exp({c_mp[0]:.6f} * L + {c_mp[1]:.6f})')
    print(f'  Slope ratio: {c_w02[0]/c_mp[0]:.8f}')
    print(f'  At large L, the difference |M_prime|-|W02| comes from the exponent difference')

    # ── Part 5: Barrier on even direction ──
    print('\n\n  PART 5: Even (u) direction barrier')
    print('  ' + '=' * 60)

    u_results = []
    for lam_sq in [100, 500, 1000, 5000, 10000]:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
        dim = 2 * N + 1
        W02, M, QW = build_all(lam_sq, N, n_quad=8000)

        ns = np.arange(-N, N + 1, dtype=float)
        u_vec = 1.0 / (L_f**2 + (4 * np.pi)**2 * ns**2)
        u_hat = u_vec / np.linalg.norm(u_vec)
        u_bar = u_hat @ QW @ u_hat

        M_diag, M_alpha, M_prime, M_full, vM = compute_M_decomposition(lam_sq, N, n_quad=8000)
        u_w02 = u_hat @ W02 @ u_hat
        u_mp = u_hat @ M_prime @ u_hat
        u_md = u_hat @ M_diag @ u_hat
        u_ma = u_hat @ M_alpha @ u_hat

        u_results.append({
            'lam_sq': lam_sq, 'L': L_f, 'barrier': u_bar,
            'w02': u_w02, 'mprime': u_mp, 'mdiag': u_md, 'malpha': u_ma,
        })

    print(f'  {"lam^2":>7s} {"L":>6s} {"W02":>10s} {"M_prime":>10s} '
          f'{"M_diag":>10s} {"M_alpha":>10s} {"Barrier":>10s}')
    print('  ' + '-' * 65)
    for r in u_results:
        print(f'  {r["lam_sq"]:>7d} {r["L"]:>6.2f} {r["w02"]:>+10.4f} '
              f'{r["mprime"]:>+10.4f} {r["mdiag"]:>+10.4f} '
              f'{r["malpha"]:>+10.4f} {r["barrier"]:>+10.6f}')

    print('\n' + '#' * 70)
    print('  SESSION 41b COMPLETE')
    print('#' * 70)
