"""
SESSION 76e -- SCHUR MARGIN AT LARGE N

At N=41 (standard), the Schur margin for M_odd is 5.2e-7.
76c showed eigenvalue count grows with N. Critical question:
does M_odd REMAIN negative definite at large N, and what
happens to the coupling ratio / margin?

This is the RH bottleneck test at higher resolution.
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def schur_analysis(Mo):
    """Full Schur step 0 analysis."""
    a1 = Mo[0, 0]
    c = Mo[0, 1:]
    B = Mo[1:, 1:]

    B_evals, B_evecs = np.linalg.eigh(B)
    all_neg_B = np.all(B_evals < 0)

    if not all_neg_B:
        return {
            'a1': a1, 'coupling': float('nan'),
            'margin': float('nan'), 'ratio': float('nan'),
            'B_all_neg': False, 'max_B_eig': B_evals.max(),
        }

    # coupling = c^T (-B^{-1}) c
    Binv_c = np.linalg.solve(B, c)
    coupling = -float(c @ Binv_c)
    margin = abs(a1) - coupling
    ratio = coupling / abs(a1)

    # Decompose coupling into B-eigencomponents
    top_contrib = 0
    for k in range(len(B_evals)):
        proj = float(c @ B_evecs[:, k])
        contrib = -proj**2 / B_evals[k]
        if contrib > top_contrib:
            top_contrib = contrib

    return {
        'a1': a1, 'coupling': coupling,
        'margin': margin, 'ratio': ratio,
        'B_all_neg': True, 'max_B_eig': B_evals.max(),
        'top_single_contrib_pct': top_contrib / coupling * 100,
    }


def run():
    print()
    print('#' * 76)
    print('  SESSION 76e -- SCHUR MARGIN AT LARGE N')
    print('#' * 76)

    # ======================================================================
    # TEST 1: Schur margin vs N at fixed lam_sq
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 1: SCHUR MARGIN vs N')
    print(f'{"="*76}\n')

    for lam_sq in [200, 1000, 5000]:
        L = np.log(lam_sq)
        print(f'\n  lam^2 = {lam_sq} (L = {L:.3f}):')
        print(f'  {"N":>4} {"a_1":>12} {"coupling":>12} {"margin":>14} '
              f'{"ratio":>14} {"top1%":>8} {"B_neg?":>6}')
        print('  ' + '-' * 76)

        for N in [15, 20, 30, 41, 50, 60, 80, 100, 120, 150]:
            try:
                _, M, _ = build_all_fast(lam_sq, N)
                Mo = odd_block(M, N)
                s = schur_analysis(Mo)

                if s['B_all_neg']:
                    print(f'  {N:>4d} {s["a1"]:>+12.6f} {s["coupling"]:>12.6f} '
                          f'{s["margin"]:>+14.6e} {s["ratio"]:>14.10f} '
                          f'{s["top_single_contrib_pct"]:>8.2f} {"YES":>6}')
                else:
                    print(f'  {N:>4d} {s["a1"]:>+12.6f} {"B NOT NEG":>12} '
                          f'{"---":>14} {"---":>14} {"---":>8} '
                          f'{"NO":>6} max_B={s["max_B_eig"]:.4e}')
            except Exception as e:
                print(f'  {N:>4d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 2: M_odd eigenvalues at large N
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 2: M_ODD EIGENVALUE SPECTRUM vs N')
    print(f'{"="*76}\n')

    lam_sq = 1000
    L = np.log(lam_sq)

    print(f'  lam^2 = {lam_sq}:')
    print(f'  {"N":>4} {"dim_odd":>8} {"eig_max":>14} {"eig_2":>14} '
          f'{"#neg":>5} {"all_neg?":>8}')
    print('  ' + '-' * 60)

    for N in [15, 20, 30, 41, 50, 60, 80, 100, 120, 150]:
        try:
            _, M, _ = build_all_fast(lam_sq, N)
            Mo = odd_block(M, N)
            eo = np.linalg.eigvalsh(Mo)
            eig_max = eo.max()
            eig_2 = sorted(eo)[-2] if len(eo) >= 2 else 0
            n_neg = np.sum(eo < 0)
            all_neg = np.all(eo < 0)
            print(f'  {N:>4d} {N:>8d} {eig_max:>+14.6e} {eig_2:>+14.6e} '
                  f'{n_neg:>5d} {"YES" if all_neg else "NO":>8}')
        except Exception as e:
            print(f'  {N:>4d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 3: Schur margin vs lam_sq at large N
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 3: SCHUR MARGIN vs LAM_SQ (N=120)')
    print(f'{"="*76}\n')

    N = 120
    print(f'  N={N} fixed:')
    print(f'  {"lam^2":>8} {"L":>8} {"a_1":>12} {"coupling":>12} '
          f'{"margin":>14} {"ratio":>14}')
    print('  ' + '-' * 74)

    margins_data = []
    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        try:
            _, M, _ = build_all_fast(lam_sq, N)
            Mo = odd_block(M, N)
            s = schur_analysis(Mo)
            L = np.log(lam_sq)

            if s['B_all_neg']:
                margins_data.append((lam_sq, L, s['margin'], s['ratio']))
                print(f'  {lam_sq:>8d} {L:>8.3f} {s["a1"]:>+12.6f} '
                      f'{s["coupling"]:>12.6f} {s["margin"]:>+14.6e} '
                      f'{s["ratio"]:>14.10f}')
            else:
                print(f'  {lam_sq:>8d} {L:>8.3f} {s["a1"]:>+12.6f} '
                      f'{"B NOT NEG":>12} {"FAIL":>14}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # Fit margin scaling
    if len(margins_data) >= 3:
        print(f'\n  Margin scaling at N={N}:')
        Ls = np.array([d[1] for d in margins_data])
        ms = np.array([d[2] for d in margins_data])
        valid = ms > 0
        if np.sum(valid) >= 3:
            Ls_v = Ls[valid]
            log_m = np.log(ms[valid])

            # Power law
            fit_pow = np.polyfit(np.log(Ls_v), log_m, 1)
            print(f'    Power law: margin ~ {np.exp(fit_pow[1]):.4e} * L^{fit_pow[0]:.2f}')

            # Exponential
            fit_exp = np.polyfit(Ls_v, log_m, 1)
            print(f'    Exponential: margin ~ {np.exp(fit_exp[1]):.4e} * exp({fit_exp[0]:.4f}*L)')

            # Power * exponential
            A = np.column_stack([np.ones_like(Ls_v), np.log(Ls_v), Ls_v])
            fit_pe, _, _, _ = np.linalg.lstsq(A, log_m, rcond=None)
            resid_pe = np.std(log_m - A @ fit_pe)
            print(f'    Power*exp: margin ~ {np.exp(fit_pe[0]):.4e} * L^{fit_pe[1]:.2f} * '
                  f'exp({fit_pe[2]:.4f}*L)  [resid={resid_pe:.4f}]')
    sys.stdout.flush()

    # ======================================================================
    # TEST 4: Compare margin at standard N vs large N
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 4: MARGIN COMPARISON — STANDARD N vs LARGE N')
    print(f'{"="*76}\n')

    print(f'  {"lam^2":>8} {"N_std":>6} {"margin_std":>14} {"N=120":>6} '
          f'{"margin_120":>14} {"ratio":>10}')
    print('  ' + '-' * 66)

    for lam_sq in [200, 500, 1000, 2000, 5000, 10000]:
        L = np.log(lam_sq)
        N_std = max(15, round(6 * L))

        try:
            _, M_std, _ = build_all_fast(lam_sq, N_std)
            Mo_std = odd_block(M_std, N_std)
            s_std = schur_analysis(Mo_std)

            _, M_big, _ = build_all_fast(lam_sq, 120)
            Mo_big = odd_block(M_big, 120)
            s_big = schur_analysis(Mo_big)

            if s_std['B_all_neg'] and s_big['B_all_neg']:
                ratio = s_big['margin'] / s_std['margin'] if s_std['margin'] != 0 else float('inf')
                print(f'  {lam_sq:>8d} {N_std:>6d} {s_std["margin"]:>+14.6e} '
                      f'{120:>6d} {s_big["margin"]:>+14.6e} {ratio:>10.4f}')
            else:
                print(f'  {lam_sq:>8d} FAIL')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 5: The critical eigenvalue (smallest |eig| of M_odd) vs N
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 5: CRITICAL EIGENVALUE OF M_ODD vs N')
    print(f'{"="*76}\n')

    lam_sq = 1000
    print(f'  lam^2={lam_sq}: tracking the least-negative eigenvalue')
    print(f'  {"N":>4} {"eig_max(Mo)":>14} {"log10|eig_max|":>16}')
    print('  ' + '-' * 38)

    for N in [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150]:
        try:
            _, M, _ = build_all_fast(lam_sq, N)
            Mo = odd_block(M, N)
            eig_max = np.linalg.eigvalsh(Mo).max()
            log_abs = np.log10(abs(eig_max)) if eig_max != 0 else -16
            marker = ' <-- POSITIVE!' if eig_max > 0 else ''
            print(f'  {N:>4d} {eig_max:>+14.6e} {log_abs:>16.4f}{marker}')
        except Exception as e:
            print(f'  {N:>4d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 76e VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
