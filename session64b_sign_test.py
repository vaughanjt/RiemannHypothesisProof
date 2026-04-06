"""
SESSION 64b -- SIGN CONSISTENCY TEST

Session 64 showed moving zero 1 off-line flips the margin.
Does this hold for ALL zeros? And at all lambda?

If the sign is UNIVERSAL (off-line always increases max_eig),
then the asymptotic proof reduces to a "sign lemma."
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast
from session64_asymptotic_test import zero_contribution_matrix, odd_block, schur_margin


def run():
    print()
    print('#' * 76)
    print('  SESSION 64b -- SIGN CONSISTENCY TEST')
    print('#' * 76)

    gammas = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832]

    # ==================================================================
    # TEST 1: Each zero individually at delta=0.01, multiple lambda
    # ==================================================================
    print('\n  === TEST 1: EACH ZERO, EACH LAMBDA ===')
    print('  Move zero k off-line by delta=0.01. Track max_eig change.\n')

    delta = 0.01

    print(f'  {"lam^2":>8} | ', end='')
    for k in range(10):
        print(f'{"z"+str(k+1):>8}', end=' ')
    print()
    print('  ' + '-' * 100)

    for lam_sq in [200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M_f, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M_f, N)
        max_eig_base = np.linalg.eigvalsh(Mo)[-1]

        shifts = []
        for k, gamma in enumerate(gammas):
            rho_on = 0.5 + 1j * gamma
            rho_off = 0.5 + delta + 1j * gamma
            dM_on = zero_contribution_matrix(rho_on, L, N)
            dM_off = zero_contribution_matrix(rho_off, L, N)
            shift_M = odd_block(dM_off, N) - odd_block(dM_on, N)
            Mo_pert = Mo + shift_M
            max_eig_pert = np.linalg.eigvalsh(Mo_pert)[-1]
            shift_eig = max_eig_pert - max_eig_base
            shifts.append(shift_eig)

        print(f'  {lam_sq:>8d} | ', end='')
        for s in shifts:
            sign = '+' if s > 0 else '-'
            print(f'{sign}{abs(s):>7.4f}', end=' ')
        print()
    sys.stdout.flush()

    # ==================================================================
    # TEST 2: Negative delta (moving zero TOWARD critical line)
    # ==================================================================
    print('\n  === TEST 2: DELTA < 0 (closer to line) ===')
    print('  If we could move zeros closer to the line, does max_eig decrease?')
    print('  (This is hypothetical -- tests the monotonicity.)\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_f, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M_f, N)
    max_eig_base = np.linalg.eigvalsh(Mo)[-1]

    gamma = gammas[0]
    print(f'  Zero 1 (gamma={gamma:.4f}) at lam^2={lam_sq}:')
    print(f'  {"delta":>8} {"max_eig":>14} {"shift":>14}')
    print('  ' + '-' * 40)

    for delta in [-0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01]:
        rho_on = 0.5 + 1j * gamma
        rho_test = 0.5 + delta + 1j * gamma
        dM_on = zero_contribution_matrix(rho_on, L, N)
        dM_test = zero_contribution_matrix(rho_test, L, N)
        shift_M = odd_block(dM_test, N) - odd_block(dM_on, N)
        Mo_pert = Mo + shift_M
        me = np.linalg.eigvalsh(Mo_pert)[-1]
        print(f'  {delta:>+8.3f} {me:>+14.6e} {me - max_eig_base:>+14.6e}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 3: Very small delta -- find the critical delta
    # ==================================================================
    print('\n  === TEST 3: CRITICAL DELTA -- SMALLEST THAT FLIPS SIGN ===')
    print('  Bisect to find the delta where max_eig crosses zero.\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_f, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M_f, N)

    gamma = gammas[0]
    rho_on = 0.5 + 1j * gamma
    dM_on = zero_contribution_matrix(rho_on, L, N)
    dMo_on = odd_block(dM_on, N)

    # Bisect delta in [0, 0.01]
    d_lo, d_hi = 0.0, 0.001
    for _ in range(50):
        d_mid = (d_lo + d_hi) / 2
        rho_test = 0.5 + d_mid + 1j * gamma
        dM_test = zero_contribution_matrix(rho_test, L, N)
        dMo_test = odd_block(dM_test, N)
        Mo_pert = Mo + dMo_test - dMo_on
        me = np.linalg.eigvalsh(Mo_pert)[-1]
        if me > 0:
            d_hi = d_mid
        else:
            d_lo = d_mid

    print(f'  Zero 1 at lam^2={lam_sq}:')
    print(f'  Critical delta = {(d_lo+d_hi)/2:.8e}')
    print(f'  (Moving zero 1 off-line by this amount flips the sign)')
    sys.stdout.flush()

    # At different lambda
    print(f'\n  Critical delta vs lambda:')
    print(f'  {"lam^2":>8} {"L":>6} {"delta_crit":>14}')
    print('  ' + '-' * 32)

    for lam_sq in [50, 200, 1000, 5000, 20000, 50000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M_f, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M_f, N)

        rho_on = 0.5 + 1j * gamma
        dM_on = zero_contribution_matrix(rho_on, L, N)
        dMo_on = odd_block(dM_on, N)

        d_lo, d_hi = 0.0, 0.01
        for _ in range(50):
            d_mid = (d_lo + d_hi) / 2
            rho_test = 0.5 + d_mid + 1j * gamma
            dM_test = zero_contribution_matrix(rho_test, L, N)
            dMo_test = odd_block(dM_test, N)
            Mo_pert = Mo + dMo_test - dMo_on
            me = np.linalg.eigvalsh(Mo_pert)[-1]
            if me > 0:
                d_hi = d_mid
            else:
                d_lo = d_mid

        dcrit = (d_lo + d_hi) / 2
        print(f'  {lam_sq:>8d} {L:>6.2f} {dcrit:>14.8e}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 64b VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
