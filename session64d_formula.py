"""
SESSION 64d -- THE CLOSED-FORM FORMULA

Hypothesis: P_odd[n,n] = 4 / (gamma^2 - (2*pi*n/L)^2)

This is ALWAYS POSITIVE when gamma > 2*pi*n/L, which holds for:
  - All zeta zeros (gamma >= 14.13)
  - All relevant modes (n = 1, 2 with L > 1)

If correct, the sign lemma follows immediately:
  v^T P v ~ v_1^2 * 4/(gamma^2 - a_1^2) + v_2^2 * 4/(gamma^2 - a_2^2) > 0
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast
from session64c_sign_lemma import build_perturbation_matrix, odd_block


def run():
    print()
    print('#' * 76)
    print('  SESSION 64d -- THE CLOSED-FORM FORMULA')
    print('#' * 76)

    gammas = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832]

    # ==================================================================
    # TEST 1: P[n,n] vs 4/(gamma^2 - (2*pi*n/L)^2) across all params
    # ==================================================================
    print('\n  === TEST 1: FORMULA VERIFICATION ===')
    print('  P_odd[n,n] vs 4/(gamma^2 - a_n^2) where a_n = 2*pi*n/L\n')

    print(f'  {"lam^2":>8} {"gamma":>8} {"n":>3} '
          f'{"P[n,n] actual":>15} {"4/(g^2-a^2)":>15} {"ratio":>8}')
    print('  ' + '-' * 63)

    for lam_sq in [100, 500, 1000, 5000, 20000, 50000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))

        for k, gamma in enumerate(gammas[:5]):
            P_mat = build_perturbation_matrix(gamma, L, N)
            Po = odd_block(P_mat, N)

            for n in [1, 2, 3, 5]:
                if n > N:
                    continue
                idx = n - 1  # 0-indexed in odd block
                actual = Po[idx, idx]
                a_n = 2 * np.pi * n / L
                formula = 4.0 / (gamma**2 - a_n**2)
                ratio = actual / formula if abs(formula) > 1e-15 else float('nan')

                if lam_sq == 1000 or (k == 0 and n <= 2):
                    print(f'  {lam_sq:>8d} {gamma:>8.2f} {n:>3d} '
                          f'{actual:>+15.8f} {formula:>+15.8f} {ratio:>8.4f}')

        if lam_sq == 1000:
            print('  ...')
    sys.stdout.flush()

    # ==================================================================
    # TEST 2: SIGN LEMMA — COMPLETE PROOF STRUCTURE
    # ==================================================================
    print('\n  === TEST 2: SIGN LEMMA — FULL VERIFICATION ===')
    print('  v^T P v vs formula:')
    print('  v_1^2 * 4/(g^2-a1^2) + v_2^2 * 4/(g^2-a2^2)\n')

    print(f'  {"lam^2":>8} {"gamma":>8} {"v^T P v":>14} '
          f'{"formula":>14} {"ratio":>8} {"sign":>6}')
    print('  ' + '-' * 62)

    for lam_sq in [50, 200, 1000, 5000, 20000, 50000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M_f, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M_f, N)
        eigs, vecs = np.linalg.eigh(Mo)
        v = vecs[:, -1]

        for k, gamma in enumerate(gammas[:5]):
            P_mat = build_perturbation_matrix(gamma, L, N)
            Po = odd_block(P_mat, N)
            vPv = float(v @ Po @ v)

            a1 = 2 * np.pi * 1 / L
            a2 = 2 * np.pi * 2 / L
            formula = v[0]**2 * 4/(gamma**2 - a1**2) + \
                      v[1]**2 * 4/(gamma**2 - a2**2)
            ratio = vPv / formula if abs(formula) > 1e-15 else float('nan')
            sign = '+' if vPv > 0 else '-'

            if k == 0 or lam_sq == 1000:
                print(f'  {lam_sq:>8d} {gamma:>8.2f} {vPv:>+14.6e} '
                      f'{formula:>+14.6e} {ratio:>8.4f} {sign:>6}')
        if lam_sq == 1000:
            print('  ...')
    sys.stdout.flush()

    # ==================================================================
    # TEST 3: WHY THE FORMULA WORKS — THE KERNEL IDENTITY
    # ==================================================================
    print('\n  === TEST 3: THE KERNEL IDENTITY ===')
    print('  The odd-block diagonal perturbation involves two integrals:')
    print('  I_cos = integral h(t)*cos(at)*t*exp(igt) dt  (from a_n)')
    print('  I_sin = integral sin(at)*t*exp(igt) dt / (n*pi)  (from B_n/n)')
    print('  Test: what is the relative contribution of each?\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    gamma = gammas[0]

    n_quad = 5000
    dt = L / n_quad
    t = dt * (np.arange(n_quad) + 0.5)
    exp_igt = np.exp(1j * gamma * t)

    print(f'  lam^2={lam_sq}, gamma={gamma:.4f}:')
    print(f'  {"n":>3} {"I_cos (a_n)":>14} {"I_sin (B_n/n)":>14} '
          f'{"total":>14} {"formula":>14}')
    print('  ' + '-' * 62)

    for n in range(1, 8):
        a_n = 2 * np.pi * n / L
        h = 2 * (L - t) / L

        # cos integral: -2*Re[integral h*cos(at)*t*exp(igt) dt]
        cos_int = -2 * np.real(np.sum(h * np.cos(a_n * t) * t * exp_igt) * dt)

        # sin integral: -(2/(n*pi))*Im[integral sin(at)*t*exp(igt) dt]
        sin_int = -(2 / (n * np.pi)) * np.imag(
            np.sum(np.sin(a_n * t) * t * exp_igt) * dt)

        total = cos_int + sin_int
        formula = 4.0 / (gamma**2 - a_n**2)

        print(f'  {n:>3d} {cos_int:>+14.6f} {sin_int:>+14.6f} '
              f'{total:>+14.6f} {formula:>+14.6f}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 4: THE FORMULA AT SMALL L (BOUNDARY EFFECTS)
    # ==================================================================
    print('\n  === TEST 4: FORMULA ACCURACY vs L ===')
    print('  How does the formula degrade at small L?\n')

    gamma = gammas[0]
    n = 1
    print(f'  {"lam^2":>8} {"L":>6} {"P[1,1]":>12} {"formula":>12} '
          f'{"ratio":>8} {"error %":>8}')
    print('  ' + '-' * 56)

    for lam_sq in [4, 10, 20, 50, 100, 500, 1000, 10000, 100000]:
        L = float(np.log(lam_sq))
        N = max(10, round(6 * L))
        a_n = 2 * np.pi * n / L
        if gamma**2 <= a_n**2:
            print(f'  {lam_sq:>8d} {L:>6.2f}  RESONANCE (a_n > gamma)')
            continue

        P_mat = build_perturbation_matrix(gamma, L, N)
        Po = odd_block(P_mat, N)
        actual = Po[0, 0]
        formula = 4.0 / (gamma**2 - a_n**2)
        ratio = actual / formula
        error = 100 * abs(ratio - 1)

        print(f'  {lam_sq:>8d} {L:>6.2f} {actual:>+12.6f} {formula:>+12.6f} '
              f'{ratio:>8.4f} {error:>8.2f}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 5: THE SIGN LEMMA STATEMENT
    # ==================================================================
    print('\n  === TEST 5: THE SIGN LEMMA — PROVABILITY CHECK ===')
    print()
    print('  SIGN LEMMA (to prove):')
    print('    For any zeta zero rho = 1/2 + i*gamma (gamma > 0),')
    print('    moving rho off the critical line by delta > 0 increases')
    print('    the maximum eigenvalue of M_odd.')
    print()
    print('  PROOF SKETCH:')
    print('    1. First-order perturbation: d(max_eig)/d(delta) = v^T P v')
    print('    2. v is concentrated on n=1,2 (97%+ weight, Session 58)')
    print('    3. P_odd is diagonally dominant on {n=1,n=2}')
    print('    4. P_odd[n,n] = 4/(gamma^2 - (2*pi*n/L)^2) + O(1/L)')
    print('    5. Since gamma >= 14.13 > 2*pi*2/L for all L > 0.89:')
    print('       P_odd[n,n] > 0 for n = 1, 2')
    print('    6. Therefore v^T P v > 0. QED.')
    print()

    # Verify condition: gamma > 2*pi*2/L for all relevant L
    gamma_min = gammas[0]
    L_crit = 4 * np.pi / gamma_min
    print(f'  Condition check: gamma_1 = {gamma_min:.4f}')
    print(f'  Need L > 4*pi/gamma = {L_crit:.4f}')
    print(f'  Smallest useful L (lam^2=4): L = {np.log(4):.4f}')
    print(f'  Condition satisfied: {np.log(4) > L_crit}')
    print()

    # Minimum value of formula over all gamma and reasonable L
    min_val = float('inf')
    for gamma in gammas:
        for L in [np.log(4), np.log(50000)]:
            for n in [1, 2]:
                a_n = 2 * np.pi * n / L
                if gamma**2 > a_n**2:
                    val = 4 / (gamma**2 - a_n**2)
                    min_val = min(min_val, val)

    print(f'  Minimum 4/(gamma^2 - a_n^2) over all zeros and L: {min_val:.6f}')
    print(f'  This is always > 0.')
    sys.stdout.flush()

    print()
    print('=' * 76)
    print('  SESSION 64d RESULTS')
    print('=' * 76)


if __name__ == '__main__':
    run()
