"""
SESSION 64e -- COLLECTIVE CANCELLATION TEST

Can we bound the total perturbation from ALL zeros above verified height T?

The sign lemma gives: each zero at (1/2 + delta + i*gamma) shifts max_eig by
approximately delta * 4/gamma^2 (first order).

For zeros above T = 10^12 (verified on-line): the VK zero-free region bounds
delta < 1/2 - c/(log gamma)^{2/3}(log log gamma)^{1/3}.

KEY QUESTION: is Sum_{gamma > T} |perturbation| < margin(L)?

Also test NON-PERTURBATIVE: actual contribution from zeros at high gamma.
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast
from session64_asymptotic_test import zero_contribution_matrix, odd_block, schur_margin


def run():
    print()
    print('#' * 76)
    print('  SESSION 64e -- COLLECTIVE CANCELLATION')
    print('#' * 76)

    # ==================================================================
    # PART 1: v^T P(gamma) v DECAY WITH GAMMA
    # ==================================================================
    print('\n  === PART 1: HOW FAST DOES v^T P v DECAY WITH GAMMA? ===')
    print('  This determines whether the tail sum converges.\n')

    gammas = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832]

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_f, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M_f, N)
    eigs, vecs = np.linalg.eigh(Mo)
    v = vecs[:, -1]

    print(f'  {"gamma":>10} {"v^T P v":>14} {"4/gamma^2":>14} {"ratio":>8}')
    print('  ' + '-' * 50)

    for gamma in gammas:
        from session64c_sign_lemma import build_perturbation_matrix
        P = build_perturbation_matrix(gamma, L, N)
        Po = odd_block(P, N)
        vPv = float(v @ Po @ v)
        formula = 4.0 / gamma**2
        ratio = vPv / formula
        print(f'  {gamma:>10.4f} {vPv:>+14.6e} {formula:>14.6e} {ratio:>8.4f}')

    # Extend to higher gammas using the formula
    print(f'\n  Extrapolation using formula 4/gamma^2:')
    print(f'  {"gamma":>10} {"4/gamma^2":>14}')
    for gamma in [100, 500, 1000, 10000, 1e6, 1e12]:
        print(f'  {gamma:>10.0f} {4/gamma**2:>14.6e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: TAIL SUM BOUND
    # ==================================================================
    print('\n  === PART 2: TAIL SUM BOUND ===')
    print('  Sum_{gamma > T} delta_max * 4/gamma^2 * (zero density)')
    print('  where delta_max comes from VK zero-free region.\n')

    c_VK = 0.05  # VK constant (approximate)

    def vk_delta_max(gamma):
        """Maximum delta allowed by VK zero-free region at height gamma."""
        logg = np.log(gamma + 2)
        loglogg = np.log(logg + 1)
        return 0.5 - c_VK / (logg**(2/3) * loglogg**(1/3))

    def zero_density(gamma):
        """Approximate zero density dN/dgamma ~ log(gamma)/(2*pi)."""
        return np.log(gamma + 2) / (2 * np.pi)

    # First-order tail bound: integral_T^inf delta_max(g) * 4/g^2 * density(g) dg
    print(f'  First-order perturbative tail bound:')
    print(f'  {"T":>12} {"tail_bound":>14} {"margin(L=12)":>14} '
          f'{"ratio":>10} {"safe?":>6}')
    print('  ' + '-' * 62)

    margin_L12 = 3e-6 * 12**(-0.97)  # margin at L ~ 12

    for T in [100, 1000, 1e4, 1e6, 1e8, 1e10, 1e12]:
        # Numerical integration from T to 10*T (captures most of tail)
        n_pts = 10000
        gs = np.linspace(T, 100*T, n_pts)
        dg = gs[1] - gs[0]
        integrand = np.array([vk_delta_max(g) * 4/g**2 * zero_density(g) for g in gs])
        tail = np.sum(integrand) * dg
        ratio = tail / margin_L12
        safe = tail < margin_L12
        print(f'  {T:>12.0f} {tail:>14.6e} {margin_L12:>14.6e} '
              f'{ratio:>10.4f} {"YES" if safe else "NO":>6}')
    sys.stdout.flush()

    # ==================================================================
    # PART 3: NON-PERTURBATIVE -- ACTUAL HIGH-GAMMA CONTRIBUTIONS
    # ==================================================================
    print('\n  === PART 3: ACTUAL CONTRIBUTION OF HIGH-GAMMA ZEROS ===')
    print('  Compute full Delta_M for zeros at gamma = 50, 100, 500.')
    print('  Move them off-line by delta = 0.01 and 0.1.\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_f, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M_f, N)
    max_eig_base = np.linalg.eigvalsh(Mo)[-1]

    print(f'  Baseline max_eig = {max_eig_base:+.6e}\n')
    print(f'  {"gamma":>8} {"delta":>8} {"max_eig":>14} {"shift":>14} '
          f'{"||dM||":>12}')
    print('  ' + '-' * 60)

    for gamma in [50, 100, 200, 500]:
        for delta in [0.01, 0.1, 0.5]:
            rho_on = 0.5 + 1j * gamma
            rho_off = 0.5 + delta + 1j * gamma
            dM_on = zero_contribution_matrix(rho_on, L, N)
            dM_off = zero_contribution_matrix(rho_off, L, N)
            shift_M = odd_block(dM_off, N) - odd_block(dM_on, N)

            Mo_pert = Mo + shift_M
            max_eig_pert = np.linalg.eigvalsh(Mo_pert)[-1]
            shift = max_eig_pert - max_eig_base

            print(f'  {gamma:>8.0f} {delta:>8.2f} {max_eig_pert:>+14.6e} '
                  f'{shift:>+14.6e} {np.linalg.norm(shift_M):>12.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 4: THE L-DEPENDENT PICTURE
    # ==================================================================
    print('\n  === PART 4: AT WHAT L DOES THE TAIL BECOME DANGEROUS? ===')
    print('  For a zero at gamma with delta off-line, the contribution')
    print('  to M entries scales as ~ e^{delta*L}/gamma (boundary term).')
    print('  When does this exceed the margin?\n')

    print(f'  Margin(L) ~ 3e-6 * L^(-0.97)')
    print('  Contribution from zero at (sigma, gamma): ~ e^{(sigma-0.5)*L}/gamma')
    print('  For sigma = 1 - c/(log gamma)^{2/3} (VK worst case):\n')

    print(f'  {"gamma":>10} {"sigma_max":>10} {"delta_max":>10} | '
          f'{"L_dangerous":>12} {"lam^2":>14}')
    print('  ' + '-' * 62)

    for gamma in [1e3, 1e4, 1e6, 1e8, 1e10, 1e12]:
        sigma_max = 1 - c_VK / (np.log(gamma)**(2/3) * np.log(np.log(gamma))**(1/3))
        delta_max = sigma_max - 0.5

        # e^{delta*L}/gamma = margin = 3e-6/L
        # e^{delta*L} = gamma * 3e-6 / L
        # delta*L = log(gamma * 3e-6 / L)
        # Solve iteratively
        L_danger = 1.0
        for _ in range(100):
            rhs = np.log(gamma * 3e-6 / max(L_danger, 0.1))
            if rhs <= 0:
                L_danger = float('inf')
                break
            L_danger = rhs / delta_max
            if L_danger > 1e15:
                L_danger = float('inf')
                break

        lam_sq_danger = np.exp(L_danger) if L_danger < 100 else float('inf')
        print(f'  {gamma:>10.0f} {sigma_max:>10.6f} {delta_max:>10.6f} | '
              f'{L_danger:>12.2f} {lam_sq_danger:>14.2e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 5: THE CRITICAL CALCULATION -- CAN WE CLOSE THE ARGUMENT?
    # ==================================================================
    print('\n  === PART 5: CAN WE CLOSE THE ARGUMENT? ===\n')

    # Known: zeros up to T verified on critical line
    T_verified = 3e12

    # At L_max (our computational verification limit):
    L_max = np.log(287000)  # lam^2 = 287000
    margin_at_Lmax = 3e-6 * L_max**(-0.97)

    print(f'  Verified zeros: gamma < T = {T_verified:.0e}')
    print(f'  Verified M_odd < 0: L < {L_max:.2f} (lam^2 = 287000)')
    print(f'  Margin at L_max: {margin_at_Lmax:.6e}')
    print()

    # First-order tail: Sum_{gamma > T} delta * 4/gamma^2 * density
    n_pts = 100000
    gs = np.logspace(np.log10(T_verified), np.log10(T_verified * 1e6), n_pts)
    dg = np.diff(gs)
    tail_first_order = 0
    for i in range(len(dg)):
        g = gs[i]
        dm = vk_delta_max(g)
        tail_first_order += dm * 4/g**2 * zero_density(g) * dg[i]

    print(f'  First-order tail bound (gamma > T):')
    print(f'    Sum delta_max * 4/gamma^2 * density = {tail_first_order:.6e}')
    print(f'    Margin at L_max = {margin_at_Lmax:.6e}')
    print(f'    Ratio: {tail_first_order / margin_at_Lmax:.6f}')
    print(f'    TAIL < MARGIN: {tail_first_order < margin_at_Lmax}')
    print()

    # Non-perturbative: boundary term e^{delta*L}/gamma at L = L_max
    # Worst single zero at gamma just above T with delta = VK bound
    delta_at_T = vk_delta_max(T_verified)
    single_zero_contrib = np.exp(delta_at_T * L_max) / T_verified
    print(f'  Non-perturbative (single worst zero just above T):')
    print(f'    delta_max at T = {delta_at_T:.6f}')
    print(f'    e^(delta*L)/gamma = {single_zero_contrib:.6e}')
    print(f'    Margin = {margin_at_Lmax:.6e}')
    print(f'    Safe: {single_zero_contrib < margin_at_Lmax}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 64e VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
