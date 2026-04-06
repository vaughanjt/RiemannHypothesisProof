"""
SESSION 64f -- CLOSE THE NON-PERTURBATIVE GAP

The first-order tail converges (80,000x safety). The non-perturbative
correction potentially grows as e^{delta*L}. Can we kill it?

KEY INSIGHT: The prime-sum kernel K(t) = 2(L-t)/L * cos(2*pi*n*t/L)
vanishes at t=L: K(L) = 0.

When computing integral_0^L K(t) * e^{s*t} dt by parts (s = delta + i*gamma):
  = [K(t)*e^{st}/s]_0^L - integral K'(t)*e^{st}/s dt
  = K(L)*e^{sL}/s - K(0)/s - (1/s)*integral K'(t)*e^{st} dt
  = 0 - K(0)/s - (1/s)*integral K'(t)*e^{st} dt

The e^{sL} boundary term VANISHES because K(L)=0!

The surviving terms are O(1/s) = O(1/gamma), and the second integration
by parts gives O(1/gamma^2). This means the FULL (non-perturbative)
zero contribution decays as 1/gamma^2, NOT as e^{delta*L}/gamma.

If this holds, the tail converges for ALL L, and the argument closes.
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast
from session64_asymptotic_test import zero_contribution_matrix, odd_block, schur_margin


def analytic_zero_contribution_ibp(gamma, delta, L, n_val):
    """
    Compute the zero contribution to the diagonal M[n,n] using
    integration by parts, tracking the boundary terms explicitly.

    integral_0^L K(t) * e^{s*t} dt  where s = delta + i*gamma
    K(t) = 2*(L-t)/L * cos(2*pi*n*t/L)

    K(L) = 0  (KEY!)
    K(0) = 2

    K'(t) = -2/L * cos(at) - 2*(L-t)/L * a*sin(at)  where a = 2*pi*n/L
    K'(L) = -2/L * cos(2*pi*n) = -2/L
    K'(0) = -2/L

    Returns the diagonal contribution (real part, doubled for rho+conj(rho)).
    """
    s = delta + 1j * gamma
    a = 2 * np.pi * n_val / L

    # Direct numerical integration for comparison
    n_quad = 10000
    dt = L / n_quad
    t = dt * (np.arange(n_quad) + 0.5)
    K = 2 * (L - t) / L * np.cos(a * t)
    direct = -2 * np.real(np.sum(K * t * np.exp(s * t)) * dt)
    # Note: the "t" factor is for the PERTURBATION (d/d(delta)).
    # For the zero contribution itself (not perturbation):
    direct_zero = -2 * np.real(np.sum(K * np.exp(s * t)) * dt)

    # Integration by parts of integral K(t) e^{st} dt:
    # = [K(t) e^{st}/s]_0^L - (1/s) integral K'(t) e^{st} dt
    # = K(L)*e^{sL}/s - K(0)/s - (1/s) integral K'(t) e^{st} dt

    K_at_L = 0.0  # K(L) = 2*(L-L)/L * cos(2*pi*n) = 0
    K_at_0 = 2.0  # K(0) = 2*(L-0)/L * cos(0) = 2

    boundary = K_at_L * np.exp(s * L) / s - K_at_0 / s
    # = 0 - 2/s = -2/s  (the e^{sL} term is KILLED by K(L)=0)

    # The integral of K'(t) e^{st} dt:
    Kprime = -2/L * np.cos(a * t) + 2*(L-t)/L * a * np.sin(a * t)
    # Wait, K'(t) = d/dt [2*(L-t)/L * cos(at)]
    # = -2/L * cos(at) + 2*(L-t)/L * (-a*sin(at))
    # = -2/L * cos(at) - 2*a*(L-t)/L * sin(at)
    Kprime = -2/L * np.cos(a * t) - 2*a*(L-t)/L * np.sin(a * t)

    int_Kprime = np.sum(Kprime * np.exp(s * t)) * dt

    ibp_result = boundary - int_Kprime / s

    # For the perturbation (d/d(delta)), we need integral K(t)*t*e^{st}:
    # IBP: integral K*t*e^{st} = [K*t*e^{st}/s] - (1/s)*integral (K'+K*t')*e^{st}
    # Hmm, this is different. Let me compute directly.

    return {
        'direct_zero': direct_zero,  # -2*Re[integral K*e^{st} dt]
        'ibp_zero': -2 * np.real(ibp_result),
        'boundary_esl': np.abs(K_at_L * np.exp(s * L) / s),  # should be 0
        'boundary_1s': np.abs(K_at_0 / s),  # O(1/gamma)
        'int_Kprime_over_s': np.abs(int_Kprime / s),
    }


def run():
    print()
    print('#' * 76)
    print('  SESSION 64f -- CLOSE THE NON-PERTURBATIVE GAP')
    print('#' * 76)

    # ==================================================================
    # PART 1: VERIFY K(L) = 0 KILLS THE BOUNDARY TERM
    # ==================================================================
    print('\n  === PART 1: BOUNDARY TERM ANALYSIS ===')
    print('  K(t) = 2*(L-t)/L * cos(2*pi*n*t/L)')
    print('  K(L) = 0 always. K(0) = 2.')
    print('  IBP: integral = [K*e^{sL}/s] - K(0)/s - ... = 0 - 2/s - ...\n')

    L = float(np.log(1000))
    gamma = 14.134725

    print(f'  L = {L:.4f}, gamma = {gamma:.4f}')
    print(f'  {"delta":>8} {"direct":>14} {"ibp":>14} {"bdry e^sL":>14} '
          f'{"bdry 1/s":>14} {"agree?":>8}')
    print('  ' + '-' * 74)

    for delta in [0, 0.01, 0.1, 0.5, 1.0, 2.0]:
        r = analytic_zero_contribution_ibp(gamma, delta, L, 1)
        agree = abs(r['direct_zero'] - r['ibp_zero']) / (abs(r['direct_zero']) + 1e-15)
        print(f'  {delta:>8.2f} {r["direct_zero"]:>+14.6e} {r["ibp_zero"]:>+14.6e} '
              f'{r["boundary_esl"]:>14.6e} {r["boundary_1s"]:>14.6e} '
              f'{agree:>8.1e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: NON-PERTURBATIVE SCALING -- DOES IT GO AS 1/gamma^2?
    # ==================================================================
    print('\n  === PART 2: NON-PERTURBATIVE SCALING WITH GAMMA ===')
    print('  At FIXED delta=0.5, vary gamma. Does the zero contribution')
    print('  scale as 1/gamma^2 (not e^{delta*L}/gamma)?\n')

    L = float(np.log(1000))
    N = max(15, round(6 * L))
    delta = 0.5

    print(f'  delta={delta}, L={L:.2f}:')
    print(f'  {"gamma":>10} {"zero_contrib":>14} {"4/g^2":>14} '
          f'{"ratio":>8} {"e^dL/g":>14} {"ratio2":>8}')
    print('  ' + '-' * 72)

    eDL = np.exp(delta * L)
    for gamma in [20, 50, 100, 200, 500, 1000, 5000, 10000]:
        r = analytic_zero_contribution_ibp(gamma, delta, L, 1)
        zc = abs(r['direct_zero'])
        formula1 = 4 / gamma**2  # first-order
        formula2 = eDL / gamma  # naive non-pert
        print(f'  {gamma:>10.0f} {zc:>14.6e} {formula1:>14.6e} '
              f'{zc/formula1:>8.2f} {formula2:>14.6e} {zc/formula2:>8.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 3: THE CRITICAL TEST -- EIGENVALUE SHIFT AT LARGE DELTA
    # ==================================================================
    print('\n  === PART 3: EIGENVALUE SHIFT vs GAMMA AT delta=0.5 ===')
    print('  Full non-perturbative eigenvalue shift from moving one zero.')
    print('  This is what actually matters for the tail bound.\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_f, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M_f, N)
    base_eig = np.linalg.eigvalsh(Mo)[-1]

    delta = 0.5
    print(f'  lam^2={lam_sq}, L={L:.2f}, delta={delta}:')
    print(f'  {"gamma":>10} {"eig_shift":>14} {"4*d/g^2":>14} '
          f'{"ratio":>8} {"C/g^alpha":>14}')
    print('  ' + '-' * 62)

    shifts = []
    gammas_test = [20, 30, 50, 100, 200, 500, 1000]
    for gamma in gammas_test:
        rho_on = 0.5 + 1j * gamma
        rho_off = 0.5 + delta + 1j * gamma
        dM_on = zero_contribution_matrix(rho_on, L, N)
        dM_off = zero_contribution_matrix(rho_off, L, N)
        shift_M = odd_block(dM_off, N) - odd_block(dM_on, N)
        Mo_pert = Mo + shift_M
        eig_shift = np.linalg.eigvalsh(Mo_pert)[-1] - base_eig
        fo = delta * 4 / gamma**2
        shifts.append((gamma, eig_shift))
        print(f'  {gamma:>10.0f} {eig_shift:>+14.6e} {fo:>14.6e} '
              f'{eig_shift/fo:>8.2f}')

    # Fit power law to the shift
    gs = np.array([s[0] for s in shifts if s[0] >= 50])
    ss = np.array([abs(s[1]) for s in shifts if s[0] >= 50])
    if len(gs) > 2:
        log_g = np.log(gs)
        log_s = np.log(ss)
        slope, intercept = np.polyfit(log_g, log_s, 1)
        C = np.exp(intercept)
        print(f'\n  Power law fit (gamma >= 50): shift ~ {C:.4f} * gamma^{slope:.3f}')
        print(f'  Expected exponent for 1/gamma^2: -2.000')
    sys.stdout.flush()

    # ==================================================================
    # PART 4: RIGOROUS TAIL BOUND FOR ALL L
    # ==================================================================
    print('\n  === PART 4: CAN THE TAIL STAY BELOW MARGIN FOR ALL L? ===')
    print('  Using the power law from Part 3 to extrapolate.\n')

    # From the fit: eigenvalue shift ~ C * gamma^alpha where alpha ~ -2
    # (or slightly different). Use the actual data.

    # The tail sum: Sum_{gamma > T} shift(gamma) * density(gamma)
    # where shift(gamma) ~ C/gamma^|alpha| and density ~ log(gamma)/(2*pi)

    # If |alpha| > 1: the sum converges as integral C*log(g)/g^|alpha| dg
    # For |alpha| = 2: integral C*log(g)/g^2 dg = C*(log(T)+1)/T

    # This is INDEPENDENT OF L (if the shift truly goes as 1/gamma^2 not e^{delta*L}/gamma^2)

    # Let's check: does the shift at delta=0.5 depend on L?
    print(f'  Eigenvalue shift at delta=0.5, gamma=200 vs L:')
    print(f'  {"lam^2":>8} {"L":>6} {"shift":>14} {"e^(dL)":>10}')
    print('  ' + '-' * 42)

    gamma_test = 200
    for lam_sq_test in [50, 200, 1000, 5000, 20000, 50000, 100000]:
        L_test = float(np.log(lam_sq_test))
        N_test = max(15, round(6 * L_test))
        _, M_test, _ = build_all_fast(lam_sq_test, N_test)
        Mo_test = odd_block(M_test, N_test)
        base = np.linalg.eigvalsh(Mo_test)[-1]

        rho_on = 0.5 + 1j * gamma_test
        rho_off = 0.5 + delta + 1j * gamma_test
        dM_on = zero_contribution_matrix(rho_on, L_test, N_test)
        dM_off = zero_contribution_matrix(rho_off, L_test, N_test)
        shift_M = odd_block(dM_off, N_test) - odd_block(dM_on, N_test)
        Mo_pert = Mo_test + shift_M
        eig_shift = np.linalg.eigvalsh(Mo_pert)[-1] - base
        eDL = np.exp(delta * L_test)

        print(f'  {lam_sq_test:>8d} {L_test:>6.2f} {eig_shift:>+14.6e} {eDL:>10.2f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 5: THE DEFINITIVE CALCULATION
    # ==================================================================
    print('\n  === PART 5: DEFINITIVE TAIL BOUND ===')
    print('  If shift ~ C/gamma^2 INDEPENDENT of L:')
    print('    Tail = C*(log T + 1)/T * (2/pi) = CONSTANT')
    print('    Margin = 3e-6/L -> 0')
    print('    Eventually tail > margin.')
    print()
    print('  If shift ~ C(L)/gamma^2 where C(L) grows:')
    print('    Need C(L) growth rate < margin decay rate.')
    print()

    # Compute C(L) = shift * gamma^2 at gamma=200, delta=0.5 for various L
    print(f'  C(L) = shift * gamma^2 at gamma={gamma_test}, delta={delta}:')
    print(f'  {"L":>6} {"shift":>14} {"C(L)":>12} {"margin":>14} '
          f'{"tail_est":>14} {"safe?":>6}')
    print('  ' + '-' * 70)

    for lam_sq_test in [50, 200, 1000, 5000, 20000, 50000, 100000]:
        L_test = float(np.log(lam_sq_test))
        N_test = max(15, round(6 * L_test))
        _, M_test, _ = build_all_fast(lam_sq_test, N_test)
        Mo_test = odd_block(M_test, N_test)
        base = np.linalg.eigvalsh(Mo_test)[-1]

        rho_on = 0.5 + 1j * gamma_test
        rho_off = 0.5 + delta + 1j * gamma_test
        dM_on = zero_contribution_matrix(rho_on, L_test, N_test)
        dM_off = zero_contribution_matrix(rho_off, L_test, N_test)
        shift_M = odd_block(dM_off, N_test) - odd_block(dM_on, N_test)
        Mo_pert = Mo_test + shift_M
        eig_shift = np.linalg.eigvalsh(Mo_pert)[-1] - base

        CL = abs(eig_shift) * gamma_test**2
        margin = 3e-6 * L_test**(-0.97)
        # Tail estimate: C(L) * (log T + 1) / T * 2/pi for T = 3e12
        T = 3e12
        tail_est = CL * (np.log(T) + 1) / T * 2 / np.pi
        safe = tail_est < margin

        print(f'  {L_test:>6.2f} {eig_shift:>+14.6e} {CL:>12.4f} '
              f'{margin:>14.6e} {tail_est:>14.6e} {"YES" if safe else "NO":>6}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 64f VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
