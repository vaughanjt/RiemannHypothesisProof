"""
SESSION 64 -- ASYMPTOTIC PROOF FEASIBILITY

Can we prove M_odd < 0 for large L using the explicit formula?

The explicit formula says:
  Sum Lambda(n) n^{-1/2} f(log n) = (pole term) - Sum_rho (zero term) + (error)

For zero rho, the contribution to M is:
  Delta_M[i,j] = -integral_0^L K(i,j,t) * exp((rho - 1/2)*t) dt

where K is the matrix kernel from the prime sum.

KEY TEST: move the first zeta zero from rho = 1/2 + i*gamma to
rho = 1/2 + delta + i*gamma. If the Schur margin flips sign,
then off-line zeros are detectable and the asymptotic approach works.
If the margin is insensitive, the approach is dead.
"""

import sys
import numpy as np
from scipy import integrate

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import build_all_fast


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def zero_contribution_matrix(rho, L, N):
    """
    Compute the matrix Delta_M contributed by a single zero rho.

    The explicit formula says the zero at rho contributes:
      Delta_M[i,j] = -integral_0^L K(i,j,t) * exp((rho - 1/2)*t) dt

    where K(i,j,t) is the prime-sum kernel (same as in build_M_prime).
    For diagonal (i=j): K(n,n,t) = 2*(L-t)/L * cos(2*pi*n*t/L)
    For off-diag (i!=j): K(n,m,t) = [sin(2*pi*m*t/L) - sin(2*pi*n*t/L)] / (pi*(n-m))

    Returns a (2N+1) x (2N+1) matrix. We take the REAL PART since
    each zero comes with its conjugate, and the combined contribution
    from rho and conj(rho) is 2*Re[Delta_M(rho)].
    """
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    s = rho - 0.5  # exponent in the integral

    # Numerical integration for each matrix entry
    n_quad = 2000
    dt = L / n_quad
    t = dt * (np.arange(n_quad) + 0.5)  # midpoints

    # exp(s*t) for all t
    exp_st = np.exp(s * t)  # complex array

    # Diagonal contributions: integral of 2*(L-t)/L * cos(2*pi*n*t/L) * exp(s*t)
    cos_mat = np.cos(2 * np.pi * ns[:, None] * t[None, :] / L)  # (dim, n_quad)
    weight_diag = 2 * (L - t) / L  # (n_quad,)
    diag_integrands = cos_mat * weight_diag[None, :] * exp_st[None, :]  # (dim, n_quad)
    diag_integrals = -diag_integrands.sum(axis=1) * dt  # (dim,) negative from explicit formula

    # Off-diagonal contributions: integral of sin_diff / (pi*(n-m)) * exp(s*t)
    sin_mat = np.sin(2 * np.pi * ns[:, None] * t[None, :] / L)  # (dim, n_quad)

    dM = np.zeros((dim, dim), dtype=complex)
    np.fill_diagonal(dM, diag_integrals)

    # Off-diagonal: for each pair (i,j), integrate [sin(m*t) - sin(n*t)]/(pi*(n-m)) * exp(s*t)
    # Vectorize over t, loop over pairs (expensive but manageable for dim ~ 80)
    nm_diff = ns[:, None] - ns[None, :]
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            n_val = ns[i]
            m_val = ns[j]
            if abs(n_val - m_val) < 0.5:
                continue
            sin_diff = sin_mat[j, :] - sin_mat[i, :]  # sin(m*t) - sin(n*t)
            kernel = sin_diff / (np.pi * (n_val - m_val))
            integral = -np.sum(kernel * exp_st) * dt
            dM[i, j] = integral

    # Return 2*Re[dM] (contribution from rho + conj(rho) pair)
    return 2 * dM.real


def schur_margin(Mo):
    """Compute the Schur margin of M_odd at step 0."""
    a1 = Mo[0, 0]
    c = Mo[0, 1:]
    B = Mo[1:, 1:]
    try:
        Binv_c = np.linalg.solve(B, c)
        coupling = -float(c @ Binv_c)  # positive
        margin = abs(a1) - coupling
        return margin, a1, coupling
    except:
        return float('nan'), a1, float('nan')


def run():
    print()
    print('#' * 76)
    print('  SESSION 64 -- ASYMPTOTIC PROOF FEASIBILITY')
    print('#' * 76)

    # First few zeta zero ordinates
    gammas = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832]

    # ==================================================================
    # PART 1: ZERO CONTRIBUTION MATRIX -- SANITY CHECK
    # ==================================================================
    print('\n  === PART 1: ZERO CONTRIBUTION MATRICES ===')
    print('  Compute Delta_M for first few zeros on the critical line.')
    print('  The norm should be O(1) for on-line zeros.\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))

    for k, gamma in enumerate(gammas[:5]):
        rho = 0.5 + 1j * gamma
        dM = zero_contribution_matrix(rho, L, N)
        dMo = odd_block(dM, N)
        norm_dM = np.linalg.norm(dMo)
        max_eig = np.linalg.eigvalsh(dMo)[-1]
        print(f'  Zero {k+1} (gamma={gamma:.4f}): '
              f'||Delta_M_odd|| = {norm_dM:.4f}, '
              f'max_eig = {max_eig:+.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: MOVE FIRST ZERO OFF-LINE
    # ==================================================================
    print('\n  === PART 2: MOVE FIRST ZERO OFF CRITICAL LINE ===')
    print('  Shift rho1 from 1/2+i*14.13 to 1/2+delta+i*14.13.')
    print('  Track the Schur margin of M_odd.\n')

    _, M_full, _ = build_all_fast(lam_sq, N)
    Mo_real = odd_block(M_full, N)
    margin_real, a1_real, coupling_real = schur_margin(Mo_real)
    max_eig_real = np.linalg.eigvalsh(Mo_real)[-1]

    print(f'  Baseline (all zeros on-line):')
    print(f'    max_eig = {max_eig_real:+.6e}')
    print(f'    Schur margin = {margin_real:+.6e}')
    print(f'    a1 = {a1_real:+.6f}, coupling = {coupling_real:.6f}')
    print()

    gamma1 = gammas[0]

    # On-line contribution from zero 1
    rho_online = 0.5 + 1j * gamma1
    dM_online = zero_contribution_matrix(rho_online, L, N)
    dMo_online = odd_block(dM_online, N)

    print(f'  {"delta":>8} {"max_eig":>14} {"margin":>14} '
          f'{"a1":>12} {"coupling":>12} {"||dM_shift||":>12}')
    print('  ' + '-' * 78)

    for delta in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        rho_offline = 0.5 + delta + 1j * gamma1
        dM_offline = zero_contribution_matrix(rho_offline, L, N)
        dMo_offline = odd_block(dM_offline, N)

        # Net shift: remove on-line contribution, add off-line
        dMo_shift = dMo_offline - dMo_online

        # Perturbed M_odd
        Mo_pert = Mo_real + dMo_shift
        margin_p, a1_p, coupling_p = schur_margin(Mo_pert)
        max_eig_p = np.linalg.eigvalsh(Mo_pert)[-1]

        print(f'  {delta:>8.3f} {max_eig_p:>+14.6e} {margin_p:>+14.6e} '
              f'{a1_p:>+12.4f} {coupling_p:>12.4f} '
              f'{np.linalg.norm(dMo_shift):>12.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 3: SENSITIVITY AT DIFFERENT LAMBDA
    # ==================================================================
    print('\n  === PART 3: DELTA=0.01 AT DIFFERENT LAMBDA ===')
    print('  Does the perturbation grow with lambda (as expected from e^{delta*L})?\n')

    delta_test = 0.01
    gamma1 = gammas[0]

    print(f'  {"lam^2":>8} {"L":>6} {"margin_real":>14} {"margin_pert":>14} '
          f'{"ratio":>10} {"||shift||":>12} {"e^(dL)":>10}')
    print('  ' + '-' * 78)

    for lam_sq in [50, 200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))

        _, M_f, _ = build_all_fast(lam_sq, N)
        Mo_f = odd_block(M_f, N)
        margin_f, _, _ = schur_margin(Mo_f)

        rho_on = 0.5 + 1j * gamma1
        rho_off = 0.5 + delta_test + 1j * gamma1
        dM_on = zero_contribution_matrix(rho_on, L, N)
        dM_off = zero_contribution_matrix(rho_off, L, N)
        dMo_on = odd_block(dM_on, N)
        dMo_off = odd_block(dM_off, N)
        shift = dMo_off - dMo_on

        Mo_pert = Mo_f + shift
        margin_p, _, _ = schur_margin(Mo_pert)
        max_eig_p = np.linalg.eigvalsh(Mo_pert)[-1]

        edL = np.exp(delta_test * L)
        ratio = margin_p / margin_f if abs(margin_f) > 1e-15 else float('inf')

        print(f'  {lam_sq:>8d} {L:>6.2f} {margin_f:>+14.6e} {margin_p:>+14.6e} '
              f'{ratio:>10.4f} {np.linalg.norm(shift):>12.4f} {edL:>10.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 4: MULTIPLE ZEROS OFF-LINE
    # ==================================================================
    print('\n  === PART 4: MOVE FIRST K ZEROS OFF-LINE (delta=0.01) ===')
    print('  Cumulative effect of moving multiple zeros.\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_f, _ = build_all_fast(lam_sq, N)
    Mo_base = odd_block(M_f, N)

    print(f'  {"K zeros":>8} {"max_eig":>14} {"margin":>14} {"neg_def":>8}')
    print('  ' + '-' * 48)

    cumulative_shift = np.zeros_like(Mo_base)
    for k in range(min(10, len(gammas))):
        gamma = gammas[k]
        rho_on = 0.5 + 1j * gamma
        rho_off = 0.5 + 0.01 + 1j * gamma
        dM_on = zero_contribution_matrix(rho_on, L, N)
        dM_off = zero_contribution_matrix(rho_off, L, N)
        shift = odd_block(dM_off, N) - odd_block(dM_on, N)
        cumulative_shift += shift

        Mo_pert = Mo_base + cumulative_shift
        margin_p, _, _ = schur_margin(Mo_pert)
        max_eig_p = np.linalg.eigvalsh(Mo_pert)[-1]
        nd = max_eig_p < 0

        print(f'  {k+1:>8d} {max_eig_p:>+14.6e} {margin_p:>+14.6e} '
              f'{"YES" if nd else "**NO**":>8}')
    sys.stdout.flush()

    # ==================================================================
    # PART 5: THE QUANTITATIVE GAP
    # ==================================================================
    print('\n  === PART 5: QUANTITATIVE GAP ANALYSIS ===')
    print('  The Vinogradov-Korobov zero-free region gives:')
    print('    sigma > 1 - c / (log t)^{2/3} (log log t)^{1/3}')
    print('  For t > T0 (verified zeros up to T0 ~ 3e12).')
    print('  What delta does this correspond to?\n')

    T0 = 3e12  # verified height
    c_VK = 0.05  # approximate Vinogradov-Korobov constant
    logT = np.log(T0)
    loglogT = np.log(logT)
    delta_VK = c_VK / (logT ** (2/3) * loglogT ** (1/3))

    print(f'  T0 = {T0:.0e} (verified zero-free)')
    print(f'  log(T0) = {logT:.2f}')
    print(f'  Vinogradov-Korobov delta = {delta_VK:.6f}')
    print(f'  (zeros have sigma < 1/2 + {delta_VK:.6f} = {0.5 + delta_VK:.6f})')
    print()

    # What's the perturbation norm for this delta at various L?
    print(f'  Effect of VK-scale off-line zero at delta={delta_VK:.6f}:')
    print(f'  {"lam^2":>8} {"L":>6} {"e^(dL)":>10} {"margin":>14} {"ratio e^dL/margin":>18}')
    print('  ' + '-' * 60)

    for lam_sq in [1000, 10000, 1e6, 1e10, 1e20]:
        L = np.log(lam_sq)
        edL = np.exp(delta_VK * L)
        margin_est = 3e-6 * L ** (-0.97)
        ratio = edL / margin_est
        print(f'  {lam_sq:>8.0f} {L:>6.1f} {edL:>10.4f} {margin_est:>14.6e} {ratio:>18.2f}')

    print()
    print('  The ratio e^{delta*L} / margin grows exponentially.')
    print('  The zero-free region perturbation SWAMPS the margin.')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 64 VERDICT')
    print('=' * 76)
    print()
    print('  The asymptotic approach requires bounding the zero sum error.')
    print('  Two scenarios:')
    print()
    print('  A. Off-line zeros DISRUPT the margin:')
    print('     -> The zero-free region delta is too small to detect.')
    print('     -> Would need delta >> margin, but VK gives delta ~ 0.002')
    print('        while margin ~ 10^{-7}. Quantitative gap: 10^4.')
    print()
    print('  B. Off-line zeros are ABSORBED by the Cauchy-Loewner structure:')
    print('     -> Then we cannot distinguish on-line from off-line.')
    print('     -> The asymptotic approach is dead.')
    print()
    print('  The computational test determines which scenario holds.')


if __name__ == '__main__':
    run()
