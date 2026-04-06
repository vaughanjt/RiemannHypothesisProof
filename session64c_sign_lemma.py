"""
SESSION 64c -- BIRD-DOGGING THE SIGN LEMMA

The sign lemma: moving any zero off the critical line increases max_eig.

By first-order perturbation theory:
  d(max_eig)/d(delta) = v^T P v   at delta=0

where v = critical eigenvector of M_odd, and
  P = d/d(delta) [Delta_M_odd(1/2 + delta + i*gamma)] at delta=0

P has entries:
  P[i,j] = -2 * Re[ integral_0^L K(i,j,t) * t * exp(i*gamma*t) dt ]

where K is the prime-sum kernel. These integrals have CLOSED FORMS.

Plan:
  1. Derive closed-form P matrix
  2. Verify against numerical derivative
  3. Compute v^T P v across zeros and lambda
  4. Decompose: which components drive the sign?
  5. Seek analytical proof of v^T P v > 0
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


def analytic_t_integral(omega, L):
    """
    Compute integral_0^L t * exp(i*omega*t) dt analytically.

    = L*exp(i*omega*L)/(i*omega) - (exp(i*omega*L) - 1)/(i*omega)^2
    """
    if abs(omega) < 1e-12:
        # Taylor: integral = L^2/2 + i*omega*L^3/3 + ...
        return L**2 / 2 + 1j * omega * L**3 / 6
    iw = 1j * omega
    eiwL = np.exp(iw * L)
    return L * eiwL / iw - (eiwL - 1) / iw**2


def build_perturbation_matrix(gamma, L, N):
    """
    Build the perturbation matrix P = d/d(delta) [2*Re(Delta_M(rho))] at delta=0.

    P[i,j] = -2 * Re[ integral_0^L K(i,j,t) * t * exp(i*gamma*t) dt ]

    Diagonal K(n,n,t) = 2*(L-t)/L * cos(2*pi*n*t/L)
    Off-diag K(n,m,t) = [sin(2*pi*m*t/L) - sin(2*pi*n*t/L)] / (pi*(n-m))

    Using cos(at) = Re[exp(iat)] and sin(at) = Im[exp(iat)]:
      integral cos(at) * t * exp(igt) dt = Re[ integral t * exp(i(a+g)t) dt ]
      integral sin(at) * t * exp(igt) dt = Im[ integral t * exp(i(a+g)t) dt ]
    """
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    P = np.zeros((dim, dim))

    for i in range(dim):
        n_val = ns[i]
        for j in range(dim):
            m_val = ns[j]

            if i == j:
                # Diagonal: K = 2*(L-t)/L * cos(2*pi*n*t/L)
                # = 2*cos(a*t) - (2*t/L)*cos(a*t)
                # where a = 2*pi*n/L
                a = 2 * np.pi * n_val / L

                # integral_0^L 2*cos(at)*t*exp(igt) dt
                # = 2*Re[ integral t*exp(i(a+g)t) dt ]
                I1 = 2 * np.real(analytic_t_integral(a + gamma, L))

                # integral_0^L (2*t/L)*cos(at)*t*exp(igt) dt
                # = (2/L) * integral t^2 * cos(at) * exp(igt) dt
                # = (2/L) * Re[ integral t^2 * exp(i(a+g)t) dt ]
                # Need: integral_0^L t^2 * exp(iwt) dt
                w1 = a + gamma
                I_t2_1 = t2_integral(w1, L)

                # Also need the conjugate frequency contribution
                # cos(at) = (exp(iat) + exp(-iat))/2
                # So integral cos(at)*t*exp(igt) dt = 0.5*[I(a+g) + I(-a+g)]
                I1_full = np.real(analytic_t_integral(a + gamma, L) +
                                  analytic_t_integral(-a + gamma, L))
                I_t2_full = np.real(t2_integral(a + gamma, L) +
                                    t2_integral(-a + gamma, L))

                # K(n,n,t) = 2*(L-t)/L * cos(2*pi*n*t/L)
                # integral K*t*exp(igt) dt = 2*integral cos(at)*t*exp(igt) dt
                #                           - (2/L)*integral cos(at)*t^2*exp(igt) dt
                P[i, j] = -2 * (I1_full - (1.0/L) * I_t2_full)

            else:
                # Off-diagonal: K = [sin(2*pi*m*t/L) - sin(2*pi*n*t/L)] / (pi*(n-m))
                nm_diff = n_val - m_val
                if abs(nm_diff) < 0.5:
                    continue

                am = 2 * np.pi * m_val / L
                an = 2 * np.pi * n_val / L

                # integral sin(at)*t*exp(igt) dt = Im[integral t*exp(i(a+g)t) dt]
                # But sin(at) = (exp(iat)-exp(-iat))/(2i)
                # So integral sin(at)*t*exp(igt) = Im[I(a+g)] using the full formula:
                Im_m = np.imag(analytic_t_integral(am + gamma, L) +
                               analytic_t_integral(-am + gamma, L))
                # Wait, sin(at) = Im[exp(iat)], not (exp(iat)+exp(-iat))/2i
                # integral sin(at)*t*exp(igt) dt = Im[integral t*exp(i(a+g)t) dt]
                Im_m = np.imag(analytic_t_integral(am + gamma, L))
                Im_n = np.imag(analytic_t_integral(an + gamma, L))

                P[i, j] = -2 * (Im_m - Im_n) / (np.pi * nm_diff)

    return P


def t2_integral(omega, L):
    """
    Compute integral_0^L t^2 * exp(i*omega*t) dt analytically.

    = L^2*exp(iwL)/(iw) - 2*L*exp(iwL)/(iw)^2 + 2*(exp(iwL)-1)/(iw)^3
    """
    if abs(omega) < 1e-12:
        return L**3 / 3 + 1j * omega * L**4 / 4
    iw = 1j * omega
    eiwL = np.exp(iw * L)
    return (L**2 * eiwL / iw -
            2 * L * eiwL / iw**2 +
            2 * (eiwL - 1) / iw**3)


def run():
    print()
    print('#' * 76)
    print('  SESSION 64c -- BIRD-DOGGING THE SIGN LEMMA')
    print('#' * 76)

    gammas = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832]

    # ==================================================================
    # PART 1: VERIFY CLOSED-FORM P AGAINST NUMERICAL DERIVATIVE
    # ==================================================================
    print('\n  === PART 1: VERIFY ANALYTIC P ===')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    gamma = gammas[0]

    P_analytic = build_perturbation_matrix(gamma, L, N)
    Po_analytic = odd_block(P_analytic, N)

    # Numerical derivative: [Delta_M(0.5+eps+ig) - Delta_M(0.5+ig)] / eps
    from session64_asymptotic_test import zero_contribution_matrix
    eps = 1e-5
    dM_base = zero_contribution_matrix(0.5 + 1j*gamma, L, N)
    dM_pert = zero_contribution_matrix(0.5 + eps + 1j*gamma, L, N)
    P_numerical = (dM_pert - dM_base) / eps
    Po_numerical = odd_block(P_numerical, N)

    diff = np.max(np.abs(Po_analytic - Po_numerical))
    rel_diff = diff / (np.max(np.abs(Po_numerical)) + 1e-15)
    print(f'  gamma={gamma:.4f}, lam^2={lam_sq}:')
    print(f'  ||P_analytic - P_numerical||_max = {diff:.6e}')
    print(f'  Relative: {rel_diff:.6e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: v^T P v FOR EACH ZERO
    # ==================================================================
    print('\n  === PART 2: FIRST-ORDER EIGENVALUE SHIFT v^T P v ===')
    print('  v = critical eigenvector (max eigenvalue of M_odd).')
    print('  v^T P v > 0 <=> off-line shift increases max_eig.\n')

    for lam_sq in [200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M_f, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M_f, N)
        eigs, vecs = np.linalg.eigh(Mo)
        v = vecs[:, -1]  # critical eigenvector (max eigenvalue)

        print(f'  lam^2={lam_sq} (L={L:.2f}):')
        print(f'    max_eig = {eigs[-1]:+.6e}')
        print(f'    {"zero":>6} {"gamma":>10} {"v^T P v":>14} {"sign":>6}')
        print('    ' + '-' * 40)

        for k, gamma in enumerate(gammas):
            P_mat = build_perturbation_matrix(gamma, L, N)
            Po = odd_block(P_mat, N)
            vPv = float(v @ Po @ v)
            sign = '+' if vPv > 0 else '-'
            print(f'    {k+1:>6d} {gamma:>10.4f} {vPv:>+14.6e} {sign:>6}')
        print()
    sys.stdout.flush()

    # ==================================================================
    # PART 3: DECOMPOSE v^T P v INTO DIAGONAL + OFF-DIAGONAL
    # ==================================================================
    print('\n  === PART 3: DIAGONAL vs OFF-DIAGONAL CONTRIBUTION ===')
    print('  v^T P v = v^T P_diag v + v^T P_offdiag v\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_f, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M_f, N)
    eigs, vecs = np.linalg.eigh(Mo)
    v = vecs[:, -1]

    print(f'  lam^2={lam_sq}:')
    print(f'  {"zero":>6} {"v^T P v":>14} {"diag part":>14} {"offdiag part":>14} '
          f'{"diag %":>8}')
    print('  ' + '-' * 62)

    for k, gamma in enumerate(gammas[:5]):
        P_mat = build_perturbation_matrix(gamma, L, N)
        Po = odd_block(P_mat, N)
        Po_diag = np.diag(np.diag(Po))
        Po_offdiag = Po - Po_diag
        vPv = float(v @ Po @ v)
        vPv_d = float(v @ Po_diag @ v)
        vPv_o = float(v @ Po_offdiag @ v)
        pct_d = 100 * vPv_d / vPv if abs(vPv) > 1e-15 else float('nan')
        print(f'  {k+1:>6d} {vPv:>+14.6e} {vPv_d:>+14.6e} {vPv_o:>+14.6e} '
              f'{pct_d:>8.1f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 4: STRUCTURE OF v^T P v -- COMPONENT ANALYSIS
    # ==================================================================
    print('\n  === PART 4: WHICH COMPONENTS OF v MATTER? ===')
    print('  v = critical eigenvector. How do its components')
    print('  interact with P?\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_f, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M_f, N)
    eigs, vecs = np.linalg.eigh(Mo)
    v = vecs[:, -1]

    gamma = gammas[0]
    P_mat = build_perturbation_matrix(gamma, L, N)
    Po = odd_block(P_mat, N)

    # v^T P v = sum_{i,j} v_i * P_ij * v_j
    # Decompose by row i contribution: row_i = v_i * sum_j P_ij * v_j = v_i * (Pv)_i
    Pv = Po @ v
    row_contributions = v * Pv

    print(f'  v^T P v = {float(v @ Pv):+.6e}')
    print(f'  Critical eigenvector v (first 10 components):')
    print(f'  {"n":>4} {"v[n]":>10} {"(Pv)[n]":>12} {"v*(Pv)":>14} {"cumul":>14}')
    print('  ' + '-' * 56)

    cumul = 0
    for i in range(min(15, N)):
        cumul += row_contributions[i]
        n = i + 1
        print(f'  {n:>4d} {v[i]:>+10.6f} {Pv[i]:>+12.6f} '
              f'{row_contributions[i]:>+14.6e} {cumul:>+14.6e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 5: THE CRITICAL QUADRATIC FORM
    # ==================================================================
    print('\n  === PART 5: P RESTRICTED TO 2x2 CRITICAL SUBSPACE ===')
    print('  v ~ c1*|1> + c2*|2> (Session 58). Project P onto {|1>,|2>}.\n')

    for lam_sq in [200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M_f, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M_f, N)
        eigs, vecs = np.linalg.eigh(Mo)
        v = vecs[:, -1]

        gamma = gammas[0]
        P_mat = build_perturbation_matrix(gamma, L, N)
        Po = odd_block(P_mat, N)

        # 2x2 projection onto n=1, n=2 (indices 0,1 in odd block)
        P22 = Po[:2, :2]
        v2 = v[:2]
        v2_norm = v2 / np.linalg.norm(v2)

        vPv_full = float(v @ Po @ v)
        vPv_2x2 = float(v2_norm @ P22 @ v2_norm)

        # Eigenvalues of P22
        eP = np.linalg.eigvalsh(P22)

        print(f'  lam^2={lam_sq}: v = ({v[0]:+.4f}, {v[1]:+.4f}, ...)')
        print(f'    P[1:2,1:2] = [[{P22[0,0]:+.4f}, {P22[0,1]:+.4f}],'
              f' [{P22[1,0]:+.4f}, {P22[1,1]:+.4f}]]')
        print(f'    P22 eigenvalues: {eP[0]:+.6f}, {eP[1]:+.6f}')
        print(f'    v^T P v (full): {vPv_full:+.6e}')
        print(f'    v^T P v (2x2):  {vPv_2x2:+.6e}')
        print(f'    2x2 captures {100*vPv_2x2/vPv_full:.1f}% of full')
        print()
    sys.stdout.flush()

    # ==================================================================
    # PART 6: ANALYTICAL STRUCTURE OF P[1,1] AND P[1,2]
    # ==================================================================
    print('\n  === PART 6: ANALYTICAL FORM OF KEY P ENTRIES ===')
    print('  P_odd[0,0] = P entry at n=1 (diagonal)')
    print('  P_odd[0,1] = P entry at n=1,n=2 (off-diagonal)')
    print('  These determine v^T P v since v is concentrated on n=1,2.\n')

    for lam_sq in [200, 1000, 5000, 20000, 50000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))

        gamma = gammas[0]
        P_mat = build_perturbation_matrix(gamma, L, N)
        Po = odd_block(P_mat, N)

        print(f'  lam^2={lam_sq:>6d} L={L:.2f}: '
              f'P[1,1]={Po[0,0]:>+10.4f}, '
              f'P[1,2]={Po[0,1]:>+10.4f}, '
              f'P[2,2]={Po[1,1]:>+10.4f}, '
              f'tr(P)={np.trace(Po):>+10.4f}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 64c VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
