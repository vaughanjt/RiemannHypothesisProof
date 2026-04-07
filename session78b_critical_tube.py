"""
SESSION 78b -- THE CRITICAL TUBE

Session 78 found: dh/d(sigma) = 0 at sigma=1/2 (exact),
d²h/d(sigma²) > 0 (minimum) for all zeros.

Now: how WIDE is the tube around the critical line where M_odd stays < 0?

The margin is ~10^{-7}. The second-order curvature is known per-zero.
We can compute the exact critical displacement delta_crit where the
second-order shift equals the margin.

PROBES:
  1. Total d²(eig_max)/d(delta²) at delta=0 (all zeros shifted uniformly)
  2. Critical delta: how far can ALL zeros move before M_odd flips?
  3. Per-zero critical delta: which zero is most constraining?
  4. The sum rule: explicit formula constrains total zero contribution
  5. Non-uniform shifts: what shape of deviation is most dangerous?
  6. Connection to de Bruijn-Newman Lambda = 0
"""

import sys
import numpy as np
import mpmath

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast

mpmath.mp.dps = 30


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P, P


def h_pair(delta, gamma, L):
    """Zero pair contribution to explicit formula at sigma = 1/2 + delta."""
    re = L**2/4 + delta**2 - gamma**2
    im = 2 * delta * gamma
    return 2 * L * re / (re**2 + im**2)


def d2h_ddelta2(gamma, L):
    """Exact second derivative of h_pair w.r.t. delta at delta=0."""
    A = L**2/4 - gamma**2
    # h_pair(delta) = 2L*(A + delta^2) / ((A + delta^2)^2 + 4*delta^2*gamma^2)
    # At delta=0: h = 2L/A
    # d²h/d(delta²) at delta=0:
    # Using (f/g)'' = (f''g - 2f'g' + f(-g'^2 + gg''))/g^2... messy
    # Numerical:
    eps = 1e-6
    return (h_pair(eps, gamma, L) - 2*h_pair(0, gamma, L) + h_pair(-eps, gamma, L)) / eps**2


def run():
    print()
    print('#' * 76)
    print('  SESSION 78b -- THE CRITICAL TUBE')
    print('#' * 76)

    n_zeros = 50
    zeros = [float(mpmath.zetazero(k).imag) for k in range(1, n_zeros + 1)]

    # ======================================================================
    # PROBE 1: Per-zero perturbation matrix and Rayleigh quotient
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: PER-ZERO SECOND-ORDER EFFECT ON M_ODD EIGENVALUE')
    print(f'{"="*76}\n')

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    _, M, _ = build_all_fast(lam_sq, N)
    Mo, P_odd = odd_block(M, N)
    evals_o, evecs_o = np.linalg.eigh(Mo)
    eig_max = evals_o[-1]
    v_crit = evecs_o[:, -1]
    margin = abs(eig_max)

    print(f'  Baseline: eig_max(M_odd) = {eig_max:+.6e}, margin = {margin:.6e}')
    print()

    # For each zero gamma_k, compute the PERTURBATION MATRIX dM/d(delta²)
    # and its Rayleigh quotient on v_crit.
    #
    # When all zeros shift by delta, the change in h_pair for zero k is:
    #   h_pair(delta, gamma_k, L) - h_pair(0, gamma_k, L)
    #   ≈ (1/2) * d²h/d(delta²) * delta²
    #
    # This changes the zero contribution to M by:
    #   delta_M_k ≈ (1/2) * d²h_k * delta² * [Fourier matrix for zero k]
    #
    # The Rayleigh quotient of this on v_crit gives the second-order
    # contribution to eig_max.

    print(f'  {"zero#":>6} {"gamma":>10} {"d2h/dd2":>14} {"RQ(dM_k)":>14} '
          f'{"delta_crit_k":>14}')
    print('  ' + '-' * 64)

    total_d2_rq = 0
    per_zero_data = []

    for k in range(min(30, len(zeros))):
        gamma = zeros[k]
        d2h = d2h_ddelta2(gamma, L)

        # Build the perturbation matrix for this zero
        # The matrix contribution from zero k is:
        # diag: h * cos(gamma * 2*pi*n/L)
        # offdiag: h * (sin_m - sin_n) / (pi*(n-m))
        # The PERTURBATION is proportional to d²h * the same Fourier structure

        phase = gamma * 2 * np.pi * ns / L
        cos_p = np.cos(phase)
        sin_p = np.sin(phase) / np.pi

        nm = ns[:, None] - ns[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            offdiag = (sin_p[None, :] - sin_p[:, None]) / nm
        np.fill_diagonal(offdiag, 0)

        # The second-order perturbation matrix (per unit delta²)
        dM_k = d2h * (np.diag(cos_p) + offdiag)
        dM_k = (dM_k + dM_k.T) / 2

        # Project to odd block
        dMo_k = P_odd.T @ dM_k @ P_odd

        # Rayleigh quotient on critical eigenvector
        rq_k = float(v_crit @ dMo_k @ v_crit)
        total_d2_rq += rq_k

        # Critical delta for this zero alone:
        # |rq_k| * delta² / 2 = margin => delta = sqrt(2 * margin / |rq_k|)
        if abs(rq_k) > 1e-20:
            delta_crit_k = np.sqrt(2 * margin / abs(rq_k))
        else:
            delta_crit_k = float('inf')

        per_zero_data.append((gamma, d2h, rq_k, delta_crit_k))
        print(f'  {k+1:>6d} {gamma:>10.4f} {d2h:>+14.6e} {rq_k:>+14.6e} '
              f'{delta_crit_k:>14.6e}')

    print(f'\n  Total d²(eig)/d(delta²) = {total_d2_rq:+.6e}')
    print(f'  (summed over {min(30, len(zeros))} zeros)')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Critical delta (uniform shift of all zeros)
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: CRITICAL DELTA FOR UNIFORM SHIFT')
    print(f'{"="*76}\n')

    # eig_max(delta) ≈ eig_max(0) + (1/2) * total_d2_rq * delta²
    # Flip when eig_max(delta) = 0:
    #   delta_crit = sqrt(2 * |eig_max(0)| / |total_d2_rq|)

    if abs(total_d2_rq) > 1e-20:
        delta_crit_uniform = np.sqrt(2 * margin / abs(total_d2_rq))
        print(f'  Second-order estimate:')
        print(f'    margin = {margin:.6e}')
        print(f'    total curvature = {total_d2_rq:+.6e}')
        print(f'    delta_crit = sqrt(2 * {margin:.2e} / {abs(total_d2_rq):.2e})')
        print(f'    delta_crit = {delta_crit_uniform:.6e}')
        print()
        print(f'  Interpretation: zeros can move off-line by at most')
        print(f'    delta = {delta_crit_uniform:.6e} = {delta_crit_uniform:.2e}')
        print(f'  before M_odd loses negative definiteness.')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: Which zero is most constraining?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: MOST CONSTRAINING ZEROS')
    print(f'{"="*76}\n')

    # Sort by per-zero critical delta
    sorted_data = sorted(per_zero_data, key=lambda x: x[3])
    print(f'  Zeros sorted by critical delta (most constraining first):')
    print(f'  {"zero#":>6} {"gamma":>10} {"delta_crit":>14} {"RQ":>14}')
    print('  ' + '-' * 48)

    for i, (gamma, d2h, rq, dc) in enumerate(sorted_data[:15]):
        k = zeros.index(gamma) + 1
        print(f'  {k:>6d} {gamma:>10.4f} {dc:>14.6e} {rq:>+14.6e}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: Verify with direct computation at small delta
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: DIRECT VERIFICATION AT FINITE DELTA')
    print(f'{"="*76}\n')

    # Build the perturbation at finite delta and check eigenvalue
    print(f'  Testing uniform shift of h_pair at various delta:')
    print(f'  {"delta":>12} {"eig_shift (2nd order)":>22} {"sign":>6}')
    print('  ' + '-' * 44)

    for delta_test in [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
        # Total perturbation to M_odd
        dMo_total = np.zeros((N, N))
        for k in range(min(30, len(zeros))):
            gamma = zeros[k]
            dh = h_pair(delta_test, gamma, L) - h_pair(0, gamma, L)

            phase = gamma * 2 * np.pi * ns / L
            cos_p = np.cos(phase)
            sin_p = np.sin(phase) / np.pi

            nm = ns[:, None] - ns[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                offdiag = (sin_p[None, :] - sin_p[:, None]) / nm
            np.fill_diagonal(offdiag, 0)

            dM_k = dh * (np.diag(cos_p) + offdiag)
            dM_k = (dM_k + dM_k.T) / 2
            dMo_total += P_odd.T @ dM_k @ P_odd

        # Eigenvalue shift
        eig_shift = float(v_crit @ dMo_total @ v_crit)
        new_eig = eig_max + eig_shift
        print(f'  {delta_test:>12.6f} {eig_shift:>+22.6e} '
              f'{"POS" if new_eig > 0 else "neg":>6}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: The tube width vs lambda
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: CRITICAL TUBE WIDTH vs LAMBDA')
    print(f'{"="*76}\n')

    print(f'  {"lam^2":>8} {"L":>8} {"margin":>14} {"total_d2_RQ":>14} '
          f'{"delta_crit":>14}')
    print('  ' + '-' * 64)

    for lam_sq_test in [100, 200, 500, 1000, 2000, 5000, 10000]:
        L_t = np.log(lam_sq_test)
        N_t = max(15, round(6 * L_t))
        dim_t = 2 * N_t + 1
        ns_t = np.arange(-N_t, N_t + 1, dtype=float)

        _, M_t, _ = build_all_fast(lam_sq_test, N_t)
        Mo_t, P_t = odd_block(M_t, N_t)
        ev_t, evc_t = np.linalg.eigh(Mo_t)
        eig_max_t = ev_t[-1]
        v_t = evc_t[:, -1]
        margin_t = abs(eig_max_t)

        # Sum d² Rayleigh quotients
        total_d2 = 0
        for k in range(min(30, len(zeros))):
            gamma = zeros[k]
            d2h = d2h_ddelta2(gamma, L_t)

            phase = gamma * 2 * np.pi * ns_t / L_t
            cos_p = np.cos(phase)
            sin_p = np.sin(phase) / np.pi

            nm_t = ns_t[:, None] - ns_t[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                offdiag = (sin_p[None, :] - sin_p[:, None]) / nm_t
            np.fill_diagonal(offdiag, 0)

            dM_k = d2h * (np.diag(cos_p) + offdiag)
            dM_k = (dM_k + dM_k.T) / 2
            dMo_k = P_t.T @ dM_k @ P_t

            total_d2 += float(v_t @ dMo_k @ v_t)

        if abs(total_d2) > 1e-20:
            dc = np.sqrt(2 * margin_t / abs(total_d2))
        else:
            dc = float('inf')

        print(f'  {lam_sq_test:>8d} {L_t:>8.3f} {margin_t:>14.6e} '
              f'{total_d2:>+14.6e} {dc:>14.6e}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: The sum rule — explicit formula constraint
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: EXPLICIT FORMULA SUM RULE')
    print(f'{"="*76}\n')

    # The explicit formula: sum_rho h(rho - 1/2) = (arch terms) - (prime terms)
    # At delta=0: sum_k h_pair(0, gamma_k, L) = known function of L
    #
    # This constrains the TOTAL zero contribution.
    # If we know the total, and we know d²h/d(delta²) per zero,
    # we can bound how much the total can change under uniform shift.

    lam_sq = 1000
    L = np.log(lam_sq)

    zero_sum = sum(h_pair(0, g, L) for g in zeros)
    d2_sum = sum(d2h_ddelta2(g, L) for g in zeros)

    print(f'  lam^2={lam_sq}:')
    print(f'    Sum h_pair(0, gamma_k) = {zero_sum:+.6f}  ({len(zeros)} zeros)')
    print(f'    Sum d²h/d(delta²)      = {d2_sum:+.6e}')
    print(f'    At delta: sum h_pair(delta) ~ {zero_sum:.6f} + {d2_sum/2:.4e} * delta^2')
    print()

    # The explicit formula gives:
    # sum_rho h(rho-1/2) = h(0) + h(-1) - sum_p log(p) h(log p) / p^{1/2}
    # where h(r) = L/(L²/4 + r²)
    # This is a FIXED quantity (depends only on primes and L).
    # If zeros shift to sigma != 1/2, the sum changes, which means
    # the explicit formula would be VIOLATED — impossible for actual zeros.
    #
    # So the constraint is: ACTUAL zeros MUST satisfy the sum rule.
    # Any deviation from sigma=1/2 would change the zero sum and break
    # the explicit formula. The only way to maintain the sum rule is
    # to compensate — but how?
    #
    # If ONE zero moves off-line, the sum changes by ~d²h_k * delta² / 2.
    # For the sum rule to hold, OTHER zeros would need to compensate.
    # But they're at discrete positions (the zeros of zeta), not free to move.

    print(f'  The explicit formula FIXES the total zero sum.')
    print(f'  Any shift delta changes the sum by ~{d2_sum/2:.4e} * delta².')
    print(f'  For delta = delta_crit = {delta_crit_uniform:.4e}:')
    change_at_crit = d2_sum / 2 * delta_crit_uniform**2
    print(f'    Change in sum = {change_at_crit:+.6e}')
    print(f'    Relative to sum: {abs(change_at_crit / zero_sum) * 100:.6f}%')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 78b VERDICT')
    print('=' * 76)
    print()
    print('  The critical tube has a definite width that can be computed')
    print('  from the margin and the per-zero curvatures.')
    print()
    print('  The functional equation gives dh/d(sigma) = 0 (first order free).')
    print('  The second order curvature determines the tube width.')
    print('  The explicit formula fixes the total zero sum.')
    print()
    print('  Key question: does the tube width shrink to zero as lambda -> inf?')
    print('  If yes: the constraint surface touches the critical line only.')
    print('  If no: there is slack, and the argument is incomplete.')
    print()


if __name__ == '__main__':
    run()
