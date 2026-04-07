"""
SESSION 75 -- WHY DOES THE COUPLING RATIO STAY BELOW 1?

At every Schur step of M_odd:
  s_k = a_k - c_k^T B_k^{-1} c_k < 0

The ratio coupling/|diagonal| = 0.997-0.999. It stays just below 1.
If it ever reaches 1.0, M_odd gets a positive eigenvalue and RH fails.

WHY does it stay below 1? This is the tightest bottleneck.

Probes:
  1. Track the ratio at step 0 across lambda: does it approach 1?
  2. Decompose coupling into eigencomponents: which mode dominates?
  3. Express the ratio in terms of Cauchy-Loewner quantities (a_n, B_n)
  4. Test: is there an IDENTITY constraining the ratio?
  5. Connection to the displacement equation [D,M] = 1*B^T - B*1^T
  6. What happens if we define r(lambda) = coupling/|diagonal| and study r'?
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
    """Full Schur decomposition of M_odd."""
    N = Mo.shape[0]
    results = []

    R = Mo.copy()
    for step in range(N - 1):
        if R.shape[0] <= 1:
            break
        a_k = R[0, 0]
        c_k = R[0, 1:]
        B_k = R[1:, 1:]

        try:
            Binv_c = np.linalg.solve(B_k, c_k)
            coupling = -float(c_k @ Binv_c)  # positive
            s_k = a_k + coupling  # a_k < 0, coupling > 0
            ratio = coupling / abs(a_k) if abs(a_k) > 1e-15 else float('inf')
            margin = abs(a_k) - coupling

            # Eigendecomposition of B_k for component analysis
            eB, vB = np.linalg.eigh(B_k)
            projs = np.array([(c_k @ vB[:, j])**2 for j in range(len(eB))])
            contribs = projs / np.abs(eB)  # positive (since eB < 0, projs > 0)

            # Which eigencomponent dominates?
            top_idx = np.argmax(contribs)
            top_frac = contribs[top_idx] / coupling if coupling > 1e-15 else 0

            results.append({
                'step': step, 'diag': a_k, 'coupling': coupling,
                'schur': s_k, 'ratio': ratio, 'margin': margin,
                'top_eig': eB[top_idx], 'top_frac': top_frac,
                'n_eigs': len(eB)
            })
        except:
            break

        R = B_k - np.outer(c_k, Binv_c)  # Schur complement

    return results


def run():
    print()
    print('#' * 76)
    print('  SESSION 75 -- THE COUPLING RATIO')
    print('#' * 76)

    # ==================================================================
    # STEP 1: The ratio at step 0 across many lambda values
    # ==================================================================
    print(f'\n  === STEP 1: COUPLING RATIO vs LAMBDA (Step 0) ===\n')

    print(f'  {"lam^2":>8} {"L":>8} {"|a_1|":>10} {"coupling":>10} '
          f'{"ratio":>12} {"margin":>14} {"1-ratio":>12}')
    print('  ' + '-' * 78)

    ratios_vs_L = []
    for lam_sq in [20, 30, 50, 75, 100, 150, 200, 300, 500, 750,
                    1000, 1500, 2000, 3000, 5000, 7500, 10000, 20000, 50000]:
        try:
            L = np.log(lam_sq)
            N = max(15, round(6 * L))
            _, M, _ = build_all_fast(lam_sq, N)
            Mo = odd_block(M, N)

            a1 = Mo[0, 0]
            c = Mo[0, 1:]
            B = Mo[1:, 1:]
            Binv_c = np.linalg.solve(B, c)
            coupling = -float(c @ Binv_c)
            ratio = coupling / abs(a1)
            margin = abs(a1) - coupling
            one_minus_r = 1 - ratio

            ratios_vs_L.append((L, ratio, margin, one_minus_r, abs(a1)))
            print(f'  {lam_sq:>8d} {L:>8.3f} {abs(a1):>10.4f} {coupling:>10.4f} '
                  f'{ratio:>12.9f} {margin:>+14.6e} {one_minus_r:>12.6e}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # Fit 1-ratio vs L
    if len(ratios_vs_L) >= 5:
        Ls = np.array([x[0] for x in ratios_vs_L])
        one_minus = np.array([x[3] for x in ratios_vs_L])
        valid = one_minus > 0
        if np.sum(valid) >= 3:
            log_L = np.log(Ls[valid])
            log_omr = np.log(one_minus[valid])
            fit = np.polyfit(log_L, log_omr, 1)
            print(f'\n  Fit: 1 - ratio ~ {np.exp(fit[1]):.6f} * L^{fit[0]:.4f}')
            print(f'  Approaches 1 as L -> inf? Exponent = {fit[0]:.4f}')
            if fit[0] < 0:
                print(f'  YES: ratio -> 1 from below, margin shrinks as L^{fit[0]:.2f}')
            else:
                print(f'  NO: ratio moves AWAY from 1')
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: All Schur steps at lam^2=1000
    # ==================================================================
    print(f'\n  === STEP 2: ALL SCHUR STEPS (lam^2=1000) ===\n')

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    _, M, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M, N)

    results = schur_analysis(Mo)

    print(f'  {"step":>4} {"ratio":>12} {"1-ratio":>12} {"margin":>14} '
          f'{"top_eig_frac":>14}')
    print('  ' + '-' * 60)
    for r in results[:25]:
        print(f'  {r["step"]:>4d} {r["ratio"]:>12.9f} {1-r["ratio"]:>12.6e} '
              f'{r["margin"]:>+14.6e} {r["top_frac"]:>14.4f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: The dominant eigencomponent
    # ==================================================================
    print(f'\n  === STEP 3: WHAT DETERMINES THE RATIO? ===\n')

    # At step 0: coupling = Sum_j (c.v_j)^2 / |lambda_j(B)|
    # The ratio is coupling / |a_1|
    # Dominated by the eigenvalue closest to zero (least negative)

    a1 = Mo[0, 0]
    c = Mo[0, 1:]
    B = Mo[1:, 1:]
    eB, vB = np.linalg.eigh(B)

    projs = np.array([(c @ vB[:, j])**2 for j in range(len(eB))])
    contribs = projs / np.abs(eB)

    print(f'  Coupling = Sum_j (c.v_j)^2 / |lambda_j(B)|')
    print(f'  where B = M_odd[1:, 1:] (the sub-block)')
    print()

    total_coupling = contribs.sum()
    order = np.argsort(contribs)[::-1]

    print(f'  {"rank":>4} {"lambda_j":>14} {"(c.v_j)^2":>14} {"contrib":>14} {"cum%":>8}')
    print('  ' + '-' * 58)
    cumsum = 0
    for i in range(min(15, len(order))):
        j = order[i]
        cumsum += contribs[j]
        pct = cumsum / total_coupling * 100
        print(f'  {i+1:>4d} {eB[j]:>+14.6e} {projs[j]:>14.6e} '
              f'{contribs[j]:>14.6e} {pct:>7.2f}%')

    print(f'\n  Total coupling: {total_coupling:.10f}')
    print(f'  |a_1|: {abs(a1):.10f}')
    print(f'  Ratio: {total_coupling/abs(a1):.12f}')
    print(f'  Top eigencomponent fraction: {contribs[order[0]]/total_coupling:.6f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: The margin as a function of the least-negative eigenvalue
    # ==================================================================
    print(f'\n  === STEP 4: MARGIN vs LEAST-NEGATIVE EIGENVALUE ===\n')

    print(f'  If B has least-negative eigenvalue lambda_* (closest to 0):')
    print(f'  coupling ~ (c.v_*)^2 / |lambda_*|')
    print(f'  margin = |a_1| - coupling ~ |a_1| - (c.v_*)^2/|lambda_*|')
    print()

    print(f'  {"lam^2":>8} {"|a_1|":>10} {"lambda_*":>14} {"(c.v_*)^2":>14} '
          f'{"c^2/|l*|":>14} {"actual coup":>14}')
    print('  ' + '-' * 78)

    for lam_sq in [50, 200, 1000, 5000, 20000]:
        try:
            Ln = np.log(lam_sq)
            Nn = max(15, round(6 * Ln))
            _, Mn, _ = build_all_fast(lam_sq, Nn)
            Mon = odd_block(Mn, Nn)

            a1n = Mon[0, 0]
            cn = Mon[0, 1:]
            Bn = Mon[1:, 1:]
            eBn, vBn = np.linalg.eigh(Bn)

            lam_star = eBn[-1]  # least negative
            v_star = vBn[:, -1]
            proj_star = (cn @ v_star)**2
            approx_coup = proj_star / abs(lam_star)

            Binv_cn = np.linalg.solve(Bn, cn)
            actual_coup = -float(cn @ Binv_cn)

            print(f'  {lam_sq:>8d} {abs(a1n):>10.4f} {lam_star:>+14.6e} '
                  f'{proj_star:>14.6e} {approx_coup:>14.6f} {actual_coup:>14.6f}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: Is the ratio EXACTLY 1 - something?
    # ==================================================================
    print(f'\n  === STEP 5: DOES THE RATIO HAVE A CLOSED FORM? ===\n')

    # The Schur complement of M_odd at step 0:
    # s_0 = a_1 + c^T B^{-1} c
    # s_0 IS the maximum eigenvalue of M_odd (by interlacing)...
    # Actually no, it's the Schur complement, which gives the
    # determinant ratio det(M_odd)/det(B).
    # But if M_odd = [[a, c^T], [c, B]], then:
    # det(M_odd) = det(B) * (a - c^T B^{-1} c) = det(B) * s_0

    # The maximum eigenvalue lambda_max of M_odd satisfies:
    # lambda_max >= s_0 (by Schur complement interlacing)

    # Is s_0 = lambda_max exactly?
    # Not in general. But if the Schur pivot ordering matches
    # the eigenvalue ordering, it might be close.

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    _, M, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M, N)

    evals_Mo = np.linalg.eigvalsh(Mo)
    lambda_max = evals_Mo[-1]

    a1 = Mo[0, 0]
    c = Mo[0, 1:]
    B = Mo[1:, 1:]
    Binv_c = np.linalg.solve(B, c)
    s0 = a1 - float(c @ Binv_c)

    print(f'  lambda_max(M_odd) = {lambda_max:+.10e}')
    print(f'  Schur s_0         = {s0:+.10e}')
    print(f'  Ratio s_0/lambda_max = {s0/lambda_max:.6f}')
    print(f'  Are they equal? {abs(s0 - lambda_max) < 1e-12 * abs(lambda_max)}')

    # What about det(M_odd)?
    det_Mo = np.prod(evals_Mo)
    det_B = np.prod(np.linalg.eigvalsh(B))
    det_ratio = det_Mo / det_B if abs(det_B) > 1e-300 else float('inf')

    print(f'\n  det(M_odd) = {det_Mo:.6e}')
    print(f'  det(B) = {det_B:.6e}')
    print(f'  det(M_odd)/det(B) = {det_ratio:.10e}')
    print(f'  s_0 = {s0:.10e}')
    print(f'  Match: {abs(det_ratio - s0) < 1e-6 * abs(s0)}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: The ratio in terms of M's raw entries
    # ==================================================================
    print(f'\n  === STEP 6: RAW ENTRY ANALYSIS ===\n')

    # In the odd basis, M_odd[0,0] = a_{odd,1} = M[N+1,N+1] + M[N-1,N-1] - 2*M[N+1,N-1]
    # (from the parity projection)
    # Actually: odd basis |k> = (delta_{n,k} - delta_{n,-k})/sqrt(2)
    # M_odd[j,k] = (M[N+j, N+k] - M[N+j, N-k] - M[N-j, N+k] + M[N-j, N-k]) / 2

    # For j=k=1 (the diagonal):
    # a_1^odd = (M[N+1,N+1] - M[N+1,N-1] - M[N-1,N+1] + M[N-1,N-1]) / 2
    # = M[N+1,N+1] - M[N+1,N-1]  (since M symmetric, M[N+1,N+1]=M[N-1,N-1] by even symmetry)

    # In Cauchy-Loewner form: M[n,m] = a_n*delta + (B_m-B_n)/(n-m)
    # M[1,1] = a_1 (diagonal)
    # M[1,-1] = (B_{-1} - B_1)/(1-(-1)) = (B_{-1}-B_1)/2

    # Since B is odd (B_{-n} = -B_n):
    # M[1,-1] = (-B_1 - B_1)/2 = -B_1

    # And M[-1,-1] = a_{-1} = a_1 (even symmetry)

    # So a_1^odd = a_1 - (-B_1) = a_1 + B_1

    print(f'  In Cauchy-Loewner terms:')
    print(f'  M_odd[1,1] = a_1 + B_1  (diagonal + Cauchy correction)')
    print()

    # Get a_1 and B_1 directly
    from session49c_weil_residual import _compute_alpha, _compute_wr_diag
    from session41g_uncapped_barrier import sieve_primes

    ns = np.arange(-N, N + 1, dtype=float)
    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)
    primes = sieve_primes(int(lam_sq))
    a_prime = np.zeros(2*N+1)
    B_prime = np.zeros(2*N+1)
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk**(-0.5)
            y = np.log(pk)
            a_prime += w * 2 * np.cos(2*np.pi*ns*y/L)
            B_prime += w * np.sin(2*np.pi*ns*y/L) / np.pi
            pk *= int(p)

    a_full = np.array([wr[abs(int(n))] for n in ns]) + a_prime
    B_full = alpha + B_prime

    a_1_val = a_full[N+1]  # a at n=1
    B_1_val = B_full[N+1]  # B at n=1

    # Check
    actual_diag = Mo[0, 0]
    predicted_diag = a_1_val + B_1_val

    print(f'  a_1 (diagonal term at n=1) = {a_1_val:+.10f}')
    print(f'  B_1 (Cauchy function at n=1) = {B_1_val:+.10f}')
    print(f'  a_1 + B_1 = {a_1_val + B_1_val:+.10f}')
    print(f'  M_odd[0,0] = {actual_diag:+.10f}')
    print(f'  Match: {abs(predicted_diag - actual_diag) < 1e-8}')

    print(f'\n  Components of the diagonal at n=1:')
    print(f'    wr_diag(1) = {wr[1]:+.10f}')
    print(f'    a_prime(1) = {a_prime[N+1]:+.10f}')
    print(f'    alpha(1) = {alpha[N+1]:+.10f}')
    print(f'    B_prime(1) = {B_prime[N+1]:+.10f}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 75 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
