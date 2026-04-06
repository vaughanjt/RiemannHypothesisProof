"""
SESSION 62b -- BOUNDING THE SCHUR COMPLEMENT

The Schur descent works: M_odd < 0 at every tested lambda.
At each step: diagonal a_n < 0 and Schur s = a_n - c^T B^{-1} c < 0.

BUT: the coupling c^T B^{-1} c almost exactly cancels |a_n|.
  Step 0: a_1 = -12.24, c^T B^{-1} c = -12.24 + 1e-6 => s = -1e-6

The coupling LIFTS the Schur toward zero. The diagonal barely wins.

For a proof we need: c^T (-B^{-1}) c < |a_n| at every step.

Key observations:
  1. ALL diagonals of M_odd are NEGATIVE (verified below)
  2. The coupling cost is O(a_n) — same order as the diagonal
  3. The margin is O(10^{-6}) — extremely tight

If we can show the diagonal of M_odd is negative for ALL lambda
(from the specific formulas for a_n, B_n), then the Schur descent
gives M_odd < 0 provided the coupling doesn't exceed the diagonal.

The tight part: WHY does the coupling stay below the diagonal?
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


def run():
    print()
    print('#' * 76)
    print('  SESSION 62b -- BOUNDING THE SCHUR COMPLEMENT')
    print('#' * 76)

    # == Part 1: Verify ALL diagonals of M_odd are negative ==
    print('\n  === PART 1: M_ODD DIAGONAL — ALL NEGATIVE? ===')

    for lam_sq in [50, 200, 1000, 5000, 20000, 50000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M, N)
        diag = np.diag(Mo)
        max_diag = diag.max()
        min_diag = diag.min()
        all_neg = max_diag < 0
        print(f'  lam^2={lam_sq:>6d}: diag range [{min_diag:+.4f}, {max_diag:+.4f}], '
              f'all negative: {all_neg}')
    sys.stdout.flush()

    # == Part 2: Decompose Schur into diagonal + coupling ==
    print('\n  === PART 2: SCHUR DECOMPOSITION ===')
    print(f'  s_k = diag_k - coupling_k')
    print(f'  where coupling_k = c_k^T B_k^{{-1}} c_k (positive since B<0)')
    print()

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    _, M, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M, N)

    print(f'  At lam^2={lam_sq}, N={N}:')
    print(f'  {"step":>5} {"diag":>12} {"coupling":>12} {"Schur s":>14} '
          f'{"ratio c/|d|":>12} {"margin":>14}')
    print('  ' + '-' * 72)

    R = Mo.copy()
    schur_data = []
    for step in range(min(20, N)):
        if R.shape[0] <= 1:
            break
        a_k = R[0, 0]
        c_k = R[0, 1:]
        B_k = R[1:, 1:]

        try:
            Binv_c = np.linalg.solve(B_k, c_k)
            coupling = float(c_k @ Binv_c)  # c^T B^{-1} c (negative since B<0)
            neg_coupling = -coupling  # positive: this is what "lifts" toward zero
            s_k = a_k - coupling
            ratio = neg_coupling / abs(a_k) if abs(a_k) > 1e-15 else 0
            margin = abs(s_k)
        except:
            coupling = float('nan')
            neg_coupling = float('nan')
            s_k = float('nan')
            ratio = float('nan')
            margin = float('nan')

        schur_data.append((step, a_k, neg_coupling, s_k, ratio, margin))
        print(f'  {step:>5d} {a_k:>+12.6f} {neg_coupling:>+12.6f} {s_k:>+14.8e} '
              f'{ratio:>12.8f} {margin:>14.8e}')

        R = B_k
    sys.stdout.flush()

    # == Part 3: How does the ratio coupling/|diagonal| vary? ==
    print('\n  === PART 3: COUPLING RATIO vs LAMBDA ===')
    print(f'  At step 0 (n=1, tightest point):')
    print()
    print(f'  {"lam^2":>8} {"diag":>12} {"coupling":>12} {"ratio":>12} '
          f'{"Schur":>14}')
    print('  ' + '-' * 62)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M, N)

        a_k = Mo[0, 0]
        c_k = Mo[0, 1:]
        B_k = Mo[1:, 1:]
        try:
            Binv_c = np.linalg.solve(B_k, c_k)
            coupling = -float(c_k @ Binv_c)
            s_k = a_k + coupling  # a_k is negative, coupling is positive
            ratio = coupling / abs(a_k)
        except:
            coupling = float('nan')
            s_k = float('nan')
            ratio = float('nan')
        print(f'  {lam_sq:>8d} {a_k:>+12.4f} {coupling:>+12.4f} {ratio:>12.8f} '
              f'{s_k:>+14.8e}')
    sys.stdout.flush()

    # == Part 4: The coupling vector anatomy ==
    print('\n  === PART 4: COUPLING VECTOR c AT STEP 0 ===')
    print(f'  c_k = M_odd[0, k] for k=1,...,N-1')
    print(f'  These are the off-diagonal couplings from n=1 to n=2,...,N')
    print()

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    _, M, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M, N)
    c = Mo[0, 1:]

    print(f'  First 15 entries of c (lam^2=1000):')
    for k in range(min(15, len(c))):
        n_target = k + 2  # coupling to n=k+2
        print(f'    c[{k}] (n=1 -> n={n_target}): {c[k]:+.6f}')

    print(f'\n  c norm: {np.linalg.norm(c):.6f}')
    print(f'  c decay: |c[k]| ~ ?')

    # Fit |c[k]| to power law
    k_vals = np.arange(1, len(c) + 1, dtype=float)
    abs_c = np.abs(c)
    mask = abs_c > 1e-10
    if np.sum(mask) > 5:
        log_k = np.log(k_vals[mask])
        log_c = np.log(abs_c[mask])
        slope, intercept = np.polyfit(log_k, log_c, 1)
        print(f'  |c[k]| ~ {np.exp(intercept):.4f} * k^{slope:.3f}')
    sys.stdout.flush()

    # == Part 5: Can we bound c^T B^{-1} c analytically? ==
    print('\n  === PART 5: BOUNDING c^T B^{-1} c ===')
    print(f'  Since B < 0: c^T B^{{-1}} c < 0, so -c^T B^{{-1}} c > 0.')
    print(f'  Bound: -c^T B^{{-1}} c <= ||c||^2 / |lambda_max(B)|')
    print(f'  where lambda_max(B) is the LEAST negative eigenvalue of B.')
    print()

    for lam_sq in [200, 1000, 5000, 20000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M, N)

        a_k = Mo[0, 0]
        c_k = Mo[0, 1:]
        B_k = Mo[1:, 1:]
        eB = np.linalg.eigvalsh(B_k)
        lambda_max_B = eB[-1]  # least negative

        Binv_c = np.linalg.solve(B_k, c_k)
        actual_coupling = -float(c_k @ Binv_c)
        bound = np.linalg.norm(c_k)**2 / abs(lambda_max_B)

        print(f'  lam^2={lam_sq:>6d}:')
        print(f'    |a_1| = {abs(a_k):.4f}')
        print(f'    actual coupling = {actual_coupling:.6f}')
        print(f'    ||c||^2/|lam_max(B)| = {bound:.4f}')
        print(f'    ratio actual/bound = {actual_coupling/bound:.6f}')
        print(f'    need: coupling < |a_1|: {actual_coupling < abs(a_k)} '
              f'(margin {abs(a_k) - actual_coupling:.6e})')
    sys.stdout.flush()

    # == Part 6: What if we use a TIGHTER bound? ==
    print('\n  === PART 6: TIGHT BOUND via EIGENVALUE DECOMPOSITION ===')
    print(f'  c^T B^{{-1}} c = sum_j (c . v_j)^2 / lambda_j')
    print(f'  where (v_j, lambda_j) are eigenpairs of B.')
    print(f'  Since all lambda_j < 0: each term is negative.')
    print(f'  The sum is dominated by the term with smallest |lambda_j|.')
    print()

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    _, M, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M, N)
    c = Mo[0, 1:]
    B = Mo[1:, 1:]
    eB, vB = np.linalg.eigh(B)

    # Decompose coupling into eigencomponents
    projections = np.array([float(c @ vB[:, j])**2 for j in range(len(eB))])
    contributions = projections / eB  # each is negative

    print(f'  Top contributions to c^T B^{{-1}} c (most negative lambda):')
    order = np.argsort(np.abs(contributions))[::-1]
    for i in range(min(10, len(order))):
        j = order[i]
        print(f'    j={j:>3d}: lambda={eB[j]:+.6f}, '
              f'(c.v)^2={projections[j]:.6f}, '
              f'contrib={contributions[j]:+.8f}')

    total = contributions.sum()
    print(f'\n  Total c^T B^{{-1}} c = {total:+.10f}')
    print(f'  Dominated by least-negative eigenvalue: '
          f'lambda={eB[-1]:+.8f}')
    print(f'  Top-1 contribution accounts for '
          f'{abs(contributions[order[0]]/total)*100:.1f}% of total')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
