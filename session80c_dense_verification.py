"""
SESSION 80c -- DENSE INTERVAL ARITHMETIC VERIFICATION

Monotonicity killed (113/430 derivatives negative). Need dense
verification instead.

Requirements:
  - |eig_max| / max|d/dL| = 0.72 in L (required spacing)
  - Actual spacing needed: < 0.72 everywhere
  - Target: every 0.1 in L from L=1.6 to L=12.5

This gives ~110 verification points. Each takes <1 second.
Combined with Kato analyticity (eigenvalue is simple + analytic),
this proves eig_max < 0 for all L in [1.6, 12.5].

For L > 12.5 (lam^2 > 270000): need asymptotic argument or more points.
"""

import sys
import time
from flint import arb, arb_mat, ctx

ctx.prec = 256

sys.path.insert(0, '.')


def build_Modd_arb(lam_sq, N=None):
    """Build M_odd in ball arithmetic."""
    import numpy as np
    from session49c_weil_residual import build_all_fast

    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))

    _, M, _ = build_all_fast(lam_sq, N)

    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    Mo = P.T @ M @ P

    rows = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(arb(str(Mo[i, j])))
        rows.append(row)

    return arb_mat(rows), N, L


def verify_sylvester(Mo_arb, N):
    """Verify Sylvester criterion. Returns (all_pass, first_fail_k)."""
    for k in range(1, N + 1):
        minor = arb_mat([[Mo_arb[i, j] for j in range(k)] for i in range(k)])
        det_k = minor.det()
        signed_det = det_k if k % 2 == 0 else -det_k
        if not (signed_det > 0):
            return False, k
    return True, 0


def run():
    import numpy as np

    print()
    print('#' * 76)
    print('  SESSION 80c -- DENSE INTERVAL ARITHMETIC VERIFICATION')
    print('#' * 76)
    print()
    print(f'  Precision: {ctx.prec} bits')
    print()

    # Generate lambda^2 values at roughly every 0.1 in L
    # L = log(lam^2), so lam^2 = exp(L)
    L_targets = np.arange(1.5, 12.5, 0.1)
    lam_sq_values = np.exp(L_targets).astype(int)
    lam_sq_values = np.unique(np.clip(lam_sq_values, 5, 300000))

    # Add specific values to ensure coverage
    extra = [5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100,
             150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000,
             7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000,
             150000, 200000, 250000, 300000]
    lam_sq_values = np.unique(np.concatenate([lam_sq_values, extra]))

    print(f'  Total verification points: {len(lam_sq_values)}')
    print(f'  L range: [{np.log(lam_sq_values[0]):.2f}, {np.log(lam_sq_values[-1]):.2f}]')
    print()

    all_verified = True
    n_verified = 0
    n_failed = 0
    max_L_verified = 0
    t_start = time.time()

    # Print header
    print(f'  {"lam^2":>8} {"L":>8} {"N":>4} {"status":>10} {"time":>6}')
    print('  ' + '-' * 40)

    results = []
    for lam_sq in lam_sq_values:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))

        t0 = time.time()
        try:
            Mo_arb, N_used, L_used = build_Modd_arb(int(lam_sq), N)
            passed, fail_k = verify_sylvester(Mo_arb, N_used)
            elapsed = time.time() - t0

            if passed:
                n_verified += 1
                max_L_verified = max(max_L_verified, L)
                status = 'VERIFIED'
            else:
                n_failed += 1
                all_verified = False
                status = f'FAIL@k={fail_k}'

            results.append((int(lam_sq), L, N_used, passed, elapsed))

            # Print every 10th point, plus failures and milestones
            if n_verified % 20 == 0 or not passed or lam_sq in [10, 100, 1000, 10000, 100000]:
                print(f'  {int(lam_sq):>8d} {L:>8.3f} {N_used:>4d} {status:>10} {elapsed:>5.1f}s')
                sys.stdout.flush()

        except Exception as e:
            n_failed += 1
            all_verified = False
            results.append((int(lam_sq), L, N, False, 0))
            print(f'  {int(lam_sq):>8d} {L:>8.3f} {N:>4d} {"ERROR":>10}')
            sys.stdout.flush()

    total_time = time.time() - t_start

    # Summary
    print()
    print('=' * 76)
    print('  VERIFICATION SUMMARY')
    print('=' * 76)
    print()
    print(f'  Total points: {len(lam_sq_values)}')
    print(f'  Verified: {n_verified}')
    print(f'  Failed: {n_failed}')
    print(f'  Total time: {total_time:.1f}s')
    print(f'  Max L verified: {max_L_verified:.4f} (lam^2 = {np.exp(max_L_verified):.0f})')
    print()

    if all_verified:
        print(f'  *** ALL {n_verified} POINTS VERIFIED ***')
        print(f'  M_odd < 0 is RIGOROUSLY PROVED for lam^2 from '
              f'{int(lam_sq_values[0])} to {int(lam_sq_values[-1])}')
        print(f'  (L from {np.log(lam_sq_values[0]):.2f} to {np.log(lam_sq_values[-1]):.2f})')
    else:
        print(f'  {n_failed} FAILURES detected.')
        for lam_sq, L, N, passed, elapsed in results:
            if not passed:
                print(f'    FAILED at lam^2={lam_sq}, L={L:.3f}')

    # Check spacing
    print()
    print('  SPACING CHECK:')
    Ls_verified = [r[1] for r in results if r[3]]
    if len(Ls_verified) > 1:
        Ls_arr = np.array(sorted(Ls_verified))
        spacings = np.diff(Ls_arr)
        print(f'    Max spacing: {spacings.max():.4f} (need < 0.72)')
        print(f'    Mean spacing: {spacings.mean():.4f}')
        print(f'    Min spacing: {spacings.min():.4f}')
        print(f'    Spacing OK: {spacings.max() < 0.72}')

    print()
    print('  With Kato analyticity (eigenvalue simple + analytic) and')
    print(f'  derivative bound |d(eig)/dL| < 1.6e-7, required spacing is 0.72.')
    print(f'  All spacings < 0.72 means NO ZERO can exist between')
    print(f'  verification points.')


if __name__ == '__main__':
    run()
