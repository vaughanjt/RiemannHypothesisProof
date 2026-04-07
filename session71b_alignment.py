"""
SESSION 71b -- POSITIVE EIGENVECTOR AND LOG-CONCAVITY

Session 71 found: the xi coefficient vector x_n = c_{|n|} gives x^T M x > 0.
This means x is aligned with M's unique positive eigenspace.

Key questions:
  1. What does M's positive eigenvector v_+ look like?
  2. Is v_+ itself a "log-concave" sequence (|v_+[n]| log-concave in |n|)?
  3. How close is v_+ to the coefficient vector c_{|n|}?
  4. The DEEP question: does Lorentzian signature of M FORCE the positive
     eigenvector to be log-concave? If so, this would connect:
     M Lorentzian (our RH-equiv) -> v_+ log-concave -> c_k log-concave -> LP -> RH
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import build_all_fast


def analyze_positive_eigenvector(lam_sq):
    """Extract and analyze M's positive eigenvector."""
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    dim = 2 * N + 1

    W02, M, QW = build_all_fast(lam_sq, N)

    evals, evecs = np.linalg.eigh(M)
    v_plus = evecs[:, -1]  # positive eigenvector

    # Ensure consistent sign: v_+[N] > 0 (center element positive)
    if v_plus[N] < 0:
        v_plus = -v_plus

    # Extract the even part: v_even[k] = v_+[N+k] + v_+[N-k] for k>0
    # (M's positive eigenvector is purely even by Session 57)
    v_even = np.zeros(N + 1)
    v_even[0] = v_plus[N]
    for k in range(1, N + 1):
        v_even[k] = (v_plus[N + k] + v_plus[N - k]) / 2

    # Normalize: v_even[0] = 1
    v_even = v_even / v_even[0]

    return v_plus, v_even, M, evals, evecs, N, L


def check_log_concavity(seq, label=''):
    """Check if |seq[k]| is log-concave: seq[k]^2 >= seq[k-1]*seq[k+1]."""
    print(f'  Log-concavity of {label}:')
    print(f'  {"k":>3} {"value":>18} {"R_k":>14} {">= 1?":>8}')
    print('  ' + '-' * 46)

    ratios = []
    for k in range(1, len(seq) - 1):
        if abs(seq[k - 1]) > 1e-50 and abs(seq[k + 1]) > 1e-50:
            R = seq[k]**2 / (seq[k - 1] * seq[k + 1])
            ok = R >= 1 - 1e-10
            ratios.append((k, R, ok))
            if k <= 15 or not ok:
                print(f'  {k:>3d} {seq[k]:>+18.10e} {R:>14.8f} '
                      f'{"YES" if ok else "**NO**":>8}')

    n_pass = sum(1 for _, _, ok in ratios if ok)
    n_total = len(ratios)
    print(f'  Result: {n_pass}/{n_total} pass log-concavity')
    return ratios


def run():
    print()
    print('#' * 76)
    print('  SESSION 71b -- POSITIVE EIGENVECTOR AND LOG-CONCAVITY')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Extract positive eigenvector at several lambda
    # ==================================================================
    print(f'\n  === STEP 1: POSITIVE EIGENVECTORS ===\n')

    for lam_sq in [50, 200, 1000, 5000]:
        v_plus, v_even, M, evals, evecs, N, L = analyze_positive_eigenvector(lam_sq)

        print(f'  lambda^2 = {lam_sq}, N = {N}, dim = {2*N+1}')
        print(f'  v_even (first 12 components, normalized v_even[0]=1):')
        for k in range(min(12, N + 1)):
            print(f'    v_even[{k:>2d}] = {v_even[k]:>+18.10e}')

        # Check log-concavity of v_even
        check_log_concavity(v_even[:min(N + 1, 20)], f'v_even (lam^2={lam_sq})')
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Compare v_even to xi coefficients
    # ==================================================================
    print(f'  === STEP 2: v_even vs c_k COMPARISON ===\n')

    import mpmath
    from mpmath import mp, mpf
    mp.dps = 50

    def xi_func(s):
        return mpf('0.5') * s * (s - 1) * mpmath.power(mpmath.pi, -s / 2) * \
               mpmath.gamma(s / 2) * mpmath.zeta(s)

    K = 12
    print(f'  Computing xi coefficients c_0..c_{K}...')
    s = mpf('0.5')
    xi_val = xi_func(s)
    c_xi = [1.0]
    for k in range(1, K + 1):
        deriv = mpmath.diff(xi_func, s, n=2 * k)
        c_xi.append(float(deriv / xi_val * mpf(-1)**k / mpmath.factorial(2 * k)))
    c_xi = np.array(c_xi)

    # Normalize c_xi[0] = 1 (already true)
    print(f'  c_xi[0] = {c_xi[0]}, c_xi[1] = {c_xi[1]:.6e}')

    for lam_sq in [200, 1000, 5000]:
        v_plus, v_even, M, evals, evecs, N, L = analyze_positive_eigenvector(lam_sq)

        print(f'\n  Comparison at lambda^2 = {lam_sq}:')
        print(f'  {"k":>3} {"v_even[k]":>18} {"c_k":>18} {"ratio v/c":>14} {"log ratio":>12}')
        print('  ' + '-' * 68)

        K_comp = min(K, N)
        for k in range(K_comp + 1):
            ratio = v_even[k] / c_xi[k] if abs(c_xi[k]) > 1e-50 else 0
            log_ratio = np.log(abs(ratio)) if abs(ratio) > 1e-50 else 0
            print(f'  {k:>3d} {v_even[k]:>+18.10e} {c_xi[k]:>+18.10e} {ratio:>14.6f} {log_ratio:>12.4f}')

        # Are they proportional? Compute ratio sequence
        ratios = [v_even[k] / c_xi[k] for k in range(1, K_comp + 1)
                  if abs(c_xi[k]) > 1e-50]
        if len(ratios) >= 2:
            print(f'  Ratio v/c spread: min={min(ratios):.4f}, max={max(ratios):.4f}')
            print(f'  NOT proportional (v_even != const * c_k)')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: Quotient structure of v_even
    # ==================================================================
    print(f'\n  === STEP 3: QUOTIENT STRUCTURE ===\n')

    for lam_sq in [200, 1000, 5000]:
        v_plus, v_even, M, evals, evecs, N, L = analyze_positive_eigenvector(lam_sq)

        print(f'  lambda^2 = {lam_sq}:')
        print(f'  {"k":>3} {"v_even[k]/v_even[k-1]":>22} {"|ratio of quotients|":>22}')
        print('  ' + '-' * 50)

        q_prev = None
        for k in range(1, min(N + 1, 15)):
            q = v_even[k] / v_even[k - 1] if abs(v_even[k - 1]) > 1e-30 else 0
            ratio = abs(q / q_prev) if q_prev and abs(q_prev) > 1e-30 else 0
            print(f'  {k:>3d} {q:>+22.14e} {ratio:>22.8f}')
            q_prev = q
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: Does M Lorentzian -> v_+ log-concave?
    # ==================================================================
    print(f'  === STEP 4: LORENTZIAN -> LOG-CONCAVE TEST ===\n')

    # Generate random Lorentzian matrices and check if positive eigenvector
    # is always log-concave

    np.random.seed(42)
    n_random = 100
    n_pass = 0
    n_fail = 0
    d = 15

    for trial in range(n_random):
        # Generate random Lorentzian matrix: A = v v^T - D
        # where D is positive definite diagonal (makes all but 1 eig negative)
        v = np.random.randn(d)
        diag_vals = np.abs(np.random.randn(d)) * 2 + 0.1
        A = np.outer(v, v) - np.diag(diag_vals)

        evals_A = np.linalg.eigvalsh(A)
        n_pos_A = np.sum(evals_A > 1e-10)

        if n_pos_A != 1:
            continue  # skip non-Lorentzian

        # Extract positive eigenvector
        _, evecs_A = np.linalg.eigh(A)
        v_pos = evecs_A[:, -1]
        if v_pos[0] < 0:
            v_pos = -v_pos

        # Check log-concavity of |v_pos|
        abs_v = np.abs(v_pos)
        is_lc = True
        for k in range(1, d - 1):
            if abs_v[k - 1] > 1e-10 and abs_v[k + 1] > 1e-10:
                if abs_v[k]**2 < abs_v[k - 1] * abs_v[k + 1] * (1 - 1e-8):
                    is_lc = False
                    break

        if is_lc:
            n_pass += 1
        else:
            n_fail += 1

    print(f'  Random Lorentzian matrices (d={d}, {n_random} trials):')
    print(f'  v_+ log-concave: {n_pass}/{n_pass + n_fail}')
    print(f'  v_+ NOT log-concave: {n_fail}/{n_pass + n_fail}')
    print()

    if n_fail > 0:
        print(f'  CONCLUSION: Lorentzian signature does NOT automatically')
        print(f'  imply log-concavity of the positive eigenvector.')
        print(f'  The log-concavity of M\'s v_+ is EXTRA structure beyond Lorentzian.')
    else:
        print(f'  CAUTION: all random tests passed, but sample may not cover')
        print(f'  adversarial cases. Lorentzian might imply log-concavity for')
        print(f'  this class of matrices.')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: The DEEP test -- does v_+(lambda) relate to c_k functionally?
    # ==================================================================
    print(f'\n  === STEP 5: v_+ AS FUNCTION OF LAMBDA ===\n')

    # Track how v_even[k] changes with lambda for fixed small k
    print(f'  {"lam^2":>8} {"v[1]":>14} {"v[2]":>14} {"v[3]":>14} {"v[4]":>14} {"v[5]":>14}')
    print('  ' + '-' * 78)

    for lam_sq in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        try:
            v_plus, v_even, M, evals, evecs, N, L = analyze_positive_eigenvector(lam_sq)
            vals = [v_even[k] if k < len(v_even) else 0 for k in range(1, 6)]
            print(f'  {lam_sq:>8d} ' + ' '.join(f'{v:>+14.8e}' for v in vals))
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 71b VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
