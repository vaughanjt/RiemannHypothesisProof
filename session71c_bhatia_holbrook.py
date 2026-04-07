"""
SESSION 71c -- BHATIA-HOLBROOK LOEWNER MATRIX INERTIA

Bhatia-Holbrook (2000) / Bhatia-Friedland-Jain (2015, arXiv:1501.01505):
For Loewner matrix L_f[i,j] = (f(p_i) - f(p_j))/(p_i - p_j):
  - f(t) = t^r, r in (0,1): L is PSD
  - f(t) = t^r, r in (1,2): L has EXACTLY 1 positive eigenvalue (Lorentzian!)
  - f(t) = t^r, r in (2,3): L has (n-1) positive eigenvalues

Our M has exact Cauchy off-diagonal (Session 59b):
  M[n,m] = a_n * delta + (B_m - B_n) / (n - m)

This IS a Loewner matrix of the function B, at integer nodes, plus diagonal.

KEY QUESTION: Does B_n behave like n^r for some r in (1,2)?
If so, Bhatia-Holbrook predicts Lorentzian signature for the Cauchy part.
Then the diagonal a_n just needs to not destroy it.

Plan:
  1. Extract B_n and a_n from M at several lambda
  2. Analyze B_n: fit to n^r, check divided difference signs
  3. Build pure Loewner matrix (without diagonal) and check signature
  4. Understand the role of the diagonal perturbation
  5. Check Bhatia-Holbrook's exact conditions
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _build_M_prime, _compute_alpha, _compute_wr_diag
)


def extract_cauchy(lam_sq):
    """Extract a_n, B_n from M decomposition."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    primes = sieve_primes(int(lam_sq))
    a_prime = np.zeros(dim)
    B_prime = np.zeros(dim)

    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            a_prime += w * 2 * np.cos(2 * np.pi * ns * y / L)
            B_prime += w * np.sin(2 * np.pi * ns * y / L) / np.pi
            pk *= int(p)

    a_n = np.array([wr[abs(int(n))] for n in ns]) + a_prime
    B_n = alpha + B_prime

    # Build actual M for reference
    _, M, _ = build_all_fast(lam_sq, N)

    return a_n, B_n, M, N, L, dim, ns


def build_pure_loewner(B_n, ns):
    """Build the pure Loewner matrix L[i,j] = (B_j - B_i)/(j - i).
    Diagonal: L[i,i] = B'(i) estimated by centered difference.
    """
    dim = len(ns)
    nm = ns[:, None] - ns[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        L = (B_n[None, :] - B_n[:, None]) / nm

    # Fill diagonal with centered finite difference: B'(n) ~ (B_{n+1} - B_{n-1})/2
    for i in range(dim):
        if 0 < i < dim - 1:
            L[i, i] = (B_n[i + 1] - B_n[i - 1]) / (ns[i + 1] - ns[i - 1])
        elif i == 0:
            L[i, i] = (B_n[1] - B_n[0]) / (ns[1] - ns[0])
        else:
            L[i, i] = (B_n[-1] - B_n[-2]) / (ns[-1] - ns[-2])

    # Symmetrize
    L = (L + L.T) / 2
    return L


def analyze_function_class(B_n, ns, N):
    """Analyze what function class B_n belongs to.

    Bhatia-Holbrook conditions for Lorentzian Loewner matrix:
    f must be in the class C^{n-1} with the divided differences having
    a specific sign pattern.

    For f(t) = t^r, r in (1,2):
    - f' > 0 (increasing)
    - f'' > 0 (convex)
    - f''' < 0
    """
    # Use only positive n (B is odd in n for the alpha part)
    # Focus on even part for the even block analysis
    idx_center = N  # index of n=0
    n_pos = ns[idx_center:]  # n = 0, 1, ..., N
    B_pos = B_n[idx_center:]

    print(f'  B_n for n = 0, 1, ..., {N}:')
    for k in range(min(15, N + 1)):
        print(f'    B[{k:>2d}] = {B_pos[k]:>+18.10e}')

    # First differences: Delta B[k] = B[k+1] - B[k]
    dB = np.diff(B_pos)
    print(f'\n  First differences (B[k+1] - B[k]):')
    for k in range(min(12, len(dB))):
        print(f'    dB[{k:>2d}] = {dB[k]:>+18.10e}')

    n_pos_dB = np.sum(dB[:N] > 0)
    n_neg_dB = np.sum(dB[:N] < 0)
    print(f'  Sign pattern of dB: {n_pos_dB} positive, {n_neg_dB} negative')

    # Second differences: Delta^2 B[k] = dB[k+1] - dB[k]
    d2B = np.diff(dB)
    print(f'\n  Second differences:')
    for k in range(min(12, len(d2B))):
        print(f'    d2B[{k:>2d}] = {d2B[k]:>+18.10e}')

    n_pos_d2B = np.sum(d2B[:N-1] > 0)
    n_neg_d2B = np.sum(d2B[:N-1] < 0)
    print(f'  Sign pattern of d2B: {n_pos_d2B} positive, {n_neg_d2B} negative')

    # Third differences
    d3B = np.diff(d2B)
    print(f'\n  Third differences:')
    for k in range(min(12, len(d3B))):
        print(f'    d3B[{k:>2d}] = {d3B[k]:>+18.10e}')

    n_pos_d3B = np.sum(d3B[:N-2] > 0)
    n_neg_d3B = np.sum(d3B[:N-2] < 0)
    print(f'  Sign pattern of d3B: {n_pos_d3B} positive, {n_neg_d3B} negative')

    # For r in (1,2): need dB > 0, d2B > 0, d3B < 0
    print(f'\n  Bhatia-Holbrook pattern for r in (1,2):')
    print(f'    dB > 0 (increasing):  {n_pos_dB}/{n_pos_dB+n_neg_dB}')
    print(f'    d2B > 0 (convex):     {n_pos_d2B}/{n_pos_d2B+n_neg_d2B}')
    print(f'    d3B < 0:              {n_neg_d3B}/{n_pos_d3B+n_neg_d3B}')

    # Fit B_pos to n^r for n >= 1
    n_fit = n_pos[1:min(N+1, 20)]  # avoid n=0
    B_fit = B_pos[1:min(N+1, 20)]

    # log(B) vs log(n) for power law fit (only if B > 0)
    valid = B_fit > 0
    if np.sum(valid) >= 3:
        log_n = np.log(n_fit[valid])
        log_B = np.log(B_fit[valid])
        fit = np.polyfit(log_n, log_B, 1)
        r_est = fit[0]
        print(f'\n  Power-law fit (B ~ n^r for n >= 1):')
        print(f'    r = {r_est:.6f}')
        print(f'    In (1,2)? {"YES" if 1 < r_est < 2 else "NO"}')
    else:
        print(f'\n  Power-law fit: not enough positive B values')
        r_est = None

    return B_pos, dB, d2B, d3B, r_est


def run():
    print()
    print('#' * 76)
    print('  SESSION 71c -- BHATIA-HOLBROOK LOEWNER INERTIA')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Extract B_n and analyze function class
    # ==================================================================
    print(f'\n  === STEP 1: B_n FUNCTION CLASS ===\n')

    for lam_sq in [200, 1000, 5000]:
        print(f'  --- lambda^2 = {lam_sq} ---')
        a_n, B_n, M, N, L, dim, ns = extract_cauchy(lam_sq)

        B_pos, dB, d2B, d3B, r_est = analyze_function_class(B_n, ns, N)
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Pure Loewner matrix signature
    # ==================================================================
    print(f'  === STEP 2: PURE LOEWNER MATRIX (NO DIAGONAL) ===\n')

    for lam_sq in [200, 1000, 5000]:
        a_n, B_n, M, N, L, dim, ns = extract_cauchy(lam_sq)

        L_pure = build_pure_loewner(B_n, ns)
        evals_L = np.linalg.eigvalsh(L_pure)

        n_pos = np.sum(evals_L > 1e-10)
        n_neg = np.sum(evals_L < -1e-10)

        print(f'  lam^2={lam_sq}: Pure Loewner sig ({n_pos}+, {n_neg}-, {dim-n_pos-n_neg}z)')
        print(f'    Top 5: {evals_L[-5:][::-1]}')
        print(f'    Bot 5: {evals_L[:5]}')

        # Full M signature for comparison
        evals_M = np.linalg.eigvalsh(M)
        n_pos_M = np.sum(evals_M > 1e-10)
        n_neg_M = np.sum(evals_M < -1e-10)
        print(f'    Full M sig: ({n_pos_M}+, {n_neg_M}-)')
        print(f'    M top: {evals_M[-1]:.6f}, M 2nd: {evals_M[-2]:.6e}')
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: Role of diagonal perturbation
    # ==================================================================
    print(f'  === STEP 3: DIAGONAL PERTURBATION ANALYSIS ===\n')

    for lam_sq in [1000]:
        a_n, B_n, M, N, L, dim, ns = extract_cauchy(lam_sq)

        L_pure = build_pure_loewner(B_n, ns)

        # Diagonal perturbation: D = M - L_pure (off-diagonal should be ~0)
        D = M - L_pure
        D_diag = np.diag(D)

        print(f'  lam^2={lam_sq}:')
        print(f'  Diagonal perturbation D = diag(M) - diag(L_pure):')
        for k in range(min(10, N + 1)):
            idx = N + k
            print(f'    n={k:>2d}: a_n={a_n[idx]:>+12.6f}, L_diag={L_pure[idx,idx]:>+12.6f}, '
                  f'D={D_diag[idx]:>+12.6f}')

        # Off-diagonal residual (should be zero)
        mask = ~np.eye(dim, dtype=bool)
        offdiag_resid = np.abs(D[mask]).max()
        print(f'  Max off-diagonal residual: {offdiag_resid:.2e}')

        # Eigenvalue tracking: L_pure, L_pure + t*D for t=0..1
        print(f'\n  Eigenvalue evolution: L_pure + t*diag(D)')
        print(f'  {"t":>6} {"#pos":>5} {"eig_1":>12} {"eig_2":>12}')
        print('  ' + '-' * 38)
        for t in [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]:
            M_t = L_pure + t * np.diag(D_diag)
            evals_t = np.linalg.eigvalsh(M_t)
            n_pos_t = np.sum(evals_t > 1e-10)
            print(f'  {t:>6.2f} {n_pos_t:>5d} {evals_t[-1]:>+12.4f} {evals_t[-2]:>+12.6e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: Even subspace Loewner
    # ==================================================================
    print(f'\n  === STEP 4: EVEN SUBSPACE LOEWNER ===\n')

    for lam_sq in [200, 1000, 5000]:
        a_n, B_n, M, N, L, dim, ns = extract_cauchy(lam_sq)

        # Even projection
        dim_even = N + 1
        P_even = np.zeros((dim, dim_even))
        P_even[N, 0] = 1.0
        for k in range(1, N + 1):
            P_even[N + k, k] = 1.0 / np.sqrt(2)
            P_even[N - k, k] = 1.0 / np.sqrt(2)

        M_even = P_even.T @ M @ P_even
        L_even = P_even.T @ build_pure_loewner(B_n, ns) @ P_even

        evals_Me = np.linalg.eigvalsh(M_even)
        evals_Le = np.linalg.eigvalsh(L_even)

        n_pos_Me = np.sum(evals_Me > 1e-10)
        n_pos_Le = np.sum(evals_Le > 1e-10)

        print(f'  lam^2={lam_sq}: M_even sig ({n_pos_Me}+), '
              f'L_even sig ({n_pos_Le}+)')
        print(f'    M_even top: {evals_Me[-1]:.4f}, 2nd: {evals_Me[-2]:.6e}')
        print(f'    L_even top: {evals_Le[-1]:.4f}, 2nd: {evals_Le[-2]:.6e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: Odd subspace Loewner
    # ==================================================================
    print(f'\n  === STEP 5: ODD SUBSPACE LOEWNER ===\n')

    for lam_sq in [200, 1000, 5000]:
        a_n, B_n, M, N, L, dim, ns = extract_cauchy(lam_sq)

        dim_odd = N
        P_odd = np.zeros((dim, dim_odd))
        for k in range(1, N + 1):
            P_odd[N + k, k - 1] = 1.0 / np.sqrt(2)
            P_odd[N - k, k - 1] = -1.0 / np.sqrt(2)

        M_odd = P_odd.T @ M @ P_odd
        L_odd = P_odd.T @ build_pure_loewner(B_n, ns) @ P_odd

        evals_Mo = np.linalg.eigvalsh(M_odd)
        evals_Lo = np.linalg.eigvalsh(L_odd)

        n_pos_Mo = np.sum(evals_Mo > 1e-10)
        n_neg_Mo = np.sum(evals_Mo < -1e-10)
        n_pos_Lo = np.sum(evals_Lo > 1e-10)
        n_neg_Lo = np.sum(evals_Lo < -1e-10)

        print(f'  lam^2={lam_sq}: M_odd ({n_pos_Mo}+, {n_neg_Mo}-), '
              f'L_odd ({n_pos_Lo}+, {n_neg_Lo}-)')
        print(f'    M_odd top: {evals_Mo[-1]:.6e}')
        print(f'    L_odd top: {evals_Lo[-1]:.6e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: Comparison with t^r Loewner matrices
    # ==================================================================
    print(f'\n  === STEP 6: t^r LOEWNER COMPARISON ===\n')

    # Build Loewner matrices of f(t) = |t|^r at integer nodes -N..N
    N_test = 20
    ns_test = np.arange(-N_test, N_test + 1, dtype=float)
    dim_test = 2 * N_test + 1

    print(f'  Loewner matrix of f(t) = sign(t)*|t|^r at nodes {{-{N_test},...,{N_test}}}:')
    print(f'  {"r":>6} {"#pos":>5} {"eig_1":>12} {"eig_2":>12}')
    print('  ' + '-' * 38)

    for r in [0.5, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]:
        # f(t) = sign(t) * |t|^r to handle negative t
        f_vals = np.sign(ns_test) * np.abs(ns_test)**r
        f_vals[N_test] = 0  # f(0) = 0

        nm_test = ns_test[:, None] - ns_test[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            L_test = (f_vals[None, :] - f_vals[:, None]) / nm_test

        # Fill diagonal
        for i in range(dim_test):
            if 0 < i < dim_test - 1:
                L_test[i, i] = (f_vals[i + 1] - f_vals[i - 1]) / 2
            elif i == 0:
                L_test[i, i] = f_vals[1] - f_vals[0]
            else:
                L_test[i, i] = f_vals[-1] - f_vals[-2]

        L_test = (L_test + L_test.T) / 2
        evals_test = np.linalg.eigvalsh(L_test)
        n_pos_test = np.sum(evals_test > 1e-10)

        print(f'  {r:>6.2f} {n_pos_test:>5d} {evals_test[-1]:>+12.4f} {evals_test[-2]:>+12.6e}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 71c VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
