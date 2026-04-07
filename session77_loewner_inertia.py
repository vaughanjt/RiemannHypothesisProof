"""
SESSION 77 -- GEOMETRY LAND: LOEWNER INERTIA AND CND STRUCTURE

M has exact Cauchy-Loewner structure (Session 59b):
  M[n,m] = a_n * delta_{nm} + (B_m - B_n)/(n - m)   for n != m

Bharali-Holtz (arXiv:1501.01505) proved: the Loewner matrix
  L[i,j] = (f(x_i) - f(x_j)) / (x_i - x_j)
has signature (1, n-1) when f is a power function x^r with 1 < r < 2.

Key question: does our function B(n) have properties that force
the Loewner matrix to have a specific inertia?

Also: is M (or M_odd) CONDITIONALLY NEGATIVE DEFINITE?
CND means x^T M x <= 0 whenever sum(x_i) = 0.
If M_odd is CND: Schoenberg => exp(-tM_odd) PD for all t > 0
=> all eigenvalues same sign => neg def (since trace < 0).

Probes:
  1. Inertia of the pure Cauchy part L[n,m] = (B_m - B_n)/(n-m)
  2. Is L conditionally negative definite?
  3. Is M_odd conditionally negative definite?
  4. Decompose: M = L + diag(a). What does diag(a) do to inertia?
  5. What function class does B(n) belong to? Convexity, monotonicity?
  6. Test Bharali-Holtz conditions on our specific nodes and function.
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def decompose_LDa(lam_sq, N=None):
    """Decompose M = L_cauchy + diag(a_n) exactly."""
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    _, M, QW = build_all_fast(lam_sq, N)

    # Compute B_n for Cauchy off-diagonal
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

    B_n = alpha + B_prime
    a_n = np.array([wr[abs(int(n))] for n in ns]) + a_prime

    # Build pure Cauchy matrix L[i,j] = (B_j - B_i) / (n_i - n_j)
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        L_cauchy = (B_n[None, :] - B_n[:, None]) / nm

    # Diagonal: B'(n) by finite difference
    for i in range(dim):
        if 0 < i < dim - 1:
            L_cauchy[i, i] = (B_n[i + 1] - B_n[i - 1]) / 2
        elif i == 0:
            L_cauchy[i, i] = B_n[1] - B_n[0]
        else:
            L_cauchy[i, i] = B_n[-1] - B_n[-2]
    L_cauchy = (L_cauchy + L_cauchy.T) / 2

    # Diagonal remainder
    D_a = np.diag(M) - np.diag(L_cauchy)

    # Verify
    err = np.max(np.abs(M - L_cauchy - np.diag(D_a)))

    return L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P, P


def is_CND(M, verbose=False):
    """Test if M is conditionally negative definite.
    CND: x^T M x <= 0 for all x with sum(x_i) = 0.
    Equivalently: M restricted to the orthogonal complement of 1
    is negative semidefinite.
    """
    d = M.shape[0]
    ones = np.ones(d)
    ones /= np.linalg.norm(ones)

    # Project M onto sum-zero subspace
    # P = I - (1/d)*ones*ones^T
    P_proj = np.eye(d) - np.outer(ones, ones)
    M_proj = P_proj @ M @ P_proj

    # Eigenvalues of projected matrix (one will be ~0 from projection)
    evals = np.linalg.eigvalsh(M_proj)
    # Remove the eigenvalue corresponding to the ones direction
    evals_sumzero = evals[np.abs(evals) > 1e-12 * np.max(np.abs(evals))]

    max_eval = evals_sumzero.max() if len(evals_sumzero) > 0 else 0
    is_cnd = max_eval <= 1e-10

    if verbose:
        print(f'    max eigenvalue on sum-zero: {max_eval:.6e}')
        print(f'    min eigenvalue on sum-zero: {evals_sumzero.min():.6e}')
        print(f'    CND: {is_cnd}')

    return is_cnd, max_eval, evals_sumzero


def run():
    print()
    print('#' * 76)
    print('  SESSION 77 -- LOEWNER INERTIA AND CND STRUCTURE')
    print('#' * 76)

    # ======================================================================
    # PROBE 1: Inertia of the pure Cauchy part L
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: INERTIA OF PURE CAUCHY MATRIX L')
    print(f'{"="*76}\n')

    for lam_sq in [200, 1000, 5000]:
        L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err = decompose_LDa(lam_sq)

        evals_L = np.linalg.eigvalsh(L_cauchy)
        n_pos = np.sum(evals_L > 1e-10)
        n_neg = np.sum(evals_L < -1e-10)
        n_zero = dim - n_pos - n_neg

        print(f'  lam^2={lam_sq}: dim={dim}, decomp err={err:.2e}')
        print(f'    L_cauchy: #pos={n_pos}, #neg={n_neg}, #zero={n_zero}')
        print(f'    L evals: [{evals_L.min():.4f}, {evals_L.max():.4f}]')
        print(f'    tr(L) = {np.trace(L_cauchy):.6f}')

        # Inertia of M for comparison
        evals_M = np.linalg.eigvalsh(M)
        n_pos_M = np.sum(evals_M > 1e-10)
        print(f'    M:       #pos={n_pos_M} (Lorentzian = 1)')

        # Inertia of diag(D_a)
        n_pos_D = np.sum(D_a > 1e-10)
        n_neg_D = np.sum(D_a < -1e-10)
        print(f'    diag(a): #pos={n_pos_D}, #neg={n_neg_D}, '
              f'range=[{D_a.min():.4f}, {D_a.max():.4f}]')
        print()
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Is L conditionally negative definite?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: IS L (PURE CAUCHY) CONDITIONALLY NEGATIVE DEFINITE?')
    print(f'{"="*76}\n')

    for lam_sq in [200, 1000, 5000]:
        L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err = decompose_LDa(lam_sq)

        print(f'  lam^2={lam_sq}:')
        cnd, max_ev, _ = is_CND(L_cauchy, verbose=True)

        # Also test -L (maybe L is conditionally POSITIVE definite)
        cnd_neg, max_ev_neg, _ = is_CND(-L_cauchy)
        print(f'    -L CND (L is CPD)? {cnd_neg} (max_ev={max_ev_neg:.6e})')
        print()
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: Is M conditionally negative definite?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: IS M CONDITIONALLY NEGATIVE DEFINITE?')
    print(f'{"="*76}\n')

    for lam_sq in [200, 1000, 5000]:
        L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err = decompose_LDa(lam_sq)

        print(f'  lam^2={lam_sq}:')
        cnd_M, max_ev_M, evals_sz_M = is_CND(M, verbose=True)
        print()
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: Is M_odd conditionally negative definite?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: IS M_ODD CONDITIONALLY NEGATIVE DEFINITE?')
    print(f'{"="*76}\n')

    for lam_sq in [200, 1000, 5000]:
        L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err = decompose_LDa(lam_sq)
        Mo, P = odd_block(M, N)
        Lo, _ = odd_block(L_cauchy, N)

        print(f'  lam^2={lam_sq}:')
        print(f'  M_odd:')
        cnd_Mo, max_ev_Mo, evals_sz_Mo = is_CND(Mo, verbose=True)

        print(f'  L_odd (pure Cauchy, odd block):')
        cnd_Lo, max_ev_Lo, _ = is_CND(Lo, verbose=True)
        print()
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: B(n) function properties — convexity, monotonicity
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: PROPERTIES OF B(n)')
    print(f'{"="*76}\n')

    lam_sq = 1000
    L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err = decompose_LDa(lam_sq)

    print(f'  B(n) values at lam^2={lam_sq}:')
    print(f'  {"n":>4} {"B(n)":>14} {"B\'(n)":>14} {"B\'\'(n)":>14}')
    print('  ' + '-' * 50)

    # Compute B', B'' by finite differences
    B_prime_fd = np.zeros(dim)
    B_pp_fd = np.zeros(dim)
    for i in range(1, dim - 1):
        B_prime_fd[i] = (B_n[i+1] - B_n[i-1]) / 2
        B_pp_fd[i] = B_n[i+1] - 2*B_n[i] + B_n[i-1]

    for i in range(0, dim, max(1, dim // 20)):
        n = int(ns[i])
        print(f'  {n:>4d} {B_n[i]:>+14.8f} {B_prime_fd[i]:>+14.8f} {B_pp_fd[i]:>+14.8f}')

    # Is B convex? concave?
    n_convex = np.sum(B_pp_fd[1:-1] > 0)
    n_concave = np.sum(B_pp_fd[1:-1] < 0)
    print(f'\n  B\'\'(n): {n_convex} convex points, {n_concave} concave points')
    print(f'  B is {"convex" if n_concave == 0 else "concave" if n_convex == 0 else "NEITHER"}')

    # Is B odd? (B(-n) = -B(n))
    B_odd_err = np.max(np.abs(B_n[N:] + B_n[N::-1]))
    print(f'  B(-n) + B(n) max error: {B_odd_err:.6e} (odd function? {B_odd_err < 1e-10})')

    # Symmetry of a_n
    a_even_err = np.max(np.abs(a_n[N:] - a_n[N::-1]))
    print(f'  a(-n) - a(n) max error: {a_even_err:.6e} (even function? {a_even_err < 1e-10})')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: Loewner function class test
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: LOEWNER FUNCTION CLASS')
    print(f'{"="*76}\n')

    # Bharali-Holtz: for f(x) = x^r, the Loewner matrix
    # L[i,j] = (x_i^r - x_j^r) / (x_i - x_j)
    # has signature (1, n-1) for 1 < r < 2.
    #
    # Our L[n,m] = (B(m) - B(n)) / (n - m) = -(B(m) - B(n)) / (m - n)
    # This is the Loewner matrix of B at the nodes {-N, ..., N}.
    #
    # For this to have Lorentzian signature, B must NOT be operator monotone
    # (which would give PSD). B must behave like x^r for 1 < r < 2.
    #
    # Test: what is the "effective r" of B? Fit B(n) ~ c * |n|^r * sign(n)
    # on positive integers.

    lam_sq = 1000
    L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err = decompose_LDa(lam_sq)

    # B is odd, so look at B(n) for n > 0
    ns_pos = np.arange(1, N + 1, dtype=float)
    B_pos = B_n[N + 1:]  # B(1), B(2), ..., B(N)

    # Fit log|B(n)| = c + r * log(n) for small n
    valid = np.abs(B_pos) > 1e-15
    if np.sum(valid) > 5:
        log_n = np.log(ns_pos[valid])
        log_B = np.log(np.abs(B_pos[valid]))

        # Fit on first 10 points
        k = min(10, np.sum(valid))
        fit_r = np.polyfit(log_n[:k], log_B[:k], 1)
        print(f'  Fit B(n) ~ {np.exp(fit_r[1]):.4f} * n^{fit_r[0]:.4f} for n=1..{k}')

        # Fit on all points
        fit_r_all = np.polyfit(log_n, log_B, 1)
        print(f'  Fit B(n) ~ {np.exp(fit_r_all[1]):.4f} * n^{fit_r_all[0]:.4f} for all n')

        # Is the effective r in (1, 2)?
        r_eff = fit_r[0]
        print(f'  Effective r = {r_eff:.4f}')
        print(f'  In Bharali-Holtz range (1,2)? {1 < r_eff < 2}')

    # Also: is B(n) itself a Pick function evaluated at integers?
    # Pick functions: analytic on upper half-plane, map it to itself
    # Characterized by: B(n) = a + b*n + integral of 1/(t-n) - t/(1+t^2) d\mu(t)
    # Test: does B(n+1) - B(n) have a specific decay pattern?
    diffs = np.diff(B_pos)
    print(f'\n  B(n+1) - B(n) first differences:')
    for i in range(min(15, len(diffs))):
        print(f'    n={i+1}: {diffs[i]:+.8f}')
    print(f'  Monotone decreasing? {np.all(np.diff(diffs) < 0)}')
    print(f'  All positive? {np.all(diffs > 0)}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 7: Direct Bharali-Holtz test — build Loewner matrix of x^r
    #          at our nodes and compare inertia to our L
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 7: BHARALI-HOLTZ COMPARISON')
    print(f'{"="*76}\n')

    # Use small dimension for clarity
    lam_sq = 200
    L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err = decompose_LDa(lam_sq)

    evals_L = np.linalg.eigvalsh(L_cauchy)
    n_pos_L = np.sum(evals_L > 1e-10)
    n_neg_L = np.sum(evals_L < -1e-10)
    print(f'  Our L_cauchy at lam^2={lam_sq}: signature ({n_pos_L}, {n_neg_L})')

    # Build Loewner matrix for f(x) = x^r at nodes ns
    # L_r[i,j] = (n_i^r - n_j^r) / (n_i - n_j) for n_i != n_j
    # Problem: ns contains 0 and negative integers. Use |n|^r * sign(n) for odd extension.
    # Use only positive nodes to avoid x=0 singularity
    ns_pos = ns[ns > 0]
    dim_pos = len(ns_pos)
    print(f'  Using {dim_pos} positive nodes (n=1..{N})')
    for r in [0.5, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5]:
        L_r = np.zeros((dim_pos, dim_pos))
        for i in range(dim_pos):
            for j in range(dim_pos):
                if i == j:
                    L_r[i, i] = r * ns_pos[i]**(r - 1)
                else:
                    L_r[i, j] = (ns_pos[j]**r - ns_pos[i]**r) / (ns_pos[j] - ns_pos[i])
        L_r = (L_r + L_r.T) / 2
        ev_r = np.linalg.eigvalsh(L_r)
        np_r = np.sum(ev_r > 1e-10)
        nn_r = np.sum(ev_r < -1e-10)
        print(f'  f(x) = x^{r:.1f}: signature ({np_r}, {nn_r})')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 8: Operator monotonicity test on B
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 8: IS THE CAUCHY MATRIX PSD? (OPERATOR MONOTONE TEST)')
    print(f'{"="*76}\n')

    # If B is operator monotone, then L = Loewner(B) is PSD.
    # If L has some negative eigenvalues, B is NOT operator monotone.
    # The Bharali-Holtz theory applies when f is "between" r=1 and r=2.

    for lam_sq in [200, 1000, 5000]:
        L_cauchy, D_a, M, B_n, a_n, N, L, dim, ns, err = decompose_LDa(lam_sq)
        evals_L = np.linalg.eigvalsh(L_cauchy)
        n_pos = np.sum(evals_L > 1e-10)
        n_neg = np.sum(evals_L < -1e-10)
        print(f'  lam^2={lam_sq}: L_cauchy signature ({n_pos}, {n_neg}), '
              f'PSD={n_neg == 0}')
        print(f'    => B is {"" if n_neg == 0 else "NOT "}operator monotone')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 77 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
