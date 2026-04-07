"""
SESSION 74 -- SPECTRAL INVOLUTION FOR DIVIDED-DIFFERENCE MATRICES

Pushnitski-Treil (arXiv:2504.18707, April 2025) proved:
For weighted Cauchy matrices C[j,k] = sqrt(A_j*A_k)/(a_j+a_k):
  - Displacement: C*D + D*C = v*v^T (rank 1 Lyapunov)
  - Spectral map Omega: (a,A) -> (eigenvalues, projections) is an INVOLUTION

Our M has divided-difference (Loewner) off-diagonal:
  - Displacement: D*M - M*D = 1*B^T - B*1^T (rank 2 Sylvester)

If U diagonalizes M (U^T M U = diag(lam)), then:
  G = U^T D U has off-diagonal G[j,k] = (w_j*z_k - z_j*w_k)/(lam_j - lam_k)
  where w = U^T*1, z = U^T*B.

Questions:
  1. Verify the displacement equation D*M - M*D = 1*B^T - B*1^T
  2. Compute G = U^T D U and check its structure
  3. Is G approximately a divided-difference matrix? (requires z ~ c*w)
  4. What are the eigenvalues of G? (should be the integers -N,...,N)
  5. Does an approximate involution hold?
  6. What does the spectral map look like for our specific M?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def extract_all(lam_sq):
    """Get M, its Cauchy decomposition, B_n, and everything else."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    _, M, QW = build_all_fast(lam_sq, N)

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    primes = sieve_primes(int(lam_sq))
    B_prime = np.zeros(dim)
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            B_prime += w * np.sin(2 * np.pi * ns * y / L) / np.pi
            pk *= int(p)

    B_n = alpha + B_prime
    D = np.diag(ns)
    ones = np.ones(dim)

    return M, D, B_n, ones, ns, N, L, dim


def run():
    print()
    print('#' * 76)
    print('  SESSION 74 -- SPECTRAL INVOLUTION FOR LOEWNER MATRICES')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Verify displacement equation
    # ==================================================================
    print(f'\n  === STEP 1: DISPLACEMENT EQUATION ===\n')

    for lam_sq in [200, 1000]:
        M, D, B_n, ones, ns, N, L, dim = extract_all(lam_sq)

        # Compute D*M - M*D
        commutator = D @ M - M @ D

        # Expected: 1*B^T - B*1^T (rank-2 antisymmetric)
        expected = np.outer(ones, B_n) - np.outer(B_n, ones)

        # Compare off-diagonal (diagonal of commutator should be 0)
        diff = commutator - expected
        max_offdiag = np.max(np.abs(diff - np.diag(np.diag(diff))))
        max_diag = np.max(np.abs(np.diag(commutator)))

        print(f'  lam^2={lam_sq}: dim={dim}')
        print(f'    ||[D,M] - (1*B^T - B*1^T)||_offdiag = {max_offdiag:.2e}')
        print(f'    ||diag([D,M])|| = {max_diag:.2e}')
        print(f'    Displacement rank 2 verified: {max_offdiag < 1e-10}')

        # Check rank of the displacement
        U_disp, s_disp, _ = np.linalg.svd(commutator)
        print(f'    Singular values of [D,M]: {s_disp[:5]}')
        print(f'    Effective rank (>1e-10): {np.sum(s_disp > 1e-10)}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Compute G = U^T D U
    # ==================================================================
    print(f'\n  === STEP 2: THE DUAL MATRIX G = U^T D U ===\n')

    lam_sq = 1000
    M, D, B_n, ones, ns, N, L, dim = extract_all(lam_sq)

    evals_M, U = np.linalg.eigh(M)
    # U diagonalizes M: U^T M U = diag(evals_M)

    G = U.T @ D @ U

    # G should have eigenvalues = ns (integers -N,...,N)
    evals_G = np.linalg.eigvalsh(G)
    evals_G_sorted = np.sort(evals_G)
    ns_sorted = np.sort(ns)

    print(f'  G = U^T D U at lam^2={lam_sq}')
    print(f'  Eigenvalues of G (should be integers -N,...,N):')
    print(f'    First 5: {evals_G_sorted[:5]}')
    print(f'    Expected: {ns_sorted[:5]}')
    print(f'    Last 5: {evals_G_sorted[-5:]}')
    print(f'    Expected: {ns_sorted[-5:]}')
    print(f'    Max |eig(G) - ns|: {np.max(np.abs(evals_G_sorted - ns_sorted)):.2e}')

    # Is G a Cauchy-type matrix?
    # Off-diagonal: G[j,k] should be (w_j*z_k - z_j*w_k)/(lam_j - lam_k)
    w = U.T @ ones  # projections of 1 onto eigenvectors
    z = U.T @ B_n   # projections of B onto eigenvectors

    # Build expected G from the formula
    G_expected = np.zeros((dim, dim))
    for j in range(dim):
        for k in range(dim):
            if j != k:
                G_expected[j, k] = (w[j]*z[k] - z[j]*w[k]) / (evals_M[j] - evals_M[k])

    # Compare off-diagonal
    mask = ~np.eye(dim, dtype=bool)
    offdiag_err = np.max(np.abs(G[mask] - G_expected[mask]))
    print(f'\n  Off-diagonal structure:')
    print(f'    G[j,k] = (w_j*z_k - z_j*w_k)/(lam_j - lam_k)?')
    print(f'    Max error: {offdiag_err:.2e}')
    print(f'    Verified: {offdiag_err < 1e-8}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: Is z proportional to w? (needed for involution closure)
    # ==================================================================
    print(f'\n  === STEP 3: z/w PROPORTIONALITY TEST ===\n')

    # If z = c*w, then G is a divided-difference matrix and involution closes
    ratios = z / w
    # Filter out near-zero w components
    valid = np.abs(w) > 1e-10
    ratios_valid = ratios[valid]

    print(f'  z_k / w_k for eigenvectors with |w_k| > 1e-10:')
    print(f'    Count: {np.sum(valid)}/{dim}')
    print(f'    Min ratio: {ratios_valid.min():.6f}')
    print(f'    Max ratio: {ratios_valid.max():.6f}')
    print(f'    Mean ratio: {ratios_valid.mean():.6f}')
    print(f'    Std: {ratios_valid.std():.6f}')
    print(f'    CV (std/mean): {ratios_valid.std()/abs(ratios_valid.mean()):.6f}')
    print(f'    Proportional (CV < 0.01)? {ratios_valid.std()/abs(ratios_valid.mean()) < 0.01}')

    # Show ratios for the top eigenvalues (most important)
    print(f'\n  Ratios by eigenvalue magnitude:')
    print(f'  {"k":>3} {"eig_M[k]":>14} {"w_k":>14} {"z_k":>14} {"z/w":>14}')
    print('  ' + '-' * 62)
    order = np.argsort(np.abs(evals_M))[::-1]
    for rank, k in enumerate(order[:20]):
        print(f'  {rank:>3d} {evals_M[k]:>+14.6e} {w[k]:>+14.6e} {z[k]:>+14.6e} '
              f'{ratios[k]:>+14.6f}' if abs(w[k]) > 1e-12 else
              f'  {rank:>3d} {evals_M[k]:>+14.6e} {w[k]:>+14.6e} {z[k]:>+14.6e} '
              f'{"div/0":>14}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: Structure of G — is it close to divided-difference?
    # ==================================================================
    print(f'\n  === STEP 4: G STRUCTURE ANALYSIS ===\n')

    # If G were a divided-difference matrix, there would exist a function F
    # such that G[j,k] = (F_k - F_j)/(lam_j - lam_k)
    # i.e., G[j,k] * (lam_j - lam_k) = F_k - F_j
    # This means G[j,k]*(lam_j-lam_k) + G[k,j]*(lam_k-lam_j) = (F_k-F_j) + (F_j-F_k) = 0
    # So G must be antisymmetric after multiplying by the denominator:
    # G[j,k]*(lam_j-lam_k) = -G[k,j]*(lam_k-lam_j) = G[k,j]*(lam_j-lam_k)
    # Wait, that's just G[j,k] = G[k,j], which is true since G = U^T D U is symmetric!

    # Actually, for a divided-difference matrix:
    # L[j,k] = (F_k - F_j)/(j - k) [nodes are j,k]
    # Here the "nodes" are the eigenvalues lam_j, so:
    # G[j,k] = (F_k - F_j)/(lam_j - lam_k) = -(F_k - F_j)/(lam_k - lam_j)

    # But G is SYMMETRIC (G = U^T D U), while (F_k-F_j)/(lam_j-lam_k)
    # is antisymmetric in (j,k) if F is generic. So:
    # G[j,k] = G[k,j] => (F_k-F_j)/(lam_j-lam_k) = (F_j-F_k)/(lam_k-lam_j) = (F_k-F_j)/(lam_j-lam_k)
    # This is always true! So symmetry is automatic for divided differences.

    # To check if G is a divided-difference matrix, try to extract F:
    # G[0,k] = (F_k - F_0)/(lam_0 - lam_k) for all k
    # F_k = F_0 + G[0,k]*(lam_0 - lam_k)

    F = np.zeros(dim)
    F[0] = 0  # arbitrary
    for k in range(1, dim):
        F[k] = G[0, k] * (evals_M[0] - evals_M[k])

    # Now check: does G[j,k] = (F_k - F_j)/(lam_j - lam_k) for ALL j,k?
    G_from_F = np.zeros((dim, dim))
    for j in range(dim):
        for k in range(dim):
            if j != k:
                G_from_F[j, k] = (F[k] - F[j]) / (evals_M[j] - evals_M[k])

    offdiag_err2 = np.max(np.abs((G - G_from_F)[mask]))
    print(f'  Divided-difference test:')
    print(f'    Extract F from G[0,k], then check G[j,k] = (F_k-F_j)/(lam_j-lam_k)')
    print(f'    Max error: {offdiag_err2:.2e}')
    print(f'    IS a divided-difference matrix: {offdiag_err2 < 1e-8}')

    if offdiag_err2 < 1e-8:
        print(f'\n  G IS a divided-difference matrix!')
        print(f'  F values (first 10): {F[:10]}')
        print(f'  F values (last 10): {F[-10:]}')

        # Check if F = c*lam + d (linear in eigenvalues)
        fit = np.polyfit(evals_M, F, 1)
        residual = F - np.polyval(fit, evals_M)
        print(f'\n  Linear fit F = a*lam + b:')
        print(f'    a = {fit[0]:.10f}, b = {fit[1]:.10f}')
        print(f'    Max residual: {np.max(np.abs(residual)):.6e}')
        print(f'    F is linear in eigenvalues: {np.max(np.abs(residual)) < 1e-6}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: The involution test
    # ==================================================================
    print(f'\n  === STEP 5: INVOLUTION TEST ===\n')

    # If G is a divided-difference matrix with generating function F
    # evaluated at nodes lam_j, then:
    # "Spectral map": (ns, B_n) -> (evals_M, F_k)
    #
    # Involution would mean: starting from (evals_M, F_k), build the
    # divided-difference matrix G, diagonalize it, and recover (ns, B_n).
    #
    # We already know G has eigenvalues = ns (the integers).
    # The question: when we diagonalize G, do we get back B_n?

    evals_G_check, V = np.linalg.eigh(G)

    # V diagonalizes G: V^T G V = diag(evals_G)
    # Compute V^T diag(evals_M) V = ?
    H = V.T @ np.diag(evals_M) @ V

    # If involution holds, H should be a divided-difference matrix
    # with generating function related to B_n

    # Extract candidate B-function from H
    H_F = np.zeros(dim)
    for k in range(1, dim):
        if abs(evals_G_check[0] - evals_G_check[k]) > 1e-10:
            H_F[k] = H[0, k] * (evals_G_check[0] - evals_G_check[k])

    # Check if H is a divided-difference matrix
    H_from_F = np.zeros((dim, dim))
    for j in range(dim):
        for k in range(dim):
            if abs(evals_G_check[j] - evals_G_check[k]) > 1e-10:
                H_from_F[j, k] = (H_F[k] - H_F[j]) / (evals_G_check[j] - evals_G_check[k])

    h_mask = ~np.eye(dim, dtype=bool)
    h_err = np.max(np.abs((H - H_from_F)[h_mask]))

    print(f'  Round trip: (ns, B) -> (lam, F) -> G -> eig(G) -> (ns_back, B_back?)')
    print(f'  Step 1: eigenvalues of G = ns? Max err: {np.max(np.abs(np.sort(evals_G_check) - ns_sorted)):.2e}')
    print(f'  Step 2: H = V^T diag(lam) V is divided-difference? Max err: {h_err:.2e}')

    if h_err < 1e-6:
        # Compare H_F to B_n (reordered by eigenvalue ordering)
        # Need to match the eigenvalue ordering of G to the integer ordering
        perm = np.argsort(evals_G_check)
        H_F_reordered = H_F[perm]

        # B_n is indexed by ns = -N,...,N. H_F_reordered should match.
        print(f'  Step 3: recovered function vs original B_n:')
        print(f'    {"n":>4} {"B_n":>14} {"recovered":>14} {"diff":>14}')
        print('    ' + '-' * 46)
        for i in range(min(15, dim)):
            print(f'    {int(ns_sorted[i]):>4d} {B_n[i]:>+14.6f} '
                  f'{H_F_reordered[i]:>+14.6f} {B_n[i]-H_F_reordered[i]:>+14.6e}')

        max_B_err = np.max(np.abs(B_n - H_F_reordered))
        print(f'    Max |B_n - recovered|: {max_B_err:.6e}')
        print(f'    INVOLUTION HOLDS: {max_B_err < 1e-4}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: Lambda dependence
    # ==================================================================
    print(f'\n  === STEP 6: INVOLUTION AT MULTIPLE LAMBDA ===\n')

    for lam_sq in [50, 200, 500, 1000, 5000]:
        try:
            M2, D2, B2, ones2, ns2, N2, L2, dim2 = extract_all(lam_sq)
            evals2, U2 = np.linalg.eigh(M2)
            G2 = U2.T @ D2 @ U2

            # Divided-difference test for G
            F2 = np.zeros(dim2)
            for k in range(1, dim2):
                F2[k] = G2[0, k] * (evals2[0] - evals2[k])

            G2_from_F = np.zeros((dim2, dim2))
            mask2 = ~np.eye(dim2, dtype=bool)
            for j in range(dim2):
                for k in range(dim2):
                    if j != k:
                        G2_from_F[j, k] = (F2[k] - F2[j]) / (evals2[j] - evals2[k])

            dd_err = np.max(np.abs((G2 - G2_from_F)[mask2]))

            # Eigenvalues of G = integers?
            evals_G2 = np.sort(np.linalg.eigvalsh(G2))
            ns2_sorted = np.sort(ns2)
            eig_err = np.max(np.abs(evals_G2 - ns2_sorted))

            print(f'  lam^2={lam_sq:>5d}: G is div-diff? err={dd_err:.2e}, '
                  f'eig(G)=ns? err={eig_err:.2e}')
        except Exception as e:
            print(f'  lam^2={lam_sq:>5d}: ERROR {e}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 74 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
