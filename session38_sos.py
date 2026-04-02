"""
SESSION 38c — SEARCH FOR PRIME-BUILT SOS FACTORIZATION

We need: M_null = -A^T * A  where A is built from primes and special functions.

APPROACH 1: Cholesky factor of -M_null
  Compute L such that -M_null = L*L^T. Analyze L for prime structure.

APPROACH 2: Factor through individual T(p^k)
  Each T(p^k) is rank ~2. Can we build A from the "negative parts" of T(p^k)?
  M = D + alpha + sum w(pk)*T(pk)
  If we write T(pk) = u*u^T - v*v^T (rank-2 eigendecomposition),
  can the sum of v*v^T terms dominate on null(W02)?

APPROACH 3: The explicit formula as a factorization
  The Weil explicit formula IS an algebraic identity relating primes to zeros.
  Can we use it to construct A without knowing zero locations?
  The explicit formula in matrix form:
    M = (analytic) + (prime sum)
    Q_W = W02 - M
  The analytic part has a KNOWN structure (special functions).
  If Q_W = W02 - (analytic) - (prime sum) = (W02 - analytic) - (prime sum)
  and if (W02 - analytic) has a specific factorization...

APPROACH 4: Congruence / Sylvester's law of inertia
  If we can find a congruence transform C such that C^T * M * C is visibly NSD
  (e.g., diagonal with non-positive entries), and C is built from primes...

APPROACH 5: Direct numerical search
  Parameterize A in terms of prime-related quantities, optimize to minimize
  ||M_null + A^T*A||.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def approach1_cholesky_structure(lam_sq, N=None):
    """
    Compute Cholesky-like factor of -M_null.
    Since -M_null has zero eigenvalues (silent modes), use eigendecomposition
    to get the square root: -M_null = U * Lambda * U^T, then A = Lambda^{1/2} * U^T.

    Analyze the rows of A: do they have prime-related structure?
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    M_null = P_null.T @ M @ P_null
    neg_M = -M_null

    # Eigendecomposition
    evals, evecs = np.linalg.eigh(neg_M)
    # Clip small negatives (numerical noise)
    evals = np.maximum(evals, 0)

    # Square root: A = diag(sqrt(evals)) @ evecs^T
    sqrt_evals = np.sqrt(evals)
    A = np.diag(sqrt_evals) @ evecs.T  # rows of A are sqrt(lambda_i) * u_i

    # Verify: A^T @ A should equal -M_null
    recon = A.T @ A
    err = np.linalg.norm(recon - neg_M, 'fro') / np.linalg.norm(neg_M, 'fro')
    print(f"APPROACH 1: CHOLESKY STRUCTURE at lam^2={lam_sq}", flush=True)
    print(f"  Reconstruction error: {err:.2e}", flush=True)
    print(f"  Nonzero rows of A: {np.sum(sqrt_evals > 1e-6)}", flush=True)

    # The nonzero rows of A correspond to seeing modes
    # Express each row of A (= sqrt(lambda)*eigenvector) in the ORIGINAL Fourier basis
    # by mapping back through P_null
    A_full = A @ P_null.T  # rows of A in the full dim-dimensional space

    # Analyze: do the rows of A_full have prime-related structure?
    print(f"\n  Structure of A rows (seeing mode square roots):", flush=True)
    print(f"  Each row is sqrt(lambda_i) * u_i mapped to Fourier basis", flush=True)

    # For each nonzero row, compute its correlation with prime-frequency oscillations
    # Prime frequencies: cos(2*pi*n*log(p)/L) for each prime p
    primes_list = []
    limit = int(lam_sq)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    for p in range(2, limit + 1):
        if sieve[p]:
            primes_list.append(p)

    # Build prime-frequency vectors: v_p[n] = cos(2*pi*n*log(p)/L)
    V_prime = np.zeros((len(primes_list), dim))
    for idx, p in enumerate(primes_list):
        for i in range(dim):
            V_prime[idx, i] = np.cos(2 * np.pi * ns[i] * np.log(p) / L_f)
        V_prime[idx] /= np.linalg.norm(V_prime[idx])

    # Correlation of each A row with the prime-frequency vectors
    active_rows = np.where(sqrt_evals > 1e-4)[0]
    print(f"\n  Correlation of A rows with prime-frequency vectors:", flush=True)
    print(f"  {'row':>4} {'sqrt(eig)':>10} {'best_p':>7} {'corr':>8} {'2nd_p':>7} {'corr2':>8}", flush=True)

    for row_idx in active_rows[:10]:
        a_row = A_full[row_idx]
        # Project to unit vector
        if np.linalg.norm(a_row) < 1e-10:
            continue
        a_unit = a_row / np.linalg.norm(a_row)

        # Correlations with all prime vectors
        corrs = V_prime @ a_unit
        sorted_idx = np.argsort(np.abs(corrs))[::-1]
        best = sorted_idx[0]
        second = sorted_idx[1]

        print(f"  {row_idx:>4} {sqrt_evals[row_idx]:>10.4f} "
              f"{primes_list[best]:>7} {corrs[best]:>+8.4f} "
              f"{primes_list[second]:>7} {corrs[second]:>+8.4f}", flush=True)

    return A, A_full, primes_list, V_prime


def approach2_prime_factored_decomposition(lam_sq, N=None):
    """
    Try to build A from the T(p^k) matrices directly.

    Each T(pk) = w * Q(pk) where Q is the oscillatory kernel.
    Q has eigendecomposition: Q = sum_i lambda_i * u_i * u_i^T

    For the NEGATIVE eigenvalues of Q (projected to null(W02)):
    these contribute negative terms to M.

    If we collect all negative eigenvectors across all p^k:
    B = [sqrt(|neg_eig|) * neg_eigvec for each (pk, neg_eig)]

    Then sum of negative parts = -B^T * B (NSD by construction)
    But M = (positive parts) + (negative parts) + (analytic)
    We need: (analytic) + (positive parts) <= 0 on null(W02)

    Or differently: build the FULL A from ALL eigenvalues of ALL T(pk),
    treating M as a sum of rank-2 contributions.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    M_null = P_null.T @ M @ P_null
    Ma_null = P_null.T @ (M_diag + M_alpha) @ P_null
    Mp_null = P_null.T @ M_prime @ P_null

    print(f"\nAPPROACH 2: PRIME-FACTORED DECOMPOSITION at lam^2={lam_sq}", flush=True)

    # Decompose each T(pk) on null(W02) into positive and negative parts
    pos_vectors = []  # (weight, vector) for positive eigenvalues
    neg_vectors = []  # (weight, vector) for negative eigenvalues

    for pk, logp, logpk in primes:
        Q = np.zeros((dim, dim))
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n = ns[j]
                if m != n:
                    Q[i, j] = (np.sin(2*np.pi*n*logpk/L_f) -
                               np.sin(2*np.pi*m*logpk/L_f)) / (np.pi*(m-n))
                else:
                    Q[i, j] = 2*(L_f - logpk)/L_f * np.cos(2*np.pi*m*logpk/L_f)
        Q = (Q + Q.T) / 2
        w = logp * pk**(-0.5)
        T = w * Q

        # Project to null(W02)
        T_null = P_null.T @ T @ P_null
        evals_T, evecs_T = np.linalg.eigh(T_null)

        for i in range(d_null):
            if evals_T[i] > 1e-12:
                pos_vectors.append((evals_T[i], evecs_T[:, i]))
            elif evals_T[i] < -1e-12:
                neg_vectors.append((-evals_T[i], evecs_T[:, i]))  # store positive weight

    print(f"  Positive eigenvectors from T(pk): {len(pos_vectors)}", flush=True)
    print(f"  Negative eigenvectors from T(pk): {len(neg_vectors)}", flush=True)

    # Reconstruct: M_prime_null = (positive Gram) - (negative Gram)
    if pos_vectors:
        B_pos = np.array([np.sqrt(w) * v for w, v in pos_vectors])
        M_prime_pos = B_pos.T @ B_pos
    else:
        M_prime_pos = np.zeros((d_null, d_null))

    if neg_vectors:
        B_neg = np.array([np.sqrt(w) * v for w, v in neg_vectors])
        M_prime_neg = B_neg.T @ B_neg
    else:
        M_prime_neg = np.zeros((d_null, d_null))

    recon = M_prime_pos - M_prime_neg
    err = np.linalg.norm(recon - Mp_null, 'fro') / np.linalg.norm(Mp_null, 'fro')
    print(f"  Reconstruction M_prime = Pos - Neg: error {err:.4e}", flush=True)

    # Now: M_null = Ma_null + Mp_null = Ma_null + Pos - Neg
    # For M_null <= 0: need Ma_null + Pos <= Neg
    # i.e.: Neg - Ma_null - Pos >= 0 (should be PSD)

    residual = M_prime_neg - Ma_null - M_prime_pos
    evals_res = np.linalg.eigvalsh(residual)
    print(f"\n  CRITICAL: Is (Neg - Analytic - Pos) PSD?", flush=True)
    print(f"  Eigenvalues: [{np.min(evals_res):.4f}, {np.max(evals_res):.4f}]", flush=True)
    print(f"  PSD: {np.min(evals_res) > -1e-6}", flush=True)

    # Also check: is Neg >= M_prime_pos + Ma_null element-by-element (in a spectral sense)?
    print(f"\n  Component norms on null(W02):", flush=True)
    print(f"  ||M_prime_pos||_op = {np.linalg.norm(M_prime_pos, 2):.4f} (positive part)", flush=True)
    print(f"  ||M_prime_neg||_op = {np.linalg.norm(M_prime_neg, 2):.4f} (negative part)", flush=True)
    print(f"  ||Ma_null||_op     = {np.linalg.norm(Ma_null, 2):.4f} (analytic)", flush=True)
    print(f"  ||Mp_null||_op     = {np.linalg.norm(Mp_null, 2):.4f} (full prime)", flush=True)

    # What fraction of M_prime's negative eigenvectors account for the negativity?
    neg_weights = sorted([w for w, v in neg_vectors], reverse=True)
    pos_weights = sorted([w for w, v in pos_vectors], reverse=True)

    print(f"\n  Top 10 negative eigenvalues (from T(pk) decomposition):", flush=True)
    for i, w in enumerate(neg_weights[:10]):
        print(f"    {i}: {w:.4f}", flush=True)
    print(f"  Sum of all negative: {sum(neg_weights):.4f}", flush=True)
    print(f"  Sum of all positive: {sum(pos_weights):.4f}", flush=True)

    return pos_vectors, neg_vectors, Ma_null


def approach3_congruence_search(lam_sq, N=None):
    """
    APPROACH 3: Find a congruence C such that C^T * M_null * C is visibly NSD.

    If C is chosen to "diagonalize" M_null in a prime-aware basis,
    the diagonal entries of C^T*M*C might have a pattern.

    Natural choices for C:
    (a) DFT matrix restricted to null(W02) -- makes M approximately Toeplitz
    (b) Prime-frequency basis: columns = cos/sin at prime frequencies
    (c) The T(p^k) eigenvectors themselves
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    M_null = P_null.T @ M @ P_null

    print(f"\nAPPROACH 3: CONGRUENCE SEARCH at lam^2={lam_sq}", flush=True)

    # (a) DFT basis
    # The DFT of M_null would reveal its spectral structure
    F = np.fft.fft(np.eye(d_null), axis=0) / np.sqrt(d_null)
    M_dft = np.real(F.conj().T @ M_null @ F)
    diag_dft = np.diag(M_dft)
    print(f"\n  (a) DFT diagonalization:", flush=True)
    print(f"  DFT diagonal: [{np.min(diag_dft):.4f}, {np.max(diag_dft):.4f}]", flush=True)
    print(f"  All non-positive: {np.max(diag_dft) < 1e-6}", flush=True)
    off_diag_norm = np.linalg.norm(M_dft - np.diag(diag_dft), 'fro')
    total_norm = np.linalg.norm(M_dft, 'fro')
    print(f"  Off-diagonal energy: {off_diag_norm/total_norm*100:.1f}%", flush=True)

    # (b) M_null expressed in the FOURIER basis (the original n-indexed basis)
    # M_null is already in the null(W02) basis. Map to Fourier:
    M_fourier = P_null @ M_null @ P_null.T  # This is M projected, in Fourier basis
    diag_fourier = np.diag(M_fourier)  # = wr_diag[n] + alpha[n,n] + prime_diag[n,n]

    print(f"\n  (b) Fourier basis diagonal (full M projected):", flush=True)
    n_pos_diag = np.sum(diag_fourier > 1e-6)
    n_neg_diag = np.sum(diag_fourier < -1e-6)
    print(f"  Positive diag entries: {n_pos_diag}", flush=True)
    print(f"  Negative diag entries: {n_neg_diag}", flush=True)
    print(f"  Diag range: [{np.min(diag_fourier):.4f}, {np.max(diag_fourier):.4f}]", flush=True)

    # If the diagonal is all non-positive, and the off-diagonal is "small",
    # Gershgorin circles would prove NSD!
    print(f"\n  GERSHGORIN CHECK on Fourier-basis M:", flush=True)
    gershgorin_ok = True
    worst_margin = float('inf')
    for i in range(dim):
        diag_val = M_fourier[i, i]
        off_diag_sum = sum(abs(M_fourier[i, j]) for j in range(dim) if j != i)
        upper = diag_val + off_diag_sum
        if upper > 0:
            gershgorin_ok = False
        margin = -upper
        if margin < worst_margin:
            worst_margin = margin
            worst_idx = i

    if gershgorin_ok:
        print(f"  *** GERSHGORIN PROVES M <= 0! ***", flush=True)
        print(f"  Worst margin: {worst_margin:.6f} at n={ns[worst_idx]:.0f}", flush=True)
    else:
        print(f"  Gershgorin fails. Worst upper bound: {-worst_margin:.4f} at n={ns[worst_idx]:.0f}", flush=True)

    # (c) WEIGHTED Gershgorin: use D^{-1} M D for diagonal D
    # Choose D to emphasize the negative diagonal entries
    # D[i,i] = 1/sqrt(|diag[i]|) normalizes the diagonal to +-1
    print(f"\n  WEIGHTED GERSHGORIN (normalizing diagonal):", flush=True)
    D = np.diag(1.0 / np.sqrt(np.maximum(np.abs(diag_fourier), 1e-10)))
    M_norm = D @ M_fourier @ D
    gershgorin_ok2 = True
    worst_margin2 = float('inf')
    for i in range(dim):
        dv = M_norm[i, i]
        od = sum(abs(M_norm[i, j]) for j in range(dim) if j != i)
        upper = dv + od
        if upper > 0:
            gershgorin_ok2 = False
        margin = -upper
        if margin < worst_margin2:
            worst_margin2 = margin

    if gershgorin_ok2:
        print(f"  *** WEIGHTED GERSHGORIN PROVES M <= 0! ***", flush=True)
    else:
        print(f"  Weighted Gershgorin fails. Worst: {-worst_margin2:.4f}", flush=True)

    return M_null, M_fourier


def approach4_explicit_formula_factorization(lam_sq, N=None):
    """
    APPROACH 4: Use the EXPLICIT FORMULA STRUCTURE directly.

    The explicit formula at bandwidth L gives (schematically):
    W02[n,m] = M_analytic[n,m] + M_prime[n,m] + spectral_side[n,m]

    where spectral_side = sum_rho (Mellin_n(rho) * conj(Mellin_m(rho)))

    Rearranging: spectral_side = W02 - M = Q_W

    So Q_W = sum_rho Mellin(rho) * Mellin(rho)^H  (a Gram matrix of Mellin vectors)

    This factorization uses zeros. Can we re-express the Mellin vectors
    in terms of PRIME quantities?

    The Mellin transform at s = 1/2 + it of our basis function omega_n is:
    omega_n_hat(t) = integral_0^L 2(1-x/L)cos(2*pi*n*x/L) * e^{itx} dx

    This is a KNOWN function of t, n, L. It doesn't depend on primes or zeros.
    The zeros just tell us WHERE to evaluate it.

    But the EXPLICIT FORMULA says: evaluating at zeros = prime arithmetic.
    So the Mellin vectors at zeros are DETERMINED by the primes.

    Can we express: sum_rho |Mellin(rho)|^2 = f(primes) without knowing rho?

    YES — that's exactly what the explicit formula does!
    Q_W = W02 - M = f(primes, analytic)

    But Q_W is already computed from primes. The question is whether
    Q_W = W02 - M is PSD, which is what we're trying to prove.

    THE CIRCULARITY: the explicit formula gives us Q_W = W02 - M = sum |Mellin|^2.
    Knowing Q_W = sum |Mellin|^2 tells us Q_W >= 0 (sum of squares).
    But evaluating Q_W from the PRIME SIDE gives a matrix we need to CHECK
    is PSD — which we can't do without RH.

    UNLESS: there's an algebraic identity that Q_W = W02 - M factors as
    a sum of squares DIRECTLY from the prime/analytic structure.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    M_null = P_null.T @ M @ P_null
    Ma_null = P_null.T @ (M_diag + M_alpha) @ P_null
    Mp_null = P_null.T @ M_prime @ P_null

    print(f"\nAPPROACH 4: EXPLICIT FORMULA FACTORIZATION at lam^2={lam_sq}", flush=True)

    # The key insight: M = Ma + Mp where
    # Ma = (digamma diagonal) + (alpha off-diagonal) -- COMPUTABLE, no primes
    # Mp = sum_{pk} w(pk) * T(pk) -- prime sum

    # On null(W02): M_null = Ma_null + Mp_null

    # If we can show: Mp_null <= -Ma_null (prime part compensates analytic part)
    # then M_null <= 0.

    # Check: is -Ma_null - Mp_null PSD? (This is -M_null, should be PSD)
    neg_M_null = -M_null
    evals_neg_M = np.linalg.eigvalsh(neg_M_null)
    print(f"  -M_null PSD: {np.min(evals_neg_M) > -1e-6}", flush=True)
    print(f"  -M_null eigenvalues: [{np.min(evals_neg_M):.6e}, {np.max(evals_neg_M):.4f}]", flush=True)

    # What does Ma_null look like?
    evals_Ma = np.linalg.eigvalsh(Ma_null)
    print(f"\n  Ma_null (analytic) eigenvalues: [{np.min(evals_Ma):.4f}, {np.max(evals_Ma):.4f}]", flush=True)
    print(f"  Ma_null is {'NSD' if np.max(evals_Ma) < 1e-6 else 'INDEFINITE'}", flush=True)

    # What does Mp_null look like?
    evals_Mp = np.linalg.eigvalsh(Mp_null)
    print(f"  Mp_null (prime) eigenvalues: [{np.min(evals_Mp):.4f}, {np.max(evals_Mp):.4f}]", flush=True)
    print(f"  Mp_null is {'NSD' if np.max(evals_Mp) < 1e-6 else 'INDEFINITE'}", flush=True)

    # THE MIRACLE CHECK:
    # Is there a scalar alpha such that M_null + alpha * I <= 0?
    # I.e., is M_null bounded above by -alpha * I?
    # This would mean all eigenvalues of M <= -alpha.
    # From the data: max eigenvalue is ~0 (silent modes), so alpha ~ 0.
    # Not useful.

    # DIFFERENT ANGLE: the prime sum Mp_null is indefinite.
    # Can we SPLIT it: Mp_null = Mp_neg + Mp_pos
    # where Mp_neg <= 0 and Mp_pos >= 0?
    # Then M_null = Ma_null + Mp_neg + Mp_pos
    # Need: Ma_null + Mp_pos <= -Mp_neg

    # Eigendecomposition split of Mp_null
    evals_Mp_full, evecs_Mp = np.linalg.eigh(Mp_null)
    pos_mask = evals_Mp_full > 1e-10
    neg_mask = evals_Mp_full < -1e-10

    Mp_pos_part = evecs_Mp[:, pos_mask] @ np.diag(evals_Mp_full[pos_mask]) @ evecs_Mp[:, pos_mask].T
    Mp_neg_part = evecs_Mp[:, neg_mask] @ np.diag(evals_Mp_full[neg_mask]) @ evecs_Mp[:, neg_mask].T

    print(f"\n  Mp_null split:", flush=True)
    print(f"  Positive part: {np.sum(pos_mask)} modes, max eig = {np.max(evals_Mp_full[pos_mask]):.4f}" if np.any(pos_mask) else "  No positive part", flush=True)
    print(f"  Negative part: {np.sum(neg_mask)} modes, min eig = {np.min(evals_Mp_full[neg_mask]):.4f}" if np.any(neg_mask) else "  No negative part", flush=True)

    # The RESIDUAL that needs to be NSD: Ma_null + Mp_pos_part
    residual = Ma_null + Mp_pos_part
    evals_res = np.linalg.eigvalsh(residual)
    print(f"\n  Residual (Ma + Mp_positive):", flush=True)
    print(f"  Eigenvalues: [{np.min(evals_res):.4f}, {np.max(evals_res):.4f}]", flush=True)
    print(f"  NSD: {np.max(evals_res) < 1e-6}", flush=True)

    if np.max(evals_res) < 1e-6:
        print(f"\n  *** Ma + Mp_positive IS NSD on null(W02)! ***", flush=True)
        print(f"  This means: the positive part of the prime sum is COMPENSATED", flush=True)
        print(f"  by the negative analytic terms. Only the analytic terms matter!", flush=True)
        print(f"  M = (Ma + Mp_pos) + Mp_neg <= 0 because both summands <= 0.", flush=True)
    else:
        print(f"\n  Ma + Mp_positive is NOT NSD. Deficit: {np.max(evals_res):.6f}", flush=True)
        print(f"  Need: Mp_neg to compensate the deficit as well.", flush=True)

        # How much of the deficit does Mp_neg cover?
        # The full picture: M_null = (Ma + Mp_pos) + Mp_neg
        # = (indefinite residual) + (NSD part)
        # For M_null <= 0: need ||Mp_neg|| large enough to dominate residual

        evals_combined = np.linalg.eigvalsh(Ma_null + Mp_pos_part + Mp_neg_part)
        print(f"  Full M_null eigenvalues: [{np.min(evals_combined):.4f}, {np.max(evals_combined):.6e}]", flush=True)
        print(f"  (Verification: should match original M_null)", flush=True)


if __name__ == "__main__":
    print("SESSION 38c — PRIME-BUILT SOS FACTORIZATION SEARCH", flush=True)
    print("=" * 80, flush=True)

    # Approach 1: Cholesky structure
    A, A_full, primes_list, V_prime = approach1_cholesky_structure(50)

    # Approach 2: Prime-factored decomposition
    approach2_prime_factored_decomposition(50)

    # Approach 3: Congruence / Gershgorin
    approach3_congruence_search(50)

    # Approach 4: Explicit formula factorization
    approach4_explicit_formula_factorization(50)
    approach4_explicit_formula_factorization(200)

    print(f"\nDone.", flush=True)
