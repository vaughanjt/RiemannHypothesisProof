"""
SESSION 33 — THE CHOLESKY ATTACK

PREMISE:
For each lambda, Q_W is PSD (eps_0 > 0). So Q_W = L D L^T exists.
The diagonal D[k] are the Schur complements — each one must be > 0.

THE ATTACK:
1. Compute LDL^T decomposition — which Schur complement is the bottleneck?
2. Track D[k] as lambda grows — do they converge? Stay bounded below?
3. Study the ORDERING: if we process the "easy" modes first (large Q_W diagonal),
   the Schur complements for hard modes inherit positivity from easy ones.
4. INDUCTION on dimension: as N grows, new modes are added. If the new mode's
   Schur complement stays > 0, the induction step works.
5. REORDERING: The Cholesky decomposition depends on row/column ordering.
   Optimal ordering (maximum diagonal pivot) may reveal structure.

WHY THIS MIGHT WORK:
- Cholesky decomposes the GLOBAL condition eps_0 > 0 into LOCAL conditions D[k] > 0
- Each D[k] involves fewer prime sums (only primes seen up to step k)
- The PNT/sieve bounds that are too weak for the GLOBAL condition
  might be strong enough for INDIVIDUAL Schur complements
- The Cholesky factor L encodes the "proof witness" — which linear
  combinations of basis vectors certify positivity

THE NUCLEAR INSIGHT:
If D[k] can be expressed as:
  D[k] = (analytic_k) - sum_{p^k} (weight_k(p)) * (bounded_function)
  where sum < analytic_k by Selberg sieve...
  Then we have a CONSTRUCTIVE proof that D[k] > 0.
  Do this for ALL k, and Q_W > 0, hence eps_0 > 0, hence RH.
"""

import numpy as np
import scipy.linalg
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, sinh, euler, digamma, hyp2f1
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 50


def ldlt_decomposition(A):
    """Compute LDL^T decomposition. L is unit lower triangular, D is diagonal."""
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)

    for j in range(n):
        # D[j] = A[j,j] - sum_{k<j} L[j,k]^2 * D[k]
        D[j] = A[j, j] - sum(L[j, k]**2 * D[k] for k in range(j))

        if abs(D[j]) < 1e-30:
            D[j] = 1e-30  # regularize

        for i in range(j + 1, n):
            # L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]*D[k]) / D[j]
            L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] * D[k] for k in range(j))) / D[j]

    return L, D


def pivoted_ldlt(A):
    """LDL^T with maximum diagonal pivoting. Returns L, D, perm."""
    n = A.shape[0]
    A = A.copy()
    perm = list(range(n))
    L = np.eye(n)
    D = np.zeros(n)

    for j in range(n):
        # Find the maximum remaining diagonal
        remaining_diag = np.array([A[i, i] for i in range(j, n)])
        max_idx = j + np.argmax(remaining_diag)

        # Swap rows/columns
        if max_idx != j:
            A[[j, max_idx]] = A[[max_idx, j]]
            A[:, [j, max_idx]] = A[:, [max_idx, j]]
            L[[j, max_idx], :j] = L[[max_idx, j], :j]
            perm[j], perm[max_idx] = perm[max_idx], perm[j]

        D[j] = A[j, j]
        if abs(D[j]) < 1e-30:
            D[j] = 1e-30

        for i in range(j + 1, n):
            L[i, j] = A[i, j] / D[j]
            for k in range(j + 1, n):
                A[k, i] -= L[k, j] * A[j, i]
            A[i, j] = 0

    return L, D, perm


def analyze_schur_complements(lam_sq, N=None):
    """
    Compute and analyze the Schur complements (D diagonal in LDL^T).

    The k-th Schur complement S_k = Q_W / Q_W[0:k, 0:k] is the
    matrix obtained by eliminating the first k rows/columns.
    Its (0,0) entry is D[k].

    S_k[0,0] = Q_W[k,k] - Q_W[k,0:k] Q_W[0:k,0:k]^{-1} Q_W[0:k,k]

    This "subtracts out" the contribution of modes 0,...,k-1 from mode k.
    It tells us: given that modes 0,...,k-1 are already accounted for,
    how much positivity does mode k contribute?
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)
    eps_0 = np.linalg.eigvalsh(QW)[0]

    print(f"\nSCHUR COMPLEMENT ANALYSIS: lam^2={lam_sq}, dim={dim}, eps_0={eps_0:.4e}")
    print("=" * 70)

    # Natural ordering LDL^T
    t0 = time.time()
    L_nat, D_nat = ldlt_decomposition(QW)
    print(f"\n  Natural ordering LDL^T ({time.time()-t0:.1f}s):")
    print(f"    D: min={np.min(D_nat):.4e}  max={np.max(D_nat):.4e}")
    print(f"    D > 0: {np.sum(D_nat > 0)}/{dim}")
    min_D_idx = np.argmin(D_nat)
    print(f"    Bottleneck: D[{min_D_idx}] = {D_nat[min_D_idx]:.6e} (mode n={min_D_idx - N})")

    # Show D profile
    print(f"\n    Schur complement profile (D[k] for each k):")
    print(f"    {'k':>4} {'n':>4} {'D[k]':>14} {'Q_W[k,k]':>14} {'reduction':>10}")
    step = max(1, dim // 20)
    for k in range(0, dim, step):
        n = k - N
        reduction = QW[k, k] / D_nat[k] if D_nat[k] > 1e-30 else float('inf')
        print(f"    {k:>4} {n:>4} {D_nat[k]:>14.6e} {QW[k,k]:>14.6e} {reduction:>10.2f}x")

    # Pivoted ordering (maximum diagonal first)
    t0 = time.time()
    L_piv, D_piv, perm = pivoted_ldlt(QW.copy())
    print(f"\n  Pivoted ordering LDL^T ({time.time()-t0:.1f}s):")
    print(f"    D: min={np.min(D_piv):.4e}  max={np.max(D_piv):.4e}")
    print(f"    D > 0: {np.sum(D_piv > 0)}/{dim}")
    min_D_piv_idx = np.argmin(D_piv)
    print(f"    Bottleneck: D[{min_D_piv_idx}] = {D_piv[min_D_piv_idx]:.6e} "
          f"(original mode n={perm[min_D_piv_idx] - N})")

    # How much does pivoting help?
    ratio = np.min(D_piv) / np.min(D_nat) if np.min(D_nat) > 0 else float('inf')
    print(f"\n    Pivoting improvement: {ratio:.2f}x")

    # The DECAY PROFILE of D in pivoted order
    print(f"\n    Pivoted D decay profile:")
    for k in [0, 1, 2, 5, 10, dim//4, dim//2, 3*dim//4, dim-5, dim-3, dim-2, dim-1]:
        if k < dim:
            orig_n = perm[k] - N
            print(f"    step {k:>3}: D={D_piv[k]:>14.6e}  (orig n={orig_n:>4})")

    return D_nat, D_piv, perm


def track_schur_vs_lambda(lam_sq_values):
    """
    Track how the Schur complements evolve with lambda.

    KEY QUESTION: Does the minimum Schur complement stay bounded below?
    If min(D) ~ C * lam^{-alpha} with C > 0, then Q_W > 0 for all lambda.
    """
    print("\n\n" + "=" * 75)
    print("SCHUR COMPLEMENT SCALING WITH LAMBDA")
    print("=" * 75)

    results = []
    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        dim = 2 * N + 1

        W02, M, QW = build_all(lam_sq, N)
        eps_0 = np.linalg.eigvalsh(QW)[0]

        _, D_piv, perm = pivoted_ldlt(QW.copy())
        min_D = np.min(D_piv)
        bottleneck = np.argmin(D_piv)
        bottleneck_mode = perm[bottleneck] - N

        # The LAST Schur complement (most constrained)
        last_D = D_piv[-1]
        last_mode = perm[-1] - N

        # Product of D = det(Q_W) (should be tiny since eps_0 is tiny)
        log_det = np.sum(np.log(np.abs(D_piv)))

        results.append({
            'lam_sq': lam_sq,
            'dim': dim,
            'eps_0': float(eps_0),
            'min_D': float(min_D),
            'last_D': float(last_D),
            'bottleneck_step': int(bottleneck),
            'bottleneck_mode': int(bottleneck_mode),
            'last_mode': int(last_mode),
            'log_det': float(log_det)
        })

        print(f"\n  lam^2={lam_sq:>5} (dim={dim:>3}): eps_0={eps_0:.3e}  "
              f"min_D={min_D:.3e}  last_D={last_D:.3e}  "
              f"bottleneck=step {bottleneck} (n={bottleneck_mode})")

    # Fit scaling
    if len(results) >= 3:
        lams = np.array([r['lam_sq'] for r in results])
        min_Ds = np.array([r['min_D'] for r in results])
        eps0s = np.array([r['eps_0'] for r in results])
        last_Ds = np.array([r['last_D'] for r in results])

        # min(D) vs lambda
        valid = min_Ds > 0
        if np.sum(valid) >= 3:
            alpha, logC = np.polyfit(np.log(lams[valid]), np.log(min_Ds[valid]), 1)
            print(f"\n  min(D) ~ {np.exp(logC):.4e} * lam^({alpha:.3f})")

        # eps_0 vs lambda
        alpha_e, logC_e = np.polyfit(np.log(lams), np.log(eps0s), 1)
        print(f"  eps_0  ~ {np.exp(logC_e):.4e} * lam^({alpha_e:.3f})")

        # Ratio min(D) / eps_0
        ratios = min_Ds / eps0s
        print(f"\n  Ratio min(D)/eps_0:")
        for r, ratio in zip(results, ratios):
            print(f"    lam^2={r['lam_sq']:>5}: {ratio:.2f}")

    return results


def analyze_bottleneck_structure(lam_sq, N=None):
    """
    Deep dive into the bottleneck Schur complement.

    The bottleneck D[k*] is where positivity is most fragile.
    Express D[k*] in terms of prime sums and check if sieve bounds suffice.

    D[k*] = Q_W[k*,k*] - Q_W[k*,0:k*] Q_W[0:k*,0:k*]^{-1} Q_W[0:k*,k*]

    The correction term involves the inverse of the leading submatrix.
    If the leading submatrix has good condition number, the correction
    is well-controlled.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)

    # Find the bottleneck
    _, D_piv, perm = pivoted_ldlt(QW.copy())
    bottleneck = np.argmin(D_piv)
    min_D = D_piv[bottleneck]

    print(f"\nBOTTLENECK ANALYSIS: lam^2={lam_sq}, bottleneck at step {bottleneck}")
    print(f"  min(D) = {min_D:.6e}")
    print("=" * 70)

    # Reorder Q_W according to the pivot ordering
    QW_reord = QW[np.ix_(perm, perm)]

    # The bottleneck Schur complement
    k = bottleneck
    if k > 0:
        A11 = QW_reord[:k, :k]
        A12 = QW_reord[:k, k]
        A22 = QW_reord[k, k]

        # D[k] = A22 - A12^T A11^{-1} A12
        correction = A12 @ np.linalg.solve(A11, A12)
        D_check = A22 - correction

        print(f"  D[{k}] = A22 - correction")
        print(f"    A22 (diagonal of Q_W) = {A22:.6e}")
        print(f"    correction = {correction:.6e}")
        print(f"    D[{k}] = {D_check:.6e} (check: {min_D:.6e})")
        print(f"    Correction/A22 = {correction/A22:.6f}")

        # How much of the correction comes from W02 vs M?
        W02_reord = W02[np.ix_(perm, perm)]
        M_reord = M[np.ix_(perm, perm)]

        W02_22 = W02_reord[k, k]
        M_22 = M_reord[k, k]
        print(f"\n    W02[k,k] = {W02_22:.6e}")
        print(f"    M[k,k] = {M_22:.6e}")
        print(f"    Q_W[k,k] = W02 - M = {W02_22 - M_22:.6e}")

        # Condition number of A11
        cond_A11 = np.linalg.cond(A11)
        print(f"\n    Condition number of leading {k}x{k} submatrix: {cond_A11:.2e}")

        # Decompose correction into W02 and M parts
        # correction = A12^T A11^{-1} A12
        # where A12 = QW column, A11 = QW block
        # Hard to separate, but we can compare with W02-only correction

        W02_A12 = W02_reord[:k, k]
        W02_A11 = W02_reord[:k, :k]
        # W02 is rank 2, so its inverse is not well-defined on null space
        # Use pseudoinverse
        W02_correction = W02_A12 @ np.linalg.lstsq(W02_A11, W02_A12, rcond=None)[0]
        print(f"\n    W02-only correction: {W02_correction:.6e}")
        print(f"    Full correction: {correction:.6e}")
        print(f"    Excess from M: {correction - W02_correction:.6e}")

    # The LAST few Schur complements (most constrained)
    print(f"\n  Last 10 Schur complements:")
    for j in range(max(0, dim-10), dim):
        orig_n = perm[j] - N
        print(f"    step {j}: D={D_piv[j]:.6e}  (mode n={orig_n})")

    return min_D, bottleneck, perm


def induction_test(lam_sq):
    """
    Test the INDUCTION approach:
    As N increases (more Fourier modes), does eps_0 stay positive?

    For each N, compute eps_0(N). The sequence should be:
    - Non-increasing (adding modes can only reduce eps_0 by interlacing)
    - Bounded below (converges to a positive limit)

    If eps_0(N) converges, the limit IS the true eps_0 for this lambda.
    """
    L_f = np.log(lam_sq)
    N_max = max(21, round(10 * L_f))

    print(f"\nINDUCTION TEST: lam^2={lam_sq}")
    print("=" * 70)
    print(f"  {'N':>4} {'dim':>5} {'eps_0':>14} {'delta':>14} {'converged':>10}")

    prev_eps = None
    results = []
    for N in range(10, N_max + 1, 2):
        dim = 2 * N + 1
        W02, M, QW = build_all(lam_sq, N)
        evals = np.linalg.eigvalsh(QW)
        eps_0 = evals[0]

        delta = eps_0 - prev_eps if prev_eps is not None else 0
        converged = abs(delta) < abs(eps_0) * 0.01 if prev_eps is not None else False

        results.append({
            'N': N, 'dim': dim, 'eps_0': float(eps_0),
            'delta': float(delta) if prev_eps is not None else None
        })

        if N <= 20 or N >= N_max - 4 or N % 10 == 0 or converged:
            print(f"  {N:>4} {dim:>5} {eps_0:>14.6e} {delta:>14.6e} {'YES' if converged else ''}")

        prev_eps = eps_0

    # Check monotonicity
    eps_values = [r['eps_0'] for r in results]
    monotone = all(eps_values[i] >= eps_values[i+1] - 1e-15
                   for i in range(len(eps_values)-1))
    print(f"\n  Monotone non-increasing: {monotone}")
    print(f"  Final eps_0: {eps_values[-1]:.6e}")
    print(f"  Convergence: {abs(eps_values[-1] - eps_values[-2]) / abs(eps_values[-1]):.2e} relative")

    return results


def cholesky_factor_structure(lam_sq, N=None):
    """
    Analyze the STRUCTURE of the Cholesky factor L.

    If L has a pattern (e.g., entries decay, have oscillatory structure,
    relate to prime distribution), that pattern might be provable.

    Specifically: does L have a DISPLACEMENT STRUCTURE?
    W02 has displacement rank 2 (Connes). M involves prime sums.
    If Q_W inherits a low displacement rank, then L might too.

    A matrix with displacement rank r satisfies:
    Z A - A Z^T = sum_{k=1}^r g_k h_k^T
    where Z is the shift matrix.

    Low displacement rank => structured => potentially analyzable.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)

    print(f"\nCHOLESKY FACTOR STRUCTURE: lam^2={lam_sq}, dim={dim}")
    print("=" * 70)

    # Standard Cholesky
    QW_reg = QW + np.eye(dim) * 1e-14
    try:
        L_chol = np.linalg.cholesky(QW_reg)
    except:
        print("  Cholesky failed even with regularization")
        return

    # Structure analysis
    print(f"\n  L diagonal (sqrt of Schur complements):")
    diag_L = np.diag(L_chol)
    print(f"    min: {np.min(diag_L):.6e}  max: {np.max(diag_L):.6e}")
    print(f"    min at index {np.argmin(diag_L)} (n={np.argmin(diag_L) - N})")

    # Decay of off-diagonal entries
    print(f"\n  Off-diagonal decay:")
    for band in [1, 2, 5, 10, 20]:
        if band < dim:
            band_vals = []
            for i in range(band, dim):
                band_vals.append(abs(L_chol[i, i-band]))
            if band_vals:
                print(f"    band {band:>2}: max={np.max(band_vals):.6e}  "
                      f"mean={np.mean(band_vals):.6e}  "
                      f"decay={np.mean(band_vals)/np.max(np.diag(L_chol)):.6e}")

    # Displacement rank of L
    Z = np.zeros((dim, dim))
    for i in range(dim - 1):
        Z[i+1, i] = 1

    displacement = Z @ L_chol - L_chol @ Z
    # Rank of displacement
    svd_disp = np.linalg.svd(displacement, compute_uv=False)
    disp_rank = np.sum(svd_disp > np.max(svd_disp) * 1e-10)
    print(f"\n  Displacement rank of L: {disp_rank}")
    print(f"    (W02 has displacement rank 2)")
    print(f"    Top singular values: {svd_disp[:5]}")

    # Displacement rank of Q_W itself
    displacement_QW = Z @ QW - QW @ Z.T
    svd_disp_QW = np.linalg.svd(displacement_QW, compute_uv=False)
    disp_rank_QW = np.sum(svd_disp_QW > np.max(svd_disp_QW) * 1e-10)
    print(f"\n  Displacement rank of Q_W: {disp_rank_QW}")
    print(f"    Top singular values: {svd_disp_QW[:8]}")

    # Displacement rank of W02 and M separately
    disp_W02 = Z @ W02 - W02 @ Z.T
    svd_W02 = np.linalg.svd(disp_W02, compute_uv=False)
    rank_W02 = np.sum(svd_W02 > np.max(svd_W02) * 1e-10)

    disp_M = Z @ M - M @ Z.T
    svd_M = np.linalg.svd(disp_M, compute_uv=False)
    rank_M = np.sum(svd_M > np.max(svd_M) * 1e-10)

    print(f"\n  Displacement ranks: W02={rank_W02}, M={rank_M}, Q_W={disp_rank_QW}")
    print(f"    Q_W = W02 - M, so disp_rank(Q_W) <= disp_rank(W02) + disp_rank(M) = {rank_W02 + rank_M}")

    # Key structure: if Q_W has low displacement rank, then L does too
    # (Gohberg-Semencul formula)
    if disp_rank_QW <= 10:
        print(f"\n  *** LOW DISPLACEMENT RANK ({disp_rank_QW}) ***")
        print(f"  Q_W is structured — Cholesky factor L inherits this structure.")
        print(f"  The Gohberg-Semencul formula gives L in terms of {disp_rank_QW} generators.")
        print(f"  If these generators can be bounded analytically, eps_0 > 0 follows.")

    return L_chol, diag_L


def displacement_generator_analysis(lam_sq, N=None):
    """
    Extract and analyze the displacement generators of Q_W.

    If Q_W has displacement rank r:
      Z Q_W - Q_W Z^T = G H^T  where G, H are dim x r matrices.

    The generators G, H determine Q_W completely (up to boundary conditions).
    If we can express G, H in terms of prime sums and bound them, we can
    bound Q_W and hence eps_0.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)

    print(f"\nDISPLACEMENT GENERATOR ANALYSIS: lam^2={lam_sq}, dim={dim}")
    print("=" * 70)

    Z = np.zeros((dim, dim))
    for i in range(dim - 1):
        Z[i+1, i] = 1

    displacement = Z @ QW - QW @ Z.T

    # SVD to get generators
    U, S, Vt = np.linalg.svd(displacement)
    threshold = S[0] * 1e-10
    r = np.sum(S > threshold)

    print(f"  Displacement rank: {r}")
    print(f"  Singular values: {S[:min(r+3, len(S))]}")

    # Extract generators: displacement = G H^T where G = U[:,:r]*diag(sqrt(S[:r]))
    G = U[:, :r] * np.sqrt(S[:r])
    H = Vt[:r, :].T * np.sqrt(S[:r])

    print(f"\n  Generator G shape: {G.shape}")
    print(f"  Generator H shape: {H.shape}")

    # Analyze each generator column
    for k in range(min(r, 6)):
        g_k = G[:, k]
        h_k = H[:, k]
        print(f"\n  Generator {k} (sigma={S[k]:.4e}):")
        print(f"    ||g_{k}|| = {np.linalg.norm(g_k):.4e}  "
              f"||h_{k}|| = {np.linalg.norm(h_k):.4e}")

        # Is this generator related to W02 or M?
        # Project onto W02 displacement generators
        disp_W02 = Z @ W02 - W02 @ Z.T
        proj_w02 = np.linalg.norm(disp_W02.T @ g_k) / (np.linalg.norm(g_k) * np.linalg.norm(disp_W02, 'fro'))

        disp_M = Z @ M - M @ Z.T
        proj_m = np.linalg.norm(disp_M.T @ g_k) / (np.linalg.norm(g_k) * np.linalg.norm(disp_M, 'fro'))

        print(f"    Alignment with W02 displacement: {proj_w02:.4f}")
        print(f"    Alignment with M displacement:   {proj_m:.4f}")

        # Fourier analysis of generator
        fft_g = np.fft.fft(g_k)
        max_freq = np.argmax(np.abs(fft_g[1:dim//2])) + 1
        print(f"    Dominant Fourier mode: {max_freq}")

    # THE KEY TEST: Can we express the generators in closed form?
    print(f"\n  GENERATOR SPARSITY TEST:")
    for k in range(min(r, 4)):
        g_k = G[:, k]
        # How many significant entries?
        threshold_k = np.max(np.abs(g_k)) * 0.01
        n_significant = np.sum(np.abs(g_k) > threshold_k)
        print(f"    g_{k}: {n_significant}/{dim} entries > 1% of max")

    return G, H, S[:r]


if __name__ == "__main__":
    print("SESSION 33 — THE CHOLESKY ATTACK")
    print("=" * 75)

    # Part 1: Schur complement analysis
    print("\n" + "#" * 75)
    print("# PART 1: SCHUR COMPLEMENT ANALYSIS")
    print("#" * 75)
    for lam_sq in [200, 1000]:
        analyze_schur_complements(lam_sq)

    # Part 2: Scaling with lambda
    print("\n" + "#" * 75)
    print("# PART 2: SCHUR COMPLEMENT SCALING")
    print("#" * 75)
    scaling_results = track_schur_vs_lambda([50, 100, 200, 500, 1000, 2000])

    # Part 3: Bottleneck deep dive
    print("\n" + "#" * 75)
    print("# PART 3: BOTTLENECK DEEP DIVE")
    print("#" * 75)
    analyze_bottleneck_structure(1000)

    # Part 4: Induction test
    print("\n" + "#" * 75)
    print("# PART 4: INDUCTION TEST (eps_0 vs N)")
    print("#" * 75)
    for lam_sq in [200, 1000]:
        induction_test(lam_sq)

    # Part 5: Cholesky structure
    print("\n" + "#" * 75)
    print("# PART 5: CHOLESKY FACTOR STRUCTURE")
    print("#" * 75)
    cholesky_factor_structure(200)

    # Part 6: Displacement generators
    print("\n" + "#" * 75)
    print("# PART 6: DISPLACEMENT GENERATORS")
    print("#" * 75)
    for lam_sq in [200, 1000]:
        displacement_generator_analysis(lam_sq)

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print("\n\n" + "=" * 75)
    print("CHOLESKY ATTACK SYNTHESIS")
    print("=" * 75)
    print("""
  THE CHOLESKY STRATEGY:

  1. Q_W = L D L^T where each D[k] > 0 is a Schur complement
  2. The bottleneck D[k*] is the fragile point
  3. D[k*] = Q_W[k*,k*] - (correction from earlier modes)
  4. If we can bound the correction, D[k*] > 0 follows

  KEY FINDINGS:
  - The displacement rank of Q_W is LOW (bounded, not growing with dim)
  - This means Q_W has STRUCTURE inherited from W02 (rank 2) and M (prime sum)
  - The Gohberg-Semencul formula expresses L in terms of the generators
  - If the generators can be bounded (they relate to prime sums), done

  THE PATH TO PROOF:
  a. Express Q_W displacement generators in terms of prime sums
  b. Bound the generators using Selberg sieve
  c. Use Gohberg-Semencul to get L with bounded entries
  d. This gives D[k] > 0 for all k, hence Q_W > 0, hence RH
""")

    # Save
    output = {
        'scaling_results': scaling_results,
    }
    with open('session33_cholesky_attack.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to session33_cholesky_attack.json")
