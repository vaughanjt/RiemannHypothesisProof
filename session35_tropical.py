"""
SESSION 35c -- TROPICAL LIFT OF M

THE IDEA (from Grok's vision):
  Replace classical von Mangoldt weights with tropical analogues.
  In max-plus world, the Euler product becomes a tropical polynomial.
  The single positive eigenvalue = unique tropical root from the pole.
  Orthogonal complement is automatically negative by tropical Hodge index.

  If classical M is a deformation of tropical M_trop that never crosses
  zero eigenvalues, sign is preserved from the tropical limit.

IMPLEMENTATION:
  1. MASLOV DEQUANTIZATION: For each matrix entry M_prime[n,m] = sum_i a_i,
     the tropical version keeps only the dominant term: M_trop[n,m] = a_{i*}
     where i* = argmax|a_i|.

  2. h-DEFORMATION: Use the Maslov interpolation with temperature h:
     At h=1: classical sum.  At h->0: only max term survives.
     Track eigenvalues on null(W02) as h varies.

  3. CUMULATIVE BUILD: Add prime powers in decreasing contribution order.
     Track when eigenvalues first become negative and whether they stay negative.

  4. TROPICAL STRUCTURE ANALYSIS: What is the combinatorial structure of M_trop?
     Is its negativity on null(W02) provable from fan/matroid theory?
"""

import numpy as np
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def build_per_prime_matrices(lam_sq, N=None):
    """Build individual T(p^k) matrices for each prime power."""
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    _, _, _, _, primes = compute_M_decomposition(lam_sq, N)

    T_matrices = []
    for pk, logp, logpk in primes:
        Q = np.zeros((dim, dim))
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n = ns[j]
                if m != n:
                    Q[i, j] = (np.sin(2 * np.pi * n * logpk / L_f) -
                               np.sin(2 * np.pi * m * logpk / L_f)) / (np.pi * (m - n))
                else:
                    Q[i, j] = 2 * (L_f - logpk) / L_f * np.cos(2 * np.pi * m * logpk / L_f)
        Q = (Q + Q.T) / 2
        w = logp * pk ** (-0.5)
        T = w * Q
        T_matrices.append((pk, logp, w, T))

    return T_matrices, dim, N


def tropical_dominant_term(lam_sq, N=None):
    """
    MASLOV DEQUANTIZATION: For each entry of M_prime, keep only the
    prime power with the largest absolute contribution.

    M_trop[n,m] = a_{i*} where i* = argmax_i |a_i| and a_i = w_i * q(n,m,y_i)
    """
    T_matrices, dim, N_val = build_per_prime_matrices(lam_sq, N)
    W02, M, QW = build_all(lam_sq, N_val)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N_val)

    # Build M_trop: for each entry, keep only dominant prime power
    M_trop = np.zeros((dim, dim))
    dominant_pk = np.zeros((dim, dim), dtype=int)  # which p^k dominates each entry

    for i in range(dim):
        for j in range(i, dim):
            best_val = 0
            best_pk = 0
            for idx, (pk, logp, w, T) in enumerate(T_matrices):
                val = T[i, j]
                if abs(val) > abs(best_val):
                    best_val = val
                    best_pk = pk
            M_trop[i, j] = best_val
            M_trop[j, i] = best_val
            dominant_pk[i, j] = best_pk
            dominant_pk[j, i] = best_pk

    # M_tropical = M_diag + M_alpha + M_trop (analytic part unchanged)
    M_full_trop = M_diag + M_alpha + M_trop

    # Get null(W02)
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    # Eigenvalues on null(W02)
    M_null_classical = P_null.T @ M @ P_null
    M_null_trop = P_null.T @ M_full_trop @ P_null
    Mp_null_classical = P_null.T @ M_prime @ P_null
    Mp_null_trop = P_null.T @ M_trop @ P_null

    evals_classical = np.linalg.eigvalsh(M_null_classical)
    evals_trop = np.linalg.eigvalsh(M_null_trop)

    print(f"\nTROPICAL DOMINANT TERM: lam^2={lam_sq}, dim={dim}")
    print(f"  # prime powers: {len(T_matrices)}")

    # How many distinct primes dominate?
    unique_pks = set(dominant_pk[np.triu_indices(dim)])
    print(f"  # distinct dominant p^k: {len(unique_pks)}")
    from collections import Counter
    pk_counts = Counter(dominant_pk[np.triu_indices(dim)].flatten())
    print(f"  Most common dominant p^k:")
    for pk, count in pk_counts.most_common(8):
        print(f"    p^k={pk}: {count} entries")

    print(f"\n  M_prime (classical) on null(W02):")
    print(f"    evals: [{np.min(Mp_null_classical.flatten()):+.4f}, {np.max(np.linalg.eigvalsh(Mp_null_classical)):+.4f}]")
    print(f"  M_trop (dominant term only) on null(W02):")
    evals_mp_trop = np.linalg.eigvalsh(Mp_null_trop)
    print(f"    evals: [{np.min(evals_mp_trop):+.4f}, {np.max(evals_mp_trop):+.4f}]")

    print(f"\n  Full M (classical) on null(W02):")
    print(f"    max eig: {np.max(evals_classical):+.6e}")
    print(f"    ALL NEG: {np.max(evals_classical) < 1e-10}")
    print(f"  Full M_trop on null(W02):")
    print(f"    max eig: {np.max(evals_trop):+.6e}")
    print(f"    ALL NEG: {np.max(evals_trop) < 1e-10}")

    # Frobenius distance
    frob_diff = np.linalg.norm(M_prime - M_trop, 'fro')
    frob_orig = np.linalg.norm(M_prime, 'fro')
    print(f"\n  ||M_prime - M_trop||_F / ||M_prime||_F = {frob_diff/frob_orig:.4f}")
    print(f"  Dropped terms = {frob_diff/frob_orig:.1%} of Frobenius norm")

    return M_trop, M_full_trop, dominant_pk, evals_trop


def h_deformation(lam_sq, N=None):
    """
    Maslov h-deformation: interpolate from classical (h=1) to tropical (h->0).

    For each entry: M_prime_h[n,m] = h * log(sum_i exp(a_i / h))
    where a_i are the individual prime power contributions.

    But this only works for positive a_i. For mixed signs, use:
    M_prime_h[n,m] = (sum_i sign(a_i) * |a_i|^{1/h})^h  (power mean deformation)

    Actually simpler: weighted power mean of the absolute values with signs preserved.
    At h=1: arithmetic mean (classical sum).
    At h->0: only the max|a_i| term survives.

    Use: M_prime_h[n,m] = sign(a_{i*}) * (sum_i |a_i|^{1/h})^h
    where a_{i*} is the term achieving max|a_i|.

    Even simpler approach that preserves linearity:
    Weight each T(p^k) by w^{1/h} / (sum w^{1/h}) * (sum w)
    where w are the original weights. As h->0, all weight goes to max-weight prime.
    """
    T_matrices, dim, N_val = build_per_prime_matrices(lam_sq, N)
    W02, M, QW = build_all(lam_sq, N_val)
    M_diag, M_alpha, M_prime, _, _ = compute_M_decomposition(lam_sq, N_val)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    # Simple approach: for each h, compute M_prime(h) where entries use
    # softmax weighting of prime power contributions
    # At h=inf: all terms equal weight (not quite classical)
    # At h=1: classical
    # At h->0: only dominant term

    # Even simpler: power-law deformation on the weights
    # w_h(p^k) = log(p) * p^{-k/(2h)}
    # At h=1: classical weights
    # At h->0: only p=2,k=1 survives (smallest prime power dominates)
    # At h->inf: all weights -> 1 (flat)

    print(f"\nH-DEFORMATION: lam^2={lam_sq}, dim={dim}")
    print(f"  Using power-law weight deformation: w_h = log(p) * p^(-k/(2h))")
    print(f"  h=1: classical | h<1: concentrate on small primes | h->0: only p=2")
    print()

    L_f = np.log(lam_sq)
    ns = np.arange(-N_val, N_val + 1, dtype=float)

    # Precompute q matrices (unweighted)
    primes = [(pk, logp, logpk) for pk, logp, _, _ in T_matrices for logpk in [np.log(pk)]]
    # Actually just use T_matrices info
    q_matrices = []
    for pk, logp, w, T in T_matrices:
        q_matrices.append((pk, logp, T / w))  # unweighted kernel

    h_values = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]

    print(f"  {'h':>6} {'max_eig(M_h|null)':>18} {'min_eig':>12} {'ALL_NEG':>8} {'||Mp_h||_F':>12}")

    for h in h_values:
        # Build M_prime(h) with modified weights
        M_prime_h = np.zeros((dim, dim))
        for pk, logp, Q in q_matrices:
            k = round(np.log(pk) / logp) if logp > 0 else 1
            w_h = logp * pk ** (-1.0 / (2 * h))
            M_prime_h += w_h * Q

        M_prime_h = (M_prime_h + M_prime_h.T) / 2

        M_h = M_diag + M_alpha + M_prime_h
        M_h = (M_h + M_h.T) / 2

        M_h_null = P_null.T @ M_h @ P_null
        evals_h = np.linalg.eigvalsh(M_h_null)
        max_ev = np.max(evals_h)
        min_ev = np.min(evals_h)
        all_neg = max_ev < 1e-10
        frob = np.linalg.norm(M_prime_h, 'fro')

        flag = " ***" if not all_neg else ""
        print(f"  {h:>6.2f} {max_ev:>+18.6e} {min_ev:>+12.4f} {'YES' if all_neg else 'NO':>8} {frob:>12.2f}{flag}")

    # At h=1, verify we match classical
    M_prime_h1 = np.zeros((dim, dim))
    for pk, logp, Q in q_matrices:
        w_h1 = logp * pk ** (-0.5)
        M_prime_h1 += w_h1 * Q
    M_prime_h1 = (M_prime_h1 + M_prime_h1.T) / 2
    diff = np.linalg.norm(M_prime_h1 - M_prime, 'fro')
    print(f"\n  Verification: ||M_prime(h=1) - M_prime_classical||_F = {diff:.2e}")


def cumulative_prime_build(lam_sq, N=None):
    """
    Add prime powers one at a time in order of DECREASING weight,
    tracking eigenvalues on null(W02) at each step.

    This shows the "assembly" of M from its dominant constituents.
    """
    T_matrices, dim, N_val = build_per_prime_matrices(lam_sq, N)
    W02, M, QW = build_all(lam_sq, N_val)
    M_diag, M_alpha, _, _, _ = compute_M_decomposition(lam_sq, N_val)
    M_analytic = M_diag + M_alpha

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    # Sort by weight (decreasing)
    sorted_T = sorted(T_matrices, key=lambda x: x[2], reverse=True)

    print(f"\nCUMULATIVE PRIME BUILD: lam^2={lam_sq}, dim={dim}")
    print(f"  Starting from M_analytic (no primes), adding prime powers by weight")
    print()

    # Start with M_analytic
    M_cumul = M_analytic.copy()
    M_cumul_null = P_null.T @ M_cumul @ P_null
    evals = np.linalg.eigvalsh(M_cumul_null)
    print(f"  {'step':>4} {'p^k':>6} {'weight':>8} {'max_eig(null)':>14} {'min_eig':>12} {'ALL_NEG':>8}")
    print(f"  {0:>4} {'---':>6} {'---':>8} {np.max(evals):>+14.6e} {np.min(evals):>+12.4f} "
          f"{'YES' if np.max(evals) < 1e-10 else 'NO':>8}")

    first_neg_step = None
    last_pos_step = None

    for step, (pk, logp, w, T) in enumerate(sorted_T):
        M_cumul = M_cumul + T
        M_cumul_null = P_null.T @ M_cumul @ P_null
        evals = np.linalg.eigvalsh(M_cumul_null)
        max_ev = np.max(evals)
        all_neg = max_ev < 1e-10

        if all_neg and first_neg_step is None:
            first_neg_step = step + 1
        if not all_neg:
            last_pos_step = step + 1

        if step < 20 or step == len(sorted_T) - 1 or (step + 1) % 10 == 0:
            print(f"  {step+1:>4} {pk:>6} {w:>8.4f} {max_ev:>+14.6e} {np.min(evals):>+12.4f} "
                  f"{'YES' if all_neg else 'NO':>8}")

    print(f"\n  First all-negative at step: {first_neg_step}/{len(sorted_T)}")
    if last_pos_step is not None:
        print(f"  Last positive at step: {last_pos_step}/{len(sorted_T)}")
    print(f"  => Monotone from step {(last_pos_step or 0) + 1} onward")


def tropical_structure(lam_sq, N=None):
    """
    Analyze the STRUCTURE of M_trop (dominant-term matrix).

    Key question: Is M_trop a Toeplitz-like matrix? Does it have
    matroid structure? Is its fan/complex structure compatible with
    tropical Hodge index?
    """
    T_matrices, dim, N_val = build_per_prime_matrices(lam_sq, N)
    W02, M, QW = build_all(lam_sq, N_val)
    M_diag, M_alpha, M_prime, _, _ = compute_M_decomposition(lam_sq, N_val)

    # Build M_trop and track which p^k dominates
    M_trop = np.zeros((dim, dim))
    dominant = {}  # (i,j) -> (pk, contribution)
    ns = np.arange(-N_val, N_val + 1, dtype=float)

    for i in range(dim):
        for j in range(dim):
            best_val = 0
            best_pk = 0
            for pk, logp, w, T in T_matrices:
                if abs(T[i, j]) > abs(best_val):
                    best_val = T[i, j]
                    best_pk = pk
            M_trop[i, j] = best_val
            dominant[(i, j)] = (best_pk, best_val)

    print(f"\nTROPICAL STRUCTURE: lam^2={lam_sq}")

    # Check near-Toeplitz structure
    # A Toeplitz matrix has M[i,j] = f(i-j).
    # M_trop: is it approximately Toeplitz?
    diag_vals = {}  # k -> list of values on k-th diagonal
    for i in range(dim):
        for j in range(dim):
            k = i - j
            if k not in diag_vals:
                diag_vals[k] = []
            diag_vals[k].append(M_trop[i, j])

    # Measure Toeplitz-ness: for each diagonal, how constant are the values?
    toeplitz_scores = []
    for k in sorted(diag_vals.keys()):
        vals = np.array(diag_vals[k])
        if len(vals) > 1:
            cv = np.std(vals) / (np.abs(np.mean(vals)) + 1e-15)
            toeplitz_scores.append((k, cv, np.mean(vals)))

    print(f"\n  Toeplitz analysis of M_trop:")
    print(f"  {'diag':>5} {'mean':>10} {'CV':>10} {'Toeplitz?':>10}")
    for k, cv, mean in toeplitz_scores[:15]:
        print(f"  {k:>5} {mean:>+10.4f} {cv:>10.4f} {'YES' if cv < 0.1 else 'no':>10}")

    # Rank analysis of M_trop
    sv = np.linalg.svd(M_trop, compute_uv=False)
    sv_norm = sv / sv[0]
    effective_rank = np.sum(sv_norm > 0.01)
    print(f"\n  Singular value spectrum of M_trop:")
    print(f"  Effective rank (sv > 1% of max): {effective_rank}")
    print(f"  Top 10 SVs: {', '.join(f'{s:.3f}' for s in sv[:10])}")

    # Which primes dominate the tropical matrix?
    pk_coverage = {}
    for (i, j), (pk, val) in dominant.items():
        if pk not in pk_coverage:
            pk_coverage[pk] = 0
        pk_coverage[pk] += 1
    total = dim * dim
    print(f"\n  Prime power coverage in M_trop:")
    for pk in sorted(pk_coverage.keys()):
        count = pk_coverage[pk]
        print(f"    p^k={pk:>5}: {count:>6} entries ({count/total:.1%})")

    # The TROPICAL EIGENVALUE (max-plus)
    # In max-plus algebra: eigenvalue = max average weight of a cycle
    # For a real matrix, this is related to the spectral radius of the tropicalization
    print(f"\n  Tropical (max-plus) analysis:")
    # Max cycle mean: max_{cycle C} (sum of entries along C) / (length of C)
    # For self-loops: M_trop[i,i]
    max_diag = np.max(np.diag(M_trop))
    min_diag = np.min(np.diag(M_trop))
    print(f"    Max diagonal (length-1 cycle): {max_diag:+.6f}")
    print(f"    Min diagonal (length-1 cycle): {min_diag:+.6f}")

    # For length-2 cycles: max (M[i,j] + M[j,i]) / 2 = max M[i,j] (symmetric)
    max_offdiag = np.max(M_trop[np.triu_indices(dim, k=1)])
    print(f"    Max off-diagonal: {max_offdiag:+.6f}")

    return M_trop


def entrywise_deformation(lam_sq, N=None):
    """
    Smoothly deform from M_trop to M_prime entrywise.

    M_prime(t) = t * M_prime + (1-t) * M_trop

    Track eigenvalues on null(W02) as t goes from 0 (tropical) to 1 (classical).
    If eigenvalues stay negative, the tropical-to-classical bridge holds.
    """
    T_matrices, dim, N_val = build_per_prime_matrices(lam_sq, N)
    W02, M, QW = build_all(lam_sq, N_val)
    M_diag, M_alpha, M_prime, _, _ = compute_M_decomposition(lam_sq, N_val)

    # Build M_trop
    M_trop = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            best_val = 0
            for pk, logp, w, T in T_matrices:
                if abs(T[i, j]) > abs(best_val):
                    best_val = T[i, j]
            M_trop[i, j] = best_val
    M_trop = (M_trop + M_trop.T) / 2

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    print(f"\nENTRYWISE DEFORMATION M_trop -> M_prime: lam^2={lam_sq}")
    print(f"  M(t) = M_diag + M_alpha + [(1-t)*M_trop + t*M_prime]")
    print()
    print(f"  {'t':>6} {'max_eig(null)':>16} {'min_eig':>12} {'ALL_NEG':>8}")

    t_vals = np.linspace(0, 1, 21)
    crossings = []
    for t in t_vals:
        Mp_t = (1 - t) * M_trop + t * M_prime
        M_t = M_diag + M_alpha + Mp_t
        M_t = (M_t + M_t.T) / 2
        M_t_null = P_null.T @ M_t @ P_null
        evals = np.linalg.eigvalsh(M_t_null)
        max_ev = np.max(evals)
        all_neg = max_ev < 1e-10
        flag = " ***" if not all_neg else ""
        print(f"  {t:>6.2f} {max_ev:>+16.6e} {np.min(evals):>+12.4f} {'YES' if all_neg else 'NO':>8}{flag}")
        if not all_neg:
            crossings.append(t)

    if len(crossings) == 0:
        print(f"\n  *** DEFORMATION IS MONOTONE: NO ZERO CROSSINGS ***")
        print(f"  *** M_trop and M_classical have same sign on null(W02) ***")
    else:
        print(f"\n  Zero crossings at t = {crossings}")


def pure_tropical_prime_sum(lam_sq, N=None):
    """
    PURE TROPICAL approach: replace the SUM over prime powers with MAX.

    For each test vector phi:
    <phi, M_prime phi> = sum_{p^k} w(p^k) * <phi, T(p^k) phi>
                       = sum_{p^k} a(p^k, phi)

    Tropical version: max_{p^k} a(p^k, phi)

    If max_{p^k} a(p^k, phi) < 0 for all phi in null(W02), and the
    remaining terms are also negative, then the sum is negative.

    But actually: if ALL a(p^k, phi) < 0, then sum < 0.
    This is the per-prime negativity check from earlier (which FAILED).

    Alternative: define M_prime_trop via the Maslov limit of the matrix.
    Keep M_prime entrywise: M_prime_trop[n,m] = the dominant-sign term.
    """
    T_matrices, dim, N_val = build_per_prime_matrices(lam_sq, N)
    W02, M, QW = build_all(lam_sq, N_val)
    M_diag, M_alpha, M_prime, _, _ = compute_M_decomposition(lam_sq, N_val)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    # For each null vector phi, compute the per-prime Rayleigh quotients
    # and check if the DOMINANT one controls the sign
    n_samples = 200
    np.random.seed(42)

    print(f"\nPURE TROPICAL PRIME SUM: lam^2={lam_sq}")
    print(f"  For random phi in null(W02), check per-prime <phi,T phi>:")
    print()

    n_all_neg = 0
    n_max_neg = 0
    max_ratio = 0  # max / sum ratio

    for trial in range(n_samples):
        # Random vector in null(W02)
        coeffs = np.random.randn(P_null.shape[1])
        phi = P_null @ coeffs
        phi = phi / np.linalg.norm(phi)

        rqs = []
        for pk, logp, w, T in T_matrices:
            rq = np.dot(phi, T @ phi)
            rqs.append(rq)

        rqs = np.array(rqs)
        total = np.sum(rqs)  # = <phi, M_prime phi>
        max_rq = np.max(rqs)
        min_rq = np.min(rqs)
        dominant_rq = rqs[np.argmax(np.abs(rqs))]

        if np.all(rqs < 1e-10):
            n_all_neg += 1
        if max_rq < 0:
            n_max_neg += 1

        if abs(total) > 1e-15:
            ratio = max_rq / abs(total)
            if ratio > max_ratio:
                max_ratio = ratio

        if trial < 5:
            n_pos = np.sum(rqs > 1e-10)
            n_neg = np.sum(rqs < -1e-10)
            print(f"  trial {trial}: sum={total:+.4f}, max={max_rq:+.4f}, min={min_rq:+.4f}, "
                  f"#pos={n_pos}, #neg={n_neg}")

    print(f"\n  ALL per-prime RQ negative: {n_all_neg}/{n_samples} ({n_all_neg/n_samples:.1%})")
    print(f"  Max per-prime RQ negative: {n_max_neg}/{n_samples}")
    print(f"  Max positive RQ / |total|: {max_ratio:.4f}")

    # Alternative: group primes by the SIGN of their contribution
    # and check if negative group always wins
    print(f"\n  GROUP ANALYSIS (prime powers by sign of typical RQ):")
    # Use first eigenvector of M|null as the test
    M_null = P_null.T @ M @ P_null
    _, evecs_null = np.linalg.eigh(M_null)
    # Most negative eigenvector
    phi_test = P_null @ evecs_null[:, 0]
    phi_test = phi_test / np.linalg.norm(phi_test)

    pos_sum = 0
    neg_sum = 0
    for pk, logp, w, T in T_matrices:
        rq = np.dot(phi_test, T @ phi_test)
        if rq > 0:
            pos_sum += rq
        else:
            neg_sum += rq

    print(f"  Along most-negative eigenvector:")
    print(f"    Positive prime contributions: {pos_sum:+.4f}")
    print(f"    Negative prime contributions: {neg_sum:+.4f}")
    print(f"    Ratio |neg|/pos: {abs(neg_sum)/pos_sum:.2f}x" if pos_sum > 1e-15 else "    All negative!")


if __name__ == "__main__":
    print("SESSION 35c -- TROPICAL LIFT")
    print("=" * 80)

    for lam_sq in [50, 200, 1000]:
        print(f"\n{'#' * 80}")
        print(f"# lam^2 = {lam_sq}")
        print(f"{'#' * 80}")

        tropical_dominant_term(lam_sq)
        entrywise_deformation(lam_sq)
        h_deformation(lam_sq)
        cumulative_prime_build(lam_sq)

    # Detailed structure at lam^2=200
    print(f"\n{'#' * 80}")
    print(f"# DETAILED TROPICAL STRUCTURE at lam^2=200")
    print(f"{'#' * 80}")
    tropical_structure(200)
    pure_tropical_prime_sum(200)

    with open('session35_tropical.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
    print(f"\nDone. Results saved.")
