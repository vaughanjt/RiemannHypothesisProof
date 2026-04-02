"""
SESSION 34 — WHY DOES M HAVE EXACTLY ONE POSITIVE EIGENVALUE?

M = M_diag + M_alpha + M_prime

M_diag: diagonal, has BOTH positive and negative entries
M_alpha: off-diagonal, small
M_prime: sum over prime powers, dominant contributor

If we can show M has at most 1 positive eigenvalue, then:
  Q_W = W02 - M = (rank-2, signature (1,1)) - (signature (1, dim-1))
  The positive eigenvalue of M gets cancelled by W02
  => Q_W is PSD

APPROACH 1: Show M_prime has at most 1 positive eigenvalue.
  M_prime = sum_{p^k} Lambda(p^k)/sqrt(p^k) * T(p^k)
  Each T(p^k) is a self-adjoint matrix. What's its signature?
  If each T(p^k) has at most 1 positive eigenvalue, and the
  positive eigenvalues are ALIGNED (same direction), the sum
  also has at most 1 positive eigenvalue.

APPROACH 2: Show M is a rank-1 perturbation of a negative definite matrix.
  M = N + v*v^T where N <= 0 and v is a specific vector.
  Then M has at most 1 positive eigenvalue (from v*v^T).

APPROACH 3: The INERTIA theorem.
  If M = A + B where signature of A is known and ||B|| is small,
  then Sylvester's inertia law constrains the signature of M.
"""

import numpy as np
import time, json, sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def analyze_per_prime_signature(lam_sq, N=None):
    """
    Analyze the SIGNATURE of each individual T(p^k) matrix.

    If each T(p^k) has at most 1 positive eigenvalue,
    and the positive eigenvectors are aligned,
    the sum M_prime also has at most 1 positive eigenvalue.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N+1, dtype=float)

    _, _, _, _, primes_used = compute_M_decomposition(lam_sq, N)

    print(f"\nPER-PRIME SIGNATURE: lam^2={lam_sq}, dim={dim}")
    print(f"  {'p^k':>5} {'weight':>8} {'n_pos':>6} {'n_neg':>6} {'n_zero':>6} {'max_eig':>10} {'min_eig':>10}")

    total_one_pos = 0
    all_have_one_pos = True
    pos_eigvecs = []

    for pk, logp, logpk in primes_used[:30]:  # first 30
        # Build T(p^k)
        Q = np.zeros((dim, dim))
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n = ns[j]
                if m != n:
                    Q[i,j] = (np.sin(2*np.pi*n*logpk/L_f) -
                              np.sin(2*np.pi*m*logpk/L_f)) / (np.pi*(m-n))
                else:
                    Q[i,j] = 2*(L_f-logpk)/L_f * np.cos(2*np.pi*m*logpk/L_f)
        Q = (Q + Q.T) / 2
        weight = logp * pk**(-0.5)
        T = weight * Q

        evals = np.linalg.eigvalsh(T)
        n_pos = np.sum(evals > 1e-12)
        n_neg = np.sum(evals < -1e-12)
        n_zero = dim - n_pos - n_neg

        if n_pos != 1:
            all_have_one_pos = False
        if n_pos == 1:
            total_one_pos += 1
            # Store the positive eigenvector for alignment check
            idx_pos = np.argmax(evals)
            pos_eigvecs.append((pk, evals[idx_pos], np.linalg.eigh(T)[1][:, idx_pos]))

        print(f"  {pk:>5} {weight:>8.4f} {n_pos:>6} {n_neg:>6} {n_zero:>6} "
              f"{evals[-1]:>10.4e} {evals[0]:>10.4e}")

    print(f"\n  All T(p^k) have exactly 1 positive eigenvalue: {all_have_one_pos}")
    print(f"  {total_one_pos}/{min(len(primes_used), 30)} have exactly 1 positive eigenvalue")

    # Check ALIGNMENT of positive eigenvectors
    if len(pos_eigvecs) >= 2:
        print(f"\n  ALIGNMENT of positive eigenvectors:")
        v0 = pos_eigvecs[0][2]
        for pk, eig, v in pos_eigvecs[:10]:
            alignment = abs(np.dot(v0, v))
            print(f"    p^k={pk:>5}: alignment with p=2 eigvec = {alignment:.6f}")

    return all_have_one_pos


def rank_one_decomposition(lam_sq, N=None):
    """
    APPROACH 2: Try to write M = N + sigma * v * v^T
    where N is negative semidefinite and v is a specific vector.

    Strategy: find the positive eigenvalue lambda_+ and eigenvector v_+.
    Set sigma = lambda_+ and v = v_+. Then N = M - sigma * v_+ * v_+^T.
    Check if N is negative semidefinite.
    """
    if N is None:
        L = np.log(lam_sq)
        N_val = max(15, round(6*L))
    else:
        N_val = N
    dim = 2*N_val+1

    W02, M, QW = build_all(lam_sq, N_val)

    evals_M, evecs_M = np.linalg.eigh(M)

    # The positive eigenvalue and eigenvector
    pos_idx = evals_M > 1e-10
    n_pos = np.sum(pos_idx)

    print(f"\nRANK-1 DECOMPOSITION: lam^2={lam_sq}")
    print(f"  M has {n_pos} positive eigenvalue(s)")

    if n_pos == 1:
        idx = np.argmax(evals_M)
        lambda_plus = evals_M[idx]
        v_plus = evecs_M[:, idx]

        # N = M - lambda_plus * v_plus * v_plus^T
        N_mat = M - lambda_plus * np.outer(v_plus, v_plus)
        evals_N = np.linalg.eigvalsh(N_mat)

        print(f"  lambda_+ = {lambda_plus:.6f}")
        print(f"  N = M - lambda_+ * v_+ v_+^T")
        print(f"  N eigenvalues: [{evals_N[0]:.4e}, ..., {evals_N[-1]:.4e}]")
        print(f"  N negative semidefinite: {evals_N[-1] < 1e-10}")

        if evals_N[-1] < 1e-10:
            print(f"\n  *** M = (neg semidef) + (rank-1 positive) ***")
            print(f"  *** M has AT MOST 1 positive eigenvalue by construction ***")

            # What does v_+ look like?
            ns = np.arange(-N_val, N_val+1)
            print(f"\n  Positive eigenvector v_+:")
            sorted_idx = np.argsort(np.abs(v_plus))[::-1]
            for i in range(min(10, dim)):
                idx_i = sorted_idx[i]
                print(f"    n={ns[idx_i]:>4}: {v_plus[idx_i]:>+.6f}")

            # Is v_+ aligned with the EVEN eigenvector of W02?
            # W02 has eigenvectors u_v (even) and u_w (odd)
            ew, ev = np.linalg.eigh(W02)
            range_idx = np.abs(ew) > np.max(np.abs(ew)) * 1e-10
            u_range = ev[:, range_idx]  # 2 eigenvectors

            for j in range(u_range.shape[1]):
                alignment = abs(np.dot(v_plus, u_range[:, j]))
                print(f"    Alignment with W02 eigenvec {j}: {alignment:.6f}")

    elif n_pos == 0:
        print(f"  M is NEGATIVE SEMIDEFINITE — even better!")
    else:
        print(f"  M has {n_pos} positive eigenvalues — rank-1 decomposition insufficient")


def why_one_positive(lam_sq, N=None):
    """
    DEEP DIVE: Why does M have exactly 1 positive eigenvalue?

    Decompose: M = M_diag + M_alpha + M_prime

    M_diag: diagonal. How many positive eigenvalues?
    M_alpha: off-diagonal. How many positive eigenvalues?
    M_prime: prime sum. How many positive eigenvalues?

    Track how the number of positive eigenvalues changes as we
    add components. Where does the "collapse" to 1 happen?
    """
    if N is None:
        L = np.log(lam_sq)
        N_val = max(15, round(6*L))
    else:
        N_val = N
    dim = 2*N_val+1

    W02, M, QW = build_all(lam_sq, N_val)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N_val)

    def count_pos(A):
        return np.sum(np.linalg.eigvalsh(A) > 1e-10)

    n_diag = count_pos(M_diag)
    n_alpha = count_pos(M_alpha)
    n_prime = count_pos(M_prime)
    n_da = count_pos(M_diag + M_alpha)
    n_dp = count_pos(M_diag + M_prime)
    n_ap = count_pos(M_alpha + M_prime)
    n_full = count_pos(M)

    print(f"\nWHY ONE POSITIVE: lam^2={lam_sq}, dim={dim}")
    print(f"  Component signatures (positive eigenvalue count):")
    print(f"    M_diag:              {n_diag:>3} positive out of {dim}")
    print(f"    M_alpha:             {n_alpha:>3} positive out of {dim}")
    print(f"    M_prime:             {n_prime:>3} positive out of {dim}")
    print(f"    M_diag + M_alpha:    {n_da:>3} positive")
    print(f"    M_diag + M_prime:    {n_dp:>3} positive")
    print(f"    M_alpha + M_prime:   {n_ap:>3} positive")
    print(f"    M_full (all three):  {n_full:>3} positive")

    # The TRACE tells us about the average eigenvalue
    tr_diag = np.trace(M_diag)
    tr_alpha = np.trace(M_alpha)
    tr_prime = np.trace(M_prime)
    tr_full = np.trace(M)

    print(f"\n  Traces:")
    print(f"    M_diag:  {tr_diag:>10.4f}")
    print(f"    M_alpha: {tr_alpha:>10.4f}")
    print(f"    M_prime: {tr_prime:>10.4f}")
    print(f"    M_full:  {tr_full:>10.4f}")

    # The single positive eigenvalue: where does it come from?
    evals_M = np.linalg.eigvalsh(M)
    lambda_plus = evals_M[-1]
    sum_neg = np.sum(evals_M[evals_M < 0])

    print(f"\n  The single positive eigenvalue: {lambda_plus:.4f}")
    print(f"  Sum of negative eigenvalues: {sum_neg:.4f}")
    print(f"  Trace = pos + neg = {lambda_plus + sum_neg:.4f} (check: {tr_full:.4f})")
    print(f"  Ratio: lambda_+ / |sum_neg| = {lambda_plus / abs(sum_neg):.6f}")

    # M_prime has how many positive eigenvalues?
    evals_prime = np.linalg.eigvalsh(M_prime)
    n_pos_prime = np.sum(evals_prime > 1e-10)
    print(f"\n  M_prime signature: {n_pos_prime} pos, {np.sum(evals_prime < -1e-10)} neg")
    print(f"  M_prime max eigenvalue: {evals_prime[-1]:.4f}")
    print(f"  M_prime min eigenvalue: {evals_prime[0]:.4f}")

    # KEY TEST: is M_prime's max eigenvalue eigenvector aligned with M's?
    _, evecs_prime = np.linalg.eigh(M_prime)
    _, evecs_M = np.linalg.eigh(M)
    alignment = abs(np.dot(evecs_prime[:, -1], evecs_M[:, -1]))
    print(f"\n  Alignment of max eigenvectors (M_prime vs M): {alignment:.6f}")

    # What about M_diag's contribution?
    # M_diag has many positive eigenvalues but M has only 1.
    # The prime sum KILLS the extra positive eigenvalues.
    print(f"\n  M_diag has {n_diag} positive eigenvalues")
    print(f"  After adding M_prime: collapses to {n_dp} positive")
    print(f"  M_prime KILLS {n_diag - n_dp} positive eigenvalues of M_diag")


if __name__ == "__main__":
    print("SESSION 34 -- WHY ONE POSITIVE EIGENVALUE?")
    print("=" * 75)

    for lam_sq in [50, 200, 1000]:
        print(f"\n{'#'*75}")
        print(f"# lam^2 = {lam_sq}")
        print(f"{'#'*75}")

        analyze_per_prime_signature(lam_sq)
        rank_one_decomposition(lam_sq)
        why_one_positive(lam_sq)

    with open('session34_one_positive.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
    print(f"\nSaved to session34_one_positive.json")
