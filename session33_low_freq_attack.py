"""
SESSION 33 — LOW FREQUENCY ATTACK

THE INSIGHT:
  The dangerous eigenvector lives in |n| <= 5 (99.9% energy).
  Restrict M to this ~10-dim LOW-FREQUENCY subspace within null(W02).
  On this small space, only SMALL PRIMES matter (p <= ~50).
  These can be computed EXACTLY — no PNT bounds needed.

  Combined with:
  - 2x2 range block PROVED
  - Schur-Horn proves 75% of null eigenvalues
  - Low-frequency attack handles the transition zone

  This might close the full proof for specific lambda values.

ALSO: Test the Connes 2026 connection — primes < 13 suffice for
finite approximations. Does restricting to p <= 13 preserve negativity?
"""

import numpy as np
import time, json, sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def low_freq_subspace(lam_sq, n_max=5, N=None):
    """
    Restrict to the low-frequency subspace |n| <= n_max within null(W02).
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)

    # null(W02)
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    D_null = P_null.shape[1]

    # Low-frequency projector: modes with |n| <= n_max
    low_freq_idx = [i for i in range(dim) if abs(i - N) <= n_max]
    P_low = np.eye(dim)[:, low_freq_idx]

    # Intersection: low-freq & null(W02)
    # Project P_low columns onto null(W02)
    P_low_null = P_null.T @ P_low  # D_null x (2*n_max+1)
    U, S, _ = np.linalg.svd(P_low_null, full_matrices=False)
    sig_mask = S > 0.01
    P_lf_null = P_null @ U[:, sig_mask]  # dim x d_effective
    d_eff = np.sum(sig_mask)

    # M restricted to this subspace
    M_lf = P_lf_null.T @ M @ P_lf_null
    QW_lf = P_lf_null.T @ QW @ P_lf_null

    evals_m = np.linalg.eigvalsh(M_lf)
    evals_qw = np.linalg.eigvalsh(QW_lf)

    # Decompose M
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)
    Md_lf = P_lf_null.T @ M_diag @ P_lf_null
    Ma_lf = P_lf_null.T @ M_alpha @ P_lf_null
    Mp_lf = P_lf_null.T @ M_prime @ P_lf_null

    # Schur-Horn on this small subspace
    mu = np.mean(evals_m)
    sigma = np.std(evals_m)
    sh = mu + sigma * np.sqrt((d_eff-1)/d_eff) if d_eff > 1 and sigma > 1e-15 else mu
    ratio = abs(mu)/sigma if sigma > 1e-15 else float('inf')

    print(f"\n  LOW FREQ |n|<={n_max}: lam^2={lam_sq}, eff_dim={d_eff}")
    print(f"    M eigenvalues: [{evals_m[0]:.6e}, ..., {evals_m[-1]:.6e}]")
    print(f"    M all negative: {evals_m[-1] < 1e-10}")
    print(f"    SH: mu={mu:.4e} sigma={sigma:.4e} |mu|/sig={ratio:.4f} bound={sh:.4e}")
    print(f"    M_diag trace: {np.trace(Md_lf):.6f}")
    print(f"    M_alpha trace: {np.trace(Ma_lf):.6f}")
    print(f"    M_prime trace: {np.trace(Mp_lf):.6f}")

    return d_eff, evals_m, M_lf, P_lf_null, primes


def small_prime_test(lam_sq, p_max_values=[7, 11, 13, 23, 50], N=None):
    """
    Test: if we restrict the prime sum to primes <= p_max,
    is M still negative on the low-frequency subspace?

    Connes 2026 showed primes < 13 suffice for finite approx.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1
    L_val = np.log(lam_sq)

    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    # Low-frequency subspace
    n_max = 5
    low_freq_idx = [i for i in range(dim) if abs(i - N) <= n_max]
    P_low = np.eye(dim)[:, low_freq_idx]
    P_low_null = P_null.T @ P_low
    U, S, _ = np.linalg.svd(P_low_null, full_matrices=False)
    P_lf_null = P_null @ U[:, S > 0.01]
    d_eff = np.sum(S > 0.01)

    Md_lf = P_lf_null.T @ M_diag @ P_lf_null
    Ma_lf = P_lf_null.T @ M_alpha @ P_lf_null

    ns = np.arange(-N, N+1, dtype=float)

    print(f"\n  SMALL PRIME TEST: lam^2={lam_sq}, |n|<=5 (dim={d_eff})")
    print(f"    Analytic part (M_diag+alpha) max eig: "
          f"{np.max(np.linalg.eigvalsh(Md_lf + Ma_lf)):.4e}")

    for p_max in p_max_values:
        # Build M_prime using only primes <= p_max
        Mp_restricted = np.zeros((dim, dim))
        for pk, logp, logpk in primes:
            p = int(np.round(np.exp(logp)))
            if p > p_max:
                continue
            for i in range(dim):
                m = ns[i]
                for j in range(dim):
                    n = ns[j]
                    if m != n:
                        q = (np.sin(2*np.pi*n*logpk/L_val) -
                             np.sin(2*np.pi*m*logpk/L_val)) / (np.pi*(m-n))
                    else:
                        q = 2*(L_val-logpk)/L_val * np.cos(2*np.pi*m*logpk/L_val)
                    Mp_restricted[i,j] += logp * pk**(-0.5) * q
        Mp_restricted = (Mp_restricted + Mp_restricted.T)/2

        Mp_lf = P_lf_null.T @ Mp_restricted @ P_lf_null
        M_total_lf = Md_lf + Ma_lf + Mp_lf

        evals = np.linalg.eigvalsh(M_total_lf)
        all_neg = evals[-1] < 1e-10

        # Count how many prime powers used
        n_pp = sum(1 for pk, logp, _ in primes if int(np.round(np.exp(logp))) <= p_max)

        print(f"    p<={p_max:>3} ({n_pp:>3} prime powers): "
              f"M evals [{evals[0]:.4e}, {evals[-1]:.4e}] "
              f"{'ALL NEG' if all_neg else 'HAS POS'}")


def explicit_small_matrix(lam_sq, N=None):
    """
    Build the EXPLICIT small matrix M on the low-frequency null subspace.
    Show each entry can be computed from:
    - M_diag entries (closed form — digamma, hypergeometric)
    - M_alpha entries (closed form — alpha coefficients)
    - M_prime entries (sum over small primes — EXACT for p <= lam)

    If this matrix is negative definite for all lambda, we have the proof
    (combined with Schur-Horn on the rest).
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    _, _, _, M_full, primes = compute_M_decomposition(lam_sq, N)
    W02, M, QW = build_all(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    # Get the low-frequency null subspace explicitly
    n_max = 3  # even smaller: |n| <= 3 for tractability
    low_idx = [i for i in range(dim) if abs(i-N) <= n_max]
    P_low = np.eye(dim)[:, low_idx]
    P_low_null = P_null.T @ P_low
    U, S, _ = np.linalg.svd(P_low_null, full_matrices=False)
    mask = S > 0.01
    P = P_null @ U[:, mask]
    d = np.sum(mask)

    M_small = P.T @ M @ P

    print(f"\n  EXPLICIT SMALL MATRIX: lam^2={lam_sq}, |n|<={n_max}, dim={d}")
    print(f"    M matrix ({d}x{d}):")
    for i in range(d):
        row = "    ["
        for j in range(d):
            row += f" {M_small[i,j]:>10.6f}"
        row += " ]"
        print(row)

    evals = np.linalg.eigvalsh(M_small)
    print(f"    Eigenvalues: {evals}")
    print(f"    ALL NEGATIVE: {evals[-1] < 1e-10}")
    print(f"    det(M_small) = {np.linalg.det(M_small):.6e}")
    print(f"    trace = {np.trace(M_small):.6f}")

    # Principal minors (Sylvester criterion for negative definiteness)
    # -M should be PD, so principal minors of -M should alternate: +, +, +, ...
    print(f"    Principal minors of -M_small:")
    for k in range(1, d+1):
        minor = np.linalg.det(-M_small[:k, :k])
        sign = "+" if minor > 0 else "-"
        print(f"      k={k}: det = {minor:>12.6e} ({sign})")

    return M_small, d


if __name__ == "__main__":
    print("SESSION 33 -- LOW FREQUENCY ATTACK")
    print("=" * 75)

    for lam_sq in [50, 200, 1000]:
        print(f"\n{'#'*75}")
        print(f"# lam^2 = {lam_sq}")
        print(f"{'#'*75}")

        # Low-frequency subspace analysis
        for n_max in [3, 5, 8]:
            low_freq_subspace(lam_sq, n_max)

        # Small prime test
        small_prime_test(lam_sq)

        # Explicit small matrix
        explicit_small_matrix(lam_sq)

    with open('session33_low_freq.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
    print(f"\nSaved to session33_low_freq.json")
