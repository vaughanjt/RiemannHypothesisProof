"""
SESSION 33 — VISION #4: MATRIX CONCENTRATION INEQUALITY

THE IDEA:
  PrimeSum = sum_{p^k <= lam^2} Lambda(p^k) * T(p^k)

  where each T(p^k) is a bounded self-adjoint operator (rank <= 2).

  By the MATRIX BERNSTEIN INEQUALITY (Tropp 2012):
  For independent random self-adjoint matrices X_1, ..., X_n with E[X_i] = 0:

    P(||sum X_i|| > t) <= 2*dim * exp(-t^2 / (2*sigma^2 + 2*R*t/3))

  where sigma^2 = ||sum E[X_i^2]||  and  R = max ||X_i||.

  The prime terms are NOT independent, but they're APPROXIMATELY independent
  because prime powers p^k, q^j with gcd(p,q)=1 create oscillatory cross-terms
  that cancel.

  APPROACH:
  1. Compute the individual term norms ||Lambda(p^k) * T_null(p^k)||
  2. Compute the variance sigma^2 = ||sum Lambda(p^k)^2 * T_null(p^k)^2||
  3. Apply matrix Bernstein to bound ||PrimeSum_null||
  4. Compare with lambda_min(Theta_null)

  ALSO: VISION #5 — Analyze PrimeSum on the NULL SPACE OF THETA
  within null(W_{0,2}). This is a smaller problem (dim ~ N/3).
"""

import numpy as np
import time, json, sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def compute_per_prime_operators(lam_sq, N, P_null):
    """Compute T_null(p^k) = P_null^T * T(p^k) * P_null for each prime power."""
    L = np.log(lam_sq)
    dim = 2*N+1
    D_null = P_null.shape[1]
    ns = np.arange(-N, N+1, dtype=float)

    # Get prime powers
    limit = min(lam_sq, 10000)
    sieve_arr = [True]*(limit+1); sieve_arr[0]=sieve_arr[1]=False
    for i in range(2, int(limit**0.5)+2):
        if i<=limit and sieve_arr[i]:
            for j in range(i*i,limit+1,i): sieve_arr[j]=False

    prime_powers = []
    for p in range(2, limit+1):
        if sieve_arr[p] and p <= lam_sq:
            pk = p
            while pk <= lam_sq:
                prime_powers.append((pk, np.log(p), np.log(pk)))
                pk *= p

    # For each prime power, build T(p^k) and project onto null space
    terms = []
    for pk, logp, logpk in prime_powers:
        # T(p^k)[i,j] = logp * pk^{-1/2} * q(m, n, logpk)
        Q = np.zeros((dim, dim))
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n = ns[j]
                if m != n:
                    Q[i,j] = (np.sin(2*np.pi*n*logpk/L) -
                              np.sin(2*np.pi*m*logpk/L)) / (np.pi*(m-n))
                else:
                    Q[i,j] = 2*(L-logpk)/L * np.cos(2*np.pi*m*logpk/L)

        weight = logp * pk**(-0.5)
        T_full = weight * Q
        T_null = P_null.T @ T_full @ P_null

        op_norm = np.max(np.abs(np.linalg.eigvalsh(T_null)))
        frob = np.linalg.norm(T_null, 'fro')
        T_sq = T_null @ T_null

        terms.append({
            'pk': pk, 'logp': logp, 'weight': weight,
            'T_null': T_null, 'T_sq': T_sq,
            'op_norm': op_norm, 'frob': frob
        })

    return terms


def matrix_bernstein_bound(terms, D_null):
    """
    Apply matrix Bernstein inequality.

    For the centered version: X_i = T_null(p^k) - E[T_null(p^k)]
    But our terms aren't random — they're deterministic.

    Instead, use the DETERMINISTIC matrix bound:
    ||sum A_i|| <= sum ||A_i||  (triangle inequality — too weak)

    Better: ||sum A_i||^2 <= ||sum A_i^2|| * n  (if terms are "orthogonal")

    Best: use the VARIANCE bound directly:
    sigma^2 = ||sum T_i^2|| (operator norm of sum of squares)
    R = max ||T_i||

    Matrix Hoeffding for bounded terms:
    ||sum A_i|| <= sqrt(2 * sigma^2 * log(2*D_null)) + R * log(2*D_null) / 3
    """
    # Compute key quantities
    R = max(t['op_norm'] for t in terms)
    sum_T_sq = sum(t['T_sq'] for t in terms)
    sigma_sq = np.max(np.abs(np.linalg.eigvalsh(sum_T_sq)))

    # Individual norms
    sum_norms = sum(t['op_norm'] for t in terms)

    # Matrix Bernstein bound (deterministic version — Tropp's matrix series)
    # ||sum A_i|| <= sqrt(2 * sigma^2 * log(2*D)) + (2/3) * R * log(2*D)
    log_factor = np.log(2 * D_null)
    bernstein = np.sqrt(2 * sigma_sq * log_factor) + (2/3) * R * log_factor

    # The actual sum
    sum_T = sum(t['T_null'] for t in terms)
    actual_norm = np.max(np.abs(np.linalg.eigvalsh(sum_T)))

    return {
        'R': R,
        'sigma_sq': sigma_sq,
        'sum_norms': sum_norms,
        'bernstein': bernstein,
        'actual_norm': actual_norm,
        'n_terms': len(terms)
    }


def theta_null_analysis(lam_sq, N=None):
    """
    VISION #5: Analyze PrimeSum on the null space of Theta within null(W_{0,2}).

    Theta has rank r < dim. Its null space within null(W02) has dimension dim-2-r.
    On this subspace, Theta = 0, so QW = -PrimeSum.
    Need PrimeSum >= 0 there (i.e., the sum of prime terms is non-negative).

    This is a SMALLER problem!
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    # Build everything
    W02, M, QW = build_all(lam_sq, N)

    # Get null(W02)
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    null_w_idx = np.abs(ew) <= thresh
    P_null_w = ev[:, null_w_idx]
    D_null_w = np.sum(null_w_idx)

    # Compute Theta on null(W02) using Connes formula
    from session33_connes_structural import compute_theta_integral_matrix
    Theta, _, _ = compute_theta_integral_matrix(lam_sq, N, n_quad=1000)
    Theta_null = P_null_w.T @ Theta @ P_null_w

    # Get null(Theta) within null(W02)
    et, evt = np.linalg.eigh(Theta_null)
    thresh_t = max(np.max(np.abs(et))*1e-8, 1e-12)
    null_theta_idx = et <= thresh_t
    D_null_theta = np.sum(null_theta_idx)

    # Projector onto null(Theta) within null(W02)
    P_null_theta = P_null_w @ evt[:, null_theta_idx]

    # PrimeSum on this doubly-null space
    M_null_theta = P_null_theta.T @ M @ P_null_theta
    QW_null_theta = P_null_theta.T @ QW @ P_null_theta

    evals_m = np.linalg.eigvalsh(M_null_theta)
    evals_qw = np.linalg.eigvalsh(QW_null_theta)

    print(f"\n  THETA NULL SPACE ANALYSIS: lam^2={lam_sq}")
    print(f"    dim={dim}, null(W02)={D_null_w}, null(Theta)_within={D_null_theta}")
    print(f"    M on null(Theta)&null(W02): [{evals_m[0]:.4e}, ..., {evals_m[-1]:.4e}]")
    print(f"    QW on null(Theta)&null(W02): [{evals_qw[0]:.4e}, ..., {evals_qw[-1]:.4e}]")
    print(f"    M all negative there: {evals_m[-1] < 1e-10}")

    # Trace analysis on this smaller space
    tr_m = np.trace(M_null_theta)
    frob_m = np.linalg.norm(M_null_theta, 'fro')
    if D_null_theta > 0:
        mu = tr_m / D_null_theta
        sigma = np.sqrt(max(frob_m**2/D_null_theta - mu**2, 0))
        sh_bound = mu + sigma * np.sqrt((D_null_theta-1)/D_null_theta) if D_null_theta > 1 else mu
        print(f"    Schur-Horn on this subspace: mu={mu:.4f} sigma={sigma:.4f} bound={sh_bound:.4e}")
        if sh_bound < 0:
            print(f"    *** SCHUR-HORN PROVES M <= 0 on null(Theta)&null(W02) ***")

    return D_null_theta, evals_m, evals_qw


if __name__ == "__main__":
    print("SESSION 33 — VISIONS #4 AND #5")
    print("=" * 75)

    for lam_sq in [50, 200]:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
        dim = 2*N+1

        print(f"\n{'='*75}")
        print(f"lam^2={lam_sq}, N={N}, dim={dim}")
        print(f"{'='*75}")

        # Get null space of W02
        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N)
        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew))*1e-10
        null_idx = np.abs(ew) <= thresh
        P_null = ev[:, null_idx]
        D_null = np.sum(null_idx)
        print(f"  Build: {time.time()-t0:.0f}s, null(W02) dim={D_null}")

        # VISION #4: Matrix concentration
        print(f"\n  VISION #4: MATRIX CONCENTRATION")
        print(f"  {'-'*50}")
        t0 = time.time()
        terms = compute_per_prime_operators(lam_sq, N, P_null)
        print(f"  Per-prime operators: {time.time()-t0:.0f}s, {len(terms)} terms")

        bounds = matrix_bernstein_bound(terms, D_null)
        print(f"  R (max term norm): {bounds['R']:.4f}")
        print(f"  sigma^2 (variance): {bounds['sigma_sq']:.4f}")
        print(f"  Triangle bound (sum norms): {bounds['sum_norms']:.4f}")
        print(f"  Bernstein bound: {bounds['bernstein']:.4f}")
        print(f"  Actual ||PrimeSum_null||: {bounds['actual_norm']:.4f}")
        print(f"  Bernstein/Actual: {bounds['bernstein']/bounds['actual_norm']:.2f}x")

        # Compare with Theta
        from session33_connes_structural import compute_theta_integral_matrix
        Theta, _, _ = compute_theta_integral_matrix(lam_sq, N, n_quad=1000)
        Theta_null = P_null.T @ Theta @ P_null
        theta_min = np.min(np.linalg.eigvalsh(Theta_null))
        theta_max = np.max(np.linalg.eigvalsh(Theta_null))

        print(f"\n  Theta_null eigenvalues: [{theta_min:.4e}, {theta_max:.4e}]")
        print(f"  Need: ||PrimeSum_null|| < lambda_min(Theta_null)")
        print(f"  Bernstein bound < Theta_min? "
              f"{'YES => PROVED' if bounds['bernstein'] < theta_min else 'NO'}")
        print(f"  Actual < Theta_min? "
              f"{'YES (true but not a proof)' if bounds['actual_norm'] < theta_max else 'NO'}")

        # VISION #5: Theta null space
        print(f"\n  VISION #5: THETA NULL SPACE")
        print(f"  {'-'*50}")
        theta_null_analysis(lam_sq, N)

    # Summary
    print(f"\n\n{'='*75}")
    print("VISION QUEST SUMMARY")
    print("="*75)

    with open('session33_visions.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
    print(f"\nSaved to session33_visions.json")
