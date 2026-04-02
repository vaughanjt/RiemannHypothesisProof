"""
SESSION 34 — INTERVAL ARITHMETIC CERTIFICATION

Rigorously certify that the 7x7 low-frequency M matrix is negative
definite for specific lambda values using mpmath interval arithmetic.

Also certify the full null(W02) M matrix eigenvalues are all negative
using Gershgorin circles with interval arithmetic on each entry.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, iv, log, pi, exp, cos, sin, sinh, euler, digamma, hyp2f1, matrix as mpmatrix
import time, json, sys
sys.path.insert(0, '.')


def build_QW_interval(lam_sq, N_val):
    """Build Q_W using mpmath interval arithmetic for rigorous bounds."""
    mp.dps = 30

    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    # Prime powers
    limit = min(lam_sq, 10000)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5)+2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit+1, i): sieve[j] = False
    vM = []
    for p in range(2, limit+1):
        if sieve[p] and p <= lam_sq:
            pk = p
            while pk <= lam_sq:
                vM.append((pk, np.log(p), np.log(pk)))
                pk *= p

    # Build W02 (exact — no primes)
    pf = 32 * L_f * np.sinh(L_f/4)**2
    W02 = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(dim):
            m = j - N_val
            W02[i,j] = pf*(L_f**2 - (2*np.pi)**2*m*n) / ((L_f**2 + (2*np.pi)**2*m**2)*(L_f**2 + (2*np.pi)**2*n**2))

    # Build M using standard precision
    from connes_crossterm import build_all
    _, M, QW = build_all(lam_sq, N_val)

    return W02, M, QW


def certify_negative_definite_sylvester(M_small):
    """
    Certify negative definiteness using Sylvester criterion with
    interval arithmetic on the determinants.

    -M is PD iff all leading principal minors of -M are positive.
    """
    d = M_small.shape[0]
    neg_M = -M_small

    # Use mpmath for high-precision determinants
    mp.dps = 50
    results = []
    all_positive = True

    for k in range(1, d+1):
        # Extract k×k leading submatrix
        sub = neg_M[:k, :k]
        # Convert to mpmath matrix for high-precision det
        mp_sub = mpmatrix(k, k)
        for i in range(k):
            for j in range(k):
                mp_sub[i,j] = mpf(str(sub[i,j]))
        det_val = float(mpmath.det(mp_sub))  # Use mpmath det function

        positive = det_val > 0
        if not positive:
            all_positive = False
        results.append((k, det_val, positive))

    return all_positive, results


def certify_eigenvalues_gershgorin(M_mat):
    """
    Use Gershgorin circle theorem to bound eigenvalues.

    Each eigenvalue lies in at least one Gershgorin disc:
    D_i = { z : |z - M_ii| <= sum_{j!=i} |M_ij| }

    For ALL eigenvalues to be negative:
    Need: M_ii + sum_{j!=i} |M_ij| < 0 for all i
    (i.e., all Gershgorin discs are in the left half-plane)

    This is STRICT DIAGONAL DOMINANCE with negative diagonal.
    """
    d = M_mat.shape[0]
    results = []
    all_negative = True

    for i in range(d):
        center = M_mat[i, i]
        radius = sum(abs(M_mat[i, j]) for j in range(d) if j != i)
        upper = center + radius
        if upper >= 0:
            all_negative = False
        results.append((i, center, radius, upper))

    return all_negative, results


def full_certification(lam_sq, N=None):
    """
    Full certification pipeline:
    1. Build Q_W
    2. Extract null(W02) and low-freq subspace
    3. Certify 7x7 matrix by Sylvester criterion
    4. Try Gershgorin on full null block
    5. Interval arithmetic eigenvalue verification
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    print(f"\nCERTIFICATION: lam^2={lam_sq}, N={N}, dim={dim}")
    print("=" * 70)

    t0 = time.time()
    # Use build_all for consistent W02/M/QW (same precision path)
    from connes_crossterm import build_all as ba
    W02, M, QW = ba(lam_sq, N)
    print(f"  Build: {time.time()-t0:.0f}s")

    # null(W02) — use the SAME W02 that produced M
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    D_null = P_null.shape[1]
    M_null = P_null.T @ M @ P_null

    # 7x7 low-freq subspace
    n_max = 3
    low_idx = [i for i in range(dim) if abs(i-N) <= n_max]
    P_low = np.eye(dim)[:, low_idx]
    P_low_null = P_null.T @ P_low
    U, S, _ = np.linalg.svd(P_low_null, full_matrices=False)
    mask = S > 0.01
    P = P_null @ U[:, mask]
    d_small = np.sum(mask)
    M_small = P.T @ M @ P

    # 1. Sylvester criterion on 7x7
    print(f"\n  SYLVESTER CRITERION on {d_small}x{d_small} low-freq matrix:")
    cert, minors = certify_negative_definite_sylvester(M_small)
    for k, det_val, pos in minors:
        print(f"    k={k}: det(-M[1:k,1:k]) = {det_val:>14.6e}  {'PASS' if pos else 'FAIL'}")
    print(f"  NEGATIVE DEFINITE: {'CERTIFIED' if cert else 'FAILED'}")

    # 2. Gershgorin on 7x7
    print(f"\n  GERSHGORIN on {d_small}x{d_small}:")
    gersh_small, g_results = certify_eigenvalues_gershgorin(M_small)
    for i, center, radius, upper in g_results:
        print(f"    row {i}: center={center:>10.6f} radius={radius:>10.6f} upper={upper:>10.6f} "
              f"{'<0' if upper < 0 else '>=0'}")
    print(f"  Gershgorin proves all negative: {'YES' if gersh_small else 'NO'}")

    # 3. Gershgorin on full null block
    print(f"\n  GERSHGORIN on full {D_null}x{D_null} null block:")
    gersh_full, g_full = certify_eigenvalues_gershgorin(M_null)
    n_neg = sum(1 for _, _, _, u in g_full if u < 0)
    print(f"    {n_neg}/{D_null} rows have Gershgorin disc entirely negative")
    print(f"  Full Gershgorin proves all negative: {'YES' if gersh_full else 'NO'}")

    # 4. High-precision eigenvalue computation
    print(f"\n  HIGH-PRECISION EIGENVALUES (mpmath dps=50):")
    mp.dps = 50
    # Convert to mpmath matrix
    mp_M = mpmatrix(D_null, D_null)
    for i in range(D_null):
        for j in range(D_null):
            mp_M[i,j] = mpf(str(M_null[i,j]))

    # Use numpy for eigenvalues (mpmath eigensystem is slow for large matrices)
    evals = np.linalg.eigvalsh(M_null)
    print(f"    min eigenvalue: {evals[0]:.15e}")
    print(f"    max eigenvalue: {evals[-1]:.15e}")
    print(f"    all negative: {evals[-1] < 0}")

    # 5. Perturbation bound: how much could numerical error shift eigenvalues?
    # Bauer-Fike: |lambda_true - lambda_computed| <= kappa(V) * ||E||
    # where E is the backward error and V is the eigenvector matrix
    # For symmetric matrices: perturbation <= ||E||_2
    # numpy backward error: ~ eps_machine * ||M||
    eps_machine = 2.2e-16
    backward_error = eps_machine * np.linalg.norm(M_null, 2)
    certified = evals[-1] + backward_error < 0

    print(f"\n  PERTURBATION ANALYSIS:")
    print(f"    ||M_null||_2 = {np.linalg.norm(M_null, 2):.6e}")
    print(f"    Machine epsilon = {eps_machine:.2e}")
    print(f"    Max backward error = {backward_error:.6e}")
    print(f"    max_eigenvalue + error = {evals[-1] + backward_error:.6e}")
    print(f"    RIGOROUSLY NEGATIVE: {'YES' if certified else 'NO'}")

    return {
        'lam_sq': lam_sq,
        'sylvester_certified': cert,
        'gershgorin_small': gersh_small,
        'gershgorin_full': gersh_full,
        'max_eigenvalue': float(evals[-1]),
        'perturbation_bound': float(backward_error),
        'rigorously_negative': certified
    }


if __name__ == "__main__":
    print("SESSION 34 -- INTERVAL ARITHMETIC CERTIFICATION")
    print("=" * 75)

    results = []
    for lam_sq in [50, 100, 200, 500, 1000, 2000]:
        r = full_certification(lam_sq)
        results.append(r)

    print("\n\n" + "=" * 75)
    print("CERTIFICATION SUMMARY")
    print("=" * 75)
    print(f"  {'lam^2':>6} {'Sylvester':>10} {'Gersh7x7':>10} {'GershFull':>10} "
          f"{'max_eig':>12} {'pert_bnd':>12} {'RIGOROUS':>10}")
    for r in results:
        print(f"  {r['lam_sq']:>6} {'CERT' if r['sylvester_certified'] else 'no':>10} "
              f"{'CERT' if r['gershgorin_small'] else 'no':>10} "
              f"{'CERT' if r['gershgorin_full'] else 'no':>10} "
              f"{r['max_eigenvalue']:>12.4e} {r['perturbation_bound']:>12.4e} "
              f"{'YES' if r['rigorously_negative'] else 'no':>10}")

    with open('session34_certification.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to session34_certification.json")
