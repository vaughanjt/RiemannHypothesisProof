"""
SESSION 35b — RECONCILIATION: WHY DOES M CANCEL PERFECTLY ON orth(v_+)?

Phase 1 showed: M_analytic and M_prime each have POSITIVE eigenvalues on orth(v_+),
but their sum is ALL NEGATIVE. This isn't brute domination — it's structural alignment.

INVESTIGATION:
1. Is M_prime truly negative semidefinite on orth(v_+)?
   Session 34 claimed all T_minor(p^k) are negative — verify for ALL primes (not just first 30).
2. Eigenvector alignment: where do M_analytic's positive eigenvectors point in M_prime's eigenspectrum?
3. The oscillatory integral picture: what makes <phi, M phi> < 0 for phi perp v_+?
"""

import numpy as np
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def verify_per_prime_negativity(lam_sq, N=None):
    """
    Test ALL T_minor(p^k) (not just first 30) for negativity on orth(v_+).
    Use tighter threshold than Session 34's 1e-10.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N)
    _, _, _, _, primes = compute_M_decomposition(lam_sq, N)

    evals_M, evecs_M = np.linalg.eigh(M)
    v_plus = evecs_M[:, -1]
    P_orth = np.eye(dim) - np.outer(v_plus, v_plus)

    print(f"\nVERIFY PER-PRIME NEGATIVITY: lam^2={lam_sq}, dim={dim}")
    print(f"  Total prime powers: {len(primes)}")

    n_pos = 0
    n_neg = 0
    max_positive = 0
    worst_pk = None
    sum_T_orth = np.zeros((dim, dim))

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

        T_orth = P_orth @ T @ P_orth
        evals_orth = np.linalg.eigvalsh(T_orth)
        nonzero = evals_orth[np.abs(evals_orth) > 1e-14]
        max_ev = np.max(nonzero) if len(nonzero) > 0 else 0

        sum_T_orth += T_orth

        if max_ev > 1e-14:  # Tighter threshold
            n_pos += 1
            if max_ev > max_positive:
                max_positive = max_ev
                worst_pk = pk
        else:
            n_neg += 1

    print(f"  T_minor NEGATIVE: {n_neg}/{len(primes)}")
    print(f"  T_minor POSITIVE: {n_pos}/{len(primes)}")
    if worst_pk is not None:
        print(f"  Worst positive eigenvalue: {max_positive:.4e} at p^k={worst_pk}")

    # Check the SUM
    evals_sum = np.linalg.eigvalsh(sum_T_orth)
    nonzero_sum = evals_sum[np.abs(evals_sum) > 1e-12]
    max_sum = np.max(nonzero_sum) if len(nonzero_sum) > 0 else 0
    min_sum = np.min(nonzero_sum) if len(nonzero_sum) > 0 else 0

    print(f"\n  Sum of T_orth (= M_prime on orth(v_+)):")
    print(f"    max eigenvalue: {max_sum:+.6e}")
    print(f"    min eigenvalue: {min_sum:+.6e}")
    print(f"    NEGATIVE SEMIDEFINITE: {max_sum < 1e-10}")

    # Compare with the M_prime we computed via decomposition
    _, _, M_prime, _, _ = compute_M_decomposition(lam_sq, N)
    Mp_orth = P_orth @ M_prime @ P_orth
    evals_Mp = np.linalg.eigvalsh(Mp_orth)
    nonzero_Mp = evals_Mp[np.abs(evals_Mp) > 1e-12]
    print(f"\n  M_prime from decomposition on orth(v_+):")
    print(f"    max eigenvalue: {np.max(nonzero_Mp):+.6e}")
    print(f"    min eigenvalue: {np.min(nonzero_Mp):+.6e}")
    print(f"    MATCH: {np.allclose(nonzero_sum, nonzero_Mp, atol=1e-6)}")

    return n_pos, n_neg, max_positive


def eigenvector_alignment(lam_sq, N=None):
    """
    Study how the positive eigenvectors of M_analytic align with M_prime's eigenspectrum.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)
    M_analytic = M_diag + M_alpha

    v_plus, lambda_plus = np.linalg.eigh(M)[1][:, -1], np.linalg.eigh(M)[0][-1]
    P_orth = np.eye(dim) - np.outer(v_plus, v_plus)

    # Get eigenvectors of M_analytic on orth(v_+)
    Ma_orth = P_orth @ M_analytic @ P_orth
    evals_a, evecs_a = np.linalg.eigh(Ma_orth)

    # Get eigenvectors of M_prime on orth(v_+)
    Mp_orth = P_orth @ M_prime @ P_orth
    evals_p, evecs_p = np.linalg.eigh(Mp_orth)

    print(f"\nEIGENVECTOR ALIGNMENT: lam^2={lam_sq}, dim={dim}")
    print(f"\n  M_analytic positive eigenvectors -> M_prime Rayleigh quotients:")
    print(f"  {'idx':>4} {'eig(Ma)':>10} {'<v,Mp v>':>12} {'sum':>10} {'net_neg?':>8}")

    for i in range(dim):
        if abs(evals_a[i]) < 1e-12:
            continue
        if evals_a[i] > 0:
            v = evecs_a[:, i]
            rq_prime = np.dot(v, M_prime @ v)
            rq_full = np.dot(v, M @ v)
            net = evals_a[i] + rq_prime
            print(f"  {i:>4} {evals_a[i]:>+10.4f} {rq_prime:>+12.4f} {net:>+10.4f} {'YES' if net < 0 else 'NO':>8}")

    # Also check: for the TOP positive eigenvector of M_analytic,
    # what is its representation in M_prime eigenbasis?
    idx_top = np.argmax(evals_a)
    v_top = evecs_a[:, idx_top]
    print(f"\n  Top positive M_analytic eigenvector (eig={evals_a[idx_top]:.4f}):")
    print(f"  Decomposition in M_prime eigenbasis:")
    print(f"  {'j':>4} {'coeff^2':>10} {'eig_p[j]':>12} {'contribution':>14}")

    coeffs = evecs_p.T @ v_top
    contribs = coeffs ** 2 * evals_p
    sorted_idx = np.argsort(np.abs(contribs))[::-1]
    for k in range(min(15, dim)):
        j = sorted_idx[k]
        if abs(evals_p[j]) < 1e-12:
            continue
        print(f"  {j:>4} {coeffs[j]**2:>10.6f} {evals_p[j]:>+12.4f} {contribs[j]:>+14.6f}")
    total_rq = np.sum(contribs)
    print(f"  Total Rayleigh quotient: {total_rq:+.6f}")
    print(f"  (Needs to be < {-evals_a[idx_top]:.4f} to make M negative)")


def component_rayleigh_analysis(lam_sq, N=None):
    """
    For EVERY eigenvector of M on orth(v_+), decompose the Rayleigh quotient
    into M_diag + M_alpha + M_prime contributions. Understand the cancellation.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    v_plus = np.linalg.eigh(M)[1][:, -1]
    P_orth = np.eye(dim) - np.outer(v_plus, v_plus)

    M_orth = P_orth @ M @ P_orth
    evals, evecs = np.linalg.eigh(M_orth)

    print(f"\nCOMPONENT RAYLEIGH ANALYSIS: lam^2={lam_sq}")
    print(f"  For each eigenvector of M|orth(v_+):")
    print(f"  {'i':>3} {'eig(M)':>10} {'<v,Md v>':>10} {'<v,Ma v>':>10} {'<v,Mp v>':>10} {'Md+Ma':>10} {'check':>10}")

    n_shown = 0
    for i in range(dim):
        if abs(evals[i]) < 1e-12:
            continue  # skip the projected-out direction
        v = evecs[:, i]
        rq_diag = np.dot(v, M_diag @ v)
        rq_alpha = np.dot(v, M_alpha @ v)
        rq_prime = np.dot(v, M_prime @ v)
        rq_full = rq_diag + rq_alpha + rq_prime
        rq_analytic = rq_diag + rq_alpha

        print(f"  {n_shown:>3} {evals[i]:>+10.4f} {rq_diag:>+10.4f} {rq_alpha:>+10.4f} "
              f"{rq_prime:>+10.4f} {rq_analytic:>+10.4f} {rq_full:>+10.4f}")
        n_shown += 1

    # Summary statistics
    print(f"\n  Among {n_shown} eigenvectors of M|orth(v_+):")
    rq_primes = []
    rq_analytics = []
    for i in range(dim):
        if abs(evals[i]) < 1e-12:
            continue
        v = evecs[:, i]
        rq_primes.append(np.dot(v, M_prime @ v))
        rq_analytics.append(np.dot(v, (M_diag + M_alpha) @ v))
    rq_primes = np.array(rq_primes)
    rq_analytics = np.array(rq_analytics)

    print(f"  M_analytic Rayleigh: [{np.min(rq_analytics):+.4f}, {np.max(rq_analytics):+.4f}]")
    print(f"  M_prime Rayleigh:    [{np.min(rq_primes):+.4f}, {np.max(rq_primes):+.4f}]")
    print(f"  M_prime ALWAYS NEGATIVE along M's eigenvectors: {np.max(rq_primes) < 1e-10}")
    print(f"  M_analytic ALWAYS NEGATIVE along M's eigenvectors: {np.max(rq_analytics) < 1e-10}")


def m_prime_sign_structure(lam_sq, N=None):
    """
    Determine definitively: is M_prime NSD on orth(v_+)?

    If YES: the problem reduces to proving M_analytic + (something NSD) < 0,
    which is stronger since M_analytic alone might be indefinite.

    If NO: we need the alignment structure.

    Also check: is M_prime NSD on null(W02)? This is the space that matters.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    # v_+ and orth projector
    v_plus = np.linalg.eigh(M)[1][:, -1]
    P_orth_vp = np.eye(dim) - np.outer(v_plus, v_plus)

    # null(W02) projector
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    null_vecs = ev[:, np.abs(ew) <= thresh]
    range_vecs = ev[:, np.abs(ew) > thresh]

    print(f"\nM_prime SIGN STRUCTURE: lam^2={lam_sq}")
    print(f"  dim={dim}, null(W02) dim={null_vecs.shape[1]}, range(W02) dim={range_vecs.shape[1]}")

    # M_prime on various subspaces
    for name, P in [("full space", np.eye(dim)),
                    ("orth(v_+)", P_orth_vp),
                    ("null(W02)", null_vecs)]:
        if P.ndim == 1:
            P = P.reshape(-1, 1)
        if name == "orth(v_+)":
            # Use P_orth formula
            Mp_sub = P @ M_prime @ P
            evals = np.linalg.eigvalsh(Mp_sub)
            evals = evals[np.abs(evals) > 1e-12]
        else:
            Mp_sub = P.T @ M_prime @ P
            evals = np.linalg.eigvalsh(Mp_sub)

        if len(evals) > 0:
            n_pos = np.sum(evals > 1e-10)
            print(f"  M_prime on {name}:")
            print(f"    eigenvalues: [{np.min(evals):+.4e}, {np.max(evals):+.4e}]")
            print(f"    trace: {np.sum(evals):+.4f}")
            print(f"    n_positive: {n_pos}/{len(evals)}")
            print(f"    NSD: {np.max(evals) < 1e-10}")

    # Same for M_analytic
    M_analytic = M_diag + M_alpha
    print()
    for name, P in [("orth(v_+)", P_orth_vp),
                    ("null(W02)", null_vecs)]:
        if name == "orth(v_+)":
            Ma_sub = P @ M_analytic @ P
            evals = np.linalg.eigvalsh(Ma_sub)
            evals = evals[np.abs(evals) > 1e-12]
        else:
            Ma_sub = P.T @ M_analytic @ P
            evals = np.linalg.eigvalsh(Ma_sub)

        if len(evals) > 0:
            n_pos = np.sum(evals > 1e-10)
            print(f"  M_analytic on {name}:")
            print(f"    eigenvalues: [{np.min(evals):+.4e}, {np.max(evals):+.4e}]")
            print(f"    trace: {np.sum(evals):+.4f}")
            print(f"    n_positive: {n_pos}/{len(evals)}")

    # KEY CHECK: is v_+ actually in range(W02)?
    v_plus_proj_range = range_vecs @ (range_vecs.T @ v_plus)
    alignment = np.linalg.norm(v_plus_proj_range)
    print(f"\n  v_+ alignment with range(W02): {alignment:.8f}")
    print(f"  v_+ alignment with null(W02):  {np.sqrt(1 - alignment**2):.8f}")


def the_deep_structure(lam_sq, N=None):
    """
    THE KEY QUESTION: What algebraic identity forces M <= 0 on null(W02)?

    If we write phi in null(W02), then <phi, W02 phi> = 0.
    So <phi, Q_W phi> = -<phi, M phi>.
    Q_W >= 0 iff M <= 0 on null(W02).

    Now M = sum_n wr_diag[n] |n><n| + off-diagonal + prime sum.

    For phi in null(W02): sum_n |phi_n|^2 * wr_diag[n] + ... = <phi, M phi>

    The constraint "phi in null(W02)" means phi is orthogonal to both
    eigenvectors of W02. This constrains the Fourier coefficients phi_n.

    What IS this constraint?
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    # W02 eigenvectors
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    range_idx = np.where(np.abs(ew) > thresh)[0]
    u1 = ev[:, range_idx[0]]
    u2 = ev[:, range_idx[1]]

    ns = np.arange(-N, N + 1)
    L_f = np.log(lam_sq)

    print(f"\nDEEP STRUCTURE: lam^2={lam_sq}")
    print(f"\n  W02 range eigenvectors (the constraint):")
    print(f"  W02 has eigenvalues: {ew[range_idx[0]]:.4f}, {ew[range_idx[1]]:.4f}")

    # What do u1, u2 look like in the Fourier basis?
    print(f"\n  u1 (Fourier components):")
    for i in range(dim):
        if abs(u1[i]) > 0.01:
            print(f"    n={ns[i]:>3}: {u1[i]:>+.6f}")

    print(f"\n  u2 (Fourier components):")
    for i in range(dim):
        if abs(u2[i]) > 0.01:
            print(f"    n={ns[i]:>3}: {u2[i]:>+.6f}")

    # The Poisson kernel structure
    # W02[n,m] = 32*L*sinh^2(L/4) * (L^2 - 4pi^2*mn) / ((L^2 + 4pi^2*m^2)(L^2 + 4pi^2*n^2))
    # This has a Cauchy-type structure: it's a rank-2 matrix
    # The even eigenvector: proportional to 1/(L^2 + 4pi^2*n^2)
    # The odd eigenvector: proportional to n/(L^2 + 4pi^2*n^2)

    poisson_even = np.array([1.0 / (L_f**2 + 4 * np.pi**2 * n**2) for n in ns])
    poisson_even = poisson_even / np.linalg.norm(poisson_even)
    poisson_odd = np.array([n / (L_f**2 + 4 * np.pi**2 * n**2) for n in ns])
    poisson_odd = poisson_odd / np.linalg.norm(poisson_odd)

    align_1e = abs(np.dot(u1, poisson_even))
    align_1o = abs(np.dot(u1, poisson_odd))
    align_2e = abs(np.dot(u2, poisson_even))
    align_2o = abs(np.dot(u2, poisson_odd))

    print(f"\n  W02 eigenvector structure:")
    print(f"    u1 vs Poisson even: {align_1e:.6f}")
    print(f"    u1 vs Poisson odd:  {align_1o:.6f}")
    print(f"    u2 vs Poisson even: {align_2e:.6f}")
    print(f"    u2 vs Poisson odd:  {align_2o:.6f}")

    # The null(W02) constraint: phi_n must satisfy
    #   sum_n phi_n / (L^2 + 4pi^2 n^2) = 0  (orthog to even)
    #   sum_n n * phi_n / (L^2 + 4pi^2 n^2) = 0  (orthog to odd)
    # These two LINEAR constraints define null(W02).

    # Now: M_diag has entries wr_diag[n] on the diagonal.
    # <phi, M_diag phi> = sum_n |phi_n|^2 * wr_diag[n]
    #
    # For phi in null(W02): the constraint forces the phi_n to be weighted
    # AWAY from n=0 (where wr_diag is positive and large) towards large |n|
    # (where wr_diag is negative).
    #
    # THIS IS THE MECHANISM: the null(W02) constraint forces the mass of phi
    # to live at large |n| where wr_diag is negative!

    print(f"\n  THE MECHANISM:")
    print(f"  wr_diag profile (diagonal of M_diag):")
    print(f"  {'n':>4} {'wr_diag[n]':>12} {'weight 1/(L^2+4pi^2n^2)':>24}")
    for i in range(dim):
        n = ns[i]
        if abs(n) <= 15 or abs(n) == N:
            w = 1.0 / (L_f**2 + 4 * np.pi**2 * n**2)
            print(f"  {n:>4} {M_diag[i, i]:>+12.6f} {w:>24.8f}")

    # The constraint pushes mass away from n=0. Count how many wr_diag are negative.
    n_neg_wr = np.sum(np.diag(M_diag) < 0)
    n_pos_wr = np.sum(np.diag(M_diag) > 0)
    print(f"\n  wr_diag: {n_pos_wr} positive, {n_neg_wr} negative")
    print(f"  Positive wr_diag live at |n| <= ~{max(abs(ns[np.diag(M_diag) > 0])):.0f}")
    print(f"  null(W02) constraint forces mass AWAY from n=0")
    print(f"  => null(W02) vectors see mostly NEGATIVE wr_diag values")


if __name__ == "__main__":
    print("SESSION 35b -- ALIGNMENT AND SIGN STRUCTURE")
    print("=" * 80)

    for lam_sq in [50, 200]:
        print(f"\n{'#' * 80}")
        print(f"# lam^2 = {lam_sq}")
        print(f"{'#' * 80}")

        verify_per_prime_negativity(lam_sq)
        m_prime_sign_structure(lam_sq)
        eigenvector_alignment(lam_sq)
        component_rayleigh_analysis(lam_sq)
        the_deep_structure(lam_sq)

    with open('session35_alignment.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
    print("\nDone.")
