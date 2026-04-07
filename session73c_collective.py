"""
SESSION 73c -- THE COLLECTIVE ZERO-CROSSING

73b found: all 66 middle eigenvalues of M(t) are proportional to (1-t).
They cross zero SIMULTANEOUSLY at t=1. This is a collective phenomenon.

Key questions:
  1. Is the proportionality eig_k ~ (1-t) EXACT or approximate?
  2. Do the 66 eigenvectors share a common structure?
  3. What subspace do the 66 near-zero eigenvectors span?
  4. Is M restricted to this subspace approximately zero?
  5. Does M decompose as: (big positive) + (big negative) + (near zero cluster)?
  6. What's the RANK of M minus its extreme eigenvalues?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def extract_decomposition(lam_sq):
    """Full decomposition of M and its Cauchy structure."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

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

    a_n = np.array([wr[abs(int(n))] for n in ns]) + a_prime
    B_n = alpha + B_prime

    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        L_pure = (B_n[None, :] - B_n[:, None]) / nm

    for i in range(dim):
        if 0 < i < dim - 1:
            L_pure[i, i] = (B_n[i + 1] - B_n[i - 1]) / 2
        elif i == 0:
            L_pure[i, i] = B_n[1] - B_n[0]
        else:
            L_pure[i, i] = B_n[-1] - B_n[-2]
    L_pure = (L_pure + L_pure.T) / 2

    _, M, QW = build_all_fast(lam_sq, N)
    D_diag = np.diag(M) - np.diag(L_pure)

    return L_pure, D_diag, M, QW, N, L, dim, ns


def run():
    print()
    print('#' * 76)
    print('  SESSION 73c -- THE COLLECTIVE ZERO-CROSSING')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Identify the three spectral groups
    # ==================================================================
    print(f'\n  === STEP 1: THREE SPECTRAL GROUPS ===\n')

    for lam_sq in [200, 1000, 5000]:
        Lp, Dd, M, QW, N, L, dim, ns = extract_decomposition(lam_sq)
        evals, evecs = np.linalg.eigh(M)

        # Group 1: the top positive eigenvalue
        eig_top = evals[-1]

        # Group 3: clearly negative (say |eig| > 0.01)
        bulk_neg = evals[np.abs(evals) > 0.01]
        bulk_neg = bulk_neg[bulk_neg < 0]
        n_bulk = len(bulk_neg)

        # Group 2: near-zero cluster (|eig| < 0.01)
        near_zero = evals[np.abs(evals) < 0.01]
        n_near_zero = len(near_zero)

        print(f'  lam^2={lam_sq}: dim={dim}')
        print(f'    Group 1 (top positive): eig = {eig_top:.4f} (1 eigenvalue)')
        print(f'    Group 2 (near-zero):    {n_near_zero} eigenvalues, '
              f'range [{near_zero.min():.2e}, {near_zero.max():.2e}]')
        print(f'    Group 3 (bulk negative): {n_bulk} eigenvalues, '
              f'range [{bulk_neg.min():.4f}, {bulk_neg.max():.4f}]')
        print(f'    Total: 1 + {n_near_zero} + {n_bulk} = {1 + n_near_zero + n_bulk} (dim={dim})')
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: The near-zero cluster at t=1 — are they EXACTLY zero?
    # ==================================================================
    print(f'  === STEP 2: NEAR-ZERO CLUSTER PRECISION ===\n')

    lam_sq = 1000
    Lp, Dd, M, QW, N, L, dim, ns = extract_decomposition(lam_sq)
    evals, evecs = np.linalg.eigh(M)

    # The near-zero eigenvalues
    near_zero_mask = np.abs(evals) < 0.01
    near_zero_evals = evals[near_zero_mask]
    near_zero_indices = np.where(near_zero_mask)[0]

    print(f'  lam^2=1000: {len(near_zero_evals)} near-zero eigenvalues')
    print(f'  {"rank":>4} {"eigenvalue":>18} {"log10|eig|":>12}')
    print('  ' + '-' * 36)
    for i, e in enumerate(sorted(near_zero_evals)):
        log_e = np.log10(abs(e)) if abs(e) > 0 else -16
        print(f'  {i+1:>4d} {e:>+18.10e} {log_e:>12.2f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: The proportionality eig_k ~ (1-t) — precision test
    # ==================================================================
    print(f'\n  === STEP 3: LINEARITY TEST eig_k(t) = c_k * (1-t) ===\n')

    # For each near-zero eigenvalue, compute the slope c_k = eig_k(t) / (1-t)
    # at several t values and check consistency
    print(f'  Testing if eig_k(0.999) / 0.001 = eig_k(0.99) / 0.01 = eig_k(0.9) / 0.1')
    print()

    t_tests = [0.9, 0.95, 0.99, 0.999]
    slopes = {}
    for t in t_tests:
        Mt = Lp + t * np.diag(Dd)
        et = np.linalg.eigvalsh(Mt)
        # Match by eigenvalue ordering (sorted)
        slopes[t] = et

    # Focus on the eigenvalues that are near-zero at t=1
    # These are the ones just above the bulk negative
    # At t=0.9, they should be the small positive ones
    print(f'  Eigenvalue index (from top): slope = eig(t)/(1-t)')
    print(f'  {"idx":>4} {"t=0.9":>12} {"t=0.95":>12} {"t=0.99":>12} {"t=0.999":>12} {"consistent?":>12}')
    print('  ' + '-' * 66)

    for j_from_top in range(1, 20):
        idx = dim - 1 - j_from_top
        vals = []
        for t in t_tests:
            e = slopes[t][idx]
            s = e / (1 - t) if abs(1 - t) > 1e-10 else 0
            vals.append(s)

        # Check consistency
        if all(abs(v) > 1e-6 for v in vals):
            spread = (max(vals) - min(vals)) / abs(np.mean(vals)) if abs(np.mean(vals)) > 1e-10 else 0
            consistent = spread < 0.1
        else:
            spread = 0
            consistent = False

        val_str = '  '.join(f'{v:>+12.4f}' for v in vals)
        print(f'  {j_from_top:>4d} {val_str} {"YES" if consistent else "no":>12}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: M restricted to near-zero subspace
    # ==================================================================
    print(f'\n  === STEP 4: M ON THE NEAR-ZERO SUBSPACE ===\n')

    # Project M onto the subspace spanned by near-zero eigenvectors
    V_nz = evecs[:, near_zero_indices]
    M_nz = V_nz.T @ M @ V_nz
    Lp_nz = V_nz.T @ Lp @ V_nz
    D_nz = V_nz.T @ np.diag(Dd) @ V_nz

    print(f'  M restricted to {len(near_zero_indices)}-dim near-zero subspace:')
    print(f'    ||M_nz|| = {np.linalg.norm(M_nz):.6e}')
    print(f'    ||L_nz|| = {np.linalg.norm(Lp_nz):.6e}')
    print(f'    ||D_nz|| = {np.linalg.norm(D_nz):.6e}')
    print(f'    tr(M_nz) = {np.trace(M_nz):.6e}')
    print(f'    tr(L_nz) = {np.trace(Lp_nz):.6e}')
    print(f'    tr(D_nz) = {np.trace(D_nz):.6e}')
    print(f'    M_nz ~= L_nz + D_nz residual: {np.linalg.norm(M_nz - Lp_nz - D_nz):.6e}')

    # Eigenvalues of L_nz and D_nz separately
    evals_Lnz = np.linalg.eigvalsh(Lp_nz)
    evals_Dnz = np.linalg.eigvalsh(D_nz)

    print(f'\n  L_pure on near-zero subspace:')
    print(f'    Eigenvalues: [{evals_Lnz.min():.4e}, {evals_Lnz.max():.4e}]')
    print(f'    #pos: {np.sum(evals_Lnz > 1e-10)}, #neg: {np.sum(evals_Lnz < -1e-10)}')

    print(f'  D (diagonal) on near-zero subspace:')
    print(f'    Eigenvalues: [{evals_Dnz.min():.4e}, {evals_Dnz.max():.4e}]')
    print(f'    #pos: {np.sum(evals_Dnz > 1e-10)}, #neg: {np.sum(evals_Dnz < -1e-10)}')

    # KEY: L_nz and D_nz should nearly cancel on this subspace
    print(f'\n  L_nz + D_nz cancellation:')
    print(f'    ||L_nz + D_nz|| / ||L_nz|| = {np.linalg.norm(Lp_nz + D_nz)/np.linalg.norm(Lp_nz):.6e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: The Hellmann-Feynman slopes of near-zero eigenvalues
    # ==================================================================
    print(f'\n  === STEP 5: HF SLOPES OF NEAR-ZERO CLUSTER ===\n')

    # For the near-zero eigenvectors, v_k^T D v_k gives the slope
    hf_slopes = np.array([np.sum(Dd * evecs[:, k]**2) for k in near_zero_indices])

    print(f'  Hellmann-Feynman slopes for {len(near_zero_indices)} near-zero eigenvectors:')
    print(f'  All slopes should be negative (eigenvalues decrease with t)')
    print()
    print(f'    Min slope: {hf_slopes.min():.6e}')
    print(f'    Max slope: {hf_slopes.max():.6e}')
    print(f'    Mean slope: {hf_slopes.mean():.6e}')
    print(f'    All negative: {np.all(hf_slopes < 0)}')
    print(f'    Spread (max/min): {hf_slopes.max()/hf_slopes.min():.4f}')

    # The slopes determine when each crosses zero:
    # eig_k(t) ~= eig_k(1) + (t-1)*slope_k
    # Crosses zero at t_cross = 1 - eig_k(1)/slope_k
    # Since eig_k(1) ~= 0 and slope_k < 0, t_cross ~= 1

    print(f'\n  Predicted crossing points:')
    t_predicted = 1 - near_zero_evals / hf_slopes
    print(f'    Range: [{t_predicted.min():.10f}, {t_predicted.max():.10f}]')
    print(f'    Mean: {t_predicted.mean():.10f}')
    print(f'    Std: {t_predicted.std():.10e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: Lambda scaling of the near-zero cluster
    # ==================================================================
    print(f'\n  === STEP 6: LAMBDA SCALING ===\n')

    print(f'  {"lam^2":>8} {"#near-zero":>10} {"max|eig_nz|":>14} {"mean|eig_nz|":>14} {"log10 max":>10}')
    print('  ' + '-' * 60)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        try:
            Lp, Dd, M, QW, N, L, dim, ns = extract_decomposition(lam_sq)
            evals = np.linalg.eigvalsh(M)

            nz = evals[np.abs(evals) < 0.01]
            n_nz = len(nz)
            max_nz = np.max(np.abs(nz)) if n_nz > 0 else 0
            mean_nz = np.mean(np.abs(nz)) if n_nz > 0 else 0
            log_max = np.log10(max_nz) if max_nz > 0 else -16

            print(f'  {lam_sq:>8d} {n_nz:>10d} {max_nz:>14.6e} {mean_nz:>14.6e} {log_max:>10.2f}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 7: Effective rank of M
    # ==================================================================
    print(f'\n  === STEP 7: EFFECTIVE RANK ANALYSIS ===\n')

    for lam_sq in [200, 1000, 5000]:
        Lp, Dd, M, QW, N, L, dim, ns = extract_decomposition(lam_sq)
        evals = np.linalg.eigvalsh(M)

        # Singular values of M
        svals = np.abs(evals)
        svals_sorted = np.sort(svals)[::-1]

        # Effective rank at various thresholds
        for thresh_exp in [-2, -4, -6, -8]:
            thresh = 10**thresh_exp
            eff_rank = np.sum(svals > thresh * svals[0] if svals[0] > 0 else svals > thresh)
            print(f'  lam^2={lam_sq}: effective rank (thresh=10^{thresh_exp}) = {eff_rank}/{dim}')
        print()
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 73c VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
