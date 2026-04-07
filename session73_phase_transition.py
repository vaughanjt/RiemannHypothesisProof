"""
SESSION 73 -- THE PHASE TRANSITION AT t=1.0

Session 71c found: M(t) = L_pure + t*D has 69 positive eigenvalues at t=0,
68 at t=0.8, and 1 at t=1.0. 67 eigenvalues get killed in the last 20%.

WHY does the transition happen at exactly t=1.0?
Is t=1.0 special, or is it a smooth crossover that happens to be sharp?

Plan:
  1. Fine-grained eigenvalue tracking: t from 0.8 to 1.2 in tiny steps
  2. Plot the eigenvalue flow: each eigenvalue as a function of t
  3. Find the CRITICAL t where each eigenvalue crosses zero
  4. Is there a clustering of critical t values near t=1.0?
  5. What happens at t > 1.0? (over-shooting the diagonal)
  6. Does the transition point depend on lambda?
  7. Is t=1.0 a phase transition (discontinuity) or a crossover (smooth)?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def extract_cauchy_and_diagonal(lam_sq):
    """Extract L_pure (Cauchy) and D (diagonal perturbation) from M."""
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

    # Build pure Loewner matrix
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        L_pure = (B_n[None, :] - B_n[:, None]) / nm

    # Fill diagonal with centered finite difference
    for i in range(dim):
        if 0 < i < dim - 1:
            L_pure[i, i] = (B_n[i + 1] - B_n[i - 1]) / 2
        elif i == 0:
            L_pure[i, i] = B_n[1] - B_n[0]
        else:
            L_pure[i, i] = B_n[-1] - B_n[-2]

    L_pure = (L_pure + L_pure.T) / 2

    # Get actual M
    _, M, _ = build_all_fast(lam_sq, N)

    # Diagonal perturbation
    D_diag = np.diag(M) - np.diag(L_pure)

    return L_pure, D_diag, M, N, L, dim


def track_eigenvalues(L_pure, D_diag, t_values):
    """Track all eigenvalues of M(t) = L_pure + t*diag(D) as t varies."""
    dim = L_pure.shape[0]
    all_evals = np.zeros((len(t_values), dim))

    for i, t in enumerate(t_values):
        M_t = L_pure + t * np.diag(D_diag)
        evals = np.linalg.eigvalsh(M_t)
        all_evals[i, :] = evals

    return all_evals


def find_zero_crossings(t_values, evals_trajectory):
    """Find the t value where each eigenvalue crosses zero."""
    n_eigs = evals_trajectory.shape[1]
    crossings = []

    for j in range(n_eigs):
        traj = evals_trajectory[:, j]
        for i in range(len(t_values) - 1):
            if traj[i] > 0 and traj[i + 1] <= 0:
                # Linear interpolation
                t_cross = t_values[i] + (0 - traj[i]) / (traj[i + 1] - traj[i]) * (t_values[i + 1] - t_values[i])
                crossings.append((t_cross, j, traj[i], traj[i + 1]))
            elif traj[i] <= 0 and traj[i + 1] > 0:
                t_cross = t_values[i] + (0 - traj[i]) / (traj[i + 1] - traj[i]) * (t_values[i + 1] - t_values[i])
                crossings.append((t_cross, j, traj[i], traj[i + 1]))

    crossings.sort()
    return crossings


def run():
    print()
    print('#' * 76)
    print('  SESSION 73 -- THE PHASE TRANSITION AT t=1.0')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Fine-grained eigenvalue tracking at lam^2=1000
    # ==================================================================
    print(f'\n  === STEP 1: EIGENVALUE FLOW (lam^2=1000) ===\n')

    lam_sq = 1000
    L_pure, D_diag, M, N, L, dim = extract_cauchy_and_diagonal(lam_sq)

    print(f'  dim = {dim}, L = {L:.4f}')
    print(f'  L_pure: {np.sum(np.linalg.eigvalsh(L_pure) > 1e-10)} positive eigenvalues')

    # Coarse sweep
    t_coarse = np.linspace(0, 1.5, 151)
    evals_coarse = track_eigenvalues(L_pure, D_diag, t_coarse)

    print(f'\n  Coarse sweep t=0 to 1.5:')
    print(f'  {"t":>6} {"#pos":>5} {"eig_max":>12} {"eig_2nd":>12} {"eig_3rd":>12}')
    print('  ' + '-' * 50)
    for i in range(0, len(t_coarse), 10):
        t = t_coarse[i]
        evals = evals_coarse[i]
        n_pos = np.sum(evals > 1e-10)
        print(f'  {t:>6.2f} {n_pos:>5d} {evals[-1]:>+12.4f} {evals[-2]:>+12.6e} {evals[-3]:>+12.6e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Find ALL zero-crossing t values
    # ==================================================================
    print(f'\n  === STEP 2: ZERO-CROSSING t VALUES ===\n')

    # Fine sweep around the transition
    t_fine = np.linspace(0.5, 1.2, 7001)
    evals_fine = track_eigenvalues(L_pure, D_diag, t_fine)

    crossings = find_zero_crossings(t_fine, evals_fine)

    print(f'  Found {len(crossings)} zero crossings:')
    print(f'  {"t_cross":>10} {"eig_idx":>8} {"before":>14} {"after":>14}')
    print('  ' + '-' * 50)
    for t_c, idx, before, after in crossings[:40]:
        direction = "pos->neg" if before > 0 else "neg->pos"
        print(f'  {t_c:>10.6f} {idx:>8d} {before:>+14.6e} {after:>+14.6e}  {direction}')
    if len(crossings) > 40:
        print(f'  ... ({len(crossings) - 40} more)')
    sys.stdout.flush()

    # Distribution of crossing points
    t_crosses = np.array([c[0] for c in crossings if c[2] > 0])  # pos->neg only
    if len(t_crosses) > 0:
        print(f'\n  Distribution of positive->negative crossings:')
        print(f'  Count: {len(t_crosses)}')
        print(f'  Min:   {t_crosses.min():.6f}')
        print(f'  Max:   {t_crosses.max():.6f}')
        print(f'  Mean:  {t_crosses.mean():.6f}')
        print(f'  Std:   {t_crosses.std():.6f}')

        # Histogram in bins
        bins = np.linspace(0.5, 1.2, 15)
        counts, _ = np.histogram(t_crosses, bins)
        print(f'\n  Histogram of crossing points:')
        for i in range(len(counts)):
            bar = '#' * counts[i]
            print(f'  [{bins[i]:.2f}, {bins[i+1]:.2f}): {counts[i]:>3d} {bar}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: What happens at t > 1.0?
    # ==================================================================
    print(f'\n  === STEP 3: BEYOND t=1.0 ===\n')

    t_beyond = np.linspace(1.0, 2.0, 101)
    evals_beyond = track_eigenvalues(L_pure, D_diag, t_beyond)

    print(f'  {"t":>6} {"#pos":>5} {"eig_max":>12} {"eig_2nd":>12}')
    print('  ' + '-' * 38)
    for i in range(0, len(t_beyond), 10):
        t = t_beyond[i]
        evals = evals_beyond[i]
        n_pos = np.sum(evals > 1e-10)
        print(f'  {t:>6.2f} {n_pos:>5d} {evals[-1]:>+12.4f} {evals[-2]:>+12.6e}')

    # At what t does the LAST positive eigenvalue die?
    for i in range(len(t_beyond)):
        evals = evals_beyond[i]
        if np.sum(evals > 1e-10) == 0:
            print(f'\n  ALL eigenvalues negative at t = {t_beyond[i]:.4f}')
            print(f'  The ONE surviving positive eigenvalue dies between '
                  f't={t_beyond[i-1]:.4f} and t={t_beyond[i]:.4f}')
            break
    else:
        print(f'\n  Still has positive eigenvalue at t = {t_beyond[-1]:.2f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: Lambda dependence of the transition
    # ==================================================================
    print(f'\n  === STEP 4: LAMBDA DEPENDENCE ===\n')

    print(f'  How does the transition sharpness depend on lambda?')
    print(f'  {"lam^2":>8} {"dim":>5} {"#pos(t=0)":>10} {"t(#pos=2)":>10} {"t(#pos=1)":>10} {"t(#pos=0)":>10}')
    print('  ' + '-' * 58)

    for lam_sq_test in [50, 100, 200, 500, 1000, 2000, 5000]:
        try:
            Lp, Dd, Mt, Nt, Lt, dt = extract_cauchy_and_diagonal(lam_sq_test)
            t_scan = np.linspace(0, 2.5, 2501)

            n_pos_at_0 = np.sum(np.linalg.eigvalsh(Lp) > 1e-10)
            t_2 = t_1 = t_0 = None

            for t in t_scan:
                M_t = Lp + t * np.diag(Dd)
                evals = np.linalg.eigvalsh(M_t)
                n_pos = np.sum(evals > 1e-10)
                if n_pos <= 2 and t_2 is None:
                    t_2 = t
                if n_pos <= 1 and t_1 is None:
                    t_1 = t
                if n_pos == 0 and t_0 is None:
                    t_0 = t

            t_2s = f'{t_2:.4f}' if t_2 is not None else '>2.5'
            t_1s = f'{t_1:.4f}' if t_1 is not None else '>2.5'
            t_0s = f'{t_0:.4f}' if t_0 is not None else '>2.5'

            print(f'  {lam_sq_test:>8d} {dt:>5d} {n_pos_at_0:>10d} {t_2s:>10} {t_1s:>10} {t_0s:>10}')
        except Exception as e:
            print(f'  {lam_sq_test:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: The critical eigenvalue anatomy
    # ==================================================================
    print(f'\n  === STEP 5: ANATOMY OF THE LAST EIGENVALUES TO DIE ===\n')

    lam_sq = 1000
    Lp, Dd, Mt, Nt, Lt, dt = extract_cauchy_and_diagonal(lam_sq)

    # At t just below the transition (say t=0.95), which eigenvalues are still positive?
    for t in [0.90, 0.95, 0.98, 0.99, 0.995, 1.0, 1.001, 1.01]:
        M_t = Lp + t * np.diag(Dd)
        evals = np.linalg.eigvalsh(M_t)
        n_pos = np.sum(evals > 1e-10)

        # Show the top few eigenvalues
        top = evals[-5:][::-1]
        top_str = ', '.join(f'{e:+.4e}' for e in top)
        print(f'  t={t:.3f}: #pos={n_pos:>3d}, top 5: [{top_str}]')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: Eigenvector tracking through the transition
    # ==================================================================
    print(f'\n  === STEP 6: EIGENVECTOR EVOLUTION ===\n')

    # Track how the 2nd eigenvector (the last to die) evolves
    t_track = [0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05]
    print(f'  2nd eigenvector alignment with W02 range and v_+ of full M:')

    # Get W02 for reference
    W02, _, _ = build_all_fast(lam_sq, Nt)
    ew, vw = np.linalg.eigh(W02)
    # W02 range: top 2 eigenvectors
    w02_range = vw[:, -2:]

    # Full M positive eigenvector
    em, vm = np.linalg.eigh(Mt)
    v_plus_full = vm[:, -1]

    print(f'  {"t":>6} {"eig_2":>12} {"align w/ W02":>14} {"align w/ v+":>14}')
    print('  ' + '-' * 50)

    for t in t_track:
        M_t = Lp + t * np.diag(Dd)
        evals_t, evecs_t = np.linalg.eigh(M_t)

        eig_2 = evals_t[-2]
        v2 = evecs_t[:, -2]

        # Alignment with W02 range
        proj_w02 = w02_range @ (w02_range.T @ v2)
        align_w02 = np.linalg.norm(proj_w02)

        # Alignment with full M's v_+
        align_vp = abs(np.dot(v2, v_plus_full))

        print(f'  {t:>6.3f} {eig_2:>+12.4e} {align_w02:>14.6f} {align_vp:>14.6f}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 73 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
