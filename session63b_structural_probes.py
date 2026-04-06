"""
SESSION 63b -- STRUCTURAL PROBES

Part 1 showed M_odd < 0 is 100% prime-specific (Cramer model gives 0%).
Part 7 showed the coupling alignment is in a vanishingly narrow cone.

Key question: WHAT about real primes makes M_odd < 0?

Probes:
  A. The constant 7 negative eigenvalues of L (off-diagonal Loewner part)
  B. Conditional primes: plant first K primes at real positions, randomize rest
  C. The archimedean part alone: is D + L_alpha neg def? (without prime off-diagonal)
  D. Decompose L into prime and archimedean parts separately
  E. The eigenvalue gap structure of B_rest = M_odd[1:, 1:]
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _build_M_prime, _compute_alpha, _compute_wr_diag
)


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def build_M_with_custom_primes(L, N, lam_sq, prime_positions):
    """Build full M using custom prime positions (INCLUDING all prime powers)."""
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # Archimedean parts (same regardless of primes)
    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    M = np.zeros((dim, dim))

    # Prime part from custom positions (including all prime powers p^k <= lam_sq)
    for p in prime_positions:
        if p < 2:
            continue
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            if y >= L:
                break
            sin_arr = np.sin(2 * np.pi * ns * y / L)
            cos_arr = np.cos(2 * np.pi * ns * y / L)
            diag = 2 * (L - y) / L * cos_arr
            np.fill_diagonal(M, M.diagonal() + w * diag)
            nm_diff = ns[:, None] - ns[None, :]
            sin_diff = sin_arr[None, :] - sin_arr[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                off = sin_diff / (np.pi * nm_diff)
            np.fill_diagonal(off, 0.0)
            M += w * off
            pk *= int(p)

    # Add archimedean
    for n in range(-N, N + 1):
        M[N + n, N + n] += wr[abs(n)]
    a_m = alpha[None, :]
    a_n_arr = alpha[:, None]
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        offdiag = (a_m - a_n_arr) / nm
    np.fill_diagonal(offdiag, 0.0)
    M += offdiag
    M = (M + M.T) / 2
    return M


def run():
    print()
    print('#' * 76)
    print('  SESSION 63b -- STRUCTURAL PROBES')
    print('#' * 76)

    # ==================================================================
    # PROBE A: WHY 7 NEGATIVE EIGENVALUES OF L?
    # ==================================================================
    print('\n  === PROBE A: THE CONSTANT 7 NEG EIGS OF L ===')
    print('  L is the off-diagonal Loewner part of M_odd.')
    print('  It consistently has exactly 7 negative eigenvalues.')
    print('  Hypothesis: related to structure of B_n sequence.\n')

    for lam_sq in [50, 100, 200, 500, 1000, 5000, 20000, 50000]:
        L_val = float(np.log(lam_sq))
        N = max(15, round(6 * L_val))
        _, M_full, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M_full, N)

        # Extract L (off-diagonal part)
        L_mat = Mo.copy()
        np.fill_diagonal(L_mat, 0.0)
        eL = np.linalg.eigvalsh(L_mat)
        n_neg = np.sum(eL < -1e-10)
        n_pos = np.sum(eL > 1e-10)

        # Also decompose L into prime and alpha parts
        ns = np.arange(-N, N + 1, dtype=float)
        alpha = _compute_alpha(L_val, N)

        # Alpha-only off-diagonal (archimedean Loewner)
        M_alpha = np.zeros((2*N+1, 2*N+1))
        a_m_arr = alpha[None, :]
        a_n_arr2 = alpha[:, None]
        nm = ns[:, None] - ns[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            M_alpha = (a_m_arr - a_n_arr2) / nm
        np.fill_diagonal(M_alpha, 0.0)
        M_alpha = (M_alpha + M_alpha.T) / 2
        L_alpha = odd_block(M_alpha, N)
        np.fill_diagonal(L_alpha, 0.0)  # remove any diagonal from projection
        eL_alpha = np.linalg.eigvalsh(L_alpha)
        n_neg_alpha = np.sum(eL_alpha < -1e-10)

        # Prime-only off-diagonal
        M_prime = _build_M_prime(L_val, N, lam_sq)
        M_prime = (M_prime + M_prime.T) / 2
        L_prime = odd_block(M_prime, N)
        # Remove diagonal to get just off-diagonal
        L_prime_offdiag = L_prime.copy()
        np.fill_diagonal(L_prime_offdiag, 0.0)
        eL_prime = np.linalg.eigvalsh(L_prime_offdiag)
        n_neg_prime = np.sum(eL_prime < -1e-10)

        print(f'  lam^2={lam_sq:>6d} N={N:>3d}: '
              f'L({n_neg},{n_pos}), '
              f'L_alpha({n_neg_alpha} neg), '
              f'L_prime({n_neg_prime} neg)')
    sys.stdout.flush()

    # ==================================================================
    # PROBE B: CONDITIONAL PRIMES -- PLANT FIRST K AT REAL POSITIONS
    # ==================================================================
    print('\n  === PROBE B: CONDITIONAL PRIMES ===')
    print('  Fix first K real primes. Randomize the rest (Cramer).')
    print('  At what K does M_odd become (usually) neg def?\n')

    np.random.seed(123)
    lam_sq = 1000
    L_val = float(np.log(lam_sq))
    N = max(15, round(6 * L_val))
    real_primes = list(sieve_primes(int(lam_sq)))
    n_trials = 30

    for K in [0, 5, 10, 20, 50, 100, len(real_primes)]:
        K_actual = min(K, len(real_primes))
        planted = real_primes[:K_actual]
        n_neg_def = 0
        max_eigs = []

        for trial in range(n_trials):
            # Generate Cramer primes for positions > planted[-1]
            if K_actual < len(real_primes):
                start = planted[-1] + 1 if planted else 2
                cramer_rest = []
                for n in range(start, int(lam_sq) + 1):
                    if np.random.random() < 1.0 / max(np.log(n), 1.01):
                        cramer_rest.append(n)
                all_primes = planted + cramer_rest
            else:
                all_primes = real_primes

            M_cust = build_M_with_custom_primes(L_val, N, lam_sq, all_primes)
            Mo_cust = odd_block(M_cust, N)
            eigs = np.linalg.eigvalsh(Mo_cust)
            max_eigs.append(eigs[-1])
            if eigs[-1] < 0:
                n_neg_def += 1

        max_eigs = np.array(max_eigs)
        pct = 100 * n_neg_def / n_trials
        print(f'  K={K_actual:>4d} ({K_actual}/{len(real_primes)} primes planted): '
              f'neg_def={n_neg_def}/{n_trials} ({pct:.0f}%), '
              f'max_eig: mean={max_eigs.mean():+.4f}, '
              f'std={max_eigs.std():.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PROBE C: ARCHIMEDEAN-ONLY M_ODD
    # ==================================================================
    print('\n  === PROBE C: ARCHIMEDEAN-ONLY M_ODD ===')
    print('  M_arch = M_diag + M_alpha (no primes).')
    print('  Is M_arch_odd neg def? This tests if primes help or hurt.\n')

    for lam_sq in [200, 1000, 5000, 20000]:
        L_val = float(np.log(lam_sq))
        N = max(15, round(6 * L_val))
        dim = 2 * N + 1
        ns = np.arange(-N, N + 1, dtype=float)

        wr = _compute_wr_diag(L_val, N)
        alpha = _compute_alpha(L_val, N)

        M_arch = np.zeros((dim, dim))
        for n in range(-N, N + 1):
            M_arch[N + n, N + n] = wr[abs(n)]
        a_m = alpha[None, :]
        a_n_arr = alpha[:, None]
        nm = ns[:, None] - ns[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            offdiag = (a_m - a_n_arr) / nm
        np.fill_diagonal(offdiag, 0.0)
        M_arch += offdiag
        M_arch = (M_arch + M_arch.T) / 2

        Mo_arch = odd_block(M_arch, N)
        eigs_arch = np.linalg.eigvalsh(Mo_arch)
        n_pos = np.sum(eigs_arch > 1e-10)

        # For comparison: full M_odd
        _, M_full, _ = build_all_fast(lam_sq, N)
        Mo_full = odd_block(M_full, N)
        eigs_full = np.linalg.eigvalsh(Mo_full)

        print(f'  lam^2={lam_sq:>6d}:')
        print(f'    M_arch_odd: max_eig={eigs_arch[-1]:+.6f}, '
              f'{n_pos} pos eigs, trace={np.trace(Mo_arch):+.4f}')
        print(f'    M_full_odd: max_eig={eigs_full[-1]:+.2e}, '
              f'trace={np.trace(Mo_full):+.4f}')
        print(f'    Diff: max_eig shift = {eigs_full[-1] - eigs_arch[-1]:+.6f}')
    sys.stdout.flush()

    # ==================================================================
    # PROBE D: THE EIGENVALUE GAP STRUCTURE OF B_rest
    # ==================================================================
    print('\n  === PROBE D: B_rest EIGENVALUE STRUCTURE ===')
    print('  B_rest = M_odd[1:, 1:]. Its eigenvalue nearest 0 controls')
    print('  the Schur complement. What determines this eigenvalue?\n')

    for lam_sq in [200, 1000, 5000, 20000]:
        L_val = float(np.log(lam_sq))
        N = max(15, round(6 * L_val))
        _, M_full, _ = build_all_fast(lam_sq, N)
        Mo = odd_block(M_full, N)

        B_rest = Mo[1:, 1:]
        eB = np.linalg.eigvalsh(B_rest)

        # Eigenvalue nearest to 0
        nearest_0 = eB[-1]
        # Gap between nearest-0 and next
        gap_to_next = eB[-1] - eB[-2] if len(eB) > 1 else float('inf')
        # Most negative
        most_neg = eB[0]

        print(f'  lam^2={lam_sq:>6d} (B_rest is {N-1}x{N-1}):')
        print(f'    Nearest to 0: {nearest_0:+.8e}')
        print(f'    Gap to next:  {gap_to_next:+.6f}')
        print(f'    2nd nearest:  {eB[-2]:+.6f}')
        print(f'    3rd nearest:  {eB[-3]:+.6f}')
        print(f'    Most negative: {most_neg:+.4f}')
        print(f'    Trace: {np.trace(B_rest):+.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PROBE E: THE DETERMINANT SIGN PATTERN
    # ==================================================================
    print('\n  === PROBE E: LEADING MINOR DETERMINANT SIGNS ===')
    print('  M_odd < 0 iff (-1)^k * det(M_odd[0:k, 0:k]) > 0 for all k.')
    print('  Track the sign pattern and where it gets tightest.\n')

    lam_sq = 1000
    L_val = float(np.log(lam_sq))
    N = max(15, round(6 * L_val))
    _, M_full, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M_full, N)

    print(f'  At lam^2={lam_sq} (N={N}):')
    print(f'  {"k":>4} {"det(Mo[0:k])":>18} {"(-1)^k*det":>18} {"sign ok":>8}')
    print('  ' + '-' * 52)

    for k in range(1, min(20, N + 1)):
        submat = Mo[:k, :k]
        # Use log-determinant for numerical stability
        sign, logdet = np.linalg.slogdet(submat)
        det_val = sign * np.exp(logdet) if logdet < 300 else float('inf')
        expected_sign = (-1) ** k
        signed = expected_sign * det_val
        ok = signed > 0
        print(f'  {k:>4d} {det_val:>+18.6e} {signed:>+18.6e} '
              f'{"YES" if ok else "**NO**":>8}')
    sys.stdout.flush()

    # ==================================================================
    # PROBE F: WHAT IF WE ADD ONE FAKE ZERO?
    # ==================================================================
    print('\n  === PROBE F: SENSITIVITY TO PRIME PERTURBATION ===')
    print('  Remove one real prime and see how max_eig of M_odd changes.')
    print('  This shows which primes are load-bearing for negativity.\n')

    lam_sq = 1000
    L_val = float(np.log(lam_sq))
    N = max(15, round(6 * L_val))
    real_primes = list(sieve_primes(int(lam_sq)))

    # Baseline
    M_base = build_M_with_custom_primes(L_val, N, lam_sq, real_primes)
    Mo_base = odd_block(M_base, N)
    base_max = np.linalg.eigvalsh(Mo_base)[-1]
    print(f'  Baseline (all {len(real_primes)} primes): max_eig = {base_max:+.6e}\n')

    print(f'  {"removed p":>10} {"max_eig":>14} {"shift":>14} {"still neg":>10}')
    print('  ' + '-' * 52)

    for p in real_primes[:25]:  # test removing each of first 25 primes
        reduced = [q for q in real_primes if q != p]
        M_red = build_M_with_custom_primes(L_val, N, lam_sq, reduced)
        Mo_red = odd_block(M_red, N)
        red_max = np.linalg.eigvalsh(Mo_red)[-1]
        shift = red_max - base_max
        print(f'  {p:>10d} {red_max:>+14.6e} {shift:>+14.6e} '
              f'{"YES" if red_max < 0 else "**NO**":>10}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 63b RESULTS')
    print('=' * 76)


if __name__ == '__main__':
    run()
