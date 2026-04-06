"""
SESSION 63c -- THE 7 NEGATIVE EIGENVALUES INVARIANT

L_prime (prime off-diagonal of M_odd) has EXACTLY 7 negative eigenvalues
at every tested lambda. This is a structural invariant.

Questions:
  1. Which primes create the 7 negative eigenvalues?
  2. Is it the small primes? The prime powers?
  3. Does the number change if we vary the prime set?
  4. What is the algebraic mechanism?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import _compute_alpha, _compute_wr_diag


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def build_prime_offdiag(L, N, prime_powers):
    """Build just M_prime off-diagonal from a list of (weight, y) pairs."""
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    M = np.zeros((dim, dim))
    nm_diff = ns[:, None] - ns[None, :]
    for w, y in prime_powers:
        if y >= L:
            continue
        sin_arr = np.sin(2 * np.pi * ns * y / L)
        cos_arr = np.cos(2 * np.pi * ns * y / L)
        # Full M_prime (diag + off-diag)
        diag = 2 * (L - y) / L * cos_arr
        np.fill_diagonal(M, M.diagonal() + w * diag)
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off, 0.0)
        M += w * off
    return M


def get_prime_power_list(lam_sq):
    """Get list of (weight, y) for all prime powers <= lam_sq."""
    primes = sieve_primes(int(lam_sq))
    result = []
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            result.append((logp * pk ** (-0.5), np.log(pk), p, pk))
            pk *= int(p)
    return result


def run():
    print()
    print('#' * 76)
    print('  SESSION 63c -- THE 7 NEGATIVE EIGENVALUES')
    print('#' * 76)

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))

    # =================================================================
    # TEST 1: Add primes one at a time, track neg eig count of L_prime
    # =================================================================
    print('\n  === TEST 1: INCREMENTAL PRIME ADDITION ===')
    print('  Add prime powers one at a time. Track signature of L_prime_odd.\n')

    pps = get_prime_power_list(lam_sq)
    # Sort by y (log of prime power)
    pps.sort(key=lambda x: x[1])

    print(f'  Total prime powers: {len(pps)}')
    print(f'  {"#pp added":>10} {"prime power":>12} {"p":>4} {"neg eigs":>9} '
          f'{"pos eigs":>9} {"max_eig":>12}')
    print('  ' + '-' * 62)

    accumulated = []
    checkpoints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, len(pps)]

    for i, (w, y, p, pk) in enumerate(pps):
        accumulated.append((w, y))
        if (i + 1) in checkpoints:
            M_p = build_prime_offdiag(L, N, accumulated)
            Mo_p = odd_block(M_p, N)
            # Extract just off-diagonal
            L_off = Mo_p.copy()
            np.fill_diagonal(L_off, 0.0)
            eL = np.linalg.eigvalsh(L_off)
            n_neg = np.sum(eL < -1e-10)
            n_pos = np.sum(eL > 1e-10)
            print(f'  {i+1:>10d} {pk:>12d} {p:>4d} {n_neg:>9d} '
                  f'{n_pos:>9d} {eL[-1]:>+12.4f}')
    sys.stdout.flush()

    # =================================================================
    # TEST 2: Only powers of p=2
    # =================================================================
    print('\n  === TEST 2: SINGLE PRIME CONTRIBUTIONS ===')
    print('  Use only powers of one prime at a time.\n')

    print(f'  {"prime p":>8} {"#powers":>8} {"neg eigs":>9} {"pos eigs":>9}')
    print('  ' + '-' * 38)

    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        pk_list = []
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            pk_list.append((logp * pk ** (-0.5), np.log(pk)))
            pk *= int(p)
        if not pk_list:
            continue
        M_single = build_prime_offdiag(L, N, pk_list)
        Mo_single = odd_block(M_single, N)
        L_off = Mo_single.copy()
        np.fill_diagonal(L_off, 0.0)
        eL = np.linalg.eigvalsh(L_off)
        n_neg = np.sum(eL < -1e-10)
        n_pos = np.sum(eL > 1e-10)
        print(f'  {p:>8d} {len(pk_list):>8d} {n_neg:>9d} {n_pos:>9d}')
    sys.stdout.flush()

    # =================================================================
    # TEST 3: Primes only (no higher powers)
    # =================================================================
    print('\n  === TEST 3: PRIMES ONLY (k=1) vs ALL POWERS ===')
    print('  Compare signature of L_prime with/without higher powers.\n')

    primes = list(sieve_primes(int(lam_sq)))

    # k=1 only
    pk1_only = [(np.log(p) * p**(-0.5), np.log(p)) for p in primes
                if np.log(p) < L]
    M_k1 = build_prime_offdiag(L, N, pk1_only)
    Mo_k1 = odd_block(M_k1, N)
    L_k1 = Mo_k1.copy()
    np.fill_diagonal(L_k1, 0.0)
    eL_k1 = np.linalg.eigvalsh(L_k1)
    n_neg_k1 = np.sum(eL_k1 < -1e-10)

    # All powers
    all_pps = [(w, y) for w, y, p, pk in pps]
    M_all = build_prime_offdiag(L, N, all_pps)
    Mo_all = odd_block(M_all, N)
    L_all = Mo_all.copy()
    np.fill_diagonal(L_all, 0.0)
    eL_all = np.linalg.eigvalsh(L_all)
    n_neg_all = np.sum(eL_all < -1e-10)

    # k >= 2 only (higher powers)
    pk_higher = [(w, y) for w, y, p, pk in pps if pk != p]
    M_higher = build_prime_offdiag(L, N, pk_higher)
    Mo_higher = odd_block(M_higher, N)
    L_higher = Mo_higher.copy()
    np.fill_diagonal(L_higher, 0.0)
    eL_higher = np.linalg.eigvalsh(L_higher)
    n_neg_higher = np.sum(eL_higher < -1e-10)

    print(f'  k=1 only ({len(pk1_only)} primes): {n_neg_k1} neg eigs of L_prime_odd')
    print(f'  All powers ({len(all_pps)} prime powers): {n_neg_all} neg eigs')
    print(f'  k>=2 only ({len(pk_higher)} higher powers): {n_neg_higher} neg eigs')
    sys.stdout.flush()

    # =================================================================
    # TEST 4: Does the number depend on truncation N?
    # =================================================================
    print('\n  === TEST 4: N-DEPENDENCE ===')
    print('  Fix lam^2=1000, vary N. Does the count of 7 depend on N?\n')

    for N_test in [10, 20, 30, 41, 60, 80, 100]:
        M_p = build_prime_offdiag(L, N_test, all_pps)
        Mo_p = odd_block(M_p, N_test)
        L_off = Mo_p.copy()
        np.fill_diagonal(L_off, 0.0)
        eL = np.linalg.eigvalsh(L_off)
        n_neg = np.sum(eL < -1e-10)
        n_pos = np.sum(eL > 1e-10)
        print(f'  N={N_test:>4d}: {n_neg} neg, {n_pos} pos eigs of L_prime_odd')
    sys.stdout.flush()

    # =================================================================
    # TEST 5: The 7 as function of lam^2 at very small lambda
    # =================================================================
    print('\n  === TEST 5: SMALL LAMBDA SCAN ===')
    print('  Check 7-invariant at very small lam^2 where few primes exist.\n')

    for lam_sq_test in [4, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000]:
        L_test = float(np.log(lam_sq_test))
        N_test = max(10, round(6 * L_test))
        pps_test = get_prime_power_list(lam_sq_test)
        if not pps_test:
            print(f'  lam^2={lam_sq_test:>6d}: no prime powers')
            continue
        pps_wy = [(w, y) for w, y, p, pk in pps_test]
        M_test = build_prime_offdiag(L_test, N_test, pps_wy)
        Mo_test = odd_block(M_test, N_test)
        L_off = Mo_test.copy()
        np.fill_diagonal(L_off, 0.0)
        eL = np.linalg.eigvalsh(L_off)
        n_neg = np.sum(eL < -1e-10)
        n_pos = np.sum(eL > 1e-10)
        n_pp = len(pps_test)
        primes_in = set(p for _, _, p, pk in pps_test)
        print(f'  lam^2={lam_sq_test:>6d} (L={L_test:.2f}, {n_pp} pp, '
              f'{len(primes_in)} primes): {n_neg} neg, {n_pos} pos')
    sys.stdout.flush()

    # =================================================================
    # TEST 6: Probe the FULL M_odd (not just L_prime) with
    #         archimedean + first K primes to find transition point
    # =================================================================
    print('\n  === TEST 6: WHEN DOES M_ODD_FULL BECOME NEG DEF? ===')
    print('  Add real primes (with powers) one at a time to archimedean base.')
    print('  Track max eigenvalue of full M_odd.\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # Build archimedean base
    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)
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
    arch_max = np.linalg.eigvalsh(Mo_arch)[-1]
    print(f'  Archimedean only: max_eig = {arch_max:+.6f}')

    # Get ALL prime powers sorted by y
    pps = get_prime_power_list(lam_sq)
    pps.sort(key=lambda x: x[1])

    # Add cumulatively
    M_running = M_arch.copy()
    print(f'\n  {"#pp":>5} {"pk":>8} {"p":>4} {"max_eig":>14} {"neg def":>8}')
    print('  ' + '-' * 45)

    for i, (w, y, p, pk) in enumerate(pps):
        sin_arr = np.sin(2 * np.pi * ns * y / L)
        cos_arr = np.cos(2 * np.pi * ns * y / L)
        diag = 2 * (L - y) / L * cos_arr
        np.fill_diagonal(M_running, M_running.diagonal() + w * diag)
        nm_diff = ns[:, None] - ns[None, :]
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off, 0.0)
        M_running += w * off

        if (i + 1) in [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 180,
                        190, 195, 198, 199, 200, len(pps)] or \
           (i > 190 and i < len(pps)):
            M_sym = (M_running + M_running.T) / 2
            Mo_run = odd_block(M_sym, N)
            max_e = np.linalg.eigvalsh(Mo_run)[-1]
            nd = max_e < 0
            print(f'  {i+1:>5d} {pk:>8d} {p:>4d} {max_e:>+14.6e} '
                  f'{"YES" if nd else "no":>8}')

    sys.stdout.flush()

    # =================================================================
    print()
    print('=' * 76)
    print('  SESSION 63c RESULTS')
    print('=' * 76)


if __name__ == '__main__':
    run()
