"""
SESSION 76b -- DEEP SEARCH: WHAT IS THE 15-DIM SIGNAL SPACE?

Session 76 found:
  1. Signal dim = 16-17, CONSTANT across lam^2 from 50 to 50000
  2. Prolate c=0.2 matches signal dim at lam^2=1000 (Shannon=16.4)
  3. Zero directions do NOT efficiently span signal space (~2% each)
  4. Coupling vector is 52% signal, 48% null
  5. On signal space, L and D REINFORCE (cancel ratio 1.98)

This script digs deeper:
  A. Does signal dim stay constant at VERY large lambda?
  B. What if N is much larger than 6*L? Is it an N effect?
  C. Build the EXACT prolate operator that reproduces the split
  D. Spectral gap: how wide is the gap between signal and null eigenvalues?
  E. Signal eigenvector overlap matrix across lambda: do they track?
  F. Is signal dim = rank of the prime contribution M_prime?
  G. Archimedean-only (no primes): what's the signal dim?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def build_M_components(lam_sq, N=None):
    """Build M and its archimedean + prime decomposition."""
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    # Archimedean diagonal
    a_arch = np.array([wr[abs(int(n))] for n in ns])

    # Archimedean off-diagonal (alpha Cauchy)
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_offdiag = (alpha[None, :] - alpha[:, None]) / nm
    np.fill_diagonal(alpha_offdiag, 0)

    M_arch = np.diag(a_arch) + alpha_offdiag
    M_arch = (M_arch + M_arch.T) / 2

    # Prime contribution
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

    # Prime off-diagonal (Cauchy from B_prime)
    with np.errstate(divide='ignore', invalid='ignore'):
        prime_offdiag = (B_prime[None, :] - B_prime[:, None]) / nm
    np.fill_diagonal(prime_offdiag, 0)

    M_prime = np.diag(a_prime) + prime_offdiag
    M_prime = (M_prime + M_prime.T) / 2

    _, M_full, _ = build_all_fast(lam_sq, N)

    return M_full, M_arch, M_prime, N, L, dim, ns


def run():
    print()
    print('#' * 76)
    print('  SESSION 76b -- DEEP SEARCH: WHAT IS THE SIGNAL SPACE?')
    print('#' * 76)

    # ======================================================================
    # TEST A: Signal dim at very large lambda
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST A: SIGNAL DIM AT VERY LARGE LAMBDA')
    print(f'{"="*76}\n')

    print(f'  {"lam^2":>8} {"L":>8} {"N":>4} {"dim":>5} {"#sig(0.01)":>10} '
          f'{"#sig(0.1)":>10} {"gap":>14}')
    print('  ' + '-' * 66)

    for lam_sq in [100, 500, 2000, 10000, 50000, 100000, 200000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        try:
            _, M, _ = build_all_fast(lam_sq, N)
            evals = np.linalg.eigvalsh(M)
            n_sig_01 = np.sum(np.abs(evals) > 0.01)
            n_sig_1 = np.sum(np.abs(evals) > 0.1)

            # Spectral gap: largest "near-zero" vs smallest "signal"
            sig_evals = evals[np.abs(evals) > 0.01]
            nz_evals = evals[np.abs(evals) <= 0.01]
            if len(nz_evals) > 0 and len(sig_evals) > 0:
                gap = min(np.abs(sig_evals)) - max(np.abs(nz_evals))
            else:
                gap = float('nan')

            print(f'  {lam_sq:>8d} {L:>8.3f} {N:>4d} {2*N+1:>5d} {n_sig_01:>10d} '
                  f'{n_sig_1:>10d} {gap:>14.6e}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST B: Vary N at fixed lam_sq -- is signal dim an N effect?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST B: VARY N AT FIXED lam^2=1000')
    print(f'{"="*76}\n')

    lam_sq = 1000
    print(f'  {"N":>4} {"dim":>5} {"#sig(0.01)":>10} {"#sig(0.1)":>10} {"eig_max":>12}')
    print('  ' + '-' * 46)

    for N in [15, 20, 30, 41, 60, 80, 100, 150]:
        try:
            _, M, _ = build_all_fast(lam_sq, N)
            evals = np.linalg.eigvalsh(M)
            n_sig_01 = np.sum(np.abs(evals) > 0.01)
            n_sig_1 = np.sum(np.abs(evals) > 0.1)
            eig_max = evals.max()
            print(f'  {N:>4d} {2*N+1:>5d} {n_sig_01:>10d} {n_sig_1:>10d} {eig_max:>12.4f}')
        except Exception as e:
            print(f'  {N:>4d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST C: Build M from archimedean-only vs full — where does signal come from?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST C: ARCHIMEDEAN vs PRIME SIGNAL CONTRIBUTION')
    print(f'{"="*76}\n')

    for lam_sq in [200, 1000, 5000]:
        M_full, M_arch, M_prime, N, L, dim, ns = build_M_components(lam_sq)

        evals_full = np.linalg.eigvalsh(M_full)
        evals_arch = np.linalg.eigvalsh(M_arch)
        evals_prime = np.linalg.eigvalsh(M_prime)

        n_sig_full = np.sum(np.abs(evals_full) > 0.01)
        n_sig_arch = np.sum(np.abs(evals_arch) > 0.01)
        n_sig_prime = np.sum(np.abs(evals_prime) > 0.01)

        print(f'  lam^2={lam_sq}:')
        print(f'    Full M:     #signal={n_sig_full}, eig range [{evals_full.min():.4f}, {evals_full.max():.4f}]')
        print(f'    Arch only:  #signal={n_sig_arch}, eig range [{evals_arch.min():.4f}, {evals_arch.max():.4f}]')
        print(f'    Prime only: #signal={n_sig_prime}, eig range [{evals_prime.min():.4f}, {evals_prime.max():.4f}]')

        # Rank of M_prime
        sv_prime = np.linalg.svd(M_prime, compute_uv=False)
        rank_01 = np.sum(sv_prime > 0.01 * sv_prime[0])
        rank_001 = np.sum(sv_prime > 0.001 * sv_prime[0])
        print(f'    M_prime rank: {rank_01} (1% thresh), {rank_001} (0.1% thresh)')
        print(f'    M_prime trace: {np.trace(M_prime):.4f}')
        print(f'    M_arch trace:  {np.trace(M_arch):.4f}')
        print()
    sys.stdout.flush()

    # ======================================================================
    # TEST D: Spectral gap analysis
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST D: SPECTRAL GAP -- HOW SHARP IS THE SIGNAL/NULL BOUNDARY?')
    print(f'{"="*76}\n')

    lam_sq = 1000
    _, M, _ = build_all_fast(lam_sq, max(15, round(6 * np.log(lam_sq))))
    evals = np.sort(np.abs(np.linalg.eigvalsh(M)))[::-1]

    print(f'  Eigenvalue magnitudes (sorted, lam^2={lam_sq}):')
    print(f'  {"rank":>4} {"|eigenvalue|":>16} {"log10":>8} {"ratio to next":>14}')
    print('  ' + '-' * 46)

    for i in range(min(25, len(evals))):
        log_e = np.log10(evals[i]) if evals[i] > 0 else -16
        ratio = evals[i] / evals[i+1] if i < len(evals) - 1 and evals[i+1] > 0 else 0
        marker = ' <-- GAP' if ratio > 10 else ''
        print(f'  {i+1:>4d} {evals[i]:>16.8e} {log_e:>8.2f} {ratio:>14.2f}{marker}')

    # Show the rest in blocks
    print(f'\n  Tail summary:')
    for block_start in [25, 40, 60, 80]:
        block_end = min(block_start + 10, len(evals))
        if block_start < len(evals):
            block = evals[block_start:block_end]
            print(f'    ranks {block_start+1}-{block_end}: '
                  f'[{block.min():.4e}, {block.max():.4e}]')
    sys.stdout.flush()

    # ======================================================================
    # TEST E: Signal eigenvector tracking across lambda
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST E: DO SIGNAL EIGENVECTORS TRACK ACROSS LAMBDA?')
    print(f'{"="*76}\n')

    # At two different lambda values with same N, compute overlap of
    # signal eigenvector subspaces
    N_fixed = 41
    ref_lam = 1000
    _, M_ref, _ = build_all_fast(ref_lam, N_fixed)
    evals_ref, evecs_ref = np.linalg.eigh(M_ref)
    sig_mask_ref = np.abs(evals_ref) > 0.01
    V_ref = evecs_ref[:, sig_mask_ref]
    n_sig_ref = V_ref.shape[1]

    print(f'  Reference: lam^2={ref_lam}, N={N_fixed}, #signal={n_sig_ref}')
    print(f'  Subspace overlap = sum of squared principal angles cosines / min(d1, d2)')
    print()
    print(f'  {"lam^2":>8} {"#signal":>8} {"overlap":>10} {"min_angle":>12} {"max_angle":>12}')
    print('  ' + '-' * 54)

    for lam_sq_test in [200, 500, 800, 900, 1100, 1200, 1500, 2000, 5000]:
        try:
            _, M_t, _ = build_all_fast(lam_sq_test, N_fixed)
            evals_t, evecs_t = np.linalg.eigh(M_t)
            sig_mask_t = np.abs(evals_t) > 0.01
            V_t = evecs_t[:, sig_mask_t]
            n_sig_t = V_t.shape[1]

            # Principal angles between subspaces
            sv = np.linalg.svd(V_ref.T @ V_t, compute_uv=False)
            sv = np.clip(sv, 0, 1)
            angles = np.arccos(sv)
            overlap = np.sum(sv**2) / min(n_sig_ref, n_sig_t)

            print(f'  {lam_sq_test:>8d} {n_sig_t:>8d} {overlap:>10.6f} '
                  f'{np.degrees(angles.min()):>12.4f}° {np.degrees(angles.max()):>12.4f}°')
        except Exception as e:
            print(f'  {lam_sq_test:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST F: Is signal space the range of M_prime?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST F: IS SIGNAL SPACE = RANGE OF M_PRIME?')
    print(f'{"="*76}\n')

    lam_sq = 1000
    M_full, M_arch, M_prime, N, L, dim, ns = build_M_components(lam_sq)

    evals_full, evecs_full = np.linalg.eigh(M_full)
    sig_mask = np.abs(evals_full) > 0.01
    V_signal = evecs_full[:, sig_mask]

    # SVD of M_prime
    U, S, Vt = np.linalg.svd(M_prime)
    # Top-k range of M_prime
    for k in [5, 10, 15, 17, 20, 30]:
        V_range_k = U[:, :k]
        # Overlap with signal space
        sv = np.linalg.svd(V_signal.T @ V_range_k, compute_uv=False)
        overlap = np.sum(sv**2) / V_signal.shape[1]
        print(f'  range(M_prime, rank={k:>2d}) overlap with signal: {overlap:.6f}')

    # Also test: is signal space the range of M_arch?
    U_a, S_a, _ = np.linalg.svd(M_arch)
    for k in [5, 10, 15, 17, 20]:
        V_range_k = U_a[:, :k]
        sv = np.linalg.svd(V_signal.T @ V_range_k, compute_uv=False)
        overlap = np.sum(sv**2) / V_signal.shape[1]
        print(f'  range(M_arch, rank={k:>2d}) overlap with signal: {overlap:.6f}')
    sys.stdout.flush()

    # ======================================================================
    # TEST G: Eigenvalue histogram -- continuous spectrum structure
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST G: IS THE SIGNAL DIM RELATED TO A PRIME COUNT?')
    print(f'{"="*76}\n')

    print(f'  {"lam^2":>8} {"#primes":>8} {"#prime_pows":>12} {"#signal":>8} '
          f'{"sig/primes":>10} {"sig/ppows":>10}')
    print('  ' + '-' * 62)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        primes = list(sieve_primes(int(lam_sq)))
        n_primes = len(primes)

        # Count prime powers
        n_ppows = 0
        for p in primes:
            pk = p
            while pk <= lam_sq:
                n_ppows += 1
                pk *= p

        _, M, _ = build_all_fast(lam_sq, N)
        evals = np.linalg.eigvalsh(M)
        n_sig = np.sum(np.abs(evals) > 0.01)

        print(f'  {lam_sq:>8d} {n_primes:>8d} {n_ppows:>12d} {n_sig:>8d} '
              f'{n_sig/n_primes:>10.4f} {n_sig/n_ppows:>10.4f}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 76b VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
