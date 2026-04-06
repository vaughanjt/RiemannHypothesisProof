"""
SESSION 59c -- CONNES' DECOMPOSITION: ARCHIMEDEAN + PRIME PERTURBATION

Connes' Theorem 1.1: D = D_scaling - |D_scaling * xi><delta_N|
  - D_scaling = archimedean part (scaling operator on [1/lam, lam])
  - rank-1 perturbation = where the primes enter

Our analogue:
  M = (M_diag + M_alpha) + M_prime
  M_diag + M_alpha = archimedean (wr_diag + alpha off-diagonal)
  M_prime = prime sum (potentially low effective rank)

Questions:
  1. What is the signature/definiteness of M_diag + M_alpha on odd block?
     If it's already negative definite, M_prime just needs to preserve that.
  2. What is the effective rank of M_prime_odd?
     If low-rank, Weyl perturbation might close the gap.
  3. Does M_diag + M_alpha have Toeplitz/CF structure on the odd block?
  4. Can we bound: max_eig(M_odd) <= max_eig(M_diag+alpha_odd) + ||M_prime_odd||_2?
     If ||M_prime_odd||_2 < |max_eig(M_diag+alpha_odd)|, done.
"""

import sys
import time

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


def build_components_odd(lam_sq):
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1

    Mp = _build_M_prime(L, N, lam_sq)
    Mp = (Mp + Mp.T) / 2
    wr = _compute_wr_diag(L, N)
    Md = np.diag([wr[abs(int(n))] for n in np.arange(-N, N + 1)])
    alpha = _compute_alpha(L, N)
    ns = np.arange(-N, N + 1, dtype=float)
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        Ma = (alpha[None, :] - alpha[:, None]) / nm
    np.fill_diagonal(Ma, 0.0)
    Ma = (Ma + Ma.T) / 2

    Mda = Md + Ma  # archimedean
    Mt = Mp + Mda  # total

    return (odd_block(Mp, N), odd_block(Md, N), odd_block(Ma, N),
            odd_block(Mda, N), odd_block(Mt, N), L, N)


def run():
    print()
    print('#' * 76)
    print('  SESSION 59c -- CONNES DECOMPOSITION: ARCHIMEDEAN + PRIME')
    print('#' * 76)

    # == Part 1: Archimedean block eigenvalues ==
    print('\n  === PART 1: M_diag+alpha ON ODD BLOCK ===')
    print(f'  Is the archimedean part already negative definite?')
    print()

    print(f'  {"lam^2":>8} {"N":>4} {"max_eig(Mda)":>14} {"min_eig(Mda)":>14} '
          f'{"neg def?":>10} {"trace(Mda)":>12}')
    print('  ' + '-' * 68)

    for lam_sq in [50, 200, 1000, 5000, 20000, 50000]:
        Mp_o, Md_o, Ma_o, Mda_o, Mt_o, L, N = build_components_odd(lam_sq)
        e = np.linalg.eigvalsh(Mda_o)
        nd = e[-1] < -1e-10
        print(f'  {lam_sq:>8d} {N:>4d} {e[-1]:>+14.6f} {e[0]:>+14.6f} '
              f'{"YES" if nd else "NO":>10} {np.trace(Mda_o):>+12.4f}')
    sys.stdout.flush()

    # == Part 2: M_prime effective rank ==
    print('\n  === PART 2: M_PRIME_ODD SPECTRAL STRUCTURE ===')
    print(f'  What is the effective rank of M_prime on the odd block?')
    print(f'  (Number of eigenvalues > 1% of max)')
    print()

    for lam_sq in [200, 1000, 5000, 20000]:
        Mp_o, _, _, _, _, L, N = build_components_odd(lam_sq)
        ep = np.linalg.eigvalsh(Mp_o)
        abs_ep = np.abs(ep)
        max_abs = abs_ep.max()
        eff_rank_1pct = np.sum(abs_ep > 0.01 * max_abs)
        eff_rank_10pct = np.sum(abs_ep > 0.10 * max_abs)
        op_norm = max_abs

        print(f'  lam^2={lam_sq:>6d}: ||Mp_odd||_2={op_norm:>8.3f}, '
              f'eff_rank(1%)={eff_rank_1pct:>3d}/{N}, '
              f'eff_rank(10%)={eff_rank_10pct:>3d}/{N}')
        # Top 5 singular values
        print(f'    top 5 |eig|: {abs_ep[-1]:.3f}, {abs_ep[-2]:.3f}, '
              f'{abs_ep[-3]:.3f}, {abs_ep[-4]:.3f}, {abs_ep[-5]:.3f}')
    sys.stdout.flush()

    # == Part 3: Weyl perturbation bound ==
    print('\n  === PART 3: WEYL PERTURBATION BOUND ===')
    print(f'  max_eig(M_odd) <= max_eig(Mda_odd) + ||Mp_odd||_2')
    print(f'  Need: ||Mp_odd||_2 < |max_eig(Mda_odd)| for proof.')
    print()

    print(f'  {"lam^2":>8} {"max(Mda)":>12} {"||Mp||_2":>12} '
          f'{"bound":>12} {"actual":>14} {"gap":>12}')
    print('  ' + '-' * 76)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        Mp_o, _, _, Mda_o, Mt_o, L, N = build_components_odd(lam_sq)
        e_mda = np.linalg.eigvalsh(Mda_o)
        e_mt = np.linalg.eigvalsh(Mt_o)
        norm_mp = np.linalg.norm(Mp_o, 2)  # operator norm = max |eigenvalue|
        bound = e_mda[-1] + norm_mp
        actual = e_mt[-1]
        gap = e_mda[-1] + norm_mp  # if this is < 0, Weyl proves neg def

        print(f'  {lam_sq:>8d} {e_mda[-1]:>+12.4f} {norm_mp:>12.4f} '
              f'{bound:>+12.4f} {actual:>+14.8e} {gap:>+12.4f}')
    sys.stdout.flush()

    # == Part 4: Refined perturbation — project out dominant directions ==
    print('\n  === PART 4: REFINED PERTURBATION ===')
    print(f'  M_prime_odd has a few large eigenvalues and many small ones.')
    print(f'  Split: Mp_odd = Mp_large + Mp_small')
    print(f'  where Mp_large = sum of top-k eigenvalue components.')
    print(f'  Then: Mda_odd + Mp_small might be neg def if ||Mp_small||_2')
    print(f'  is smaller than |max_eig(Mda_odd)|.')
    print()

    lam_sq = 1000
    Mp_o, _, _, Mda_o, Mt_o, L, N = build_components_odd(lam_sq)
    ep, vp = np.linalg.eigh(Mp_o)

    print(f'  At lam^2=1000 (N={N}):')
    print(f'  max_eig(Mda_odd) = {np.linalg.eigvalsh(Mda_o)[-1]:+.6f}')
    print()

    for k in [1, 2, 3, 5, 10, 15, 20]:
        if k >= N:
            break
        # Remove top-k and bottom-k eigenvalue components from Mp
        # Keep only the "middle" eigenvalues
        ep_small = ep.copy()
        ep_small[-k:] = 0  # zero out top k
        ep_small[:k] = 0   # zero out bottom k
        Mp_small = (vp * ep_small) @ vp.T
        Mp_large = Mp_o - Mp_small
        norm_small = np.abs(ep_small).max()

        # Test: Mda + Mp_small
        combined = Mda_o + Mp_small
        e_combined = np.linalg.eigvalsh(combined)
        max_combined = e_combined[-1]

        # Test: full thing with Mp_large added back
        # (should equal Mt_o)

        e_mda = np.linalg.eigvalsh(Mda_o)[-1]
        print(f'  k={k:>2d}: ||Mp_small||_2={norm_small:>8.4f}, '
              f'max_eig(Mda+Mp_small)={max_combined:>+10.6f}, '
              f'Weyl bound={e_mda + norm_small:>+10.4f}')
    sys.stdout.flush()

    # == Part 5: How many prime eigenvalues need removing? ==
    print('\n  === PART 5: MINIMUM k TO MAKE WEYL CLOSE ===')
    print(f'  Find smallest k such that removing top+bottom k eigenvalues')
    print(f'  of Mp_odd makes ||Mp_small||_2 < |max_eig(Mda_odd)|.')
    print()

    for lam_sq in [200, 1000, 5000, 20000]:
        Mp_o, _, _, Mda_o, Mt_o, L, N = build_components_odd(lam_sq)
        ep = np.sort(np.abs(np.linalg.eigvalsh(Mp_o)))[::-1]  # sorted desc
        e_mda_max = np.linalg.eigvalsh(Mda_o)[-1]
        threshold = abs(e_mda_max)

        k_needed = 0
        for k in range(N):
            if k < N and ep[k] > threshold:
                k_needed = k + 1
            else:
                break

        print(f'  lam^2={lam_sq:>6d}: |max(Mda)|={threshold:.4f}, '
              f'k_needed={k_needed} out of {N} '
              f'({100*k_needed/N:.1f}%)')
    sys.stdout.flush()

    # == Part 6: Archimedean Toeplitz structure ==
    print('\n  === PART 6: IS M_diag+alpha_odd TOEPLITZ? ===')
    from session59_toeplitz_probe import toeplitz_deviation

    for lam_sq in [200, 1000, 5000, 20000]:
        _, _, _, Mda_o, _, L, N = build_components_odd(lam_sq)
        dev, T, _ = toeplitz_deviation(Mda_o)
        eT = np.linalg.eigvalsh(T)
        eMda = np.linalg.eigvalsh(Mda_o)
        print(f'  lam^2={lam_sq:>6d}: Toeplitz dev={dev:.4f}, '
              f'max_eig(Mda)={eMda[-1]:+.4f}, '
              f'max_eig(Toeplitz(Mda))={eT[-1]:+.4f}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
