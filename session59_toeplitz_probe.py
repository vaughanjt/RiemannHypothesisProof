"""
SESSION 59 -- IS M_ODD TOEPLITZ? (CONNES SPECTRAL TRIPLE CONNECTION)

Connes' 2025 paper (arXiv:2511.22755) constructs zeta spectral triples
from rank-1 perturbations of the scaling operator, using Caratheodory-
Fejer (CF) theory for Toeplitz matrices. The CF theorem guarantees
positivity of Toeplitz matrices from their symbol — a NON-CIRCULAR
positivity criterion that doesn't require knowing zeros or primes.

If M_odd (or -M_odd) has Toeplitz structure, CF theory might prove
its definiteness without any circular argument.

A Toeplitz matrix has T[i,j] = t[i-j] — constant along diagonals.
On the odd subspace (basis n=1,2,...,N), Toeplitz means:
  M_odd[i,j] depends only on (i-j).

Plan:
  1. Check if M_odd is Toeplitz (measure deviation from Toeplitz).
  2. Check each component (M_prime_odd, M_diag_odd, M_alpha_odd).
  3. If not Toeplitz, check if it's ASYMPTOTICALLY Toeplitz (diagonal
     bands converge as N grows).
  4. Check if there's a basis change that makes it Toeplitz.
  5. If Toeplitz-like: extract the symbol and check CF positivity.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import (
    build_all_fast, _build_M_prime, _compute_alpha, _compute_wr_diag
)


def odd_block(M, N):
    """Extract the N x N odd block."""
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def toeplitz_deviation(A):
    """
    Measure how far A is from being Toeplitz.
    Returns: (relative deviation, Toeplitz approximation).

    For each diagonal k, compute mean and std of entries.
    Deviation = sqrt(sum of variances) / Frobenius norm.
    """
    N = A.shape[0]
    diag_means = {}
    diag_vars = {}

    for k in range(-N + 1, N):
        entries = np.diag(A, k)
        diag_means[k] = entries.mean()
        diag_vars[k] = entries.var()

    # Build Toeplitz approximation
    T = np.zeros_like(A)
    for k in range(-N + 1, N):
        np.fill_diagonal(T[max(0, -k):, max(0, k):], diag_means[k])

    total_var = sum(diag_vars.values())
    frob = np.linalg.norm(A, 'fro')
    rel_dev = np.sqrt(total_var) / frob if frob > 0 else 0

    return rel_dev, T, diag_means


def run():
    print()
    print('#' * 76)
    print('  SESSION 59 -- IS M_ODD TOEPLITZ?')
    print('#' * 76)

    # == Part 1: Toeplitz deviation of M_odd and components ==
    print('\n  === PART 1: TOEPLITZ DEVIATION ===')
    print(f'  Deviation = 0 means perfectly Toeplitz.')
    print(f'  Deviation = 1 means maximally non-Toeplitz.')
    print()

    print(f'  {"lam^2":>8} {"M_prime_o":>12} {"M_diag_o":>12} '
          f'{"M_alpha_o":>12} {"M_total_o":>12} {"-M_total_o":>12}')
    print('  ' + '-' * 68)

    for lam_sq in [50, 200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))

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
        Mt = Mp + Md + Ma

        Mp_o = odd_block(Mp, N)
        Md_o = odd_block(Md, N)
        Ma_o = odd_block(Ma, N)
        Mt_o = odd_block(Mt, N)

        dp = toeplitz_deviation(Mp_o)[0]
        dd = toeplitz_deviation(Md_o)[0]
        da = toeplitz_deviation(Ma_o)[0]
        dt = toeplitz_deviation(Mt_o)[0]
        dnt = toeplitz_deviation(-Mt_o)[0]

        print(f'  {lam_sq:>8d} {dp:>12.6f} {dd:>12.6f} '
              f'{da:>12.6f} {dt:>12.6f} {dnt:>12.6f}')
    sys.stdout.flush()

    # == Part 2: Detailed diagonal structure at lam^2=1000 ==
    print('\n  === PART 2: DIAGONAL BAND STRUCTURE (lam^2=1000) ===')
    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))

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
    Mt = Mp + Md + Ma

    Mt_o = odd_block(Mt, N)

    print(f'  M_odd is {N}x{N}. Diagonal band analysis:')
    print(f'  {"diag k":>8} {"mean":>12} {"std":>12} {"std/|mean|":>12} '
          f'{"n_entries":>10}')
    print('  ' + '-' * 58)

    _, _, diag_means = toeplitz_deviation(Mt_o)
    for k in range(min(N, 15)):
        entries = np.diag(Mt_o, k)
        if len(entries) > 0:
            mn = entries.mean()
            sd = entries.std()
            rel = sd / abs(mn) if abs(mn) > 1e-15 else float('inf')
            print(f'  {k:>8d} {mn:>+12.6f} {sd:>12.6f} {rel:>12.4f} '
                  f'{len(entries):>10d}')
    sys.stdout.flush()

    # == Part 3: M_diag_odd — is it Toeplitz? ==
    print('\n  === PART 3: M_DIAG ON ODD BLOCK ===')
    print(f'  M_diag is diagonal in the Fourier basis.')
    print(f'  On the odd block, M_diag_odd[i,j] = wr_diag[i+1] * delta_ij.')
    print(f'  This is diagonal (hence Toeplitz only if constant diagonal).')
    print(f'  wr_diag[n] varies with n, so M_diag_odd is NOT Toeplitz.')
    print(f'  BUT: is wr_diag[n] approximately linear in n?')
    print()

    Md_o = odd_block(Md, N)
    diag_vals = np.diag(Md_o)
    ns_odd = np.arange(1, N + 1)

    # Fit wr_diag[n] = a + b*log(n)
    log_ns = np.log(ns_odd)
    A_fit = np.column_stack([np.ones(N), log_ns])
    coeffs, *_ = np.linalg.lstsq(A_fit, diag_vals, rcond=None)
    fit = A_fit @ coeffs
    residual = diag_vals - fit
    print(f'  wr_diag[n] ~ {coeffs[0]:+.6f} + {coeffs[1]:+.6f} * log(n)')
    print(f'  Residual RMS: {np.sqrt((residual**2).mean()):.6f}')
    print(f'  Residual max: {np.max(np.abs(residual)):.6f}')
    print(f'  This is the known asymptotics: wr_diag[n] = C(L) - log(n) + O(1/n)')
    sys.stdout.flush()

    # == Part 4: M_alpha_odd — Toeplitz structure? ==
    print('\n  === PART 4: M_ALPHA ON ODD BLOCK ===')
    Ma_o = odd_block(Ma, N)
    da, Ta, _ = toeplitz_deviation(Ma_o)
    print(f'  Toeplitz deviation: {da:.6f}')
    print(f'  M_alpha_odd diagonal bands:')
    for k in range(min(N, 8)):
        entries = np.diag(Ma_o, k)
        print(f'    k={k}: mean={entries.mean():+.6f}, std={entries.std():.6f}')
    sys.stdout.flush()

    # == Part 5: M_prime_odd — Toeplitz structure? ==
    print('\n  === PART 5: M_PRIME ON ODD BLOCK ===')
    Mp_o = odd_block(Mp, N)
    dp, Tp, _ = toeplitz_deviation(Mp_o)
    print(f'  Toeplitz deviation: {dp:.6f}')
    print(f'  M_prime_odd diagonal bands:')
    for k in range(min(N, 8)):
        entries = np.diag(Mp_o, k)
        print(f'    k={k}: mean={entries.mean():+.6f}, std={entries.std():.6f}')
    sys.stdout.flush()

    # == Part 6: Toeplitz approximation — does it preserve negativity? ==
    print('\n  === PART 6: TOEPLITZ APPROXIMATION EIGENVALUES ===')
    print(f'  Replace M_odd with its Toeplitz average. Is it still neg def?')
    print()

    _, T_approx, _ = toeplitz_deviation(Mt_o)
    eT = np.linalg.eigvalsh(T_approx)
    eM = np.linalg.eigvalsh(Mt_o)

    print(f'  M_odd:            max_eig = {eM[-1]:+.6e}, min_eig = {eM[0]:+.6f}')
    print(f'  Toeplitz(M_odd):  max_eig = {eT[-1]:+.6e}, min_eig = {eT[0]:+.6f}')
    print(f'  Toeplitz approx is {"neg def" if eT[-1] < 0 else "NOT neg def"}')

    if eT[-1] < 0:
        print(f'\n  The Toeplitz approximation IS negative definite!')
        print(f'  For Toeplitz matrices, eigenvalues are determined by the')
        print(f'  symbol f(theta) = sum_k t_k exp(i*k*theta).')
        print(f'  If f(theta) < 0 for all theta, the Toeplitz matrix is neg def.')
        print()

        # Compute the symbol
        print(f'  Symbol f(theta) of the Toeplitz approximation:')
        _, _, dm = toeplitz_deviation(Mt_o)
        thetas = np.linspace(0, 2 * np.pi, 500)
        symbol = np.zeros(len(thetas))
        for k in range(-N + 1, N):
            symbol += dm[k] * np.exp(1j * k * thetas).real

        print(f'    max f(theta): {symbol.max():+.6f}')
        print(f'    min f(theta): {symbol.min():+.6f}')
        print(f'    f(theta) < 0 everywhere: {symbol.max() < 0}')
        if symbol.max() < 0:
            print(f'\n  ** SYMBOL IS STRICTLY NEGATIVE **')
            print(f'  This means the Toeplitz approximation of M_odd is')
            print(f'  negative definite BY THE CF THEOREM.')
    else:
        print(f'\n  Toeplitz approximation is NOT negative definite.')
        print(f'  CF route does not directly apply.')

    # == Part 7: How good is the Toeplitz approximation? ==
    print(f'\n  === PART 7: TOEPLITZ APPROXIMATION QUALITY ===')

    for lam_sq in [200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M_t, _ = build_all_fast(lam_sq, N)
        Mt_o_t = odd_block(M_t, N)
        dev, T_t, _ = toeplitz_deviation(Mt_o_t)
        eT_t = np.linalg.eigvalsh(T_t)
        eM_t = np.linalg.eigvalsh(Mt_o_t)
        diff_norm = np.linalg.norm(Mt_o_t - T_t, 'fro') / np.linalg.norm(Mt_o_t, 'fro')

        print(f'  lam^2={lam_sq:>6d}: dev={dev:.4f}, '
              f'max_eig(M)={eM_t[-1]:+.2e}, '
              f'max_eig(T)={eT_t[-1]:+.2e}, '
              f'||M-T||/||M||={diff_norm:.4f}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
