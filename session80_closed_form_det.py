"""
SESSION 80 -- THE 2x2 LEADING MINOR HAS A CLOSED FORM

Key insight: the odd block entries have closed forms:
  Mo[j-1, k-1] = (B_k - B_j) / (j - k)   for j != k
  Mo[j-1, j-1] = a_j + B_j / j

where a_n and B_n are determined by the explicit formula.

The 2x2 leading minor determinant is:
  det = (a_1 + B_1)(a_2 + B_2/2) - (B_1 - B_2)^2

This is a CLOSED-FORM expression in a_1, a_2, B_1, B_2 -- all from
the explicit formula. No eigenvalues needed.

If det > 0 AND Mo[0,0] < 0: the 2x2 leading minor is negative definite.
This is necessary (but not sufficient) for M_odd < 0.

QUESTION: does det > 0 match the Schur gap ~4?
If yes: proving det > 0 from the explicit formula is the path.
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)
from session41g_uncapped_barrier import sieve_primes


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def compute_explicit_formula(lam_sq):
    """Compute a_n and B_n from the explicit formula."""
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
            a_prime += w * 2 * (L - y) / L * np.cos(2 * np.pi * ns * y / L)
            B_prime += w * np.sin(2 * np.pi * ns * y / L) / np.pi
            pk *= int(p)

    a_n = np.array([wr[abs(int(n))] for n in ns]) + a_prime
    B_n = alpha + B_prime

    return a_n, B_n, N, L, dim, ns


def run():
    print()
    print('#' * 76)
    print('  SESSION 80 -- CLOSED-FORM 2x2 DETERMINANT')
    print('#' * 76)

    # ======================================================================
    # STEP 1: Verify the closed forms
    # ======================================================================
    print('\n  === STEP 1: VERIFY CLOSED FORMS ===\n')

    lam_sq = 1000
    a_n, B_n, N, L, dim, ns = compute_explicit_formula(lam_sq)
    _, M, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M, N)

    # Mo[j-1, j-1] = a_j + B_j / j
    # Mo[j-1, k-1] = (B_k - B_j) / (j - k)  for j != k
    print(f'  Odd block closed forms (lam^2={lam_sq}):')
    for j in range(1, 6):
        formula = a_n[N + j] + B_n[N + j] / j
        actual = Mo[j - 1, j - 1]
        print(f'    Mo[{j},{j}] = a_{j} + B_{j}/{j} = {formula:+.8f} '
              f'(actual: {actual:+.8f}, err: {abs(formula - actual):.2e})')

    for j, k in [(1, 2), (1, 3), (2, 3), (1, 5)]:
        formula = (B_n[N + k] - B_n[N + j]) / (j - k)
        actual = Mo[j - 1, k - 1]
        print(f'    Mo[{j},{k}] = (B_{k}-B_{j})/({j}-{k}) = {formula:+.8f} '
              f'(actual: {actual:+.8f}, err: {abs(formula - actual):.2e})')
    sys.stdout.flush()

    # ======================================================================
    # STEP 2: The 2x2 determinant in closed form
    # ======================================================================
    print(f'\n  === STEP 2: 2x2 DETERMINANT ===\n')

    # det = (a_1 + B_1)(a_2 + B_2/2) - (B_1 - B_2)^2
    a1 = a_n[N + 1]
    a2 = a_n[N + 2]
    B1 = B_n[N + 1]
    B2 = B_n[N + 2]

    d00 = a1 + B1
    d11 = a2 + B2 / 2
    d01 = B1 - B2
    det_2x2 = d00 * d11 - d01**2

    print(f'  a_1 = {a1:+.8f}')
    print(f'  a_2 = {a2:+.8f}')
    print(f'  B_1 = {B1:+.8f}')
    print(f'  B_2 = {B2:+.8f}')
    print()
    print(f'  Mo[0,0] = a_1 + B_1 = {d00:+.8f}')
    print(f'  Mo[1,1] = a_2 + B_2/2 = {d11:+.8f}')
    print(f'  Mo[0,1] = B_1 - B_2 = {d01:+.8f}')
    print()
    print(f'  det = {d00:.4f} * {d11:.4f} - {d01:.4f}^2')
    print(f'      = {d00 * d11:.4f} - {d01**2:.4f}')
    print(f'      = {det_2x2:+.6f}')
    print()

    # Compare to Schur gap
    c = Mo[0, 1:]
    B_mat = Mo[1:, 1:]
    Be, Bv = np.linalg.eigh(B_mat)
    schur_gap = abs(Mo[0, 0]) * abs(Be[0]) - (float(c @ Bv[:, 0]))**2
    print(f'  Schur gap = {schur_gap:+.6f}')
    print(f'  2x2 det / Schur gap = {det_2x2 / schur_gap:.4f}')
    sys.stdout.flush()

    # ======================================================================
    # STEP 3: Dense scan of 2x2 determinant
    # ======================================================================
    print(f'\n  === STEP 3: 2x2 DETERMINANT vs LAMBDA ===\n')

    print(f'  {"lam^2":>8} {"a_1+B_1":>10} {"a_2+B_2/2":>10} '
          f'{"B_1-B_2":>10} {"2x2 det":>10} {"det>0?":>6}')
    print('  ' + '-' * 58)

    all_positive = True
    for ls in [10, 15, 20, 30, 50, 75, 100, 200, 500, 1000,
               2000, 5000, 10000, 50000, 100000, 200000]:
        a_n, B_n, N, L, dim, ns = compute_explicit_formula(ls)
        d00 = a_n[N + 1] + B_n[N + 1]
        d11 = a_n[N + 2] + B_n[N + 2] / 2
        d01 = B_n[N + 1] - B_n[N + 2]
        det2 = d00 * d11 - d01**2
        ok = det2 > 0
        if not ok:
            all_positive = False
        print(f'  {ls:>8d} {d00:>+10.4f} {d11:>+10.4f} '
              f'{d01:>+10.4f} {det2:>+10.4f} {"YES" if ok else "**NO**":>6}')

    print(f'\n  All positive: {all_positive}')
    sys.stdout.flush()

    # ======================================================================
    # STEP 4: Decompose into archimedean + prime
    # ======================================================================
    print(f'\n  === STEP 4: ARCHIMEDEAN vs PRIME DECOMPOSITION ===\n')

    for ls in [200, 1000, 10000]:
        L = np.log(ls)
        N = max(15, round(6 * L))
        dim = 2 * N + 1
        ns = np.arange(-N, N + 1, dtype=float)

        wr = _compute_wr_diag(L, N)
        alpha = _compute_alpha(L, N)

        # Archimedean parts
        a1_arch = wr[1]
        a2_arch = wr[2]
        B1_arch = alpha[N + 1]
        B2_arch = alpha[N + 2]

        # Full parts
        a_n, B_n, _, _, _, _ = compute_explicit_formula(ls)
        a1_full = a_n[N + 1]
        a2_full = a_n[N + 2]
        B1_full = B_n[N + 1]
        B2_full = B_n[N + 2]

        # Prime parts
        a1_prime = a1_full - a1_arch
        a2_prime = a2_full - a2_arch
        B1_prime = B1_full - B1_arch
        B2_prime = B2_full - B2_arch

        # Det with arch only
        d00_a = a1_arch + B1_arch
        d11_a = a2_arch + B2_arch / 2
        d01_a = B1_arch - B2_arch
        det_arch = d00_a * d11_a - d01_a**2

        # Det with full
        d00_f = a1_full + B1_full
        d11_f = a2_full + B2_full / 2
        d01_f = B1_full - B2_full
        det_full = d00_f * d11_f - d01_f**2

        print(f'  lam^2={ls}:')
        print(f'    Arch: a1={a1_arch:+.4f}, a2={a2_arch:+.4f}, '
              f'B1={B1_arch:+.4f}, B2={B2_arch:+.4f}')
        print(f'    Prime: a1={a1_prime:+.4f}, a2={a2_prime:+.4f}, '
              f'B1={B1_prime:+.4f}, B2={B2_prime:+.4f}')
        print(f'    det(arch only) = {det_arch:+.6f}')
        print(f'    det(full)      = {det_full:+.6f}')
        print(f'    Prime effect on det: {det_full - det_arch:+.6f}')
        print()
    sys.stdout.flush()

    # ======================================================================
    # STEP 5: Expand the determinant algebraically
    # ======================================================================
    print(f'\n  === STEP 5: ALGEBRAIC EXPANSION ===\n')

    # det = (a_1 + B_1)(a_2 + B_2/2) - (B_1 - B_2)^2
    # = a_1*a_2 + a_1*B_2/2 + B_1*a_2 + B_1*B_2/2 - B_1^2 + 2*B_1*B_2 - B_2^2
    # = a_1*a_2 + a_1*B_2/2 + B_1*a_2 + B_1*B_2/2 - B_1^2 + 2*B_1*B_2 - B_2^2
    # = a_1*a_2 + a_1*B_2/2 + a_2*B_1 + 5*B_1*B_2/2 - B_1^2 - B_2^2

    ls = 1000
    a_n, B_n, N, L, dim, ns = compute_explicit_formula(ls)
    a1 = a_n[N + 1]
    a2 = a_n[N + 2]
    B1 = B_n[N + 1]
    B2 = B_n[N + 2]

    terms = {
        'a1*a2': a1 * a2,
        'a1*B2/2': a1 * B2 / 2,
        'a2*B1': a2 * B1,
        'B1*B2/2': B1 * B2 / 2,
        '-B1^2': -B1**2,
        '2*B1*B2': 2 * B1 * B2,
        '-B2^2': -B2**2,
    }

    print(f'  det = (a_1+B_1)(a_2+B_2/2) - (B_1-B_2)^2')
    print(f'  Expanding:')
    total = 0
    for name, val in terms.items():
        total += val
        print(f'    {name:>12}: {val:>+12.4f}')
    print(f'    {"TOTAL":>12}: {total:>+12.4f}')
    print(f'    Actual det:  {(a1 + B1) * (a2 + B2 / 2) - (B1 - B2)**2:>+12.4f}')
    sys.stdout.flush()

    # ======================================================================
    # STEP 6: ALL Sylvester minors (necessary for full proof)
    # ======================================================================
    print(f'\n  === STEP 6: ALL SYLVESTER LEADING MINORS ===\n')

    ls = 1000
    _, M, _ = build_all_fast(ls, max(15, round(6 * np.log(ls))))
    N = max(15, round(6 * np.log(ls)))
    Mo = odd_block(M, N)

    # For neg-def: (-1)^k * det(Mo[0:k, 0:k]) > 0 for k=1,...,N
    print(f'  Sylvester criterion for M_odd (lam^2={ls}):')
    print(f'  {"k":>4} {"(-1)^k * det":>16} {">0?":>6}')
    print('  ' + '-' * 30)

    all_pass = True
    for k in range(1, min(N + 1, 20)):
        minor = Mo[:k, :k]
        sign, logdet = np.linalg.slogdet(minor)
        sylv = ((-1)**k) * sign * np.exp(logdet)
        ok = sylv > 0
        if not ok:
            all_pass = False
        if k <= 10 or not ok:
            print(f'  {k:>4d} {sylv:>+16.6e} {"YES" if ok else "**NO**":>6}')

    if all_pass:
        print(f'  ...')
        print(f'  All {min(N, 19)} leading minors satisfy Sylvester: YES')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 80 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
