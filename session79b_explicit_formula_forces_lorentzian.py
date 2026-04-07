"""
SESSION 79b -- DOES THE EXPLICIT FORMULA FORCE LORENTZIAN SIGNATURE?

The phase transition is at EXACTLY scale=1.0. This is the scale where
M satisfies the explicit formula identity. At any other scale, the
matrix doesn't correspond to a valid zeta function.

HYPOTHESIS: The Lorentzian property is a CONSEQUENCE of the explicit
formula identity. Proving this would close the gap.

APPROACH: The explicit formula says:
  sum_rho h(rho - 1/2) = (archimedean) + (prime)
  The LHS is the zero sum. The RHS builds M.

  At scale=1: M is built from the RHS. The zero sum equals the prime sum.
  The matrix M encodes the explicit formula.

KEY TEST: If we build M from RANDOM coefficients (not satisfying any
explicit formula), does it ever have Lorentzian signature?

If Lorentzian signature is GENERIC for random matrices: the explicit
formula is unnecessary, and the proof path is dead.

If Lorentzian signature is EXTREMELY RARE: the explicit formula is the
constraint that forces it, and we're on track.

PROBES:
  1. Random diagonal + same Cauchy off-diagonal: how often Lorentzian?
  2. Random diagonal with -log(n) envelope: how often?
  3. Random primes (Cramer) + actual Gamma diagonal: how often?
  4. The explicit formula as a LINEAR CONSTRAINT on eigenvalues
  5. Can we express "sum h_pair = C" as a constraint on M's spectrum?
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


def build_M_cramer(lam_sq, N=None, seed=None):
    """Build M using Cramer random primes instead of actual primes."""
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    actual_primes = list(sieve_primes(int(lam_sq)))
    n_primes = len(actual_primes)

    rng = np.random.RandomState(seed)
    cramer = []
    n = 2
    while len(cramer) < n_primes and n < 5 * lam_sq:
        if rng.random() < 1.0 / max(np.log(n), 1):
            cramer.append(n)
        n += 1
    cramer = cramer[:n_primes]

    # Build M with Cramer primes
    a_prime = np.zeros(dim)
    B_prime = np.zeros(dim)
    for p in cramer:
        pk = int(p)
        logp = np.log(float(p))
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(float(pk))
            a_prime += w * 2 * (L - y) / L * np.cos(2 * np.pi * ns * y / L)
            B_prime += w * np.sin(2 * np.pi * ns * y / L) / np.pi
            pk *= int(p)

    a_n = np.array([wr[abs(int(n))] for n in ns]) + a_prime
    B_n = alpha + B_prime

    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        offdiag = (B_n[None, :] - B_n[:, None]) / nm
    np.fill_diagonal(offdiag, 0)

    M = np.diag(a_n) + offdiag
    M = (M + M.T) / 2

    return M, N


def run():
    print()
    print('#' * 76)
    print('  SESSION 79b -- DOES EXPLICIT FORMULA FORCE LORENTZIAN?')
    print('#' * 76)

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))

    # ======================================================================
    # TEST 1: Cramer primes + actual Gamma diagonal
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 1: CRAMER PRIMES + ACTUAL GAMMA DIAGONAL')
    print(f'{"="*76}\n')

    print(f'  lam^2={lam_sq}: using actual Gamma diagonal + random Cramer primes')
    print(f'  {"trial":>6} {"#pos(M)":>8} {"#pos(Mo)":>8} {"eig_max(Mo)":>14} {"Lorentzian?":>12}')
    print('  ' + '-' * 54)

    n_lor = 0
    n_trials = 200
    for trial in range(n_trials):
        M_c, N_c = build_M_cramer(lam_sq, N, seed=trial)
        evals = np.linalg.eigvalsh(M_c)
        npos = np.sum(evals > 1e-10)

        Mo = odd_block(M_c, N_c)
        eo = np.linalg.eigvalsh(Mo)
        npos_o = np.sum(eo > 1e-10)
        emax_o = eo[-1]

        is_lor = npos <= 1
        if is_lor:
            n_lor += 1

        if trial < 20 or is_lor:
            print(f'  {trial+1:>6d} {npos:>8d} {npos_o:>8d} '
                  f'{emax_o:>+14.6e} {"YES" if is_lor else "no":>12}')

    print(f'\n  RESULT: {n_lor}/{n_trials} Cramer trials are Lorentzian')
    print(f'  Fraction: {n_lor/n_trials:.4f}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 2: Actual primes + WRONG Gamma diagonal
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 2: ACTUAL PRIMES + RANDOM DIAGONAL PERTURBATION')
    print(f'{"="*76}\n')

    _, M_real, _ = build_all_fast(lam_sq, N)
    wr = _compute_wr_diag(L, N)
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    wr_vec = np.array([wr[abs(int(n))] for n in ns])

    print(f'  Actual M is Lorentzian (#pos=1). Now perturb the diagonal:')
    print(f'  {"perturbation":>16} {"#pos(M)":>8} {"eig_max(Mo)":>14} {"Lorentzian?":>12}')
    print('  ' + '-' * 56)

    np.random.seed(42)
    for eps_name, eps in [('0', 0), ('1e-7', 1e-7), ('1e-6', 1e-6),
                           ('1e-5', 1e-5), ('1e-4', 1e-4), ('1e-3', 1e-3),
                           ('1e-2', 1e-2), ('0.1', 0.1), ('1.0', 1.0)]:
        perturbation = eps * np.random.randn(dim) * np.abs(wr_vec)
        M_pert = M_real + np.diag(perturbation)
        evals = np.linalg.eigvalsh(M_pert)
        npos = np.sum(evals > 1e-10)
        Mo = odd_block(M_pert, N)
        emax_o = np.linalg.eigvalsh(Mo)[-1]
        lor = npos <= 1
        print(f'  {eps_name:>16s} {npos:>8d} {emax_o:>+14.6e} '
              f'{"YES" if lor else "no":>12}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 3: The explicit formula as eigenvalue constraint
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 3: EXPLICIT FORMULA CONSTRAINS EIGENVALUE SUM')
    print(f'{"="*76}\n')

    _, M_real, QW = build_all_fast(lam_sq, N)
    evals_M = np.linalg.eigvalsh(M_real)

    # The trace of M = sum of eigenvalues = sum of diagonal entries
    tr_M = np.trace(M_real)
    tr_diag = np.sum(np.diag(M_real))

    print(f'  tr(M) = {tr_M:.6f}')
    print(f'  sum(diag(M)) = {tr_diag:.6f}')
    print(f'  sum(eigenvalues) = {np.sum(evals_M):.6f}')
    print()

    # The trace decomposes as:
    # tr(M) = tr(M_diag) + tr(M_alpha) + tr(M_prime)
    # tr(M_diag) = sum_n wr_diag(|n|) = archimedean
    # tr(M_alpha) = 0 (off-diagonal matrix)
    # tr(M_prime) = sum_n sum_{p^k} w * 2*(L-y)/L * cos(...)
    tr_wr = sum(wr[abs(int(n))] for n in ns)
    tr_prime = tr_M - tr_wr  # everything else

    print(f'  Decomposition:')
    print(f'    tr(M_diag) = {tr_wr:.6f}  (archimedean / Gamma)')
    print(f'    tr(rest)   = {tr_prime:.6f}  (primes + alpha)')
    print(f'    Total      = {tr_M:.6f}')
    print()

    # The Lorentzian property means: 1 positive eigenvalue (say lambda_+),
    # and 2N negative eigenvalues. The trace constrains:
    #   lambda_+ + sum(lambda_-) = tr(M)
    # Since sum(lambda_-) < 0 and lambda_+ > 0:
    #   lambda_+ = tr(M) - sum(lambda_-)
    #   lambda_+ < tr(M) (if sum(lambda_-) < 0, which it is)

    lam_plus = evals_M[-1]
    sum_neg = np.sum(evals_M[:-1])
    print(f'  lambda_+ = {lam_plus:.6f}')
    print(f'  sum(lambda_-) = {sum_neg:.6f}')
    print(f'  lambda_+ + sum(lambda_-) = {lam_plus + sum_neg:.6f} = tr(M)')
    print()

    # For M_odd (all negative):
    Mo = odd_block(M_real, N)
    eo = np.linalg.eigvalsh(Mo)
    tr_Mo = np.trace(Mo)
    print(f'  M_odd: tr = {tr_Mo:.6f}, sum(evals) = {np.sum(eo):.6f}')
    print(f'  All negative: {np.all(eo < 0)}')
    print(f'  eig_max = {eo[-1]:+.6e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 4: What fraction of the "diagonal space" gives Lorentzian?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 4: VOLUME OF LORENTZIAN REGION IN DIAGONAL SPACE')
    print(f'{"="*76}\n')

    # Fix the off-diagonal of M, vary the diagonal randomly.
    # What fraction of random diagonals give Lorentzian signature?
    M_offdiag = M_real - np.diag(np.diag(M_real))

    n_lor_random = 0
    n_lor_loglike = 0
    n_random = 1000

    np.random.seed(123)
    for trial in range(n_random):
        # Completely random diagonal (same scale as actual)
        d_rand = np.random.randn(dim) * np.std(np.diag(M_real))
        M_rand = M_offdiag + np.diag(d_rand)
        evals = np.linalg.eigvalsh(M_rand)
        if np.sum(evals > 1e-10) <= 1:
            n_lor_random += 1

    for trial in range(n_random):
        # Log-like diagonal: d_n = a + b * log(|n|+1) + noise
        a = np.random.uniform(2, 6)
        b = np.random.uniform(-2, 0)
        noise = np.random.randn(dim) * 0.1
        d_log = np.array([a + b * np.log(abs(int(n)) + 1) + noise[i]
                          for i, n in enumerate(ns)])
        M_log = M_offdiag + np.diag(d_log)
        evals = np.linalg.eigvalsh(M_log)
        if np.sum(evals > 1e-10) <= 1:
            n_lor_loglike += 1

    print(f'  Random diagonal (Gaussian): {n_lor_random}/{n_random} Lorentzian '
          f'({n_lor_random/n_random*100:.2f}%)')
    print(f'  Log-like diagonal (a + b*log(n) + noise): '
          f'{n_lor_loglike}/{n_random} Lorentzian ({n_lor_loglike/n_random*100:.2f}%)')
    sys.stdout.flush()

    # ======================================================================
    # TEST 5: Does Cramer + Gamma EVER give M_odd < 0?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 5: CRAMER PRIMES — IS M_ODD EVER NEGATIVE DEFINITE?')
    print(f'{"="*76}\n')

    n_nd = 0
    n_test = 200
    min_emax = float('inf')

    for trial in range(n_test):
        M_c, N_c = build_M_cramer(lam_sq, N, seed=1000 + trial)
        Mo = odd_block(M_c, N_c)
        emax = np.linalg.eigvalsh(Mo)[-1]
        if emax < 0:
            n_nd += 1
        min_emax = min(min_emax, emax)

    print(f'  {n_nd}/{n_test} Cramer trials have M_odd negative definite')
    print(f'  Minimum eig_max(M_odd) across trials: {min_emax:+.6e}')
    print(f'  (actual: {eo[-1]:+.6e})')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 79b VERDICT')
    print('=' * 76)
    print()
    print('  If Cramer gives 0% Lorentzian and 0% M_odd<0:')
    print('    The explicit formula (actual primes) is the UNIQUE constraint')
    print('    that creates the Lorentzian property. Proof path is viable.')
    print()
    print('  If Cramer gives >0%:')
    print('    The Lorentzian property has a generic component.')
    print('    The explicit formula helps but is not the sole cause.')
    print()


if __name__ == '__main__':
    run()
