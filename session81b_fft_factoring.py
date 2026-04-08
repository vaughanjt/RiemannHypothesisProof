"""
SESSION 81b -- FFT FACTORING IN ZERO-SPACE

The naive zero-sieve checks candidates one-by-one (slow).
FFT insight: instead of deconvolving with each candidate p separately,
compute ALL deconvolutions simultaneously via correlation.

The idea:
  1. Build a "prime signature dictionary" S(log p) for all primes p < sqrt(n)
  2. Compute the "target signal" T(x) = cos(gamma_k * log(n) - gamma_k * x)
     for x in a dense grid
  3. The correlation C(x) = sum_k T_k(x) * S_k(x) peaks at x = log(p)
     whenever p divides n
  4. This correlation is a CONVOLUTION, computable via FFT in O(G log G)
     where G is the grid size, instead of O(P * K) for P primes and K zeros

This turns factoring into: find the peaks of a spectral correlation function.

PROBES:
  1. Build the correlation function C(x) and check it peaks at factors
  2. How sharp are the peaks? Do non-factors create false peaks?
  3. Timing: FFT correlation vs trial division vs naive zero-sieve
  4. Scale test: how does performance grow with n?
  5. Can we use FEWER zeros with FFT than with naive sieve?
"""

import sys
import numpy as np
import time

sys.path.insert(0, '.')

import mpmath
mpmath.mp.dps = 30


def load_zeros(K):
    return [float(mpmath.zetazero(k).imag) for k in range(1, K + 1)]


def run():
    print()
    print('#' * 76)
    print('  SESSION 81b -- FFT FACTORING IN ZERO-SPACE')
    print('#' * 76)

    K = 30
    zeros = np.array(load_zeros(K))

    # ======================================================================
    # PROBE 1: The correlation function
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: ZERO-SPACE CORRELATION FUNCTION')
    print(f'{"="*76}\n')

    n = 10403  # = 101 * 103
    log_n = np.log(float(n))

    # Build correlation on a grid of x = log(p_candidate)
    # x ranges from log(2) to log(sqrt(n))
    G = 10000  # grid points
    x_min = np.log(2)
    x_max = np.log(np.sqrt(n))
    x_grid = np.linspace(x_min, x_max, G)

    # For each grid point x, the deconvolved signature is:
    #   D_k(x) = cos(gamma_k * (log_n - x))
    # The "integer test" asks: is (log_n - x) the log of an integer?
    # An integer q has signature S_k(q) = cos(gamma_k * log(q))
    # Correlation: C(x) = (1/K) * sum_k D_k(x) * S_k(round(exp(log_n - x)))
    #
    # But this still requires evaluating at each grid point.
    #
    # BETTER: Build the correlation as a function of x directly.
    # C(x) = (1/K) * sum_k cos(gamma_k * (log_n - x)) * cos(gamma_k * log(round(n/exp(x))))
    #
    # The key insight: if x = log(p) for a factor p, then n/exp(x) = n/p = q (integer),
    # and C(x) = (1/K) * sum_k cos(gamma_k * log(q))^2 = high.
    # If x != log(p) for any factor, n/exp(x) is not an integer, and the
    # rounding error scrambles the correlation.
    #
    # SIMPLIFICATION: skip the rounding. Just compute:
    # C(x) = (1/K) * sum_k cos(gamma_k * (log_n - x))^2
    # This is ALWAYS high (average 0.5). Not useful.
    #
    # The RIGHT approach: use the EXPLICIT FORMULA as a matched filter.
    # The von Mangoldt function Lambda(n) = log(p) if n=p^k, 0 otherwise.
    # Its "zero expansion" is: Lambda(n) = 1 - sum_rho n^{rho-1} - ...
    # Peaks of the zero-expansion at integer arguments detect prime powers.
    #
    # For FACTORING: we want peaks at divisors of n.
    # Define: F(x) = sum_k cos(gamma_k * (log_n - x))
    # This oscillates wildly, BUT at x = log(p) where p | n,
    # log_n - x = log(n/p) = log(q) for integer q, and the
    # phases cos(gamma_k * log(q)) are "coherent" (they correspond
    # to an integer, which has structured phase relationships).
    #
    # For non-integer n/exp(x), the phases are "random" and cancel.

    # Compute F(x) = sum_k cos(gamma_k * (log_n - x))
    # This is a sum of K cosines at frequencies gamma_k.
    # It's a DETERMINISTIC function of x -- no randomness.

    F = np.zeros(G)
    for k in range(K):
        F += np.cos(zeros[k] * (log_n - x_grid))
    F /= K

    # Find peaks
    print(f'  n = {n} = 101 * 103, K = {K} zeros')
    print(f'  Grid: {G} points from x={x_min:.3f} to x={x_max:.3f}')
    print()

    # The peaks of F should be at x = log(p) for factors p
    log_101 = np.log(101)
    log_103 = np.log(103)

    # Find local maxima
    peaks = []
    for i in range(1, G - 1):
        if F[i] > F[i-1] and F[i] > F[i+1] and F[i] > 0.5:
            peaks.append((x_grid[i], F[i], int(round(np.exp(x_grid[i])))))

    print(f'  Peaks of F(x) above 0.5:')
    print(f'  {"x":>10} {"F(x)":>10} {"exp(x)":>10} {"integer?":>10} {"factor?":>8}')
    print('  ' + '-' * 52)

    for x_peak, f_peak, exp_x in peaks[:20]:
        is_factor = n % exp_x == 0 if exp_x > 1 else False
        near_int = abs(np.exp(x_peak) - exp_x) < 0.5
        print(f'  {x_peak:>10.4f} {f_peak:>10.6f} {exp_x:>10d} '
              f'{"yes" if near_int else "no":>10} '
              f'{"FACTOR" if is_factor else "":>8}')

    print(f'\n  log(101) = {log_101:.4f}, log(103) = {log_103:.4f}')
    print(f'  F(log(101)) = {np.interp(log_101, x_grid, F):.6f}')
    print(f'  F(log(103)) = {np.interp(log_103, x_grid, F):.6f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: The MATCHED FILTER approach
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: MATCHED FILTER — CORRELATE WITH INTEGER SIGNATURES')
    print(f'{"="*76}\n')

    # Better approach: for each candidate q = 2, 3, ..., sqrt(n):
    # Compute score(q) = |sum_k exp(i * gamma_k * log(n/q))|^2 / K^2
    # If n/q is an integer, the phases are coherent -> score ~ 1
    # If n/q is not an integer, phases are "random" -> score ~ 1/K

    # This is still O(sqrt(n) * K), but we can vectorize it.
    # AND: the sum_k exp(i * gamma_k * log(q)) for ALL q at once
    # is a Non-Uniform DFT (NUDFT), computable via NUFFT in O(K + G log G).

    q_candidates = np.arange(2, int(np.sqrt(n)) + 1)
    log_q = np.log(q_candidates.astype(float))
    log_nq = log_n - log_q  # log(n/q)

    # Vectorized score computation
    # score(q) = |sum_k exp(i * gamma_k * log(n/q))|^2 / K^2
    scores = np.zeros(len(q_candidates))
    for k in range(K):
        scores += np.cos(zeros[k] * log_nq)
    scores = (scores / K) ** 2

    # Add the sin part for full complex magnitude
    scores_sin = np.zeros(len(q_candidates))
    for k in range(K):
        scores_sin += np.sin(zeros[k] * log_nq)
    scores = (scores + (scores_sin / K) ** 2)

    # Top candidates
    top_idx = np.argsort(scores)[::-1]

    print(f'  Matched filter scores for n={n}:')
    print(f'  {"q":>6} {"n/q":>10} {"score":>10} {"n%q==0?":>8}')
    print('  ' + '-' * 38)

    for i in range(min(15, len(top_idx))):
        idx = top_idx[i]
        q = q_candidates[idx]
        nq = n / q
        is_div = n % q == 0
        print(f'  {q:>6d} {nq:>10.2f} {scores[idx]:>10.6f} '
              f'{"YES" if is_div else "":>8}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: NUFFT-style batch computation
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: BATCH COMPUTATION TIMING')
    print(f'{"="*76}\n')

    # The vectorized approach above is already O(sqrt(n) * K).
    # True FFT would require mapping to a uniform grid and using FFT.
    # Let's compare:
    #   A) Trial division: O(sqrt(n))
    #   B) Vectorized zero-score: O(sqrt(n) * K)
    #   C) FFT-based: O(G * log(G) + K * G) where G = grid size

    test_cases = [
        (143, 11, 13),
        (10403, 101, 103),
        (1018081, 1009, 1009),
        (10007 * 10009, 10007, 10009),  # ~100M
    ]

    print(f'  {"n":>12} {"trial (us)":>12} {"vec-zero (us)":>14} '
          f'{"ratio":>8} {"correct?":>10}')
    print('  ' + '-' * 62)

    for n, p_true, q_true in test_cases:
        log_n = np.log(float(n))
        sq = int(np.sqrt(n)) + 1

        # Trial division
        t0 = time.perf_counter()
        reps = max(1, min(1000, 10000000 // sq))
        for _ in range(reps):
            factor_td = 0
            for p in range(2, sq):
                if n % p == 0:
                    factor_td = p
                    break
        td_time = (time.perf_counter() - t0) / reps * 1e6

        # Vectorized zero-score
        t0 = time.perf_counter()
        reps_z = max(1, min(100, 1000000 // (sq * K)))
        for _ in range(reps_z):
            q_cands = np.arange(2, sq)
            log_nq = log_n - np.log(q_cands.astype(float))
            sc_cos = np.zeros(len(q_cands))
            sc_sin = np.zeros(len(q_cands))
            for k in range(K):
                sc_cos += np.cos(zeros[k] * log_nq)
                sc_sin += np.sin(zeros[k] * log_nq)
            sc = (sc_cos / K)**2 + (sc_sin / K)**2
            factor_zs = q_cands[np.argmax(sc)]
        zs_time = (time.perf_counter() - t0) / reps_z * 1e6

        correct = (factor_zs == p_true or factor_zs == q_true)
        ratio = zs_time / td_time if td_time > 0 else 0

        print(f'  {n:>12d} {td_time:>12.1f} {zs_time:>14.1f} '
              f'{ratio:>7.1f}x {"YES" if correct else "NO":>10}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: The SPECTRAL PEAK approach (true FFT)
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: TRUE FFT — SPECTRAL PEAK DETECTION')
    print(f'{"="*76}\n')

    # Map the problem to a uniform grid for FFT:
    # Define f(x) = sum_k exp(i * gamma_k * x) for x on a uniform grid
    # Then f(log(n/q)) peaks when n/q is an "integer-like" number.
    #
    # Substitution: let t = log(n) - x, so we want peaks at t = log(q)
    # for integer q. Compute:
    #   F(t) = sum_k exp(i * gamma_k * t)
    # on a uniform grid of t values.
    # Then: F(log(q)) is large when q is "resonant" with the zeros.
    #
    # But ALL integers q are equally resonant (the zeros don't prefer
    # any particular integer). So F(t) peaks at ALL log-integers,
    # not just at divisors of n.
    #
    # The TARGET-SPECIFIC part is the PHASE: we want
    #   G(t) = sum_k exp(i * gamma_k * (log_n - t))
    #        = exp(i * gamma_k * log_n) * conjugate(exp(i * gamma_k * t))
    #        = A_k * conj(B_k(t))
    #
    # where A_k = exp(i * gamma_k * log_n) encodes n.
    # This is a CORRELATION between A and B, computable via FFT.

    n = 10403
    log_n = np.log(float(n))
    G = 65536  # FFT size (power of 2)

    # Uniform grid in t = log(q) space
    t_min = np.log(2)
    t_max = np.log(float(n))
    dt = (t_max - t_min) / G
    t_grid = t_min + np.arange(G) * dt

    # Build the "target signal" A_k = exp(i * gamma_k * log_n)
    # Build the "dictionary signal" on the grid:
    #   B(t) = sum_k exp(i * gamma_k * t)

    t0 = time.perf_counter()

    # Compute B(t) on the uniform grid via vectorized sum
    B = np.zeros(G, dtype=complex)
    for k in range(K):
        B += np.exp(1j * zeros[k] * t_grid)

    # Compute A = sum_k exp(i * gamma_k * log_n) (scalar-ish, one per k)
    A = np.exp(1j * zeros * log_n)  # shape (K,)

    # Correlation: G(t) = sum_k A_k * conj(exp(i * gamma_k * t))
    # = sum_k exp(i * gamma_k * log_n) * exp(-i * gamma_k * t)
    # = sum_k exp(i * gamma_k * (log_n - t))
    # This is just B evaluated at (log_n - t), i.e., B_shifted.

    # Direct computation (already vectorized above):
    G_corr = np.zeros(G, dtype=complex)
    for k in range(K):
        G_corr += np.exp(1j * zeros[k] * (log_n - t_grid))
    G_mag = np.abs(G_corr) ** 2 / K**2

    fft_time = (time.perf_counter() - t0) * 1e6

    # Find peaks in the correlation
    # At t = log(p) where p|n: n/p = q (integer), phases coherent
    log_factors = [np.log(101), np.log(103)]

    print(f'  FFT correlation for n={n}, K={K}, G={G}:')
    print(f'  Computation time: {fft_time:.0f} us')
    print()

    # Top peaks
    peak_idx = np.argsort(G_mag)[::-1]
    print(f'  Top 10 peaks:')
    print(f'  {"t":>10} {"exp(t)":>10} {"|G|^2":>10} {"n/exp(t)":>10} {"factor?":>8}')
    print('  ' + '-' * 52)

    shown = 0
    seen_ints = set()
    for idx in peak_idx:
        t_val = t_grid[idx]
        exp_t = np.exp(t_val)
        exp_t_round = round(exp_t)
        if exp_t_round in seen_ints or exp_t_round < 2:
            continue
        seen_ints.add(exp_t_round)
        nq = n / exp_t
        is_factor = n % exp_t_round == 0 if exp_t_round > 0 else False
        print(f'  {t_val:>10.4f} {exp_t_round:>10d} {G_mag[idx]:>10.6f} '
              f'{nq:>10.1f} {"FACTOR" if is_factor else "":>8}')
        shown += 1
        if shown >= 10:
            break

    # Value at the actual factors
    print(f'\n  At factor locations:')
    for p in [101, 103]:
        t_p = np.log(float(p))
        idx_p = np.argmin(np.abs(t_grid - t_p))
        print(f'    p={p}: G_mag = {G_mag[idx_p]:.6f}, '
              f'rank = {np.sum(G_mag > G_mag[idx_p]) + 1}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 81b VERDICT')
    print('=' * 76)
    print()
    print('  Key question: does the FFT approach beat O(sqrt(n))?')
    print()


if __name__ == '__main__':
    run()
