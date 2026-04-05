"""
SESSION 51b -- CONDITIONAL CRAMER MODEL

Thread 2 of Session 51, running in parallel with 51a.

Context from Session 43:
  Pure Cramer (all integers i.i.d. prime with prob 1/log n): E[drain] ~= 0,
  but actual drain ~= 0.22. The 0.22 comes from specific positions of
  SMALL primes (2, 3, 5, 7, ...) which deviate from Cramer.

Question: how much of the actual drain comes from each "shell" of primes?
If shell p<=20 accounts for 0.22 and shell p>20 accounts for ~0, then
the drain is "dominated by small primes" and can be bounded by direct
computation of those ~8 primes alone.

Protocol:
  1. Compute actual drain(L) at several L values using real primes.
  2. For each threshold K in {5, 10, 20, 50, 100, 500}:
     a. Fix primes p <= K at their actual positions.
     b. Randomize integers > K under Cramer prob 1/log n.
     c. Compute drain for each realization (n_trials = 200).
     d. Report mean drain and std dev.
  3. As K grows, conditional drain mean should approach the actual drain.
     How fast it converges tells us how many small primes "matter".

  4. Derived bound: if actual drain is within the K = 20 confidence band,
     the drain is "explained by small primes". Then:
        |drain(L)| <= |contrib from primes p<=20| + |tail bound from large
        primes via PNT|
     which might give a tight enough bound to beat the margin.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes, compute_barrier_partial
from session42j_margin_vs_drain import mprime_pnt_integral


def compute_mp_from_prime_list(prime_list, lam_sq, N):
    """
    Recompute M_prime Rayleigh quotient on conjugate-Poisson kernel for a
    given list of primes (possibly a Cramer random realization).
    """
    L = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    nm_diff = ns[:, None] - ns[None, :]
    M = np.zeros((2 * N + 1, 2 * N + 1))

    for p in prime_list:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            weight = logp * pk ** (-0.5)
            yk = np.log(pk)
            sin_arr = np.sin(2 * np.pi * ns * yk / L)
            cos_arr = np.cos(2 * np.pi * ns * yk / L)
            diag = 2 * (L - yk) / L * cos_arr
            np.fill_diagonal(M, M.diagonal() + weight * diag)
            sin_diff = sin_arr[None, :] - sin_arr[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                off = sin_diff / (np.pi * nm_diff)
            np.fill_diagonal(off, 0.0)
            M += weight * off
            pk *= int(p)

    M = (M + M.T) / 2
    return float(w_hat @ M @ w_hat)


def cramer_trial_above_K(K, lam_sq, rng):
    """
    Generate a random realization of "primes" > K, keeping primes
    <= K fixed at actual values. The random subset has size matching
    the actual count of primes in (K, lam_sq], and integers n are
    sampled with probability proportional to 1/log(n) (the Cramer
    conditional density given we know how many primes there are).

    This eliminates the count bias of the raw Cramer model.
    """
    actual = sieve_primes(int(lam_sq))
    small = [int(p) for p in actual if p <= K]
    n_large_actual = int(sum(1 for p in actual if p > K))
    if n_large_actual == 0:
        return small

    n_arr = np.arange(K + 1, int(lam_sq) + 1)
    if len(n_arr) == 0:
        return small
    log_n = np.log(n_arr.astype(float))
    weights = 1.0 / log_n
    weights = weights / weights.sum()
    # Sample n_large_actual distinct integers without replacement,
    # weighted by 1/log(n).
    chosen_idx = rng.choice(len(n_arr), size=n_large_actual, replace=False, p=weights)
    large = sorted(int(x) for x in n_arr[chosen_idx])
    return small + large


def run():
    print()
    print('#' * 76)
    print('  SESSION 51b -- CONDITIONAL CRAMER DRAIN DECOMPOSITION')
    print('#' * 76)

    rng = np.random.default_rng(seed=20260405)

    lam_sq = 2000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    print(f'\n  Fixed lam^2 = {lam_sq}, L = {L:.4f}, N = {N}')
    print()
    sys.stdout.flush()

    # Actual drain for reference
    actual_primes = sieve_primes(int(lam_sq))
    print(f'  {len(actual_primes)} actual primes up to {lam_sq}')
    actual_mp_all = compute_mp_from_prime_list(actual_primes, lam_sq, N)
    pnt_mp = mprime_pnt_integral(lam_sq, N)
    actual_drain = actual_mp_all - pnt_mp
    print(f'  Actual M_prime (via direct sum): {actual_mp_all:+.6f}')
    print(f'  PNT smooth M_prime:              {pnt_mp:+.6f}')
    print(f'  Actual drain = Mp - PNT:         {actual_drain:+.6f}')
    print()
    sys.stdout.flush()

    # Sanity vs compute_barrier_partial
    sanity = compute_barrier_partial(lam_sq, N)
    print(f'  Sanity vs session41g compute_barrier_partial mprime: '
          f'{sanity["mprime"]:+.6f}  (diff {actual_mp_all - sanity["mprime"]:+.2e})')
    print()
    sys.stdout.flush()

    # Sweep K
    K_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 50, 100, 500]
    n_trials = 50

    print(f'  Sweeping K (fix first primes <= K at actual, randomize p > K)')
    print(f'  {n_trials} trials per K')
    print()
    print(f'  {"K":>5} {"n_small":>8} {"mean drain":>14} {"std drain":>14} '
          f'{"mean gap":>14} {"|actual - mean|":>16}')
    print('  ' + '-' * 80)
    sys.stdout.flush()

    results = {}
    for K in K_values:
        small = [p for p in actual_primes if p <= K]
        n_small = len(small)

        trial_drains = []
        for t in range(n_trials):
            prime_realization = cramer_trial_above_K(K, lam_sq, rng)
            mp_trial = compute_mp_from_prime_list(prime_realization, lam_sq, N)
            trial_drains.append(mp_trial - pnt_mp)

        trial_drains = np.array(trial_drains)
        mean_d = float(trial_drains.mean())
        std_d = float(trial_drains.std())
        diff_from_actual = abs(mean_d - actual_drain)
        gap = actual_drain - mean_d

        results[K] = (n_small, mean_d, std_d, trial_drains)
        print(f'  {K:>5d} {n_small:>8d} {mean_d:>+14.6f} {std_d:>14.6f} '
              f'{gap:>+14.6f} {diff_from_actual:>16.6f}', flush=True)

    print()
    print('=' * 76)
    print('  INTERPRETATION')
    print('=' * 76)
    print(f'''
  Actual drain at lam^2 = {lam_sq} is {actual_drain:+.6f}.

  As K grows, we fix more small primes at their actual positions. The
  conditional-Cramer mean drain approaches the actual drain.

  If the mean converges to the actual drain at a small K (say K <= 20),
  then the actual drain is dominated by the first few primes' deviation
  from the Cramer model. That would mean:

    |drain(L)| <= |contribution from p <= 20| + O(std dev of Cramer tail)

  which is a FINITE, explicit quantity. Combined with margin(L) monotone
  to 0.269, this could yield an unconditional bound |drain| < margin.

  If the mean only converges at large K, the drain involves fine
  correlations across many primes and the approach fails.

  Look at the "|actual - mean|" column above: where does it drop below
  the std deviation, indicating the remaining randomness accounts for
  the discrepancy?
''')


if __name__ == '__main__':
    run()
