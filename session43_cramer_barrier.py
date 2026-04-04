"""
SESSION 43 — CRAMER MODEL: IS THE BARRIER POSITIVE FOR RANDOM PRIMES?

Cramer's model: each integer n >= 2 is "prime" independently with
probability 1/log(n). The actual primes are one realization.

Questions:
1. What is E[drain] under the Cramer model?
2. What is Var[drain]?
3. Is E[barrier] = margin - E[drain] > 0?
4. How many sigma is the barrier above zero?
5. How does the actual drain compare to the Cramer prediction?

If the barrier is robustly positive for random primes, and the actual
primes aren't too different from random, we have a probabilistic proof.

The key insight: the drain = sum_p f(p) - integral f(t) dt.
Under Cramer: E[sum_p f(p)] = sum_n f(n)/log(n) ~ integral f(t)/log(t) dt * log(t)
                              = integral f(t) dt  (by PNT heuristic)
So E[drain] ~ 0. The drain's expectation IS zero under Cramer!

The actual drain of ~0.22 comes from the SPECIFIC positions of small primes
(2, 3, 5, 7) which deviate from the Cramer prediction. These are FIXED —
not random — so the Cramer model doesn't capture them.

Refined approach: condition on the first K primes being fixed at their
actual positions, and randomize only the large primes.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes


def cramer_barrier_simulation(lam_sq, n_trials=1000, N_basis=None):
    """
    Simulate the barrier under Cramer's random prime model.

    For each trial:
    1. Generate random "primes" (each n prime with prob 1/log(n))
    2. Compute M_prime from these random primes
    3. Compute the barrier (W02 - M_prime - M_diag - M_alpha)

    Since M_diag and M_alpha don't depend on primes, we only need
    to simulate M_prime.

    Returns: array of simulated drain values.
    """
    L = np.log(lam_sq)
    if N_basis is None:
        N_basis = max(15, round(6 * L))
    dim = 2 * N_basis + 1
    ns = np.arange(-N_basis, N_basis + 1, dtype=float)

    w = ns / (L**2 + (4*np.pi)**2 * ns**2)
    w[N_basis] = 0.0
    w_hat = w / np.linalg.norm(w)
    nm_diff = ns[:, None] - ns[None, :]

    # Precompute the per-integer contribution function
    # For each integer n, if it's "prime", it contributes:
    # log(n)/sqrt(n) * <w_hat, Q_{log(n)/L}, w_hat>
    # (ignoring prime powers k >= 2 for simplicity)

    max_n = int(lam_sq)
    log_n = np.log(np.arange(2, max_n + 1, dtype=float))  # log(2), log(3), ...
    sqrt_n = np.sqrt(np.arange(2, max_n + 1, dtype=float))

    # Precompute F(n) = <w_hat, Q_{log(n)/L}, w_hat> for each n
    F_values = np.zeros(max_n - 1)
    for idx in range(max_n - 1):
        n = idx + 2
        y = np.log(n)
        if y >= L:
            continue
        sin_arr = np.sin(2*np.pi*ns*y/L)
        cos_arr = np.cos(2*np.pi*ns*y/L)
        diag = 2*(L-y)/L * np.sum(w_hat**2 * cos_arr)
        sin_diff = sin_arr[None,:] - sin_arr[:,None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off, 0.0)
        off_val = w_hat @ off @ w_hat
        F_values[idx] = diag + off_val

    # Weight for each integer: log(n)/sqrt(n) * F(n)
    weights = log_n / sqrt_n * F_values

    # Cramer probability: P(n is prime) = 1/log(n)
    probs = 1.0 / log_n
    probs = np.clip(probs, 0, 1)

    # Actual primes for comparison
    actual_primes = sieve_primes(max_n)
    actual_mask = np.zeros(max_n - 1, dtype=bool)
    for p in actual_primes:
        if p >= 2 and p <= max_n:
            actual_mask[p - 2] = True

    actual_mprime = np.sum(weights[actual_mask])

    # PNT integral approximation
    pnt_integral = np.sum(weights * probs * log_n)  # E[sum] under Cramer with correct weighting

    # Actually, under Cramer: E[sum_p f(p)] = sum_n P(n prime) * f(n)
    #                                       = sum_n (1/log n) * log(n)/sqrt(n) * F(n)
    #                                       = sum_n F(n)/sqrt(n)
    # Which is approximately integral F(t)/sqrt(t) dt = PNT_integral
    cramer_expected = np.sum(weights * probs * log_n)  # sum_n (1/log n) * (log n / sqrt n) * F(n) = sum F/sqrt
    # Hmm, let me redo this:
    # weight[n] = log(n)/sqrt(n) * F(n)
    # Under Cramer: E[Mp] = sum_n P(n prime) * weight[n] = sum_n weight[n] / log(n)
    cramer_expected_mp = np.sum(weights / log_n)  # = sum_n F(n)/sqrt(n)

    # Simulate
    drain_samples = np.zeros(n_trials)
    mp_samples = np.zeros(n_trials)

    for trial in range(n_trials):
        # Random primes
        random_mask = np.random.random(max_n - 1) < probs
        mp = np.sum(weights[random_mask])
        mp_samples[trial] = mp
        drain_samples[trial] = mp - cramer_expected_mp

    return {
        'actual_mp': actual_mprime,
        'cramer_expected_mp': cramer_expected_mp,
        'actual_drain': actual_mprime - cramer_expected_mp,
        'drain_mean': drain_samples.mean(),
        'drain_std': drain_samples.std(),
        'drain_min': drain_samples.min(),
        'drain_max': drain_samples.max(),
        'mp_mean': mp_samples.mean(),
        'mp_std': mp_samples.std(),
        'n_trials': n_trials,
        'drain_samples': drain_samples,
    }


def cramer_conditioned(lam_sq, P_fixed=50, n_trials=1000, N_basis=None):
    """
    Cramer model CONDITIONED on small primes being fixed.

    Fix the primes p <= P_fixed at their actual positions.
    Randomize only n > P_fixed with P(n prime) = 1/log(n).

    This captures the dominant source of the drain (small prime excess)
    while randomizing the large-prime tail.
    """
    L = np.log(lam_sq)
    if N_basis is None:
        N_basis = max(15, round(6 * L))
    dim = 2 * N_basis + 1
    ns = np.arange(-N_basis, N_basis + 1, dtype=float)

    w = ns / (L**2 + (4*np.pi)**2 * ns**2)
    w[N_basis] = 0.0
    w_hat = w / np.linalg.norm(w)
    nm_diff = ns[:, None] - ns[None, :]

    max_n = int(lam_sq)
    log_n = np.log(np.arange(2, max_n + 1, dtype=float))
    sqrt_n = np.sqrt(np.arange(2, max_n + 1, dtype=float))

    # Precompute F and weights
    F_values = np.zeros(max_n - 1)
    for idx in range(max_n - 1):
        n = idx + 2
        y = np.log(n)
        if y >= L:
            continue
        sin_arr = np.sin(2*np.pi*ns*y/L)
        cos_arr = np.cos(2*np.pi*ns*y/L)
        diag = 2*(L-y)/L * np.sum(w_hat**2 * cos_arr)
        sin_diff = sin_arr[None,:] - sin_arr[:,None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off, 0.0)
        off_val = w_hat @ off @ w_hat
        F_values[idx] = diag + off_val

    weights = log_n / sqrt_n * F_values
    probs = np.clip(1.0 / log_n, 0, 1)

    # Split: fixed (n <= P_fixed) and random (n > P_fixed)
    actual_primes = sieve_primes(max_n)
    fixed_mask = np.zeros(max_n - 1, dtype=bool)
    for p in actual_primes:
        if 2 <= p <= P_fixed:
            fixed_mask[p - 2] = True

    random_range = np.arange(max_n - 1) >= (P_fixed - 1)  # indices for n > P_fixed

    # Fixed contribution
    fixed_mp = np.sum(weights[fixed_mask])

    # Expected random contribution
    random_expected = np.sum(weights[random_range] / log_n[random_range])

    # Actual full Mp
    actual_mask = np.zeros(max_n - 1, dtype=bool)
    for p in actual_primes:
        if p >= 2:
            actual_mask[p - 2] = True
    actual_mp = np.sum(weights[actual_mask])

    # Simulate random part
    drain_samples = np.zeros(n_trials)
    for trial in range(n_trials):
        random_mask = np.zeros(max_n - 1, dtype=bool)
        random_draws = np.random.random(np.sum(random_range)) < probs[random_range]
        random_mask[random_range] = random_draws
        mp = fixed_mp + np.sum(weights[random_mask])
        drain_samples[trial] = mp - (fixed_mp + random_expected)

    return {
        'actual_mp': actual_mp,
        'fixed_mp': fixed_mp,
        'random_expected': random_expected,
        'total_expected': fixed_mp + random_expected,
        'actual_drain': actual_mp - (fixed_mp + random_expected),
        'drain_mean': drain_samples.mean(),
        'drain_std': drain_samples.std(),
        'n_trials': n_trials,
        'P_fixed': P_fixed,
    }


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  SESSION 43 — CRAMER MODEL BARRIER')
    print('#' * 72)

    # ── Part 1: Pure Cramer (all primes random) ──
    print('\n  PART 1: Pure Cramer model')
    print('  ' + '=' * 60)

    for lam_sq in [500, 1000, 5000, 10000]:
        t0 = time.time()
        r = cramer_barrier_simulation(lam_sq, n_trials=2000)
        dt = time.time() - t0

        print(f'\n  lam^2 = {lam_sq}:  ({dt:.0f}s)')
        print(f'    Actual Mp:      {r["actual_mp"]:+.6f}')
        print(f'    Cramer E[Mp]:   {r["cramer_expected_mp"]:+.6f}')
        print(f'    Actual drain:   {r["actual_drain"]:+.6f}')
        print(f'    Cramer E[drain]: {r["drain_mean"]:+.6f}')
        print(f'    Cramer std[drain]: {r["drain_std"]:.6f}')
        print(f'    Actual drain in sigma: {(r["actual_drain"] - r["drain_mean"]) / r["drain_std"]:.2f}')

    # ── Part 2: Conditioned Cramer (fix small primes) ──
    print('\n\n  PART 2: Conditioned Cramer (fix primes <= P)')
    print('  ' + '=' * 60)

    for lam_sq in [1000, 5000, 10000]:
        for P_fixed in [10, 50, 100]:
            t0 = time.time()
            r = cramer_conditioned(lam_sq, P_fixed=P_fixed, n_trials=2000)
            dt = time.time() - t0

            print(f'\n  lam^2={lam_sq}, P_fixed={P_fixed}:  ({dt:.0f}s)')
            print(f'    Fixed Mp (p<={P_fixed}):  {r["fixed_mp"]:+.6f}')
            print(f'    Random expected:    {r["random_expected"]:+.6f}')
            print(f'    Total expected:     {r["total_expected"]:+.6f}')
            print(f'    Actual Mp:          {r["actual_mp"]:+.6f}')
            print(f'    Actual drain:       {r["actual_drain"]:+.6f}')
            print(f'    Simulated drain std: {r["drain_std"]:.6f}')
            print(f'    Actual in sigma:    {r["actual_drain"] / r["drain_std"]:.2f}' if r["drain_std"] > 0 else '')

    # ── Part 3: What does this mean for the proof? ──
    print('\n\n  PART 3: Implications')
    print('  ' + '=' * 60)

    # The margin is ~0.264. The drain is ~0.22.
    # Under pure Cramer: E[drain] = 0, so E[barrier] = margin = 0.264.
    # Under conditioned Cramer: E[drain] = actual_drain from small primes.
    # The random part has std ~ some value.

    # For the proof: need P(drain > margin) < epsilon
    # Under conditioned Cramer with P_fixed=100:
    # drain = fixed_drain + random_drain
    # fixed_drain is deterministic
    # random_drain has mean ~0 and small std

    r = cramer_conditioned(10000, P_fixed=100, n_trials=5000)
    margin = 0.264

    print(f'\n  At lam^2=10000, P_fixed=100:')
    print(f'    Margin:             {margin:.6f}')
    print(f'    Fixed drain (p<=100): deterministic, included in expected')
    print(f'    Random drain std:   {r["drain_std"]:.6f}')
    print(f'    Actual total drain: {r["actual_drain"]:+.6f}')
    print(f'    Barrier = margin - drain')
    print(f'    Expected barrier:   {margin - r["actual_drain"]:.6f}')
    print(f'    Barrier std:        {r["drain_std"]:.6f}')
    z = (margin - r["actual_drain"]) / r["drain_std"] if r["drain_std"] > 0 else float('inf')
    print(f'    Z-score:            {z:.2f}')
    print(f'    P(barrier < 0):     exp(-{z**2/2:.1f}) ~ {np.exp(-z**2/2):.2e}')

    print('\n' + '#' * 72)
    print('  SESSION 43 COMPLETE')
    print('#' * 72)
