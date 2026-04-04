"""
SESSION 45o — THE ADELIC BARRIER: EACH PRIME IN ITS OWN DIMENSION

The insight: in H, pi is coherent (j/a=0.49) but primes decohere (0.0005)
because 95 primes share 1 extra dimension. Give each prime its OWN dimension.

Use Clifford algebra Cl(n) with n generators e_1, ..., e_n satisfying:
  e_i * e_j + e_j * e_i = -2 * delta_{ij}  (anticommute, square to -1)

Dimension of Cl(n) = 2^n. For n primes, we need 2^n dimensional space.
Too large for computation. Instead, use a TRUNCATED approach:

For K primes p_1, ..., p_K, assign each to a Clifford generator:
  p_k's contribution lives in the e_k direction.

The barrier becomes:
  B = W02 * e_0 - sum_k M_{p_k} * e_k

where e_0 = 1 (scalar, the pi direction) and e_k are the prime directions.

|B|^2 = W02^2 + sum_k M_{p_k}^2  (because e_k are orthogonal!)

THIS IS ALWAYS POSITIVE (sum of squares)!

But wait — this is too easy. The actual barrier is B = W02 - sum M_{p_k},
not B = W02 - sum M_{p_k}*e_k. The adelic version puts each prime in its
own direction, but the REAL barrier has them all in the SAME direction.

The question is: does the adelic decomposition tell us something about
the real barrier? Yes — through the NORM INEQUALITY:

|W02 - sum M_{p_k}| >= |W02| - |sum M_{p_k}|  (reverse triangle inequality)
                      >= |W02| - sum |M_{p_k}|  (triangle inequality)

But also:
|W02 - sum M_{p_k}| = real barrier (can be positive or negative)
|W02*e_0 - sum M_{p_k}*e_k| = sqrt(W02^2 + sum M_{p_k}^2) (always positive!)

The adelic barrier is ALWAYS positive. The real barrier is the 1D projection.
The question: how much does the projection shrink the norm?

PROJECTION FACTOR:
  real_barrier = <B_adelic, e_0> = W02 - sum M_{p_k} * <e_k, e_0> = W02 (wrong!)

Actually, the projection is more subtle. Let me think...

The real barrier is: B_real = W02 - sum_k M_{p_k}  (all in the same direction)
The adelic barrier is: B_adelic = W02 * e_0 - sum_k M_{p_k} * e_k  (each in its own)

|B_adelic|^2 = W02^2 + sum_k M_{p_k}^2  (Pythagorean, orthogonal directions)
|B_real|^2 = (W02 - sum M_{p_k})^2        (all in one direction, can cancel)

The difference:
|B_adelic|^2 - |B_real|^2 = W02^2 + sum M_{p_k}^2 - (W02 - sum M_{p_k})^2
                           = 2 * W02 * sum M_{p_k} - 2 * sum_{j<k} M_{p_j} * M_{p_k}
                           = 2 * sum_k M_{p_k} * (W02 - sum_{j!=k} M_{p_j})

This measures the CROSS-TERMS that exist in the 1D projection but not in
the adelic version. The adelic barrier is always >= real barrier (in norm).

The RATIO |B_real|/|B_adelic| measures how much the 1D projection shrinks
the barrier. If this ratio is bounded below, we can bound the real barrier
using the adelic one.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes, compute_barrier_partial
from session45n_pi_predicts_primes import w02_only, prime_contribution


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45o — THE ADELIC BARRIER')
    print('  Each prime in its own dimension')
    print('=' * 76)

    N = 15

    # ══════════════════════════════════════════════════════════════
    # 1. THE ADELIC vs REAL BARRIER
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. ADELIC BARRIER vs REAL BARRIER')
    print('#' * 76)

    print(f'''
  Real barrier:   B_real = W02 - (Mp_2 + Mp_3 + Mp_5 + ...)
                  All primes in the SAME direction. Can cancel.

  Adelic barrier: B_adelic = W02*e_0 - Mp_2*e_2 - Mp_3*e_3 - Mp_5*e_5 - ...
                  Each prime in its OWN direction. Orthogonal. Can't cancel.

  |B_adelic|^2 = W02^2 + Mp_2^2 + Mp_3^2 + Mp_5^2 + ...
               = W02^2 + sum_p Mp_p^2    (ALWAYS POSITIVE)

  |B_real|^2 = (W02 - sum_p Mp_p)^2      (can be zero if sum = W02)
  ''')

    for lam_sq in [100, 200, 500, 1000, 2000, 5000, 10000]:
        t0 = time.time()
        w02 = w02_only(lam_sq, N)
        primes = list(sieve_primes(int(lam_sq)))

        # Per-prime contributions
        mp_per_prime = []
        for p in primes:
            c = prime_contribution(int(p), lam_sq, N)
            mp_per_prime.append(c)

        mp_total = sum(mp_per_prime)
        b_real = w02 - mp_total

        # Adelic norm
        adelic_norm_sq = w02**2 + sum(c**2 for c in mp_per_prime)
        adelic_norm = np.sqrt(adelic_norm_sq)

        # Ratio
        ratio = abs(b_real) / adelic_norm if adelic_norm > 0 else 0

        # Cross-term sum (what's lost in projection)
        cross_terms = sum(mp_per_prime[j] * mp_per_prime[k]
                         for j in range(len(primes))
                         for k in range(j+1, len(primes)))

        dt = time.time() - t0

        print(f'\n  lam^2 = {lam_sq} ({len(primes)} primes, {dt:.1f}s):')
        print(f'    W02 (pi):        {w02:+.6f}')
        print(f'    sum Mp (primes): {mp_total:+.6f}')
        print(f'    B_real:          {b_real:+.6f}')
        print(f'    |B_adelic|:      {adelic_norm:.6f}')
        print(f'    |B_real|/|B_adelic|: {ratio:.6f}')
        print(f'    Cross-terms:     {cross_terms:+.6f}')
        print(f'    |B_adelic|^2 = {w02**2:.4f} + {sum(c**2 for c in mp_per_prime):.4f}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 2. THE PROJECTION FACTOR
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. THE PROJECTION FACTOR: how much does 1D shrink the barrier?')
    print('#' * 76)

    print(f'\n  The adelic barrier is ALWAYS positive (Pythagorean sum).')
    print(f'  The real barrier is the 1D projection, which CAN be smaller.')
    print(f'  If the projection factor |B_real|/|B_adelic| is bounded below,')
    print(f'  we can bound the real barrier using the adelic one.\n')

    print(f'  {"lam^2":>8s} {"B_real":>12s} {"|B_adelic|":>12s} {"ratio":>10s} '
          f'{"primes":>8s} {"1/sqrt(K)":>10s}')
    print('  ' + '-' * 64)

    ratios = []
    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        w02 = w02_only(lam_sq, N)
        primes = list(sieve_primes(int(lam_sq)))
        K = len(primes)

        mp_per_prime = [prime_contribution(int(p), lam_sq, N) for p in primes]
        mp_total = sum(mp_per_prime)
        b_real = w02 - mp_total

        adelic_sq = w02**2 + sum(c**2 for c in mp_per_prime)
        adelic = np.sqrt(adelic_sq)

        ratio = abs(b_real) / adelic if adelic > 0 else 0
        inv_sqrt_k = 1 / np.sqrt(K) if K > 0 else 0
        ratios.append(ratio)

        print(f'  {lam_sq:>8d} {b_real:>+12.6f} {adelic:>12.6f} {ratio:>10.6f} '
              f'{K:>8d} {inv_sqrt_k:>10.6f}')
    sys.stdout.flush()

    # Does the ratio scale like 1/sqrt(K)?
    # If so: random walk — primes adding incoherently
    print(f'\n  If ratio ~ 1/sqrt(K): primes add like a RANDOM WALK.')
    print(f'  The real barrier ~ |B_adelic| / sqrt(K) ~ bounded.')

    # ══════════════════════════════════════════════════════════════
    # 3. THE RANDOM WALK MODEL
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. THE RANDOM WALK: primes as incoherent steps')
    print('#' * 76)

    print(f'''
  In the adelic picture, the total prime contribution is:
    Mp_total = sum_k Mp_k    (all in one direction = real barrier)

  If the Mp_k have RANDOM SIGNS (incoherent), then by CLT:
    Mp_total ~ sqrt(K) * sigma    where sigma = std of individual Mp_k

  While |B_adelic| = sqrt(W02^2 + sum Mp_k^2) ~ sqrt(K) * sigma_rms

  So: B_real = W02 - Mp_total
     |B_real| ~ |W02 - sqrt(K)*sigma|
     |B_adelic| ~ sqrt(W02^2 + K*sigma^2)

  The ratio |B_real|/|B_adelic| depends on W02/sigma and K.
  ''')

    # Compute statistics of per-prime contributions
    for lam_sq in [500, 2000, 10000]:
        primes = list(sieve_primes(int(lam_sq)))
        mp_per_prime = np.array([prime_contribution(int(p), lam_sq, N) for p in primes])
        w02 = w02_only(lam_sq, N)
        K = len(primes)

        mean_mp = np.mean(mp_per_prime)
        std_mp = np.std(mp_per_prime)
        rms_mp = np.sqrt(np.mean(mp_per_prime**2))

        print(f'\n  lam^2 = {lam_sq} ({K} primes):')
        print(f'    Per-prime: mean = {mean_mp:+.6f}, std = {std_mp:.6f}, rms = {rms_mp:.6f}')
        print(f'    Sum Mp = {np.sum(mp_per_prime):+.6f}')
        print(f'    sqrt(K)*std = {np.sqrt(K)*std_mp:.6f}')
        print(f'    |Sum| / (sqrt(K)*std) = {abs(np.sum(mp_per_prime))/(np.sqrt(K)*std_mp):.4f}')
        print(f'    W02 = {w02:+.6f}')
        print(f'    W02 / (sqrt(K)*rms) = {abs(w02)/(np.sqrt(K)*rms_mp):.4f}')

        # Are the contributions truly incoherent?
        # Check: autocorrelation of mp_per_prime (sorted by prime)
        if K > 10:
            corr_1 = np.corrcoef(mp_per_prime[:-1], mp_per_prime[1:])[0, 1]
            print(f'    Lag-1 autocorrelation: {corr_1:+.4f} '
                  f'(0 = independent, +/-1 = correlated)')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 4. THE ADELIC DECOMPOSITION OF THE BARRIER
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. ADELIC DECOMPOSITION: barrier = pi-term + noise')
    print('#' * 76)

    lam_sq = 2000
    primes = list(sieve_primes(int(lam_sq)))
    mp_per_prime = np.array([prime_contribution(int(p), lam_sq, N) for p in primes])
    w02 = w02_only(lam_sq, N)
    K = len(primes)

    print(f'\n  lam^2 = {lam_sq}, {K} primes')
    print(f'\n  B_adelic^2 = W02^2 + sum Mp_k^2')
    print(f'    W02^2 (pi contribution):        {w02**2:.6f}')
    print(f'    sum Mp_k^2 (prime contribution): {np.sum(mp_per_prime**2):.6f}')
    print(f'    Total |B_adelic|^2:              {w02**2 + np.sum(mp_per_prime**2):.6f}')
    print(f'    Pi fraction of adelic norm:      {w02**2 / (w02**2 + np.sum(mp_per_prime**2)):.6f}')

    # In the adelic picture, the barrier is a VECTOR:
    # B = (W02, -Mp_2, -Mp_3, -Mp_5, ...)
    # Its direction in the (K+1)-dimensional space encodes the
    # relative contributions of pi and each prime.

    # The REAL barrier is the projection onto the (1,1,1,...,1) direction
    # (because in 1D, all components add)
    ones = np.ones(K + 1)
    ones[0] = 1  # W02 direction
    ones[1:] = -1  # prime directions (subtracted)

    B_vec = np.zeros(K + 1)
    B_vec[0] = w02
    B_vec[1:] = mp_per_prime

    B_real = w02 - np.sum(mp_per_prime)  # the actual barrier
    B_proj = np.dot(B_vec, ones) / np.sqrt(K + 1)  # projection onto (1,-1,-1,...,-1)/sqrt(K+1)

    print(f'\n  B_real = W02 - sum Mp = {B_real:+.6f}')
    print(f'  |B_vec| (adelic) = {np.linalg.norm(B_vec):.6f}')
    print(f'  B_vec . direction = {B_real:.6f}')
    print(f'  cos(angle to projection) = {B_real / np.linalg.norm(B_vec):.6f}')

    # ══════════════════════════════════════════════════════════════
    # 5. CAN THE ADELIC PICTURE PROVE POSITIVITY?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. CAN THE ADELIC PICTURE PROVE B_real > 0?')
    print('#' * 76)

    print(f'''
  The adelic barrier |B_adelic| = sqrt(W02^2 + sum Mp_k^2) > 0 always.

  The real barrier B_real = W02 - sum Mp_k can be positive or negative.

  B_real > 0 iff W02 > sum Mp_k.

  In the adelic picture, this is:
    The e_0 component of B_adelic is larger than the SUM of
    all e_k components projected onto the e_0 direction.

  But e_k are ORTHOGONAL to e_0! The projection of e_k onto e_0 is ZERO.

  The real barrier exists because in the PHYSICAL (1D) world, all
  components collapse onto the same axis. The adelic orthogonality
  is BROKEN by the projection to 1D.

  HOWEVER: the adelic picture tells us that the prime contributions
  are INDIVIDUALLY bounded: |Mp_k| is a specific number for each prime.
  The question is whether their SUM (with consistent signs) can exceed W02.

  From our data:
  ''')

    # Check: what fraction of primes have negative Mp_k?
    n_neg = np.sum(mp_per_prime < 0)
    n_pos = np.sum(mp_per_prime > 0)
    print(f'    Primes with Mp_k < 0 (help the barrier): {n_neg}/{K} ({100*n_neg/K:.1f}%)')
    print(f'    Primes with Mp_k > 0 (hurt the barrier):  {n_pos}/{K} ({100*n_pos/K:.1f}%)')
    print(f'    Sum of negative Mp_k: {np.sum(mp_per_prime[mp_per_prime < 0]):+.6f}')
    print(f'    Sum of positive Mp_k: {np.sum(mp_per_prime[mp_per_prime > 0]):+.6f}')
    print(f'    Net: {np.sum(mp_per_prime):+.6f}')
    print(f'    W02: {w02:+.6f}')
    print(f'    B_real = W02 - net = {B_real:+.6f}')

    # The sign pattern
    print(f'\n  Sign pattern of per-prime contributions (first 30):')
    print(f'  ', end='')
    for i in range(min(30, K)):
        print('+' if mp_per_prime[i] > 0 else '-', end='')
    print()

    # Cumulative: how the sign balance evolves
    print(f'\n  Cumulative sign balance:')
    cum = 0
    print(f'  {"primes":>8s} {"cum sum":>12s} {"W02-cum":>12s} {"positive?":>10s}')
    print('  ' + '-' * 46)
    for i in range(K):
        cum += mp_per_prime[i]
        if i < 15 or i % 50 == 0 or i == K-1:
            b = w02 - cum
            print(f'  {i+1:>8d} {cum:>+12.6f} {b:>+12.6f} '
                  f'{"YES" if b > 0 else "no":>10s}')

    # ══════════════════════════════════════════════════════════════
    # 6. THE INEQUALITY THAT WOULD PROVE RH
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. THE INEQUALITY THAT WOULD PROVE RH')
    print('#' * 76)

    print(f'''
  In the adelic picture:

  |B_adelic|^2 = W02^2 + sum_k Mp_k^2     (Pythagorean, always positive)
  |B_real|^2   = (W02 - sum_k Mp_k)^2     (1D projection, can be zero)

  The DIFFERENCE is the cross-term:
  |B_adelic|^2 - |B_real|^2 = 2 * sum_{{j<k}} Mp_j * Mp_k

  This cross-term represents the INTERFERENCE between primes.
  In the adelic world: no interference (orthogonal).
  In the real world: full interference (same direction).

  RH is equivalent to: the interference is not TOO destructive.
  Specifically: sum_{{j<k}} Mp_j * Mp_k < (|B_adelic|^2 - 0) / 2

  Or equivalently: the primes don't conspire to cancel W02.

  From our data at lam^2 = {lam_sq}:
    |B_adelic|^2 = {w02**2 + np.sum(mp_per_prime**2):.6f}
    |B_real|^2   = {B_real**2:.6f}
    Cross-term   = {(w02**2 + np.sum(mp_per_prime**2) - B_real**2)/2:.6f}
    Cross/Adelic^2 = {(w02**2 + np.sum(mp_per_prime**2) - B_real**2)/(2*(w02**2 + np.sum(mp_per_prime**2))):.6f}

  The cross-term is {(w02**2 + np.sum(mp_per_prime**2) - B_real**2)/(2*(w02**2 + np.sum(mp_per_prime**2)))*100:.1f}% of |B_adelic|^2.
  ''')

    cross = (w02**2 + np.sum(mp_per_prime**2) - B_real**2) / 2
    print(f'  The interference between primes accounts for {cross:.4f}')
    print(f'  out of the adelic norm {w02**2 + np.sum(mp_per_prime**2):.4f}.')
    print(f'  The 1D projection preserves {B_real**2/(w02**2 + np.sum(mp_per_prime**2))*100:.4f}%')
    print(f'  of the adelic norm squared.')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45o FINAL SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE ADELIC BARRIER:

  In the adelic world (each prime in its own dimension):
    |B_adelic|^2 = W02^2 + sum Mp_k^2 > 0  ALWAYS.
    Barrier positivity is AUTOMATIC. Primes can't cancel pi.

  In the real world (all primes in one dimension):
    B_real = W02 - sum Mp_k  (can be positive or negative)
    Positivity = RH. The 1D projection CAN destroy the barrier.

  THE GAP between adelic and real is the CROSS-TERM:
    2 * sum_{{j<k}} Mp_j * Mp_k = prime-prime interference

  RH says: this interference is bounded. The primes never conspire
  to perfectly cancel the archimedean (pi) contribution.

  THE SESSION 45 ARC (a through o):
    45a-d: Wick rotation, spectral shift, completeness reformulation
    45e-f: Quaternionic barrier, component separation
    45g-h: Fueter zeros, |F(rho)| = (2/gamma)|zeta'|, interleaving
    45i:   QFT — primes and pi parallel in low modes (PNT)
    45j:   Pi as archimedean prime: 1000x j-separation
    45k:   Topological argument: v=0 on CL, departure rate = Fueter
    45L:   w-function, w=0 and u=0 curves don't intersect
    45m:   Pi is a SPHERE in H, coherent signal vs incoherent primes
    45n:   Pi predicts primes (bound, explicit formula, resonance)
    45o:   Adelic barrier: positivity automatic in infinite dimensions,
           lost in 1D projection through prime-prime interference.

  RH = the interference between primes is bounded.
  Pi sets the envelope. Primes fill it incoherently.
  The barrier is the margin between envelope and filling.
  In the adeles, the margin is infinite-dimensional and always positive.
  In one dimension, it's the Riemann Hypothesis.
''')

    print('=' * 76)
    print('  SESSION 45 COMPLETE (a through o)')
    print('=' * 76)
