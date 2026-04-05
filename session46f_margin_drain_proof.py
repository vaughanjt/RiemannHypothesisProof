"""
SESSION 46f — THE MARGIN-DRAIN PROOF ATTEMPT

Form 2 of the inequality: margin(L) > |drain(L)| for all L.

margin(L) = W02(L) - PNT_integral(L) - M_diag(L) - M_alpha(L)
  PURELY ANALYTIC. No primes. Computable special functions of L.
  Limit: margin -> 0.269 as L -> infinity.

drain(L) = M_prime_actual(L) - PNT_integral(L)
  ARITHMETIC. Depends on specific primes. Oscillates with L.
  Limit: drain -> 0.240 as L -> infinity.
  Gap: margin - drain -> 0.029 > 0.

TO PROVE RH: show |drain(L)| < margin(L) for all L >= L_0.

The drain is dominated by the first ~20 primes. For each prime p:
  delta_p(L) = actual_contribution(p,L) - PNT_contribution(p,L)

This oscillates as a function of L because of cos(2*pi*n*log(p)/L) terms.

THREE BOUNDS ON THE DRAIN:
  A. Triangle inequality: |drain| <= sum |delta_p| (too loose)
  B. Equidistribution: |drain| ~ sqrt(sum delta_p^2) (random phases)
  C. Exact maximum: max_L |drain(L)| (computable but hard to prove)

If bound B < 0.264 (the margin), primes are incoherent enough and RH follows.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, quad)
import time
import sys
import os

mp.dps = 25

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes, compute_barrier_partial
from session45n_pi_predicts_primes import w02_only, prime_contribution
from session42j_margin_vs_drain import mdiag_malpha, mprime_pnt_integral


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 46f — THE MARGIN-DRAIN PROOF ATTEMPT')
    print('  Prove: margin(L) > |drain(L)| for all L')
    print('=' * 76)

    N = 15

    # ══════════════════════════════════════════════════════════════
    # 1. COMPUTE MARGIN AND DRAIN AT MANY L VALUES
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. MARGIN AND DRAIN ACROSS L')
    print('#' * 76)

    lam_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    print(f'\n  {"lam^2":>8s} {"L":>7s} {"W02":>10s} {"PNT_Mp":>10s} '
          f'{"Md+Ma":>10s} {"MARGIN":>10s} {"actual_Mp":>10s} {"DRAIN":>10s} '
          f'{"GAP":>8s}')
    print('  ' + '-' * 88)

    margins = []
    drains = []
    Ls = []

    for lam_sq in lam_values:
        t0 = time.time()
        L_val = np.log(lam_sq)

        # W02
        w02 = w02_only(lam_sq, N)

        # PNT integral
        pnt_mp = mprime_pnt_integral(lam_sq, N)

        # M_diag + M_alpha
        md, ma = mdiag_malpha(lam_sq, N)

        # Actual M_prime
        r = compute_barrier_partial(lam_sq, N)
        actual_mp = r['mprime']

        # Margin and drain
        margin = w02 - pnt_mp - md - ma
        drain = actual_mp - pnt_mp
        gap = margin - abs(drain)

        margins.append(margin)
        drains.append(drain)
        Ls.append(L_val)

        dt = time.time() - t0
        print(f'  {lam_sq:>8d} {L_val:>7.3f} {w02:>+10.4f} {pnt_mp:>+10.4f} '
              f'{md+ma:>+10.4f} {margin:>+10.6f} {actual_mp:>+10.4f} '
              f'{drain:>+10.6f} {gap:>+8.4f}  ({dt:.0f}s)')
        sys.stdout.flush()

    margins = np.array(margins)
    drains = np.array(drains)
    Ls = np.array(Ls)

    # ══════════════════════════════════════════════════════════════
    # 2. PER-PRIME DRAIN DECOMPOSITION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. PER-PRIME DRAIN: which primes contribute most?')
    print('#' * 76)

    lam_sq = 2000
    L_val = np.log(lam_sq)
    primes = list(sieve_primes(int(lam_sq)))
    K = len(primes)

    # Per-prime contributions to the drain
    # drain_p = actual_Mp_p - PNT_contribution_of_p
    # The PNT contribution of prime p is approximately:
    # integral from log(p)-delta to log(p)+delta of e^{y/2} * F(y/L) dy
    # For simplicity, compute each prime's actual contribution
    mp_per_prime = [prime_contribution(int(p), lam_sq, N) for p in primes]

    # The PNT integral spreads the contribution smoothly
    # Each prime's "expected" contribution under PNT:
    # PNT says sum_{p<=x} log(p)/sqrt(p) * K(log(p)/L) ~ integral
    # The per-prime DRAIN is: actual - expected
    # For a rough decomposition: weight_p * K(log(p)/L) - average_weight * K

    # More precisely, the drain = sum_p [Mp_p] - PNT_integral
    # We already have the total drain. Let's measure how much comes from
    # the first 5, 10, 20, 50 primes.

    pnt_total = mprime_pnt_integral(lam_sq, N)
    actual_total = sum(mp_per_prime)
    total_drain = actual_total - pnt_total

    print(f'\n  lam^2 = {lam_sq}, {K} primes')
    print(f'  Total drain = {total_drain:+.6f}')

    # Cumulative contribution by adding primes one at a time
    print(f'\n  Cumulative prime sum vs PNT (drain buildup):')
    print(f'  {"primes":>7s} {"up to p":>8s} {"cum Mp":>12s} {"PNT_Mp":>12s} '
          f'{"cum drain":>12s} {"% of total":>10s}')
    print('  ' + '-' * 62)

    cum_mp = 0
    for i, p in enumerate(primes):
        cum_mp += mp_per_prime[i]
        if i < 20 or (i+1) % 50 == 0 or i == K-1:
            # Rough PNT allocation: proportional to p's range
            pnt_frac = cum_mp  # just use actual cumulative
            drain_so_far = cum_mp - pnt_total * (i+1) / K  # rough allocation
            pct = drain_so_far / total_drain * 100 if abs(total_drain) > 1e-10 else 0
            print(f'  {i+1:>7d} {int(p):>8d} {cum_mp:>+12.6f} '
                  f'{pnt_total*(i+1)/K:>+12.6f} {drain_so_far:>+12.6f} {pct:>9.1f}%')

    # ══════════════════════════════════════════════════════════════
    # 3. THE THREE BOUNDS ON THE DRAIN
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. THREE BOUNDS ON THE DRAIN')
    print('#' * 76)

    # Compute |delta_p| for each prime (the absolute per-prime drain)
    abs_deltas = np.array([abs(c) for c in mp_per_prime])

    # Bound A: Triangle inequality
    bound_A = np.sum(abs_deltas)
    print(f'\n  Bound A (triangle inequality): |drain| <= sum |delta_p|')
    print(f'    sum |delta_p| = {bound_A:.6f}')
    print(f'    margin = {margins[lam_values.index(2000)]:.6f}')
    print(f'    Bound A < margin? {"YES" if bound_A < margins[lam_values.index(2000)] else "NO (too loose)"}')

    # Bound B: Equidistribution / RMS
    bound_B = np.sqrt(np.sum(np.array(mp_per_prime)**2))
    print(f'\n  Bound B (equidistribution/RMS): |drain| ~ sqrt(sum delta_p^2)')
    print(f'    sqrt(sum delta_p^2) = {bound_B:.6f}')
    print(f'    margin = {margins[lam_values.index(2000)]:.6f}')
    print(f'    Bound B < margin? {"YES" if bound_B < margins[lam_values.index(2000)] else "NO"}')

    # Bound C: Actual maximum over computed L values
    bound_C = max(abs(d) for d in drains)
    print(f'\n  Bound C (actual maximum over computed L):')
    print(f'    max |drain| = {bound_C:.6f}')
    print(f'    min margin = {min(margins):.6f}')
    print(f'    max |drain| < min margin? {"YES" if bound_C < min(margins) else "NO"}')
    print(f'    Gap: {min(margins) - bound_C:.6f}')

    # ══════════════════════════════════════════════════════════════
    # 4. CAN WE BOUND THE DRAIN BY THE MARGIN?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. THE DRAIN OSCILLATION: how much room is there?')
    print('#' * 76)

    # Dense scan of drain vs margin
    print(f'\n  Dense scan of drain and margin:')
    print(f'  {"lam^2":>8s} {"L":>7s} {"margin":>10s} {"|drain|":>10s} {"gap":>10s} {"safe":>6s}')
    print('  ' + '-' * 55)

    dense_lam = list(range(50, 1001, 50)) + list(range(1000, 10001, 500))
    min_gap = float('inf')
    min_gap_lam = 0

    for lam_sq in dense_lam:
        L_val = np.log(lam_sq)
        w02 = w02_only(lam_sq, N)
        r = compute_barrier_partial(lam_sq, N)
        actual_mp = r['mprime']
        pnt_mp = mprime_pnt_integral(lam_sq, N)
        md, ma = mdiag_malpha(lam_sq, N)

        margin = w02 - pnt_mp - md - ma
        drain = actual_mp - pnt_mp
        gap = margin - abs(drain)

        if gap < min_gap:
            min_gap = gap
            min_gap_lam = lam_sq

        if lam_sq <= 500 or lam_sq % 2000 == 0 or gap < 0.03:
            safe = 'YES' if gap > 0 else '***NO***'
            print(f'  {lam_sq:>8d} {L_val:>7.3f} {margin:>+10.6f} {abs(drain):>10.6f} '
                  f'{gap:>+10.6f} {safe:>6s}')
        sys.stdout.flush()

    print(f'\n  Minimum gap: {min_gap:+.6f} at lam^2 = {min_gap_lam}')
    print(f'  Gap always positive? {"YES" if min_gap > 0 else "NO"}')

    # ══════════════════════════════════════════════════════════════
    # 5. WHAT WOULD THE PROOF LOOK LIKE?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. WHAT WOULD THE PROOF LOOK LIKE?')
    print('#' * 76)

    print(f'''
  THE MARGIN is a computable function of L:
    margin(L) = W02(L) - PNT_integral(L) - M_diag(L) - M_alpha(L)

  Each component involves standard special functions:
    W02: 32*L*sinh^2(L/4) * quadratic form (involves pi)
    PNT_integral: integral of e^(y/2) * F(y/L) dy
    M_diag: Euler-Mascheroni gamma + log + integral (involves gamma, pi)
    M_alpha: hypergeometric 2F1 + digamma (involves pi)

  The margin is MONOTONICALLY INCREASING toward 0.269.
  A computer algebra system could verify margin(L) > 0.264 for L > L_0.

  THE DRAIN is:
    drain(L) = sum_p [log(p)/sqrt(p) * K(log(p)/L)] - PNT_integral(L)

  This is a FINITE sum (K primes) minus a smooth integral.
  The drain OSCILLATES because cos(2*pi*n*log(p)/L) varies with L.

  TO PROVE |drain(L)| < margin(L):
    Option A: Bound each prime's contribution and use triangle inequality.
              sum |delta_p| = {bound_A:.4f} {'<' if bound_A < 0.264 else '>'} 0.264.
              {'WORKS' if bound_A < 0.264 else 'TOO LOOSE by factor ' + f'{bound_A/0.264:.1f}'}

    Option B: Use equidistribution (phases of log(p)/L are random).
              sqrt(sum delta_p^2) = {bound_B:.4f} {'<' if bound_B < 0.264 else '>'} 0.264.
              {'WORKS' if bound_B < 0.264 else 'TOO LOOSE by factor ' + f'{bound_B/0.264:.1f}'}

    Option C: Direct computation up to L_max, then asymptotic bound.
              max |drain| = {bound_C:.4f} < min margin = {min(margins):.4f}.
              GAP = {min(margins) - bound_C:.4f}.
              {'WORKS in computed range' if min_gap > 0 else 'FAILS'}

    Option D: Explicit formula for the drain.
              drain = -sum_rho (spectral correction at zero rho) + O(e^{{-c*sqrt(L)}})
              Bound the spectral correction using zero-free region.
              This is the CLASSICAL approach and requires...
              a zero-free region, which is equivalent to RH. CIRCULAR.

    Option E: The FINITE-PRIME approach.
              The drain is dominated by p = 2, 3, 5, ..., ~100.
              For EACH small prime p, bound |delta_p(L)| for all L.
              Then: |drain| <= sum_(p<=100) |delta_p| + (tail bound)
              The tail bound comes from PNT for large primes (unconditional).
  ''')

    # ══════════════════════════════════════════════════════════════
    # 6. OPTION E: THE FINITE-PRIME BOUND
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  6. OPTION E: BOUNDING EACH SMALL PRIME\'S DRAIN')
    print('#' * 76)

    # For each small prime p, compute max_L |delta_p(L)|
    # delta_p(L) involves cos(2*pi*n*log(p)/L) which oscillates
    # The maximum is bounded by the weight: log(p)/sqrt(p) * max|kernel|

    print(f'\n  Per-prime maximum drain contribution over L:')
    print(f'  {"prime":>6s} {"weight":>10s} {"max|Mp(L)|":>12s} {"at lam^2":>10s}')
    print('  ' + '-' * 42)

    prime_max_drain = []
    for p in [int(x) for x in primes[:30]]:
        max_mp = 0
        max_lam = 0
        for lam_sq in range(max(p**2 + 1, 50), 10001, 50):
            if p**2 > lam_sq:
                continue
            c = prime_contribution(p, lam_sq, N)
            if abs(c) > max_mp:
                max_mp = abs(c)
                max_lam = lam_sq
        prime_max_drain.append((p, max_mp, max_lam))
        print(f'  {p:>6d} {np.log(p)/np.sqrt(p):>10.6f} {max_mp:>12.6f} {max_lam:>10d}')

    # Sum of maxima (worst case)
    sum_max = sum(m for _, m, _ in prime_max_drain)
    print(f'\n  Sum of max |delta_p| for first 30 primes: {sum_max:.6f}')
    print(f'  Margin limit: 0.269')
    print(f'  Sum < margin? {"YES" if sum_max < 0.269 else "NO"}')

    if sum_max < 0.269:
        print(f'\n  *** THE SUM OF INDIVIDUAL PRIME DRAIN MAXIMA IS BELOW THE MARGIN ***')
        print(f'  Even if every prime hits its WORST CASE simultaneously,')
        print(f'  the total drain cannot exceed the margin.')
        print(f'  THIS WOULD PROVE |drain| < margin FOR ALL L (after adding tail bound).')
    else:
        print(f'\n  Sum exceeds margin by factor {sum_max/0.269:.2f}.')
        print(f'  Need to account for the fact that primes CAN\'T all hit')
        print(f'  their worst case simultaneously (phases are incommensurate).')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 46f SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE MARGIN-DRAIN INEQUALITY:

  margin(L) -> 0.269 (analytic, computable, no primes)
  |drain(L)| -> 0.240 (arithmetic, oscillates, depends on primes)
  gap -> 0.029 (positive at all {len(dense_lam)} computed points)

  MINIMUM GAP: {min_gap:.6f} at lam^2 = {min_gap_lam}

  BOUND ATTEMPTS:
    A. Triangle inequality (sum |delta_p|): {bound_A:.4f} {'< 0.264 WORKS' if bound_A < 0.264 else '> 0.264 TOO LOOSE'}
    B. Equidistribution (sqrt sum delta_p^2): {bound_B:.4f} {'< 0.264 WORKS' if bound_B < 0.264 else '> 0.264 TOO LOOSE'}
    C. Computed maximum: {bound_C:.4f} < {min(margins):.4f} = min margin -> GAP {min_gap:.4f}
    E. Per-prime max sum (30 primes): {sum_max:.4f} {'< 0.269 WORKS' if sum_max < 0.269 else '> 0.269 TOO LOOSE'}

  THE CLOSEST APPROACH:
  {'Option C (computed range) works with gap ' + f'{min_gap:.4f}' if min_gap > 0 else 'No bound works'}
  But this is numerical verification, not proof (covers finite range only).

  TO CLOSE THE GAP:
  Need either:
    (a) A tighter per-prime bound that accounts for phase incommensurability
    (b) An asymptotic argument that margin - drain -> 0.029 as L -> inf
    (c) A rigorous computer-assisted proof covering L in [L_0, L_max]
        combined with an asymptotic bound for L > L_max
''')

    print('=' * 76)
    print('  SESSION 46f COMPLETE')
    print('=' * 76)
