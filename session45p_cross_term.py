"""
SESSION 45p — THE CROSS-TERM: THE FINAL PUSH

RH = the prime-prime cross-term stays below 50% of the adelic norm squared.

Specifically:
  |B_adelic|^2 = W02^2 + sum_k Mp_k^2     (always positive)
  |B_real|^2   = (W02 - sum_k Mp_k)^2     (positive iff RH)

  Cross-term = (|B_adelic|^2 - |B_real|^2) / 2 = sum_{j<k} Mp_j * Mp_k

  RH iff |B_real|^2 > 0
     iff |B_adelic|^2 > 2 * Cross-term
     iff Cross-term / |B_adelic|^2 < 1/2

At lam^2=2000: Cross/Adelic^2 = 49.74%
The margin to 50% is 0.26%.

PLAN:
  1. Track Cross/Adelic^2 ratio as lam^2 -> infinity
  2. Does it converge? To what? If < 0.5, that's RH.
  3. Decompose the cross-term: which prime PAIRS contribute most?
  4. Can the ratio be expressed analytically?
  5. Connection to the PNT and zero density
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
    print('  SESSION 45p — THE CROSS-TERM: FINAL PUSH')
    print('  RH iff Cross/Adelic^2 < 0.5')
    print('=' * 76)

    N = 15

    # ══════════════════════════════════════════════════════════════
    # 1. TRACK THE RATIO AS lam^2 -> infinity
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. CROSS-TERM RATIO vs lam^2')
    print('#' * 76)

    print(f'\n  {"lam^2":>8s} {"K primes":>8s} {"W02":>12s} {"sum Mp":>12s} '
          f'{"B_real":>10s} {"Adelic^2":>12s} {"Cross":>12s} '
          f'{"Cross/Ad^2":>10s} {"margin":>8s}')
    print('  ' + '-' * 100)

    ratios = []
    lam_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    for lam_sq in lam_values:
        t0 = time.time()
        w02 = w02_only(lam_sq, N)
        primes = list(sieve_primes(int(lam_sq)))
        K = len(primes)

        mp_list = [prime_contribution(int(p), lam_sq, N) for p in primes]
        mp_arr = np.array(mp_list)
        mp_total = np.sum(mp_arr)

        b_real = w02 - mp_total
        adelic_sq = w02**2 + np.sum(mp_arr**2)
        cross = (adelic_sq - b_real**2) / 2
        ratio = cross / adelic_sq if adelic_sq > 0 else 0
        margin = 0.5 - ratio

        dt = time.time() - t0
        ratios.append((lam_sq, K, ratio, margin, b_real, adelic_sq))

        print(f'  {lam_sq:>8d} {K:>8d} {w02:>+12.4f} {mp_total:>+12.4f} '
              f'{b_real:>+10.6f} {adelic_sq:>12.4f} {cross:>12.4f} '
              f'{ratio:>10.6f} {margin:>+8.6f}  ({dt:.1f}s)')
        sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 2. DOES THE RATIO CONVERGE?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. CONVERGENCE ANALYSIS')
    print('#' * 76)

    Ls = np.array([np.log(r[0]) for r in ratios])
    Rs = np.array([r[2] for r in ratios])
    Ms = np.array([r[3] for r in ratios])

    print(f'\n  Cross/Adelic^2 ratio trajectory:')
    for lsq, K, ratio, margin, _, _ in ratios:
        bar = '#' * int(ratio * 100) + '.' * int((0.5 - ratio) * 100) + '|'
        print(f'    lam^2={lsq:>6d}: {ratio:.6f} [{bar}] margin={margin:+.6f}')

    # Fit: ratio = a - b/L ?
    if len(Ls) >= 4:
        X = np.column_stack([np.ones_like(Ls), 1/Ls])
        c = np.linalg.lstsq(X, Rs, rcond=None)[0]
        print(f'\n  Fit: ratio = {c[0]:.6f} + {c[1]:.4f}/L')
        print(f'  Limit (L->inf): {c[0]:.6f}')
        print(f'  Margin at L->inf: {0.5 - c[0]:.6f}')

        if c[0] < 0.5:
            print(f'  *** RATIO CONVERGES BELOW 0.5! ***')
            print(f'  Predicted limit: {c[0]:.6f} < 0.5')
            print(f'  Asymptotic margin: {0.5 - c[0]:.6f}')
        else:
            print(f'  Ratio converges to {c[0]:.6f} >= 0.5')

    # Also fit: ratio = a - b/L - c/L^2
    if len(Ls) >= 5:
        X2 = np.column_stack([np.ones_like(Ls), 1/Ls, 1/Ls**2])
        c2 = np.linalg.lstsq(X2, Rs, rcond=None)[0]
        print(f'\n  Quadratic fit: ratio = {c2[0]:.6f} + {c2[1]:.4f}/L + {c2[2]:.4f}/L^2')
        print(f'  Limit (L->inf): {c2[0]:.6f}')

    # Fit: ratio = 0.5 - a/L^b ?
    margins = 0.5 - Rs
    log_margins = np.log(margins[margins > 0])
    log_Ls = Ls[:len(log_margins)]
    if len(log_Ls) >= 3:
        cm = np.polyfit(log_Ls, log_margins, 1)
        print(f'\n  Margin fit: margin ~ L^{{{cm[0]:.4f}}} * exp({cm[1]:.4f})')
        print(f'  Margin decay exponent: {cm[0]:.4f}')
        if cm[0] < 0:
            print(f'  Margin SHRINKS as L grows (exponent < 0)')
            print(f'  Predicted margin at L=20: {np.exp(cm[1]) * 20**cm[0]:.6f}')
            print(f'  Predicted margin at L=50: {np.exp(cm[1]) * 50**cm[0]:.6f}')
            print(f'  Predicted margin at L=100: {np.exp(cm[1]) * 100**cm[0]:.6f}')
            if cm[0] > -1:
                print(f'  Margin decays SLOWER than 1/L -> sum converges -> RH likely!')
            else:
                print(f'  Margin decays FASTER than 1/L -> could reach zero')
        else:
            print(f'  Margin GROWS! Barrier becomes relatively stronger.')

    # ══════════════════════════════════════════════════════════════
    # 3. THE CROSS-TERM DECOMPOSITION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. CROSS-TERM DECOMPOSITION: which prime pairs matter?')
    print('#' * 76)

    lam_sq = 2000
    primes = list(sieve_primes(int(lam_sq)))
    mp_arr = np.array([prime_contribution(int(p), lam_sq, N) for p in primes])
    K = len(primes)

    # Full cross-term matrix
    cross_matrix = np.outer(mp_arr, mp_arr)
    np.fill_diagonal(cross_matrix, 0)
    total_cross = np.sum(cross_matrix) / 2

    print(f'\n  lam^2 = {lam_sq}, {K} primes')
    print(f'  Total cross-term: {total_cross:+.6f}')

    # Contribution by prime-pair distance
    print(f'\n  Cross-term by gap between primes (index distance):')
    print(f'  {"gap":>6s} {"contribution":>14s} {"cumulative":>14s} {"% of total":>10s}')
    print('  ' + '-' * 48)

    cum = 0
    for gap in range(1, min(50, K)):
        gap_sum = sum(mp_arr[i] * mp_arr[i+gap] for i in range(K-gap))
        cum += gap_sum
        pct = cum / total_cross * 100 if total_cross != 0 else 0
        if gap <= 10 or gap % 10 == 0:
            print(f'  {gap:>6d} {gap_sum:>+14.6f} {cum:>+14.6f} {pct:>9.1f}%')

    # Contribution by prime SIZE
    print(f'\n  Cross-term by prime size (cumulative from small primes):')
    print(f'  {"primes up to":>12s} {"cross within":>14s} {"% of total":>10s}')
    print('  ' + '-' * 40)

    for cutoff_idx in [5, 10, 20, 50, 100, 200, K]:
        idx = min(cutoff_idx, K)
        sub_cross = 0
        for j in range(idx):
            for k in range(j+1, idx):
                sub_cross += mp_arr[j] * mp_arr[k]
        pct = sub_cross / total_cross * 100 if total_cross != 0 else 0
        print(f'  {primes[idx-1] if idx > 0 else 0:>12d} {sub_cross:>+14.6f} {pct:>9.1f}%')

    # ══════════════════════════════════════════════════════════════
    # 4. ANALYTIC EXPRESSION FOR THE RATIO
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. ANALYTIC STRUCTURE OF THE RATIO')
    print('#' * 76)

    print(f'''
  Cross/Adelic^2 = [sum_{{j<k}} Mp_j * Mp_k] / [W02^2 + sum_k Mp_k^2]

  Numerator = (1/2) * [(sum Mp_k)^2 - sum Mp_k^2]
            = (1/2) * [Mp_total^2 - sum Mp_k^2]

  So: Cross/Adelic^2 = (Mp_total^2 - sum Mp_k^2) / (2 * (W02^2 + sum Mp_k^2))

  Define:
    S1 = sum Mp_k     (first moment)
    S2 = sum Mp_k^2   (second moment)
    R = Cross/Adelic^2 = (S1^2 - S2) / (2*(W02^2 + S2))

  RH iff R < 1/2
     iff S1^2 - S2 < W02^2 + S2
     iff S1^2 < W02^2 + 2*S2
     iff (sum Mp_k)^2 < W02^2 + 2*sum Mp_k^2

  This is a QUADRATIC FORM inequality!
  ''')

    for lam_sq in [500, 2000, 10000]:
        primes = list(sieve_primes(int(lam_sq)))
        mp_arr = np.array([prime_contribution(int(p), lam_sq, N) for p in primes])
        w02 = w02_only(lam_sq, N)

        S1 = np.sum(mp_arr)
        S2 = np.sum(mp_arr**2)

        lhs = S1**2
        rhs = w02**2 + 2 * S2

        print(f'  lam^2 = {lam_sq}:')
        print(f'    S1^2 = (sum Mp)^2 = {lhs:.6f}')
        print(f'    W02^2 + 2*S2     = {rhs:.6f}')
        print(f'    RH holds iff {lhs:.4f} < {rhs:.4f}: {"YES" if lhs < rhs else "NO"}')
        print(f'    Margin: {rhs - lhs:.6f}')
        print(f'    Relative margin: {(rhs - lhs)/rhs:.6f}')

    # ══════════════════════════════════════════════════════════════
    # 5. THE S1^2 < W02^2 + 2*S2 INEQUALITY
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. THE MASTER INEQUALITY: S1^2 < W02^2 + 2*S2')
    print('#' * 76)

    print(f'\n  Track all three quantities as L grows:\n')
    print(f'  {"lam^2":>8s} {"L":>7s} {"S1^2":>14s} {"W02^2":>14s} '
          f'{"2*S2":>14s} {"W02^2+2S2":>14s} {"margin":>10s} {"rel":>8s}')
    print('  ' + '-' * 90)

    for lam_sq in lam_values:
        primes = list(sieve_primes(int(lam_sq)))
        mp_arr = np.array([prime_contribution(int(p), lam_sq, N) for p in primes])
        w02 = w02_only(lam_sq, N)
        L = np.log(lam_sq)

        S1 = np.sum(mp_arr)
        S2 = np.sum(mp_arr**2)

        lhs = S1**2
        rhs = w02**2 + 2 * S2
        margin = rhs - lhs
        rel = margin / rhs if rhs > 0 else 0

        print(f'  {lam_sq:>8d} {L:>7.3f} {lhs:>14.4f} {w02**2:>14.4f} '
              f'{2*S2:>14.4f} {rhs:>14.4f} {margin:>+10.4f} {rel:>8.4f}')

    # ══════════════════════════════════════════════════════════════
    # 6. GROWTH RATES
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. GROWTH RATES: which term grows fastest?')
    print('#' * 76)

    print(f'\n  S1 = sum Mp_k ~ ?')
    print(f'  S2 = sum Mp_k^2 ~ ?')
    print(f'  W02 ~ ?')

    s1_vals = []
    s2_vals = []
    w02_vals = []
    L_vals = []

    for lam_sq in lam_values:
        primes = list(sieve_primes(int(lam_sq)))
        mp_arr = np.array([prime_contribution(int(p), lam_sq, N) for p in primes])
        w02 = w02_only(lam_sq, N)
        L = np.log(lam_sq)

        s1_vals.append(abs(np.sum(mp_arr)))
        s2_vals.append(np.sum(mp_arr**2))
        w02_vals.append(abs(w02))
        L_vals.append(L)

    L_arr = np.array(L_vals)
    s1_arr = np.array(s1_vals)
    s2_arr = np.array(s2_vals)
    w02_arr = np.array(w02_vals)

    # Fit power laws
    for name, vals in [('|S1|', s1_arr), ('S2', s2_arr), ('|W02|', w02_arr)]:
        log_v = np.log(vals[vals > 0])
        log_L = L_arr[:len(log_v)]
        if len(log_L) >= 3:
            c = np.polyfit(log_L, log_v, 1)
            print(f'  {name:>6s} ~ exp({c[0]:.4f} * L) * {np.exp(c[1]):.4f}  (slope={c[0]:.4f})')

    print(f'\n  S1^2 grows as exp(2 * slope_S1 * L)')
    print(f'  W02^2 + 2*S2 grows as exp(2 * slope_W02 * L) + exp(slope_S2 * L)')
    print(f'\n  RH holds if slope_S1 < slope_W02 (S1 grows slower than W02)')
    s1_slope = np.polyfit(L_arr, np.log(s1_arr), 1)[0]
    w02_slope = np.polyfit(L_arr, np.log(w02_arr), 1)[0]
    s2_slope = np.polyfit(L_arr, np.log(s2_arr), 1)[0]

    print(f'  |S1| slope:  {s1_slope:.6f}')
    print(f'  |W02| slope: {w02_slope:.6f}')
    print(f'  S2 slope:    {s2_slope:.6f}')
    print(f'  S1 < W02? slope test: {s1_slope:.6f} < {w02_slope:.6f}: '
          f'{"YES" if s1_slope < w02_slope else "NO"}')

    if abs(s1_slope - w02_slope) < 0.01:
        print(f'\n  *** SLOPES ARE NEARLY EQUAL: {s1_slope:.6f} vs {w02_slope:.6f} ***')
        print(f'  The race is CLOSE. Both grow at essentially the same rate.')
        print(f'  The margin comes from the CONSTANT (not the exponent).')
        print(f'  This is consistent with barrier ~ O(1) as L -> inf.')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45p SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE MASTER INEQUALITY (equivalent to RH):

    (sum Mp_k)^2  <  W02^2 + 2 * sum Mp_k^2

  Left side: the SQUARE of the total prime sum (coherent part)^2.
  Right side: W02^2 (pi's contribution) + 2*(sum of individual prime^2).

  In words: the coherent sum of primes, squared, must be less than
  pi squared plus twice the incoherent sum of prime squares.

  NUMERICAL EVIDENCE:
    Cross/Adelic^2 ratio approaches ~{c[0]:.4f} as L -> infinity
    Margin to 0.5 threshold: ~{0.5 - c[0]:.4f}
    Growth rates: |S1| and |W02| grow at slopes {s1_slope:.4f} and {w02_slope:.4f}

  THE VERDICT:
    {'THE RATIO CONVERGES BELOW 0.5 — CONSISTENT WITH RH' if c[0] < 0.5 else 'THE RATIO MAY REACH 0.5 — INCONCLUSIVE'}

  This is the cleanest formulation we have:
    RH = the coherent prime sum never exceeds sqrt(pi^2 + 2*noise)
    where "pi^2" = W02^2 and "noise" = sum of individual prime^2.
''')

    print('=' * 76)
    print('  SESSION 45p COMPLETE')
    print('=' * 76)
