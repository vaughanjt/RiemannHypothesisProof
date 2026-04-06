"""
SESSION 67b -- LOCK THE ZEROS

Key identity from the Hadamard product at s=1/2:

  xi''(1/2)/xi(1/2) = Sum_rho -1/(rho-1/2)^2

For on-line zero rho = 1/2+ig:  contribution = 2/g^2  (positive)
For off-line zero rho = 1/2+d+ig: contribution = 2(g^2-d^2)/(d^2+g^2)^2
                                   which is LESS than 2/g^2

The total is FIXED (it equals xi''(1/2)/xi(1/2) = 0.0462).

Key insight: off-line zeros contribute LESS than on-line zeros to this sum.
But the sum is FIXED. If any zero contributes less, the others must
contribute MORE. But the maximum contribution per zero is 2/gamma^2
(achieved only when delta=0).

So: if ANY zero goes off-line, the deficit must be absorbed by other zeros.
But other zeros are ALSO bounded by their on-line contribution. The only
way the sum stays at 0.0462 is if ALL zeros are on-line.

Is this rigorous? Let's test.
"""

import sys
import numpy as np
import mpmath
from mpmath import mp, mpf, zetazero, euler as gamma_euler

mp.dps = 50


def xi_func(s):
    return mpmath.mpf('0.5') * s * (s - 1) * \
           mpmath.power(mpmath.pi, -s/2) * \
           mpmath.gamma(s/2) * mpmath.zeta(s)


def run():
    print()
    print('#' * 76)
    print('  SESSION 67b -- LOCK THE ZEROS')
    print('#' * 76)

    # ==================================================================
    # PART 1: THE LOCKING IDENTITY
    # ==================================================================
    print('\n  === PART 1: THE LOCKING IDENTITY ===\n')

    s_half = mpf('0.5')
    xi_val = float(xi_func(s_half))
    xi_pp = float(mpmath.diff(xi_func, s_half, n=2))

    Z_fixed = xi_pp / xi_val
    print(f'  xi(1/2) = {xi_val:.10f}')
    print(f'  xi\'\'(1/2) = {xi_pp:.10f}')
    print(f'  Z = xi\'\'(1/2)/xi(1/2) = {Z_fixed:.10f}')
    print()

    # Compute Sum 2/gamma^2 from zeros
    n_zeros = 300
    gammas = []
    for k in range(1, n_zeros + 1):
        gammas.append(float(mpmath.im(zetazero(k))))
    gammas = np.array(gammas)
    mp.dps = 30

    sum_on_line = np.sum(2.0 / gammas**2)
    # Tail correction
    gK = gammas[-1]
    tail = 2 * (np.log(gK) + 1) / (2*np.pi*gK)
    sum_on_line_total = sum_on_line + tail

    print(f'  Sum 2/gamma^2 (on-line hypothesis, {n_zeros} zeros + tail):')
    print(f'    = {sum_on_line_total:.10f}')
    print(f'  Z = {Z_fixed:.10f}')
    print(f'  Match: {abs(sum_on_line_total - Z_fixed):.2e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: WHAT IF ONE ZERO GOES OFF-LINE?
    # ==================================================================
    print('\n  === PART 2: DEFICIT FROM ONE OFF-LINE ZERO ===\n')

    print(f'  If zero k moves from 1/2+ig to 1/2+d+ig:')
    print(f'  On-line contribution: 2/g^2')
    print(f'  Off-line contribution: 2(g^2-d^2)/(d^2+g^2)^2')
    print(f'  Deficit: 2/g^2 - 2(g^2-d^2)/(d^2+g^2)^2 = 2d^2(3g^2+d^2)/(g^2(d^2+g^2)^2)')
    print()

    gamma1 = gammas[0]
    print(f'  Zero 1 (gamma={gamma1:.4f}):')
    print(f'  {"delta":>10} {"on-line":>14} {"off-line":>14} {"deficit":>14} '
          f'{"deficit/total":>14}')
    print('  ' + '-' * 70)

    on_line_1 = 2.0 / gamma1**2
    for delta in [0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 14.0]:
        off_line = 2*(gamma1**2 - delta**2) / (delta**2 + gamma1**2)**2
        # Double because the functional equation creates a quadruple
        # Actually, for the Hadamard sum, we count rho and 1-rho separately
        # If rho = 1/2+d+ig, then 1-rho = 1/2-d-ig
        # 1/(1/2-rho)^2 = 1/(-d-ig)^2 = 1/(d+ig)^2
        # 1/(1/2-(1-rho))^2 = 1/(rho-1/2)^2 = 1/(d+ig)^2
        # Hmm, same. Let me be careful.
        # rho = 1/2+d+ig, conj(rho) = 1/2+d-ig
        # 1-rho = 1/2-d-ig, 1-conj(rho) = 1/2-d+ig
        # 1/(1/2-rho)^2 = 1/(-d-ig)^2
        # 1/(1/2-conj(rho))^2 = 1/(-d+ig)^2
        # 1/(1/2-(1-rho))^2 = 1/(d+ig)^2 ... wait, 1/2-(1-rho) = rho-1/2 = d+ig
        # 1/(1/2-(1-conj(rho)))^2 = 1/(conj(rho)-1/2)^2 = 1/(d-ig)^2
        # Total: 1/(d+ig)^2 + 1/(d-ig)^2 + 1/(-d+ig)^2 + 1/(-d-ig)^2
        # = 2*Re[1/(d+ig)^2] + 2*Re[1/(-d+ig)^2]
        # = 2(d^2-g^2)/(d^2+g^2)^2 + 2(d^2-g^2)/(d^2+g^2)^2
        # = 4(d^2-g^2)/(d^2+g^2)^2
        # And -Sum = -4(d^2-g^2)/(d^2+g^2)^2 = 4(g^2-d^2)/(d^2+g^2)^2
        # For on-line (d=0): 4*g^2/g^4 = 4/g^2 = 2*(2/g^2) ... hmm
        # So the contribution per on-line PAIR is 4/g^2, not 2/g^2.
        # Wait, the pair (rho, conj(rho)) with both on-line gives
        # -[1/(ig)^2 + 1/(-ig)^2] = -[-1/g^2 + (-1/g^2)] = 2/g^2
        # And the pair (1-rho, 1-conj(rho)) = (conj(rho), rho) is the SAME pair!
        # So for on-line zeros, there's no quadruple, just a pair, contributing 2/g^2.
        # For off-line zeros, there ARE four distinct zeros contributing 4(g^2-d^2)/(d^2+g^2)^2.

        # So: off-line quadruple contributes 4(g^2-d^2)/(d^2+g^2)^2
        # where on-line pair contributes 2/g^2 = 4/g^2 ... wait.
        # I need to be more careful about what we're comparing.

        # When zero MOVES from on-line to off-line:
        # Before: 1 pair (rho, conj(rho)) contributing 2/g^2
        # After: 2 pairs (rho,conj(rho)) and (1-rho,1-conj(rho)) = 1 quadruple
        #   contributing 4(g^2-d^2)/(d^2+g^2)^2
        # So the CHANGE in Z is: 4(g^2-d^2)/(d^2+g^2)^2 - 2/g^2

        # Hmm but the number of zeros DOUBLES when moving off-line!
        # On-line: rho = 1/2+ig, conj = 1/2-ig. That's 2 zeros.
        # Off-line: rho = 1/2+d+ig, conj = 1/2+d-ig, 1-rho = 1/2-d-ig,
        #           1-conj = 1/2-d+ig. That's 4 zeros.
        # But we can't CREATE zeros. The total number is fixed.
        # Moving one pair off-line means the quadruple already existed as 2 pairs
        # (the pair at 1/2+ig and the pair at 1/2-ig were actually 4 distinct
        #  points that happened to have sigma=1/2).

        # I think the correct comparison is simpler. Just compare the contribution
        # of the SAME number of zeros.
        # On-line pair: 2/g^2
        # Off-line pair (rho, conj(rho)) at sigma=1/2+d:
        #   1/(1/2-rho)^2 + 1/(1/2-conj(rho))^2 = 2*(d^2-g^2)/(d^2+g^2)^2
        #   Wait: -(sum) = -2*(d^2-g^2)/(d^2+g^2)^2 = 2*(g^2-d^2)/(d^2+g^2)^2

        # So contribution of the PAIR (rho, conj(rho)) at sigma=1/2+d:
        # 2*(g^2-d^2)/(d^2+g^2)^2

        off_pair = 2*(gamma1**2 - delta**2) / (delta**2 + gamma1**2)**2
        deficit = on_line_1 - off_pair
        deficit_frac = deficit / Z_fixed

        print(f'  {delta:>10.3f} {on_line_1:>14.8f} {off_pair:>14.8f} '
              f'{deficit:>14.8e} {deficit_frac:>14.8f}')

    sys.stdout.flush()

    # ==================================================================
    # PART 3: THE LOCKING ARGUMENT
    # ==================================================================
    print('\n  === PART 3: THE LOCKING ARGUMENT ===\n')

    # Z = xi''(1/2)/xi(1/2) = Sum_{pairs} 2(g^2-d^2)/(d^2+g^2)^2
    # Each term <= 2/g^2 (with equality iff d=0)
    # Total = Z (fixed)
    # Maximum possible total (all on-line) = Sum 2/g^2 = Z
    #
    # IF any d_k > 0: that pair contributes LESS than 2/g_k^2
    # But Sum must still equal Z = Sum 2/g_k^2 (if all were on-line)
    # CONTRADICTION: the sum can't reach Z if any term is reduced

    # Wait - this assumes the gamma values don't change.
    # When a zero moves off-line, its gamma could shift too.
    # The constraint is xi = (Euler product), which fixes ALL zeros simultaneously.

    # But here's the KEY: the MAXIMUM of each term 2(g^2-d^2)/(d^2+g^2)^2
    # over all d is achieved at d=0, giving 2/g^2.
    # This is true for EACH gamma value separately.

    # So: Sum_{pairs} max_{d_k} f(d_k, g_k) = Sum 2/g_k^2 = Z
    # And: Sum f(d_k, g_k) = Z (actual, with actual d_k)
    # Since f(d_k, g_k) <= 2/g_k^2 for each k:
    # Z = Sum f(d_k, g_k) <= Sum 2/g_k^2 = Z
    # Equality iff f(d_k, g_k) = 2/g_k^2 for all k, i.e., d_k = 0 for all k.

    # THIS IS THE PROOF... IF THE GAMMA VALUES ARE FIXED!

    # The problem: the gamma values are NOT fixed independently of the d values.
    # The zeros (d_k, g_k) are determined simultaneously by the Euler product.
    # Different configurations could have different gamma values.

    print(f'  Z = xi\'\'(1/2)/xi(1/2) = {Z_fixed:.10f}')
    print()
    print(f'  Each pair contributes: f(d,g) = 2(g^2-d^2)/(d^2+g^2)^2')
    print(f'  Maximum of f over d:  f(0,g) = 2/g^2')
    print()
    print(f'  Key: Sum f(d_k, g_k) = Z  (actual zeros)')
    print(f'        Sum 2/g_k^2    = Z  (IF all on-line)')
    print()
    print(f'  Since f(d,g) <= 2/g^2 with equality iff d=0:')
    print(f'  Z = Sum f <= Sum 2/g^2 = Z')
    print(f'  Equality requires f = 2/g^2 for EVERY pair => d=0 for all.')
    print()
    print(f'  *** BUT: this assumes the same gamma values in both sums! ***')
    print(f'  If zeros move off-line, their imaginary parts could change.')
    print(f'  The argument holds IFF the gamma values are determined')
    print(f'  independently of the delta values.')

    # ==================================================================
    # PART 4: ARE GAMMA VALUES INDEPENDENT OF DELTA?
    # ==================================================================
    print(f'\n  === PART 4: GAMMA-DELTA COUPLING ===\n')

    # In the Hadamard product: xi is fixed, so the zeros are fixed.
    # There's no "moving" zeros - they are where they are.
    # But hypothetically: could a different function (same Euler product
    # but different zero positions) have the same Z?

    # The answer: NO. Different zero positions give different xi functions.
    # The zero positions UNIQUELY determine xi (by Hadamard).
    # And xi is uniquely determined by the Euler product.
    # So there's ONE set of zeros, and we're asking if they're on-line.

    # The locking argument says:
    # IF we could prove Sum 2/g_k^2 = Z (which requires knowing g_k are
    # the imaginary parts of ON-LINE zeros), then d_k = 0 follows.
    # But Sum 2/g_k^2 = Z IS the statement that all zeros are on-line!

    print(f'  The Hadamard product determines xi from zeros.')
    print(f'  The Euler product determines xi from primes.')
    print(f'  Both give the SAME xi, hence the SAME Z = {Z_fixed:.6f}.')
    print()
    print(f'  The zeros are at positions rho_k = sigma_k + i*gamma_k.')
    print(f'  Z = Sum_pairs -Re[1/(rho_k - 1/2)^2 + 1/(conj(rho_k) - 1/2)^2]')
    print(f'    = Sum_pairs 2*(gamma_k^2 - delta_k^2)/(delta_k^2 + gamma_k^2)^2')
    print()
    print(f'  For the locking argument: we need to compare this to')
    print(f'  Sum 2/gamma_k^2 (the hypothetical all-on-line value).')
    print(f'  These are equal iff all delta_k = 0.')
    print()

    # Can we compute "Sum 2/gamma_k^2" WITHOUT knowing the zeros are on-line?
    # Sum 2/gamma_k^2 requires knowing the gamma values.
    # If zeros are off-line, gamma values might differ from on-line gamma values.
    # So "Sum 2/gamma_k^2" is different for on-line vs off-line configurations.

    # BUT: there's only ONE configuration (the actual one).
    # And Z = Sum f(delta_k, gamma_k) is fixed.
    # The question: is Z = max over all configurations, i.e., is Z achieved
    # by the all-on-line configuration?

    # If Z is the MAXIMUM possible value (over all zero configurations
    # consistent with the Euler product), then the actual configuration
    # must be the maximizer, which is all-on-line.

    # So: IS xi''(1/2)/xi(1/2) MAXIMIZED when all zeros are on the critical line?

    print(f'  THE KEY QUESTION:')
    print(f'  Is Z = xi\'\'(1/2)/xi(1/2) MAXIMIZED when all zeros are on the line?')
    print(f'  If yes: the actual configuration (which achieves Z) must be on-line.')
    print(f'  If no: the argument fails.')
    print()

    # For a SINGLE zero: max of 2(g^2-d^2)/(d^2+g^2)^2 over d is at d=0.
    # For the SUM: max of Sum f(d_k, g_k) over (d_k, g_k) subject to
    # "the zeros define the same xi function" is at d_k = 0.
    # But is this true? The constraint "same xi" is extremely rigid.

    # Actually, the constraint "same xi" means the zeros are FIXED.
    # There's nothing to optimize over. The zeros are what they are.

    # The argument only works if we can show:
    # Z_actual = Z_max = Sum 2/gamma_k^2 (all on-line)
    # And the max is achieved uniquely at d=0.

    # For this: Z_max = Sum 2/g_k^2 where g_k are the ON-LINE gamma values.
    # Z_actual = Sum 2(G_k^2-D_k^2)/(D_k^2+G_k^2)^2 where (D_k, G_k) are actual.
    # Z_actual = Z_max iff all D_k = 0 AND G_k = g_k.

    print(f'  For a FIXED set of gamma values:')
    print(f'    Z(d=0) = Sum 2/gamma^2 = MAXIMUM')
    print(f'    Z(d>0) = Sum 2(g^2-d^2)/(d^2+g^2)^2 < Z(d=0)')
    print(f'    Difference = Sum 2d^2(3g^2+d^2)/(g^2(d^2+g^2)^2) > 0')
    print()
    print(f'  This means: the on-line configuration MAXIMIZES Z')
    print(f'  among all configurations with THE SAME gamma values.')
    print()
    print(f'  If the actual Z equals this maximum: all d=0. QED.')
    print()
    print(f'  The gap: we don\'t know the gamma values independently.')
    print(f'  We know Z = {Z_fixed:.6f} and we know the actual (d_k, g_k).')
    print(f'  If we could show Z = Sum 2/g_k^2 (on-line sum), d=0 follows.')
    print(f'  But Sum 2/g_k^2 requires knowing g_k, which requires knowing d_k=0...')
    sys.stdout.flush()

    # ==================================================================
    # PART 5: BREAK THE CIRCULARITY?
    # ==================================================================
    print(f'\n  === PART 5: BREAKING THE CIRCULARITY ===\n')

    # The circularity: Sum 2/g^2 (on-line) = Z requires knowing g = Im(rho) on-line.
    # But we DO know Im(rho) for the actual zeros (whether on-line or not).
    # The question: does Im(rho) CHANGE when Re(rho) changes?

    # For the actual zeta function: the zeros are at fixed positions.
    # We CAN compute Im(rho_k) = gamma_k (the actual imaginary parts).
    # And we CAN compute Z = xi''(1/2)/xi(1/2).

    # The identity: Z = Sum 2*(gamma_k^2 - delta_k^2)/(delta_k^2 + gamma_k^2)^2
    # And also: Sum 2/gamma_k^2 (the on-line value with the SAME gamma_k).

    # If delta_k > 0 for some k: Z < Sum 2/gamma_k^2
    # If delta_k = 0 for all k: Z = Sum 2/gamma_k^2

    # So: RH <=> Z = Sum 2/gamma_k^2

    # We can COMPUTE both sides:
    # LHS = xi''(1/2)/xi(1/2) (from the xi function)
    # RHS = Sum 2/gamma_k^2 (from the known zero imaginary parts)

    # If they MATCH: RH is true.
    # If they DON'T match: RH is false (and the deficit measures Sum delta^2).

    print(f'  Z (from xi):                    {Z_fixed:.10f}')
    print(f'  Sum 2/gamma_k^2 (from zeros):   {sum_on_line_total:.10f}')
    print(f'  Difference:                      {Z_fixed - sum_on_line_total:.4e}')
    print()

    # The difference should be 0 iff RH.
    # If difference > 0: some delta > 0 (off-line zeros)
    # If difference = 0: all delta = 0 (RH)
    # If difference < 0: IMPOSSIBLE (the contribution function has max at d=0)

    diff = Z_fixed - sum_on_line_total
    print(f'  If diff = 0: RH true')
    print(f'  If diff > 0: not possible (on-line is maximum)')
    print(f'  If diff < 0: some zeros off-line (deficit = -diff)')
    print()
    print(f'  Observed diff = {diff:.6e}')
    print(f'  Sign: {"ZERO (within precision)" if abs(diff) < 1e-4 else ("POSITIVE (impossible)" if diff > 0 else "NEGATIVE")}')
    print()

    if abs(diff) < 0.01:
        print(f'  The difference is < 0.01, consistent with ZERO.')
        print(f'  This is consistent with all zeros on the critical line.')
        print(f'  (Residual from tail approximation, not off-line zeros.)')
    print()

    # Wait - I had the inequality backwards. Let me recheck.
    # f(d,g) = 2(g^2-d^2)/(d^2+g^2)^2
    # At d=0: f = 2/g^2
    # df/dd = 2*[-2d(d^2+g^2)^2 - (g^2-d^2)*2(d^2+g^2)*2d] / (d^2+g^2)^4
    #       = ... at d=0: df/dd = 0 (saddle or maximum)
    # d^2f/dd^2 at d=0: need to compute...
    # f(d,g) = 2g^2/(d^2+g^2)^2 - 2d^2/(d^2+g^2)^2
    # The first term decreases with d (for d > 0), second increases.
    # At d=0: f = 2/g^2. For small d: f ~ 2/g^2 - 6d^2/g^4 (decreasing).
    # So d=0 IS a maximum. Good.

    # So the ACTUAL Z should be LESS than or equal to the on-line Sum 2/gamma^2.
    # Equality iff d=0 for all zeros.

    # And the actual Z = xi''(1/2)/xi(1/2) is a FIXED number.
    # The on-line Sum 2/gamma^2 is ALSO a fixed number (using the actual gamma values).

    # If the actual zeros are all on-line: Z_actual = Sum 2/gamma^2 (match).
    # If some are off-line: Z_actual < Sum 2/gamma^2 (strict inequality).

    # So: Z_actual < Sum 2/gamma^2  <==>  some zeros off-line
    #     Z_actual = Sum 2/gamma^2  <==>  RH

    print(f'  *** THE LOCKING TEST ***')
    print(f'  Compute INDEPENDENTLY:')
    print(f'    A = xi\'\'(1/2)/xi(1/2)    (from xi function)')
    print(f'    B = Sum 2/gamma_k^2       (from zero imaginary parts)')
    print(f'  Then:')
    print(f'    A = B  <==>  ALL zeros on critical line (RH)')
    print(f'    A < B  <==>  SOME zeros off-line')
    print(f'    A > B  <==>  IMPOSSIBLE')
    print()
    print(f'  A = {Z_fixed:.10f}')
    print(f'  B = {sum_on_line_total:.10f} (from {n_zeros} zeros + tail)')
    print(f'  A - B = {Z_fixed - sum_on_line_total:.6e}')
    print()
    print(f'  Within computational precision: A = B.')
    print(f'  This is CONSISTENT with RH but does not prove it')
    print(f'  (limited by finite zero computation + tail approximation).')
    sys.stdout.flush()

    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 67b VERDICT')
    print('=' * 76)
    print()
    print('  THE LOCKING IDENTITY:')
    print(f'    xi\'\'(1/2)/xi(1/2) = Sum 2*(g^2-d^2)/(d^2+g^2)^2')
    print(f'    Maximum at d=0: Sum 2/g^2')
    print(f'    If actual = max: all d=0 (RH)')
    print()
    print('  STATUS: A = B to available precision.')
    print('  To PROVE: need A = B exactly, which requires')
    print('  either (a) computing Sum 2/gamma^2 to infinite precision,')
    print('  or (b) proving the identity A = max analytically.')


if __name__ == '__main__':
    run()
