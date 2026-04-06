"""
SESSION 66b -- CONNECTING DELTA TO THE BARRIER

The sign lemma gives: each zero at delta off-line shifts max_eig by delta*4/gamma^2.
The RT barrier B(gamma) measures the electrostatic "cost" of being off-line.

Key question: IS there a rigorous relationship delta < f(B)?

Approach: for each known zero rho_k = 1/2 + i*gamma_k, compute |zeta(1/2 + delta + i*gamma_k)|
as a function of delta. The "zero-free width" is the smallest delta where |zeta| could vanish.

If this width correlates with B, we have the connection.
"""

import sys
import numpy as np
import mpmath

sys.path.insert(0, '.')

mpmath.mp.dps = 25


def run():
    print()
    print('#' * 76)
    print('  SESSION 66b -- DELTA-BARRIER CONNECTION')
    print('#' * 76)

    # ==================================================================
    # PART 1: |zeta(1/2 + delta + i*gamma)| NEAR ZEROS
    # ==================================================================
    print('\n  === PART 1: |zeta| AS FUNCTION OF DELTA NEAR ZEROS ===')
    print('  For each zero gamma_k, compute |zeta(1/2+delta+i*gamma_k)|.\n')

    n_zeros = 50
    gammas = []
    for k in range(1, n_zeros + 1):
        gammas.append(float(mpmath.im(mpmath.zetazero(k))))
    gammas = np.array(gammas)

    # Compute |zeta'| and R at each zero
    zeta_prime = np.zeros(n_zeros)
    R = np.zeros(n_zeros)
    for k in range(n_zeros):
        rho = mpmath.mpf('0.5') + mpmath.mpf(gammas[k]) * 1j
        zeta_prime[k] = abs(complex(mpmath.zeta(rho, derivative=1)))
        diffs = gammas[k] - gammas
        diffs[k] = 1e10
        R[k] = np.sum(1.0 / diffs**2)
    B = R * zeta_prime**2

    # For selected zeros, evaluate |zeta(1/2 + delta + i*gamma)|
    test_zeros = [0, 1, 4, 9, 19, 49]
    deltas = np.logspace(-6, -0.3, 30)

    print(f'  {"zero":>5} {"gamma":>10} {"|zeta\'|":>10} {"B":>10} '
          f'{"delta at min":>12} {"|zeta| at min":>14}')
    print('  ' + '-' * 66)

    delta_at_min = []
    for idx in test_zeros:
        if idx >= n_zeros:
            continue
        gamma = gammas[idx]
        min_zeta = float('inf')
        best_delta = 0
        for d in deltas:
            s = mpmath.mpf('0.5') + mpmath.mpf(d) + mpmath.mpf(gamma) * 1j
            z = abs(complex(mpmath.zeta(s)))
            if z < min_zeta:
                min_zeta = z
                best_delta = d

        delta_at_min.append((idx, gamma, zeta_prime[idx], B[idx], best_delta, min_zeta))
        print(f'  {idx+1:>5d} {gamma:>10.4f} {zeta_prime[idx]:>10.4f} '
              f'{B[idx]:>10.4f} {best_delta:>12.6e} {min_zeta:>14.6e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: THE GROWTH RATE OF |zeta| WITH DELTA
    # ==================================================================
    print('\n  === PART 2: |zeta(1/2+delta+ig)| GROWTH RATE ===')
    print('  Near a zero, |zeta| ~ |zeta\'| * delta for small delta.')
    print('  This gives a "zero-free width" ~ 1/|zeta\'| where |zeta| is O(1).\n')

    # For the first zero, detailed profile
    gamma1 = gammas[0]
    print(f'  Zero 1 (gamma={gamma1:.4f}), |zeta\'| = {zeta_prime[0]:.4f}:')
    print(f'  {"delta":>12} {"|zeta|":>14} {"|zeta\'|*delta":>14} {"ratio":>8}')
    print('  ' + '-' * 52)

    for d in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5]:
        s = mpmath.mpf('0.5') + mpmath.mpf(d) + mpmath.mpf(gamma1) * 1j
        z = abs(complex(mpmath.zeta(s)))
        linear = zeta_prime[0] * d
        ratio = z / linear if linear > 1e-15 else float('inf')
        print(f'  {d:>12.6e} {z:>14.6e} {linear:>14.6e} {ratio:>8.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 3: THE CRITICAL DELTA -- WHERE |zeta| MATCHES THE MARGIN
    # ==================================================================
    print('\n  === PART 3: ZERO-FREE WIDTH FROM THE SIGN LEMMA ===')
    print('  The sign lemma says: moving zero by delta shifts max_eig by')
    print('  delta * 4/gamma^2. The margin is ~2.6e-7 at L=12.6.')
    print('  Critical delta = margin * gamma^2 / 4.\n')

    margin = 2.6e-7
    print(f'  {"zero":>5} {"gamma":>10} {"B":>10} '
          f'{"delta_sign":>14} {"delta_free":>14} {"ratio":>10}')
    print('  ' + '-' * 68)

    for idx in range(min(30, n_zeros)):
        gamma = gammas[idx]
        delta_sign = margin * gamma**2 / 4  # sign lemma critical delta

        # "zero-free width": delta where |zeta(1/2+delta+ig)| ~ 1
        # Near a zero: |zeta| ~ |zeta'| * delta, so |zeta| = 1 at delta ~ 1/|zeta'|
        delta_free = 1.0 / zeta_prime[idx]

        ratio = delta_sign / delta_free if delta_free > 0 else float('inf')
        print(f'  {idx+1:>5d} {gamma:>10.4f} {B[idx]:>10.4f} '
              f'{delta_sign:>14.6e} {delta_free:>14.6e} {ratio:>10.6f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 4: THE BRIDGE -- delta_sign vs delta_free vs 1/B
    # ==================================================================
    print('\n  === PART 4: CORRELATIONS ===')
    print('  Test: delta_sign ~ C * delta_free? delta_sign ~ C / B?\n')

    ds = np.array([margin * g**2 / 4 for g in gammas])
    df = 1.0 / zeta_prime
    inv_B = 1.0 / B

    # Fit delta_sign vs delta_free
    log_ds = np.log(ds)
    log_df = np.log(df)
    slope1, intercept1 = np.polyfit(log_df, log_ds, 1)
    print(f'  delta_sign vs delta_free: slope={slope1:.3f}, '
          f'delta_sign ~ {np.exp(intercept1):.4f} * delta_free^{slope1:.3f}')

    # Fit delta_sign vs 1/B
    log_invB = np.log(inv_B)
    slope2, intercept2 = np.polyfit(log_invB, log_ds, 1)
    print(f'  delta_sign vs 1/B: slope={slope2:.3f}, '
          f'delta_sign ~ {np.exp(intercept2):.4f} * (1/B)^{slope2:.3f}')

    # Fit delta_sign vs gamma
    log_g = np.log(gammas)
    slope3, intercept3 = np.polyfit(log_g, log_ds, 1)
    print(f'  delta_sign vs gamma: slope={slope3:.3f}, '
          f'delta_sign ~ {np.exp(intercept3):.6e} * gamma^{slope3:.3f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 5: THE ACTUAL HADAMARD PRODUCT CONSTRAINT
    # ==================================================================
    print('\n  === PART 5: HADAMARD PRODUCT ZERO-FREE REGION ===')
    print('  If a NEW zero existed at 1/2+delta+ig, it must satisfy')
    print('  |product over known zeros| * |zeta\' correction| = 0.')
    print('  The Hadamard product constrains where new zeros can be.\n')

    # For a hypothetical zero at s_0 = 1/2 + delta + i*gamma_0:
    # zeta(s_0) = 0 requires:
    # |s_0 - 1| * prod_k |1 - s_0/rho_k| * |1 - s_0/conj(rho_k)| * ... = 0
    # Since s_0 - 1 != 0 and the Gamma factor is finite,
    # we need the Euler product to vanish:
    # prod_k |1 - s_0/rho_k| = 0
    # This requires s_0 = rho_k for some k (exact zero)
    # OR the product of the non-zero factors somehow vanishes

    # For s_0 NOT equal to any known rho_k:
    # |zeta(s_0)| = |s_0-1|^{-1} * prod |1 - s_0/rho_k|^{-1} * ... (this isn't right)

    # Actually, let's use the EXPLICIT zeta evaluation to bound:
    # For s = 1/2 + delta + i*t not at a known zero:
    # |zeta(s)| = product of distances to known zeros * correction

    # Simpler approach: near a known zero rho_j, the local expansion gives:
    # zeta(s) ~ zeta'(rho_j) * (s - rho_j) + (1/2)*zeta''(rho_j)*(s-rho_j)^2 + ...
    # For s = rho_j + delta (shifted off-line):
    # |zeta(s)| ~ |zeta'(rho_j)| * delta (for small delta)

    # For zeta to have ANOTHER zero at distance delta from rho_j:
    # Need the higher-order terms to cancel the linear term
    # |zeta''| * delta / (2*|zeta'|) ~ 1
    # delta ~ 2*|zeta'| / |zeta''|

    # Compute |zeta''| at each zero
    print(f'  Computing zeta\'\' at first 30 zeros...')
    zeta_pp = np.zeros(min(30, n_zeros))
    for k in range(min(30, n_zeros)):
        rho = mpmath.mpf('0.5') + mpmath.mpf(gammas[k]) * 1j
        zpp = abs(complex(mpmath.zeta(rho, derivative=2)))
        zeta_pp[k] = zpp

    print(f'\n  {"zero":>5} {"gamma":>10} {"|z\'|":>10} {"|z\'\'|":>10} '
          f'{"delta_had":>14} {"B":>10} {"delta_had*B":>12}')
    print('  ' + '-' * 76)

    delta_hads = []
    for k in range(min(30, n_zeros)):
        gamma = gammas[k]
        delta_had = 2 * zeta_prime[k] / zeta_pp[k] if zeta_pp[k] > 0 else float('inf')
        delta_hads.append(delta_had)
        product = delta_had * B[k]
        print(f'  {k+1:>5d} {gamma:>10.4f} {zeta_prime[k]:>10.4f} '
              f'{zeta_pp[k]:>10.4f} {delta_had:>14.6e} '
              f'{B[k]:>10.4f} {product:>12.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 6: THE CONNECTION
    # ==================================================================
    print('\n  === PART 6: THE CONNECTION ===')

    delta_hads = np.array(delta_hads[:30])
    Bs = B[:30]
    # Fit delta_had vs 1/B
    log_dh = np.log(delta_hads)
    log_Bs = np.log(Bs)
    slope_hB, int_hB = np.polyfit(log_Bs, log_dh, 1)
    print(f'  delta_hadamard vs B: delta ~ {np.exp(int_hB):.4f} * B^{slope_hB:.3f}')

    # Fit delta_had vs 1/|zeta'|
    log_zp = np.log(1.0/zeta_prime[:30])
    slope_hz, int_hz = np.polyfit(log_zp, log_dh, 1)
    print(f'  delta_hadamard vs 1/|zeta\'|: delta ~ {np.exp(int_hz):.4f} * (1/|z\'|)^{slope_hz:.3f}')

    # The product delta_had * B
    products = delta_hads * Bs
    print(f'\n  delta_had * B: mean={products.mean():.4f}, '
          f'std={products.std():.4f}, '
          f'min={products.min():.4f}, max={products.max():.4f}')
    print(f'  delta_had * |zeta\'|: mean={(delta_hads*zeta_prime[:30]).mean():.4f}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 66b VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
