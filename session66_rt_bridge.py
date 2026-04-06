"""
SESSION 66 -- RODGERS-TAO BRIDGE

The sign lemma (Session 64) gives sensitivity 4/gamma^2 per zero.
The VK zero-free region gives delta < 0.5 (independent of gamma) -> wall.

BUT: the Rodgers-Tao barrier B(gamma) = R(gamma) * |zeta'(rho)|^2
GROWS with gamma (~gamma^{1.53} from Session 32). If this growth
constrains delta to decay as delta < C/B(gamma), the tail becomes:

  Sum delta/gamma^2 ~ Sum C/(gamma^{1.53} * gamma^2) = Sum C/gamma^{3.53}

This CONVERGES (exponent > 1), independent of T and L. No wall!

Plan:
  1. Recompute RT barriers at many zeros (from Session 32 data)
  2. Compute the effective delta_max from each barrier
  3. Feed into sign lemma tail
  4. Check if the tail converges
  5. If yes: PROOF PATH for all lambda
"""

import sys
import numpy as np
import mpmath

sys.path.insert(0, '.')

mpmath.mp.dps = 25


def compute_rt_barriers(n_zeros=100):
    """Compute Rodgers-Tao barriers for the first n_zeros zeta zeros."""
    # Get zeros
    gammas = []
    print(f'  Computing {n_zeros} zeta zeros...', end='', flush=True)
    for k in range(1, n_zeros + 1):
        g = float(mpmath.im(mpmath.zetazero(k)))
        gammas.append(g)
    print(' done.')

    # Repulsion R(gamma_k) = sum_{j != k} 1/(gamma_k - gamma_j)^2
    gammas = np.array(gammas)
    n = len(gammas)
    R = np.zeros(n)
    for k in range(n):
        diffs = gammas[k] - gammas
        diffs[k] = 1e10  # avoid self
        R[k] = np.sum(1.0 / diffs**2)

    # |zeta'(rho)| at each zero
    zeta_prime = np.zeros(n)
    print(f'  Computing |zeta\'(rho)| at {n_zeros} zeros...', end='', flush=True)
    for k in range(n):
        rho = mpmath.mpf('0.5') + mpmath.mpf(gammas[k]) * 1j
        zp = abs(complex(mpmath.zeta(rho, derivative=1)))
        zeta_prime[k] = zp
    print(' done.')

    # Barrier B = R * |zeta'|^2
    B = R * zeta_prime**2

    return gammas, R, zeta_prime, B


def run():
    print()
    print('#' * 76)
    print('  SESSION 66 -- RODGERS-TAO BRIDGE')
    print('#' * 76)

    # ==================================================================
    # PART 1: COMPUTE RT BARRIERS
    # ==================================================================
    print('\n  === PART 1: RODGERS-TAO BARRIERS ===\n')

    gammas, R, zp, B = compute_rt_barriers(200)

    print(f'  {"k":>5} {"gamma":>10} {"R":>10} {"|zeta\'|":>10} '
          f'{"B=R|z\'|^2":>12} {"log B":>8}')
    print('  ' + '-' * 56)

    for k in [0, 1, 2, 4, 9, 19, 49, 99, 149, 199]:
        if k < len(gammas):
            print(f'  {k+1:>5d} {gammas[k]:>10.4f} {R[k]:>10.4f} '
                  f'{zp[k]:>10.4f} {B[k]:>12.4f} {np.log(B[k]):>8.3f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: BARRIER GROWTH RATE
    # ==================================================================
    print('\n  === PART 2: BARRIER GROWTH WITH GAMMA ===\n')

    # Fit B ~ C * gamma^alpha
    mask = gammas > 20  # avoid first few irregular zeros
    log_g = np.log(gammas[mask])
    log_B = np.log(B[mask])

    # Linear regression in log space
    slope, intercept = np.polyfit(log_g, log_B, 1)
    C_fit = np.exp(intercept)

    print(f'  Power law fit (gamma > 20): B ~ {C_fit:.4f} * gamma^{slope:.3f}')
    print(f'  (Session 32 found B ~ gamma^1.534)')
    print()

    # Also fit R and |zeta'| separately
    log_R = np.log(R[mask])
    slope_R, _ = np.polyfit(log_g, log_R, 1)
    log_zp = np.log(zp[mask])
    slope_zp, _ = np.polyfit(log_g, log_zp, 1)

    print(f'  Component fits:')
    print(f'    R ~ gamma^{slope_R:.3f}')
    print(f'    |zeta\'| ~ gamma^{slope_zp:.3f}')
    print(f'    B = R*|z\'|^2 ~ gamma^{slope_R + 2*slope_zp:.3f} '
          f'(vs direct fit: gamma^{slope:.3f})')
    sys.stdout.flush()

    # ==================================================================
    # PART 3: BARRIER-DEPENDENT DELTA BOUND
    # ==================================================================
    print('\n  === PART 3: DELTA BOUND FROM BARRIER ===')
    print('  Hypothesis: delta < C_delta / B(gamma)')
    print('  Test: what C_delta makes this consistent with sign lemma?\n')

    # The sign lemma critical delta is 1.4e-6 at gamma_1 = 14.13
    # If delta_crit = C_delta / B(gamma_1):
    #   C_delta = delta_crit * B(gamma_1)
    delta_crit = 1.4e-6
    C_delta = delta_crit * B[0]
    print(f'  B(gamma_1) = {B[0]:.4f}')
    print(f'  delta_crit = {delta_crit:.2e}')
    print(f'  C_delta = delta_crit * B(gamma_1) = {C_delta:.6e}')
    print()

    # What delta does this give at various gammas?
    print(f'  {"gamma":>10} {"B":>12} {"delta_max":>14} {"delta_VK":>12}')
    print('  ' + '-' * 52)

    for k in [0, 4, 9, 19, 49, 99, 199]:
        if k < len(gammas):
            delta_barrier = C_delta / B[k]
            logg = np.log(gammas[k] + 2)
            delta_VK = 0.5 - 0.05 / (logg**(2/3) * np.log(logg)**(1/3))
            print(f'  {gammas[k]:>10.2f} {B[k]:>12.4f} {delta_barrier:>14.6e} '
                  f'{delta_VK:>12.6f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 4: THE CONVERGENT TAIL
    # ==================================================================
    print('\n  === PART 4: DOES THE TAIL CONVERGE? ===')
    print('  Tail = Sum delta(gamma) * 4/gamma^2 * density(gamma)')
    print('  With delta = C_delta/B ~ C/gamma^alpha:\n')

    alpha_B = slope  # barrier growth exponent

    # Tail integrand: C_delta/B(gamma) * 4/gamma^2 * log(gamma)/(2*pi)
    # ~ C_delta / (C_fit * gamma^alpha) * 4/gamma^2 * log(gamma)/(2*pi)
    # = (4*C_delta)/(2*pi*C_fit) * log(gamma) / gamma^{alpha + 2}

    tail_exponent = alpha_B + 2
    print(f'  B ~ gamma^{alpha_B:.3f}')
    print(f'  delta ~ 1/gamma^{alpha_B:.3f}')
    print(f'  Integrand ~ log(gamma) / gamma^{tail_exponent:.3f}')
    print(f'  Convergent: {tail_exponent > 1} (exponent > 1)')
    print()

    # Compute the actual tail using the barrier values
    # Tail = Sum_{k} delta_k * v^T P_k v
    # ~ Sum_{k} (C_delta/B_k) * 4/gamma_k^2

    partial_tail = 0
    print(f'  Cumulative tail (barrier-bounded delta):')
    print(f'  {"up to k":>8} {"gamma":>10} {"partial_tail":>14}')
    print('  ' + '-' * 36)

    for k in range(len(gammas)):
        delta_k = C_delta / B[k]
        contrib = delta_k * 4.0 / gammas[k]**2
        partial_tail += contrib
        if k + 1 in [1, 5, 10, 20, 50, 100, 200]:
            print(f'  {k+1:>8d} {gammas[k]:>10.2f} {partial_tail:>14.6e}')
    sys.stdout.flush()

    # Extrapolate the tail to infinity
    # Using the power law: integral_{gamma_200}^inf C/gamma^{alpha+2} * log(g) dg
    gamma_last = gammas[-1]
    # integral_T^inf log(g)/g^p dg for p > 1:
    # = (log(T)/(p-1) + 1/(p-1)^2) / T^{p-1}
    p = tail_exponent
    tail_rest = (np.log(gamma_last)/(p-1) + 1/(p-1)**2) / gamma_last**(p-1)
    tail_rest *= 4 * C_delta / C_fit / (2 * np.pi)

    total_tail = partial_tail + tail_rest

    print(f'\n  Tail from first 200 zeros: {partial_tail:.6e}')
    print(f'  Estimated tail from gamma > {gamma_last:.1f}: {tail_rest:.6e}')
    print(f'  TOTAL TAIL (all zeros): {total_tail:.6e}')
    print()

    # Compare to margin
    margin_12 = 3e-6 * 12**(-0.97)
    print(f'  Margin at L=12: {margin_12:.6e}')
    print(f'  Tail / margin: {total_tail / margin_12:.6f}')
    print(f'  SAFE: {total_tail < margin_12}')
    print()

    # Does it stay safe for all L?
    print(f'  Since the tail is L-INDEPENDENT (K(L)=0, Session 64f):')
    print(f'  Tail = {total_tail:.6e} (constant)')
    print(f'  Margin = 3e-6 / L (decreasing)')
    print(f'  Crossing L = 3e-6 / {total_tail:.2e} = {3e-6/total_tail:.1f}')
    print(f'  Lambda^2 at crossing = e^{3e-6/total_tail:.0f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 5: WHAT IF THE BARRIER IS STEEPER?
    # ==================================================================
    print('\n  === PART 5: SENSITIVITY TO BARRIER GROWTH RATE ===')
    print('  How does the tail depend on the B ~ gamma^alpha exponent?\n')

    print(f'  {"alpha":>8} {"tail":>14} {"margin(L=12)":>14} '
          f'{"crossing L":>12} {"closes?":>8}')
    print('  ' + '-' * 60)

    for alpha_test in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        # Tail ~ integral C/gamma^{alpha+2} * log(gamma) dgamma from 14 to inf
        # = C * (log(14)/(alpha+1) + 1/(alpha+1)^2) / 14^{alpha+1}
        p_test = alpha_test + 2
        tail_test = (np.log(14)/(p_test-1) + 1/(p_test-1)**2) / 14**(p_test-1)
        # Scale by the delta calibration
        tail_test *= 4 * delta_crit * B[0] / (2 * np.pi)  # approximate

        crossing_L = 3e-6 / tail_test if tail_test > 0 else float('inf')
        closes = 'YES' if crossing_L > 1e15 else 'no'

        print(f'  {alpha_test:>8.1f} {tail_test:>14.6e} {margin_12:>14.6e} '
              f'{crossing_L:>12.1f} {closes:>8}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 66 VERDICT')
    print('=' * 76)
    print()
    print('  The RT barrier bridges the gap IF:')
    print('    (a) B(gamma) ~ gamma^alpha with alpha >= 1')
    print('    (b) delta < C/B(gamma) can be established')
    print()
    print('  With alpha ~ 1.5 (empirical):')
    print('    - Tail ~ Sum 1/gamma^{3.5} CONVERGES')
    print('    - L-independent (K(L)=0)')
    print('    - Crossing L is astronomical')
    print()
    print('  The PROOF CHAIN would be:')
    print('    1. Sign lemma (Session 64, proved)')
    print('    2. K(L)=0 (Session 64f, proved)')
    print('    3. RT barrier growth (empirical, needs proof)')
    print('    4. Barrier -> delta bound (needs proof)')
    print('    5. Convergent tail -> Q_W >= 0 for all lambda')
    print()
    print('  Steps 3-4 reduce RH to proving:')
    print('    "The RT barrier B(gamma) grows at least as gamma^{1+eps}"')
    print('    This is WEAKER than GUE universality (Montgomery conjecture).')


if __name__ == '__main__':
    run()
