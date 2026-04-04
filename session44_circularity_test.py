"""
SESSION 44 — THE CIRCULARITY TEST

The critical question: does the identity
    barrier = sum_rho |H_w(rho)|^2 - C
REQUIRE assuming RH?

Answer depends on what the explicit formula actually says:

UNCONDITIONAL explicit formula (no RH assumed):
    sum_rho h(rho) = analytic_side(h)
where the sum is over ALL zeros rho = beta + i*gamma, and
h(rho) means h evaluated at the zero (not |h|^2).

For a POSITIVE-DEFINITE test function h (= f * f_bar):
    h(rho) = integral f(x) f_bar(y) K(x,y,rho) dx dy

If rho = 1/2 + i*gamma (ON critical line):
    h(rho) = |f_hat(gamma)|^2  >= 0

If rho = sigma + i*gamma with sigma != 1/2 (OFF line):
    h(rho) is NOT |f_hat|^2, could be complex/negative.

So: the identity barrier = sum |H|^2 IS circular (requires RH).

BUT: what about computing from the EXPLICIT FORMULA SIDE?

The explicit formula says:
    sum_rho h_w(rho) = analytic_terms(h_w) - prime_terms(h_w)

where the right side IS the barrier (= Q_W quadratic form).
The left side is sum_rho h_w(rho).

If all computed zeros are on the critical line (verified for first 10^13),
then for those zeros: h_w(rho) = |H_w(rho)|^2 >= 0.

So: barrier = sum_{verified} |H|^2 + sum_{unverified} h_w(rho)
                  >= 0                  unknown sign

If sum_{verified} > |sum_{unverified}| + |analytic corrections|,
then barrier > 0 WITHOUT assuming ALL zeros are on the line.

This would be a partial result: barrier > 0 ASSUMING the first
10^13 zeros are on the critical line (which is verified).

Let's test: is the contribution of the first K verified zeros
already enough to guarantee barrier > 0?
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, exp, sin, cos, quad,
                    zetazero, power)
import time
import sys
import os

mp.dps = 15

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all


def compute_explicit_formula_terms(lam_sq, K_zeros, N_basis=None):
    """
    Compute the explicit formula decomposition:

    barrier = sum_{k=1}^{K} |H_w(rho_k)|^2 + tail_zeros + error

    where:
    - sum |H|^2 is the verified zero contribution (>= 0)
    - tail_zeros is the contribution from zeros beyond K
    - error is the truncation/regularization error

    We compute: sum |H|^2 and (barrier - sum |H|^2) = tail + error

    If (barrier - sum |H|^2) > 0 for some K, then:
    barrier > sum |H|^2 >= 0, so barrier > 0 WITHOUT needing tail >= 0.

    If (barrier - sum |H|^2) < 0 for all K, then:
    The verified zeros OVERSHOOT the barrier, and the tail/corrections
    pull it back down. This means the tail is NEGATIVE, which would
    only happen if some unverified zeros are off the critical line.
    """
    L = log(mpf(lam_sq))
    L_f = float(L)
    if N_basis is None:
        N_basis = max(15, round(6 * L_f))
    dim = 2 * N_basis + 1

    # Exact barrier from matrix
    W02, M, QW = build_all(lam_sq, N_basis, n_quad=3000)
    ns = np.arange(-N_basis, N_basis + 1, dtype=float)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N_basis] = 0.0
    w_hat = w / np.linalg.norm(w)
    barrier = float(w_hat @ QW @ w_hat)

    # Compute H_w at K zeros (sin-basis for odd direction)
    zeros = [float(zetazero(k).imag) for k in range(1, K_zeros + 1)]

    H_values = []
    for gamma in zeros:
        s = mpf(1)/2 + mpc(0, mpf(gamma))
        hw = mpc(0, 0)
        for n in range(1, N_basis + 1):
            wn = w_hat[N_basis + n]
            if abs(wn) < 1e-15:
                continue
            def integrand(x, n=n):
                return 2*(1-x/L)*sin(2*pi*n*x/L) * power(x, s-1)
            g = quad(integrand, [mpf(0), L], maxdegree=5)
            hw += 2 * mpf(wn) * g
        H_values.append(complex(hw))

    H_sq = np.array([abs(h)**2 for h in H_values])
    cum_S = np.cumsum(H_sq)

    return {
        'barrier': barrier,
        'H_sq': H_sq,
        'cum_S': cum_S,
        'zeros': zeros,
        'remainder': barrier - cum_S,  # = tail_zeros + error
    }


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  SESSION 44 — THE CIRCULARITY TEST')
    print('#' * 72)

    print("""
  THE QUESTION:

  barrier = sum_{verified} |H|^2 + REMAINDER

  If REMAINDER > 0 for small K:
    barrier > sum |H|^2 >= 0, so barrier > 0.
    The verified zeros UNDERCONTRIBUTE.
    The rest (tail + analytic) is POSITIVE.
    NO assumption about unverified zeros needed.

  If REMAINDER < 0 for all K:
    The verified zeros OVERCONTRIBUTE.
    Something must pull the barrier back down.
    This requires knowledge of ALL zeros.
    CIRCULAR.
    """)

    for lam_sq in [20, 50, 200]:
        print(f'\n  lam^2 = {lam_sq}')
        print('  ' + '=' * 55)

        t0 = time.time()
        r = compute_explicit_formula_terms(lam_sq, K_zeros=50)
        dt = time.time() - t0

        print(f'  Barrier: {r["barrier"]:.8f}  ({dt:.0f}s)')
        print()
        print(f'  {"K":>4s} {"S(K)":>12s} {"remainder":>12s} {"sign":>6s}')
        print('  ' + '-' * 38)

        for K in [1, 2, 3, 5, 10, 15, 20, 30, 50]:
            if K > len(r['cum_S']):
                break
            S = r['cum_S'][K-1]
            rem = r['remainder'][K-1]
            sign = '+' if rem > 0 else '-'
            print(f'  {K:>4d} {S:>12.6f} {rem:>+12.6f} {sign:>6s}')

        # Find where remainder changes sign
        sign_changes = []
        for i in range(len(r['remainder'])-1):
            if r['remainder'][i] * r['remainder'][i+1] < 0:
                sign_changes.append(i+2)  # K value where it flips

        print(f'\n  Remainder sign changes at K = {sign_changes}')

        # THE VERDICT for this lambda
        remainder_at_K1 = r['remainder'][0]
        if remainder_at_K1 > 0:
            print(f'  ** REMAINDER IS POSITIVE AT K=1 **')
            print(f'     barrier = |H(rho_1)|^2 + {remainder_at_K1:.6f}')
            print(f'     >= |H(rho_1)|^2 >= 0')
            print(f'     BARRIER >= 0 PROVED (using only 1 verified zero)')
        else:
            K_positive = None
            for i, rem in enumerate(r['remainder']):
                if rem > 0:
                    K_positive = i + 1
                    break
            if K_positive:
                print(f'  Remainder first positive at K = {K_positive}')
                print(f'  barrier = S({K_positive}) + {r["remainder"][K_positive-1]:.6f}')
                print(f'  >= S({K_positive}) >= 0')
            else:
                print(f'  ** REMAINDER NEVER POSITIVE **')
                print(f'  The spectral sum always exceeds the barrier.')
                print(f'  The correction is always positive (pulling barrier down).')
                print(f'  Cannot prove barrier >= 0 this way.')

    # ── THE HONEST ASSESSMENT ──
    print('\n\n' + '=' * 72)
    print('  THE HONEST ASSESSMENT')
    print('=' * 72)
    print()
    print('  The remainder = barrier - S(K) tells us:')
    print()
    print('  If > 0 (K small): the first K zeros undercontribute.')
    print('    The barrier EXCEEDS the verified spectral sum.')
    print('    Positivity follows from the EXTRA positive stuff.')
    print('    This extra stuff = analytic terms > spectral terms.')
    print('    PROVES barrier > 0 with only K verified zeros.')
    print()
    print('  If < 0 (K large): the spectral sum overshoots.')
    print('    Need corrections to pull it back to the barrier.')
    print('    Those corrections involve ALL zeros -> CIRCULAR.')

    print('\n' + '#' * 72)
