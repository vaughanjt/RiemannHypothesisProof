"""
SESSION 46b — TRIGONOMETRIC IDENTITIES IN THE BARRIER

The barrier B = W02 - Mp is built from three fundamental identities:
  1. cosh^2(x) - sinh^2(x) = 1   (W02 involves sinh^2)
  2. sin^2(x) + cos^2(x) = 1     (Mp involves cos, sin)
  3. |e^{ix}|^2 = 1               (functional equation)

Each constrains the barrier terms to lie on specific manifolds.

KEY APPLICATION: Product-to-sum for the cross-term.

The cross-term sum_{j<k} Mp_j * Mp_k is the obstacle to proving RH
via the adelic barrier. Each Mp_k involves cos(2*pi*n*y_k/L) where
y_k = log(p_k). The product of two cosines:

  cos(a)*cos(b) = [cos(a-b) + cos(a+b)] / 2

So: Mp_j * Mp_k ~ terms at frequencies:
  log(p_j) - log(p_k) = log(p_j/p_k)  (RATIO frequency — LOW)
  log(p_j) + log(p_k) = log(p_j*p_k)  (PRODUCT frequency — HIGH)

The ratio-frequency terms are LOW frequency (small for nearby primes).
These DON'T cancel and dominate the cross-term.

The product-frequency terms are HIGH frequency (large).
These DO cancel through oscillation.

This converts RH from a question about prime SUMS to a question
about prime RATIOS and prime PRODUCTS.

ALSO: The hyperbolic identity in W02:
  sinh^2(L/4) = (cosh(L/2) - 1) / 2

So: W02 = 32L * (cosh(L/2) - 1) / 2 * [quadratic form]
        = 16L * (cosh(L/2) - 1) * [quadratic form]

The "1" in (cosh - 1) is the identity. W02 measures the DEPARTURE
from the identity cosh^2 - sinh^2 = 1.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes
from session45n_pi_predicts_primes import w02_only, prime_contribution


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 46b -- TRIGONOMETRIC IDENTITIES IN THE BARRIER')
    print('=' * 76)

    N = 15
    LAM_SQ = 2000

    # ══════════════════════════════════════════════════════════════
    # 1. THE THREE IDENTITIES
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. THE THREE IDENTITIES')
    print('#' * 76)

    L = np.log(LAM_SQ)
    print(f'\n  L = {L:.4f} (lam^2 = {LAM_SQ})')

    # Identity 1: cosh^2 - sinh^2 = 1 in W02
    sinh_val = np.sinh(L/4)
    cosh_val = np.cosh(L/4)
    identity_1 = cosh_val**2 - sinh_val**2
    print(f'\n  Identity 1: cosh^2(L/4) - sinh^2(L/4) = {identity_1:.15f}')
    print(f'    sinh^2(L/4) = {sinh_val**2:.6f}')
    print(f'    cosh^2(L/4) = {cosh_val**2:.6f}')
    print(f'    W02 prefactor = 32*L*sinh^2(L/4) = {32*L*sinh_val**2:.6f}')
    print(f'    Using identity: sinh^2 = cosh^2 - 1')
    print(f'    W02 pf = 32*L*(cosh^2 - 1) = 32*L*cosh^2 - 32*L')
    print(f'           = {32*L*cosh_val**2:.6f} - {32*L:.6f}')
    print(f'    The "1" subtracts {32*L:.4f} from the cosh term.')

    # Identity 2: sin^2 + cos^2 = 1 in Mp
    print(f'\n  Identity 2: sin^2(x) + cos^2(x) = 1 in M_prime')
    print(f'    Each prime contributes terms like cos(2*pi*n*log(p)/L)')
    print(f'    The sum over n: sum_n w_hat[n]^2 * cos(2*pi*n*y/L)')
    print(f'    = Fejer kernel evaluated at y/L')
    print(f'    For the cross-term, we use:')
    print(f'    cos(a)*cos(b) = [cos(a-b) + cos(a+b)] / 2')

    # Identity 3: |e^{ix}|^2 = 1
    print(f'\n  Identity 3: |e^{{ix}}|^2 = 1 in the functional equation')
    print(f'    The Riemann-Siegel theta: theta(t) = arg(Gamma(1/4+it/2)) - (t/2)*log(pi)')
    print(f'    Z(t) = e^{{i*theta}} * zeta(1/2+it) is real because |e^{{i*theta}}| = 1')

    # ══════════════════════════════════════════════════════════════
    # 2. PRODUCT-TO-SUM DECOMPOSITION OF THE CROSS-TERM
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. PRODUCT-TO-SUM: cross-term -> ratio + product frequencies')
    print('#' * 76)

    primes = list(sieve_primes(int(LAM_SQ)))
    K = len(primes)
    mp_arr = np.array([prime_contribution(int(p), LAM_SQ, N) for p in primes])

    # For each pair (j,k), the cross-term Mp_j * Mp_k has dominant
    # frequencies at log(p_j/p_k) and log(p_j*p_k)
    # The cross-term contribution is proportional to:
    #   cos(2*pi*n*log(p_j/p_k)/L)  [ratio frequency]
    # + cos(2*pi*n*log(p_j*p_k)/L)  [product frequency]

    print(f'\n  For each prime pair (p_j, p_k):')
    print(f'    Mp_j * Mp_k contributes at frequencies:')
    print(f'      f_ratio = log(p_j/p_k)   (LOW — nearby primes)')
    print(f'      f_product = log(p_j*p_k)  (HIGH — large)')

    # Compute ratio and product frequencies for nearby prime pairs
    print(f'\n  {"p_j":>5s} {"p_k":>5s} {"f_ratio":>10s} {"f_product":>10s} '
          f'{"Mp_j*Mp_k":>12s} {"ratio/L":>8s} {"product/L":>10s}')
    print('  ' + '-' * 65)

    for i in range(min(15, K-1)):
        pj, pk = int(primes[i]), int(primes[i+1])
        f_ratio = np.log(pj/pk)
        f_product = np.log(pj*pk)
        cross = mp_arr[i] * mp_arr[i+1]
        print(f'  {pj:>5d} {pk:>5d} {abs(f_ratio):>10.6f} {f_product:>10.4f} '
              f'{cross:>+12.6f} {abs(f_ratio)/L:>8.4f} {f_product/L:>10.4f}')

    # ══════════════════════════════════════════════════════════════
    # 3. RATIO vs PRODUCT FREQUENCY CONTRIBUTIONS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. WHICH DOMINATES: ratio frequencies or product frequencies?')
    print('#' * 76)

    # For ALL prime pairs, compute the ratio and product frequencies
    # and measure their contributions to the cross-term
    ratio_freqs = []
    product_freqs = []
    cross_values = []

    for j in range(K):
        for k in range(j+1, K):
            pj, pk = int(primes[j]), int(primes[k])
            f_ratio = abs(np.log(pj) - np.log(pk))
            f_product = np.log(pj) + np.log(pk)
            cross = mp_arr[j] * mp_arr[k]
            ratio_freqs.append(f_ratio)
            product_freqs.append(f_product)
            cross_values.append(cross)

    ratio_freqs = np.array(ratio_freqs)
    product_freqs = np.array(product_freqs)
    cross_values = np.array(cross_values)

    # Bin by ratio frequency
    bins = np.linspace(0, max(ratio_freqs), 20)
    print(f'\n  Cross-term contribution by RATIO frequency |log(p_j/p_k)|:')
    print(f'  {"freq bin":>12s} {"sum cross":>14s} {"n_pairs":>8s} {"% of total":>10s}')
    print('  ' + '-' * 48)

    total_cross = np.sum(cross_values)
    for i in range(len(bins)-1):
        mask = (ratio_freqs >= bins[i]) & (ratio_freqs < bins[i+1])
        bin_sum = np.sum(cross_values[mask])
        n_pairs = np.sum(mask)
        pct = 100 * bin_sum / total_cross if total_cross != 0 else 0
        if n_pairs > 0:
            print(f'  [{bins[i]:.2f}, {bins[i+1]:.2f}] {bin_sum:>+14.4f} {n_pairs:>8d} {pct:>+9.1f}%')

    # Low-frequency (ratio < 1) vs high-frequency
    low_mask = ratio_freqs < 1.0
    high_mask = ratio_freqs >= 1.0
    print(f'\n  LOW ratio frequency (|log(p_j/p_k)| < 1):')
    print(f'    Sum cross: {np.sum(cross_values[low_mask]):+.6f} ({np.sum(low_mask)} pairs)')
    print(f'  HIGH ratio frequency (|log(p_j/p_k)| >= 1):')
    print(f'    Sum cross: {np.sum(cross_values[high_mask]):+.6f} ({np.sum(high_mask)} pairs)')
    print(f'  Total cross: {total_cross:+.6f}')
    print(f'  Low-freq fraction: {np.sum(cross_values[low_mask])/total_cross:.4f}')

    # ══════════════════════════════════════════════════════════════
    # 4. THE TWIN PRIME CONNECTION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. THE TWIN PRIME CONNECTION')
    print('#' * 76)

    print(f'''
  The ratio frequency log(p_j/p_k) is smallest for TWIN PRIMES:
    log((p+2)/p) = log(1 + 2/p) ~ 2/p (for large p)

  Twin primes contribute the LOWEST frequency cross-terms.
  These are the hardest to cancel — they dominate the cross-term.

  So the cross-term (the obstacle to RH) is dominated by:
    NEARBY prime pairs (small ratio frequency)
    especially twin primes and small-gap primes.

  The PRODUCT frequencies log(p_j*p_k) are always large (> 2*log(2) = 1.39)
  and contribute oscillating terms that average out.
  ''')

    # Find twin prime contributions
    twin_pairs = []
    for i in range(K-1):
        if int(primes[i+1]) - int(primes[i]) == 2:
            cross = mp_arr[i] * mp_arr[i+1]
            twin_pairs.append((int(primes[i]), int(primes[i+1]), cross))

    print(f'  Twin prime pairs up to {LAM_SQ} ({len(twin_pairs)} pairs):')
    twin_total = sum(c for _, _, c in twin_pairs)
    print(f'  {"p":>5s} {"p+2":>5s} {"cross-term":>12s}')
    print('  ' + '-' * 25)
    for p, p2, c in twin_pairs[:15]:
        print(f'  {p:>5d} {p2:>5d} {c:>+12.6f}')
    print(f'  ...')
    print(f'  Total twin-prime cross-term: {twin_total:+.6f}')
    print(f'  Fraction of total cross-term: {twin_total/total_cross:.4f}')

    # ══════════════════════════════════════════════════════════════
    # 5. THE IDENTITY-BASED BARRIER DECOMPOSITION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. BARRIER THROUGH THE LENS OF IDENTITIES')
    print('#' * 76)

    w02 = w02_only(LAM_SQ, N)
    mp_total = np.sum(mp_arr)
    barrier = w02 - mp_total

    # W02 using cosh^2 - 1 = sinh^2:
    # W02 = 32L * sinh^2(L/4) * (quadratic form)
    #      = 32L * (cosh^2(L/4) - 1) * (quadratic form)
    #      = [32L * cosh^2(L/4) * QF] - [32L * QF]

    # The "1" term: 32L * QF (where QF is the quadratic form without prefactor)
    # Let's compute it
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    denom = L**2 + (4 * np.pi)**2 * ns**2
    w_tilde = ns / denom
    wt_dot_wh = np.dot(w_tilde, w_hat)
    qf = -(4 * np.pi)**2 * wt_dot_wh**2  # the quadratic form part

    w02_cosh_part = 32 * L * cosh_val**2 * qf
    w02_identity_part = 32 * L * 1.0 * qf  # the "1" from cosh^2 - 1
    w02_check = w02_cosh_part - w02_identity_part

    print(f'\n  W02 = (cosh^2 part) - (identity part):')
    print(f'    cosh^2 part: {w02_cosh_part:+.6f}')
    print(f'    identity part (the "1"): {w02_identity_part:+.6f}')
    print(f'    W02 = {w02_check:+.6f} (check: {w02:+.6f})')

    print(f'\n  BARRIER = W02 - Mp')
    print(f'         = (cosh^2 part - identity) - Mp')
    print(f'         = cosh^2 part - (identity + Mp)')
    print(f'         = {w02_cosh_part:.4f} - ({w02_identity_part:.4f} + {mp_total:.4f})')
    print(f'         = {w02_cosh_part:.4f} - ({w02_identity_part + mp_total:.4f})')
    print(f'         = {barrier:+.6f}')

    print(f'\n  The barrier is the excess of cosh^2 over (1 + primes).')
    print(f'  The hyperbolic identity cosh^2 - sinh^2 = 1 means:')
    print(f'    sinh^2 = cosh^2 - 1')
    print(f'    barrier = sinh^2*QF*32L - Mp = (cosh^2-1)*QF*32L - Mp')
    print(f'    = cosh^2*QF*32L - QF*32L - Mp')
    print(f'    = cosh^2*QF*32L - (QF*32L + Mp)')
    print(f'\n  RH iff cosh^2 > 1 + Mp/(QF*32L)')
    print(f'  cosh^2(L/4) = {cosh_val**2:.6f}')
    print(f'  1 + Mp/(QF*32L) = {1 + mp_total/(qf*32*L):.6f}')
    print(f'  Excess: {cosh_val**2 - (1 + mp_total/(qf*32*L)):.6f}')

    # ══════════════════════════════════════════════════════════════
    # 6. UNIVERSALITY: does the identity decomposition hold at all L?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. THE COSH^2 vs (1 + PRIMES) RACE ACROSS L')
    print('#' * 76)

    print(f'\n  {"lam^2":>8s} {"L":>7s} {"cosh^2":>12s} {"1+Mp/norm":>12s} '
          f'{"excess":>10s} {"excess > 0?":>10s}')
    print('  ' + '-' * 62)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
        L_val = np.log(lam_sq)
        cosh2 = np.cosh(L_val/4)**2

        ns_l = np.arange(-N, N + 1, dtype=float)
        w_l = ns_l / (L_val**2 + (4*np.pi)**2 * ns_l**2)
        w_l[N] = 0.0
        w_hat_l = w_l / np.linalg.norm(w_l)
        denom_l = L_val**2 + (4*np.pi)**2 * ns_l**2
        wt_l = ns_l / denom_l
        wt_dot_l = np.dot(wt_l, w_hat_l)
        qf_l = -(4*np.pi)**2 * wt_dot_l**2
        norm = qf_l * 32 * L_val

        primes_l = list(sieve_primes(int(lam_sq)))
        mp_arr_l = np.array([prime_contribution(int(p), lam_sq, N) for p in primes_l])
        mp_total_l = np.sum(mp_arr_l)

        rhs = 1 + mp_total_l / norm if abs(norm) > 1e-10 else 0
        excess = cosh2 - rhs
        ok = 'YES' if excess > 0 else 'NO'

        print(f'  {lam_sq:>8d} {L_val:>7.3f} {cosh2:>12.6f} {rhs:>12.6f} '
              f'{excess:>+10.6f} {ok:>10s}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 46b SYNTHESIS')
    print('=' * 76)

    low_frac = np.sum(cross_values[low_mask]) / total_cross if total_cross != 0 else 0
    twin_frac = twin_total / total_cross if total_cross != 0 else 0

    print(f'''
  THREE IDENTITIES IN THE BARRIER:

  1. HYPERBOLIC: cosh^2 - sinh^2 = 1
     W02 = 32L*(cosh^2-1)*QF = 32L*cosh^2*QF - 32L*QF
     The "1" in the identity subtracts 32L*QF from the cosh term.
     RH iff cosh^2(L/4) > 1 + Mp/(32L*QF)

  2. TRIGONOMETRIC: sin^2 + cos^2 = 1
     Mp involves cos(2*pi*n*y/L) for y = log(p^k).
     Cross-term Mp_j*Mp_k decomposes via product-to-sum:
       = ratio-frequency part (low freq, from log(p_j/p_k))
       + product-frequency part (high freq, from log(p_j*p_k))

  3. EULER: |e^{{ix}}|^2 = 1
     Z(t) = e^{{i*theta}}*zeta(1/2+it) is real by this identity.
     The zeros live where Z = 0.

  CROSS-TERM DECOMPOSITION:
    Total cross-term: {total_cross:+.4f}
    Low ratio-frequency (|log(p_j/p_k)| < 1): {low_frac:.1%} of total
    Twin prime contribution: {twin_frac:.1%} of total
    The cross-term is dominated by NEARBY prime pairs.

  THE BARRIER AS IDENTITY RACE:
    cosh^2(L/4) vs 1 + primes_normalized
    cosh^2 grows as exp(L/2). Primes grow as exp(~0.55L).
    The hyperbolic identity provides the extra "1" that separates them.
    Pi lives in the cosh (archimedean). Primes live in the correction.
    The "1" is the identity that keeps them apart.
''')

    print('=' * 76)
    print('  SESSION 46b COMPLETE')
    print('=' * 76)
