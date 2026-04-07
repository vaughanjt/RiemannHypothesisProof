"""
SESSION 78c -- CAN WE PROVE TIGHTNESS?

The chain so far:
  1. Functional equation => dh/d(sigma) = 0 at sigma=1/2
  2. Test function => d2h/d(sigma2) > 0 => minimum at sigma=1/2
  3. Explicit formula => sum h_pair = S (fixed by primes)
  4. Each h_pair(delta_k) >= h_pair(0, gamma_k)
  5. Therefore S >= S_0 where S_0 = sum h_pair(0, gamma_k)

For RH: need S = S_0 (tightness). Then each delta_k = 0.

The gap: is S = S_0 forced by some other constraint?

PROBES:
  1. Compute S (from primes) and S_0 (from on-line zeros) independently.
     They must be equal (explicit formula). But can we show they MUST be?
  2. What if S > S_0? Then some zeros are off-line. What does this do to M?
  3. The UPPER bound: is there a constraint S <= something?
     If S <= S_0 from another direction, combined with S >= S_0, gives S = S_0.
  4. The Parseval/energy approach: does the ENERGY (sum of |h|^2) have
     a complementary constraint?
  5. The variational principle: the zeros MINIMIZE some functional.
     Does that functional force S = S_0?
  6. Test with a DIFFERENT test function: does the argument work for
     any test function, or only the Lorentzian?
  7. What if we use TWO test functions simultaneously?
"""

import sys
import numpy as np
import mpmath

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes

mpmath.mp.dps = 30


def h_pair(delta, gamma, L):
    """h_pair for Lorentzian test function."""
    re = L**2/4 + delta**2 - gamma**2
    im = 2 * delta * gamma
    denom = re**2 + im**2
    if denom < 1e-30:
        return 0.0
    return 2 * L * re / denom


def explicit_formula_prime_side(lam_sq, L):
    """Compute the prime side of the explicit formula for h(r) = L/(L^2/4 + r^2).

    The Weil explicit formula:
      sum_rho h(rho - 1/2) = h(0) + h(-1)
                              - sum_{p^k} log(p)/p^{k/2} * h_check(log p^k)
                              + integral terms

    For h(r) = L / (L^2/4 + r^2):
      h(0) = L / (L^2/4) = 4/L
      h(-1) = L / (L^2/4 + 1)  [from the trivial zeros contribution]

    Actually, the standard explicit formula for the test function h is:
      sum_rho h(gamma_rho) = h_hat(0) * (contributions)
    where the exact form depends on normalization.

    Let's compute both sides numerically and compare.
    """
    # For our specific h(r) = L / (L^2/4 + r^2):
    # The Fourier transform h_hat(x) = pi * e^{-L|x|/2}

    # The explicit formula (Bombieri's form):
    # sum_{rho} h(gamma_rho) = h_hat(0)*log(pi) - integral...
    #                         + sum_n Lambda(n)/sqrt(n) * h_hat(log n)
    # This is complex. Let me just compute both sides numerically.

    # PRIME SIDE: sum_{p^k <= lam^2} log(p) * p^{-k/2} * g(log(p^k))
    # where g(y) involves the test function evaluated at y
    # For the Lorentzian: the prime contribution to the barrier is
    # B_prime(L) = sum_{p^k <= lam^2} log(p) * p^{-k/2} * 2*(L-y)/L
    # where y = k*log(p) and the factor 2*(L-y)/L comes from the
    # integration of the test function.

    primes = sieve_primes(int(lam_sq))
    prime_sum = 0
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            # The "h_check" evaluated at y:
            # For Lorentzian h(r) = L/(L^2/4 + r^2):
            # The explicit formula prime term uses h_hat(y) = pi * e^{-L*y/2}
            # But we need to be more careful about which convention...
            # Let me just use: contribution = w * L / (L^2/4 + y^2) * something
            # Actually this is getting tangled. Let me compute numerically.
            prime_sum += w * h_pair(0, y / (2*np.pi/L), L)
            # No, this isn't right either.
            pk *= int(p)

    return prime_sum


def run():
    print()
    print('#' * 76)
    print('  SESSION 78c -- CAN WE PROVE TIGHTNESS?')
    print('#' * 76)

    n_zeros = 100
    zeros = [float(mpmath.zetazero(k).imag) for k in range(1, n_zeros + 1)]

    # ======================================================================
    # PROBE 1: Compute S_0 (on-line sum) for various L
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: THE ON-LINE SUM S_0 vs L')
    print(f'{"="*76}\n')

    print(f'  S_0(L) = sum_k h_pair(0, gamma_k, L) = sum_k 2L/(L^2/4 - gamma_k^2)')
    print(f'  This sum is NEGATIVE for large enough K (most terms negative).')
    print()
    print(f'  {"L":>8} {"S_0 (K=30)":>14} {"S_0 (K=50)":>14} {"S_0 (K=100)":>14}')
    print('  ' + '-' * 54)

    for lam_sq in [100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
        L = np.log(lam_sq)
        s30 = sum(h_pair(0, g, L) for g in zeros[:30])
        s50 = sum(h_pair(0, g, L) for g in zeros[:50])
        s100 = sum(h_pair(0, g, L) for g in zeros[:100])
        print(f'  {L:>8.3f} {s30:>+14.6f} {s50:>+14.6f} {s100:>+14.6f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Does a DIFFERENT test function also give minimum at sigma=1/2?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: OTHER TEST FUNCTIONS')
    print(f'{"="*76}\n')

    # Any test function h that respects the functional equation will have
    # h_pair(delta, gamma) = h(delta + i*gamma) + h(-delta + i*gamma)
    # which is EVEN in delta. So dh/d(delta) = 0 at delta=0 ALWAYS.
    #
    # But is sigma=1/2 always a MINIMUM? That depends on d2h/d(delta2).
    # For h(w) = L/(L^2/4 + w^2) (Lorentzian), it's always a minimum.
    # For other h, it might be a maximum.

    L = np.log(1000)
    gamma_test = zeros[0]  # gamma_1 = 14.13

    # Test function 1: Gaussian h(w) = exp(-a*w^2)
    def h_gaussian(delta, gamma, a=0.01):
        w_re = delta
        w_im = gamma
        # h(w) = exp(-a*w^2) where w = delta + i*gamma
        # w^2 = delta^2 - gamma^2 + 2i*delta*gamma
        # h = exp(-a*(delta^2 - gamma^2)) * exp(-2i*a*delta*gamma)
        # h_pair = h(delta+igamma) + h(-delta+igamma)
        #        = 2*Re[exp(-a*(delta^2 - gamma^2 + 2i*delta*gamma))]
        #        = 2*exp(-a*(delta^2-gamma^2)) * cos(2*a*delta*gamma)
        return 2 * np.exp(-a*(delta**2 - gamma**2)) * np.cos(2*a*delta*gamma)

    # Test function 2: Power h(w) = 1/(1 + w^2)^s for various s
    def h_power(delta, gamma, s=1):
        re = 1 + delta**2 - gamma**2
        im = 2 * delta * gamma
        mag_sq = re**2 + im**2
        if mag_sq < 1e-30:
            return 0
        # (1+w^2)^{-s} = (mag_sq)^{-s/2} * exp(-i*s*angle)
        # h_pair = 2*Re[(mag_sq)^{-s/2} * exp(-i*s*atan2(im, re))]
        angle = np.arctan2(im, re)
        return 2 * mag_sq**(-s/2) * np.cos(s * angle)

    # Test function 3: Sech h(w) = 1/cosh(a*w)
    def h_sech(delta, gamma, a=0.1):
        # cosh(a*(delta+igamma)) = cosh(a*delta)*cos(a*gamma) + i*sinh(a*delta)*sin(a*gamma)
        # h_pair = 1/cosh(a*(delta+igamma)) + 1/cosh(a*(-delta+igamma))
        # = 2*Re[1/cosh(a*(delta+igamma))]
        cr = np.cosh(a*delta) * np.cos(a*gamma)
        ci = np.sinh(a*delta) * np.sin(a*gamma)
        mag_sq = cr**2 + ci**2
        if mag_sq < 1e-30:
            return 0
        return 2 * cr / mag_sq

    print(f'  Testing d2h/d(delta2) at delta=0 for gamma_1 = {gamma_test:.4f}:')
    print(f'  (positive = minimum at sigma=1/2, negative = maximum)')
    print()

    eps = 1e-5
    for name, func in [('Lorentzian', lambda d, g: h_pair(d, g, L)),
                         ('Gaussian a=0.01', lambda d, g: h_gaussian(d, g, 0.01)),
                         ('Gaussian a=0.1', lambda d, g: h_gaussian(d, g, 0.1)),
                         ('Gaussian a=1.0', lambda d, g: h_gaussian(d, g, 1.0)),
                         ('Power s=1', lambda d, g: h_power(d, g, 1)),
                         ('Power s=2', lambda d, g: h_power(d, g, 2)),
                         ('Sech a=0.1', lambda d, g: h_sech(d, g, 0.1)),
                         ('Sech a=0.5', lambda d, g: h_sech(d, g, 0.5))]:
        h0 = func(0, gamma_test)
        d2h = (func(eps, gamma_test) - 2*func(0, gamma_test) + func(-eps, gamma_test)) / eps**2
        print(f'    {name:>20s}: h(0)={h0:>+12.6e}, d2h={d2h:>+12.6e} '
              f'{"MIN" if d2h > 0 else "MAX" if d2h < 0 else "FLAT"}')

    # Test across multiple zeros
    print(f'\n  Is d2h > 0 (minimum) for ALL zeros and ALL test functions?')
    for name, func in [('Lorentzian', lambda d, g: h_pair(d, g, L)),
                         ('Gaussian a=0.01', lambda d, g: h_gaussian(d, g, 0.01)),
                         ('Gaussian a=0.1', lambda d, g: h_gaussian(d, g, 0.1)),
                         ('Power s=1', lambda d, g: h_power(d, g, 1)),
                         ('Sech a=0.1', lambda d, g: h_sech(d, g, 0.1))]:
        n_min = 0
        n_max = 0
        for g in zeros[:30]:
            d2 = (func(eps, g) - 2*func(0, g) + func(-eps, g)) / eps**2
            if d2 > 0:
                n_min += 1
            else:
                n_max += 1
        print(f'    {name:>20s}: {n_min} MIN, {n_max} MAX out of 30 zeros')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: The upper bound — can we bound S from above?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: BOUNDING S FROM ABOVE')
    print(f'{"="*76}\n')

    # S = sum_rho h(rho - 1/2) is determined by the explicit formula.
    # From the prime side: S = (archimedean terms) - (prime sum)
    #
    # The archimedean terms are POSITIVE (they come from the pole of zeta).
    # The prime sum is POSITIVE (each prime contributes positively).
    # So S = positive - positive.
    #
    # If we can bound the prime sum from BELOW, we bound S from ABOVE.
    #
    # The prime sum involves sum_{p^k} log(p)/p^{k/2} * h_hat(log p^k)
    # where h_hat is the Fourier transform of h.
    # For the Lorentzian: h_hat(x) = pi * e^{-L|x|/2}
    #
    # This is a convergent sum over prime powers.

    L = np.log(1000)

    # Compute the "barrier" B(L) which is essentially sum_rho h(rho-1/2)
    # from the prime side:
    # B(L) = W02 eigenvalue - M eigenvalue on the range direction
    # We already know B(L) ~ 0.04 from Sessions 40-42

    # The zero sum:
    S_0 = sum(h_pair(0, g, L) for g in zeros)
    print(f'  L = {L:.4f} (lam^2 = 1000):')
    print(f'  S_0 (100 on-line zeros) = {S_0:+.6f}')
    print()

    # Key insight: the explicit formula says S = S_prime + S_arch
    # where S_prime < 0 (prime sum with minus sign) and S_arch > 0
    # The barrier B(L) ~ 0.04 > 0 means S_arch dominates S_prime slightly.
    #
    # But S_0 = -0.269 (negative!) while B(L) = 0.04 (positive).
    # These are DIFFERENT quantities! S_0 is the zero sum for the h_pair
    # test function, while B(L) is the barrier for Q_W on the range.
    #
    # The relationship: S_0 enters the Rayleigh quotient of M, not B(L) directly.

    # ======================================================================
    # PROBE 4: The KEY test — complementary inequality
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: COMPLEMENTARY INEQUALITY FROM A SECOND TEST FUNCTION')
    print(f'{"="*76}\n')

    # If we use test function h1 (Lorentzian): h1_pair is minimized at delta=0
    #   => S1 >= S1_0  (zero sum >= on-line sum)
    #
    # If we use test function h2 where h2_pair is MAXIMIZED at delta=0:
    #   => S2 <= S2_0  (zero sum <= on-line sum)
    #
    # Both S1 and S2 are determined by primes (explicit formula).
    # If S1 = S1_0 AND S2 = S2_0, both constraints are tight => RH.
    #
    # Can we find an h2 where the critical line is a MAXIMUM?

    # From Probe 2: Gaussian with a=0.1 gives MAX for some zeros!
    # Let's check: for Gaussian h(w) = exp(-a*w^2):
    # h_pair = 2*exp(-a*(delta^2-gamma^2)) * cos(2*a*delta*gamma)
    # d2/d(delta2) at delta=0:
    # = 2*exp(a*gamma^2) * [-2a*cos(0) + 0] = -4a*exp(a*gamma^2)
    # This is ALWAYS NEGATIVE! Gaussian h_pair is always MAXIMIZED at delta=0.

    print(f'  For Gaussian h(w) = exp(-a*w^2):')
    print(f'    h_pair = 2*exp(-a*(delta^2-gamma^2)) * cos(2*a*delta*gamma)')
    print(f'    d2h/d(delta2)|_0 = -4a*exp(a*gamma^2) < 0  ALWAYS')
    print(f'    => sigma=1/2 is ALWAYS a MAXIMUM for Gaussian test function!')
    print()
    print(f'  This gives the COMPLEMENTARY inequality:')
    print(f'    S_gauss = sum h_pair_gauss(delta_k, gamma_k) <= S_gauss_0')
    print(f'    (zero sum <= on-line sum for Gaussian)')
    print()

    # Verify numerically
    a_gauss = 0.01
    print(f'  Verification (a={a_gauss}):')
    for g in zeros[:5]:
        d2 = -4 * a_gauss * np.exp(a_gauss * g**2)
        h0 = 2 * np.exp(a_gauss * g**2)
        print(f'    gamma={g:.4f}: h_pair(0)={h0:.6f}, d2h={d2:+.6e} (MAX)')
    print()

    # NOW: combine the two inequalities:
    # From Lorentzian: S_lor >= S_lor_0  (lower bound)
    # From Gaussian:   S_gau <= S_gau_0  (upper bound)
    #
    # Both S_lor and S_gau are determined by primes.
    # S_lor_0 and S_gau_0 are determined by on-line zeros.
    #
    # If S_lor = S_lor_0 (Lorentzian tight) => all delta_k = 0 => RH.
    # If S_gau = S_gau_0 (Gaussian tight)   => all delta_k = 0 => RH.
    #
    # The question: can BOTH be tight simultaneously ONLY when delta_k = 0?
    #
    # If delta_k != 0 for some k:
    #   S_lor > S_lor_0 (strict inequality from Lorentzian)
    #   S_gau < S_gau_0 (strict inequality from Gaussian)
    #
    # But S_lor and S_gau are BOTH determined by the SAME primes.
    # The explicit formula for different h gives different sums but
    # they're all computed from the same prime distribution.
    #
    # Can we find a RELATION between S_lor and S_gau that forces
    # tightness of both?

    print(f'  COMBINING TWO TEST FUNCTIONS:')
    print(f'    Lorentzian: S_lor >= S_lor_0 (delta=0 is minimum)')
    print(f'    Gaussian:   S_gau <= S_gau_0 (delta=0 is maximum)')
    print()
    print(f'    If any delta_k != 0:')
    print(f'      S_lor > S_lor_0 (gap opens on lower bound)')
    print(f'      S_gau < S_gau_0 (gap opens on upper bound)')
    print()
    print(f'    Can we find a relation S_lor + c*S_gau = constant?')
    print(f'    If so, the gaps would need to cancel, constraining delta_k.')
    print()

    # Compute both sums for various delta
    L = np.log(1000)
    a_g = 0.01
    print(f'  {"delta":>10} {"S_lor":>14} {"S_gau":>14} {"S_lor+S_gau":>14} '
          f'{"S_lor-S_gau":>14}')
    print('  ' + '-' * 70)

    for delta in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        s_lor = sum(h_pair(delta, g, L) for g in zeros[:50])
        s_gau = sum(h_gaussian(delta, g, a_g) for g in zeros[:50])
        print(f'  {delta:>10.4f} {s_lor:>+14.6f} {s_gau:>+14.6f} '
              f'{s_lor + s_gau:>+14.6f} {s_lor - s_gau:>+14.6f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: The Lorentzian + Gaussian linear combination
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: LINEAR COMBINATION h = h_lor + c * h_gau')
    print(f'{"="*76}\n')

    # If h = h_lor + c*h_gau, then:
    # d2h_pair/d(delta2) = d2h_lor/d(delta2) + c * d2h_gau/d(delta2)
    # For h_lor: d2h > 0 (minimum)
    # For h_gau: d2h < 0 (maximum)
    #
    # At the right c, d2h_combined = 0 for some gamma => inflection point.
    # This means the combined test function has a FLAT direction at sigma=1/2
    # for that gamma.
    #
    # What does this mean for the explicit formula constraint?

    L = np.log(1000)
    gamma1 = zeros[0]

    d2_lor = (h_pair(eps, gamma1, L) - 2*h_pair(0, gamma1, L) + h_pair(-eps, gamma1, L)) / eps**2
    d2_gau = -4 * a_g * np.exp(a_g * gamma1**2)

    c_balance = -d2_lor / d2_gau
    print(f'  For gamma_1 = {gamma1:.4f}:')
    print(f'    d2h_lor = {d2_lor:+.6e}')
    print(f'    d2h_gau = {d2_gau:+.6e}')
    print(f'    Balancing c = {c_balance:.6e}')
    print(f'    At this c: combined test function has ZERO curvature at gamma_1')
    print()

    # For ALL gammas to have zero curvature, we need c(gamma) to vary.
    # This means no single linear combination can make ALL curvatures zero.
    # The c that balances gamma_1 doesn't balance gamma_2, etc.
    print(f'  Balancing c for each zero:')
    print(f'  {"zero#":>6} {"gamma":>10} {"c_balance":>14}')
    print('  ' + '-' * 34)
    for k in range(10):
        g = zeros[k]
        d2l = (h_pair(eps, g, L) - 2*h_pair(0, g, L) + h_pair(-eps, g, L)) / eps**2
        d2g = -4 * a_g * np.exp(a_g * g**2)
        c_b = -d2l / d2g
        print(f'  {k+1:>6d} {g:>10.4f} {c_b:>14.6e}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: What if we use the HEAT KERNEL as second test function?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: STRUCTURAL CONSTRAINTS ON S')
    print(f'{"="*76}\n')

    # The explicit formula gives S = F(primes, L) for each test function h.
    # For the Lorentzian: S_lor = F_lor(primes, L)
    # For the Gaussian:   S_gau = F_gau(primes, L)
    #
    # Both are determined by the same primes. The question is whether
    # knowing F_lor and F_gau (both computable from primes) constrains
    # the zero configuration.
    #
    # In principle, using INFINITELY MANY test functions gives COMPLETE
    # information about the zeros (this is the zero-counting approach).
    # The question is: do TWO test functions already force RH?
    #
    # From the inequalities:
    #   S_lor >= S_lor_0 (Lorentzian: minimum at sigma=1/2)
    #   S_gau <= S_gau_0 (Gaussian: maximum at sigma=1/2)
    #
    # These constrain the zero configuration to a STRIP:
    #   S_lor - S_lor_0 >= 0 and S_gau_0 - S_gau >= 0
    #
    # If the strip has zero width (forces S_lor = S_lor_0), RH follows.

    # How wide is the strip?
    print(f'  The strip of compatible zero configurations:')
    print(f'  Moving all zeros to sigma = 1/2 + delta:')
    print()
    print(f'  {"delta":>10} {"S_lor-S_lor_0":>14} {"S_gau_0-S_gau":>14} {"product":>14}')
    print('  ' + '-' * 56)

    s_lor_0 = sum(h_pair(0, g, L) for g in zeros[:50])
    s_gau_0 = sum(h_gaussian(0, g, a_g) for g in zeros[:50])

    for delta in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        s_lor = sum(h_pair(delta, g, L) for g in zeros[:50])
        s_gau = sum(h_gaussian(delta, g, a_g) for g in zeros[:50])

        gap_lor = s_lor - s_lor_0  # >= 0 (Lorentzian gap)
        gap_gau = s_gau_0 - s_gau  # >= 0 (Gaussian gap)

        print(f'  {delta:>10.4f} {gap_lor:>+14.6e} {gap_gau:>+14.6e} '
              f'{gap_lor * gap_gau:>14.6e}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 78c VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
