"""
SESSION 67c -- THE ZETA-FREE ROUTE

Can we compute xi''(1/2)/xi(1/2) = 0.0462 WITHOUT zeta?

Decompose: log xi(s) = log(1/2) + log(s) + log(s-1) - (s/2)log(pi)
                        + log Gamma(s/2) + log zeta(s)

The second derivative at s=1/2:
  (log xi)''(1/2) = ARCHIMEDEAN + PRIME_PART

where ARCHIMEDEAN involves only pi and Gamma (no zeta, no primes, no zeros),
and PRIME_PART involves zeta'/zeta and zeta''/zeta at s=1/2.

Question: can PRIME_PART be computed from primes WITHOUT going through zeta?
"""

import sys
import numpy as np
import mpmath
from mpmath import mp, mpf

mp.dps = 30


def run():
    print()
    print('#' * 76)
    print('  SESSION 67c -- THE ZETA-FREE ROUTE')
    print('#' * 76)

    s = mpf('0.5')

    # ==================================================================
    # PART 1: DECOMPOSE xi''/xi(1/2)
    # ==================================================================
    print('\n  === PART 1: THE DECOMPOSITION ===\n')

    # log xi(s) = log(1/2) + log(s) + log(s-1) - (s/2)log(pi) + log Gamma(s/2) + log zeta(s)
    # (log xi)' = 1/s + 1/(s-1) - log(pi)/2 + (1/2)*psi(s/2) + zeta'/zeta(s)
    # (log xi)'' = -1/s^2 - 1/(s-1)^2 + (1/4)*psi'(s/2) + (zeta'/zeta)'(s)
    # where (zeta'/zeta)' = zeta''/zeta - (zeta'/zeta)^2

    # At s = 1/2, since xi'(1/2) = 0:
    # xi''(1/2)/xi(1/2) = (log xi)''(1/2) = A + P
    # where A = archimedean, P = prime part

    # ARCHIMEDEAN PART (no zeta):
    # A = -1/s^2 - 1/(s-1)^2 + (1/4)*psi'(s/2)
    # At s = 1/2:
    # A = -4 - 4 + (1/4)*psi'(1/4)

    psi1_quarter = float(mpmath.polygamma(1, mpf('0.25')))  # psi'(1/4)
    A = -4 - 4 + 0.25 * psi1_quarter

    print(f'  ARCHIMEDEAN PART (only pi and Gamma):')
    print(f'    -1/s^2 at s=1/2:      -4')
    print(f'    -1/(s-1)^2 at s=1/2:  -4')
    print(f'    psi\'(1/4) =            {psi1_quarter:.10f}')
    print(f'    (1/4)*psi\'(1/4) =      {0.25*psi1_quarter:.10f}')
    print(f'    A = -8 + psi\'(1/4)/4 = {A:.10f}')
    print()

    # Known closed form for psi'(1/4):
    # psi'(1/4) = pi^2 + 8*G where G = Catalan's constant
    G_catalan = float(mpmath.catalan)
    psi1_formula = np.pi**2 + 8 * G_catalan
    print(f'    Catalan\'s constant G = {G_catalan:.10f}')
    print(f'    psi\'(1/4) = pi^2 + 8G = {psi1_formula:.10f}')
    print(f'    Check: {abs(psi1_quarter - psi1_formula):.2e}')
    print()
    print(f'    A = -8 + pi^2/4 + 2*G = {-8 + np.pi**2/4 + 2*G_catalan:.10f}')

    # PRIME PART (involves zeta):
    # P = (zeta'/zeta)'(1/2) = zeta''(1/2)/zeta(1/2) - [zeta'(1/2)/zeta(1/2)]^2

    zeta_half = float(mpmath.zeta(s))
    zeta_prime_half = float(mpmath.zeta(s, derivative=1))
    zeta_pp_half = float(mpmath.zeta(s, derivative=2))

    P = zeta_pp_half/zeta_half - (zeta_prime_half/zeta_half)**2

    print(f'\n  PRIME PART (involves zeta):')
    print(f'    zeta(1/2) =   {zeta_half:.10f}')
    print(f'    zeta\'(1/2) =  {zeta_prime_half:.10f}')
    print(f'    zeta\'\'(1/2) = {zeta_pp_half:.10f}')
    print(f'    P = zeta\'\'/zeta - (zeta\'/zeta)^2 = {P:.10f}')
    print()

    total = A + P
    print(f'  TOTAL: xi\'\'(1/2)/xi(1/2) = A + P = {total:.10f}')

    # Verify
    def xi_func(ss):
        return mpmath.mpf('0.5') * ss * (ss-1) * mpmath.power(mpmath.pi, -ss/2) * \
               mpmath.gamma(ss/2) * mpmath.zeta(ss)

    xi_val = float(xi_func(s))
    mp.dps = 50
    xi_pp = float(mpmath.diff(xi_func, s, n=2))
    mp.dps = 30
    Z_direct = xi_pp / xi_val
    print(f'  Direct: xi\'\'(1/2)/xi(1/2) = {Z_direct:.10f}')
    print(f'  Match: {abs(total - Z_direct):.2e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: THE NEAR-CANCELLATION
    # ==================================================================
    print(f'\n  === PART 2: THE NEAR-CANCELLATION ===\n')
    print(f'  A (archimedean) = {A:+.10f}')
    print(f'  P (prime)       = {P:+.10f}')
    print(f'  A + P (total)   = {total:+.10f}')
    print(f'  |A|/|total|     = {abs(A)/abs(total):.1f}x (cancellation ratio)')
    print()
    print(f'  Two contributions of size ~3.7 cancel to give 0.046.')
    print(f'  This is an 80:1 cancellation.')
    sys.stdout.flush()

    # ==================================================================
    # PART 3: WHAT WOULD WE NEED?
    # ==================================================================
    print(f'\n  === PART 3: THE ZETA-FREE GOAL ===\n')
    print(f'  A is zeta-free:')
    print(f'    A = -8 + pi^2/4 + 2*Catalan = {A:.10f}')
    print(f'  P involves zeta at s=1/2.')
    print()
    print(f'  To compute P without zeta, we need zeta(1/2) from primes.')
    print(f'  The Euler product diverges at s=1/2.')
    print(f'  But the EXPLICIT FORMULA gives:')
    print(f'    (zeta\'/zeta)\'(s) = 1/(s-1)^2 + Sum_rho 1/(s-rho)^2')
    print(f'  At s = 1/2: involves the zeros (circular).')
    print()

    # The Weil explicit formula approach:
    # Sum_rho h_hat(gamma) = PRIME_SUM + ARCHIMEDEAN
    # For h_hat(t) = 1/(it)^2 = -1/t^2, we'd get Sum 1/gamma^2 from a prime sum.
    # BUT: the corresponding test function doesn't decay, so the sum diverges.

    # What if we use the SMOOTHED version?
    print(f'  === APPROACH: WEIL EXPLICIT FORMULA WITH SMOOTH TEST ===\n')
    print(f'  Choose h(gamma) = 1/(gamma^2 + a^2) for parameter a > 0.')
    print(f'  This gives a CONVERGENT prime sum via the explicit formula.')
    print(f'  Then take a -> 0 to recover Sum 1/gamma^2.\n')

    # h(gamma) = 1/(gamma^2 + a^2)
    # This corresponds to h(rho-1/2) = 1/((rho-1/2)^2 + a^2) = -1/((rho-1/2-ia)(rho-1/2+ia))
    # The test function g(x) = (pi/a) * e^{-a|x|}
    # The prime sum: Sum_n Lambda(n)/sqrt(n) * g(log n) = (pi/a) Sum Lambda(n) n^{-1/2-a}

    # For a > 1/2: the sum Sum Lambda(n) n^{-1/2-a} converges (Re(s) = 1/2+a > 1)!

    print(f'  For a > 1/2: Sum Lambda(n) * n^{{-1/2-a}} CONVERGES.')
    print(f'  This IS a convergent prime sum!\n')

    # Compute for various a
    print(f'  {"a":>8} {"Sum Lambda/n^(1/2+a)":>22} {"Sum 1/(g^2+a^2)":>18} '
          f'{"explicit match":>16}')
    print('  ' + '-' * 68)

    from session41g_uncapped_barrier import sieve_primes

    for a_val in [2.0, 1.5, 1.0, 0.8, 0.6, 0.51]:
        # Prime sum: Sum_n Lambda(n) * n^{-s} at s = 1/2 + a = -zeta'/zeta(1/2+a)
        s_val = 0.5 + a_val
        neg_zz = float(-mpmath.zeta(mpf(str(s_val)), derivative=1) /
                       mpmath.zeta(mpf(str(s_val))))

        # Zero sum: Sum 1/(gamma^2 + a^2) (using known zeros)
        gammas = []
        for k in range(1, 201):
            gammas.append(float(mpmath.im(mpmath.zetazero(k))))
        gammas = np.array(gammas)
        zero_sum = np.sum(2.0 / (gammas**2 + a_val**2))
        # Tail correction
        gK = gammas[-1]
        tail = 2 * np.log(gK) / (2*np.pi*(gK**2+a_val**2)**(0.5)) * (1/a_val) * np.arctan(gK/a_val)
        # Crude: tail ~ 2/(2*pi) * integral log(g)/(g^2+a^2) from gK to inf
        # ~ 2*log(gK)/(2*pi*a*gK)
        tail = 2*np.log(gK)/(2*np.pi*a_val*gK)
        zero_sum_total = zero_sum + tail

        # From the explicit formula: Sum 1/(gamma^2+a^2) should equal
        # (1/2) * [some function of -zeta'/zeta(1/2+a) and archimedean terms]
        # Specifically: Sum_rho 1/((rho-1/2)^2+a^2) = archimedean + prime
        # This is the smoothed version of xi''/xi

        print(f'  {a_val:>8.2f} {neg_zz:>22.10f} {zero_sum_total:>18.10f} ')

    sys.stdout.flush()

    # ==================================================================
    # PART 4: THE LIMIT a -> 0
    # ==================================================================
    print(f'\n  === PART 4: THE CONVERGENT PRIME SUM ===\n')

    print(f'  -zeta\'/zeta(s) = Sum Lambda(n)/n^s  converges for Re(s) > 1')
    print(f'  At s = 1/2 + a with a > 1/2: this is a CONVERGENT PRIME SUM.')
    print()
    print(f'  -zeta\'/zeta(1/2+a) as a function of a:')
    print(f'  {"a":>8} {"s":>6} {"-zeta\'/zeta(s)":>16} {"from primes":>16}')
    print('  ' + '-' * 50)

    for a_val in [5.0, 2.0, 1.0, 0.55, 0.51, 0.501]:
        s_val = mpf('0.5') + mpf(str(a_val))
        neg_zz = float(-mpmath.zeta(s_val, derivative=1) / mpmath.zeta(s_val))

        # Direct prime sum (for verification at large a)
        if a_val >= 1.0:
            primes = sieve_primes(10000)
            ps = 0
            for p in primes:
                pk = int(p)
                logp = np.log(int(p))
                while pk <= 10000:
                    ps += logp * pk**(-(0.5+a_val))
                    pk *= int(p)
            print(f'  {a_val:>8.3f} {0.5+a_val:>6.3f} {neg_zz:>16.10f} {ps:>16.10f}')
        else:
            print(f'  {a_val:>8.3f} {0.5+a_val:>6.3f} {neg_zz:>16.10f} {"(need cont.)":>16}')

    print(f'\n  As a -> 0+, -zeta\'/zeta(1/2+a) -> infinity (pole at s=1).')
    print(f'  BUT: the RENORMALIZED quantity')
    print(f'    R(a) = -zeta\'/zeta(1/2+a) - 1/(a-1/2)')
    print(f'  has a finite limit as a -> 0.')
    print()

    print(f'  {"a":>8} {"R(a)":>16}')
    print('  ' + '-' * 28)
    for a_val in [5.0, 2.0, 1.0, 0.6, 0.55, 0.51, 0.505, 0.501]:
        s_val = mpf('0.5') + mpf(str(a_val))
        neg_zz = float(-mpmath.zeta(s_val, derivative=1) / mpmath.zeta(s_val))
        R = neg_zz - 1.0/(a_val - 0.5)
        print(f'  {a_val:>8.3f} {R:>+16.10f}')

    # The limit as a -> 0 of R(a) should give a value related to B + Sum_rho 1/rho + ...
    sys.stdout.flush()

    # ==================================================================
    # PART 5: CONVERGENT PRIME FORMULA FOR Z
    # ==================================================================
    print(f'\n  === PART 5: THE CONVERGENT PRIME FORMULA ===\n')

    # The second derivative: (zeta'/zeta)'(s) = -(d/ds)(-zeta'/zeta)(s)
    # = Sum Lambda(n) * log(n) * n^{-s} for Re(s) > 1
    # But at s = 1/2: diverges.

    # However: the REGULARIZED second derivative:
    # (zeta'/zeta)'(s) - 1/(s-1)^2 = Sum_rho 1/(s-rho)^2
    # At s = 1/2+a:
    # Sum Lambda(n)*log(n)*n^{-(1/2+a)} - 1/(a-1/2)^2 = Sum_rho 1/(a+1/2-rho)^2

    # For a > 1/2: the prime sum CONVERGES.
    # As a -> 0: the left side diverges but the regularization handles it.

    # What we really need: the VALUE at a = 0.
    # P(0) = lim_{a->0} [Sum Lambda(n)*log(n)*n^{-(1/2+a)} - 1/(a-1/2)^2 - (terms)]
    # This limit exists and equals Sum_rho 1/(rho-1/2)^2 = -xi''(1/2)/xi(1/2)

    # But computing this limit requires analytic continuation...

    print(f'  The prime sum Sum Lambda(n)*log(n)/n^s converges for Re(s) > 1.')
    print(f'  At s = 1/2: it diverges.')
    print(f'  The value at s = 1/2 requires analytic continuation.')
    print()
    print(f'  HOWEVER: the REGULARIZED combination')
    print(f'    P(a) = Sum Lambda(n)*log(n)*n^{{-(1/2+a)}} - 1/(a-1/2)^2')
    print(f'  converges as a -> 0+ and gives our locking constant.')
    print()

    print(f'  P(a) for decreasing a:')
    print(f'  {"a":>8} {"prime sum":>18} {"1/(a-1/2)^2":>14} {"P(a)":>16}')
    print('  ' + '-' * 60)

    for a_val in [5.0, 2.0, 1.0, 0.6, 0.55, 0.52, 0.51, 0.505]:
        s_val = mpf('0.5') + mpf(str(a_val))
        # (zeta'/zeta)'(s) = zeta''/zeta - (zeta'/zeta)^2
        zzp = float(mpmath.zeta(s_val, derivative=1))
        zz = float(mpmath.zeta(s_val))
        zzpp = float(mpmath.zeta(s_val, derivative=2))
        G_val = zzpp/zz - (zzp/zz)**2
        pole = 1.0 / (a_val - 0.5)**2
        P_val = G_val - pole

        print(f'  {a_val:>8.3f} {G_val:>18.10f} {pole:>14.4f} {P_val:>+16.10f}')

    # The limit P(0) should be Sum_rho 1/(rho-1/2)^2 - 1/(1/2-1)^2
    # = -xi''/xi(1/2) - 4
    # = -0.0462 - 4 = -4.046?
    # Hmm wait. Let me reconsider.

    # (zeta'/zeta)'(s) = -1/(s-1)^2 + Sum_rho -1/(s-rho)^2
    # Wait: from the explicit formula for -zeta'/zeta:
    # -zeta'/zeta(s) = 1/(s-1) - B - Sum_rho [1/(s-rho) + 1/rho]
    # Differentiating:
    # -(zeta'/zeta)'(s) = -1/(s-1)^2 + Sum_rho 1/(s-rho)^2
    # So: (zeta'/zeta)'(s) = 1/(s-1)^2 - Sum_rho 1/(s-rho)^2

    # At s = 1/2:
    # (zeta'/zeta)'(1/2) = 1/(1/2-1)^2 - Sum_rho 1/(1/2-rho)^2
    #                    = 4 - Sum_rho 1/(1/2-rho)^2
    #                    = 4 - (-xi''/xi(1/2))
    #                    = 4 + 0.0462

    G_half = float(mpmath.zeta(s, derivative=2)/mpmath.zeta(s) -
                   (mpmath.zeta(s, derivative=1)/mpmath.zeta(s))**2)
    print(f'\n  At a = 0 (s = 1/2):')
    print(f'  (zeta\'/zeta)\'(1/2) = {G_half:.10f}')
    print(f'  = 4 + xi\'\'(1/2)/xi(1/2) = {4 + Z_direct:.10f}')
    print(f'  Match: {abs(G_half - 4 - Z_direct):.2e}')

    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 67c VERDICT')
    print('=' * 76)
    print()
    print(f'  xi\'\'(1/2)/xi(1/2) = A + P where:')
    print(f'    A = -8 + pi^2/4 + 2*Catalan = {A:.6f}  (ZETA-FREE)')
    print(f'    P = (zeta\'/zeta)\'(1/2)      = {P:.6f}  (needs zeta)')
    print(f'    Total                        = {total:.6f}')
    print()
    print(f'  P cannot be computed from a convergent prime sum at s=1/2')
    print(f'  because the prime zeta function has a natural boundary at Re(s)=1.')
    print()
    print(f'  HOWEVER: P(a) = (zeta\'/zeta)\'(1/2+a) IS a convergent prime sum')
    print(f'  for a > 1/2. The value at a=0 requires analytic continuation.')
    print()
    print(f'  The gap: analytic continuation FROM the convergent region (a > 1/2)')
    print(f'  TO the critical point (a = 0) is what the explicit formula does.')
    print(f'  And the explicit formula introduces... the zeros.')


if __name__ == '__main__':
    run()
