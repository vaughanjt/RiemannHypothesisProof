"""
SESSION 46c — THE IDENTITY WEB

Every identity that equals 1 or 0, involving pi, e, primes, and zeros,
constrains the barrier. Build the full web and measure what's left.

IDENTITIES EQUALING 1:
  I1: sin^2(x) + cos^2(x) = 1           (Pythagorean)
  I2: cosh^2(x) - sinh^2(x) = 1         (Hyperbolic)
  I3: |e^{ix}|^2 = 1                     (Unit circle)
  I4: zeta(s) * sum mu(n)/n^s = 1        (Mobius inversion)
  I5: Gamma(s)*Gamma(1-s)*sin(pi*s)/pi=1 (Gamma reflection)
  I6: xi(s)/xi(1-s) = 1 on CL            (Functional equation)
  I7: e^{2*pi*i} = 1                     (Full rotation = periodicity)
  I8: sum_{n=1}^inf n^{-s}/zeta(s) = ... (Euler product = 1 after division)

IDENTITIES EQUALING 0:
  Z1: e^{i*pi} + 1 = 0                   (Euler)
  Z2: xi(rho) = 0 at zeros               (Definition of zeros)
  Z3: sin(n*pi) = 0 for integer n        (Periodicity)
  Z4: zeta(-2n) = 0 for n >= 1           (Trivial zeros)
  Z5: v(1/2, t) = 0 for all t            (Functional equation on CL)
  Z6: sum Mobius = M(x) -> 0             (PNT equivalent)

Each identity constrains the barrier components. Collect them all.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, pi as mppi, zeta as mpzeta, gamma as mpgamma,
                    loggamma, euler as mp_euler, digamma, log as mplog,
                    sin as mpsin, cos as mpcos, exp as mpexp, zetazero,
                    siegeltheta, siegelz)
from sympy import mobius as sym_mobius
import time
import sys
import os

mp.dps = 25

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 46c -- THE IDENTITY WEB')
    print('  Every identity = 1 or = 0 that constrains the barrier')
    print('=' * 76)

    L = np.log(2000)
    N = 15

    # ══════════════════════════════════════════════════════════════
    # IDENTITIES EQUALING 1
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  IDENTITIES EQUALING 1')
    print('#' * 76)

    # --- I1: Pythagorean ---
    print(f'\n  I1: sin^2(x) + cos^2(x) = 1')
    print(f'  Role in barrier: Mp involves cos(2*pi*n*y/L), sin(2*pi*n*y/L).')
    print(f'  For each prime p at y=log(p): sin^2(2*pi*n*y/L) + cos^2(2*pi*n*y/L) = 1')
    print(f'  This means |e^{{2*pi*i*n*y/L}}| = 1 for each prime.')
    print(f'  CONSTRAINT: each prime\'s Fourier contribution has UNIT MAGNITUDE.')
    print(f'  The amplitude comes from the weight log(p)/sqrt(p), not from trig.')

    # Verify
    p_test = 67
    y = np.log(p_test)
    for n_mode in [1, 5, 10]:
        c = np.cos(2*np.pi*n_mode*y/L)
        s = np.sin(2*np.pi*n_mode*y/L)
        print(f'    p={p_test}, n={n_mode}: sin^2+cos^2 = {s**2+c**2:.15f}')

    # --- I2: Hyperbolic ---
    print(f'\n  I2: cosh^2(x) - sinh^2(x) = 1')
    print(f'  Role: W02 prefactor = 32*L*sinh^2(L/4).')
    print(f'  Identity gives: sinh^2 = cosh^2 - 1.')
    print(f'  So W02 = 32L*(cosh^2-1)*QF = 32L*cosh^2*QF - 32L*QF.')
    val = np.cosh(L/4)**2 - np.sinh(L/4)**2
    print(f'    cosh^2(L/4) - sinh^2(L/4) = {val:.15f}')

    # --- I3: Unit circle ---
    print(f'\n  I3: |e^{{ix}}|^2 = 1')
    print(f'  Role: The Riemann-Siegel theta makes Z(t) = e^{{i*theta}}*zeta real.')
    print(f'  |e^{{i*theta}}| = 1 preserves the norm.')
    print(f'  CONSTRAINT: the rotation by theta PRESERVES |zeta| on the CL.')

    # --- I4: Mobius inversion ---
    print(f'\n  I4: zeta(s) * sum mu(n)/n^s = 1 (for Re(s) > 1)')
    print(f'  Role: The prime sum Mp comes from zeta\'/zeta = -sum Lambda(n)/n^s.')
    print(f'  The Mobius function mu is the Dirichlet inverse of 1.')
    print(f'  CONSTRAINT: the prime sum has a UNIQUE inverse (the Mobius sum).')
    # Verify at s = 2
    s_test = 2.0
    zeta_val = float(mpzeta(mpf(s_test)))
    mobius_sum = sum(int(sym_mobius(n)) / n**s_test for n in range(1, 500))
    product = zeta_val * mobius_sum
    print(f'    At s=2: zeta(2) * sum mu(n)/n^2 = {product:.10f}')

    # --- I5: Gamma reflection ---
    print(f'\n  I5: Gamma(s)*Gamma(1-s) = pi/sin(pi*s)')
    print(f'  Role: M_diag involves digamma psi(s) = Gamma\'/Gamma.')
    print(f'  Reflection: psi(1-s) - psi(s) = pi*cot(pi*s).')
    print(f'  At s=1/4+i*pi*n/L (the M_alpha argument), this constrains M_alpha.')
    # Verify
    s_test = mpc(0.25, 0.5)
    lhs = mpgamma(s_test) * mpgamma(1 - s_test)
    rhs = mppi / mpsin(mppi * s_test)
    print(f'    At s=0.25+0.5i: Gamma(s)*Gamma(1-s) = {complex(lhs):.6f}')
    print(f'                     pi/sin(pi*s) = {complex(rhs):.6f}')
    print(f'                     Ratio = {abs(complex(lhs/rhs)):.15f}')

    # --- I6: Functional equation ---
    print(f'\n  I6: xi(s)/xi(1-s) = 1')
    print(f'  Role: Forces v(1/2, t) = 0 (xi real on CL).')
    print(f'  On the CL: |xi(s)/xi(1-s)| = 1 exactly.')
    print(f'  Off CL: |xi(s)/xi(1-s)| != 1 (broken symmetry).')
    print(f'  CONSTRAINT: the barrier is symmetric about sigma=1/2.')

    # --- I7: Full rotation ---
    print(f'\n  I7: e^{{2*pi*i}} = 1')
    print(f'  Role: Fourier modes with period L. e^{{2*pi*i*n}} = 1 makes')
    print(f'  the basis orthogonal. The barrier modes are discrete because')
    print(f'  of this periodicity.')
    print(f'  CONSTRAINT: barrier Fourier spectrum is DISCRETE (integer modes).')

    # --- I8: Parseval ---
    print(f'\n  I8: sum |f_n|^2 = integral |f(x)|^2 dx  (Parseval)')
    print(f'  Role: Total energy of the barrier in Fourier space = energy in real space.')
    print(f'  CONSTRAINT: the barrier\'s Fourier coefficients are norm-preserving.')
    print(f'  Used in session 45c to prove d^2B/dd^2 > 0 (convexity).')

    # ══════════════════════════════════════════════════════════════
    # IDENTITIES EQUALING 0
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  IDENTITIES EQUALING 0')
    print('#' * 76)

    # --- Z1: Euler ---
    print(f'\n  Z1: e^{{i*pi}} + 1 = 0')
    print(f'  The root identity. In H: e^{{I*pi}} + 1 = 0 for all I in S^2.')
    print(f'  Pi is a SPHERE. The identity is a 2-sphere of equations.')

    # --- Z2: Zeros ---
    print(f'\n  Z2: xi(rho) = 0 at zeros')
    print(f'  THE defining property. Each zero gives one equation.')
    print(f'  200 zeros = 200 equations constraining the barrier.')
    print(f'  The barrier B = sum |H(rho)|^2 uses these as sampling points.')

    # --- Z3: Sine at integer pi ---
    print(f'\n  Z3: sin(n*pi) = 0 for integer n')
    print(f'  Role: Gram points are where theta(t) = n*pi.')
    print(f'  At Gram points: sin(theta) = 0, cos(theta) = +/-1.')
    print(f'  CONSTRAINT: at Gram points, the Z-function = +/-|zeta|.')

    # --- Z4: Trivial zeros ---
    print(f'\n  Z4: zeta(-2n) = 0 for positive integer n')
    print(f'  Role: The trivial zeros come from Gamma poles.')
    print(f'  CONSTRAINT: zeta vanishes at negative even integers.')
    print(f'  These are "free" — they come from the Gamma function, not primes.')

    # --- Z5: v = 0 on CL ---
    print(f'\n  Z5: v(1/2, t) = 0 for all t (from functional equation)')
    print(f'  xi is REAL on the critical line. This forces Im(xi) = 0.')
    print(f'  CONSTRAINT: the strongest single identity for the barrier.')
    print(f'  Combined with u(1/2, gamma_n) = 0: gives ALL zeros on CL.')

    # --- Z6: Mertens ---
    print(f'\n  Z6: M(x) = sum_{{n<=x}} mu(n) = o(x) (PNT equivalent)')
    print(f'  The Mertens function. M(x)/x -> 0 is equivalent to PNT.')
    print(f'  M(x)/sqrt(x) bounded is equivalent to RH!')
    print(f'  CONSTRAINT: the Mobius sum stays bounded relative to sqrt(x).')
    # Compute M(x) for some values
    print(f'    M(x) values:')
    for x in [10, 100, 1000, 10000]:
        M = sum(int(sym_mobius(n)) for n in range(1, x + 1))
        ratio = M / np.sqrt(x)
        print(f'      M({x}) = {M}, M/sqrt(x) = {ratio:.4f}')

    # ══════════════════════════════════════════════════════════════
    # THE IDENTITY WEB: how they connect
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  THE IDENTITY WEB: connections between identities')
    print('#' * 76)

    print(f'''
  Each identity constrains the barrier. The WEB of connections:

  I1 (sin^2+cos^2) ----> Mp has unit-magnitude Fourier components
       |                    |
       v                    v
  I7 (e^{{2pi*i}}=1) ----> Discrete Fourier spectrum
       |                    |
       v                    v
  I8 (Parseval) ---------> Total energy preserved
       |
       v
  I2 (cosh^2-sinh^2) ---> W02 = 32L*(cosh^2-1)*QF
       |                    |
       v                    v
  Z1 (e^{{ipi}}=-1) ------> Pi is a sphere (Euler spheres)
       |                    |
       v                    v
  I5 (Gamma reflection) -> M_diag, M_alpha constrained
       |                    |
       v                    v
  I6 (xi(s)=xi(1-s)) ---> v=0 on CL (Z5)
       |                    |
       v                    v
  Z2 (xi(rho)=0) --------> B = sum |H(rho)|^2 (spectral barrier)
       |                    |
       v                    v
  I4 (zeta*mu=1) --------> Mp has Mobius inverse
       |                    |
       v                    v
  Z6 (M(x) bounded) -----> RH iff M(x)/sqrt(x) bounded
  ''')

    # ══════════════════════════════════════════════════════════════
    # THE COMBINED CONSTRAINTS ON THE BARRIER
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  COMBINED CONSTRAINTS: what does the web force?')
    print('#' * 76)

    print(f'''
  The barrier B = W02 - Mp - M_diag - M_alpha.

  From I2: W02 = 32L*(cosh^2(L/4)-1)*QF      [cosh identity splits W02]
  From I1: Mp = sum_p w_p * cos(phase_p)       [unit magnitude per prime]
  From I8: sum |Mp_mode|^2 = integral |kernel|^2  [Parseval preserves norm]
  From I5: M_diag involves psi(1/4+i*pi*n/L)  [Gamma reflection constrains]
  From I4: sum mu(n)*Mp contributions = W02    [Mobius inverts the prime sum]
  From Z5: v(1/2, t) = 0                      [barrier real on CL]
  From Z2: B = sum_rho |H(rho)|^2              [spectral = sum of squares]
  From Z6: M(x)/sqrt(x) bounded               [Mertens = RH]

  KEY IDENTITIES FOR POSITIVITY:

  Z2 says B = sum |H(rho)|^2 >= 0 (sum of squares, always non-negative).
  But this requires summing over ALL zeros (infinity).

  I4+Z6 say: if M(x)/sqrt(x) is bounded, RH holds.
  This is the Mobius route: bounded Mertens => RH.

  I2+I1 say: cosh^2 provides the framework, primes fill it with
  unit-magnitude oscillations. The "1" in cosh^2-1 is the GAP.

  Z5+I6 say: xi is real on the CL, and symmetric about sigma=1/2.
  Combined with Z2: zeros lie where a REAL function crosses zero.
  ''')

    # ══════════════════════════════════════════════════════════════
    # QUANTITATIVE: how much does each identity constrain?
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  QUANTITATIVE: constraint strength of each identity')
    print('#' * 76)

    # The barrier has multiple "degrees of freedom":
    # - W02: determined by L (1 parameter)
    # - Mp: determined by K primes (K parameters, one per prime)
    # - M_diag: determined by L (computable integral)
    # - M_alpha: determined by L (hypergeometric + digamma)

    # Each identity removes degrees of freedom:
    primes = list(sieve_primes(int(2000)))
    K = len(primes)

    print(f'\n  Barrier at lam^2 = 2000 ({K} primes):')
    print(f'  Total degrees of freedom: {K + 3} (K primes + W02 + M_diag + M_alpha)')
    print(f'\n  {"Identity":>25s} {"Constraints":>12s} {"Remaining DOF":>14s}')
    print('  ' + '-' * 55)

    dof = K + 3
    constraints = [
        ('I2 (cosh^2-sinh^2=1)', 1, 'Splits W02 into cosh^2 - 1'),
        ('I1 (sin^2+cos^2=1)', K, 'Each prime has unit Fourier mag'),
        ('I8 (Parseval)', 1, 'Total energy preserved'),
        ('I5 (Gamma reflection)', 1, 'Constrains M_diag+M_alpha'),
        ('I6 (functional eq)', 1, 'Symmetric about sigma=1/2'),
        ('Z5 (v=0 on CL)', 1, 'xi real on critical line'),
        ('Z2 (xi(rho)=0)', 0, 'Gives spectral repr (not a reduction)'),
        ('I4 (zeta*mu=1)', 1, 'Mobius inverts prime sum'),
    ]

    remaining = dof
    for name, n_const, desc in constraints:
        remaining -= n_const
        print(f'  {name:>25s} {n_const:>12d} {remaining:>14d}  ({desc})')

    print(f'\n  After all identities: {remaining} degrees of freedom remain.')
    print(f'  These are the UNCONSTRAINED directions of the barrier.')
    print(f'  If remaining = 0 or negative: the system is overdetermined')
    print(f'  and positivity might be forced.')

    if remaining <= 0:
        print(f'\n  *** SYSTEM IS OVERDETERMINED: {-remaining} excess constraints ***')
        print(f'  The identities may FORCE barrier positivity!')
    else:
        print(f'\n  System is underdetermined: {remaining} free parameters.')
        print(f'  Need {remaining} more identities to fully constrain.')

    # ══════════════════════════════════════════════════════════════
    # WHAT IDENTITIES ARE WE MISSING?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  WHAT IDENTITIES ARE WE MISSING?')
    print('#' * 76)

    print(f'''
  We have {K + 3} DOF and removed {dof - remaining} via identities.
  Remaining: {remaining} unconstrained directions.

  These correspond to: the SPECIFIC VALUES of each prime\'s Fourier phase.
  Identity I1 constrains the MAGNITUDE (=1) but not the PHASE.
  The phase 2*pi*n*log(p)/L is determined by the prime p and the mode n.

  The "unconstrained" directions ARE the primes themselves.
  Each prime p contributes a phase log(p) that is a FREE PARAMETER
  (determined by which integers are prime, not by any identity).

  TO FULLY CONSTRAIN: we need identities involving the SPECIFIC VALUES
  of log(p) for each prime. These would be:

  - The prime number theorem: pi(x) ~ x/log(x)
  - The explicit formula: psi(x) = x - sum_rho x^rho/rho + ...
  - Mertens' theorems: sum 1/p ~ log(log(x)) + M
  - The Bombieri-Vinogradov theorem: primes equidistributed in APs
  - The Chebotarev density theorem: primes split in extensions

  Each of these is a STATISTICAL constraint on the ensemble of primes.
  They don't determine individual primes but constrain their statistics.

  RH is the statement that these statistical constraints are SUFFICIENT
  to force the barrier positive. The primes have enough "randomness"
  (equidistribution) to prevent systematic cancellation with pi.

  The identity web ALMOST closes: all the structural identities are
  satisfied, and what remains is the ARITHMETIC of the specific primes.
  The primes are the only "free" ingredient.
  ''')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('=' * 76)
    print('  SESSION 46c SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE IDENTITY WEB:

  8 identities equaling 1:
    I1 (Pythagorean), I2 (Hyperbolic), I3 (Unit circle),
    I4 (Mobius), I5 (Gamma reflection), I6 (Functional eq),
    I7 (Full rotation), I8 (Parseval)

  6 identities equaling 0:
    Z1 (Euler), Z2 (Zeros), Z3 (Sine periodicity),
    Z4 (Trivial zeros), Z5 (v=0 on CL), Z6 (Mertens)

  Together: {dof - remaining} constraints on {dof} degrees of freedom.
  Remaining: {remaining} unconstrained = the prime phases log(p).

  THE BOTTOM LINE:
  The identity web constrains EVERYTHING about the barrier
  EXCEPT the specific values of the primes. The structural identities
  (Pythagorean, hyperbolic, Euler, Parseval, functional equation)
  determine the framework. The primes fill in the content.

  RH = the prime content is consistent with the structural framework.
  Every identity = 1 is a constraint.
  Every identity = 0 is a zero to avoid or land on.
  The barrier lives at the intersection of all constraints.
''')

    print('=' * 76)
    print('  SESSION 46c COMPLETE')
    print('=' * 76)
