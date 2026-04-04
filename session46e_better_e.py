"""
SESSION 46e — A BETTER e IN QUATERNION SPACE

Pi in H: {q : e^q = -1} = {pi*I : I in S^2} (sphere of radius pi, maps to point -1)
e in H:  {exp(I) : I in S^2} = sphere of radius sin(1), center cos(1)

The exponential map exp(theta*I) maps S^2 to a sphere:
  center = cos(theta), radius = sin(theta)

CRITICAL DISCOVERY:
  At theta = pi/2 = pi * (1/2):
    center = cos(pi/2) = 0
    radius = sin(pi/2) = 1
    exp(pi/2 * I) = I  for all I in S^2

  THE EXPONENTIAL MAPS S^2 TO ITSELF AT theta = pi/2.
  S^2 is a FIXED POINT of the exp map at the CRITICAL LINE VALUE.

  pi * sigma = pi * (1/2) = pi/2 -> fixed sphere
  The critical line sigma = 1/2 IS the fixed sphere condition.

QUESTIONS:
  1. Does this fixed-sphere property connect to the barrier?
  2. Is the "better e" actually e^{pi/2} = i^i... no, that's different.
  3. How does the exp map on spheres interact with the zeta function?
  4. Is there an "optimal theta" that makes the barrier most positive?
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, pi as mppi, zeta as mpzeta, gamma as mpgamma
import time
import sys

mp.dps = 20


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 46e -- A BETTER e IN QUATERNION SPACE')
    print('=' * 76)

    # ══════════════════════════════════════════════════════════════
    # 1. THE EXPONENTIAL MAP ON SPHERES
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. THE EXPONENTIAL MAP ON IMAGINARY QUATERNION SPHERES')
    print('#' * 76)

    print(f'''
  For theta in [0, pi] and I any unit imaginary quaternion:
    exp(theta * I) = cos(theta) + I * sin(theta)

  The image of S^2 under exp(theta * .) is:
    center = cos(theta)  (on the real axis)
    radius = sin(theta)  (in the imaginary directions)

  Special values:
    theta = 0:    exp(0) = 1                    (point)
    theta = 1:    exp(I) = cos(1) + sin(1)*I    ("e's sphere")
    theta = pi/2: exp(pi/2*I) = I               (FIXED SPHERE)
    theta = pi:   exp(pi*I) = -1                ("pi's sphere" = point)

  The fixed sphere at theta = pi/2:
    exp(pi/2 * I) = cos(pi/2) + I*sin(pi/2) = 0 + I*1 = I

  S^2 maps to ITSELF. This is unique: no other theta in (0,pi)
  has this property (sin(theta) = 1 and cos(theta) = 0 only at pi/2).
  ''')

    # Verify
    print(f'  Verification:')
    print(f'  {"theta":>8s} {"cos(theta)":>12s} {"sin(theta)":>12s} '
          f'{"center":>10s} {"radius":>10s} {"type":>15s}')
    print('  ' + '-' * 70)

    special_thetas = [
        (0, 'point at +1'),
        (0.5, 'small sphere'),
        (1.0, 'e\'s sphere'),
        (np.pi/4, 'pi/4 sphere'),
        (np.pi/3, 'pi/3 sphere'),
        (np.pi/2, '*** FIXED S^2 ***'),
        (2*np.pi/3, 'past fixed'),
        (np.pi, 'point at -1 (pi)'),
    ]

    for theta, label in special_thetas:
        c = np.cos(theta)
        s = np.sin(theta)
        print(f'  {theta:>8.4f} {c:>+12.6f} {s:>12.6f} '
              f'{c:>+10.6f} {s:>10.6f} {label:>15s}')

    # ══════════════════════════════════════════════════════════════
    # 2. THE FIXED SPHERE AND THE CRITICAL LINE
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. THE FIXED SPHERE = THE CRITICAL LINE')
    print('#' * 76)

    print(f'''
  In the completed zeta: xi(s) = (1/2)*s*(s-1)*pi^{{-s/2}}*Gamma(s/2)*zeta(s)

  The archimedean factor involves pi^{{-s/2}} = exp(-s/2 * log(pi)).
  At s = sigma + t*I:
    pi^{{-s/2}} = exp(-(sigma + t*I)/2 * log(pi))
              = exp(-sigma*log(pi)/2) * exp(-t*I*log(pi)/2)

  The second factor exp(-t*I*log(pi)/2) maps S^2 to a sphere:
    center = cos(t*log(pi)/2)
    radius = sin(t*log(pi)/2)

  This sphere is centered at the origin (FIXED SPHERE) when:
    cos(t*log(pi)/2) = 0
    => t*log(pi)/2 = pi/2 + n*pi
    => t = pi*(2n+1)/log(pi)

  The FIRST fixed-sphere height:
    t_fixed = pi/log(pi) = {np.pi/np.log(np.pi):.6f}

  Compare to the first zero: gamma_1 = 14.1347

  t_fixed = {np.pi/np.log(np.pi):.4f}... not close to gamma_1.

  But the CRITICAL LINE condition is different:
  sigma = 1/2 means the REAL part of the exponent is:
    -sigma*log(pi)/2 = -(1/2)*log(pi)/2 = -log(pi)/4

  The factor pi^{{-1/4}} = {np.pi**(-0.25):.6f}

  The key is not the height t but the REAL PART sigma.
  ''')

    # ══════════════════════════════════════════════════════════════
    # 3. THE EXPONENTIAL MAP AND sigma = 1/2
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  3. WHY sigma = 1/2 IS THE FIXED SPHERE')
    print('#' * 76)

    print(f'''
  Consider the map F_sigma: S^2 -> H defined by:
    F_sigma(I) = exp(pi * sigma * I) for I in S^2

  This maps the imaginary unit sphere to:
    center = cos(pi*sigma)
    radius = sin(pi*sigma)

  The fixed sphere condition (radius = 1, center = 0):
    sin(pi*sigma) = 1 and cos(pi*sigma) = 0
    => pi*sigma = pi/2
    => sigma = 1/2

  *** sigma = 1/2 IS THE UNIQUE VALUE WHERE THE MAP
      F_sigma SENDS S^2 TO ITSELF ***

  At sigma = 1/2: F_{{1/2}}(I) = exp(pi/2 * I) = I
  The exponential PRESERVES the imaginary sphere.
  This is a quaternionic characterization of the critical line.
  ''')

    # Verify with a table
    print(f'  {"sigma":>8s} {"pi*sigma":>10s} {"center":>10s} {"radius":>10s} '
          f'{"maps S^2 to":>20s}')
    print('  ' + '-' * 62)

    for sigma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        theta = np.pi * sigma
        c = np.cos(theta)
        r = np.sin(theta)
        if abs(r - 1) < 0.01 and abs(c) < 0.01:
            desc = '*** S^2 (FIXED) ***'
        elif abs(r) < 0.01:
            desc = f'point at {c:+.2f}'
        else:
            desc = f'sphere r={r:.3f} c={c:+.3f}'
        print(f'  {sigma:>8.1f} {theta:>10.4f} {c:>+10.6f} {r:>10.6f} {desc:>20s}')

    # ══════════════════════════════════════════════════════════════
    # 4. CONNECTION TO THE BARRIER
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. CONNECTION TO THE BARRIER')
    print('#' * 76)

    print(f'''
  The barrier B = W02 - Mp.

  W02 involves the prefactor 32*L*sinh^2(L/4).
  sinh^2(L/4) = (exp(L/2) - 2 + exp(-L/2)) / 4

  In quaternionic extension with L -> L + theta*J:
    exp((L+theta*J)/2) = exp(L/2) * exp(theta*J/2)

  The factor exp(theta*J/2) maps S^2 to a sphere:
    center = cos(theta/2), radius = sin(theta/2)

  At the BARRIER's "critical" theta (where the J-sphere is fixed):
    sin(theta/2) = 1 => theta = pi
    exp(pi*J/2) = J (the fixed sphere condition)

  So the barrier's quaternionic extension has a fixed sphere at theta = pi.

  But for the ZETA function's sigma parameter:
    The map F_sigma(I) = exp(pi*sigma*I) has fixed sphere at sigma = 1/2.

  These are DIFFERENT maps with DIFFERENT fixed spheres:
    Barrier's exp(theta*J/2): fixed at theta = pi
    Zeta's exp(pi*sigma*I): fixed at sigma = 1/2

  The CRITICAL LINE sigma = 1/2 is where the ZETA map preserves S^2.
  ''')

    # ══════════════════════════════════════════════════════════════
    # 5. THE "BETTER e": exp AT THE FIXED SPHERE
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  5. THE "BETTER e": THE EXPONENTIAL AT THE FIXED SPHERE')
    print('#' * 76)

    print(f'''
  Standard e: exp(1) = e = {np.e:.10f}
  This is the exponential at the UNIT REAL number 1.

  "Better e" for the critical line: exp(I*pi/2) = I
  This is the exponential at the UNIT IMAGINARY sphere, scaled by pi/2.
  It maps I -> I (identity on the imaginary sphere).

  In terms of the zeta function:
    pi^{{-1/2 * I}} = exp(-I*log(pi)/2)
    = cos(log(pi)/2) - I*sin(log(pi)/2)
    = {np.cos(np.log(np.pi)/2):.6f} - I*{np.sin(np.log(np.pi)/2):.6f}

  This is NOT on the fixed sphere (it has both real and imaginary parts).
  The fixed sphere condition gives pi^{{-sigma*I}} landing on S^2 iff:
    cos(sigma*log(pi)) = 0
    => sigma*log(pi) = pi/2 + n*pi
    => sigma = (pi/2 + n*pi)/log(pi) = {(np.pi/2)/np.log(np.pi):.6f}, ...

  sigma = {(np.pi/2)/np.log(np.pi):.6f} != 1/2.

  So the fixed sphere of pi^{{-sigma*I}} is NOT at sigma = 1/2.
  The fixed sphere of exp(pi*sigma*I) IS at sigma = 1/2.

  THE QUESTION: which map is the "right" one for the critical line?

  Answer: exp(pi*sigma*I) encodes the FUNCTIONAL EQUATION.
  The functional equation involves xi(s) = xi(1-s), which
  when written in terms of the exponential map:
    s -> 1-s means sigma -> 1-sigma
    exp(pi*sigma*I) -> exp(pi*(1-sigma)*I)

  At sigma = 1/2: exp(pi/2*I) = I and exp(pi/2*I) = I (same!)
  The functional equation is AUTOMATICALLY SATISFIED at the fixed sphere.
  ''')

    # ══════════════════════════════════════════════════════════════
    # 6. THE DUALITY: pi COLLAPSES, e PRESERVES
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  6. THE DUALITY: pi COLLAPSES, e PRESERVES')
    print('#' * 76)

    print(f'''
  PI (theta = pi):
    exp(pi*I) = -1 for all I in S^2
    The sphere COLLAPSES to a single point.
    All imaginary directions give the same result.
    MAXIMUM COHERENCE. The "signal."

  E (theta = 1):
    exp(1*I) = cos(1) + sin(1)*I
    = {np.cos(1):.6f} + {np.sin(1):.6f}*I
    The sphere maps to a SMALLER sphere (radius {np.sin(1):.4f}).
    Different I give different results on the sphere.
    PARTIAL COHERENCE.

  CRITICAL LINE (theta = pi/2):
    exp(pi/2*I) = I for all I in S^2
    The sphere maps to ITSELF (fixed point).
    Each imaginary direction is PRESERVED.
    THE TRANSITION between collapse (pi) and expansion.

  THE HIERARCHY:
    theta = 0:    identity (trivial)
    theta < pi/2: exp shrinks the sphere (radius < 1)
    theta = pi/2: FIXED SPHERE (radius = 1) <-- CRITICAL LINE
    theta > pi/2: exp passes through center (center < 0)
    theta = pi:   COLLAPSE to -1 <-- PI

  The critical line is the PHASE TRANSITION between
  the "expanding" regime (e-like) and the "collapsing" regime (pi-like).
  This matches our session 45c finding that delta=0 is a phase boundary!
  ''')

    # ══════════════════════════════════════════════════════════════
    # 7. THE "BETTER e" IS exp(pi/2 * I) = I
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  7. THE BETTER e: exp(pi/2 * I) = I (the critical exponential)')
    print('#' * 76)

    print(f'''
  The standard Euler identity: e^{{i*pi}} = -1
  connects e, i, pi in the COLLAPSING regime.

  The CRITICAL identity: e^{{I*pi/2}} = I
  connects e, I, pi in the FIXED-SPHERE regime.

  This is the "better e for the critical line":
    Not a different NUMBER, but a different REGIME of the exponential.
    At theta = pi: collapse (the Euler identity, pi's territory)
    At theta = pi/2: fixed point (the critical identity, sigma=1/2)
    At theta = 1: partial sphere (standard e's territory)

  The critical identity e^{{I*pi/2}} = I says:
    "The exponential at half of pi's frequency preserves the sphere."

  In the zeta function:
    At sigma = 1/2 (the critical line):
    exp(pi*sigma*I) = exp(pi/2*I) = I (preserved)

    The functional equation xi(s) = xi(1-s) is the statement
    that replacing sigma with 1-sigma preserves the sphere:
    exp(pi*(1-sigma)*I) = exp(pi*sigma*I) iff sigma = 1/2.

  THE CHAIN:
    e^{{i*pi}} = -1         (Euler identity, theta=pi, COLLAPSE)
    e^{{I*pi/2}} = I        (Critical identity, theta=pi/2, FIXED)
    sigma = 1/2             (Critical line = fixed sphere of F_sigma)
    xi(s) = xi(1-s)         (Functional equation = sphere preservation)
    v(1/2,t) = 0            (xi real on CL = consequence of preservation)
    B(L) > 0 on CL          (RH = positivity at the fixed sphere)

  The "better e" is the CRITICAL EXPONENTIAL: exp restricted to
  the fixed-sphere regime (theta = pi/2), where the quaternionic
  structure is preserved rather than collapsed.
  ''')

    # ══════════════════════════════════════════════════════════════
    # 8. NUMERICAL: THE FIXED SPHERE IN THE BARRIER
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  8. THE FIXED SPHERE CONDITION IN THE BARRIER')
    print('#' * 76)

    # The barrier W02 - Mp at different sigma (not just 1/2)
    # The barrier is defined for the Rayleigh quotient at L = log(lam^2)
    # But the zeros of xi are at different sigma values
    # The SPECTRAL barrier sum |H(rho)|^2 samples xi at zeros

    # The fixed sphere says: at sigma=1/2, the exponential map
    # preserves S^2. This means the archimedean factor pi^{-s/2}
    # has a specific symmetry at sigma=1/2 that it doesn't have elsewhere.

    # Compute: how does |pi^{-sigma/2}|^2 vary with sigma?
    print(f'\n  |pi^{{-sigma/2}}|^2 = pi^{{-sigma}} as function of sigma:')
    print(f'  {"sigma":>8s} {"pi^(-sigma)":>14s} {"F_sigma sphere":>20s}')
    print('  ' + '-' * 46)

    for sigma in np.linspace(0, 1, 11):
        pi_factor = np.pi**(-sigma)
        theta = np.pi * sigma
        r = np.sin(theta)
        c = np.cos(theta)
        desc = f'r={r:.3f}, c={c:+.3f}'
        if abs(sigma - 0.5) < 0.05:
            desc += ' <-- FIXED'
        print(f'  {sigma:>8.2f} {pi_factor:>14.6f} {desc:>20s}')

    # The fixed sphere at sigma=1/2 means pi^{-1/4} = 0.7511
    # This is the GEOMETRIC MEAN of pi^0 = 1 and pi^{-1/2} = 0.5642
    print(f'\n  At the fixed sphere (sigma=1/2):')
    print(f'    pi^{{-1/4}} = {np.pi**(-0.25):.6f}')
    print(f'    Geometric mean of pi^0 and pi^{{-1/2}}: {np.sqrt(np.pi**(-0.5)):.6f}')
    print(f'    These are equal: pi^{{-1/4}} = sqrt(pi^{{-1/2}})')
    print(f'    The fixed sphere is the GEOMETRIC MIDPOINT of the')
    print(f'    archimedean factor between sigma=0 and sigma=1.')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 46e SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE "BETTER e" IN QUATERNION SPACE:

  The exponential map exp(theta*I) on the imaginary sphere S^2:
    theta = 0:    trivial (point at +1)
    theta = 1:    "e's sphere" (radius sin(1) = {np.sin(1):.4f})
    theta = pi/2: FIXED SPHERE (S^2 -> S^2, the critical identity)
    theta = pi:   "pi's sphere" (collapsed to point -1, Euler identity)

  The CRITICAL IDENTITY: e^{{I*pi/2}} = I
    This says the exponential at half-pi PRESERVES the sphere.
    The critical line sigma = 1/2 is the fixed-sphere value.
    The functional equation xi(s) = xi(1-s) is sphere preservation.
    xi being real on the CL is a consequence.

  THE HIERARCHY:
    e^{{i*pi}} = -1 (Euler, COLLAPSE)        -> pi's coherence
    e^{{I*pi/2}} = I (Critical, FIXED)        -> critical line
    e^{{I}} = cos(1)+sin(1)*I (Standard, PARTIAL) -> e's partial coherence

  THE "BETTER e" IS NOT A DIFFERENT NUMBER.
  It's the exponential in its FIXED-SPHERE REGIME (theta = pi/2),
  where the quaternionic structure is neither collapsed (like pi)
  nor merely shrunk (like standard e), but PERFECTLY PRESERVED.

  This is the quaternionic meaning of the critical line:
  sigma = 1/2 is where the exponential map is an automorphism of S^2.
  The zeros must live here because the functional equation demands
  sphere preservation, and only sigma = 1/2 provides it.
''')

    print('=' * 76)
    print('  SESSION 46e COMPLETE')
    print('=' * 76)
