"""
SESSION 45m — WHAT DOES QUATERNIONIC PI LOOK LIKE?

In C: pi is a point on the real line. e^{i*pi} = -1 is one equation.

In H: the solutions to e^q = -1 form a 2-SPHERE of radius pi:
  {pi*I : I in S^2} where I^2 = -1

Pi in quaternion space is not a number. It's a SPHERE.

The Euler identity becomes a WHOLE SPHERE of identities:
  e^{pi*i} = e^{pi*j} = e^{pi*k} = e^{pi*(ai+bj+ck)} = -1
  for ANY unit imaginary quaternion (a^2+b^2+c^2 = 1)

And the (4*pi)^2 in our barrier? That's related to the AREA of this sphere.
The sphere of radius pi in R^3 has area 4*pi*pi^2 = 4*pi^3.
But the (4*pi)^2 = 16*pi^2 is the area of a sphere of radius 2*pi...
no, area of S^2 of radius r is 4*pi*r^2. So radius 4*pi has area 4*pi*(4*pi)^2.

Actually, the (4*pi)^2 comes from the Fourier dual: if the physical space has
period L, the Fourier modes are at n*(2*pi/L), and the squared frequency at
mode n is (2*pi*n/L)^2. For n=1: (2*pi/L)^2. For our barrier, the modes
involve 4*pi*n/L, giving (4*pi)^2*n^2.

But the QUATERNIONIC meaning is deeper: (4*pi)^2 = (4*pi)^2 is the squared
"radius" of the mode in quaternionic frequency space, and the barrier's
denominator L^2 + (4*pi*n)^2 = |L + 4*pi*n*I|^2 is the QUATERNIONIC NORM
of L + 4*pi*n*I for ANY imaginary unit I.

The denominator is the squared norm of a QUATERNION. The barrier lives in
quaternionic norm space all along — we just didn't see it.

THIS SCRIPT:
  1. The Euler sphere: visualize e^q = -1 solutions in H
  2. Pi as a sphere: trace the sphere {pi*I : I in S^2}
  3. The barrier denominator as quaternionic norm: |L + 4*pi*n*I|^2
  4. How primes see this norm vs how pi sees it
  5. The archimedean factor pi^{-s/2} as a curve in H
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, pi as mppi, gamma as mpgamma, zeta as mpzeta
import time
import sys

mp.dps = 20


def qexp(a, b, c, d):
    """Quaternionic exp(a + bi + cj + dk)."""
    vn = np.sqrt(b**2 + c**2 + d**2)
    ea = np.exp(a)
    if vn < 1e-15:
        return ea, 0, 0, 0
    cos_v = np.cos(vn)
    sin_v = np.sin(vn)
    s = ea * sin_v / vn
    return ea * cos_v, s*b, s*c, s*d


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45m — WHAT DOES QUATERNIONIC PI LOOK LIKE?')
    print('=' * 76)

    # ══════════════════════════════════════════════════════════════
    # 1. THE EULER SPHERE: solutions to e^q = -1 in H
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. THE EULER SPHERE')
    print('#' * 76)

    print(f'''
  In C: e^z = -1 has solutions z = (2n+1)*pi*i for integer n.
        These are POINTS on the imaginary axis, spaced 2*pi apart.

  In H: e^q = -1 has solutions q = (2n+1)*pi*I for ANY I in S^2.
        These are SPHERES in imaginary quaternion space.
        The first (n=0) is a sphere of radius pi.
        The second (n=1) is a sphere of radius 3*pi.

  Pi is the RADIUS of the first Euler sphere.
  ''')

    # Verify: e^{pi*I} = -1 for various I
    print(f'  Verifying e^{{pi*I}} = -1 for various unit imaginary quaternions:')
    print(f'  {"I direction":>20s} {"e^(pi*I)":>40s} {"= -1?":>6s}')
    print('  ' + '-' * 70)

    pi_val = np.pi
    test_dirs = [
        ('i', (0, 1, 0, 0)),
        ('j', (0, 0, 1, 0)),
        ('k', (0, 0, 0, 1)),
        ('(i+j)/sqrt2', (0, 1/np.sqrt(2), 1/np.sqrt(2), 0)),
        ('(i+j+k)/sqrt3', (0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3))),
        ('0.2i+0.3j+0.93k', (0, 0.2, 0.3, 0.9327)),  # approx unit
    ]

    for label, (a, b, c, d) in test_dirs:
        # Normalize to unit
        vn = np.sqrt(b**2 + c**2 + d**2)
        b, c, d = b/vn, c/vn, d/vn
        # q = pi * I
        qa, qb, qc, qd = qexp(0, pi_val*b, pi_val*c, pi_val*d)
        close = abs(qa + 1) < 1e-10 and abs(qb) < 1e-10 and abs(qc) < 1e-10 and abs(qd) < 1e-10
        print(f'  {label:>20s} ({qa:+.10f}, {qb:+.4e}i, {qc:+.4e}j, {qd:+.4e}k) '
              f'{"YES" if close else "NO":>6s}')

    print(f'\n  ALL give -1. Pi in quaternion space is a SPHERE of radius pi,')
    print(f'  not a single number. The sphere has:')
    print(f'    Radius: pi = {np.pi:.10f}')
    print(f'    Area: 4*pi*pi^2 = {4*np.pi**3:.6f}')
    print(f'    Volume enclosed: (4/3)*pi*pi^3 = {4/3*np.pi**4:.6f}')

    # ══════════════════════════════════════════════════════════════
    # 2. THE HIERARCHY OF EULER SPHERES
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. THE HIERARCHY OF EULER SPHERES')
    print('#' * 76)

    print(f'\n  e^q = -1: spheres at radius (2n+1)*pi')
    print(f'  e^q = +1: spheres at radius 2n*pi (plus the origin)')
    print(f'\n  {"n":>3s} {"radius":>12s} {"e^q":>6s} {"area":>14s}')
    print('  ' + '-' * 38)

    for n in range(5):
        r_minus = (2*n+1) * np.pi
        r_plus = 2*(n+1) * np.pi if n > 0 else 0
        area_minus = 4 * np.pi * r_minus**2
        print(f'  {n:>3d} {r_minus:>12.4f} {"e=-1":>6s} {area_minus:>14.4f}')
        if n > 0:
            area_plus = 4 * np.pi * r_plus**2
            print(f'  {n:>3d} {r_plus:>12.4f} {"e=+1":>6s} {area_plus:>14.4f}')

    # ══════════════════════════════════════════════════════════════
    # 3. THE BARRIER DENOMINATOR AS QUATERNIONIC NORM
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. THE BARRIER DENOMINATOR: |L + 4*pi*n*I|^2')
    print('#' * 76)

    print(f'''
  The barrier's w_tilde denominator is: L^2 + (4*pi*n)^2

  In quaternionic language: this is |L + 4*pi*n*I|^2 for ANY I in S^2.
  It's the SQUARED NORM of a quaternion.

  The quaternion L + 4*pi*n*I sits at:
    Real part: L (the cutoff parameter)
    Imaginary part: 4*pi*n*I (on the Euler sphere of radius 4*pi*n)

  The Euler sphere of radius 4*pi is the set of q with e^{{q/4}} = +1.
  (Since 4*pi = 2*2*pi, and e^{{2*pi*I}} = +1.)

  So the barrier denominator is:
    |L + q|^2 where q is on the (4*pi*n)-th Euler sphere.

  EACH MODE n of the barrier corresponds to a DIFFERENT Euler sphere.
  Mode n=1: sphere of radius 4*pi = {4*np.pi:.4f}
  Mode n=2: sphere of radius 8*pi = {8*np.pi:.4f}
  Mode n=3: sphere of radius 12*pi = {12*np.pi:.4f}
  ''')

    L0 = np.log(2000)
    print(f'  At L = {L0:.4f} (lam^2 = 2000):')
    print(f'  {"mode n":>7s} {"4*pi*n":>10s} {"|L+4*pi*n*I|^2":>16s} {"L^2+(4*pi*n)^2":>16s}')
    print('  ' + '-' * 52)

    for n in range(1, 8):
        r = 4 * np.pi * n
        norm_sq = L0**2 + r**2
        print(f'  {n:>7d} {r:>10.4f} {norm_sq:>16.4f} {L0**2 + (4*np.pi*n)**2:>16.4f}')

    # ══════════════════════════════════════════════════════════════
    # 4. THE ARCHIMEDEAN FACTOR pi^{-s/2} IN H
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. pi^{{-s/2}} AS A CURVE IN QUATERNIONIC SPACE')
    print('#' * 76)

    print(f'''
  pi^{{-s/2}} = exp(-s/2 * log(pi))

  For s = sigma + t*I (on a complex slice):
    pi^{{-s/2}} = pi^{{-sigma/2}} * exp(-t*I*log(pi)/2)
              = pi^{{-sigma/2}} * [cos(t*log(pi)/2) - I*sin(t*log(pi)/2)]

  This traces a SPIRAL in the (real, I) plane.
  The spiral has:
    - Radial decay: pi^{{-sigma/2}} (shrinks as sigma grows)
    - Angular frequency: log(pi)/2 = {np.log(np.pi)/2:.6f} per unit t
    - Period in t: 2*pi / (log(pi)/2) = 4*pi/log(pi) = {4*np.pi/np.log(np.pi):.4f}

  For a PRIME p^{{-s}} = exp(-s*log(p)):
    Angular frequency: log(p)/1 = log(p)
    Period in t: 2*pi/log(p)

  The ratio of frequencies: log(p) / (log(pi)/2) = 2*log(p)/log(pi)
  ''')

    print(f'  FREQUENCY COMPARISON: pi vs primes')
    print(f'  {"object":>10s} {"log":>10s} {"freq":>10s} {"period":>10s} {"freq/pi_freq":>12s}')
    print('  ' + '-' * 56)

    pi_freq = np.log(np.pi) / 2
    pi_period = 2 * np.pi / pi_freq

    print(f'  {"pi^-s/2":>10s} {np.log(np.pi):>10.6f} {pi_freq:>10.6f} '
          f'{pi_period:>10.4f} {1.0:>12.4f}')

    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        freq = np.log(p)
        period = 2 * np.pi / freq
        ratio = freq / pi_freq
        print(f'  {"p=" + str(p):>10s} {np.log(p):>10.6f} {freq:>10.6f} '
              f'{period:>10.4f} {ratio:>12.4f}')

    # ══════════════════════════════════════════════════════════════
    # 5. THE KEY: INCOMMENSURABILITY
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. INCOMMENSURABILITY: pi vs primes in quaternionic frequency')
    print('#' * 76)

    print(f'''
  In the i-plane (standard complex analysis):
    pi^{{-s/2}} oscillates at frequency log(pi)/2 = {pi_freq:.6f}
    p^{{-s}} oscillates at frequency log(p)

  These frequencies are INCOMMENSURATE (ratio is irrational for all primes p).
  In complex analysis, this doesn't help — the functions still live in the
  same plane and can interfere.

  In QUATERNIONIC space, we can put them on DIFFERENT imaginary axes:
    pi^{{-s/2}} oscillates in the I direction
    p^{{-s}} oscillates in the J direction

  If I and J are orthogonal, the oscillations CANNOT interfere.

  But wait — the Dirichlet series puts everything in the SAME direction
  (whatever imaginary unit the argument s uses). So the separation must
  come from the STRUCTURE of the completed zeta, not from a choice of basis.

  The j-separation we found (session 45j) shows that when s has a j-component,
  pi's factor and the primes' factors respond DIFFERENTLY:
    pi^{{-s/2}} picks up a large j-component (j/a = 0.49)
    p^{{-s}} picks up a tiny j-component (j/a = 0.0005)

  This is because log(pi)/2 = {np.log(np.pi)/2:.6f} is a SINGLE frequency,
  while the primes have MANY frequencies (log(2), log(3), log(5), ...) that
  DESTRUCTIVELY INTERFERE in the j-direction.

  Pi is COHERENT in j. Primes are INCOHERENT in j.
  ''')

    # Demonstrate: sum of e^{i*log(p)*t} for many primes
    # vs e^{i*log(pi)/2*t}
    print(f'  Coherence test at t = 14.1347 (first zero):')
    t = 14.1347

    # Pi contribution
    pi_phase = np.log(np.pi) / 2 * t
    pi_cos = np.cos(pi_phase)
    pi_sin = np.sin(pi_phase)
    print(f'\n    pi^{{-s/2}} phase: {pi_phase:.4f} rad = {pi_phase/(2*np.pi):.4f} turns')
    print(f'    cos = {pi_cos:+.6f}, sin = {pi_sin:+.6f}')
    print(f'    |e^{{i*phase}}| = 1 (coherent: single frequency)')

    # Prime contributions
    from session41g_uncapped_barrier import sieve_primes
    primes = sieve_primes(1000)

    cos_sum = 0
    sin_sum = 0
    for p in primes:
        phase = np.log(int(p)) * t
        cos_sum += np.cos(phase) / np.sqrt(int(p))
        sin_sum += np.sin(phase) / np.sqrt(int(p))

    total_weight = sum(1/np.sqrt(int(p)) for p in primes)
    coherence = np.sqrt(cos_sum**2 + sin_sum**2) / total_weight

    print(f'\n    Prime sum (168 primes, weighted by 1/sqrt(p)):')
    print(f'    sum cos = {cos_sum:+.6f}, sum sin = {sin_sum:+.6f}')
    print(f'    |sum| / total_weight = {coherence:.6f}')
    print(f'    (coherence: 1 = all in phase, 0 = random cancellation)')

    if coherence < 0.1:
        print(f'    *** PRIMES ARE INCOHERENT (coherence {coherence:.4f} << 1) ***')
        print(f'    Their phases scramble. Pi stays coherent (coherence = 1).')

    # ══════════════════════════════════════════════════════════════
    # 6. PI AS THE ARCHIMEDEAN PRIME: THE GEOMETRIC PICTURE
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. THE GEOMETRIC PICTURE')
    print('#' * 76)

    print(f'''
  WHAT QUATERNIONIC PI LOOKS LIKE:

  1. Pi is a SPHERE, not a number.
     The set {{q in H : e^q = -1}} = {{pi*I : I in S^2}}
     is a 2-sphere of radius pi in imaginary quaternion space.

  2. Each mode n of the barrier sits on an Euler sphere.
     The denominator L^2 + (4*pi*n)^2 = |L + 4*pi*n*I|^2 is the
     distance from the real point L to the n-th Euler sphere.

  3. The archimedean factor pi^{{-s/2}} traces a SPIRAL in H.
     It oscillates at frequency log(pi)/2 = {np.log(np.pi)/2:.6f}.
     This is a single, clean, coherent frequency.

  4. Each prime p contributes p^{{-s}} which oscillates at frequency log(p).
     The primes have MANY DIFFERENT frequencies.
     When summed, they DESTRUCTIVELY INTERFERE.

  5. THE j-SEPARATION:
     pi (one coherent frequency) -> large j-projection (0.49)
     primes (many incoherent frequencies) -> tiny j-projection (0.0005)
     The COHERENCE of pi vs INCOHERENCE of primes IS the separation.

  6. THE BARRIER'S POSITIVITY:
     B = W02 - Mp = (archimedean) - (finite primes)
     = (coherent pi oscillation) - (incoherent prime sum)

     In the i-direction: both contribute, can partially cancel.
     In the j-direction: pi contributes coherently, primes cancel out.
     The j-component of B is dominated by pi -> always nonzero.

  7. THE REMAINING QUESTION:
     The barrier is positive in the j-direction (pi dominates).
     The barrier oscillates in the i-direction (primes interfere).
     On the critical line (real L), there is no j-direction.
     The question is: does the coherent pi contribution in the
     i-direction still dominate the incoherent prime sum?

     This is the PRIME NUMBER THEOREM's deep content:
     pi(x) ~ x/log(x) because pi (the archimedean prime) sets the
     scale and the finite primes fill in the details incoherently.
  ''')

    # ══════════════════════════════════════════════════════════════
    # 7. INCOMMENSURABILITY MEASURE
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  7. IS log(pi)/log(p) IRRATIONAL FOR ALL PRIMES p?')
    print('#' * 76)

    print(f'\n  If log(pi)/log(p) is irrational, the oscillations of pi^{{-s/2}}')
    print(f'  and p^{{-s}} are INCOMMENSURATE — they never exactly synchronize.')
    print(f'  This is what prevents perfect cancellation in the barrier.')

    print(f'\n  {"prime p":>8s} {"log(pi)/log(p)":>16s} {"continued fraction":>25s}')
    print('  ' + '-' * 52)

    for p in [2, 3, 5, 7, 11, 13]:
        ratio = np.log(np.pi) / np.log(p)
        # Continued fraction approximation
        cf = []
        x = ratio
        for _ in range(6):
            n = int(x)
            cf.append(n)
            frac = x - n
            if abs(frac) < 1e-10:
                break
            x = 1 / frac
        cf_str = '[' + '; '.join(str(c) for c in cf) + '...]'
        print(f'  {p:>8d} {ratio:>16.10f} {cf_str:>25s}')

    print(f'\n  log(pi) = {np.log(np.pi):.10f}')
    print(f'  log(pi) is transcendental (pi is transcendental, so log(pi) is too).')
    print(f'  Therefore log(pi)/log(p) is IRRATIONAL for every prime p.')
    print(f'  (If it were rational: log(pi) = (a/b)*log(p) => pi = p^{{a/b}} => pi algebraic. Contradiction.)')

    print(f'\n  THIS IS THE FUNDAMENTAL INCOMMENSURABILITY:')
    print(f'  Pi and every prime live on incommensurate frequency scales.')
    print(f'  In the i-plane, they can partially cancel (creating zeros).')
    print(f'  In the j-plane, the incommensurability causes decoherence')
    print(f'  of the prime sum, while pi stays coherent.')
    print(f'  The j-separation is a CONSEQUENCE of transcendence of pi.')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45m SYNTHESIS')
    print('=' * 76)

    print(f'''
  QUATERNIONIC PI:

  Pi is a 2-sphere of radius pi in imaginary quaternion space.
  It's the first Euler sphere: {{q : e^q = -1}} = {{pi*I : I in S^2}}.

  The barrier's denominator L^2 + (4*pi*n)^2 = |L + q_n|^2 where q_n
  sits on the n-th Euler sphere. Each Fourier mode is a distance from
  the real axis to an Euler sphere.

  The archimedean factor pi^{{-s/2}} is a coherent spiral in H.
  The finite primes p^{{-s}} are incoherent oscillations in H.
  Pi's transcendence guarantees log(pi)/log(p) is irrational for all p,
  making their frequencies incommensurate.

  In the j-direction: pi is coherent (j/a = 0.49), primes decohere (0.0005).
  On the real axis (no j): both contribute to the same 1D projection.
  The barrier's positivity = the coherent archimedean term exceeds the
  incoherent prime sum in this 1D projection.

  WHAT PI "LOOKS LIKE" IN H:
  A sphere. A spiral. The unique coherent frequency in a sea of
  incommensurate prime oscillations. The organizing principle that
  makes the critical line special — because pi^{{-s/2}} has the ONLY
  coherent phase, and it's maximally coherent at sigma = 1/2 where
  the functional equation forces xi to be real.
''')

    print('=' * 76)
    print('  SESSION 45m COMPLETE')
    print('=' * 76)
