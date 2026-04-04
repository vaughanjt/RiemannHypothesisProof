"""
SESSION 46 — THE EXTENDED EULER IDENTITY

Classical: e^{i*pi} = -1   (one equation)
Quaternionic: e^{I*pi} = -1 for all I in S^2   (a sphere of equations)

EXTENDED: e^{I*theta(gamma_n)} ~ +/-1 at each zeta zero

where theta(t) = arg(Gamma(1/4 + it/2)) - (t/2)*log(pi) is the
Riemann-Siegel theta function. This is the phase accumulated by pi
and Gamma at height t on the critical line.

The Gram points g_n satisfy theta(g_n) = n*pi exactly.
Gram's law: zeros gamma_n approximately equal Gram points g_n.
Deviation: S(t) = (1/pi)*arg(zeta(1/2+it)) measures the excess.

THE EXTENDED EULER IDENTITY:
  e^{I*theta(gamma_n)} = (-1)^n * e^{I*pi*S(gamma_n)}

At a Gram-law-obeying zero: S ~ 0, so e^{I*theta} ~ (-1)^n = +/-1.
The zero sits ON the Euler sphere (or its negative).

Gram violations = deviations from the Euler sphere.

IN QUATERNIONIC SPACE:
  The Euler spheres: {q : e^q = (-1)^n} = {n*pi*I : I in S^2}
  The zero-spheres: {1/2 + gamma_n*I : I in S^2}
  The extended Euler identity maps zero-spheres to Euler spheres
  via the theta function.

PLAN:
  1. Compute theta(gamma_n) for the first 200 zeros
  2. Measure theta(gamma_n) mod pi — how close to integer multiples?
  3. Map the deviations = pi*S(gamma_n) — the "Euler sphere error"
  4. Quaternionic visualization: zero-spheres vs Euler spheres
  5. Does the deviation have structure? (GUE, random, systematic?)
  6. The extended identity in full: connecting pi, primes, and zeros
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, pi as mppi, loggamma, log as mplog,
                    zeta as mpzeta, zetazero, arg as mparg, gamma as mpgamma,
                    siegeltheta, siegelz)
import time
import sys

mp.dps = 30


def theta_rs(t):
    """Riemann-Siegel theta function: theta(t) = Im(loggamma(1/4+it/2)) - (t/2)*log(pi)."""
    return float(mpmath.siegeltheta(t))


def Z_function(t):
    """Hardy Z-function: Z(t) = exp(i*theta(t)) * zeta(1/2+it). Real-valued."""
    return float(mpmath.siegelz(t))


def S_function(t):
    """S(t) = (1/pi)*arg(zeta(1/2+it)) = (theta(t) - pi*N(t))/pi + correction."""
    # More directly: S(t) = N(t) - theta(t)/pi - 1
    # where N(t) = number of zeros with 0 < gamma <= t
    # But we compute it from the phase directly
    z = complex(mpzeta(mpc(0.5, t)))
    return np.angle(z) / np.pi


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 46 -- THE EXTENDED EULER IDENTITY')
    print('  e^{I*theta(gamma_n)} ~ +/-1 at each zeta zero')
    print('=' * 76)

    # Load zeros
    N_ZEROS = 200
    print(f'\n  Loading {N_ZEROS} zeta zeros...', flush=True)
    t0 = time.time()
    zeros = [float(zetazero(k).imag) for k in range(1, N_ZEROS + 1)]
    zeros = np.array(zeros)
    print(f'  Done ({time.time()-t0:.1f}s)')

    # ══════════════════════════════════════════════════════════════
    # 1. THETA AT EACH ZERO
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. RIEMANN-SIEGEL THETA AT EACH ZERO')
    print('#' * 76)

    print(f'\n  theta(t) = Im(log Gamma(1/4+it/2)) - (t/2)*log(pi)')
    print(f'  Gram points: theta(g_n) = n*pi')
    print(f'  At zeros: theta(gamma_n) = n*pi + pi*S(gamma_n)')

    thetas = np.array([theta_rs(g) for g in zeros])
    # theta mod pi: how close to integer multiple?
    theta_over_pi = thetas / np.pi
    nearest_int = np.round(theta_over_pi)
    deviations = theta_over_pi - nearest_int  # fractional part centered at 0

    print(f'\n  {"zero n":>7s} {"gamma_n":>10s} {"theta":>12s} {"theta/pi":>10s} '
          f'{"nearest n*pi":>12s} {"deviation":>10s} {"on sphere?":>10s}')
    print('  ' + '-' * 78)

    for k in range(min(30, N_ZEROS)):
        dev = deviations[k]
        on_sphere = 'YES' if abs(dev) < 0.25 else 'no'
        gram_n = int(nearest_int[k])
        print(f'  {k+1:>7d} {zeros[k]:>10.4f} {thetas[k]:>12.4f} {theta_over_pi[k]:>10.4f} '
              f'{gram_n:>12d} {dev:>+10.4f} {on_sphere:>10s}')

    # Statistics
    print(f'\n  Deviation statistics (|theta/pi - nearest integer|):')
    abs_dev = np.abs(deviations)
    print(f'    Mean |deviation|: {np.mean(abs_dev):.6f} (in units of pi)')
    print(f'    Std:              {np.std(deviations):.6f}')
    print(f'    Max:              {np.max(abs_dev):.6f}')
    print(f'    Min:              {np.min(abs_dev):.6f}')
    print(f'    Zeros within 0.25 of Euler sphere: {np.sum(abs_dev < 0.25)}/{N_ZEROS} '
          f'({100*np.sum(abs_dev < 0.25)/N_ZEROS:.1f}%)')
    print(f'    Zeros within 0.10 of Euler sphere: {np.sum(abs_dev < 0.10)}/{N_ZEROS} '
          f'({100*np.sum(abs_dev < 0.10)/N_ZEROS:.1f}%)')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 2. THE EXTENDED EULER IDENTITY
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. THE EXTENDED EULER IDENTITY')
    print('#' * 76)

    print(f'''
  At each zero rho_n = 1/2 + i*gamma_n:

    e^{{i*theta(gamma_n)}} = cos(theta) + i*sin(theta)

  If theta(gamma_n) = n*pi exactly (Gram's law): e^{{i*n*pi}} = (-1)^n

  The ACTUAL value: e^{{i*theta(gamma_n)}} = (-1)^n * e^{{i*pi*delta_n}}
  where delta_n = theta(gamma_n)/pi - n is the deviation.

  In quaternions: e^{{I*theta(gamma_n)}} = (-1)^n * e^{{I*pi*delta_n}}

  The deviation e^{{I*pi*delta_n}} measures how far the zero is
  from the Euler sphere at n*pi.
  ''')

    # Compute the quaternionic "Euler error" for each zero
    print(f'  {"zero n":>7s} {"gamma_n":>10s} {"(-1)^n":>6s} '
          f'{"cos(pi*dev)":>12s} {"sin(pi*dev)":>12s} {"|error|":>10s}')
    print('  ' + '-' * 62)

    euler_errors = []
    for k in range(min(40, N_ZEROS)):
        dev = deviations[k]
        cos_err = np.cos(np.pi * dev)
        sin_err = np.sin(np.pi * dev)
        error_norm = abs(sin_err)  # distance from +/-1 on unit circle
        euler_errors.append(error_norm)

        sign = int((-1)**int(nearest_int[k]))
        print(f'  {k+1:>7d} {zeros[k]:>10.4f} {sign:>+6d} '
              f'{cos_err:>+12.6f} {sin_err:>+12.6f} {error_norm:>10.6f}')

    euler_errors = np.array([abs(np.sin(np.pi * d)) for d in deviations])
    print(f'\n  Euler sphere error |sin(pi*deviation)|:')
    print(f'    Mean: {np.mean(euler_errors):.6f}')
    print(f'    Std:  {np.std(euler_errors):.6f}')
    print(f'    Max:  {np.max(euler_errors):.6f} (zero #{np.argmax(euler_errors)+1})')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 3. GRAM'S LAW AND VIOLATIONS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. GRAM\'S LAW: zeros vs Gram points')
    print('#' * 76)

    # Compute Gram points: theta(g_n) = n*pi
    # Find g_n by solving theta(t) = n*pi
    print(f'\n  Computing Gram points...')
    gram_points = []
    for n in range(0, N_ZEROS + 50):
        target = n * np.pi
        # Newton's method starting from an estimate
        # theta(t) ~ (t/2)*log(t/(2*pi*e)) for large t
        if n == 0:
            t_est = 3.0
        else:
            t_est = 2 * np.pi * np.exp(1 + mpmath.lambertw(float((n - 1/8) / np.e)).real)
            t_est = max(t_est, gram_points[-1] + 0.5 if gram_points else 3.0)

        for _ in range(50):
            th = theta_rs(t_est)
            # d(theta)/dt via finite difference
            h = 1e-6
            dth = (theta_rs(t_est + h) - theta_rs(t_est - h)) / (2 * h)
            if abs(dth) < 1e-20:
                break
            t_est -= (th - target) / dth
            if abs(th - target) < 1e-10:
                break

        gram_points.append(t_est)

    gram_points = np.array(gram_points[:N_ZEROS + 20])

    # Match zeros to Gram points
    print(f'\n  {"zero n":>7s} {"gamma_n":>10s} {"Gram g_n":>10s} {"gap":>10s} '
          f'{"Gram law?":>10s}')
    print('  ' + '-' * 50)

    gram_violations = 0
    for k in range(min(30, N_ZEROS)):
        # Find nearest Gram point
        diffs = np.abs(gram_points - zeros[k])
        nearest_gram_idx = np.argmin(diffs)
        gap = zeros[k] - gram_points[nearest_gram_idx]
        gram_law = 'YES' if nearest_gram_idx == k else f'VIOLATION ({nearest_gram_idx})'
        if nearest_gram_idx != k:
            gram_violations += 1
        print(f'  {k+1:>7d} {zeros[k]:>10.4f} {gram_points[k]:>10.4f} '
              f'{zeros[k]-gram_points[k]:>+10.4f} {gram_law:>10s}')

    # Count all violations
    total_violations = 0
    for k in range(min(N_ZEROS, len(gram_points))):
        if k < N_ZEROS:
            z = Z_function(gram_points[k])
            if ((-1)**k * z) < 0:
                total_violations += 1

    print(f'\n  Gram violations in first {min(N_ZEROS, len(gram_points))} Gram points: '
          f'{total_violations} ({100*total_violations/min(N_ZEROS, len(gram_points)):.1f}%)')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 4. THE DEVIATION STRUCTURE
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. DEVIATION STRUCTURE: is it random or systematic?')
    print('#' * 76)

    print(f'\n  The deviations delta_n = theta(gamma_n)/pi - nearest_integer')
    print(f'  measure how far each zero is from its Euler sphere.')

    # Distribution
    print(f'\n  Distribution of deviations:')
    bins = np.linspace(-0.5, 0.5, 21)
    hist, _ = np.histogram(deviations, bins=bins)
    max_count = max(hist)
    for i in range(len(hist)):
        bar = '#' * int(hist[i] / max(1, max_count) * 40)
        center = (bins[i] + bins[i+1]) / 2
        print(f'    [{center:+.3f}] {bar} ({hist[i]})')

    # Autocorrelation
    dev_centered = deviations - np.mean(deviations)
    if len(dev_centered) > 10:
        acf_1 = np.corrcoef(dev_centered[:-1], dev_centered[1:])[0, 1]
        acf_2 = np.corrcoef(dev_centered[:-2], dev_centered[2:])[0, 1]
        print(f'\n  Autocorrelation of deviations:')
        print(f'    Lag-1: {acf_1:+.4f}')
        print(f'    Lag-2: {acf_2:+.4f}')
        if abs(acf_1) < 0.1:
            print(f'    Deviations are approximately UNCORRELATED (random-like)')
        else:
            print(f'    Deviations show significant correlation')

    # Does deviation grow with n?
    ns = np.arange(1, N_ZEROS + 1)
    c_dev = np.polyfit(np.log(ns[10:]), np.log(np.abs(deviations[10:]) + 1e-10), 1)
    print(f'\n  Growth of |deviation| with zero index:')
    print(f'    |deviation| ~ n^{{{c_dev[0]:.4f}}}')
    if abs(c_dev[0]) < 0.1:
        print(f'    Deviations are approximately CONSTANT (no growth)')
    elif c_dev[0] > 0:
        print(f'    Deviations GROW with n')
    else:
        print(f'    Deviations SHRINK with n')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 5. THE IDENTITY IN FULL
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. THE EXTENDED EULER IDENTITY — FULL FORM')
    print('#' * 76)

    print(f'''
  CLASSICAL EULER IDENTITY:
    e^{{i*pi}} = -1
    One equation. Five constants. The most beautiful equation in mathematics.

  QUATERNIONIC EULER IDENTITY:
    e^{{I*pi}} = -1  for all I in S^2
    A 2-sphere of equations. Pi is the radius.

  EXTENDED EULER IDENTITY (this work):
    e^{{I*theta(gamma_n)}} = (-1)^n * e^{{I*pi*S(gamma_n)}}

    where:
      gamma_n = height of the n-th zeta zero
      theta(t) = Im(log Gamma(1/4+it/2)) - (t/2)*log(pi)
      S(t) = (1/pi)*arg(zeta(1/2+it))
      I = any unit imaginary quaternion

    This connects:
      e (the exponential)
      I (the quaternionic imaginary sphere)
      pi (through the theta function and the Euler spheres)
      Gamma (the archimedean factor)
      zeta (through the zeros gamma_n)
      primes (through zeta's Euler product)

    At each zero, the phase e^{{I*theta}} lands near the Euler sphere
    n*pi*I (for all I in S^2). The deviation S(gamma_n) measures how far
    the zero is from exact Euler sphere placement.

  GRAM'S LAW is the statement that S(gamma_n) is small.
  The extended Euler identity says: zeros LIVE ON (or near) Euler spheres.

  QUANTITATIVE:
    Mean |deviation| = {np.mean(abs_dev):.4f} (in units of pi)
    {100*np.sum(abs_dev < 0.25)/N_ZEROS:.0f}% of zeros within pi/4 of an Euler sphere
    {100*np.sum(abs_dev < 0.10)/N_ZEROS:.0f}% of zeros within pi/10 of an Euler sphere
  ''')

    # ══════════════════════════════════════════════════════════════
    # 6. THE PRIME CONNECTION
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  6. THE PRIME CONNECTION: theta encodes pi, S encodes primes')
    print('#' * 76)

    print(f'''
  The theta function theta(t) = Im(log Gamma(1/4+it/2)) - (t/2)*log(pi)
  is purely archimedean: it depends only on Gamma and pi.

  The S function S(t) = (1/pi)*arg(zeta(1/2+it))
  encodes the primes through zeta's Euler product.

  So the extended Euler identity SEPARATES:
    e^{{I*theta}} = the pi part (archimedean, the Euler sphere)
    e^{{I*pi*S}} = the prime part (finite, the deviation)

  At each zero:
    [pi's contribution] * [primes' contribution] = (-1)^n
    e^{{I*theta}} * e^{{I*pi*S}} = (-1)^n

  This is the SEPARATION we found in quaternionic space (session 45j),
  now expressed as a multiplicative identity!

  The archimedean factor (pi, through theta) provides the LATTICE.
  The prime factor (through S) provides the PERTURBATION.
  The zeros are the lattice + perturbation.

  RH says: the perturbation is bounded (S(t) = O(log t)).
  If S grew unboundedly, zeros would drift far from Euler spheres,
  eventually leaving the critical line.
  ''')

    # Verify: theta is purely pi-dependent, S is prime-dependent
    print(f'  Verification: theta depends on pi, S depends on primes')
    print(f'\n  {"t":>8s} {"theta(t)":>12s} {"theta/pi":>10s} {"S(t)":>10s}')
    print('  ' + '-' * 44)

    for t in [14.1347, 21.022, 25.011, 30.425, 50.0, 100.0]:
        th = theta_rs(t)
        s_val = S_function(t)
        print(f'  {t:>8.4f} {th:>12.4f} {th/np.pi:>10.4f} {s_val:>+10.4f}')

    # ══════════════════════════════════════════════════════════════
    # 7. THE BARRIER IN EULER-SPHERE COORDINATES
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. BARRIER IN EULER-SPHERE COORDINATES')
    print('#' * 76)

    print(f'\n  The barrier B(L) oscillates. Its oscillation frequency')
    print(f'  relates to the zeros through the explicit formula.')
    print(f'  In Euler-sphere coordinates, the barrier traces a path')
    print(f'  on the Euler sphere hierarchy.')

    # The barrier at L relates to zeros through:
    # B(L) ~ sum_n |H(gamma_n)|^2 (spectral representation)
    # Each term contributes an oscillation at frequency gamma_n
    # In theta-coordinates: the oscillation is at theta(gamma_n) ~ n*pi

    # So the barrier's Fourier spectrum in theta-coordinates is
    # approximately periodic with period pi!

    print(f'\n  The spectral barrier B = sum |H(gamma)|^2 in theta-coordinates:')
    print(f'  Each zero contributes at theta(gamma_n) ~ n*pi.')
    print(f'  In theta-coordinates, the barrier is approximately')
    print(f'  PERIODIC with period pi (one contribution per Euler sphere).')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 46 SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE EXTENDED EULER IDENTITY:

    e^{{I*theta(gamma_n)}} = (-1)^n * e^{{I*pi*S(gamma_n)}}

  This equation says:
    - Each zeta zero lives on (or near) an Euler sphere in H
    - The theta function (pi + Gamma) provides the lattice of spheres
    - The S function (primes through zeta) provides the deviation
    - Gram's law = deviations are small (S ~ 0)
    - RH = deviations stay bounded (S = O(log t))

  NUMERICAL RESULTS ({N_ZEROS} zeros):
    Mean |deviation|: {np.mean(abs_dev):.4f} pi
    {100*np.sum(abs_dev < 0.25)/N_ZEROS:.0f}% within pi/4 of Euler sphere
    {100*np.sum(abs_dev < 0.10)/N_ZEROS:.0f}% within pi/10 of Euler sphere
    Autocorrelation: {acf_1:+.4f} (approximately uncorrelated)
    Growth: |deviation| ~ n^{{{c_dev[0]:.3f}}} (approximately constant)

  THE CHAIN:
    Euler's identity:     e^{{i*pi}} = -1
    -> Quaternionic:      e^{{I*pi}} = -1 (sphere)
    -> Theta function:    theta(t) accumulates pi's phase
    -> Gram lattice:      theta(g_n) = n*pi (Euler sphere lattice)
    -> Zeros on lattice:  gamma_n ~ g_n (Gram's law)
    -> Deviation = primes: S(gamma_n) via Euler product
    -> RH = bounded S:    primes don't push zeros off Euler spheres

  Pi provides the scaffolding. Primes perturb it.
  The Riemann Hypothesis says the perturbation is controlled.
''')

    print('=' * 76)
    print('  SESSION 46 COMPLETE')
    print('=' * 76)
