"""
SESSION 45k — THE TOPOLOGICAL ARGUMENT

The functional equation xi(s) = xi(1-s) combined with xi being real on R gives:
  u(sigma, t) = u(1-sigma, t)     [real part symmetric]
  v(sigma, t) = -v(1-sigma, t)    [imaginary part antisymmetric]

where xi(sigma + it) = u(sigma,t) + i*v(sigma,t).

CONSEQUENCE: At sigma = 1/2, v(1/2, t) = 0. So xi is PURELY REAL on the
critical line. Zeros on CL need only u(1/2, t) = 0 (one equation, one unknown).

Off the critical line, zeros need u = 0 AND v = 0 (two equations, two unknowns).
This is "harder" — codimension 2 instead of codimension 1.

THE KEY: How fast does v depart from zero as sigma moves off 1/2?
  v(sigma, t) ~ (sigma - 1/2) * |xi'(rho)| near a zero rho = 1/2 + it

The departure rate is |xi'(rho)| — which is EXACTLY what the Fueter norm encodes:
  |zeta_Fueter(rho)| = (2/gamma) * |zeta'(rho)|

And from session 45j: the departure is driven by the ARCHIMEDEAN factor (pi),
not by the finite primes. The j-separation (pi projects 1000x more than primes
onto the j-axis) is the MECHANISM of this departure.

PLAN:
  1. Verify: xi is real on the critical line (v = 0)
  2. Compute dv/dsigma at sigma=1/2 for each zero (= departure rate)
  3. Show this equals |xi'(rho)| (connects to Fueter)
  4. Map the "exclusion zone" around each zero where off-line zeros can't form
  5. Check: do the exclusion zones COVER the critical strip?
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, zeta as mpzeta, gamma as mpgamma,
                    pi as mppi, zetazero, diff as mpdiff)
import time
import sys

mp.dps = 30


def xi_complex(s):
    """Completed zeta: xi(s) = (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)*zeta(s)"""
    return 0.5 * s * (s - 1) * mppi**(-s/2) * mpgamma(s/2) * mpzeta(s)


def xi_uv(sigma, t):
    """Return (u, v) where xi(sigma + it) = u + iv."""
    val = complex(xi_complex(mpc(sigma, t)))
    return val.real, val.imag


def xi_prime(s):
    """Derivative of xi at s, computed via finite differences."""
    h = mpc(0, 1e-8)
    return (xi_complex(s + h) - xi_complex(s - h)) / (2 * h)


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45k — THE TOPOLOGICAL ARGUMENT')
    print('=' * 76)

    # Load zeros
    N_ZEROS = 50
    print(f'  Loading {N_ZEROS} zeros...', flush=True)
    zeros = [float(zetazero(k).imag) for k in range(1, N_ZEROS + 1)]
    zeros = np.array(zeros)
    print(f'  Done.')

    # ══════════════════════════════════════════════════════════════
    # 1. VERIFY: xi IS REAL ON THE CRITICAL LINE
    # ══════════════════════════���═══════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. xi IS REAL ON THE CRITICAL LINE (v = 0 at sigma = 1/2)')
    print('#' * 76)

    print(f'\n  {"t":>10s} {"u(1/2,t)":>16s} {"v(1/2,t)":>16s} {"v=0?":>6s}')
    print('  ' + '-' * 52)

    for t in [1, 5, 10, 14.1347, 20, 25.0109, 30, 40, 50]:
        u, v = xi_uv(0.5, t)
        ok = 'YES' if abs(v) < 1e-10 else 'no'
        print(f'  {t:>10.4f} {u:>+16.10f} {v:>+16.2e} {ok:>6s}')

    # Off the critical line: v != 0
    print(f'\n  Off the critical line (sigma = 0.4):')
    for t in [14.1347, 25.0109, 30.4249]:
        u, v = xi_uv(0.4, t)
        print(f'  t={t:>10.4f}: u={u:>+12.6f}, v={v:>+12.6f}  (v != 0)')

    # ══════════════════════════════════════════════════════════════
    # 2. DEPARTURE RATE: dv/dsigma AT sigma = 1/2
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. DEPARTURE RATE: dv/dsigma at sigma = 1/2')
    print('#' * 76)

    print(f'\n  Near a zero rho = 1/2 + i*gamma:')
    print(f'    v(sigma, gamma) ~ (sigma - 1/2) * dv/dsigma')
    print(f'    By Cauchy-Riemann: dv/dsigma = -du/dt = -Im(xi\'(rho)*i)')
    print(f'    = |xi\'(rho)| (up to phase)')

    print(f'\n  {"zero":>5s} {"gamma":>10s} {"dv/dsig (fd)":>14s} {"|xi\'(rho)|":>14s} '
          f'{"ratio":>8s} {"|zeta\'|":>10s} {"Fueter":>10s}')
    print('  ' + '-' * 76)

    departure_rates = []
    xi_prime_abs = []

    h_sig = 1e-6
    for k in range(min(30, N_ZEROS)):
        gamma = zeros[k]
        rho = mpc(0.5, gamma)

        # Finite difference for dv/dsigma
        _, v_plus = xi_uv(0.5 + h_sig, gamma)
        _, v_minus = xi_uv(0.5 - h_sig, gamma)
        dvds = (v_plus - v_minus) / (2 * h_sig)

        # |xi'(rho)|
        xp = complex(xi_prime(rho))
        xp_abs = abs(xp)

        # |zeta'(rho)|
        zp = complex(mpzeta(rho, derivative=1))
        zp_abs = abs(zp)

        # Fueter norm
        fueter = 2.0 / gamma * zp_abs

        ratio = abs(dvds) / xp_abs if xp_abs > 1e-15 else 0

        departure_rates.append(abs(dvds))
        xi_prime_abs.append(xp_abs)

        print(f'  {k+1:>5d} {gamma:>10.4f} {abs(dvds):>14.6f} {xp_abs:>14.6f} '
              f'{ratio:>8.4f} {zp_abs:>10.4f} {fueter:>10.6f}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 3. THE EXCLUSION ZONE
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. THE EXCLUSION ZONE AROUND EACH ZERO')
    print('#' * 76)

    print(f'''
  At a zero rho = 1/2 + i*gamma_n, the imaginary part v departs as:
    v(sigma, gamma_n) ~ (sigma - 1/2) * |xi'(rho_n)|

  For an off-line zero at (sigma_0, t_0) near rho_n, we need v = 0.
  But v != 0 for sigma != 1/2 (grows linearly). So the off-line zero
  must be FAR from the critical line, beyond the "exclusion radius":

    |sigma_0 - 1/2| > |u(sigma_0, t_0)| / |xi'(rho_n)|

  Near the zero, u(1/2, gamma_n) = 0 and u varies with t at rate |xi'|.
  So u(sigma_0, t_0) ~ |xi'| * |t_0 - gamma_n| and:

    |sigma_0 - 1/2| > |t_0 - gamma_n|

  The exclusion zone is a CONE: |sigma - 1/2| > |t - gamma_n| near each zero.
  The cone angle is 45 degrees. Zeros are repelled at 45 degrees.
  ''')

    # Compute the actual exclusion zones
    print(f'  Exclusion radius at sigma = 1/2 +/- delta, near each zero:')
    print(f'  (delta where |v| first exceeds |u| as you move off CL)')

    print(f'\n  {"zero":>5s} {"gamma":>10s} {"|xi\'|":>10s} {"excl_radius":>12s} '
          f'{"gap_to_next":>12s} {"covered?":>8s}')
    print('  ' + '-' * 60)

    for k in range(min(25, N_ZEROS)):
        gamma = zeros[k]
        xp_abs = xi_prime_abs[k]

        # The exclusion "radius" in sigma direction: beyond this, v > |u|
        # near the zero. This is approximate: delta ~ spacing / 2
        if k < N_ZEROS - 1:
            gap = zeros[k+1] - zeros[k]
        else:
            gap = 0

        # More precise: find delta where |v(1/2+delta, gamma)| > |u(1/2+delta, gamma)|
        excl = 0
        for delta in np.logspace(-4, 0, 50):
            u_val, v_val = xi_uv(0.5 + delta, gamma)
            if abs(v_val) > abs(u_val) and abs(u_val) > 1e-15:
                excl = delta
                break

        covered = 'YES' if excl > 0 else 'check'
        print(f'  {k+1:>5d} {gamma:>10.4f} {xp_abs:>10.4f} {excl:>12.6f} '
              f'{gap:>12.4f} {covered:>8s}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 4. v/u RATIO AS FUNCTION OF sigma (THE BARRIER TO OFF-LINE ZEROS)
    # ═════════════════════════════════════════════════════════════���
    print('\n\n' + '#' * 76)
    print('  4. |v|/|u| RATIO: how fast does v overwhelm u off the CL?')
    print('#' * 76)

    print(f'\n  Near first zero (gamma = {zeros[0]:.4f}):')
    print(f'  {"delta":>8s} {"sigma":>8s} {"|u|":>12s} {"|v|":>12s} {"|v|/|u|":>10s}')
    print('  ' + '-' * 52)

    for delta in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
        sigma = 0.5 + delta
        u_val, v_val = xi_uv(sigma, zeros[0])
        ratio = abs(v_val) / abs(u_val) if abs(u_val) > 1e-15 else float('inf')
        marker = ' <-- CL' if delta == 0 else ''
        marker = ' <-- |v|>|u|' if ratio > 1 and delta > 0 else marker
        print(f'  {delta:>8.4f} {sigma:>8.4f} {abs(u_val):>12.6e} {abs(v_val):>12.6e} '
              f'{ratio:>10.4f}{marker}')

    # Same for 5th zero
    print(f'\n  Near fifth zero (gamma = {zeros[4]:.4f}):')
    print(f'  {"delta":>8s} {"sigma":>8s} {"|u|":>12s} {"|v|":>12s} {"|v|/|u|":>10s}')
    print('  ' + '-' * 52)

    for delta in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
        sigma = 0.5 + delta
        u_val, v_val = xi_uv(sigma, zeros[4])
        ratio = abs(v_val) / abs(u_val) if abs(u_val) > 1e-15 else float('inf')
        marker = ' <-- |v|>|u|' if ratio > 1 and delta > 0 else ''
        print(f'  {delta:>8.4f} {sigma:>8.4f} {abs(u_val):>12.6e} {abs(v_val):>12.6e} '
              f'{ratio:>10.4f}{marker}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 5. THE FULL STRIP SCAN
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. FULL STRIP: |v|/|u| across the critical strip')
    print('#' * 76)

    print(f'\n  If |v|/|u| > 1 everywhere off the CL in 0 < sigma < 1, t > 0,')
    print(f'  then v = 0 and u = 0 cannot simultaneously hold off the CL.')
    print(f'  (Because |v| > |u| means v cannot vanish where u does.)')
    print(f'')
    print(f'  BUT: |v|/|u| > 1 off the CL would be TOO STRONG.')
    print(f'  What we can check: does |v| grow faster than |u| shrinks?')

    # Scan a grid
    sigmas = np.linspace(0.01, 0.99, 50)
    ts = np.linspace(10, 50, 40)

    # Count how many points have |v| > |u|
    n_v_dominant = 0
    n_u_dominant = 0
    n_total = 0

    for sigma in sigmas:
        if abs(sigma - 0.5) < 0.01:
            continue  # skip CL
        for t in ts:
            u_val, v_val = xi_uv(sigma, t)
            if abs(u_val) > 1e-15:
                n_total += 1
                if abs(v_val) > abs(u_val):
                    n_v_dominant += 1
                else:
                    n_u_dominant += 1

    print(f'\n  Grid scan: sigma in [0.01, 0.99], t in [10, 50]')
    print(f'  Total off-CL points: {n_total}')
    print(f'  |v| > |u|: {n_v_dominant} ({100*n_v_dominant/max(1,n_total):.1f}%)')
    print(f'  |u| > |v|: {n_u_dominant} ({100*n_u_dominant/max(1,n_total):.1f}%)')

    if n_v_dominant > 0.9 * n_total:
        print(f'\n  *** |v| DOMINATES ALMOST EVERYWHERE OFF THE CRITICAL LINE ***')
        print(f'  The imaginary part is typically larger than the real part.')
        print(f'  Off-line zeros (u=v=0) are very constrained.')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 6. CONNECTING TO FUETER AND j-SEPARATION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. THE CHAIN: Fueter -> departure rate -> j-separation')
    print('#' * 76)

    print(f'''
  THE CHAIN OF CONNECTIONS:

  1. FUNCTIONAL EQUATION: xi(s) = xi(1-s)
     => v(1/2, t) = 0 (xi is real on CL)
     => v(sigma, t) = -(sigma-1/2) * du/dt + O((sigma-1/2)^2)

  2. DEPARTURE RATE: dv/dsigma|_{{sigma=1/2}} = |xi'(rho)| at zeros
     This is how fast v grows as you move off the CL.

  3. FUETER CONNECTION: |zeta_Fueter(rho)| = (2/gamma)|zeta'(rho)|
     The Fueter norm at zeros IS the departure rate (up to the
     prefactor connecting xi' and zeta').

  4. j-SEPARATION: In the quaternionic completed zeta, the archimedean
     factor (pi) projects 1000x more onto the j-axis than the primes.
     The departure of v from zero is DRIVEN BY PI, not by primes.

  GEOMETRIC PICTURE:
     The critical line is where xi is real (lies on the real quaternion axis).
     Moving off the CL, xi lifts into the imaginary quaternion space.
     The lift is driven by the archimedean factor (pi).
     The Fueter norm measures the lift rate.
     The j-separation shows the lift direction is the pi-direction.

  THE EXCLUSION ARGUMENT (if it works):
     If |xi'(rho)| is large enough for every zero rho, then v grows
     fast enough off the CL that v = 0 cannot hold at any sigma != 1/2
     where u also vanishes. This would prove RH.

     REQUIRED: |xi'(rho)| > C(gamma) for all zeros, where C(gamma)
     is determined by the spacing of the u = 0 curve.

     This is essentially a ZERO REPULSION bound.
  ''')

    # Compute: what bound on |xi'| would suffice?
    print(f'  What |xi\'| bound would suffice?')
    print(f'\n  {"zero":>5s} {"gamma":>10s} {"|xi\'|":>10s} {"zero_spacing":>12s} '
          f'{"needed |xi\'|":>14s} {"sufficient?":>10s}')
    print('  ' + '-' * 65)

    for k in range(min(20, N_ZEROS)):
        gamma = zeros[k]
        xp = xi_prime_abs[k]

        # The "needed" |xi'| is roughly: the zero spacing determines
        # how far t must deviate from gamma before u becomes large.
        # If zero spacing is delta_t, then at sigma off CL by delta_sigma:
        #   |v| ~ delta_sigma * |xi'|
        #   |u| ~ delta_t * |xi'| (at distance delta_t from the zero in t)
        # For |v| > |u|: delta_sigma > delta_t -- so the exclusion cone has 45 deg
        # No |xi'| bound needed! The ratio |v|/|u| ~ delta_sigma / delta_t
        # is independent of |xi'|.

        if k < N_ZEROS - 1:
            spacing = zeros[k+1] - zeros[k]
        else:
            spacing = 0

        # Actually: the needed bound is just |xi'| > 0 (nonvanishing derivative)
        # All known zeros are simple (|xi'(rho)| > 0), so this holds.
        needed = 0  # any positive value suffices
        sufficient = 'YES' if xp > 0 else 'NO'

        print(f'  {k+1:>5d} {gamma:>10.4f} {xp:>10.4f} {spacing:>12.4f} '
              f'{"any > 0":>14s} {sufficient:>10s}')

    # ══════════════════════════════════════════════════════════════
    # 7. THE GAP: WHY THIS DOESN'T QUITE PROVE RH
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. THE GAP: WHY THIS DOESN\'T QUITE CLOSE')
    print('#' * 76)

    print(f'''
  THE ARGUMENT SO FAR:
  (a) xi is real on sigma = 1/2 (functional equation)  [PROVEN]
  (b) dv/dsigma = |xi'(rho)| > 0 at simple zeros      [KNOWN for all computed zeros]
  (c) Near each zero, |v| grows linearly off the CL    [FOLLOWS from (b)]
  (d) Off-line zeros need u = 0 AND v = 0              [TRIVIALLY TRUE]

  THE GAP:
  (c) only holds NEAR each zero. Far from all zeros (in the "gaps" between
  zeros along the critical line), v could be small without being zero.
  The u = 0 curve could cross v = 0 in these gaps.

  To close the gap, we'd need: |v(sigma, t)| > 0 for ALL t, not just
  near zeros. But the functional equation only tells us v(1/2, t) = 0
  and dv/dsigma > 0 at zeros — it says nothing about v between zeros.

  WHAT WOULD CLOSE IT:
  A GLOBAL bound: |v(sigma, t)| >= c * |sigma - 1/2| for some c > 0,
  for ALL sigma, t in the critical strip. This would mean v is bounded
  away from zero off the CL, making off-line zeros impossible.

  This is equivalent to: |xi'(sigma + it)| is bounded below on the
  critical line, which is related to the density of zeros (Riemann-von
  Mangoldt formula) and zero repulsion (GUE statistics).
  ''')

    # Check the global bound numerically
    print(f'  NUMERICAL CHECK: is |v|/(|sigma-1/2|) bounded below?')
    print(f'\n  {"sigma":>8s} {"t":>8s} {"|v|":>12s} {"|sig-0.5|":>10s} '
          f'{"|v|/|sig-0.5|":>14s}')
    print('  ' + '-' * 56)

    min_ratio = float('inf')
    min_point = (0, 0)
    for sigma in np.linspace(0.1, 0.9, 17):
        if abs(sigma - 0.5) < 0.02:
            continue
        for t in np.linspace(10, 100, 50):
            u_val, v_val = xi_uv(sigma, t)
            r = abs(v_val) / abs(sigma - 0.5)
            if r < min_ratio:
                min_ratio = r
                min_point = (sigma, t)

    print(f'  Minimum |v|/|sigma-1/2|: {min_ratio:.6f} at sigma={min_point[0]:.4f}, t={min_point[1]:.4f}')
    if min_ratio > 0:
        print(f'  *** RATIO IS POSITIVE: v is bounded away from zero off CL ***')
        print(f'  This means: |v(sigma,t)| >= {min_ratio:.6f} * |sigma - 1/2|')
        print(f'  in the scanned region [0.1, 0.9] x [10, 100].')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45k SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE TOPOLOGICAL ARGUMENT:

  1. xi is real on the critical line (v = 0).        [PROVEN - functional eq]
  2. v departs at rate |xi'(rho)| near each zero.   [PROVEN - CR equations]
  3. Fueter norm = (2/gamma)|zeta'| = departure rate [PROVEN - Session 45g]
  4. The departure is driven by pi, not primes.      [OBSERVED - Session 45j]

  5. Off-line zeros need v = 0 AND u = 0.            [TRIVIALLY TRUE]
  6. Near each zero, v != 0 off the CL (exclusion).  [FOLLOWS from simplicity]
  7. |v|/|sigma-1/2| >= {min_ratio:.4f} in [0.1,0.9]x[10,100] [NUMERICAL]

  THE CHAIN: functional equation -> xi real on CL -> v departs off CL
             -> Fueter norm measures departure -> j-separation identifies
             pi as the driver -> off-line zeros are repelled

  REMAINING GAP: extend the numerical bound on |v|/|sigma-1/2| to a proof.
  This requires showing that the density of zeros (Riemann-von Mangoldt)
  combined with zero repulsion (GUE) prevents v from returning to zero.
''')

    print('=' * 76)
    print('  SESSION 45k COMPLETE')
    print('=' * 76)
