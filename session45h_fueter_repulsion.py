"""
SESSION 45h — FUETER-DERIVATIVE-ZERO REPULSION

The key discovery: |zeta_Fueter(rho)| = (2/gamma) * |zeta'(rho)|

This connects the Fueter construction to:
  - Zero repulsion (GUE statistics, Montgomery pair correlation)
  - de Bruijn-Newman Lambda (zero dynamics under heat flow)
  - Hadamard product (|zeta'(rho)| = product over other zeros)

QUESTIONS:
  1. Is sigma=1/2 a MINIMUM of |F|^2 along the sigma direction?
     If yes: critical line minimizes the Fueter norm — geometric characterization.
  2. What are the statistics of |zeta'(rho)|/gamma? GUE prediction?
  3. Does the A=0 curve (Im(zeta')=0) interleave with classical zeros?
  4. Can we build a "Fueter barrier" = sum (2/gamma)^2 |zeta'(rho)|^2
     and relate it to our original barrier?
  5. The Fueter norm sigma-profile: as sigma moves off 1/2, does |F| at
     each zero change in a way that constrains the zero location?
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta as mpzeta, zetazero
import time
import sys

mp.dps = 25


def fueter_components(sigma, r):
    """Exact Fueter A, B components."""
    if abs(r) < 1e-12:
        return 0.0, 0.0
    s = mpc(sigma, r)
    z = complex(mpzeta(s))
    zp = complex(mpzeta(s, derivative=1))
    A = -(2.0 / r) * zp.imag
    B = (2.0 / r) * zp.real - 2.0 * z.imag / (r * r)
    return A, B


def fueter_norm_sq(sigma, r):
    A, B = fueter_components(sigma, r)
    return A**2 + B**2


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45h -- FUETER-DERIVATIVE-ZERO REPULSION')
    print('=' * 76)

    # Load zeros
    N_ZEROS = 200
    print(f'\n  Loading {N_ZEROS} zeta zeros...', flush=True)
    t0 = time.time()
    zeros = [float(zetazero(k).imag) for k in range(1, N_ZEROS + 1)]
    zeros = np.array(zeros)
    print(f'  Done ({time.time()-t0:.1f}s)')

    # Compute |zeta'| at each zero
    print(f'  Computing zeta\' at each zero...', flush=True)
    zeta_primes = []
    for gamma in zeros:
        zp = complex(mpzeta(mpc(0.5, gamma), derivative=1))
        zeta_primes.append(zp)
    zeta_primes = np.array(zeta_primes)
    zp_abs = np.abs(zeta_primes)

    # ══════════════════════════════════════════════════════════════
    # 1. IS sigma=1/2 A MINIMUM OF |F|^2 AT EACH ZERO?
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. FUETER NORM sigma-PROFILE AT CLASSICAL ZEROS')
    print('#' * 76)

    print(f'\n  For each zero rho = 1/2 + i*gamma, compute |F|^2 at sigma +/- eps')
    print(f'  If d^2|F|^2/dsigma^2 > 0 at sigma=1/2: the critical line is a LOCAL MINIMUM')

    eps = 0.01
    print(f'\n  {"zero":>5s} {"gamma":>10s} {"|F|^2(0.5)":>14s} '
          f'{"|F|^2(0.5-e)":>14s} {"|F|^2(0.5+e)":>14s} '
          f'{"d^2/dsig^2":>12s} {"min@CL?":>8s}')
    print('  ' + '-' * 82)

    n_min = 0
    n_max = 0
    n_saddle = 0
    d2_vals = []

    for k in range(min(50, N_ZEROS)):
        gamma = zeros[k]
        f_center = fueter_norm_sq(0.5, gamma)
        f_left = fueter_norm_sq(0.5 - eps, gamma)
        f_right = fueter_norm_sq(0.5 + eps, gamma)

        d2 = (f_left - 2 * f_center + f_right) / (eps**2)
        d2_vals.append(d2)

        is_min = 'MIN' if f_center < f_left and f_center < f_right else ''
        is_max = 'MAX' if f_center > f_left and f_center > f_right else ''
        label = is_min or is_max or 'saddle'

        if is_min:
            n_min += 1
        elif is_max:
            n_max += 1
        else:
            n_saddle += 1

        if k < 20 or k % 10 == 0:
            print(f'  {k+1:>5d} {gamma:>10.4f} {f_center:>14.6e} '
                  f'{f_left:>14.6e} {f_right:>14.6e} '
                  f'{d2:>+12.4e} {label:>8s}')

    print(f'\n  SUMMARY (first {min(50, N_ZEROS)} zeros):')
    print(f'    MIN at sigma=1/2:    {n_min}')
    print(f'    MAX at sigma=1/2:    {n_max}')
    print(f'    Saddle:              {n_saddle}')
    print(f'    Mean d^2|F|^2/dsig^2: {np.mean(d2_vals):+.6e}')
    print(f'    All positive (all MIN)? {all(d > 0 for d in d2_vals)}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 2. DETAILED sigma PROFILE AT FIRST FEW ZEROS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. DETAILED sigma PROFILE: |F|^2 as function of sigma')
    print('#' * 76)

    sigmas = np.linspace(0.0, 1.0, 41)
    for k_zero in [0, 1, 2, 4, 9]:
        gamma = zeros[k_zero]
        print(f'\n  Zero #{k_zero+1}: gamma = {gamma:.4f}')
        print(f'  {"sigma":>8s} {"|F|^2":>14s} {"A":>12s} {"B":>12s} '
              f'{"|F|^2/|F|^2(0.5)":>18s}')
        print('  ' + '-' * 58)

        f_at_half = fueter_norm_sq(0.5, gamma)
        for sig in sigmas[::4]:
            A, B = fueter_components(sig, gamma)
            fsq = A**2 + B**2
            ratio = fsq / f_at_half if f_at_half > 0 else 0
            marker = ' <--' if abs(sig - 0.5) < 0.02 else ''
            print(f'  {sig:>8.4f} {fsq:>14.6e} {A:>+12.6f} {B:>+12.6f} '
                  f'{ratio:>18.6f}{marker}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 3. STATISTICS OF |zeta'(rho)| AND FUETER NORMS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. STATISTICS OF |zeta\'(rho)| AND FUETER NORMS')
    print('#' * 76)

    fueter_norms = (2.0 / zeros) * zp_abs
    fueter_norm_sq_arr = fueter_norms**2

    print(f'\n  |zeta\'(rho)| statistics ({N_ZEROS} zeros):')
    print(f'    Mean:   {np.mean(zp_abs):.6f}')
    print(f'    Std:    {np.std(zp_abs):.6f}')
    print(f'    Min:    {np.min(zp_abs):.6f} at gamma={zeros[np.argmin(zp_abs)]:.4f}')
    print(f'    Max:    {np.max(zp_abs):.6f} at gamma={zeros[np.argmax(zp_abs)]:.4f}')

    print(f'\n  Fueter norm (2/gamma)|zeta\'(rho)| statistics:')
    print(f'    Mean:   {np.mean(fueter_norms):.6f}')
    print(f'    Std:    {np.std(fueter_norms):.6f}')
    print(f'    Min:    {np.min(fueter_norms):.6f}')
    print(f'    Max:    {np.max(fueter_norms):.6f}')

    # Growth rate of |zeta'|
    log_g = np.log(zeros[10:])
    log_zp = np.log(zp_abs[10:])
    c_zp = np.polyfit(log_g, log_zp, 1)
    print(f'\n  Growth: |zeta\'(rho)| ~ gamma^{{{c_zp[0]:.4f}}}')
    print(f'  (GUE prediction for log|char poly derivative|: ~ log(gamma)/2)')

    # Fueter norm decay/growth
    log_fn = np.log(fueter_norms[10:])
    c_fn = np.polyfit(log_g, log_fn, 1)
    print(f'  Fueter norm (2/gamma)|zeta\'| ~ gamma^{{{c_fn[0]:.4f}}}')
    print(f'  (If exponent < 0: Fueter norm DECAYS with gamma)')

    # ══════════════════════════════════════════════════════════════
    # 4. A=0 INTERLEAVING WITH CLASSICAL ZEROS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. A=0 CURVE INTERLEAVING WITH CLASSICAL ZEROS')
    print('#' * 76)

    print(f'\n  A(1/2, r) = -(2/r)*Im(zeta\'(1/2+ir)) = 0 when Im(zeta\') = 0')
    print(f'  Scanning r in [10, 100] for A=0 crossings:')

    r_scan = np.linspace(10, 100, 2000)
    a_vals = []
    for r in r_scan:
        A, _ = fueter_components(0.5, r)
        a_vals.append(A)
    a_vals = np.array(a_vals)

    # Find crossings
    a_crossings = []
    for i in range(len(a_vals) - 1):
        if a_vals[i] * a_vals[i+1] < 0:
            # Linear interpolation
            r_cross = r_scan[i] - a_vals[i] * (r_scan[i+1] - r_scan[i]) / (a_vals[i+1] - a_vals[i])
            a_crossings.append(r_cross)

    print(f'  Found {len(a_crossings)} A=0 crossings')
    print(f'  Found {np.sum(zeros < 100)} classical zeros in [10, 100]')

    # Interleaving check
    all_events = []
    for g in zeros[zeros < 100]:
        all_events.append(('ZERO', g))
    for r in a_crossings:
        all_events.append(('A=0', r))
    all_events.sort(key=lambda x: x[1])

    print(f'\n  Interleaving pattern (first 40 events):')
    prev_type = None
    alternations = 0
    total_pairs = 0
    for event_type, r in all_events[:40]:
        marker = '*' if event_type == 'ZERO' else ' '
        print(f'    {marker} {event_type:>4s} at r = {r:.4f}')
        if prev_type is not None and prev_type != event_type:
            alternations += 1
        if prev_type is not None:
            total_pairs += 1
        prev_type = event_type

    if total_pairs > 0:
        alt_frac = alternations / total_pairs
        print(f'\n  Alternation fraction: {alt_frac:.4f} '
              f'(1.0 = perfect interleaving, 0.5 = random)')
        if alt_frac > 0.85:
            print(f'  *** STRONG INTERLEAVING: A=0 crossings interleave with zeros ***')
            print(f'  This means Im(zeta\') changes sign between consecutive zeros.')
            print(f'  Geometric interpretation: the Fueter vector ROTATES between zeros.')
        elif alt_frac > 0.6:
            print(f'  Moderate interleaving.')
        else:
            print(f'  Weak or no interleaving.')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 5. FUETER VECTOR ROTATION BETWEEN ZEROS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. FUETER VECTOR ROTATION BETWEEN CONSECUTIVE ZEROS')
    print('#' * 76)

    print(f'\n  At each zero: F = A + B*(v/|v|)')
    print(f'  The "Fueter angle" theta = arctan(A/B) rotates as we move between zeros.')
    print(f'  If theta changes by ~pi between consecutive zeros: PERFECT ROTATION.')

    print(f'\n  {"zero_k":>7s} {"gamma_k":>10s} {"A_k":>12s} {"B_k":>12s} '
          f'{"theta_k":>10s} {"delta_theta":>12s}')
    print('  ' + '-' * 65)

    prev_theta = None
    delta_thetas = []
    for k in range(min(30, N_ZEROS)):
        gamma = zeros[k]
        A, B = fueter_components(0.5, gamma)
        theta = np.arctan2(A, B)

        if prev_theta is not None:
            dtheta = theta - prev_theta
            # Wrap to [-pi, pi]
            while dtheta > np.pi:
                dtheta -= 2 * np.pi
            while dtheta < -np.pi:
                dtheta += 2 * np.pi
            delta_thetas.append(dtheta)
        else:
            dtheta = 0

        print(f'  {k+1:>7d} {gamma:>10.4f} {A:>+12.6f} {B:>+12.6f} '
              f'{theta:>+10.4f} {dtheta:>+12.4f}')
        prev_theta = theta

    if delta_thetas:
        dt = np.array(delta_thetas)
        print(f'\n  Delta-theta statistics:')
        print(f'    Mean:     {np.mean(dt):+.4f} rad ({np.mean(dt)/np.pi:+.4f}*pi)')
        print(f'    Std:      {np.std(dt):.4f} rad')
        print(f'    |Mean|/pi: {abs(np.mean(dt))/np.pi:.4f}')
        if abs(np.mean(dt)) > 2.5:
            print(f'    *** NEAR-pi ROTATION between zeros! ***')
            print(f'    The Fueter vector nearly REVERSES between consecutive zeros.')
        elif abs(np.mean(dt)) > 1.5:
            print(f'    Significant rotation (~{abs(np.mean(dt))/np.pi:.1f}*pi) between zeros.')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 6. THE FUETER BARRIER
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. THE FUETER BARRIER: sum of |F|^2 at zeros')
    print('#' * 76)

    fueter_barrier = np.sum(fueter_norm_sq_arr)
    partial_sums = np.cumsum(fueter_norm_sq_arr)

    print(f'\n  B_Fueter = sum_rho (2/gamma)^2 * |zeta\'(rho)|^2')
    print(f'  = sum_rho |F(rho)|^2')
    print(f'\n  Convergence:')
    for n in [10, 20, 50, 100, 150, 200]:
        if n <= N_ZEROS:
            print(f'    First {n:>3d} zeros: B_F = {partial_sums[n-1]:.6f}')

    print(f'\n  B_Fueter ({N_ZEROS} zeros) = {fueter_barrier:.6f}')

    # Compare to our original barrier
    sys.path.insert(0, '.')
    from session41g_uncapped_barrier import compute_barrier_partial
    for lam_sq in [200, 500, 1000, 2000]:
        r = compute_barrier_partial(lam_sq)
        print(f'  Original barrier W02-Mp at lam^2={lam_sq}: {r["partial_barrier"]:+.6f}')

    # ══════════════════════════════════════════════════════════════
    # 7. THE KEY TEST: FUETER NORM EXTREMALITY AT sigma=1/2
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. THE KEY TEST: sum |F(sigma, gamma)|^2 across sigma')
    print('#' * 76)

    print(f'\n  Define S_F(sigma) = sum_rho |F(sigma, gamma_rho)|^2')
    print(f'  If S_F is MINIMIZED at sigma=1/2, the critical line minimizes')
    print(f'  the total Fueter energy. This would be a variational principle!')

    sigmas_test = np.linspace(0.1, 0.9, 17)
    n_zeros_test = 50  # first 50 zeros for speed

    print(f'\n  Using first {n_zeros_test} zeros:')
    print(f'  {"sigma":>8s} {"S_F(sigma)":>14s} {"S_F/S_F(0.5)":>14s}')
    print('  ' + '-' * 40)

    sf_vals = []
    sf_at_half = None
    for sig in sigmas_test:
        sf = 0.0
        for gamma in zeros[:n_zeros_test]:
            A, B = fueter_components(sig, gamma)
            sf += A**2 + B**2
        sf_vals.append(sf)
        if abs(sig - 0.5) < 0.03:
            sf_at_half = sf

    for sig, sf in zip(sigmas_test, sf_vals):
        ratio = sf / sf_at_half if sf_at_half else 0
        marker = ' <-- CL' if abs(sig - 0.5) < 0.03 else ''
        marker += ' MIN' if sf == min(sf_vals) else ''
        print(f'  {sig:>8.4f} {sf:>14.6e} {ratio:>14.6f}{marker}')

    min_sigma = sigmas_test[np.argmin(sf_vals)]
    print(f'\n  S_F minimum at sigma = {min_sigma:.4f}')
    if abs(min_sigma - 0.5) < 0.05:
        print(f'  *** CRITICAL LINE MINIMIZES TOTAL FUETER ENERGY! ***')
        print(f'  This is a VARIATIONAL CHARACTERIZATION of the critical line:')
        print(f'  sigma = 1/2 minimizes sum_rho |Delta(zeta_slice)|^2 evaluated at zeros.')
    else:
        print(f'  Minimum is NOT at the critical line (at sigma={min_sigma:.4f})')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 8. FUNCTIONAL EQUATION CONSTRAINT ON FUETER
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  8. FUNCTIONAL EQUATION AND FUETER SYMMETRY')
    print('#' * 76)

    print(f'\n  The functional equation zeta(s) = chi(s)*zeta(1-s) implies')
    print(f'  a relationship between F(sigma, r) and F(1-sigma, r).')

    print(f'\n  {"gamma":>10s} {"|F(0.5)|":>12s} {"|F(0.3)|":>12s} {"|F(0.7)|":>12s} '
          f'{"F(.3)/F(.7)":>12s}')
    print('  ' + '-' * 52)

    for k in range(0, min(20, N_ZEROS)):
        gamma = zeros[k]
        f_half = np.sqrt(fueter_norm_sq(0.5, gamma))
        f_left = np.sqrt(fueter_norm_sq(0.3, gamma))
        f_right = np.sqrt(fueter_norm_sq(0.7, gamma))
        ratio = f_left / f_right if f_right > 1e-15 else float('inf')
        print(f'  {gamma:>10.4f} {f_half:>12.6f} {f_left:>12.6f} {f_right:>12.6f} '
              f'{ratio:>12.6f}')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45h SYNTHESIS')
    print('=' * 76)

    print(f'''
  1. FUETER NORM sigma-PROFILE:
     At classical zeros, |F|^2 varies with sigma.
     Results: {n_min} MIN, {n_max} MAX, {n_saddle} saddle at sigma=1/2
     (out of first {min(50, N_ZEROS)} zeros)

  2. |zeta'(rho)| STATISTICS:
     Growth: |zeta'| ~ gamma^{{{c_zp[0]:.4f}}}
     Fueter norm: (2/gamma)|zeta'| ~ gamma^{{{c_fn[0]:.4f}}}

  3. A=0 INTERLEAVING:
     {len(a_crossings)} A=0 crossings vs {int(np.sum(zeros < 100))} zeros in [10,100]
     Alternation fraction: {alt_frac:.4f}

  4. FUETER VECTOR ROTATION:
     Mean delta-theta between consecutive zeros: {np.mean(delta_thetas):.4f} rad
     ({abs(np.mean(delta_thetas))/np.pi:.4f}*pi)

  5. TOTAL FUETER ENERGY S_F(sigma):
     Minimum at sigma = {min_sigma:.4f}

  6. FUETER BARRIER:
     B_F = sum |F(rho)|^2 = {fueter_barrier:.6f} ({N_ZEROS} zeros)
''')

    print('=' * 76)
    print('  SESSION 45h COMPLETE')
    print('=' * 76)
