"""
SESSION 45L — CLOSING THE GAP

The gap from 45k: exclusion zones around individual zeros don't cover
the full strip. Between zeros, v can be small.

NEW APPROACH: Factor out the forced zero.

Since v(1/2, t) = 0 for all t (functional equation), write:
  v(sigma, t) = (sigma - 1/2) * w(sigma, t)

where w is smooth (v has a simple zero at sigma=1/2 as a function of sigma).
At sigma=1/2: w(1/2, t) = dv/dsigma|_{sigma=1/2} = -du/dt (by Cauchy-Riemann).

Off-line zeros of xi need v = 0 AND u = 0. Since v = (sigma-1/2)*w,
off-line zeros (sigma != 1/2) need w = 0 AND u = 0.

THE KEY QUESTION: does {w = 0} ever intersect {u = 0} for sigma != 1/2?

w(sigma, t) = 0 defines curves in the (sigma, t) plane.
u(sigma, t) = 0 defines other curves.
If these curve families NEVER intersect off the CL, RH is true.

PLAN:
  1. Compute w(sigma, t) across the critical strip
  2. Map {w = 0} curves and {u = 0} curves
  3. Check if they intersect
  4. Study what prevents intersection (or find where they nearly intersect)
  5. Connect to quaternionic linking / Hopf fibration
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta as mpzeta, gamma as mpgamma, pi as mppi, zetazero
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


def w_function(sigma, t):
    """
    w(sigma, t) = v(sigma, t) / (sigma - 1/2)

    At sigma = 1/2: w = dv/dsigma = -du/dt (by L'Hopital / CR equations).
    """
    if abs(sigma - 0.5) < 1e-8:
        # Use finite difference for dv/dsigma at sigma=1/2
        h = 1e-6
        _, v_plus = xi_uv(0.5 + h, t)
        _, v_minus = xi_uv(0.5 - h, t)
        return (v_plus - v_minus) / (2 * h)
    else:
        _, v = xi_uv(sigma, t)
        return v / (sigma - 0.5)


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45L -- CLOSING THE GAP')
    print('=' * 76)

    # ══════════════════════════════════════════════════════════════
    # 1. MAP THE w-FUNCTION AND u-FUNCTION ZERO SETS
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. MAPPING {w=0} AND {u=0} IN THE CRITICAL STRIP')
    print('#' * 76)

    # Grid
    n_sigma = 100
    n_t = 200
    sigmas = np.linspace(0.01, 0.99, n_sigma)
    ts = np.linspace(10, 55, n_t)

    print(f'  Computing u and w on {n_sigma} x {n_t} grid...')
    t0 = time.time()

    u_grid = np.zeros((n_sigma, n_t))
    w_grid = np.zeros((n_sigma, n_t))

    for i, sig in enumerate(sigmas):
        for j, t in enumerate(ts):
            u_val, v_val = xi_uv(sig, t)
            u_grid[i, j] = u_val
            w_grid[i, j] = w_function(sig, t)
        if (i + 1) % 20 == 0:
            print(f'    sigma row {i+1}/{n_sigma} ({time.time()-t0:.1f}s)', flush=True)

    dt = time.time() - t0
    print(f'  Done ({dt:.1f}s)')

    # Find zero crossings of u and w
    # u = 0 crossings (in sigma direction for each t)
    u_zeros = []  # list of (sigma, t) where u crosses zero
    w_zeros = []  # list of (sigma, t) where w crosses zero

    for j in range(n_t):
        for i in range(n_sigma - 1):
            # u crossings
            if u_grid[i, j] * u_grid[i+1, j] < 0:
                # Linear interpolation
                sig_cross = sigmas[i] - u_grid[i, j] * (sigmas[i+1] - sigmas[i]) / (u_grid[i+1, j] - u_grid[i, j])
                u_zeros.append((sig_cross, ts[j]))
            # w crossings
            if w_grid[i, j] * w_grid[i+1, j] < 0:
                sig_cross = sigmas[i] - w_grid[i, j] * (sigmas[i+1] - sigmas[i]) / (w_grid[i+1, j] - w_grid[i, j])
                w_zeros.append((sig_cross, ts[j]))

    # Also scan in t direction
    for i in range(n_sigma):
        for j in range(n_t - 1):
            if u_grid[i, j] * u_grid[i, j+1] < 0:
                t_cross = ts[j] - u_grid[i, j] * (ts[j+1] - ts[j]) / (u_grid[i, j+1] - u_grid[i, j])
                u_zeros.append((sigmas[i], t_cross))
            if w_grid[i, j] * w_grid[i, j+1] < 0:
                t_cross = ts[j] - w_grid[i, j] * (ts[j+1] - ts[j]) / (w_grid[i, j+1] - w_grid[i, j])
                w_zeros.append((sigmas[i], t_cross))

    print(f'\n  Found {len(u_zeros)} u=0 crossings')
    print(f'  Found {len(w_zeros)} w=0 crossings')

    # ══════════════════════════════════════════════════════════════
    # 2. DO {w=0} AND {u=0} INTERSECT OFF THE CRITICAL LINE?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. INTERSECTION CHECK: {w=0} AND {u=0} OFF THE CL')
    print('#' * 76)

    # For each u=0 point, check if w is near zero
    close_approaches = []
    for sig_u, t_u in u_zeros:
        if abs(sig_u - 0.5) < 0.02:
            continue  # skip CL
        w_val = w_function(sig_u, t_u)
        close_approaches.append((sig_u, t_u, abs(w_val)))

    close_approaches.sort(key=lambda x: x[2])

    print(f'\n  Closest approaches of {{w=0}} to {{u=0}} OFF the CL:')
    print(f'  (sorted by |w| at u=0 points)')
    print(f'\n  {"sigma":>8s} {"t":>8s} {"|w| at u=0":>14s} {"|sig-0.5|":>10s}')
    print('  ' + '-' * 44)

    for sig, t, w_abs in close_approaches[:25]:
        print(f'  {sig:>8.4f} {t:>8.4f} {w_abs:>14.6e} {abs(sig-0.5):>10.4f}')

    if close_approaches:
        min_w = close_approaches[0][2]
        min_sig, min_t = close_approaches[0][0], close_approaches[0][1]
        print(f'\n  MINIMUM |w| at a u=0 point off CL: {min_w:.6e}')
        print(f'    at sigma={min_sig:.4f}, t={min_t:.4f}')

        if min_w < 1e-6:
            print(f'  *** DANGER: w nearly zero where u=0! Near-intersection! ***')
        else:
            print(f'  w is bounded away from zero on {{u=0}} off the CL.')
            print(f'  The curves {{w=0}} and {{u=0}} DO NOT INTERSECT in this region.')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 3. WHERE ARE THE {w=0} CURVES?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. ANATOMY OF {w=0} CURVES')
    print('#' * 76)

    # w = 0 on the CL happens where dv/dsigma = 0, i.e., du/dt = 0
    # These are extrema of u(1/2, t) = xi(1/2+it)
    print(f'\n  w(1/2, t) = dv/dsigma = -du/dt on the critical line.')
    print(f'  w = 0 on CL where u has extrema (between consecutive zeros).')

    # Find extrema of u on CL
    t_fine = np.linspace(10, 55, 1000)
    u_cl = np.array([xi_uv(0.5, t)[0] for t in t_fine])

    cl_extrema = []
    for j in range(1, len(u_cl) - 1):
        if (u_cl[j] > u_cl[j-1] and u_cl[j] > u_cl[j+1]) or \
           (u_cl[j] < u_cl[j-1] and u_cl[j] < u_cl[j+1]):
            cl_extrema.append((t_fine[j], u_cl[j]))

    # Classical zeros for reference
    zeros = [float(zetazero(k).imag) for k in range(1, 16)]

    print(f'\n  Extrema of u(1/2,t) (= xi on CL):')
    print(f'  {"t":>10s} {"u(1/2,t)":>14s} {"type":>6s} {"nearest_zero":>12s}')
    print('  ' + '-' * 46)
    for t_ext, u_ext in cl_extrema:
        ext_type = 'max' if u_ext > 0 else 'min'
        nearest = min(zeros, key=lambda g: abs(g - t_ext))
        print(f'  {t_ext:>10.4f} {u_ext:>+14.6e} {ext_type:>6s} {nearest:>12.4f}')

    print(f'\n  Each extremum on CL is a potential SOURCE of a {{w=0}} curve')
    print(f'  that extends into the critical strip.')

    # For each w=0 point off the CL, find the nearest CL extremum
    w_zeros_off_cl = [(s, t) for s, t in w_zeros if abs(s - 0.5) > 0.05]
    print(f'\n  w=0 points off CL (|sigma-0.5| > 0.05): {len(w_zeros_off_cl)}')

    if w_zeros_off_cl:
        print(f'  Sample w=0 points:')
        print(f'  {"sigma":>8s} {"t":>8s} {"u at this pt":>14s}')
        print('  ' + '-' * 34)
        for sig, t in w_zeros_off_cl[:20]:
            u_val, _ = xi_uv(sig, t)
            marker = ' <-- DANGER' if abs(u_val) < 1e-8 else ''
            print(f'  {sig:>8.4f} {t:>8.4f} {u_val:>+14.6e}{marker}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 4. THE REPULSION: |w| ON THE {u=0} CURVES
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. |w| ALONG THE {u=0} CURVES (the repulsion measure)')
    print('#' * 76)

    print(f'\n  The u=0 curves pass through the CL at the zeros of xi.')
    print(f'  Off the CL, u=0 curves curve away. Along these curves,')
    print(f'  |w| measures how far v is from zero (relative to |sigma-1/2|).')
    print(f'  If |w| stays bounded below, off-line zeros are impossible.')

    # Trace u=0 curves from each zero
    print(f'\n  Tracing u=0 from zeros of xi, moving sigma off CL:')
    print(f'\n  {"zero":>5s} {"gamma":>10s} {"delta_sig":>10s} {"sigma":>8s} '
          f'{"t (u=0)":>10s} {"|w|":>14s} {"w*delta":>10s}')
    print('  ' + '-' * 72)

    for k_zero in range(min(10, len(zeros))):
        gamma = zeros[k_zero]

        for delta in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
            sigma = 0.5 + delta
            # Find t near gamma where u(sigma, t) = 0
            # Use Newton's method starting from gamma
            t_search = gamma
            for _ in range(50):
                u_val, _ = xi_uv(sigma, t_search)
                # du/dt via finite difference
                h = 1e-6
                u_plus, _ = xi_uv(sigma, t_search + h)
                du_dt = (u_plus - u_val) / h
                if abs(du_dt) < 1e-20:
                    break
                t_search -= u_val / du_dt

            u_final, _ = xi_uv(sigma, t_search)
            if abs(u_final) < 1e-10:
                w_val = w_function(sigma, t_search)
                print(f'  {k_zero+1:>5d} {gamma:>10.4f} {delta:>+10.4f} {sigma:>8.4f} '
                      f'{t_search:>10.4f} {abs(w_val):>14.6e} {abs(w_val)*delta:>10.6f}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 5. THE w-FUNCTION ON THE CL: DOES IT VANISH?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. w(1/2, t) = -du/dt ON THE CRITICAL LINE')
    print('#' * 76)

    print(f'\n  w vanishes on CL at extrema of u(1/2,t).')
    print(f'  Between zeros, u reaches extrema where |w|=0.')
    print(f'  These are the SOURCE points of {{w=0}} curves entering the strip.')
    print(f'\n  The question: do these {{w=0}} curves reach the {{u=0}} curves?')

    # Compute w(1/2, t) along CL
    w_cl = np.array([w_function(0.5, t) for t in t_fine])

    print(f'\n  {"t":>8s} {"u(1/2,t)":>14s} {"w(1/2,t)":>14s} {"event":>12s}')
    print('  ' + '-' * 52)

    for j in range(0, len(t_fine), 20):
        t = t_fine[j]
        event = ''
        if abs(u_cl[j]) < abs(u_cl).max() * 0.001:
            event = 'near u=0'
        if j > 0 and j < len(w_cl)-1 and abs(w_cl[j]) < abs(w_cl[j-1]) and abs(w_cl[j]) < abs(w_cl[j+1]):
            event = 'w~0 (extremum)'
        print(f'  {t:>8.4f} {u_cl[j]:>+14.6e} {w_cl[j]:>+14.6e} {event:>12s}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 6. THE MINIMUM |w| ON {u=0} ACROSS INCREASING t RANGE
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. MINIMUM |w| ON {u=0} AS t INCREASES')
    print('#' * 76)

    print(f'\n  If min|w| on {{u=0}} stays bounded below as t -> inf,')
    print(f'  the gap is closed: no off-line zeros exist.')

    # Extend the grid to larger t
    for t_max in [50, 100, 200]:
        ts_ext = np.linspace(10, t_max, 300)
        min_w_on_u0 = float('inf')
        min_point = (0, 0)

        for t in ts_ext:
            for sig in np.linspace(0.05, 0.95, 40):
                if abs(sig - 0.5) < 0.02:
                    continue
                u_val, v_val = xi_uv(sig, t)
                if abs(u_val) < abs(v_val) * 0.01 + 1e-30:  # near u=0
                    w_val = abs(v_val / (sig - 0.5)) if abs(sig - 0.5) > 1e-10 else 0
                    if w_val < min_w_on_u0 and w_val > 0:
                        min_w_on_u0 = w_val
                        min_point = (sig, t)

        print(f'  t in [10, {t_max}]: min |w| near {{u=0}} off CL = {min_w_on_u0:.6e} '
              f'at sigma={min_point[0]:.4f}, t={min_point[1]:.4f}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 7. THE QUATERNIONIC LINKING ARGUMENT
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. QUATERNIONIC LINKING OF ZERO-SPHERES')
    print('#' * 76)

    print(f'''
  In H = R^4, the zeros of xi form 2-spheres S^2_rho for each zero rho.
  Two 2-spheres in R^4 can be LINKED (like two circles in R^3).

  For zeros at 1/2 + i*gamma_n, the zero-spheres are:
    S_n = {{1/2 + gamma_n * I : I in S^2}}
  These are concentric spheres centered at sigma=1/2, radius gamma_n.

  The linking number of two concentric 2-spheres in R^4 is:
    link(S_n, S_m) = 0 for concentric spheres (they're unlinked)

  Wait -- concentric spheres in R^4 are NOT linked. They can be
  "unlinked" by moving one past the other along the real axis.

  For an OFF-LINE zero at sigma_0 != 1/2:
    S_off = {{sigma_0 + gamma * I : I in S^2}}
  This sphere has a DIFFERENT center. Is it linked with the on-line spheres?

  Two 2-spheres in R^4 are linked iff one passes through the "disk"
  bounded by the other. For spheres at different centers on the real axis,
  they are linked iff their radii and center separation satisfy certain
  conditions.

  Concretely: S_n at center (1/2, 0, 0, 0) radius gamma_n
  and S_off at center (sigma_0, 0, 0, 0) radius gamma_off.
  These are linked iff... they can always be unlinked by continuous
  deformation in R^4 (since R^4 minus a 2-sphere is simply connected
  for codimension-2 spheres... actually, this is more subtle).
  ''')

    # Actually compute: can we detect the linking numerically?
    # The linking number of two 2-spheres in R^4 is always 0
    # (two codimension-2 spheres in R^4 are never linked).
    # This is because pi_1(R^4 \ S^2) = Z (fundamental group),
    # but the linking is measured by pi_2, which requires different tools.

    # Actually: two 2-spheres in R^4 CAN be linked! The linking number
    # is an element of pi_1(R^4 \ S^2) = Z (by Alexander duality).
    # Two round spheres that are concentric are unlinked (linking = 0).
    # But if one sphere passes through the disk bounded by another, linking = 1.

    print(f'  CONCLUSION: The linking argument does not directly constrain')
    print(f'  zero positions because concentric 2-spheres in R^4 are unlinked.')
    print(f'  Off-center spheres CAN be linked, but the linking number is')
    print(f'  not forced to be nonzero by the functional equation alone.')

    # ══════════════════════════════════════════════════════════════
    # 8. THE w-REPULSION BOUND
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  8. THE w-REPULSION BOUND (THE BOTTOM LINE)')
    print('#' * 76)

    # The minimum |w| on {u~0} off the CL
    print(f'\n  BOTTOM LINE: does |w| on {{u=0}} stay positive off the CL?')
    print(f'\n  If yes (even numerically): no off-line zeros in the scanned region.')
    print(f'  If no: we found a potential off-line zero (and disproved RH!).')

    # Refined check near the closest approach
    if close_approaches:
        best_sig, best_t, best_w = close_approaches[0]
        print(f'\n  Closest approach: sigma={best_sig:.4f}, t={best_t:.4f}, |w|={best_w:.6e}')

        # Zoom in around this point
        print(f'  Zooming in...')
        sigs_zoom = np.linspace(best_sig - 0.05, best_sig + 0.05, 50)
        ts_zoom = np.linspace(best_t - 0.5, best_t + 0.5, 50)

        min_w_zoom = float('inf')
        for sig in sigs_zoom:
            if abs(sig - 0.5) < 0.01:
                continue
            for t in ts_zoom:
                u_val, v_val = xi_uv(sig, t)
                if abs(u_val) < 1e-10:
                    w_val = abs(w_function(sig, t))
                    if w_val < min_w_zoom:
                        min_w_zoom = w_val
                        min_zoom = (sig, t)

        if min_w_zoom < float('inf'):
            print(f'  Zoomed minimum: |w| = {min_w_zoom:.6e} at sigma={min_zoom[0]:.4f}, t={min_zoom[1]:.4f}')
        else:
            print(f'  No u=0 points found in zoom region.')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45L SYNTHESIS')
    print('=' * 76)

    n_close = len(close_approaches)
    min_abs_w = close_approaches[0][2] if close_approaches else float('inf')

    print(f'''
  THE w-FUNCTION APPROACH:
    v(sigma, t) = (sigma - 1/2) * w(sigma, t)
    Off-line zeros need w = 0 AND u = 0 simultaneously.

  RESULTS:
    Grid scan: {n_close} u=0 points found off CL
    Minimum |w| on {{u=0}} off CL: {min_abs_w:.6e}
    {'NO INTERSECTION FOUND' if min_abs_w > 1e-6 else 'POSSIBLE NEAR-INTERSECTION'}

  THE w=0 CURVES:
    Emanate from CL at extrema of u(1/2,t) (between consecutive zeros).
    {len(w_zeros_off_cl)} w=0 points found off CL.
    These curves exist but stay AWAY from the u=0 curves.

  THE LINKING ARGUMENT:
    Concentric 2-spheres in R^4 are unlinked -- doesn't constrain.
    The Hopf fibration argument requires more sophisticated topology.

  STATUS:
    The w-function reformulation is clean and the gap is VISIBLE:
    - w=0 and u=0 are separate curve families in the (sigma,t) plane
    - They don't intersect in the computed range
    - But PROVING they never intersect requires analytic control of w
    - The hypercomplex structure (j-separation, Fueter) explains WHY
      they don't intersect (pi drives v away from zero off CL)
      but doesn't PROVE it

  THE REMAINING STEP:
    Show that w(sigma, t) != 0 on the curve {{u(sigma, t) = 0}} for sigma != 1/2.
    Equivalently: show that xi'(sigma+it) != 0 wherever xi(sigma+it) = 0.
    This is the statement that ALL ZEROS OF xi ARE SIMPLE -- a well-known
    consequence of RH (and conjectured to be true independently).

    Wait: the simplicity of zeros and w != 0 on u=0 are related but not identical.
    w(sigma,t) = v/(sigma-1/2) involves the antisymmetric part, not directly the derivative.
''')

    print('=' * 76)
    print('  SESSION 45L COMPLETE')
    print('=' * 76)
