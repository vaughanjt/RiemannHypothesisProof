"""Log-concavity analysis of xi(1/2+iz) — the proof barrier.

Key identity:
  log xi(s) = log(1/2) + log(s) + log(s-1) - (s/2)*log(pi)
            + log Gamma(s/2) + log zeta(s)

At s = 1/2 + iz:
  f(z) = log|xi(1/2+iz)|
  f''(z) = f''_poly(z) + f''_gamma(z) + f''_zeta(z)

Where:
  f''_poly  = from the s(s-1)/2 factor  [KNOWN, computable exactly]
  f''_gamma = from Gamma(s/2)           [KNOWN, computable exactly]
  f''_zeta  = from zeta(s)              [THE HARD PART]

If f''(z) < 0 between all zeros => du/dz > 0 => zeros repel
=> Lambda <= 0 => RH.

The question: can we prove f''_zeta has the right sign without
assuming where the zeros are?
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time


def log_xi_parts(z, dps=50):
    """Decompose log|xi(1/2+iz)| into its constituent parts.

    Returns dict with each component evaluated at s = 1/2 + iz.
    """
    with mpmath.workdps(dps):
        z_mp = mpmath.mpf(str(z))
        s = mpmath.mpc('0.5', str(z_mp))

        # Each factor of xi(s) = (1/2) * s * (s-1) * pi^(-s/2) * Gamma(s/2) * zeta(s)

        # 1. log|s| = log|1/2 + iz|
        log_s = mpmath.log(s)

        # 2. log|s-1| = log|-1/2 + iz|
        log_s1 = mpmath.log(s - 1)

        # 3. -(s/2)*log(pi)
        log_pi_part = -(s / 2) * mpmath.log(mpmath.pi)

        # 4. log Gamma(s/2)
        log_gamma = mpmath.loggamma(s / 2)

        # 5. log zeta(s)
        zeta_val = mpmath.zeta(s)
        log_zeta = mpmath.log(zeta_val) if zeta_val != 0 else mpmath.mpf('-inf')

        # 6. log(1/2)
        log_half = mpmath.log(mpmath.mpf('0.5'))

        # Total
        log_xi = log_half + log_s + log_s1 + log_pi_part + log_gamma + log_zeta

        return {
            'z': float(z),
            's_real': 0.5,
            's_imag': float(z),
            'log_half': float(mpmath.re(log_half)),
            'log_s': float(mpmath.re(log_s)),
            'log_s1': float(mpmath.re(log_s1)),
            'log_pi': float(mpmath.re(log_pi_part)),
            'log_gamma': float(mpmath.re(log_gamma)),
            'log_zeta': float(mpmath.re(log_zeta)),
            'log_xi_total': float(mpmath.re(log_xi)),
        }


def f_double_prime_exact(z, dps=50):
    """Compute f''(z) = d^2/dz^2 log|xi(1/2+iz)| using mpmath diff.

    This is the NON-CIRCULAR computation: no Hadamard product,
    no assumption about zero locations. Pure mpmath evaluation.
    """
    with mpmath.workdps(dps):
        def log_abs_xi(z_val):
            s = mpmath.mpc('0.5', z_val)
            xi_val = (s * (s - 1) / 2
                      * mpmath.power(mpmath.pi, -s / 2)
                      * mpmath.gamma(s / 2)
                      * mpmath.zeta(s))
            return mpmath.log(mpmath.fabs(xi_val))

        z_mp = mpmath.mpf(str(z))
        # Second derivative using mpmath's numerical differentiation
        result = mpmath.diff(log_abs_xi, z_mp, 2)
        return float(result)


def f_double_prime_decomposed(z, dps=50):
    """Compute f''(z) decomposed into parts.

    f(z) = Re[log xi(1/2+iz)]
         = log(1/2) + Re[log(1/2+iz)] + Re[log(-1/2+iz)]
           + Re[-(1/2+iz)/2 * log pi]
           + Re[log Gamma((1/4+iz/2))]
           + Re[log zeta(1/2+iz)]

    f''(z) = sum of second z-derivatives of each part.

    Uses mpmath.diff for each component separately.
    """
    with mpmath.workdps(dps):
        z_mp = mpmath.mpf(str(z))

        # Component 1: Re[log(1/2 + iz)] -- from s factor
        def part_s(zv):
            return mpmath.re(mpmath.log(mpmath.mpc('0.5', zv)))
        fpp_s = float(mpmath.diff(part_s, z_mp, 2))

        # Component 2: Re[log(-1/2 + iz)] -- from (s-1) factor
        def part_s1(zv):
            return mpmath.re(mpmath.log(mpmath.mpc('-0.5', zv)))
        fpp_s1 = float(mpmath.diff(part_s1, z_mp, 2))

        # Component 3: Re[-(1/2+iz)/2 * log(pi)] -- from pi^{-s/2}
        # = -(1/4)*log(pi) [constant in z] + derivative of Re[-iz/2 * log pi]
        # Re[-iz/2 * log(pi)] = Re[-i*z*log(pi)/2] = (z/2)*log(pi) * Im(-i) ... hmm
        # Actually: s = 1/2+iz, -(s/2)*log(pi) = -(1/4 + iz/2)*log(pi)
        # Re of this = -log(pi)/4  (the iz/2 * log(pi) is purely imaginary)
        # So d^2/dz^2 of this = 0
        fpp_pi = 0.0

        # Component 4: Re[log Gamma(s/2)] = Re[log Gamma(1/4 + iz/2)]
        def part_gamma(zv):
            return mpmath.re(mpmath.loggamma(mpmath.mpc('0.25', str(mpmath.mpf(str(zv))/2))))
        fpp_gamma = float(mpmath.diff(part_gamma, z_mp, 2))

        # Component 5: Re[log zeta(1/2+iz)] -- THE HARD PART
        def part_zeta(zv):
            s = mpmath.mpc('0.5', zv)
            zeta_val = mpmath.zeta(s)
            if zeta_val == 0:
                return mpmath.mpf('-inf')
            return mpmath.re(mpmath.log(zeta_val))
        fpp_zeta = float(mpmath.diff(part_zeta, z_mp, 2))

        # Total
        fpp_total = fpp_s + fpp_s1 + fpp_pi + fpp_gamma + fpp_zeta

        return {
            'z': float(z),
            'fpp_s': fpp_s,           # from s
            'fpp_s1': fpp_s1,         # from (s-1)
            'fpp_pi': fpp_pi,         # from pi^{-s/2}
            'fpp_gamma': fpp_gamma,   # from Gamma(s/2)
            'fpp_zeta': fpp_zeta,     # from zeta(s)  <-- THE BARRIER
            'fpp_known': fpp_s + fpp_s1 + fpp_pi + fpp_gamma,  # sum of known parts
            'fpp_total': fpp_total,
        }


def f_double_prime_analytical(z, dps=50):
    """Compute f'' parts using analytical formulas where possible.

    For the "known" parts, we can derive exact formulas:

    Part 1: Re[log(1/2+iz)]
      d/dz Re[log(1/2+iz)] = Re[i/(1/2+iz)] = Re[i(1/2-iz)/(1/4+z^2)]
                            = z/(1/4+z^2)
      d^2/dz^2 = (1/4 - z^2) / (1/4 + z^2)^2   [quotient rule]

    Part 2: Re[log(-1/2+iz)]
      Same formula with 1/2 -> -1/2:
      d^2/dz^2 = (1/4 - z^2) / (1/4 + z^2)^2   [same! by symmetry]

    Part 3: pi factor = 0

    Part 4: Gamma part -- use digamma/trigamma:
      Re[log Gamma(1/4+iz/2)]
      d/dz = Re[(i/2)*psi(1/4+iz/2)]
      d^2/dz^2 = Re[(i/2)^2 * psi'(1/4+iz/2)] = -(1/4)*Re[psi'(1/4+iz/2)]
      where psi' is the trigamma function.

    Part 5: Zeta part -- the hard part, computed numerically.
    """
    with mpmath.workdps(dps):
        z_mp = mpmath.mpf(str(z))
        z2 = z_mp**2

        # Part 1+2: polynomial factors (both give same contribution)
        # d^2/dz^2 Re[log(1/2+iz)] = (1/4 - z^2)/(1/4 + z^2)^2
        denom = (mpmath.mpf('0.25') + z2)**2
        fpp_s = float((mpmath.mpf('0.25') - z2) / denom)
        fpp_s1 = fpp_s  # identical by symmetry

        # Part 3: pi
        fpp_pi = 0.0

        # Part 4: Gamma -- use trigamma
        # d^2/dz^2 Re[log Gamma(1/4+iz/2)] = -(1/4)*Re[psi(1, 1/4+iz/2)]
        s_half = mpmath.mpc('0.25', str(z_mp / 2))
        trigamma_val = mpmath.polygamma(1, s_half)
        fpp_gamma = float(-mpmath.re(trigamma_val) / 4)

        # Part 5: Zeta (numerical)
        def part_zeta(zv):
            s = mpmath.mpc('0.5', zv)
            zeta_val = mpmath.zeta(s)
            if zeta_val == 0:
                return mpmath.mpf('-inf')
            return mpmath.re(mpmath.log(zeta_val))
        fpp_zeta = float(mpmath.diff(part_zeta, z_mp, 2))

        fpp_known = fpp_s + fpp_s1 + fpp_pi + fpp_gamma
        fpp_total = fpp_known + fpp_zeta

        return {
            'z': float(z),
            'fpp_s': fpp_s,
            'fpp_s1': fpp_s1,
            'fpp_pi': fpp_pi,
            'fpp_gamma': fpp_gamma,
            'fpp_zeta': fpp_zeta,
            'fpp_known': fpp_known,
            'fpp_total': fpp_total,
        }


def run_logconcavity_study(z_values=None, dps=50):
    """Comprehensive log-concavity study.

    For each z between consecutive zeros:
    1. Compute f''(z) total (non-circular, from mpmath)
    2. Decompose into known parts + zeta part
    3. Determine which part dominates the sign
    4. Track margin (how negative is f''?) vs height
    """
    all_zeros = np.load('_zeros_200.npy')

    if z_values is None:
        # Sample midpoints between consecutive zeros
        z_values = []
        for i in range(min(60, len(all_zeros) - 1)):
            mid = (all_zeros[i] + all_zeros[i + 1]) / 2.0
            z_values.append(mid)
        # Also sample at 1/4 and 3/4 points for shape info
        for i in range(min(30, len(all_zeros) - 1)):
            q1 = all_zeros[i] + 0.25 * (all_zeros[i+1] - all_zeros[i])
            q3 = all_zeros[i] + 0.75 * (all_zeros[i+1] - all_zeros[i])
            z_values.extend([q1, q3])

    z_values = sorted(set(z_values))

    print("=" * 65)
    print("LOG-CONCAVITY DECOMPOSITION STUDY")
    print("=" * 65)
    print(f"  Computing f''(z) at {len(z_values)} points")
    print(f"  Working precision: {dps} digits")
    print(f"  z range: [{min(z_values):.2f}, {max(z_values):.2f}]")

    results = []
    t0 = time.time()

    for idx, z in enumerate(z_values):
        if idx % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(z_values)}] z = {z:.4f} ({elapsed:.1f}s)")

        r = f_double_prime_analytical(z, dps=dps)

        # Also compute f'' directly as a cross-check
        fpp_direct = f_double_prime_exact(z, dps=dps)
        r['fpp_direct'] = fpp_direct
        r['decomp_error'] = abs(r['fpp_total'] - fpp_direct)

        results.append(r)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── Analysis ──
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    zs = [r['z'] for r in results]
    fpp_totals = [r['fpp_total'] for r in results]
    fpp_knowns = [r['fpp_known'] for r in results]
    fpp_zetas = [r['fpp_zeta'] for r in results]
    fpp_gammas = [r['fpp_gamma'] for r in results]
    fpp_ss = [r['fpp_s'] for r in results]

    # Check log-concavity
    n_positive = sum(1 for f in fpp_totals if f > 0)
    print(f"\n  f''(z) > 0 at {n_positive}/{len(results)} points")
    if n_positive == 0:
        print("  => log|xi(1/2+iz)| is CONCAVE at all sampled points")
        print("  => du/dz > 0 CONFIRMED (non-circular)")
    else:
        print("  => CONCAVITY FAILS at some points!")
        for r in results:
            if r['fpp_total'] > 0:
                print(f"     z = {r['z']:.6f}: f'' = {r['fpp_total']:.6e}")

    # Decomposition analysis
    print(f"\n  Decomposition (at midpoints between zeros):")
    print(f"  {'z':>10}  {'f_total':>12}  {'f_known':>12}  {'f_zeta':>12}  {'f_gamma':>12}  {'dominates':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")

    # Show every 5th result
    for r in results[::5]:
        dom = "zeta" if abs(r['fpp_zeta']) > abs(r['fpp_known']) else "known"
        print(f"  {r['z']:>10.2f}  {r['fpp_total']:>12.6f}  {r['fpp_known']:>12.6f}"
              f"  {r['fpp_zeta']:>12.6f}  {r['fpp_gamma']:>12.6f}  {dom:>10}")

    # Cross-check: decomposed vs direct
    errors = [r['decomp_error'] for r in results]
    print(f"\n  Decomposition cross-check:")
    print(f"    Max |decomposed - direct| = {max(errors):.4e}")
    print(f"    Mean = {np.mean(errors):.4e}")

    # The critical question: sign of each part
    known_signs = [1 if f > 0 else -1 for f in fpp_knowns]
    zeta_signs = [1 if f > 0 else -1 for f in fpp_zetas]

    n_known_positive = sum(1 for s in known_signs if s > 0)
    n_zeta_positive = sum(1 for s in zeta_signs if s > 0)

    print(f"\n  Sign analysis:")
    print(f"    f''_known > 0: {n_known_positive}/{len(results)}")
    print(f"    f''_zeta  > 0: {n_zeta_positive}/{len(results)}")
    print(f"    f''_total > 0: {n_positive}/{len(results)}")

    if n_known_positive == 0:
        print("    => Known parts ALWAYS contribute to concavity")
    if n_zeta_positive > 0 and n_positive == 0:
        print("    => Zeta part is ANTI-concave but known parts overpower it!")
        print("    => THIS IS THE PROOF STRUCTURE:")
        print("       Show |f''_known| > |f''_zeta| between all zeros")
    elif n_zeta_positive == 0:
        print("    => Both parts contribute to concavity!")
        print("    => Proving f''_zeta < 0 independently would give RH")

    # Margin analysis: how negative is f'' at each height?
    # Group by inter-zero interval
    margin_data = []
    for i in range(len(all_zeros) - 1):
        z_lo, z_hi = all_zeros[i], all_zeros[i + 1]
        interval_results = [r for r in results if z_lo < r['z'] < z_hi]
        if interval_results:
            max_fpp = max(r['fpp_total'] for r in interval_results)
            min_fpp = min(r['fpp_total'] for r in interval_results)
            margin_data.append({
                'interval': i,
                'z_mid': (z_lo + z_hi) / 2,
                'gap': z_hi - z_lo,
                'max_fpp': max_fpp,  # closest to 0 (worst case)
                'min_fpp': min_fpp,
            })

    if margin_data:
        print(f"\n  Concavity margin by interval:")
        print(f"  {'interval':>8}  {'z_mid':>8}  {'gap':>8}  {'max f\"':>12}  (0=boundary)")
        for md in margin_data[:15]:
            print(f"  {md['interval']:>8}  {md['z_mid']:>8.2f}  {md['gap']:>8.4f}"
                  f"  {md['max_fpp']:>12.6f}")

    # ── Plots ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # (0,0) f''(z) total
    ax = axes[0, 0]
    ax.plot(zs, fpp_totals, 'b.-', markersize=3, linewidth=0.8)
    ax.axhline(y=0, color='red', linewidth=1.5, alpha=0.7)
    ax.set_ylabel("f''(z)")
    ax.set_title("f''(z) = d^2/dz^2 log|xi(1/2+iz)|  [TOTAL]")
    ax.grid(True, alpha=0.3)

    # (0,1) Decomposition: known vs zeta parts
    ax = axes[0, 1]
    ax.plot(zs, fpp_knowns, 'g.-', markersize=2, linewidth=0.8, label='known (s, Gamma)')
    ax.plot(zs, fpp_zetas, 'r.-', markersize=2, linewidth=0.8, label='zeta part')
    ax.plot(zs, fpp_totals, 'b-', linewidth=1.5, alpha=0.5, label='total')
    ax.axhline(y=0, color='gray', linewidth=1, alpha=0.5)
    ax.set_ylabel("f'' component")
    ax.set_title("Decomposition: known parts vs zeta part")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0) Individual components
    ax = axes[1, 0]
    ax.plot(zs, fpp_ss, 'm-', linewidth=0.8, label='s*(s-1) part')
    ax.plot(zs, fpp_gammas, 'c-', linewidth=0.8, label='Gamma part')
    ax.plot(zs, fpp_zetas, 'r-', linewidth=0.8, label='zeta part')
    ax.axhline(y=0, color='gray', linewidth=1, alpha=0.5)
    ax.set_ylabel("f'' component")
    ax.set_title("Individual component contributions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) Ratio: |f''_zeta / f''_known|
    ax = axes[1, 1]
    ratios = [abs(fz) / abs(fk) if abs(fk) > 1e-15 else np.nan
              for fz, fk in zip(fpp_zetas, fpp_knowns)]
    ax.plot(zs, ratios, 'ko-', markersize=3, linewidth=0.8)
    ax.axhline(y=1.0, color='red', linewidth=1.5, alpha=0.7,
               label='|zeta| = |known| (balance)')
    ax.set_ylabel('|f"_zeta| / |f"_known|')
    ax.set_title('Relative strength: zeta vs known parts')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(2.0, np.nanmax(ratios) * 1.1) if ratios else 2.0)

    # (2,0) Margin vs height
    ax = axes[2, 0]
    if margin_data:
        z_mids = [m['z_mid'] for m in margin_data]
        margins = [m['max_fpp'] for m in margin_data]
        ax.plot(z_mids, margins, 'bs-', markersize=5, linewidth=1)
        ax.axhline(y=0, color='red', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('z (height)')
        ax.set_ylabel('max f" in interval (0 = fails)')
        ax.set_title('Concavity margin vs height')
        ax.grid(True, alpha=0.3)

    # (2,1) Margin vs gap size
    ax = axes[2, 1]
    if margin_data:
        gaps = [m['gap'] for m in margin_data]
        margins = [m['max_fpp'] for m in margin_data]
        ax.scatter(gaps, margins, c='blue', s=30, alpha=0.7)
        ax.axhline(y=0, color='red', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Zero gap')
        ax.set_ylabel('max f" in interval')
        ax.set_title('Concavity margin vs gap size')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Log-Concavity Decomposition of xi(1/2+iz)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('dbn_logconcavity.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: dbn_logconcavity.png")
    plt.close(fig)

    # Save data
    with open('dbn_logconcavity.json', 'w') as f:
        json.dump({
            'results': results,
            'margins': margin_data,
            'summary': {
                'n_points': len(results),
                'n_positive_total': n_positive,
                'n_positive_known': n_known_positive,
                'n_positive_zeta': n_zeta_positive,
            }
        }, f, indent=2)
    print("  Saved: dbn_logconcavity.json")

    return results, margin_data


def derive_collision_correction():
    """Derive the N-body collision correction factor alpha analytically.

    For the closest pair with gap g, the 2-body ODE gives:
      dg/dt = -4/g  =>  g^2(t) = g0^2 + 4t  =>  t_coll = -g0^2/4

    In the N-body case, neighboring zeros contribute additional force.
    Assume zeros are approximately uniformly spaced with mean spacing d.
    The closest pair (say zeros j, j+1 with gap g << d) feels:

    From zero j-1 (distance ~d from j): force on j toward j+1 = +2/(d)
    From zero j+2 (distance ~d from j+1): force on j+1 toward j = +2/(d)
    ... and similarly for all other neighbors.

    The total additional compression on the gap:
      F_extra = 2 * sum_{k=1}^{N} [1/(k*d - g/2)^2 - 1/(k*d + g/2)^2]
              ~ 2 * sum_{k=1}^{N} [2*g / (k*d)^3]   for g << d
              = 4g/d^3 * sum_{k=1}^{N} 1/k^3

    But wait -- the correction to dg/dt has a different structure. Let me
    think more carefully...

    The gap equation: dg/dt = dz_{j+1}/dt - dz_j/dt

    From the Coulomb ODE:
    dz_j/dt = +2/(z_j - z_{j+1}) + sum_{k != j,j+1} 2/(z_j - z_k)
    dz_{j+1}/dt = +2/(z_{j+1} - z_j) + sum_{k != j,j+1} 2/(z_{j+1} - z_k)

    dg/dt = dz_{j+1}/dt - dz_j/dt
          = 2/(z_{j+1}-z_j) - 2/(z_j-z_{j+1})   [mutual term]
            + sum_{k} [2/(z_{j+1}-z_k) - 2/(z_j-z_k)]

    The mutual term: 2/(-g) - 2/(g) = -4/g  [this is the 2-body part]

    The neighbor correction for zero k at position z_k ~ z_j + n*d (n != 0, -1):
    2/(z_{j+1} - z_k) - 2/(z_j - z_k)
    = 2/((z_j + g) - (z_j + n*d)) - 2/(z_j - (z_j + n*d))
    = 2/(g - n*d) - 2/(-n*d)
    = 2/(g - n*d) + 2/(n*d)
    = 2*g / [n*d*(g - n*d)]

    For |n| >= 1 and g << d:
    ~ 2*g / [n*d * (-n*d)] = -2*g / (n*d)^2    [for n > 0: g - nd ~ -nd]

    Wait I need to be more careful about the geometry. Let me index:
    z_j is the left zero of the closest pair
    z_{j+1} = z_j + g is the right zero

    Other zeros:
    z_{j+k} ~ z_j + g + (k-1)*d   for k >= 2 (to the right)
    z_{j-k} ~ z_j - k*d           for k >= 1 (to the left)
    """
    print("\n" + "=" * 65)
    print("COLLISION CORRECTION FACTOR DERIVATION")
    print("=" * 65)

    # Numerical calculation of the correction
    # Place zeros on a line: ..., -2d, -d, 0, g, d+g, 2d+g, ...
    # Closest pair is at 0 and g.
    # Compute dg/dt from full Coulomb sum.

    d = 1.0  # mean spacing (normalized)
    g_over_d = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5])

    print(f"\n  Mean spacing d = {d}")
    print(f"  Testing g/d ratios: {g_over_d}")
    print(f"\n  {'g/d':>8}  {'dg/dt_2body':>14}  {'dg/dt_Nbody':>14}  {'alpha':>8}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*8}")

    for ratio in g_over_d:
        g = ratio * d
        N_neighbors = 500  # enough for convergence

        # Positions of all zeros (closest pair at 0 and g)
        # Left neighbors: -d, -2d, -3d, ...
        # Right neighbors: g+d, g+2d, g+3d, ...

        # 2-body contribution
        dg_2body = -4.0 / g

        # N-body correction: sum over all other zeros
        correction = 0.0
        for k in range(1, N_neighbors + 1):
            # Zero to the left at -k*d
            z_k = -k * d
            term = 2.0 / (g - z_k) - 2.0 / (0 - z_k)  # (j+1 force) - (j force)
            correction += term

            # Zero to the right at g + k*d
            z_k = g + k * d
            term = 2.0 / (g - z_k) - 2.0 / (0 - z_k)
            correction += term

        dg_Nbody = dg_2body + correction
        alpha = dg_Nbody / dg_2body

        print(f"  {ratio:>8.3f}  {dg_2body:>14.4f}  {dg_Nbody:>14.4f}  {alpha:>8.4f}")

    # Analytical approximation for g << d:
    # correction ~ sum_{k=1}^{inf} [2/(g+kd) + 2/(kd) - 2/(-kd) - 2/(g-kd)] ... complicated
    # Simpler: use the exact Coulomb sum for equispaced + perturbed pair
    print("\n  Analytical approximation for g << d:")
    print("  The correction factor alpha = dg/dt(N-body) / dg/dt(2-body)")

    # For g -> 0, the leading correction comes from nearest neighbors:
    # zero at -d: contributes 2/(g+d) - 2/d = -2g/[d(g+d)] ~ -2g/d^2
    # zero at g+d: contributes 2/(g-g-d) - 2/(0-g-d) = -2/d + 2/(g+d) = -2g/[d(g+d)] ~ -2g/d^2
    # Each neighbor pair at distance n*d contributes ~ -4g/(n*d)^2
    # Total correction ~ -4g/d^2 * sum_{n=1}^{inf} 1/n^2 = -4g * pi^2/(6*d^2)
    #
    # So dg/dt ~ -4/g - 4g*pi^2/(6*d^2)
    #          = -4/g * [1 + g^2*pi^2/(6*d^2)]
    #
    # For g << d, the correction is O(g^2/d^2), which is tiny!
    # This doesn't match alpha ~ 1.88...
    #
    # The issue is that in our ODE, the closest pair's gap g(t) stays ~g
    # while the correction accumulates over time. The ratio of collision
    # times is NOT the same as the ratio of instantaneous forces at t=0.

    print("  The instantaneous force correction at g << d is O(g^2/d^2) -- tiny!")
    print("  But the collision time ratio ~ 0.53 is large.")
    print("  This means the correction GROWS as g shrinks (closer to collision).")
    print("")
    print("  As g -> 0: neighbors at distance ~d push INWARD on the pair.")
    print("  The pair's mutual repulsion (-4/g) dominates, but the neighbor")
    print("  compression (-4g*pi^2/6d^2) integrates to a large time correction.")
    print("")

    # More careful analysis: near collision, the gap ODE becomes
    # dg/dt = -4/g + F_bg(t)
    # where F_bg is the background force from neighbors (slowly varying).
    # As g -> 0 at t -> t_coll, the dominant balance is still g^2 ~ 4|t|
    # but the background force shifts the collision point.

    # For uniform spacing d, the background force on the gap is:
    # F_bg = sum_{k=1}^{inf} [2/(g+kd) + 2/(kd)] + [2/(-kd) + 2/(g-kd)]
    # At g=0: F_bg = 0 (by symmetry)
    # At g small: F_bg ~ -C * g with C = ...

    # Let's compute F_bg as a function of g/d more carefully
    print("  Background force F_bg(g) vs gap g:")
    print(f"  {'g/d':>8}  {'F_bg':>12}  {'F_bg*g':>12}  {'ratio_to_mutual':>18}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*18}")

    for ratio in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        g = ratio * d
        F_bg = 0.0
        for k in range(1, 1000):
            # Left neighbor at -k*d
            F_bg += 2.0 / (g + k*d) + 2.0 / (k*d)
            # Right neighbor at g+k*d
            F_bg += 2.0 / (-k*d) + 2.0 / (-g - k*d)

        F_mutual = -4.0 / g
        ratio_val = F_bg / F_mutual if abs(F_mutual) > 1e-30 else 0

        print(f"  {ratio:>8.4f}  {F_bg:>12.6f}  {F_bg*g:>12.6f}  {ratio_val:>18.6f}")

    return


if __name__ == '__main__':
    # Quick collision correction analysis
    derive_collision_correction()

    # Main log-concavity study
    results, margins = run_logconcavity_study(dps=40)
