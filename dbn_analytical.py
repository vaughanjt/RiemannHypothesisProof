"""Analytical paths from dBN zero dynamics.

Path 1: Collision time formula verification
  - 2-body Coulomb approximation: dg/dt = -4/g => g^2 = g0^2 + 4t
  - Predicted collision: t_coll = -g0^2/4
  - Test against full N-body backward ODE

Path 2: Burgers / Cole-Hopf framework
  - u(t,z) = -2 * H_t'(z) / H_t(z) satisfies Burgers equation
  - Cole-Hopf: u = -2 * d/dz [log H_t(z)]
  - Shock formation in u <=> zero collision in H_t
  - Study what property of xi prevents shocks for t >= 0
"""

import numpy as np
import mpmath
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time

from dbn_explorer import (
    xi_function, coulomb_rhs, normalized_spacings,
    spacing_statistics, wigner_surmise_gue,
)


# =====================================================================
# PATH 1: Collision time formula
# =====================================================================

def find_collision_time(z0, t_end=-0.5, n_steps=2000, rtol=1e-8, atol=1e-10):
    """Evolve backward with tight tolerances to find precise collision time.

    Uses event detection to stop at near-collision (gap < 1% of initial min).

    Returns:
        t_collision (float or None), closest_pair_history (list of dicts)
    """
    z_sorted = np.sort(z0)
    gaps = np.diff(z_sorted)
    min_gap_init = np.min(gaps)
    min_idx = np.argmin(gaps)
    collision_threshold = min_gap_init * 0.01  # 1% of initial

    def collision_event(t, z_flat):
        z_s = np.sort(z_flat)
        return np.min(np.diff(z_s)) - collision_threshold

    collision_event.terminal = True
    collision_event.direction = -1

    # Track closest pair at each step
    history = []

    sol = solve_ivp(
        coulomb_rhs,
        t_span=(0.0, t_end),
        y0=z0,
        method='DOP853',
        rtol=rtol,
        atol=atol,
        events=collision_event,
        dense_output=True,
        max_step=abs(t_end) / n_steps,
    )

    # Sample the dense output for history
    t_sample = np.linspace(0.0, sol.t[-1], min(n_steps, len(sol.t) * 2))
    for t_val in t_sample:
        z_at_t = sol.sol(t_val)
        z_s = np.sort(z_at_t)
        g = np.diff(z_s)
        mi = np.argmin(g)
        history.append({
            't': float(t_val),
            'min_gap': float(g[mi]),
            'min_gap_sq': float(g[mi]**2),
            'z1': float(z_s[mi]),
            'z2': float(z_s[mi + 1]),
        })

    t_coll = None
    if sol.t_events[0].size > 0:
        t_coll = float(sol.t_events[0][0])

    return t_coll, history


def run_collision_test(n_values=None):
    """Test t_collision = -g0^2/4 prediction across multiple N.

    The 2-body Coulomb approximation for the closest pair:
      dg/dt = -4/g  (ignoring all other zeros)
      => g(t)^2 = g0^2 + 4t
      => collision at t = -g0^2/4

    The N-body corrections come from other zeros pushing/pulling
    the closest pair. We measure the correction factor.
    """
    if n_values is None:
        n_values = [25, 50, 100, 200]

    all_zeros = np.load('_zeros_200.npy')

    print("=" * 65)
    print("COLLISION TIME FORMULA: t_coll = -g0^2 / 4")
    print("=" * 65)
    print("Testing 2-body Coulomb prediction against full N-body ODE\n")

    results = []

    for n_z in n_values:
        if n_z > len(all_zeros):
            continue

        z0 = all_zeros[:n_z].copy()
        z_sorted = np.sort(z0)
        gaps = np.diff(z_sorted)
        g0 = np.min(gaps)
        min_idx = np.argmin(gaps)
        z1, z2 = z_sorted[min_idx], z_sorted[min_idx + 1]
        t_predicted = -g0**2 / 4.0

        print(f"  -- N = {n_z} --")
        print(f"  Closest pair: z = {z1:.4f}, {z2:.4f}")
        print(f"  Initial gap g0 = {g0:.6f}")
        print(f"  Predicted collision: t = {t_predicted:.8f}")

        t0 = time.time()
        t_coll, history = find_collision_time(
            z0, t_end=t_predicted * 3, n_steps=3000)
        elapsed = time.time() - t0

        if t_coll is not None:
            ratio = t_coll / t_predicted
            print(f"  Actual collision:    t = {t_coll:.8f}")
            print(f"  Ratio (actual/predicted): {ratio:.6f}")
            print(f"  Correction factor: {1 - ratio:.4f} ({elapsed:.1f}s)")
        else:
            print(f"  No collision detected (ODE reached t = {history[-1]['t']:.6f})")
            ratio = np.nan

        results.append({
            'N': n_z,
            'g0': g0,
            'z1': z1,
            'z2': z2,
            't_predicted': t_predicted,
            't_actual': t_coll,
            'ratio': ratio,
            'history': history,
        })

    # ── Summary ──
    print("\n" + "=" * 65)
    print("COLLISION TEST SUMMARY")
    print("=" * 65)
    print(f"  {'N':>5}  {'g0':>10}  {'t_pred':>12}  {'t_actual':>12}  {'ratio':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r in results:
        ta = f"{r['t_actual']:.8f}" if r['t_actual'] is not None else "N/A"
        ra = f"{r['ratio']:.4f}" if not np.isnan(r.get('ratio', np.nan)) else "N/A"
        print(f"  {r['N']:>5}  {r['g0']:>10.6f}  {r['t_predicted']:>12.8f}"
              f"  {ta:>12}  {ra:>8}")

    # ── Plot: g^2 vs t (should be linear if 2-body holds) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, r in enumerate(results):
        if idx >= 4:
            break
        ax = axes[idx // 2, idx % 2]
        h = r['history']
        ts = [p['t'] for p in h]
        g2s = [p['min_gap_sq'] for p in h]

        ax.plot(ts, g2s, 'b.-', markersize=1, linewidth=1.5, label='ODE (N-body)')

        # 2-body prediction: g^2 = g0^2 + 4t
        t_line = np.array([min(ts), 0])
        g2_line = r['g0']**2 + 4 * t_line
        ax.plot(t_line, g2_line, 'r--', linewidth=2,
                label=f'2-body: g0^2 + 4t')

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        if r['t_predicted'] is not None:
            ax.axvline(x=r['t_predicted'], color='red', linestyle=':',
                       alpha=0.5, label=f"pred: {r['t_predicted']:.6f}")
        if r['t_actual'] is not None:
            ax.axvline(x=r['t_actual'], color='green', linestyle=':',
                       alpha=0.7, label=f"actual: {r['t_actual']:.6f}")

        ax.set_xlabel('t')
        ax.set_ylabel('g^2 (min gap squared)')
        ax.set_title(f'N = {r["N"]}: g0 = {r["g0"]:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Collision Time: 2-Body Prediction vs N-Body ODE', fontsize=13)
    plt.tight_layout()
    fig.savefig('dbn_collision_test.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: dbn_collision_test.png")
    plt.close(fig)

    with open('dbn_collision.json', 'w') as f:
        # Save without full history (too large)
        save = [{k: v for k, v in r.items() if k != 'history'} for r in results]
        json.dump(save, f, indent=2)
    print("  Saved: dbn_collision.json")

    return results


# =====================================================================
# PATH 2: Burgers / Cole-Hopf framework
# =====================================================================

def xi_and_derivative(z, dps=30):
    """Compute xi(1/2+iz) and its derivative d/dz[xi(1/2+iz)].

    d/dz[xi(1/2+iz)] = i * xi'(1/2+iz)
    where xi'(s) = d/ds[xi(s)].

    Uses numerical differentiation with Richardson extrapolation.
    """
    with mpmath.workdps(dps):
        h = mpmath.mpf('1e-8')
        z_mp = mpmath.mpf(str(z))

        # Central difference with Richardson extrapolation
        # f'(z) ~ [f(z+h) - f(z-h)] / (2h)
        # Improved: [8f(z+h) - 8f(z-h) - f(z+2h) + f(z-2h)] / (12h)
        vals = []
        for dz in [z_mp - 2*h, z_mp - h, z_mp + h, z_mp + 2*h]:
            s = mpmath.mpc('0.5', str(dz))
            v = (s * (s - 1) / 2
                 * mpmath.power(mpmath.pi, -s/2)
                 * mpmath.gamma(s/2)
                 * mpmath.zeta(s))
            vals.append(v)

        # 4th-order central difference (chain rule: d/dz = i * d/ds)
        # But s = 1/2 + iz, so ds/dz = i
        # xi(1/2+iz) as function of z: derivative = i * xi'(s)
        # For real z, xi(1/2+iz) is real, and its z-derivative is real
        # Using real-valued central differences directly on xi(1/2+iz) as f(z):
        deriv = (-vals[3] + 8*vals[2] - 8*vals[1] + vals[0]) / (12 * h)

        s0 = mpmath.mpc('0.5', str(z_mp))
        val = (s0 * (s0 - 1) / 2
               * mpmath.power(mpmath.pi, -s0/2)
               * mpmath.gamma(s0/2)
               * mpmath.zeta(s0))

        return float(mpmath.re(val)), float(mpmath.re(deriv))


def compute_burgers_field(z_grid, dps=25):
    """Compute u(0,z) = -2 * H_0'(z) / H_0(z) on a grid.

    This is the Burgers velocity field at t=0. Its poles are at the
    zeta zeros. Between poles, u measures the "repulsive force" that
    keeps zeros apart.

    Returns:
        u_vals, H0_vals, H0_prime_vals
    """
    H0 = []
    H0p = []
    u = []

    for z in z_grid:
        val, deriv = xi_and_derivative(z, dps=dps)
        H0.append(val)
        H0p.append(deriv)
        if abs(val) > 1e-30:
            u.append(-2.0 * deriv / val)
        else:
            u.append(np.nan)  # pole

    return np.array(u), np.array(H0), np.array(H0p)


def verify_burgers_equation(z_grid, u_vals, H0_vals, dt=0.001, dps=20):
    """Numerically verify that u satisfies Burgers' equation.

    du/dt + u * du/dz = d^2u/dz^2

    At t=0:
    - du/dz computed from grid (finite differences)
    - d^2u/dz^2 computed from grid
    - du/dt computed from H_t at t=dt vs t=0
    """
    dz = z_grid[1] - z_grid[0]

    # du/dz and d^2u/dz^2 from finite differences (interior points)
    du_dz = np.gradient(u_vals, dz)
    d2u_dz2 = np.gradient(du_dz, dz)

    # du/dt from time evolution: u(dt,z) - u(0,z)) / dt
    # Compute H_dt via FFT heat evolution
    from dbn_explorer import compute_Ht_fft
    H_dt = compute_Ht_fft(z_grid, H0_vals, dt)
    H_dt_prime = np.gradient(H_dt, dz)

    u_dt = np.full_like(u_vals, np.nan)
    valid = np.abs(H_dt) > 1e-20
    u_dt[valid] = -2.0 * H_dt_prime[valid] / H_dt[valid]

    du_dt = (u_dt - u_vals) / dt

    # Burgers residual: du/dt + u*du/dz - d^2u/dz^2 should be ~0
    residual = du_dt + u_vals * du_dz - d2u_dz2

    # Mask near poles (where u is huge)
    away_from_poles = np.abs(u_vals) < 100
    valid_mask = away_from_poles & np.isfinite(residual)

    return {
        'du_dt': du_dt,
        'advection': u_vals * du_dz,
        'diffusion': d2u_dz2,
        'residual': residual,
        'valid_mask': valid_mask,
        'rms_residual': float(np.sqrt(np.nanmean(residual[valid_mask]**2)))
            if valid_mask.sum() > 0 else np.nan,
        'max_residual': float(np.nanmax(np.abs(residual[valid_mask])))
            if valid_mask.sum() > 0 else np.nan,
    }


def analyze_shock_conditions(z_grid, u_vals, H0_vals, H0_prime_vals):
    """Analyze conditions for shock formation in Burgers equation.

    A shock forms when du/dz -> -inf (compressive wave steepening).
    For the inviscid Burgers equation, shocks form when du/dz < 0
    somewhere initially. But we have VISCOUS Burgers (the d^2u/dz^2
    term), which prevents actual shocks -- instead, we get rapid
    steepening that corresponds to zeros approaching each other.

    Key diagnostic: the "shock strength" = min(du/dz) at each t.
    If du/dz stays bounded from below, zeros never collide.

    For our problem:
    u = -2 H'/H => du/dz = -2 (H''H - H'^2) / H^2 = -2 H''/H + 2(H'/H)^2
                 = -2 H''/H + u^2/2

    So du/dz = u^2/2 - 2*H''/H

    At a zero of H (z = gamma_j), u has a simple pole: u ~ -2/(z - gamma_j)
    and du/dz ~ 2/(z - gamma_j)^2 > 0. So du/dz is POSITIVE near zeros.

    Between zeros, H doesn't vanish, and du/dz can be negative.
    The question: how negative can du/dz get between zeros?
    """
    dz = z_grid[1] - z_grid[0]

    # du/dz from finite differences
    du_dz = np.gradient(u_vals, dz)

    # H''/H term
    H0_pp = np.gradient(H0_prime_vals, dz)  # H''
    H_ratio = np.full_like(H0_vals, np.nan)
    valid = np.abs(H0_vals) > 1e-20
    H_ratio[valid] = H0_pp[valid] / H0_vals[valid]

    # du/dz analytical check: should equal u^2/2 - 2*H''/H
    du_dz_analytical = u_vals**2 / 2 - 2 * H_ratio

    # Find most compressive points (most negative du/dz between zeros)
    away_from_poles = np.abs(u_vals) < 50
    valid_mask = away_from_poles & np.isfinite(du_dz)

    # Identify zero locations (sign changes of H0)
    zero_crossings = np.where(np.diff(np.sign(H0_vals)))[0]

    # For each inter-zero interval, find min du/dz
    intervals = []
    for i in range(len(zero_crossings) - 1):
        i1, i2 = zero_crossings[i] + 2, zero_crossings[i + 1] - 1
        if i2 <= i1:
            continue
        segment = du_dz[i1:i2]
        mask = np.isfinite(segment)
        if mask.sum() > 0:
            min_val = float(np.min(segment[mask]))
            min_loc = z_grid[i1 + np.argmin(segment[mask])]
            intervals.append({
                'z_left': float(z_grid[zero_crossings[i]]),
                'z_right': float(z_grid[zero_crossings[i + 1]]),
                'gap': float(z_grid[zero_crossings[i + 1]] -
                             z_grid[zero_crossings[i]]),
                'min_du_dz': min_val,
                'min_du_dz_loc': float(min_loc),
            })

    return {
        'du_dz': du_dz,
        'du_dz_analytical': du_dz_analytical,
        'H_ratio': H_ratio,
        'intervals': intervals,
        'zero_crossings': zero_crossings,
    }


def run_burgers_analysis(z_range=(10, 160), n_points=3000, dps=25):
    """Full Burgers/Cole-Hopf analysis at t=0.

    1. Compute u(0,z) = -2*xi'/xi
    2. Verify Burgers equation
    3. Analyze shock conditions (du/dz profile)
    4. Study the convexity structure of log|H_0|
    """
    print("\n" + "=" * 65)
    print("BURGERS / COLE-HOPF ANALYSIS")
    print("=" * 65)

    z_grid = np.linspace(z_range[0], z_range[1], n_points)
    dz = z_grid[1] - z_grid[0]

    # 1. Compute Burgers field
    print(f"\n[1/4] Computing u(0,z) = -2*xi'/xi on {n_points} points...")
    print(f"  z in [{z_range[0]}, {z_range[1]}], dz = {dz:.4f}")
    t0 = time.time()
    u_vals, H0_vals, H0p_vals = compute_burgers_field(z_grid, dps=dps)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    n_poles = np.sum(np.isnan(u_vals))
    print(f"  Poles (zeros of xi): {n_poles}")
    finite_mask = np.isfinite(u_vals) & (np.abs(u_vals) < 200)
    print(f"  u range (away from poles): [{np.min(u_vals[finite_mask]):.2f}, "
          f"{np.max(u_vals[finite_mask]):.2f}]")

    # 2. Verify Burgers equation
    print(f"\n[2/4] Verifying Burgers equation du/dt + u*du/dz = d2u/dz2...")
    burgers = verify_burgers_equation(z_grid, u_vals, H0_vals, dt=0.001, dps=20)
    print(f"  RMS residual: {burgers['rms_residual']:.4e}")
    print(f"  Max residual: {burgers['max_residual']:.4e}")
    if burgers['rms_residual'] < 1.0:
        print("  Burgers equation VERIFIED (residual consistent with discretization)")
    else:
        print("  WARNING: large residual -- check discretization")

    # 3. Shock condition analysis
    print(f"\n[3/4] Analyzing shock conditions (du/dz profile)...")
    shock = analyze_shock_conditions(z_grid, u_vals, H0_vals, H0p_vals)

    intervals = shock['intervals']
    if intervals:
        min_dudz_all = min(iv['min_du_dz'] for iv in intervals)
        print(f"  {len(intervals)} inter-zero intervals analyzed")
        print(f"  Most compressive du/dz = {min_dudz_all:.4f}")
        print(f"  Key insight: du/dz > 0 near poles (repulsion)")
        print(f"  Shock risk: where du/dz is most negative between zeros")

        # Show top 5 most compressive intervals
        sorted_iv = sorted(intervals, key=lambda x: x['min_du_dz'])
        print(f"\n  Most compressive intervals:")
        print(f"  {'z_left':>10} {'z_right':>10} {'gap':>8} {'min du/dz':>12}")
        for iv in sorted_iv[:5]:
            print(f"  {iv['z_left']:>10.2f} {iv['z_right']:>10.2f} "
                  f"{iv['gap']:>8.3f} {iv['min_du_dz']:>12.4f}")

    # 4. Convexity of log|H_0| (the Cole-Hopf potential)
    print(f"\n[4/4] Studying convexity of phi(z) = -log|H_0(z)| (Cole-Hopf potential)...")
    # phi = -log|H0|, phi'' = -(H0''*H0 - H0'^2) / H0^2 = -H0''/H0 + (H0'/H0)^2
    # phi'' = u^2/4 + du_dz/2  (from u = -2H'/H)
    # If phi is convex (phi'' > 0 everywhere between zeros), the Cole-Hopf
    # solution has no caustics for t > 0, meaning zeros never collide.

    du_dz = shock['du_dz']
    phi_pp = u_vals**2 / 4 + du_dz / 2

    convex_check = phi_pp[finite_mask]
    n_negative = np.sum(convex_check < 0)
    print(f"  phi''(z) = u^2/4 + u'/2")
    print(f"  Points where phi'' < 0: {n_negative}/{len(convex_check)} "
          f"({100*n_negative/len(convex_check):.1f}%)")
    if n_negative == 0:
        print("  *** phi is CONVEX between all zeros ***")
        print("  => Cole-Hopf solution has no caustics for t > 0")
        print("  => This would PROVE zeros stay real (Lambda <= 0) if rigorous!")
    else:
        min_phi_pp = float(np.min(convex_check))
        print(f"  Min phi'' = {min_phi_pp:.6f}")
        if min_phi_pp > -0.1:
            print("  phi'' is nearly non-negative -- weak concavity regions exist")
            print("  These may vanish with finer grid / higher precision")

    # ── Plots ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # (0,0) H_0(z) with zeros
    ax = axes[0, 0]
    ax.plot(z_grid, H0_vals, 'b-', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    # Mark zeros
    zc = shock['zero_crossings']
    for zi in zc:
        ax.axvline(x=z_grid[zi], color='red', alpha=0.2, linewidth=0.5)
    ax.set_ylabel('H_0(z)')
    ax.set_title(f'H_0(z) = xi(1/2+iz) -- {len(zc)} zeros')
    ax.grid(True, alpha=0.3)

    # (0,1) Burgers field u(0,z)
    ax = axes[0, 1]
    u_clipped = np.clip(u_vals, -100, 100)
    ax.plot(z_grid, u_clipped, 'darkgreen', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    for zi in zc:
        ax.axvline(x=z_grid[zi], color='red', alpha=0.2, linewidth=0.5)
    ax.set_ylabel('u(0,z)')
    ax.set_title('Burgers velocity field u = -2 H\'/H')
    ax.set_ylim(-80, 80)
    ax.grid(True, alpha=0.3)

    # (1,0) du/dz profile
    ax = axes[1, 0]
    dudz_clipped = np.clip(du_dz, -500, 2000)
    ax.plot(z_grid, dudz_clipped, 'purple', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.7)
    for zi in zc:
        ax.axvline(x=z_grid[zi], color='red', alpha=0.2, linewidth=0.5)
    ax.set_ylabel('du/dz')
    ax.set_title('Velocity gradient (negative = compressive)')
    ax.set_ylim(-200, 500)
    ax.grid(True, alpha=0.3)

    # (1,1) phi'' = convexity of Cole-Hopf potential
    ax = axes[1, 1]
    phi_pp_clipped = np.clip(phi_pp, -100, 500)
    ax.plot(z_grid, phi_pp_clipped, 'darkorange', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.7)
    for zi in zc:
        ax.axvline(x=z_grid[zi], color='red', alpha=0.2, linewidth=0.5)
    ax.set_ylabel("phi''(z)")
    ax.set_title("Cole-Hopf potential convexity (phi'' > 0 => no shocks)")
    ax.set_ylim(-50, 200)
    ax.grid(True, alpha=0.3)

    # (2,0) Burgers equation residual
    ax = axes[2, 0]
    res_clipped = np.clip(np.abs(burgers['residual']), 0, 100)
    ax.semilogy(z_grid, res_clipped + 1e-10, 'gray', linewidth=0.5)
    ax.set_xlabel('z')
    ax.set_ylabel('|Burgers residual|')
    ax.set_title('Burgers equation verification')
    ax.grid(True, alpha=0.3)

    # (2,1) Inter-zero min du/dz vs gap size
    ax = axes[2, 1]
    if intervals:
        gaps = [iv['gap'] for iv in intervals]
        min_dudz = [iv['min_du_dz'] for iv in intervals]
        ax.scatter(gaps, min_dudz, c='blue', s=20, alpha=0.7)
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.7)
        ax.set_xlabel('Zero gap')
        ax.set_ylabel('Min du/dz in interval')
        ax.set_title('Compression strength vs gap size')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Burgers / Cole-Hopf Analysis at t=0', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('dbn_burgers_analysis.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: dbn_burgers_analysis.png")
    plt.close(fig)

    # Save data
    analysis_data = {
        'z_range': list(z_range),
        'n_points': n_points,
        'burgers_rms_residual': burgers['rms_residual'],
        'burgers_max_residual': burgers['max_residual'],
        'n_intervals': len(intervals),
        'intervals': intervals[:20],  # top 20
        'n_phi_negative': int(n_negative),
        'phi_pp_min': float(np.min(convex_check)) if len(convex_check) > 0 else None,
    }
    with open('dbn_burgers.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print("  Saved: dbn_burgers.json")

    return {
        'z_grid': z_grid,
        'u_vals': u_vals,
        'H0_vals': H0_vals,
        'burgers': burgers,
        'shock': shock,
        'phi_pp': phi_pp,
    }


if __name__ == '__main__':
    print("PART 1: Collision Time Formula Test")
    print("=" * 65)
    collision_results = run_collision_test()

    print("\n\nPART 2: Burgers / Cole-Hopf Analysis")
    print("=" * 65)
    burgers_results = run_burgers_analysis()
