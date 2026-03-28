"""Analytical derivation of the collision correction factor alpha.

The 2-body Coulomb prediction: t_coll = -g0^2/4
The N-body reality: t_coll ~ -g0^2/(4*alpha) with alpha ~ 1.88

Strategy:
  1. The gap ODE for the closest pair includes contributions from ALL zeros
  2. Model the neighbor distribution using GUE pair correlation
  3. Derive F_bg(g, z) = background Coulomb force from GUE-distributed neighbors
  4. Solve the modified ODE: dg/dt = -4/g + F_bg(g)
  5. Extract alpha and compare to measured values

Key insight: zeros aren't uniformly spaced. GUE pair correlation gives
R2(r) = 1 - (sin(pi*r)/(pi*r))^2, which has level repulsion at small r
and approaches 1 at large r. This changes the neighbor force profile.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import digamma, polygamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time


# -- GUE pair correlation --------------------------------------------------

def R2_gue(r):
    """GUE pair correlation function (unfolded spacing).

    R2(r) = 1 - (sin(pi*r)/(pi*r))^2

    This is the 2-point correlation for GUE eigenvalues after unfolding
    (dividing by mean spacing). It gives the probability density of finding
    another zero at distance r from a given zero, relative to Poisson.

    Properties:
    - R2(0) = 0 (level repulsion)
    - R2(r) -> 1 as r -> inf (uncorrelated at large distance)
    - R2(r) = 1 - sinc^2(r) where sinc(r) = sin(pi*r)/(pi*r)
    """
    r = np.asarray(r, dtype=float)
    result = np.ones_like(r)
    nonzero = r > 1e-15
    sinc = np.sin(np.pi * r[nonzero]) / (np.pi * r[nonzero])
    result[nonzero] = 1.0 - sinc**2
    result[~nonzero] = 0.0
    return result


def R2_poisson(r):
    """Poisson pair correlation (uniform random, no correlations)."""
    return np.ones_like(np.asarray(r, dtype=float))


# -- Background force from neighbors --------------------------------------

def gap_compression_from_density(g, rho_func, d=1.0, N_max=500, mode='continuous'):
    """Compute the net Coulomb compression on a gap of size g.

    Setup: closest pair at positions 0 and g on the real line.
    Other zeros distributed according to density rho(x) = R2(|x|/d)/d
    on both sides.

    The gap compression is:
    F_bg = integral over all other zeros of [2/(g-x) - 2/(0-x)] * rho(x) dx

    where the integral excludes x in [0, g] (no zeros inside the gap).

    For discrete neighbors at positions x_k:
    F_bg = sum_k [2/(g - x_k) - 2/(0 - x_k)]

    Parameters:
        g: gap size (in units of mean spacing d)
        rho_func: R2 function giving pair correlation
        d: mean spacing
        N_max: integration cutoff
        mode: 'continuous' (integrate R2) or 'discrete' (sum over lattice)
    """
    if mode == 'continuous':
        # Integrate: for neighbors to the LEFT of position 0
        # Zero at position -x (x > 0) with density R2(x/d)/d:
        # Contribution to dg/dt: 2/(g+x) - 2/x = -2g/[x(g+x)]
        def integrand_left(x):
            return (-2 * g / (x * (g + x))) * rho_func(x / d) / d

        # For neighbors to the RIGHT of position g
        # Zero at position g+x (x > 0):
        # Force on z_{j+1}: 2/(g - (g+x)) = -2/x
        # Force on z_j:     2/(0 - (g+x)) = -2/(g+x)
        # Contribution: -2/x - (-2/(g+x)) = -2/x + 2/(g+x) = -2g/[x(g+x)]
        def integrand_right(x):
            return (-2 * g / (x * (g + x))) * rho_func(x / d) / d

        # By symmetry both integrals are the same
        # Total: 2 * integral from epsilon to N_max*d of integrand
        eps = d * 0.01  # avoid singularity at x=0 (R2(0)=0 handles it for GUE)

        F_left, _ = quad(integrand_left, eps, N_max * d, limit=200)
        F_right, _ = quad(integrand_right, eps, N_max * d, limit=200)

        return F_left + F_right

    else:  # discrete: zeros at integer multiples of d
        F = 0.0
        for k in range(1, N_max + 1):
            x = k * d
            # Left neighbor at -x
            F += -2 * g / (x * (g + x))
            # Right neighbor at g + x
            F += -2 * g / (x * (g + x))
        return F


def gap_ode_with_background(t, g_arr, rho_func, d):
    """Modified gap ODE: dg/dt = +4/g + F_bg(g).

    The 2-body term is +4/g (mutual Coulomb repulsion going forward).
    F_bg < 0 is the background compression from neighbors.
    Going backward in t (t < 0), the gap shrinks toward collision.
    """
    g = g_arr[0]
    if g < 1e-12:
        return [-1e12]  # collision

    F_2body = +4.0 / g
    F_bg = gap_compression_from_density(g, rho_func, d=d, N_max=200)

    return [F_2body + F_bg]


def build_Fbg_interpolator(rho_func, d=1.0, g_max=2.0, n_grid=500):
    """Precompute F_bg on a fine grid and return interpolator.

    This avoids calling quad at every ODE step.
    """
    from scipy.interpolate import interp1d

    g_grid = np.linspace(1e-6, g_max * d, n_grid)
    F_grid = np.array([
        gap_compression_from_density(g, rho_func, d, N_max=200)
        for g in g_grid
    ])

    return interp1d(g_grid, F_grid, kind='cubic', fill_value='extrapolate')


def solve_collision_time(g0, rho_func, d=1.0, t_end_factor=3.0, F_interp=None):
    """Solve modified gap ODE to find collision time.

    Uses precomputed F_bg interpolator for speed.
    Returns t_collision, sol.
    """
    if F_interp is None:
        F_interp = build_Fbg_interpolator(rho_func, d, g_max=g0/d * 1.5 + 0.5)

    t_2body = -g0**2 / 4.0
    t_end = t_2body * t_end_factor

    def rhs(t, g_arr):
        g = g_arr[0]
        if g < 1e-12:
            return [-1e12]
        return [+4.0 / g + float(F_interp(g))]

    threshold = g0 * 0.01

    def collision_event(t, g_arr):
        return g_arr[0] - threshold

    collision_event.terminal = True
    collision_event.direction = -1

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end),
        y0=[g0],
        method='RK45',
        rtol=1e-10,
        atol=1e-12,
        events=collision_event,
        max_step=abs(t_end) / 1000,
        dense_output=True,
    )

    t_coll = None
    if sol.t_events[0].size > 0:
        t_coll = float(sol.t_events[0][0])

    return t_coll, sol


def run_alpha_derivation():
    """Full derivation: alpha from GUE pair correlation.

    1. Compute F_bg for GUE vs Poisson vs uniform at various gap sizes
    2. Solve modified gap ODE for GUE neighbors
    3. Extract alpha = t_2body / t_actual and compare to N-body ODE
    """
    print("=" * 70)
    print("COLLISION CORRECTION alpha FROM PAIR CORRELATION")
    print("=" * 70)

    d = 1.0  # mean spacing (unfolded)

    # -- Step 1: Background force profiles --
    print("\n[1/4] Background compression force vs gap size")
    print("      F_bg(g) = net Coulomb force from all neighbors on the gap")

    g_values = np.linspace(0.01, 1.5, 50) * d

    F_gue = [gap_compression_from_density(g, R2_gue, d, mode='continuous')
             for g in g_values]
    F_poi = [gap_compression_from_density(g, R2_poisson, d, mode='continuous')
             for g in g_values]
    F_disc = [gap_compression_from_density(g, R2_poisson, d, mode='discrete')
              for g in g_values]

    print(f"\n  {'g/d':>6}  {'F_gue':>12}  {'F_poisson':>12}  {'F_discrete':>12}  {'ratio G/P':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    for i in range(0, len(g_values), 5):
        g = g_values[i]
        ratio = F_gue[i] / F_poi[i] if abs(F_poi[i]) > 1e-15 else 0
        print(f"  {g/d:>6.3f}  {F_gue[i]:>12.6f}  {F_poi[i]:>12.6f}"
              f"  {F_disc[i]:>12.6f}  {ratio:>10.4f}")

    # -- Step 2: Instantaneous alpha vs gap size --
    print("\n[2/4] Instantaneous alpha(g) = |dg/dt_total| / |dg/dt_2body|")

    alpha_gue = []
    alpha_poi = []
    for i, g in enumerate(g_values):
        F_2body = -4.0 / g
        a_g = (F_2body + F_gue[i]) / F_2body
        a_p = (F_2body + F_poi[i]) / F_2body
        alpha_gue.append(a_g)
        alpha_poi.append(a_p)

    print(f"\n  {'g/d':>6}  {'alpha_GUE':>12}  {'alpha_Poisson':>14}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}")
    for i in range(0, len(g_values), 5):
        print(f"  {g_values[i]/d:>6.3f}  {alpha_gue[i]:>12.6f}  {alpha_poi[i]:>14.6f}")

    # -- Step 3: Solve modified gap ODE --
    print("\n[3/4] Solving modified gap ODE for collision times")

    # Test at the actual g0/d ratios from our N-body simulations
    all_zeros = np.load('_zeros_200.npy')

    test_cases = []
    for n_z in [25, 50, 100, 200]:
        z0 = all_zeros[:n_z]
        z_sorted = np.sort(z0)
        gaps = np.diff(z_sorted)
        g0 = np.min(gaps)
        mean_gap = np.mean(gaps)
        test_cases.append({
            'N': n_z,
            'g0': g0,
            'mean_gap': mean_gap,
            'g0_over_d': g0 / mean_gap,
        })

    # Load measured collision times from previous run
    try:
        with open('dbn_collision.json') as f:
            collision_data = json.load(f)
        measured = {r['N']: r for r in collision_data}
    except FileNotFoundError:
        measured = {}

    # Precompute F_bg interpolators (one per d value)
    print("  Precomputing F_bg interpolators...")
    d_values = set(tc['mean_gap'] for tc in test_cases)
    F_interps_gue = {}
    F_interps_poi = {}
    for d_val in d_values:
        g_max_d = max(tc['g0'] for tc in test_cases if tc['mean_gap'] == d_val) / d_val * 1.5 + 0.5
        F_interps_gue[d_val] = build_Fbg_interpolator(R2_gue, d_val, g_max=g_max_d)
        F_interps_poi[d_val] = build_Fbg_interpolator(R2_poisson, d_val, g_max=g_max_d)
    print("  Done.")

    print(f"\n  {'N':>5}  {'g0/d':>8}  {'t_2body':>12}  {'t_GUE_pred':>12}"
          f"  {'t_measured':>12}  {'alpha_pred':>10}  {'alpha_meas':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")

    results = []
    for tc in test_cases:
        g0 = tc['g0']
        d_eff = tc['mean_gap']
        t_2body = -g0**2 / 4.0

        # Solve with GUE correlation (using precomputed interpolator)
        t_gue, sol_gue = solve_collision_time(
            g0, R2_gue, d=d_eff, F_interp=F_interps_gue[d_eff])

        # Solve with Poisson (uniform) for comparison
        t_poi, sol_poi = solve_collision_time(
            g0, R2_poisson, d=d_eff, F_interp=F_interps_poi[d_eff])

        alpha_pred = t_2body / t_gue if t_gue is not None and t_gue != 0 else np.nan

        t_meas = measured.get(tc['N'], {}).get('t_actual', None)
        alpha_meas = t_2body / t_meas if t_meas is not None and t_meas != 0 else np.nan

        print(f"  {tc['N']:>5}  {tc['g0_over_d']:>8.4f}  {t_2body:>12.8f}"
              f"  {t_gue if t_gue else 'N/A':>12}"
              f"  {t_meas if t_meas else 'N/A':>12}"
              f"  {alpha_pred:>10.4f}"
              f"  {alpha_meas:>10.4f}")

        results.append({
            'N': tc['N'],
            'g0': g0,
            'mean_gap': d_eff,
            'g0_over_d': tc['g0_over_d'],
            't_2body': t_2body,
            't_gue_pred': t_gue,
            't_poisson_pred': t_poi,
            't_measured': t_meas,
            'alpha_gue': alpha_pred,
            'alpha_measured': alpha_meas,
            'sol_gue': sol_gue,
        })

    # -- Step 4: Analytical formula for alpha in the g0 << d limit --
    print("\n[4/4] Analytical approximation for alpha")
    print()
    print("  The gap ODE with GUE-correlated neighbors:")
    print("    dg/dt = -4/g + F_bg(g)")
    print()
    print("  where F_bg(g) = 2 * integral_0^inf [-2g/(x(g+x))] * R2(x/d)/d dx")
    print()
    print("  For g << d, F_bg(g) ~ -C * g / d^2 where:")

    # Compute the integral coefficient C
    # F_bg ~ -2g * 2 * integral_0^inf R2(x/d) / (x * d * x) dx  (x >> g)
    #       = -4g/d^2 * integral_0^inf R2(u) / u^2 du  (u = x/d)

    def C_integrand_gue(u):
        if u < 1e-10:
            return 0.0  # R2(0) = 0, so R2(u)/u^2 -> finite
        return R2_gue(u) / u**2

    def C_integrand_poisson(u):
        if u < 1e-10:
            return 1e10  # diverges for Poisson (R2=1, 1/u^2 -> inf)
        return 1.0 / u**2

    C_gue, _ = quad(C_integrand_gue, 0.01, 500, limit=500)
    print(f"    C_GUE = 4 * integral R2(u)/u^2 du = 4 * {C_gue:.6f} = {4*C_gue:.6f}")
    print(f"    (Poisson: diverges -- uniform spacing gives log-divergent force)")
    print()

    # With F_bg ~ -C*g/d^2, the gap ODE becomes:
    # dg/dt = -4/g - C*g/d^2
    # This is a separable ODE: g*dg / (4 + C*g^2/d^2) = -dt
    # Let u = g^2: du/(4 + C*u/d^2) = -2*dt
    # d^2/(C) * ln(4 + C*u/d^2) = -2*t + const
    #
    # At t=0: u0 = g0^2
    # At collision (u=0): 4 -> 4 + 0 = 4
    # d^2/C * ln(4 / (4 + C*g0^2/d^2)) = -2*t_coll
    # t_coll = d^2/(2C) * ln(1 + C*g0^2/(4*d^2))
    # For C*g0^2/(4*d^2) << 1: t_coll ~ d^2/(2C) * C*g0^2/(4*d^2) = g0^2/8
    # Hmm, that gives alpha=2 in the g0<<d limit... Let me recheck.

    # Actually: dg/dt = -4/g + F_bg, F_bg = -(4*C_gue/d^2)*g (to leading order)
    # Let's call beta = 4*C_gue/d^2
    # dg/dt = -4/g - beta*g
    # g dg = -(4 + beta*g^2) dt
    # integral from g0 to 0 of g dg / (4 + beta*g^2) = -t_coll
    # [1/(2*beta)] * ln(4 + beta*g^2) from g0 to 0 = -t_coll
    # [1/(2*beta)] * [ln(4) - ln(4 + beta*g0^2)] = -t_coll
    # t_coll = [1/(2*beta)] * ln(1 + beta*g0^2/4)
    # = [d^2/(8*C_gue)] * ln(1 + 4*C_gue*g0^2/d^2/4)
    # = [d^2/(8*C_gue)] * ln(1 + C_gue*g0^2/d^2)

    # 2-body: t_2body = -g0^2/4
    # alpha = t_2body / t_coll = -g0^2/4 / t_coll

    # F_bg ~ -4*C_gue * g/d^2 for g << d (leading order, GUE)
    # So dg/dt = +4/g - beta*g  where beta = 4*C_gue/d^2
    # With d=1: beta = 4*C_gue
    beta = 4 * C_gue  # (with d=1)
    print(f"  Modified gap ODE: dg/dt = +4/g - beta*g, beta = {beta:.4f}")
    print()
    print("  Separable ODE: g dg / (4 - beta*g^2) = dt")
    print("  Solution: t(g) = -[1/(2*beta)] * ln|4 - beta*g^2| + C")
    print("  With g(0)=g0 and collision at g=0:")
    print("  t_coll = [1/(2*beta)] * [ln(4 - beta*g0^2) - ln(4)]")
    print("         = [1/(2*beta)] * ln(1 - beta*g0^2/4)")
    print("  (requires beta*g0^2/4 < 1, i.e., g0 < 2/sqrt(beta))")
    print(f"  Critical gap: g_crit/d = 2/sqrt(beta) = {2/np.sqrt(beta):.4f}")
    print(f"  Above this, neighbors prevent collision entirely!")
    print()
    print("  2-body: t_coll = -g0^2/4  (from dg/dt = +4/g, going backward)")
    print()
    print(f"  {'g0/d':>8}  {'t_2body':>12}  {'t_analytic':>12}  {'alpha_analytic':>14}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*14}")

    g_crit = 2.0 / np.sqrt(beta)
    analytic_results = []
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        g0 = ratio  # d=1
        t_2body = -g0**2 / 4.0  # backward, so negative

        if beta * g0**2 / 4 < 1:
            t_analytic = (1.0 / (2 * beta)) * np.log(1 - beta * g0**2 / 4)
            alpha_an = t_2body / t_analytic if t_analytic != 0 else np.nan
        else:
            t_analytic = np.nan
            alpha_an = np.nan  # no collision possible

        print(f"  {ratio:>8.3f}  {t_2body:>12.8f}  "
              f"{'N/A (>crit)' if np.isnan(t_analytic) else f'{t_analytic:>12.8f}':>12}"
              f"  {alpha_an:>14.6f}" if not np.isnan(alpha_an) else
              f"  {ratio:>8.3f}  {t_2body:>12.8f}  {'N/A (>crit)':>12}  {'inf':>14}")
        analytic_results.append({
            'g0_over_d': ratio,
            't_2body': t_2body,
            't_analytic': float(t_analytic) if not np.isnan(t_analytic) else None,
            'alpha': float(alpha_an) if not np.isnan(alpha_an) else None,
        })

    # -- Plots --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Background force vs gap for GUE, Poisson, discrete
    ax = axes[0, 0]
    g_norm = g_values / d
    ax.plot(g_norm, F_gue, 'b-', linewidth=2, label='GUE R2')
    ax.plot(g_norm, F_poi, 'r--', linewidth=1.5, label='Poisson')
    ax.plot(g_norm, F_disc, 'g:', linewidth=1.5, label='Uniform lattice')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('g / d (gap / mean spacing)')
    ax.set_ylabel('F_bg (background force)')
    ax.set_title('Background compression force on closest pair')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) Instantaneous alpha vs gap
    ax = axes[0, 1]
    ax.plot(g_norm, alpha_gue, 'b-', linewidth=2, label='GUE')
    ax.plot(g_norm, alpha_poi, 'r--', linewidth=1.5, label='Poisson')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    # Mark measured values
    for tc in test_cases:
        if tc['N'] in measured:
            r = tc['g0_over_d']
            am = measured[tc['N']]['ratio']
            if am and not np.isnan(am):
                ax.plot(r, 1.0/am, 'ko', markersize=8)
                ax.annotate(f"N={tc['N']}", (r, 1.0/am),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('g / d')
    ax.set_ylabel('Instantaneous alpha')
    ax.set_title('Alpha = |dg/dt_total| / |dg/dt_2body|')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0) Gap trajectory: g^2 vs t for different models
    ax = axes[1, 0]
    for r in results:
        sol = r['sol_gue']
        if sol is not None:
            t_dense = np.linspace(0, sol.t[-1], 500)
            g_dense = sol.sol(t_dense)[0]
            ax.plot(t_dense, g_dense**2, linewidth=1.5,
                    label=f"N={r['N']}: GUE pred")
            # 2-body line
            g0 = r['g0']
            t_line = np.array([r['t_2body'], 0])
            ax.plot(t_line, [0, g0**2], '--', alpha=0.5, linewidth=1)
            # Mark measured collision
            if r['t_measured']:
                ax.axvline(x=r['t_measured'], linestyle=':', alpha=0.3)

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('g^2')
    ax.set_title('Gap^2 trajectories: GUE prediction (solid) vs 2-body (dashed)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Predicted vs measured alpha
    ax = axes[1, 1]
    ns = [r['N'] for r in results]
    a_pred = [r['alpha_gue'] for r in results]
    a_meas = [r['alpha_measured'] for r in results]

    ax.plot(ns, a_pred, 'bo-', markersize=8, linewidth=2, label='GUE prediction')
    ax.plot(ns, a_meas, 'rs-', markersize=8, linewidth=2, label='N-body ODE (measured)')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='2-body (alpha=1)')

    # Analytical formula prediction for each N
    g0d_values = [tc['g0_over_d'] for tc in test_cases]
    a_analytic = []
    for gd in g0d_values:
        t_2b = -gd**2 / 4
        t_an = -(1.0 / (2 * beta)) * np.log(1 + beta * gd**2 / 4)
        a_analytic.append(t_2b / t_an if t_an != 0 else np.nan)
    ax.plot(ns, a_analytic, 'g^-', markersize=8, linewidth=1.5,
            label='Analytical (small-g approx)')

    ax.set_xlabel('N (number of zeros)')
    ax.set_ylabel('alpha')
    ax.set_title('Collision correction: GUE prediction vs measurement')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Deriving alpha from GUE Pair Correlation', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('dbn_alpha_from_gue.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: dbn_alpha_from_gue.png")
    plt.close(fig)

    # Save
    save_data = {
        'C_gue': C_gue,
        'beta': beta,
        'results': [{k: v for k, v in r.items() if k != 'sol_gue'}
                    for r in results],
        'analytic': analytic_results,
    }
    with open('dbn_alpha_derivation.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved: dbn_alpha_derivation.json")

    return results


if __name__ == '__main__':
    run_alpha_derivation()
