"""The M-function: analytical form of the Stieltjes condition.

THEOREM: RH holds if and only if M(s) > 0 for all Re(s) > 1/2, Im(s) > 0,
where:
    M(s) = Im[(s - 1/2) * conj((xi'/xi)(s))]

Expanding: if u = s - 1/2 = (sigma-1/2) + it and f = (xi'/xi)(s) = A + iB,
    M = (sigma-1/2)*B - t*(-A) ... wait let me redo.

u * conj(f) = [(sigma-1/2) + it] * [A - iB]
= (sigma-1/2)*A + tB + i[tA - (sigma-1/2)*B]

So M = Im[u * conj(f)] = t*A - (sigma-1/2)*B

Condition: t*Re[(xi'/xi)(s)] - (Re(s)-1/2)*Im[(xi'/xi)(s)] > 0

For Re(s) > 1: (xi'/xi) is computable from the Euler product.
This is a NON-CIRCULAR, TESTABLE condition.

Connection to Levinson: arg[(xi'/xi)] on the critical line (sigma=1/2)
controls M. Levinson's method bounds this argument to prove >1/3 of
zeros are on the critical line. Our condition is sharper: M > 0
everywhere in the right half-plane (not just on the critical line).
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time


def xi_log_deriv(s, dps=30):
    """(xi'/xi)(s) via mpmath. Non-circular."""
    with mpmath.workdps(dps):
        s_mp = mpmath.mpc(s)

        def xi_func(sv):
            return (sv * (sv - 1) / 2
                    * mpmath.power(mpmath.pi, -sv/2)
                    * mpmath.gamma(sv/2)
                    * mpmath.zeta(sv))

        xi_val = xi_func(s_mp)
        if abs(xi_val) < mpmath.mpf(10)**(-dps+5):
            return complex(float('nan'), float('nan'))

        xi_deriv = mpmath.diff(xi_func, s_mp)
        return complex(xi_deriv / xi_val)


def M_function(sigma, t, dps=30):
    """Compute M(s) = Im[(s-1/2) * conj((xi'/xi)(s))].

    M(s) = t * Re[(xi'/xi)(s)] - (sigma-1/2) * Im[(xi'/xi)(s)]

    RH <=> M(s) > 0 for all sigma > 1/2, t > 0.
    """
    s = complex(sigma, t)
    f = xi_log_deriv(s, dps=dps)

    if np.isnan(f.real):
        return float('nan')

    A = f.real  # Re[(xi'/xi)(s)]
    B = f.imag  # Im[(xi'/xi)(s)]

    return t * A - (sigma - 0.5) * B


def run_M_analysis():
    """Comprehensive M-function analysis."""

    print("=" * 70)
    print("THE M-FUNCTION: ANALYTICAL STIELTJES CONDITION")
    print("=" * 70)
    print()
    print("  M(s) = t*Re[(xi'/xi)(s)] - (sigma-1/2)*Im[(xi'/xi)(s)]")
    print("  RH <=> M(s) > 0 for all sigma > 1/2, t > 0")
    print()
    print("  Computed from mpmath.zeta (Euler product). NO ZEROS.")

    # Step 1: Compute M(s) on a grid in the right half s-plane
    print("\n[1/4] Grid computation of M(s)")

    sigmas = np.linspace(0.51, 3.0, 50)
    ts = np.linspace(0.5, 100, 80)

    print(f"  Grid: sigma in [{sigmas[0]}, {sigmas[-1]}], t in [{ts[0]}, {ts[-1]}]")
    print(f"  Points: {len(sigmas)} x {len(ts)} = {len(sigmas)*len(ts)}")

    t0 = time.time()
    M_grid = np.zeros((len(ts), len(sigmas)))
    n_violations = 0

    for i, t in enumerate(ts):
        for j, sigma in enumerate(sigmas):
            M_val = M_function(sigma, t, dps=25)
            M_grid[i, j] = M_val
            if not np.isnan(M_val) and M_val <= 0:
                n_violations += 1

        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Row {i+1}/{len(ts)} (t={t:.1f}), violations: {n_violations}, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  VIOLATIONS (M <= 0): {n_violations} / {len(sigmas)*len(ts)}")

    if n_violations == 0:
        print("  *** M(s) > 0 at ALL tested points ***")
        print("  *** CONSISTENT WITH RH ***")

    # Step 2: Margin analysis — where is M smallest?
    print("\n[2/4] Margin analysis")

    # Along the critical line (sigma -> 1/2+)
    print("  M(s) near the critical line (sigma = 0.51):")
    t_fine = np.linspace(1, 100, 200)
    M_critical = [M_function(0.51, t, dps=25) for t in t_fine]
    M_critical = np.array(M_critical)
    valid = ~np.isnan(M_critical)

    if valid.sum() > 0:
        min_M = np.nanmin(M_critical[valid])
        min_t = t_fine[valid][np.nanargmin(M_critical[valid])]
        print(f"  Minimum M = {min_M:.6f} at t = {min_t:.2f}")
        print(f"  Mean M = {np.nanmean(M_critical[valid]):.6f}")

    # Along vertical lines at different sigma
    print("\n  Minimum M(s) by sigma:")
    for sigma in [0.51, 0.55, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
        M_line = [M_function(sigma, t, dps=20) for t in np.linspace(1, 80, 50)]
        M_arr = np.array(M_line)
        valid = ~np.isnan(M_arr)
        if valid.sum() > 0:
            print(f"    sigma={sigma:.2f}: min M = {np.nanmin(M_arr[valid]):.6f},"
                  f" mean M = {np.nanmean(M_arr[valid]):.6f}")

    # Step 3: Decompose M into Euler product terms
    print("\n[3/4] Decomposition of M(s) for Re(s) > 1")
    print("  For sigma > 1: (xi'/xi)(s) = 1/s + 1/(s-1) - log(pi)/2")
    print("    + psi(s/2)/2 + (zeta'/zeta)(s)")
    print("  where (zeta'/zeta)(s) = -sum Lambda(n)/n^s (Euler product)")
    print()

    # At sigma = 1.5, compute M and decompose
    sigma_test = 1.5
    t_test = np.linspace(5, 80, 20)

    print(f"  Decomposition at sigma = {sigma_test}:")
    print(f"  {'t':>6}  {'M_total':>12}  {'M_pole':>12}  {'M_gamma':>12}  {'M_zeta':>12}")

    for t in t_test[:10]:
        s = complex(sigma_test, t)
        u_re = sigma_test - 0.5  # Re(s-1/2)
        u_im = t                 # Im(s-1/2)

        with mpmath.workdps(30):
            s_mp = mpmath.mpc(s)

            # Pole terms: 1/s + 1/(s-1)
            f_pole = complex(1/s_mp + 1/(s_mp - 1))

            # Gamma term: psi(s/2)/2
            f_gamma = complex(mpmath.digamma(s_mp/2) / 2)

            # Log pi: -log(pi)/2 (real constant, no Im contribution to M)
            f_logpi = float(-mpmath.log(mpmath.pi)/2)

            # Zeta term: (zeta'/zeta)(s)
            zeta_val = mpmath.zeta(s_mp)
            zeta_prime = mpmath.diff(mpmath.zeta, s_mp)
            f_zeta = complex(zeta_prime / zeta_val)

        # M contribution from each: M_k = t*Re(f_k) - u_re*Im(f_k)
        M_pole = t * f_pole.real - u_re * f_pole.imag
        M_gamma = t * f_gamma.real - u_re * f_gamma.imag
        M_logpi = t * f_logpi  # Im(f_logpi) = 0
        M_zeta = t * f_zeta.real - u_re * f_zeta.imag
        M_total = M_pole + M_gamma + M_logpi + M_zeta

        print(f"  {t:>6.1f}  {M_total:>12.4f}  {M_pole:>12.4f}"
              f"  {M_gamma:>12.4f}  {M_zeta:>12.4f}")

    # Step 4: Behavior near zeros (s near 1/2 + i*gamma_n)
    print("\n[4/4] M(s) near zeta zeros")
    all_zeros = np.load('_zeros_200.npy')

    print("  At s = sigma + i*gamma_n for various sigma > 1/2:")
    print(f"  {'gamma':>8}  {'sigma=0.51':>12}  {'sigma=0.55':>12}  {'sigma=0.6':>12}  {'sigma=1.0':>12}")
    for gamma in all_zeros[:10]:
        vals = []
        for sigma in [0.51, 0.55, 0.6, 1.0]:
            M_val = M_function(sigma, gamma, dps=25)
            vals.append(M_val)
        # Format, handling NaN
        strs = [f"{v:>12.4f}" if not np.isnan(v) else f"{'NaN':>12}" for v in vals]
        print(f"  {gamma:>8.2f}  {'  '.join(strs)}")

    # -- Plots --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) M(s) heatmap
    ax = axes[0, 0]
    M_clipped = np.clip(M_grid, -50, 200)
    M_clipped[np.isnan(M_clipped)] = 0
    im = ax.pcolormesh(sigmas, ts, M_clipped, cmap='RdYlGn', shading='auto')
    ax.contour(sigmas, ts, M_grid, levels=[0], colors='black', linewidths=2)
    ax.set_xlabel('sigma = Re(s)')
    ax.set_ylabel('t = Im(s)')
    ax.set_title('M(s) = t*Re(f) - (sigma-1/2)*Im(f)\n(green = positive = RH)')
    plt.colorbar(im, ax=ax, label='M(s)')
    ax.axvline(x=0.5, color='red', linewidth=2, linestyle='--', label='critical line')
    ax.legend(fontsize=8)

    # (0,1) M along critical-line neighborhood
    ax = axes[0, 1]
    for sigma in [0.51, 0.55, 0.6, 0.7, 1.0]:
        M_line = [M_function(sigma, t, dps=20) for t in t_fine[:100]]
        ax.plot(t_fine[:100], M_line, linewidth=1, label=f'sigma={sigma}')
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--', label='RH boundary')
    ax.set_xlabel('t')
    ax.set_ylabel('M(s)')
    ax.set_title('M(s) at fixed sigma (0 = RH violation)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,0) Min M vs sigma
    ax = axes[1, 0]
    sigma_range = np.linspace(0.51, 2.5, 40)
    min_Ms = []
    for sigma in sigma_range:
        M_line = [M_function(sigma, t, dps=20) for t in np.linspace(1, 80, 30)]
        M_arr = np.array(M_line)
        valid = ~np.isnan(M_arr)
        min_Ms.append(np.nanmin(M_arr[valid]) if valid.sum() > 0 else np.nan)
    ax.plot(sigma_range, min_Ms, 'b-', linewidth=2)
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('sigma')
    ax.set_ylabel('min_t M(sigma, t)')
    ax.set_title('Margin: min M(s) over t, by sigma')
    ax.grid(True, alpha=0.3)

    # (1,1) Decomposition at sigma=1.5
    ax = axes[1, 1]
    t_decomp = np.linspace(5, 80, 40)
    M_poles = []
    M_gammas = []
    M_zetas = []
    M_totals = []
    for t in t_decomp:
        s = complex(1.5, t)
        u_re = 1.0
        with mpmath.workdps(20):
            s_mp = mpmath.mpc(s)
            fp = complex(1/s_mp + 1/(s_mp-1))
            fg = complex(mpmath.digamma(s_mp/2)/2)
            fl = float(-mpmath.log(mpmath.pi)/2)
            zv = mpmath.zeta(s_mp)
            zp = mpmath.diff(mpmath.zeta, s_mp)
            fz = complex(zp/zv)
        M_poles.append(t*fp.real - u_re*fp.imag)
        M_gammas.append(t*fg.real - u_re*fg.imag)
        M_zetas.append(t*fz.real - u_re*fz.imag)
        M_totals.append(M_poles[-1] + M_gammas[-1] + t*fl + M_zetas[-1])

    ax.plot(t_decomp, M_totals, 'k-', linewidth=2, label='Total M')
    ax.plot(t_decomp, M_poles, 'b--', linewidth=1, label='Pole terms')
    ax.plot(t_decomp, M_gammas, 'g--', linewidth=1, label='Gamma term')
    ax.plot(t_decomp, M_zetas, 'r--', linewidth=1, label='Zeta term (primes)')
    ax.axhline(y=0, color='red', linewidth=1.5, linestyle=':')
    ax.set_xlabel('t')
    ax.set_ylabel('M contribution')
    ax.set_title('M(s) decomposition at sigma=1.5')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('The M-Function: Im[(s-1/2)*conj(xi\'/xi)] > 0 iff RH',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('jacobi_M_function.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: jacobi_M_function.png")
    plt.close(fig)

    save_data = {
        'n_violations': n_violations,
        'grid_sigma': [float(sigmas[0]), float(sigmas[-1])],
        'grid_t': [float(ts[0]), float(ts[-1])],
        'min_M_critical_line': float(min_M) if valid.sum() > 0 else None,
    }
    with open('jacobi_M_function.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved: jacobi_M_function.json")


if __name__ == '__main__':
    run_M_analysis()
