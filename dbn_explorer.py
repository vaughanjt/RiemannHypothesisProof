"""De Bruijn-Newman zero flow explorer.

Computes H_t(z) for the de Bruijn-Newman family and tracks how zeta zeros
move under the Coulomb ODE as t varies. At t=0, H_0(z) = xi(1/2 + iz) has
real zeros iff RH is true. For t >= Lambda (the dBN constant, 0 <= Lambda <= 0.2),
all zeros are real.

Strategy:
  - H_0(z) = xi(1/2 + iz) computed directly via mpmath (exact)
  - H_t(z) computed via FFT heat evolution: hat{H_t}(k) = exp(-tk^2) hat{H_0}(k)
  - Zero dynamics: dz_j/dt = -2 sum_{k!=j} 1/(z_j - z_k)  (Coulomb ODE)
  - Initial zeros from pre-computed zeta zeros (_zeros_200.npy)

References:
  - de Bruijn (1950), Newman (1976)
  - Polymath 15 / Tao (2018): Lambda <= 0.22
  - Rodgers-Tao (2020): Lambda >= 0
"""

import numpy as np
import mpmath
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time


# -- H_0 via xi function (exact) ----------------------------------------

def xi_function(z, dps=30):
    """Compute xi(1/2 + iz) using mpmath.

    This is H_0(z) -- the de Bruijn-Newman function at t=0.
    Returns a real value for real z.
    """
    with mpmath.workdps(dps):
        s = mpmath.mpc('0.5', str(z))
        val = (s * (s - 1) / 2
               * mpmath.power(mpmath.pi, -s / 2)
               * mpmath.gamma(s / 2)
               * mpmath.zeta(s))
        return float(mpmath.re(val))


def compute_H0_grid(z_grid, dps=20):
    """Evaluate H_0(z) = xi(1/2+iz) on a grid.

    Args:
        z_grid: 1D array of real z values.
        dps: Working precision digits.

    Returns:
        1D array of H_0 values.
    """
    return np.array([xi_function(z, dps=dps) for z in z_grid])


# -- H_t via FFT heat evolution -----------------------------------------

def compute_Ht_fft(z_grid, H0_vals, t, pad_factor=4):
    """Compute H_t(z) by FFT-based heat evolution.

    H_t satisfies the heat equation dH/dt = d^2H/dz^2.
    In Fourier space: hat{H_t}(k) = exp(-t*k^2) * hat{H_0}(k).

    Args:
        z_grid: Uniformly spaced z values.
        H0_vals: H_0(z) evaluated on z_grid.
        t: Heat parameter (positive = smoothing, negative = sharpening).
        pad_factor: Zero-padding multiplier for FFT accuracy.

    Returns:
        H_t values on z_grid.
    """
    N = len(z_grid)
    dz = z_grid[1] - z_grid[0]

    # Zero-pad for better spectral resolution
    N_pad = N * pad_factor
    H0_padded = np.zeros(N_pad)
    H0_padded[:N] = H0_vals

    # FFT
    H0_hat = np.fft.fft(H0_padded)

    # Frequency array
    k = np.fft.fftfreq(N_pad, d=dz) * 2 * np.pi

    # Heat kernel in Fourier space (forward heat: damp high freq for t > 0)
    Ht_hat = H0_hat * np.exp(-t * k**2)
    # Note: t > 0 = smoothing (zeros repel), t < 0 = sharpening (zeros attract)

    # Inverse FFT and extract original range
    Ht_padded = np.fft.ifft(Ht_hat).real
    return Ht_padded[:N]


# -- Zero finding --------------------------------------------------------

def find_Ht_zeros_fft(z_grid, Ht_vals):
    """Find zeros of H_t by detecting sign changes in grid values.

    Uses Brent's method to refine each sign-change bracket.

    Returns:
        Sorted array of zero positions.
    """
    zeros = []
    for i in range(len(Ht_vals) - 1):
        if Ht_vals[i] * Ht_vals[i + 1] < 0:
            # Linear interpolation for initial guess
            frac = abs(Ht_vals[i]) / (abs(Ht_vals[i]) + abs(Ht_vals[i + 1]))
            zeros.append(z_grid[i] + frac * (z_grid[i + 1] - z_grid[i]))
    return np.array(sorted(zeros))


# -- Coulomb ODE for zero dynamics ---------------------------------------

def coulomb_rhs(t_param, z_flat):
    """Vectorized Coulomb ODE: dz_j/dt = +2 sum_{k!=j} 1/(z_j - z_k).

    Standard dBN convention (forward heat equation):
    increasing t = smoothing = repulsion between zeros.
    Uses matrix operations for speed.
    """
    n = len(z_flat)
    z = z_flat[:, None]  # (n, 1)
    diffs = z - z_flat[None, :]  # (n, n)

    # Mask diagonal
    np.fill_diagonal(diffs, np.inf)

    # Regularize near-collisions
    mask = np.abs(diffs) < 1e-12
    diffs[mask] = np.sign(diffs[mask] + 1e-30) * 1e-12

    inv_diffs = 1.0 / diffs
    np.fill_diagonal(inv_diffs, 0.0)

    return +2.0 * np.sum(inv_diffs, axis=1)


def evolve_zeros_forward(z0, t_start=0.0, t_end=0.2, n_steps=200,
                         method='DOP853', rtol=1e-6, atol=1e-8):
    """Evolve zeros forward in t (stable: repulsion).

    Returns:
        t_values, z_trajectories (n_steps, n_zeros).
    """
    t_eval = np.linspace(t_start, t_end, n_steps)

    sol = solve_ivp(
        coulomb_rhs,
        t_span=(t_start, t_end),
        y0=z0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        print(f"  ODE warning: {sol.message}")

    return sol.t, sol.y.T


def evolve_zeros_backward(z0, t_start=0.0, t_end=-0.05, n_steps=200,
                          method='DOP853', rtol=1e-6, atol=1e-8):
    """Evolve zeros backward in t (unstable: attraction, collisions).

    Stops at near-collisions via event detection.

    Returns:
        t_values, z_trajectories (may be shorter than n_steps).
    """
    min_gap = np.min(np.diff(np.sort(z0))) * 0.05  # 5% of smallest gap

    def collision_event(t, z_flat):
        z_sorted = np.sort(z_flat)
        return np.min(np.diff(z_sorted)) - min_gap

    collision_event.terminal = True
    collision_event.direction = -1

    t_eval = np.linspace(t_start, t_end, n_steps)

    sol = solve_ivp(
        coulomb_rhs,
        t_span=(t_start, t_end),
        y0=z0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        events=collision_event,
    )

    if sol.t_events[0].size > 0:
        print(f"  Collision at t = {sol.t_events[0][0]:.6f}")

    return sol.t, sol.y.T


# -- Spacing statistics --------------------------------------------------

def normalized_spacings(zeros):
    """Normalized nearest-neighbor spacings (divide by mean)."""
    z = np.sort(zeros)
    gaps = np.diff(z)
    mean_gap = np.mean(gaps)
    return gaps / mean_gap if mean_gap > 0 else gaps


def wigner_surmise_gue(s):
    """GUE Wigner surmise: P(s) = (32/pi^2) s^2 exp(-4s^2/pi)."""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def spacing_statistics(zeros):
    """Spacing stats: mean, variance, chi-squared vs Wigner surmise."""
    spacings = normalized_spacings(zeros)
    if len(spacings) < 5:
        return {"n": len(spacings), "mean": np.nan, "var": np.nan}

    bins = np.linspace(0, 4, 41)
    centers = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(spacings, bins=bins, density=True)
    wigner = wigner_surmise_gue(centers)

    mask = wigner > 0.01
    chi2 = float(np.sum((hist[mask] - wigner[mask])**2 / wigner[mask])) if mask.sum() > 0 else np.nan

    return {
        "n": len(spacings),
        "mean": float(np.mean(spacings)),
        "var": float(np.var(spacings)),
        "skew": float(np.mean(((spacings - np.mean(spacings)) / max(np.std(spacings), 1e-15))**3)),
        "chi2_wigner": chi2,
    }


# -- Visualization -------------------------------------------------------

def plot_H0_with_zeros(z_range=(0, 80), n_points=2000, dps=15):
    """Plot H_0(z) = xi(1/2+iz) and mark zeta zeros."""
    z_grid = np.linspace(z_range[0], z_range[1], n_points)

    print("  Computing xi(1/2+iz) on grid...")
    H0 = compute_H0_grid(z_grid, dps=dps)

    # Find zeros from sign changes
    h0_zeros = find_Ht_zeros_fft(z_grid, H0)

    # Load known zeta zeros for comparison
    known_zeros = np.load('_zeros_200.npy')
    known_in_range = known_zeros[(known_zeros >= z_range[0]) & (known_zeros <= z_range[1])]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(z_grid, H0, 'b-', linewidth=0.8, label='H_0(z) = xi(1/2+iz)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Mark detected zeros
    for hz in h0_zeros:
        ax.axvline(x=hz, color='red', alpha=0.3, linewidth=0.5)

    # Mark known zeta zeros
    for kz in known_in_range:
        ax.plot(kz, 0, 'go', markersize=4, alpha=0.7)

    ax.set_xlabel('z')
    ax.set_ylabel('H_0(z)')
    ax.set_title(f'H_0(z) = xi(1/2+iz) -- {len(h0_zeros)} zeros detected (red), '
                 f'{len(known_in_range)} known zeta zeros (green)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, H0, z_grid


def plot_Ht_profiles(z_grid, H0, t_values):
    """Plot H_t(z) for several t values, showing heat smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t_values)))

    for t_val, color in zip(t_values, colors):
        if t_val == 0:
            Ht = H0
        else:
            Ht = compute_Ht_fft(z_grid, H0, t_val)
        ax.plot(z_grid, Ht, color=color, linewidth=1.0, label=f't = {t_val:.3f}')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('z')
    ax.set_ylabel('H_t(z)')
    ax.set_title('De Bruijn-Newman H_t(z) for various t')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_zero_trajectories(t_values, z_trajectories, title=None):
    """Plot zero trajectories as lines in the (t, z) plane."""
    n_zeros = z_trajectories.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, n_zeros))
    for j in range(n_zeros):
        ax.plot(t_values, z_trajectories[:, j], color=colors[j],
                linewidth=0.8, alpha=0.8)

    ax.set_xlabel('t (heat parameter)')
    ax.set_ylabel('z (zero position)')
    ax.set_title(title or f'dBN Zero Flow: {n_zeros} zeros')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='t=0 (RH)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_gap_evolution(t_values, z_trajectories):
    """Plot minimum and mean gap between adjacent zeros vs t."""
    min_gaps = []
    mean_gaps = []
    for i in range(len(t_values)):
        gaps = np.diff(np.sort(z_trajectories[i]))
        min_gaps.append(np.min(gaps))
        mean_gaps.append(np.mean(gaps))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(t_values, min_gaps, 'r.-', linewidth=1, markersize=2)
    ax1.set_ylabel('Min gap')
    ax1.set_title('Gap Evolution Under Coulomb Flow')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax2.plot(t_values, mean_gaps, 'b.-', linewidth=1, markersize=2)
    ax2.set_xlabel('t')
    ax2.set_ylabel('Mean gap')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    return fig


def plot_spacing_evolution(t_values, z_trajectories, n_samples=8):
    """Spacing histograms at several time slices + Wigner overlay."""
    n_times = len(t_values)
    indices = np.linspace(0, n_times - 1, n_samples, dtype=int)

    nrows = 2
    ncols = n_samples // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), sharey=True)
    axes = axes.flatten()

    s_theory = np.linspace(0, 4, 200)
    w_theory = wigner_surmise_gue(s_theory)

    for ax_idx, time_idx in enumerate(indices):
        ax = axes[ax_idx]
        t_val = t_values[time_idx]
        zeros_at_t = z_trajectories[time_idx]

        spacings = normalized_spacings(zeros_at_t)
        stats = spacing_statistics(zeros_at_t)

        ax.hist(spacings, bins=25, range=(0, 4), density=True,
                alpha=0.6, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.plot(s_theory, w_theory, 'r--', linewidth=1.5, alpha=0.8)
        chi2_str = f'{stats["chi2_wigner"]:.2f}' if not np.isnan(stats.get("chi2_wigner", np.nan)) else "N/A"
        ax.set_title(f't={t_val:.4f}\nchi2={chi2_str}', fontsize=9)
        ax.set_xlim(0, 4)
        if ax_idx % ncols == 0:
            ax.set_ylabel('Density')
        if ax_idx >= ncols:
            ax.set_xlabel('s')

    fig.suptitle('Spacing Distribution Evolution with t', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_statistics_vs_t(t_values, z_trajectories):
    """Chi-squared vs Wigner and spacing variance as function of t."""
    chi2_vals = []
    var_vals = []
    for i in range(len(t_values)):
        stats = spacing_statistics(z_trajectories[i])
        chi2_vals.append(stats.get("chi2_wigner", np.nan))
        var_vals.append(stats.get("var", np.nan))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_values, chi2_vals, 'b.-', linewidth=1, markersize=3)
    ax1.set_ylabel('Chi-squared vs Wigner')
    ax1.set_title('GUE Departure vs Heat Parameter t')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='t=0')
    ax1.legend()

    ax2.plot(t_values, var_vals, 'g.-', linewidth=1, markersize=3)
    ax2.set_xlabel('t')
    ax2.set_ylabel('Spacing variance')
    ax2.axhline(y=0.178, color='orange', linestyle=':', label='GUE var ~ 0.178')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_pair_correlation_snapshots(t_values, z_trajectories, n_samples=4):
    """Pair correlation R_2(x) at key time slices.

    R_2 is a much more sensitive diagnostic than the spacing histogram
    for detecting departure from GUE.
    """
    n_times = len(t_values)
    indices = np.linspace(0, n_times - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(1, n_samples, figsize=(16, 4), sharey=True)
    x_theory = np.linspace(0.01, 3, 200)
    gue_r2 = 1 - (np.sin(np.pi * x_theory) / (np.pi * x_theory))**2

    for ax_idx, time_idx in enumerate(indices):
        ax = axes[ax_idx]
        t_val = t_values[time_idx]
        zeros_at_t = np.sort(z_trajectories[time_idx])
        spacings = normalized_spacings(zeros_at_t)

        # Empirical pair correlation via spacing histogram
        bins = np.linspace(0, 3, 60)
        centers = (bins[:-1] + bins[1:]) / 2
        hist, _ = np.histogram(spacings, bins=bins, density=True)

        ax.bar(centers, hist, width=centers[1]-centers[0], alpha=0.5,
               color='steelblue', label='Empirical')
        ax.plot(x_theory, gue_r2 * np.max(hist) / np.max(gue_r2),
                'r--', linewidth=1.5, label='GUE shape')
        ax.set_title(f't = {t_val:.4f}', fontsize=10)
        ax.set_xlabel('s')
        ax.set_xlim(0, 3)
        if ax_idx == 0:
            ax.set_ylabel('Density')

    axes[-1].legend(fontsize=8)
    fig.suptitle('Pair Correlation Evolution', fontsize=12)
    plt.tight_layout()
    return fig


# -- Main exploration workflow -------------------------------------------

def run_exploration(n_zeros=40, t_forward=0.1, t_backward=-0.02,
                    n_steps=300, save_plots=True):
    """Full dBN zero flow exploration.

    1. Validate H_0 zeros match zeta zeros
    2. Evolve zeros forward (stable) and backward (toward collisions)
    3. Track spacing statistics across t
    4. Generate all diagnostic plots
    """
    print("=" * 60)
    print("DE BRUIJN-NEWMAN ZERO FLOW EXPLORER")
    print("=" * 60)

    # 1. Load zeros
    print(f"\n[1/6] Loading first {n_zeros} zeta zeros...")
    all_zeros = np.load('_zeros_200.npy')
    z0 = all_zeros[:n_zeros].copy()
    print(f"  Range: [{z0[0]:.2f}, {z0[-1]:.2f}]")
    print(f"  Mean spacing: {np.mean(np.diff(z0)):.4f}")

    # 2. Validate H_0
    print(f"\n[2/6] Validating H_0(z) = xi(1/2+iz) at known zeros...")
    for j in [0, 1, 2, 4, 9]:
        if j < n_zeros:
            val = xi_function(z0[j], dps=25)
            print(f"  H_0({z0[j]:.6f}) = {val:.4e}  {'OK' if abs(val) < 1e-6 else 'NONZERO'}")

    # 3. Plot H_0 and find its zeros
    print(f"\n[3/6] Computing H_0(z) profile and finding zeros...")
    fig_h0, H0_vals, z_grid = plot_H0_with_zeros(z_range=(0, 80), n_points=400, dps=15)
    if save_plots:
        fig_h0.savefig('dbn_H0_profile.png', dpi=150, bbox_inches='tight')
        print("  Saved: dbn_H0_profile.png")
    plt.close(fig_h0)

    # H_t profiles at different t
    print("  Computing H_t profiles...")
    t_samples = [0.0, 0.01, 0.02, 0.05, 0.1]
    fig_ht = plot_Ht_profiles(z_grid, H0_vals, t_samples)
    if save_plots:
        fig_ht.savefig('dbn_Ht_profiles.png', dpi=150, bbox_inches='tight')
        print("  Saved: dbn_Ht_profiles.png")
    plt.close(fig_ht)

    # 4. Forward evolution
    print(f"\n[4/6] Evolving {n_zeros} zeros forward: t = 0 -> {t_forward}...")
    t0_clock = time.time()
    t_fwd, z_fwd = evolve_zeros_forward(z0, t_start=0.0, t_end=t_forward,
                                         n_steps=n_steps)
    dt_fwd = time.time() - t0_clock
    print(f"  Done in {dt_fwd:.1f}s, {len(t_fwd)} steps")
    final_gaps = np.diff(np.sort(z_fwd[-1]))
    print(f"  Final min gap: {np.min(final_gaps):.4f}")
    print(f"  Spread: [{z_fwd[-1].min():.2f}, {z_fwd[-1].max():.2f}]")

    # 5. Backward evolution
    print(f"\n[5/6] Evolving {n_zeros} zeros backward: t = 0 -> {t_backward}...")
    t0_clock = time.time()
    t_bwd, z_bwd = evolve_zeros_backward(z0, t_start=0.0, t_end=t_backward,
                                          n_steps=n_steps)
    dt_bwd = time.time() - t0_clock
    print(f"  Done in {dt_bwd:.1f}s, {len(t_bwd)} steps")
    if len(t_bwd) < n_steps:
        print(f"  Stopped early at t = {t_bwd[-1]:.6f}")
    bwd_gaps = np.diff(np.sort(z_bwd[-1]))
    print(f"  Final min gap: {np.min(bwd_gaps):.6f}")

    # 6. Combine and analyze
    print(f"\n[6/6] Generating trajectory and statistics plots...")

    # Stitch backward (reversed) + forward
    t_full = np.concatenate([t_bwd[::-1], t_fwd[1:]])
    z_full = np.concatenate([z_bwd[::-1], z_fwd[1:]], axis=0)

    # Statistics at each time step
    stats_log = []
    for i in range(len(t_full)):
        s = spacing_statistics(z_full[i])
        s['t'] = float(t_full[i])
        stats_log.append(s)

    # Plot A: Zero trajectories
    fig1 = plot_zero_trajectories(
        t_full, z_full,
        title=f'dBN Zero Flow: {n_zeros} zeros, t in [{t_full[0]:.3f}, {t_full[-1]:.3f}]')
    if save_plots:
        fig1.savefig('dbn_trajectories.png', dpi=150, bbox_inches='tight')
        print("  Saved: dbn_trajectories.png")
    plt.close(fig1)

    # Plot B: Gap evolution
    fig_gap = plot_gap_evolution(t_full, z_full)
    if save_plots:
        fig_gap.savefig('dbn_gap_evolution.png', dpi=150, bbox_inches='tight')
        print("  Saved: dbn_gap_evolution.png")
    plt.close(fig_gap)

    # Plot C: Spacing histograms at time slices
    fig2 = plot_spacing_evolution(t_full, z_full, n_samples=8)
    if save_plots:
        fig2.savefig('dbn_spacing_evolution.png', dpi=150, bbox_inches='tight')
        print("  Saved: dbn_spacing_evolution.png")
    plt.close(fig2)

    # Plot D: Statistics vs t
    fig3 = plot_statistics_vs_t(t_full, z_full)
    if save_plots:
        fig3.savefig('dbn_statistics_vs_t.png', dpi=150, bbox_inches='tight')
        print("  Saved: dbn_statistics_vs_t.png")
    plt.close(fig3)

    # Plot E: Pair correlation snapshots
    fig_pc = plot_pair_correlation_snapshots(t_full, z_full, n_samples=4)
    if save_plots:
        fig_pc.savefig('dbn_pair_correlation.png', dpi=150, bbox_inches='tight')
        print("  Saved: dbn_pair_correlation.png")
    plt.close(fig_pc)

    # Save stats
    with open('dbn_stats.json', 'w') as f:
        json.dump(stats_log, f, indent=2)
    print("  Saved: dbn_stats.json")

    # -- Summary --
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    t0_idx = np.argmin(np.abs(t_full))
    s0 = stats_log[t0_idx]
    sf = stats_log[-1]
    sb = stats_log[0]

    print(f"\n  At t = 0 (RH boundary):")
    print(f"    Spacing variance:  {s0.get('var', np.nan):.4f}  (GUE ~ 0.178)")
    print(f"    Chi-sq vs Wigner:  {s0.get('chi2_wigner', np.nan):.4f}")

    print(f"\n  At t = {t_full[-1]:.4f} (max forward):")
    print(f"    Spacing variance:  {sf.get('var', np.nan):.4f}")
    print(f"    Chi-sq vs Wigner:  {sf.get('chi2_wigner', np.nan):.4f}")

    print(f"\n  At t = {t_full[0]:.4f} (max backward):")
    print(f"    Spacing variance:  {sb.get('var', np.nan):.4f}")
    print(f"    Chi-sq vs Wigner:  {sb.get('chi2_wigner', np.nan):.4f}")

    chi2_arr = np.array([s.get('chi2_wigner', np.nan) for s in stats_log])
    if not np.all(np.isnan(chi2_arr)):
        peak_idx = np.nanargmax(chi2_arr)
        print(f"\n  Max GUE departure at t = {t_full[peak_idx]:.6f} "
              f"(chi2 = {chi2_arr[peak_idx]:.4f})")

    # Key observable: how does the min gap behave?
    fwd_min_gaps = [np.min(np.diff(np.sort(z_fwd[i]))) for i in range(len(t_fwd))]
    print(f"\n  Forward min-gap trend:")
    print(f"    t=0: {fwd_min_gaps[0]:.4f}")
    print(f"    t={t_fwd[-1]:.3f}: {fwd_min_gaps[-1]:.4f}")
    print(f"    Ratio: {fwd_min_gaps[-1]/fwd_min_gaps[0]:.3f}x (>1 = repulsion working)")

    return t_full, z_full, stats_log


def run_transition_analysis(n_zeros=50, t_back_max=-0.08, n_steps=500):
    """Deep analysis of the GUE -> arithmetic transition as t -> 0-.

    Evolves backward with fine t resolution and tracks multiple
    statistical measures to identify the transition structure.
    """
    print("\n" + "=" * 60)
    print("GUE -> ARITHMETIC TRANSITION ANALYSIS")
    print("=" * 60)

    all_zeros = np.load('_zeros_200.npy')
    z0 = all_zeros[:n_zeros].copy()
    init_min_gap = np.min(np.diff(np.sort(z0)))
    print(f"\n  {n_zeros} zeros, min gap at t=0: {init_min_gap:.4f}")

    # Backward evolution with fine steps
    print(f"  Evolving backward: t = 0 -> {t_back_max}...")
    t_bwd, z_bwd = evolve_zeros_backward(
        z0, t_start=0.0, t_end=t_back_max, n_steps=n_steps)
    print(f"  Reached t = {t_bwd[-1]:.6f} ({len(t_bwd)} steps)")

    # Also evolve forward for comparison
    print(f"  Evolving forward: t = 0 -> {abs(t_back_max)}...")
    t_fwd, z_fwd = evolve_zeros_forward(
        z0, t_start=0.0, t_end=abs(t_back_max), n_steps=n_steps)

    # Compute statistics at each step
    print("  Computing statistics...")
    bwd_stats = []
    for i in range(len(t_bwd)):
        z_sorted = np.sort(z_bwd[i])
        gaps = np.diff(z_sorted)
        spacings = gaps / np.mean(gaps)
        s = spacing_statistics(z_bwd[i])
        s['t'] = float(t_bwd[i])
        s['min_gap'] = float(np.min(gaps))
        s['gap_ratio'] = float(np.min(gaps) / init_min_gap)
        # Small-gap fraction: fraction of spacings < 0.5 * mean
        s['small_gap_frac'] = float(np.mean(spacings < 0.5))
        # Large-gap fraction: fraction > 2 * mean
        s['large_gap_frac'] = float(np.mean(spacings > 2.0))
        bwd_stats.append(s)

    fwd_stats = []
    for i in range(len(t_fwd)):
        z_sorted = np.sort(z_fwd[i])
        gaps = np.diff(z_sorted)
        spacings = gaps / np.mean(gaps)
        s = spacing_statistics(z_fwd[i])
        s['t'] = float(t_fwd[i])
        s['min_gap'] = float(np.min(gaps))
        s['gap_ratio'] = float(np.min(gaps) / init_min_gap)
        s['small_gap_frac'] = float(np.mean(spacings < 0.5))
        s['large_gap_frac'] = float(np.mean(spacings > 2.0))
        fwd_stats.append(s)

    # ── Multi-panel transition figure ──
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    t_b = [s['t'] for s in bwd_stats]
    t_f = [s['t'] for s in fwd_stats]

    # (0,0) Spacing variance vs t
    ax = axes[0, 0]
    ax.plot(t_b, [s['var'] for s in bwd_stats], 'r.-', markersize=2, label='backward')
    ax.plot(t_f, [s['var'] for s in fwd_stats], 'b.-', markersize=2, label='forward')
    ax.axhline(y=0.178, color='orange', linestyle=':', label='GUE ~ 0.178')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Spacing variance')
    ax.set_title('Variance (GUE ~ 0.178)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Min gap ratio vs t
    ax = axes[0, 1]
    ax.plot(t_b, [s['gap_ratio'] for s in bwd_stats], 'r.-', markersize=2)
    ax.plot(t_f, [s['gap_ratio'] for s in fwd_stats], 'b.-', markersize=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Min gap / initial min gap')
    ax.set_title('Min Gap Ratio (collision at 0)')
    ax.grid(True, alpha=0.3)

    # (1,0) Small-gap fraction
    ax = axes[1, 0]
    ax.plot(t_b, [s['small_gap_frac'] for s in bwd_stats], 'r.-', markersize=2)
    ax.plot(t_f, [s['small_gap_frac'] for s in fwd_stats], 'b.-', markersize=2)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Fraction s < 0.5')
    ax.set_title('Small-Gap Fraction (GUE level repulsion)')
    ax.grid(True, alpha=0.3)

    # (1,1) Large-gap fraction
    ax = axes[1, 1]
    ax.plot(t_b, [s['large_gap_frac'] for s in bwd_stats], 'r.-', markersize=2)
    ax.plot(t_f, [s['large_gap_frac'] for s in fwd_stats], 'b.-', markersize=2)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Fraction s > 2.0')
    ax.set_title('Large-Gap Fraction')
    ax.grid(True, alpha=0.3)

    # (2,0) Chi-squared vs Wigner
    ax = axes[2, 0]
    ax.plot(t_b, [s['chi2_wigner'] for s in bwd_stats], 'r.-', markersize=2)
    ax.plot(t_f, [s['chi2_wigner'] for s in fwd_stats], 'b.-', markersize=2)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('Chi-sq vs Wigner')
    ax.set_title('GUE Departure')
    ax.grid(True, alpha=0.3)

    # (2,1) Skewness
    ax = axes[2, 1]
    ax.plot(t_b, [s['skew'] for s in bwd_stats], 'r.-', markersize=2)
    ax.plot(t_f, [s['skew'] for s in fwd_stats], 'b.-', markersize=2)
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('Skewness')
    ax.set_title('Distribution Asymmetry')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'GUE-Arithmetic Transition ({n_zeros} zeros)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('dbn_transition_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: dbn_transition_analysis.png")
    plt.close(fig)

    # ── Closest-pair tracking ──
    print("\n  Tracking closest zero pair...")
    closest_pairs = []
    for i in range(len(t_bwd)):
        z_sorted = np.sort(z_bwd[i])
        gaps = np.diff(z_sorted)
        min_idx = np.argmin(gaps)
        closest_pairs.append({
            't': float(t_bwd[i]),
            'z1': float(z_sorted[min_idx]),
            'z2': float(z_sorted[min_idx + 1]),
            'gap': float(gaps[min_idx]),
        })

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot([cp['t'] for cp in closest_pairs],
             [cp['gap'] for cp in closest_pairs], 'r.-', markersize=3)
    ax1.set_ylabel('Closest pair gap')
    ax1.set_title('Closest Zero Pair Under Backward Flow')
    ax1.grid(True, alpha=0.3)

    ax2.plot([cp['t'] for cp in closest_pairs],
             [cp['z1'] for cp in closest_pairs], 'b.-', markersize=2, label='z_lower')
    ax2.plot([cp['t'] for cp in closest_pairs],
             [cp['z2'] for cp in closest_pairs], 'g.-', markersize=2, label='z_upper')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Position')
    ax2.set_title('Closest Pair Positions (converging toward collision)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig('dbn_closest_pair.png', dpi=150, bbox_inches='tight')
    print("  Saved: dbn_closest_pair.png")
    plt.close(fig2)

    # ── Summary ──
    print("\n  TRANSITION SUMMARY:")
    print(f"    Backward reach: t = {t_bwd[-1]:.6f}")
    final_min = bwd_stats[-1]['min_gap']
    final_var = bwd_stats[-1]['var']
    print(f"    Min gap at endpoint: {final_min:.4f} (started {init_min_gap:.4f})")
    print(f"    Variance at endpoint: {final_var:.4f} (GUE=0.178, t=0: {bwd_stats[0]['var']:.4f})")

    # Find t where variance crosses GUE
    fwd_vars = np.array([s['var'] for s in fwd_stats])
    fwd_ts = np.array([s['t'] for s in fwd_stats])
    crossings = np.where(np.diff(np.sign(fwd_vars - 0.178)))[0]
    if len(crossings) > 0:
        t_cross = fwd_ts[crossings[0]]
        print(f"    GUE variance crossing (forward): t = {t_cross:.4f}")
    else:
        print(f"    Forward variance range: [{fwd_vars.min():.4f}, {fwd_vars.max():.4f}]")

    # Save all transition data
    transition_data = {
        'backward': bwd_stats,
        'forward': fwd_stats,
        'closest_pairs': closest_pairs,
    }
    with open('dbn_transition.json', 'w') as f:
        json.dump(transition_data, f, indent=2)
    print("  Saved: dbn_transition.json")


def run_scaling_study(n_values=None, t_forward=0.15, n_steps=400,
                      save_plots=True):
    """Study how the GUE variance crossing point scales with N.

    For each N in n_values, evolves zeros forward and finds the t where
    spacing variance first drops below GUE (0.178). If the crossing
    converges to t=0 as N increases, it means RH zeros are exactly GUE.

    Returns:
        dict with N values, crossing points, and per-N statistics.
    """
    if n_values is None:
        n_values = [25, 50, 100, 200]

    GUE_VAR = 0.178
    all_zeros = np.load('_zeros_200.npy')

    print("\n" + "=" * 60)
    print("GUE CROSSING POINT vs N -- SCALING STUDY")
    print("=" * 60)
    print(f"  N values: {n_values}")
    print(f"  Forward range: t = 0 -> {t_forward}")
    print(f"  GUE variance target: {GUE_VAR}")

    results = []

    for n_z in n_values:
        if n_z > len(all_zeros):
            print(f"\n  Skipping N={n_z} (only {len(all_zeros)} zeros available)")
            continue

        z0 = all_zeros[:n_z].copy()
        init_var = spacing_statistics(z0)['var']

        print(f"\n  -- N = {n_z} --")
        print(f"  Range: [{z0[0]:.2f}, {z0[-1]:.2f}]")
        print(f"  Variance at t=0: {init_var:.4f}")

        # Loosen tolerances for large N to keep runtime manageable
        rtol = 1e-6 if n_z <= 100 else 1e-5
        atol = 1e-8 if n_z <= 100 else 1e-7

        t0 = time.time()
        t_fwd, z_fwd = evolve_zeros_forward(
            z0, t_start=0.0, t_end=t_forward, n_steps=n_steps,
            rtol=rtol, atol=atol)
        elapsed = time.time() - t0
        print(f"  Forward ODE: {elapsed:.1f}s ({len(t_fwd)} steps)")

        # Compute variance at each t
        variances = []
        chi2_vals = []
        for i in range(len(t_fwd)):
            s = spacing_statistics(z_fwd[i])
            variances.append(s['var'])
            chi2_vals.append(s.get('chi2_wigner', np.nan))

        variances = np.array(variances)
        t_arr = np.array(t_fwd)

        # Find crossing: first t where variance drops below GUE_VAR
        crossings = np.where(np.diff(np.sign(variances - GUE_VAR)))[0]
        if len(crossings) > 0:
            # Linear interpolation for precise crossing
            idx = crossings[0]
            v0, v1 = variances[idx], variances[idx + 1]
            t0_c, t1_c = t_arr[idx], t_arr[idx + 1]
            t_cross = t0_c + (GUE_VAR - v0) / (v1 - v0) * (t1_c - t0_c)
            print(f"  GUE crossing: t = {t_cross:.6f}")
        else:
            t_cross = np.nan
            if variances[-1] > GUE_VAR:
                print(f"  No crossing (variance stays above GUE: min={variances.min():.4f})")
            else:
                print(f"  Variance starts below GUE — crossing at t <= 0")
                t_cross = 0.0

        # Also find chi2 minimum (best GUE match)
        chi2_arr = np.array(chi2_vals)
        if not np.all(np.isnan(chi2_arr)):
            best_idx = np.nanargmin(chi2_arr)
            t_best_gue = float(t_arr[best_idx])
            chi2_min = float(chi2_arr[best_idx])
            print(f"  Best GUE match: t = {t_best_gue:.6f} (chi2 = {chi2_min:.4f})")
        else:
            t_best_gue = np.nan
            chi2_min = np.nan

        results.append({
            'N': n_z,
            'var_at_t0': float(init_var),
            't_cross_gue': float(t_cross),
            't_best_gue': t_best_gue,
            'chi2_at_best': chi2_min,
            'var_min': float(variances.min()),
            'var_max': float(variances.max()),
            't_values': t_arr.tolist(),
            'variances': variances.tolist(),
            'elapsed_s': elapsed,
        })

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SCALING SUMMARY")
    print("=" * 60)
    print(f"  {'N':>5}  {'Var(t=0)':>10}  {'t_cross':>10}  {'t_best':>10}  {'chi2_best':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in results:
        tc = f"{r['t_cross_gue']:.6f}" if not np.isnan(r['t_cross_gue']) else "N/A"
        tb = f"{r['t_best_gue']:.6f}" if not np.isnan(r['t_best_gue']) else "N/A"
        c2 = f"{r['chi2_at_best']:.4f}" if not np.isnan(r['chi2_at_best']) else "N/A"
        print(f"  {r['N']:>5}  {r['var_at_t0']:>10.4f}  {tc:>10}  {tb:>10}  {c2:>10}")

    # ── Convergence plot ──
    if save_plots and len(results) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) Variance vs t for all N values
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
        for r, c in zip(results, colors):
            ax.plot(r['t_values'], r['variances'], color=c, linewidth=1.5,
                    label=f"N={r['N']}")
        ax.axhline(y=GUE_VAR, color='orange', linestyle=':', linewidth=2,
                   label=f'GUE = {GUE_VAR}')
        ax.set_xlabel('t')
        ax.set_ylabel('Spacing variance')
        ax.set_title('Variance vs t for different N')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (0,1) Crossing point vs N
        ax = axes[0, 1]
        ns = [r['N'] for r in results if not np.isnan(r['t_cross_gue'])]
        tcs = [r['t_cross_gue'] for r in results if not np.isnan(r['t_cross_gue'])]
        if len(ns) >= 2:
            ax.plot(ns, tcs, 'ro-', markersize=8, linewidth=2)
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            # Fit 1/N trend if enough points
            if len(ns) >= 3:
                ns_arr = np.array(ns, dtype=float)
                tcs_arr = np.array(tcs)
                # Fit t_cross = a/N + b
                A = np.column_stack([1.0/ns_arr, np.ones(len(ns_arr))])
                coeff, _, _, _ = np.linalg.lstsq(A, tcs_arr, rcond=None)
                n_fit = np.linspace(min(ns)*0.8, max(ns)*2, 100)
                t_fit = coeff[0] / n_fit + coeff[1]
                ax.plot(n_fit, t_fit, 'b--', alpha=0.5,
                        label=f'Fit: {coeff[0]:.3f}/N + {coeff[1]:.4f}')
                ax.legend(fontsize=9)
                print(f"\n  1/N fit: t_cross = {coeff[0]:.4f}/N + {coeff[1]:.6f}")
                print(f"  Extrapolated t_cross(N->inf) = {coeff[1]:.6f}")
        ax.set_xlabel('N (number of zeros)')
        ax.set_ylabel('t at GUE variance crossing')
        ax.set_title('GUE Crossing Point vs N')
        ax.grid(True, alpha=0.3)

        # (1,0) Variance at t=0 vs N
        ax = axes[1, 0]
        ns_all = [r['N'] for r in results]
        v0s = [r['var_at_t0'] for r in results]
        ax.plot(ns_all, v0s, 'gs-', markersize=8, linewidth=2)
        ax.axhline(y=GUE_VAR, color='orange', linestyle=':', linewidth=2,
                   label=f'GUE = {GUE_VAR}')
        ax.set_xlabel('N')
        ax.set_ylabel('Variance at t=0')
        ax.set_title('How close are RH zeros to GUE?')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (1,1) Best chi2 vs N
        ax = axes[1, 1]
        chi2s = [r['chi2_at_best'] for r in results
                 if not np.isnan(r['chi2_at_best'])]
        ns_c2 = [r['N'] for r in results
                 if not np.isnan(r['chi2_at_best'])]
        t_bests = [r['t_best_gue'] for r in results
                   if not np.isnan(r['t_best_gue'])]
        if chi2s:
            ax.plot(ns_c2, t_bests, 'ms-', markersize=8, linewidth=2)
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('N')
            ax.set_ylabel('t at best GUE fit (min chi2)')
            ax.set_title('Best GUE Match Point vs N')
            ax.grid(True, alpha=0.3)

        fig.suptitle('GUE Crossing Convergence Study', fontsize=14, y=1.01)
        plt.tight_layout()
        fig.savefig('dbn_scaling_study.png', dpi=150, bbox_inches='tight')
        print("  Saved: dbn_scaling_study.png")
        plt.close(fig)

    # Save data
    save_results = [{k: v for k, v in r.items()
                     if k not in ('t_values', 'variances')}
                    for r in results]
    with open('dbn_scaling.json', 'w') as f:
        json.dump({'results': save_results,
                   'full': results}, f, indent=2)
    print("  Saved: dbn_scaling.json")

    return results


if __name__ == '__main__':
    run_scaling_study()
