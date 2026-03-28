"""Self-adjointness test via the Stieltjes property.

THEOREM: RH holds iff T(w) = sum_n 1/(w - gamma_n^2) is a Stieltjes function,
i.e., Im(T(w)) < 0 whenever Im(w) > 0.

Proof: T(w) is Stieltjes iff all its poles are on [0,inf) with positive residues.
The poles are at w = gamma_n^2. These are all real positive iff gamma_n is real
iff RH holds.

This is a GLOBAL SIGN CONDITION on a function computable from (xi'/xi) via mpmath.
No zeros needed. The margin |Im(T)| at each point gives a quantitative bound
on how far any zero can be from the critical line.

Test: evaluate T(w) at a dense grid of w in the upper half-plane, check Im(T) < 0.
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time


def T_from_xi_complex(w, dps=30):
    """Compute T(w) from (xi'/xi) for complex w. NON-CIRCULAR.

    T(w) = sum_n 1/(w - gamma_n^2)
         = i / (2*sqrt(w)) * (xi'/xi)(1/2 + i*sqrt(w))

    where sqrt uses the principal branch (Re(sqrt(w)) > 0 for w not on (-inf,0]).
    """
    with mpmath.workdps(dps):
        w_mp = mpmath.mpc(w)
        sqrt_w = mpmath.sqrt(w_mp)
        s = mpmath.mpf('0.5') + mpmath.mpc(0, 1) * sqrt_w

        # Compute (xi'/xi)(s)
        def xi_func(sv):
            return (sv * (sv - 1) / 2
                    * mpmath.power(mpmath.pi, -sv/2)
                    * mpmath.gamma(sv/2)
                    * mpmath.zeta(sv))

        xi_val = xi_func(s)
        xi_deriv = mpmath.diff(xi_func, s)

        if abs(xi_val) < mpmath.mpf(10)**(-dps + 5):
            return complex(float('nan'), float('nan'))

        xi_ratio = xi_deriv / xi_val
        T = mpmath.mpc(0, 1) / (2 * sqrt_w) * xi_ratio

        return complex(T)


def T_from_zeros_complex(w, gammas):
    """T(w) from known zeros. FOR VERIFICATION ONLY."""
    return np.sum(1.0 / (w - gammas**2))


def run_stieltjes_test():
    """Test the Stieltjes property: Im(T(w)) < 0 for Im(w) > 0."""

    print("=" * 70)
    print("SELF-ADJOINTNESS TEST: STIELTJES PROPERTY OF T(w)")
    print("=" * 70)
    print()
    print("  THEOREM: RH <=> Im(T(w)) < 0 for all w with Im(w) > 0")
    print("  T(w) computed from (xi'/xi) via mpmath. NO ZEROS USED.")

    all_zeros = np.load('_zeros_200.npy')

    # Step 1: Verify T(w) formula at a few points
    print("\n[1/4] Verification: T(w) from xi vs from zeros")

    test_ws = [complex(-100, 50), complex(0, 100), complex(500, 200),
               complex(-1000, 10), complex(100, 1)]

    print(f"\n  {'w':>20}  {'Im(T_xi)':>14}  {'Im(T_zeros)':>14}  {'both < 0?':>10}")
    for w in test_ws:
        T_xi = T_from_xi_complex(w, dps=30)
        T_z = T_from_zeros_complex(w, all_zeros)
        both_neg = "YES" if T_xi.imag < 0 and T_z.imag < 0 else "NO"
        print(f"  {w!s:>20}  {T_xi.imag:>14.8f}  {T_z.imag:>14.8f}  {both_neg:>10}")

    # Step 2: Dense grid test in upper half w-plane
    print("\n[2/4] Dense grid test: Im(T(w)) < 0 for Im(w) > 0?")

    # Grid: w = u + iv, u in [-500, 5000], v in [1, 500]
    n_u, n_v = 60, 30
    u_vals = np.linspace(-500, 5000, n_u)
    v_vals = np.logspace(0, 3, n_v)  # v from 1 to 1000

    print(f"  Grid: {n_u} x {n_v} = {n_u*n_v} points")
    print(f"  u in [{u_vals[0]}, {u_vals[-1]}], v in [{v_vals[0]:.1f}, {v_vals[-1]:.1f}]")

    t0 = time.time()
    ImT_grid = np.zeros((n_v, n_u))
    n_violations = 0
    min_ImT = 0.0
    worst_w = None

    for i, v in enumerate(v_vals):
        for j, u in enumerate(u_vals):
            w = complex(u, v)
            T = T_from_xi_complex(w, dps=25)
            ImT_grid[i, j] = T.imag

            if T.imag > 0:
                n_violations += 1
                if worst_w is None or T.imag > ImT_grid[i,j]:
                    worst_w = w

            if T.imag < min_ImT:
                min_ImT = T.imag

        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Row {i+1}/{n_v} (v={v:.1f}), violations so far: {n_violations}, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  VIOLATIONS (Im(T) > 0): {n_violations} / {n_u*n_v}")

    if n_violations == 0:
        print("  *** Im(T(w)) < 0 at ALL tested points ***")
        print("  *** Stieltjes property HOLDS (consistent with RH) ***")
    else:
        print(f"  *** VIOLATIONS FOUND at {n_violations} points ***")
        print(f"  *** This would DISPROVE RH if confirmed! ***")
        print(f"  *** Worst: w = {worst_w}, Im(T) = {ImT_grid.max():.6e} ***")

    # Step 3: Margin analysis — how negative is Im(T)?
    print("\n[3/4] Margin analysis")

    # The margin |Im(T(w))| at each w gives a bound on off-line displacement.
    # If gamma = a + ib (off-line by b), its contribution to Im(T) at w = u+iv is:
    #   ~ 2ab*v / |w - (a^2-b^2+2iab)|^2
    # For this to NOT flip the sign, we need |b| < margin / (some geometric factor).

    # Find the "thinnest" margin (most negative Im(T) closest to 0)
    # along the line v = v0 (fixed imaginary part)
    print(f"\n  Min |Im(T)| by height v:")
    for v_idx in [0, 5, 10, 15, 20, 25, 29]:
        if v_idx < n_v:
            row = ImT_grid[v_idx, :]
            min_abs = np.min(np.abs(row))
            min_val = np.max(row)  # closest to 0 (most positive = weakest)
            u_weakest = u_vals[np.argmax(row)]
            print(f"    v={v_vals[v_idx]:>8.1f}: min|Im(T)|={min_abs:.6e},"
                  f" weakest at u={u_weakest:.0f} (Im(T)={min_val:.6e})")

    # Step 4: The critical strip scan
    print("\n[4/4] Critical strip scan: w near positive real axis")
    print("  (Where the poles are closest — most sensitive to off-line zeros)")

    # Scan w = gamma_n^2 + i*epsilon for small epsilon
    # This probes the neighborhood of the poles
    epsilons = [0.1, 1.0, 10.0, 100.0]
    # Use w near the known squared zeros
    test_positions = [200, 1000, 5000, 10000, 50000, 100000]

    print(f"\n  {'u (=gamma^2)':>14}  {'epsilon':>10}  {'Im(T)':>14}  {'status':>8}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*14}  {'-'*8}")
    for u in test_positions:
        for eps in epsilons:
            w = complex(u, eps)
            T = T_from_xi_complex(w, dps=25)
            status = "OK" if T.imag < 0 else "VIOLATE"
            print(f"  {u:>14.1f}  {eps:>10.1f}  {T.imag:>14.8f}  {status:>8}")

    # -- Plots --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Im(T) heatmap
    ax = axes[0, 0]
    # Clip for visualization
    ImT_clipped = np.clip(ImT_grid, -0.01, 0.001)
    im = ax.pcolormesh(u_vals, v_vals, ImT_clipped, cmap='RdBu_r', shading='auto')
    ax.set_yscale('log')
    ax.set_xlabel('Re(w)')
    ax.set_ylabel('Im(w)')
    ax.set_title('Im(T(w)) in upper half w-plane\n(blue = negative = RH consistent)')
    plt.colorbar(im, ax=ax, label='Im(T)')
    # Mark violation boundary
    ax.contour(u_vals, v_vals, ImT_grid, levels=[0], colors='red', linewidths=2)

    # (0,1) Im(T) along horizontal cuts
    ax = axes[0, 1]
    for v_idx in [0, 5, 10, 20, 29]:
        if v_idx < n_v:
            ax.plot(u_vals, ImT_grid[v_idx, :], linewidth=1,
                    label=f'v={v_vals[v_idx]:.0f}')
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--', label='RH boundary')
    ax.set_xlabel('Re(w)')
    ax.set_ylabel('Im(T)')
    ax.set_title('Im(T) along horizontal cuts')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,0) Margin: min|Im(T)| vs v
    ax = axes[1, 0]
    margins = [np.min(np.abs(ImT_grid[i, :])) for i in range(n_v)]
    ax.loglog(v_vals, margins, 'b-', linewidth=2)
    ax.set_xlabel('v = Im(w)')
    ax.set_ylabel('min |Im(T(u+iv))| over u')
    ax.set_title('Stieltjes margin vs height (0 = RH violation)')
    ax.grid(True, alpha=0.3, which='both')

    # (1,1) Im(T) near the real axis (v small)
    ax = axes[1, 1]
    v_small = [0.1, 0.5, 1.0, 5.0]
    u_fine = np.linspace(0, 2000, 100)
    for v in v_small:
        ImT_fine = []
        for u in u_fine:
            T = T_from_xi_complex(complex(u, v), dps=20)
            ImT_fine.append(T.imag)
        ax.plot(u_fine, ImT_fine, linewidth=1, label=f'v={v}')
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Re(w)')
    ax.set_ylabel('Im(T)')
    ax.set_title('Im(T) near real axis (strongest test)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Self-Adjointness: Is T(w) Stieltjes? (Im(T)<0 iff RH)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('jacobi_stieltjes.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: jacobi_stieltjes.png")
    plt.close(fig)

    save_data = {
        'n_violations': n_violations,
        'grid_size': [n_u, n_v],
        'u_range': [float(u_vals[0]), float(u_vals[-1])],
        'v_range': [float(v_vals[0]), float(v_vals[-1])],
        'min_ImT': float(min_ImT),
        'margins': list(zip([float(v) for v in v_vals], [float(m) for m in margins])),
    }
    with open('jacobi_stieltjes.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved: jacobi_stieltjes.json")


if __name__ == '__main__':
    run_stieltjes_test()
