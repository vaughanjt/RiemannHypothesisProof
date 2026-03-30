"""
Session 28c: Probe the mechanism behind sigma_min ~ 1/N^2.

For v_min on w_perp:
  sigma_min = (1/2pi) integral |D(1/2+it)|^2 |zeta(1/2+it)|^2 / (1/4+t^2) dt

Split integral into regions to find where the contribution comes from:
  Region A: |t| <= 1 (near t=0, strong weight, D can be made small)
  Region B: 1 < |t| <= N (moderate weight, D has MV average)
  Region C: |t| > N (weak weight 1/N^2, D has MV average)

Also: compute |D(1/2+it)|^2 explicitly for the min eigenvector
to see WHERE D is nonzero.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, power, nstr
import time

mp.dps = 30


def build_gram(N, n_grid=500000):
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    return (fp @ fp.T) * dx


def D_squared(c, t_val):
    """Compute |D(1/2+it)|^2 for Dirichlet polynomial D(s) = sum c_j j^{-s}."""
    N = len(c)
    s = mpc(0.5, t_val)
    D = sum(c[j] * power(mpf(j+1), -s) for j in range(N))
    return float(abs(D)**2)


if __name__ == "__main__":
    print("SIGMA_MIN MECHANISM PROBE")
    print("=" * 70)

    for N in [50, 100]:
        t0 = time.time()
        G = build_gram(N, n_grid=500000)
        evals, evecs = np.linalg.eigh(G)
        v_min = evecs[:, 0]
        sigma_min = evals[0]

        print(f"\nN = {N}, sigma_min = {sigma_min:.6e}")

        # Compute |D(1/2+it)|^2 at many t values
        t_points = np.concatenate([
            np.linspace(0, 1, 50),
            np.linspace(1, N, 200),
            np.linspace(N, 3*N, 100),
        ])

        D_sq = np.array([D_squared(v_min, t) for t in t_points])

        # Weight function
        zeta_sq = np.array([float(abs(zeta(mpc(0.5, t)))**2) for t in t_points])
        weight = zeta_sq / (0.25 + t_points**2)

        # Integrand
        integrand = D_sq * weight / (2 * float(pi))

        # Regional contributions (approximate)
        mask_A = t_points <= 1
        mask_B = (t_points > 1) & (t_points <= N)
        mask_C = t_points > N

        # Trapezoidal integration on each region
        def trap_integral(t, y, mask):
            t_m = t[mask]; y_m = y[mask]
            if len(t_m) < 2: return 0.0
            return np.trapezoid(y_m, t_m)

        I_A = 2 * trap_integral(t_points, integrand, mask_A)  # factor 2 for symmetric
        I_B = 2 * trap_integral(t_points, integrand, mask_B)
        I_C = 2 * trap_integral(t_points, integrand, mask_C)
        I_total = I_A + I_B + I_C

        print(f"  Regional contributions to sigma_min:")
        print(f"    |t| <= 1:     {I_A:.6e}  ({100*I_A/I_total:.1f}%)")
        print(f"    1 < |t| <= N: {I_B:.6e}  ({100*I_B/I_total:.1f}%)")
        print(f"    |t| > N:      {I_C:.6e}  ({100*I_C/I_total:.1f}%)")
        print(f"    Total:        {I_total:.6e} (sigma_min = {sigma_min:.6e})")

        # Key quantities
        D_at_0 = D_squared(v_min, 0)
        D_at_1 = D_squared(v_min, 1.0)
        D_at_N = D_squared(v_min, float(N))

        # Average |D|^2 in each region
        D_avg_A = np.mean(D_sq[mask_A]) if np.sum(mask_A) > 0 else 0
        D_avg_B = np.mean(D_sq[mask_B]) if np.sum(mask_B) > 0 else 0
        D_avg_C = np.mean(D_sq[mask_C]) if np.sum(mask_C) > 0 else 0

        print(f"\n  |D(1/2+it)|^2 averages:")
        print(f"    Region A (|t|<=1):     avg={D_avg_A:.6e}, at t=0: {D_at_0:.6e}")
        print(f"    Region B (1<|t|<=N):   avg={D_avg_B:.6e}")
        print(f"    Region C (|t|>N):      avg={D_avg_C:.6e}")
        print(f"    MV prediction (avg = sum c_j^2/j): {np.sum(v_min**2 / np.arange(1,N+1)):.6e}")

        # Weight averages
        w_avg_A = np.mean(weight[mask_A])
        w_avg_B = np.mean(weight[mask_B]) if np.sum(mask_B) > 0 else 0
        w_avg_C = np.mean(weight[mask_C]) if np.sum(mask_C) > 0 else 0

        print(f"\n  Weight |zeta|^2/(1/4+t^2) averages:")
        print(f"    Region A: {w_avg_A:.6e}")
        print(f"    Region B: {w_avg_B:.6e}")
        print(f"    Region C: {w_avg_C:.6e}")

        # The key ratio: sigma_min * N^2
        print(f"\n  sigma_min * N^2 = {sigma_min * N**2:.4f}")

        # Mechanism identification
        print(f"\n  MECHANISM: sigma_min comes from:")
        dominant = max([(I_A, 'Region A'), (I_B, 'Region B'), (I_C, 'Region C')])
        print(f"    Dominant region: {dominant[1]} ({100*dominant[0]/I_total:.0f}%)")
        print(f"    |D|^2 * weight in dominant region ~ {dominant[0]:.4e}")
        print(f"    = (|D|^2 avg ~ {D_avg_B:.2e}) * (weight avg ~ {w_avg_B:.2e}) * (length ~ {N:.0f})")
        print(f"    ~ {D_avg_B * w_avg_B * N:.4e} (vs actual {I_B:.4e})")

        print(f"  ({time.time()-t0:.1f}s)")
