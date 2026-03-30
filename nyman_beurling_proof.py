"""
Session 28d: Formal proof verification — sigma_min >= c * log(N) / N^2

Proof chain:
1. v_min in w_perp => pole terms vanish, sigma_min = integral |D|^2|zeta|^2/(1/4+t^2) dt
2. v_min has energy at j ~ N => sum c_j^2/j >= 1/N
3. MV for D*zeta on [N,2N]: integral |D*zeta|^2 dt >= c * N * log(N) * sum(c_j^2/j)
4. Weight on [N,2N]: 1/(1/4+t^2) >= 1/(5N^2)
5. Combine: sigma_min >= c * log(N) / N^2

This script verifies each step numerically.
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


def compute_Dzeta_integral(c_vec, T_lo, T_hi, n_t=2000):
    """Compute integral_{T_lo}^{T_hi} |D(1/2+it) * zeta(1/2+it)|^2 dt."""
    N = len(c_vec)
    t_grid = np.linspace(T_lo, T_hi, n_t)
    dt = t_grid[1] - t_grid[0]

    total = 0.0
    for t in t_grid:
        s = mpc(0.5, t)
        D_val = sum(c_vec[j] * power(mpf(j+1), -s) for j in range(N))
        z_val = zeta(s)
        total += float(abs(D_val * z_val)**2)

    return total * dt


def compute_D_integral(c_vec, T_lo, T_hi, n_t=2000):
    """Compute integral_{T_lo}^{T_hi} |D(1/2+it)|^2 dt."""
    N = len(c_vec)
    t_grid = np.linspace(T_lo, T_hi, n_t)
    dt = t_grid[1] - t_grid[0]

    total = 0.0
    for t in t_grid:
        s = mpc(0.5, t)
        D_val = sum(c_vec[j] * power(mpf(j+1), -s) for j in range(N))
        total += float(abs(D_val)**2)

    return total * dt


if __name__ == "__main__":
    print("PROOF VERIFICATION: sigma_min >= c * log(N) / N^2")
    print("=" * 70)

    for N in [30, 50, 100]:
        t0 = time.time()
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals, evecs = np.linalg.eigh(G)
        v_min = evecs[:, 0]
        sigma_min = evals[0]

        print(f"\nN = {N}, sigma_min = {sigma_min:.6e}")

        # STEP 1: v_min in w_perp
        S = np.dot(v_min, 1.0/np.arange(1, N+1))
        print(f"\n  Step 1: v_min in w_perp?")
        print(f"    S = sum(c_j/j) = {S:.4e} ({'YES' if abs(S) < 0.01 else 'NO'})")

        # STEP 2: sum c_j^2/j >= 1/N
        sum_cj2_j = np.sum(v_min**2 / np.arange(1, N+1))
        print(f"\n  Step 2: sum(c_j^2/j) >= 1/N ?")
        print(f"    sum(c_j^2/j) = {sum_cj2_j:.6e}")
        print(f"    1/N = {1/N:.6e}")
        print(f"    ratio = {sum_cj2_j * N:.4f} ({'YES: ratio >= 1' if sum_cj2_j >= 1/N else 'NO: ratio < 1'})")

        # STEP 3: MV for D*zeta on [N, 2N]
        print(f"\n  Step 3: MV for D*zeta on [N, 2N]")
        I_Dzeta = compute_Dzeta_integral(v_min, float(N), 2.0*N, n_t=1000)
        I_D = compute_D_integral(v_min, float(N), 2.0*N, n_t=1000)
        MV_prediction = N * np.log(N) * sum_cj2_j  # expected: N * log(N) * sum(c_j^2/j)

        print(f"    integral |D*zeta|^2 dt = {I_Dzeta:.6e}")
        print(f"    integral |D|^2 dt     = {I_D:.6e}")
        print(f"    N * log(N) * sum(cj2/j) = {MV_prediction:.6e}")
        print(f"    |D*zeta|^2 / (N*log(N)*sum) = {I_Dzeta/MV_prediction:.4f}")
        print(f"    |D*zeta|^2 / |D|^2 = {I_Dzeta/I_D:.4f} (should be ~ log(N) = {np.log(N):.2f})")

        # STEP 4: Weight bound on [N, 2N]
        weight_min = 1.0 / (0.25 + (2*N)**2)
        weight_max = 1.0 / (0.25 + N**2)
        print(f"\n  Step 4: Weight on [N, 2N]")
        print(f"    min weight = {weight_min:.6e}")
        print(f"    max weight = {weight_max:.6e}")
        print(f"    1/(5N^2) = {1/(5*N**2):.6e}")

        # STEP 5: Combine for lower bound
        # sigma_min >= (1/2pi) * weight_min * I_Dzeta
        bound_from_integral = (1/(2*np.pi)) * weight_min * I_Dzeta
        # Theoretical bound: c * log(N) / N^2
        theoretical = np.log(N) / (2 * np.pi * N**2)

        print(f"\n  Step 5: Combined lower bound")
        print(f"    (1/2pi) * w_min * integral = {bound_from_integral:.6e}")
        print(f"    log(N)/(2pi*N^2)           = {theoretical:.6e}")
        print(f"    sigma_min (actual)          = {sigma_min:.6e}")
        print(f"    bound / sigma_min           = {bound_from_integral/sigma_min:.4f}")
        print(f"    theoretical / sigma_min     = {theoretical/sigma_min:.4f}")

        # THE KEY RATIO: sigma_min * N^2 / log(N)
        key_ratio = sigma_min * N**2 / np.log(N)
        print(f"\n  KEY RATIO: sigma_min * N^2 / log(N) = {key_ratio:.6f}")
        print(f"  (should be bounded below by a positive constant)")

        print(f"\n  Time: {time.time()-t0:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Is sigma_min >= c * log(N) / N^2 ?")
    print("-" * 70)
    print(f"{'N':>6} {'sigma_min':>12} {'bound(numer)':>12} {'log(N)/N^2':>12} {'sig*N^2/logN':>12}")
    print("-" * 70)
