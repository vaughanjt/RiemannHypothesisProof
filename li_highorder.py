"""
Session 17e: High-order Li coefficients with proper numerical precision.

The radius of convergence of log((s-1)*zeta(s)) around s=1 is 3
(nearest singularity: trivial zero at s=-2).

Using FFT radius r=2 keeps coefficients d_k ~ (2/3)^k, so d_200 ~ 10^{-35},
well within 150-digit precision.

The binomial sum for lambda_n involves cancellation of terms up to ~10^{60},
so we need ~200dp for n~200.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, zeta, power, loggamma, euler

mp.dps = 200  # heavy artillery


def log_sz(s):
    """log((s-1)*zeta(s)) — entire, encodes prime distribution."""
    h = s - 1
    if abs(h) < mpf(10)**(-100):
        return mpf(0)
    return log(h * zeta(s))


def smooth_gamma_part(s):
    """Gamma side of log xi (no zeta)."""
    return log(mpf(0.5)) + log(s) - (s / 2) * log(pi) + loggamma(s / 2)


def log_xi(s):
    """log(xi(s)) = smooth_gamma + log_sz."""
    h = s - 1
    if abs(h) < mpf(10)**(-100):
        return -log(mpf(2))
    return smooth_gamma_part(s) + log_sz(s)


def taylor_fft(f, center, n_terms, radius, n_points):
    """FFT-based Taylor coefficients. High precision."""
    r = mpf(radius)
    print(f"  Evaluating {n_points} points on circle of radius {radius}...")
    values = []
    for j in range(n_points):
        theta = 2 * mpf(pi) * j / n_points
        z = mpc(center, 0) + r * mpc(mpmath.cos(theta), mpmath.sin(theta))
        values.append(f(z))
    print(f"  Computing {n_terms+1} Fourier coefficients...")
    coeffs = []
    for k in range(n_terms + 1):
        ck = mpc(0, 0)
        for j in range(n_points):
            theta = 2 * mpf(pi) * j / n_points
            ck += values[j] * mpc(mpmath.cos(-k * theta), mpmath.sin(-k * theta))
        ck /= n_points
        ck /= power(r, k)
        coeffs.append(ck.real)
    return coeffs


def lambda_from_taylor(n, coeffs):
    total = mpf(0)
    for j in range(n):
        k = n - j
        if k < len(coeffs):
            total += mpmath.binomial(n - 1, j) * coeffs[k]
    return n * total


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 17e: High-Order Li Coefficients (200dp, r=2)")
    print("=" * 70)

    N_MAX = 200
    RADIUS = 2.0  # radius of convergence is 3
    N_PTS = 1024

    # --- Compute Taylor coefficients ---
    print("\nComputing zeta part coefficients...")
    zc = taylor_fft(log_sz, 1, N_MAX + 5, RADIUS, N_PTS)

    print("\nComputing gamma part coefficients...")
    gc = taylor_fft(smooth_gamma_part, 1, N_MAX + 5, RADIUS, N_PTS)

    # Verify coefficient decay
    print("\nCoefficient decay check:")
    print(f"  Expected: |d_k| ~ (r/R)^k = (2/3)^k")
    for k in [1, 5, 10, 20, 50, 100, 150, 200]:
        if k < len(zc):
            predicted = abs(float(zc[1])) * (2.0/3.0)**(k-1)
            actual = abs(float(zc[k]))
            print(f"  |d_{k:3d}| = {actual:.4e}   (predicted ~{predicted:.4e})")

    # Verify d_1 = gamma
    gamma = float(zc[1])
    print(f"\n  d_1 = {gamma:.15f}")
    print(f"  gamma= {float(euler):.15f}")
    print(f"  Match: {abs(gamma - float(euler)) < 1e-10}")

    # --- Lambda_n^{zeta} for n=1..200 ---
    print(f"\n--- lambda_n^{{zeta}} for n=1..{N_MAX} ---")
    print(f"  {'n':>4s}  {'zeta part':>16s}  {'corr ratio':>12s}  {'status':>10s}")

    zeta_vals = []
    min_zeta = (0, float('inf'))
    max_corr_ratio = (0, 0.0)
    all_positive = True

    for n in range(1, N_MAX + 1):
        z = float(lambda_from_taylor(n, zc))
        ng = n * gamma
        cr = abs(z - ng) / ng if ng > 0 else 0
        zeta_vals.append(z)

        if z < min_zeta[1]:
            min_zeta = (n, z)
        if cr > max_corr_ratio[1]:
            max_corr_ratio = (n, cr)
        if z <= 0:
            all_positive = False

        status = "OK" if z > 0 else "*** NEG ***"
        if n <= 20 or n % 20 == 0 or z < 0.3 or n in [min_zeta[0], max_corr_ratio[0]]:
            print(f"  {n:4d}  {z:+16.10f}  {cr:12.8f}  {status:>10s}")

    print(f"\n  === RESULTS ===")
    print(f"  lambda_n^{{zeta}} > 0 for all n=1..{N_MAX}? {'YES' if all_positive else 'NO'}")
    print(f"  Minimum: n={min_zeta[0]}, value={min_zeta[1]:+.12f}")
    print(f"  Peak correction ratio: n={max_corr_ratio[0]}, ratio={max_corr_ratio[1]:.10f}")

    # --- Full decomposition for key values ---
    print(f"\n--- Full decomposition at key n values ---")
    print(f"  {'n':>4s}  {'Gamma':>16s}  {'Zeta':>16s}  {'Total':>16s}  {'margin':>10s}")

    for n in [1, 5, 10, 20, 50, 88, 100, 150, 200]:
        g = float(lambda_from_taylor(n, gc))
        z = float(lambda_from_taylor(n, zc))
        total = g + z
        margin = total / abs(g) * 100 if abs(g) > 1e-10 else float('inf')
        print(f"  {n:4d}  {g:+16.8f}  {z:+16.8f}  {total:+16.8f}  {margin:9.1f}%")

    # --- Oscillation structure ---
    print(f"\n--- Oscillation structure of lambda_n^{{zeta}} ---")
    # Find local minima and maxima
    local_min = []
    local_max = []
    for i in range(1, len(zeta_vals) - 1):
        if zeta_vals[i] < zeta_vals[i-1] and zeta_vals[i] < zeta_vals[i+1]:
            local_min.append((i+1, zeta_vals[i]))
        if zeta_vals[i] > zeta_vals[i-1] and zeta_vals[i] > zeta_vals[i+1]:
            local_max.append((i+1, zeta_vals[i]))

    print(f"  Local minima:")
    for n, v in local_min:
        print(f"    n={n}: {v:+.10f}")
    print(f"  Local maxima:")
    for n, v in local_max:
        print(f"    n={n}: {v:+.10f}")

    # Period of oscillation
    if len(local_min) >= 2:
        periods = [local_min[i+1][0] - local_min[i][0] for i in range(len(local_min)-1)]
        print(f"  Oscillation periods: {periods}")
        print(f"  Average period: {np.mean(periods):.1f}")

    print("\n" + "=" * 70)
