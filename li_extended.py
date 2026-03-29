"""
Session 17c: Extended Li coefficient analysis.

Key question: Does lambda_n^{zeta} stay positive for large n?
If yes, what drives this positivity?

Also: Weil explicit formula connection.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, zeta, power, fac, loggamma

mp.dps = 80  # extra precision for large n


def taylor_fft(f, center, n_terms, radius=0.4, n_points=None):
    if n_points is None:
        n_points = max(2 * n_terms + 32, 128)
    r = mpf(radius)
    values = []
    for j in range(n_points):
        theta = 2 * mpf(pi) * j / n_points
        z = mpc(center, 0) + r * mpc(mpmath.cos(theta), mpmath.sin(theta))
        values.append(f(z))
    coeffs = []
    for k in range(n_terms + 1):
        ck = mpc(0, 0)
        for j in range(n_points):
            theta = 2 * mpf(pi) * j / n_points
            ck += values[j] * mpc(mpmath.cos(-k * theta), mpmath.sin(-k * theta))
        ck /= n_points
        ck /= power(r, k)
        coeffs.append(ck.real if abs(ck.imag) < mpf(10)**(-30) else ck)
    return coeffs


def lambda_from_taylor(n, coeffs):
    total = mpf(0)
    for j in range(n):
        k = n - j
        if k < len(coeffs):
            total += mpmath.binomial(n - 1, j) * coeffs[k]
    return n * total


def log_xi(s):
    h = s - 1
    if abs(h) < mpf(10)**(-40):
        return -log(mpf(2))
    sz = h * zeta(s)
    return log(mpf(0.5)) + log(s) + log(sz) - (s/2) * log(pi) + loggamma(s/2)

def smooth_gamma_part(s):
    return log(mpf(0.5)) + log(s) - (s/2) * log(pi) + loggamma(s/2)

def log_sz(s):
    h = s - 1
    if abs(h) < mpf(10)**(-40):
        return mpf(0)
    return log(h * zeta(s))


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 17c: Extended Li Coefficient Analysis")
    print("=" * 70)

    N_MAX = 100

    # Compute Taylor coefficients
    print(f"\nComputing Taylor coefficients (n={N_MAX}, 80 dp)...")
    print("  Total log xi...")
    coeffs_xi = taylor_fft(log_xi, 1, N_MAX + 5, radius=0.35, n_points=512)
    print("  Gamma part...")
    gamma_coeffs = taylor_fft(smooth_gamma_part, 1, N_MAX + 5, radius=0.35, n_points=512)
    print("  Zeta part...")
    zeta_coeffs = taylor_fft(log_sz, 1, N_MAX + 5, radius=0.35, n_points=512)

    # Verify c_0
    print(f"\n  c_0 check: {float(coeffs_xi[0]):+.15f} (expected {float(-log(2)):+.15f})")
    print(f"  c_1 check: {float(coeffs_xi[1]):+.15f} (expected lambda_1 = {float(1+euler/2-log(4*pi)/2):+.15f})")

    # --- Comprehensive table ---
    print(f"\n--- Li coefficients: n=1..{N_MAX} ---")
    print(f"  {'n':>4s}  {'lambda_n':>14s}  {'Gamma':>14s}  {'Zeta':>14s}  {'Zeta>0?':>7s}  {'Driver':>12s}")

    all_zeta_pos = True
    min_zeta = (0, float('inf'))
    max_gamma_neg = (0, 0.0)

    for n in range(1, N_MAX + 1):
        lam = float(lambda_from_taylor(n, coeffs_xi))
        g = float(lambda_from_taylor(n, gamma_coeffs))
        z = float(lambda_from_taylor(n, zeta_coeffs))

        z_pos = z > 0
        if not z_pos:
            all_zeta_pos = False
        if z < min_zeta[1]:
            min_zeta = (n, z)
        if g < max_gamma_neg[1] and g < 0:
            max_gamma_neg = (n, g)

        if g > 0 and z > 0:
            driver = "both"
        elif g > 0 and z <= 0:
            driver = "GAMMA wins"
        elif g <= 0 and z > 0:
            driver = "ZETA wins"
        else:
            driver = "DANGER"

        if n <= 30 or n % 10 == 0:
            z_str = "YES" if z_pos else "*** NO ***"
            print(f"  {n:4d}  {lam:+14.8f}  {g:+14.8f}  {z:+14.8f}  {z_str:>7s}  {driver:>12s}")

    print(f"\n  === SUMMARY ===")
    print(f"  lambda_n^{{zeta}} > 0 for ALL n=1..{N_MAX}? {'YES' if all_zeta_pos else '*** NO ***'}")
    print(f"  Minimum zeta part: n={min_zeta[0]}, value={min_zeta[1]:+.10f}")
    print(f"  Most negative Gamma: n={max_gamma_neg[0]}, value={max_gamma_neg[1]:+.10f}")

    # --- Asymptotic behavior ---
    print(f"\n--- Asymptotic analysis ---")
    print(f"  {'n':>4s}  {'lambda_n':>14s}  {'(n/2)ln(n)':>14s}  {'ratio':>10s}  {'Zeta/n':>10s}")
    for n in range(10, N_MAX + 1, 10):
        lam = float(lambda_from_taylor(n, coeffs_xi))
        g = float(lambda_from_taylor(n, gamma_coeffs))
        z = float(lambda_from_taylor(n, zeta_coeffs))
        predicted = (n/2) * np.log(n)
        ratio = lam / predicted if predicted > 0 else 0
        zeta_per_n = z / n
        print(f"  {n:4d}  {lam:+14.6f}  {predicted:14.6f}  {ratio:10.4f}  {zeta_per_n:10.6f}")

    # --- Critical ratio: |zeta| / |gamma| for small n ---
    print(f"\n--- Critical ratios (small n where Gamma < 0) ---")
    print(f"  For n=1..7, positivity requires |zeta| > |gamma|")
    print(f"  {'n':>3s}  {'|Zeta|':>12s}  {'|Gamma|':>12s}  {'margin':>12s}  {'safety':>8s}")
    for n in range(1, 8):
        g = float(lambda_from_taylor(n, gamma_coeffs))
        z = float(lambda_from_taylor(n, zeta_coeffs))
        margin = z + g  # total lambda_n
        safety = margin / abs(g) * 100  # % safety margin
        print(f"  {n:3d}  {abs(z):12.8f}  {abs(g):12.8f}  {margin:+12.8f}  {safety:7.1f}%")

    # --- Ratio test for lambda_n^{zeta} ---
    print(f"\n--- Ratio test: is zeta part monotonically structured? ---")
    print(f"  {'n':>3s}  {'zeta_n':>14s}  {'zeta_{n+1}/zeta_n':>18s}")
    prev = None
    for n in range(1, 51):
        z = float(lambda_from_taylor(n, zeta_coeffs))
        if prev is not None and abs(prev) > 1e-15:
            ratio = z / prev
            if n <= 20 or n % 5 == 0:
                print(f"  {n:3d}  {z:+14.8f}  {ratio:18.6f}")
        else:
            if n <= 20 or n % 5 == 0:
                print(f"  {n:3d}  {z:+14.8f}  {'---':>18s}")
        prev = z

    # --- Connection: Weil explicit formula ---
    print(f"\n--- Weil explicit formula connection ---")
    print("  The Bombieri-Lagarias (1999) explicit formula:")
    print("  lambda_n = S_0(n) + S_1(n) + S_infty(n)")
    print("  where:")
    print("    S_0(n)     = sum over trivial zeros (computable)")
    print("    S_1(n)     = sum over non-trivial zeros (= lambda_n by definition)")
    print("    S_infty(n) = 'archimedean' (Gamma factor) contribution")
    print()
    print("  Our decomposition: lambda_n = lambda_n^Gamma + lambda_n^zeta")
    print("  where:")
    print("    lambda_n^Gamma = S_infty(n) + S_0(n)  [Gamma + trivial zeros]")
    print("    lambda_n^zeta = 'prime distribution' contribution")
    print()
    print("  The observation lambda_n^zeta > 0 for all n is equivalent to:")
    print("  'The prime distribution contribution to the explicit formula")
    print("   is always positive in the Li coefficient basis.'")
    print()
    print("  This should relate to the prime number theorem and")
    print("  the distribution of prime powers near x = e^n.")

    # --- Can we identify what happens at large n? ---
    print(f"\n--- Large-n extrapolation ---")
    print("  If lambda_n^zeta ~ C for some constant C > 0 as n -> inf,")
    print("  and lambda_n^Gamma ~ (n/2)*ln(n) -> +inf,")
    print("  then lambda_n = Gamma + zeta -> +inf, consistent with RH.")
    print()
    z_vals = [float(lambda_from_taylor(n, zeta_coeffs)) for n in range(1, N_MAX + 1)]
    print(f"  Zeta part values at n=50,60,70,80,90,100:")
    for n in [50, 60, 70, 80, 90, 100]:
        print(f"    n={n}: {z_vals[n-1]:+.10f}")

    # Check if it's approaching a limit
    last_10 = z_vals[-10:]
    mean = np.mean(last_10)
    std = np.std(last_10)
    print(f"\n  Last 10 values: mean={mean:.6f}, std={std:.6f}")
    print(f"  Appears to {'converge' if std < 0.1 else 'oscillate'}")

    print("\n" + "=" * 70)
