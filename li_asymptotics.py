"""
Session 18: Asymptotic analysis of Li coefficient components.

GOAL: Determine growth rate of |lambda_n^zeta| and threshold N_0
      for Gamma dominance.

Theory:
  lambda_n^zeta / n = [x^n] F(x)  where  F(x) = log((s-1)*zeta(s))|_{s=1/(1-x)}

  F(x) has logarithmic singularities at x = 1-1/rho for each non-trivial zero rho.
  Under RH: |1-1/rho| = 1, so singularities lie ON the unit circle.

  Near x = w_rho = 1-1/rho:
    F(x) ~ -log(x - w_rho) + analytic
  (logarithmic branch point from the zero of (s-1)*zeta(s)).

  For a function with logarithmic singularities at w_j on |x|=1:
    [x^n] F(x) ~ -sum_j w_j^{-n} / n    (leading asymptotics)

  So: lambda_n^zeta ~ -sum_rho (1-1/rho)^{-n}
                     = -sum_rho (rho/(rho-1))^n

  Under RH, rho = 1/2+i*gamma, rho/(rho-1) = (1/2+ig)/(-1/2+ig) = -(1+2ig)/(1-2ig)
  which has absolute value 1. So:
    lambda_n^zeta ~ -sum_{gamma>0} 2*Re[(rho/(rho-1))^n]

  This is an OSCILLATORY SUM of unit-magnitude terms.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, zeta, power, loggamma, euler

mp.dps = 50


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
        coeffs.append(ck.real)
    return coeffs


def lambda_from_taylor(n, coeffs):
    total = mpf(0)
    for j in range(n):
        k = n - j
        if k < len(coeffs):
            total += mpmath.binomial(n - 1, j) * coeffs[k]
    return float(n * total)


def log_sz(s):
    h = s - 1
    if abs(h) < mpf(10)**(-40):
        return mpf(0)
    return log(h * zeta(s))

def smooth_gamma_part(s):
    return log(mpf(0.5)) + log(s) - (s / 2) * log(pi) + loggamma(s / 2)


# ─── Asymptotic formula from zeros ───

def lambda_zeta_asymptotic(n, gammas):
    """
    Asymptotic: lambda_n^zeta ~ -sum_{gamma>0} 2*Re[(rho/(rho-1))^n]

    rho = 1/2 + i*gamma
    rho/(rho-1) = (1/2+ig)/(-1/2+ig) = -(1+2ig)/(1-2ig)

    |rho/(rho-1)| = 1, so each term oscillates on the unit circle.
    arg(rho/(rho-1)) = pi - 2*arctan(2*gamma) + pi = 2*pi - 2*arctan(2*gamma)
    Wait, let me compute directly.
    """
    total = 0.0
    for g in gammas:
        rho = 0.5 + 1j * g
        w = rho / (rho - 1)  # unit magnitude
        term = w**n
        total += 2 * term.real
    return -total


def lambda_zeta_asymptotic_partial(n, gammas, N_zeros):
    """Use only the first N_zeros zeros."""
    return lambda_zeta_asymptotic(n, gammas[:N_zeros])


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 18: Asymptotics of Li Coefficient Components")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")
    print(f"Loaded {len(gammas)} zeros")

    # --- Step 1: Test asymptotic formula ---
    print("\n--- Step 1: Asymptotic formula vs exact (Taylor) ---")
    print("  lambda_n^zeta ~ -sum 2*Re[(rho/(rho-1))^n]")
    print("  This should match exact values for large n.")

    mp.dps = 200
    N_MAX = 200
    print(f"\n  Computing exact Taylor values (200dp, r=2)...")
    zc = taylor_fft(log_sz, 1, N_MAX + 5, radius=2.0, n_points=1024)
    gc = taylor_fft(smooth_gamma_part, 1, N_MAX + 5, radius=2.0, n_points=1024)
    mp.dps = 50

    print(f"\n  {'n':>4s}  {'Exact':>14s}  {'Asymp(500z)':>14s}  {'Asymp(100z)':>14s}  {'Asymp(20z)':>14s}")
    for n in [1, 5, 10, 20, 50, 88, 100, 150, 172, 200]:
        exact = lambda_from_taylor(n, zc)
        asym500 = lambda_zeta_asymptotic(n, gammas)
        asym100 = lambda_zeta_asymptotic_partial(n, gammas, 100)
        asym20 = lambda_zeta_asymptotic_partial(n, gammas, 20)
        print(f"  {n:4d}  {exact:+14.6f}  {asym500:+14.6f}  {asym100:+14.6f}  {asym20:+14.6f}")

    # --- Step 2: Growth rate analysis ---
    print("\n--- Step 2: Growth rate of |lambda_n^zeta| ---")
    print("  If the sum is 'random walk' over N(T)~T*log(T) zeros at height T~n:")
    print("  Expected: |lambda_n^zeta| ~ sqrt(N(n)) ~ sqrt(n*log(n))")
    print()
    print(f"  {'n':>4s}  {'|exact|':>12s}  {'sqrt(n*ln n)':>14s}  {'ratio':>10s}  {'n*ln(n)':>10s}")

    for n in range(10, N_MAX + 1, 10):
        z = lambda_from_taylor(n, zc)
        sqrtNln = np.sqrt(n * np.log(n)) if n > 1 else 1
        nln = n * np.log(n)
        ratio = abs(z) / sqrtNln if sqrtNln > 0 else 0
        print(f"  {n:4d}  {abs(z):12.6f}  {sqrtNln:14.6f}  {ratio:10.6f}  {nln:10.2f}")

    # --- Step 3: Gamma part asymptotics ---
    print("\n--- Step 3: Gamma part asymptotics ---")
    print("  lambda_n^Gamma should be ~ (n/2)*ln(n) + (n/2)*(gamma-1-ln(2*pi)) + O(1)")
    print("  From Stirling: psi(s/2) ~ ln(s/2) - 1/s + ...")
    print()
    print(f"  {'n':>4s}  {'Exact Gamma':>14s}  {'(n/2)ln(n)':>14s}  {'ratio':>10s}  {'diff':>14s}")

    for n in range(10, N_MAX + 1, 10):
        g = lambda_from_taylor(n, gc)
        pred = (n / 2) * np.log(n)
        ratio = g / pred if pred > 0 else 0
        diff = g - pred
        print(f"  {n:4d}  {g:+14.6f}  {pred:14.6f}  {ratio:10.6f}  {diff:+14.6f}")

    # Better asymptotic: (n/2)*ln(n) + (n/2)*(gamma_EM - 1 + ln(2)) + ...
    print(f"\n  Refined asymptotic: (n/2)*[ln(n) + gamma - 1 + ln(2)]")
    c = float(euler) - 1 + np.log(2)
    print(f"  constant c = gamma - 1 + ln(2) = {c:.6f}")
    print()
    print(f"  {'n':>4s}  {'Exact Gamma':>14s}  {'Refined':>14s}  {'error':>12s}")
    for n in [50, 100, 150, 200]:
        g = lambda_from_taylor(n, gc)
        refined = (n / 2) * (np.log(n) + c)
        print(f"  {n:4d}  {g:+14.6f}  {refined:+14.6f}  {g - refined:+12.6f}")

    # --- Step 4: Dominance threshold ---
    print("\n--- Step 4: Gamma dominance threshold N_0 ---")
    print("  Find smallest N_0 such that lambda_n^Gamma > |lambda_n^zeta| for all n >= N_0")
    print()

    N0 = None
    all_dominated = True
    for n in range(1, N_MAX + 1):
        g = lambda_from_taylor(n, gc)
        z = lambda_from_taylor(n, zc)
        if g <= abs(z):
            all_dominated = False
            N0 = None
        elif not all_dominated and N0 is None:
            # First n where Gamma starts dominating again
            all_dominated = True
            N0 = n

    if N0 is not None:
        # Verify it holds from N0 to N_MAX
        holds = True
        for n in range(N0, N_MAX + 1):
            g = lambda_from_taylor(n, gc)
            z = lambda_from_taylor(n, zc)
            if g <= abs(z):
                holds = False
                N0 = None
                break

    if N0 is not None:
        print(f"  N_0 = {N0} (Gamma dominates for all n >= {N0} through n={N_MAX})")
    else:
        # Find where Gamma < |zeta|
        trouble_spots = []
        for n in range(1, N_MAX + 1):
            g = lambda_from_taylor(n, gc)
            z = lambda_from_taylor(n, zc)
            if g <= abs(z):
                trouble_spots.append(n)
        print(f"  Gamma does NOT dominate at n = {trouble_spots[:20]}...")
        if trouble_spots:
            print(f"  Last trouble spot: n = {trouble_spots[-1]}")
            N0_candidate = trouble_spots[-1] + 1
            print(f"  Candidate N_0 = {N0_candidate}")

    # --- Step 5: Margin of safety ---
    print("\n--- Step 5: Safety margin lambda_n / |lambda_n^Gamma| ---")
    print("  (How much room does the total have above zero?)")
    print(f"  {'n':>4s}  {'total':>14s}  {'|Gamma|':>14s}  {'margin%':>10s}")
    for n in [1, 2, 3, 5, 88, 100, 155, 172, 186, 200]:
        g = lambda_from_taylor(n, gc)
        z = lambda_from_taylor(n, zc)
        total = g + z
        margin = total / abs(max(abs(g), abs(z))) * 100
        print(f"  {n:4d}  {total:+14.6f}  {abs(g):14.6f}  {margin:9.1f}%")

    # --- Step 6: Phase portrait of w = rho/(rho-1) ---
    print("\n--- Step 6: Phase portrait (what drives oscillation) ---")
    print("  w_rho = rho/(rho-1), |w|=1, arg(w) = pi + 2*arctan(1/(2*gamma))")
    print()
    print(f"  {'zero#':>5s}  {'gamma':>10s}  {'arg(w)/pi':>10s}  {'period':>10s}")
    for i in range(15):
        g = gammas[i]
        rho = 0.5 + 1j * g
        w = rho / (rho - 1)
        alpha = np.angle(w) / np.pi  # in units of pi
        period = 2 * np.pi / abs(np.angle(w))
        print(f"  {i+1:5d}  {g:10.4f}  {alpha:+10.6f}  {period:10.1f}")

    # --- Step 7: Random walk model ---
    print("\n--- Step 7: Random walk model for lambda_n^zeta ---")
    print("  If phases arg(w_rho) are 'pseudo-random', then:")
    print("  lambda_n^zeta = -2 * sum Re(w_rho^n) is a random walk")
    print("  with N(T) ~ T*ln(T)/2pi steps, each of size ~1")
    print()
    print("  Predicted RMS: sqrt(N) where N = n*ln(n)/(2pi)")
    print("  (Using T~n for the effective zero height)")
    print()

    # Compute empirical RMS over windows
    zeta_vals = [lambda_from_taylor(n, zc) for n in range(1, N_MAX + 1)]

    # Window RMS
    window = 20
    print(f"  {'center':>6s}  {'window RMS':>12s}  {'sqrt(n*ln n/2pi)':>18s}  {'ratio':>10s}")
    for center in range(20, N_MAX - 10, 20):
        start = max(0, center - window // 2)
        end = min(N_MAX, center + window // 2)
        window_vals = zeta_vals[start:end]
        rms = np.sqrt(np.mean(np.array(window_vals) ** 2))
        pred = np.sqrt(center * np.log(center) / (2 * np.pi))
        ratio = rms / pred if pred > 0 else 0
        print(f"  {center:6d}  {rms:12.6f}  {pred:18.6f}  {ratio:10.4f}")

    print("\n" + "=" * 70)
    print("Session 18 — Asymptotic analysis complete.")
    print("=" * 70)
