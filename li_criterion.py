"""
Session 17: Li's criterion for the Riemann Hypothesis.

Li (1997): RH iff lambda_n > 0 for all n >= 1, where
  lambda_n = sum_rho [1 - (1 - 1/rho)^n]

Strategy: Compute Taylor coefficients of log xi(s) around s=1 via
Cauchy integral (FFT-based), then extract lambda_n via convolution.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, zeta, power, fac, loggamma

mp.dps = 50


def primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


# ─── Core functions ───

def log_xi(s):
    """log(xi(s)): smooth everywhere, pole cancellation handled."""
    h = s - 1
    if abs(h) < mpf(10)**(-40):
        return -log(mpf(2))
    sz = h * zeta(s)
    return log(mpf(0.5)) + log(s) + log(sz) - (s/2) * log(pi) + loggamma(s/2)


def smooth_gamma_part(s):
    """Gamma side of log xi without the log(s-1) singularity.
    = log(1/2) + log(s) - (s/2)*log(pi) + loggamma(s/2)"""
    return log(mpf(0.5)) + log(s) - (s/2) * log(pi) + loggamma(s/2)


def log_sz(s):
    """log((s-1)*zeta(s)): entire function encoding the primes."""
    h = s - 1
    if abs(h) < mpf(10)**(-40):
        return mpf(0)
    return log(h * zeta(s))


# ─── FFT-based Taylor coefficients (Cauchy integral) ───

def taylor_fft(f, center, n_terms, radius=0.4, n_points=None):
    """
    Compute Taylor coefficients of f around center via Cauchy integral.

    c_k = (1/(2*pi*i)) * integral f(z)/(z-center)^{k+1} dz

    Discretized: c_k = (1/N) * sum_{j=0}^{N-1} f(center + r*w^j) * w^{-jk} / r^k
    where w = exp(2*pi*i/N).

    This is just FFT of the function values on a circle.
    """
    if n_points is None:
        n_points = max(2 * n_terms + 16, 64)

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
        coeffs.append(ck.real if abs(ck.imag) < mpf(10)**(-20) else ck)
    return coeffs


# ─── Lambda from Taylor coefficients ───

def lambda_from_taylor(n, coeffs):
    """lambda_n = n * sum_{j=0}^{n-1} C(n-1,j) * c_{n-j}"""
    total = mpf(0)
    for j in range(n):
        k = n - j
        if k < len(coeffs):
            total += mpmath.binomial(n - 1, j) * coeffs[k]
    return n * total


# ─── Lambda from zeros ───

def lambda_n_from_zeros(n, gammas):
    """lambda_n = sum_{gamma>0} 2*Re[1 - (1 - 1/rho)^n]"""
    total = mpf(0)
    for g in gammas:
        rho = mpc(0.5, g)
        term = 1 - power(1 - 1/rho, n)
        total += 2 * term.real
    return total


# ─── Main ───

if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 17: Li's Criterion — Dragon Hunting")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")
    print(f"\nLoaded {len(gammas)} zeros")

    N_MAX = 50

    # --- Step 1: Taylor coefficients via FFT ---
    print("\n--- Step 1: Taylor coefficients of log xi(s) around s=1 (FFT method) ---")
    coeffs_xi = taylor_fft(log_xi, 1, N_MAX + 5, radius=0.4, n_points=256)
    print(f"  c_0 = {float(coeffs_xi[0]):+.15f}  (expected -ln 2 = {float(-log(2)):.15f})")
    for k in range(1, 8):
        print(f"  c_{k} = {float(coeffs_xi[k]):+.15f}")

    # --- Step 2: Lambda_n from Taylor ---
    print(f"\n--- Step 2: Li coefficients (Taylor method) n=1..{N_MAX} ---")
    lambdas_taylor = []
    for n in range(1, N_MAX + 1):
        val = float(lambda_from_taylor(n, coeffs_xi))
        lambdas_taylor.append(val)
        if n <= 20 or n % 10 == 0:
            print(f"  lambda_{n:3d} = {val:+.10f}  {'> 0 OK' if val > 0 else '*** NEGATIVE ***'}")

    # --- Step 3: Verification ---
    print("\n--- Step 3: Verification ---")
    exact_l1 = float(1 + euler/2 - log(4*pi)/2)
    taylor_l1 = lambdas_taylor[0]
    zeros_l1 = float(lambda_n_from_zeros(1, gammas))
    print(f"  Exact formula:  lambda_1 = {exact_l1:.15f}")
    print(f"  Taylor (FFT):   lambda_1 = {taylor_l1:.15f}")
    print(f"  From 500 zeros: lambda_1 = {zeros_l1:.15f}")

    print(f"\n  {'n':>3s}  {'Taylor':>16s}  {'Zeros(500)':>16s}  {'diff':>12s}")
    for n in range(1, 11):
        z_val = float(lambda_n_from_zeros(n, gammas))
        t_val = lambdas_taylor[n-1]
        print(f"  {n:3d}  {t_val:+16.10f}  {z_val:+16.10f}  {t_val - z_val:+12.6e}")

    # --- Step 4: Gamma vs Zeta decomposition ---
    print("\n--- Step 4: Gamma vs Zeta decomposition ---")
    print("  Split: log xi = smooth_gamma(s) + log((s-1)*zeta(s))")
    print("  smooth_gamma = log(1/2) + log(s) - (s/2)log(pi) + log Gamma(s/2)")
    print("  log_sz       = log((s-1)*zeta(s))  [encodes prime distribution]")

    gamma_coeffs = taylor_fft(smooth_gamma_part, 1, N_MAX + 5, radius=0.4, n_points=256)
    zeta_coeffs = taylor_fft(log_sz, 1, N_MAX + 5, radius=0.4, n_points=256)

    print(f"\n  Gamma c_0 = {float(gamma_coeffs[0]):+.15f}")
    print(f"  Zeta  c_0 = {float(zeta_coeffs[0]):+.15f}")
    print(f"  Sum       = {float(gamma_coeffs[0]+zeta_coeffs[0]):+.15f}  (total c_0 = {float(coeffs_xi[0]):+.15f})")

    print(f"\n  {'n':>3s}  {'Gamma':>14s}  {'Zeta':>14s}  {'Total':>14s}  {'Gamma%':>8s}")
    decomp_data = []
    for n in range(1, 31):
        g_val = float(lambda_from_taylor(n, gamma_coeffs))
        z_val = float(lambda_from_taylor(n, zeta_coeffs))
        total = g_val + z_val
        g_pct = 100 * g_val / total if abs(total) > 1e-15 else 0
        decomp_data.append((n, g_val, z_val, total))
        if n <= 20 or n % 5 == 0:
            print(f"  {n:3d}  {g_val:+14.8f}  {z_val:+14.8f}  {total:+14.8f}  {g_pct:7.1f}%")

    # --- Step 5: Growth rate ---
    print("\n--- Step 5: Growth rate (RH predicts lambda_n ~ (n/2)*ln(n)) ---")
    print(f"  {'n':>3s}  {'lambda_n':>14s}  {'(n/2)ln(n)':>14s}  {'ratio':>10s}")
    for n in [5, 10, 20, 30, 40, 50]:
        predicted = (n/2) * np.log(n)
        actual = lambdas_taylor[n-1]
        ratio = actual / predicted if predicted > 0 else 0
        print(f"  {n:3d}  {actual:+14.6f}  {predicted:14.6f}  {ratio:10.4f}")

    # --- Step 6: Positivity anatomy ---
    print("\n--- Step 6: Positivity anatomy ---")
    print("  KEY QUESTION: Is Gamma dominance (from M-function, Session 16)")
    print("  visible in the lambda_n basis?")
    print(f"\n  {'n':>3s}  {'Gamma>0?':>9s}  {'Zeta>0?':>9s}  {'Driver':>14s}  {'|Zeta/Gamma|':>14s}")
    for n, g, z, t in decomp_data[:20]:
        g_pos = "YES" if g > 0 else "no"
        z_pos = "YES" if z > 0 else "no"
        if g > 0 and z > 0:
            driver = "both"
        elif g > 0 and z <= 0:
            driver = "GAMMA wins"
        elif g <= 0 and z > 0:
            driver = "ZETA wins"
        else:
            driver = "both neg"
        ratio = abs(z/g) if abs(g) > 1e-15 else float('inf')
        print(f"  {n:3d}  {g_pos:>9s}  {z_pos:>9s}  {driver:>14s}  {ratio:14.6f}")

    # --- Step 7: Crossover analysis ---
    print("\n--- Step 7: Crossover analysis ---")
    print("  At what n does the zeta part become significant?")
    for n, g, z, t in decomp_data:
        if abs(z) > 0.1 * abs(g):
            print(f"  n={n}: |zeta/gamma| = {abs(z/g):.4f} — zeta part is >10% of gamma")
            break

    # --- Step 8: Connection to M-function ---
    print("\n--- Step 8: M-function connection ---")
    print("  M(s) = Im[(s-1/2) * conj(xi'/xi(s))] > 0 iff RH (pointwise)")
    print("  lambda_n tests the same positivity in a discrete (power) basis")
    print()
    print("  The Taylor coefficient c_k of log xi at s=1 is related to")
    print("  the k-th Stieltjes constant and the k-th moment of 1/rho.")
    print()

    # Compute: what fraction of lambda_n comes from large vs small zeros?
    print("  Convergence of lambda_n from zeros (how many zeros needed):")
    for n in [1, 5, 10, 20]:
        vals = []
        for N_z in [10, 50, 100, 200, 500]:
            v = float(lambda_n_from_zeros(n, gammas[:N_z]))
            vals.append(v)
        exact_v = lambdas_taylor[n-1]
        print(f"  n={n:2d}: 10z={vals[0]:+.6f}  50z={vals[1]:+.6f}  100z={vals[2]:+.6f}  "
              f"200z={vals[3]:+.6f}  500z={vals[4]:+.6f}  exact={exact_v:+.6f}")

    # --- Step 9: Per-zero contribution profile ---
    print("\n--- Step 9: Per-zero contribution to lambda_n ---")
    print("  Which zeros contribute most to positivity?")
    for n in [1, 10, 20]:
        contribs = []
        for i, g in enumerate(gammas[:50]):
            rho = mpc(0.5, g)
            c = 2 * (1 - power(1 - 1/rho, n)).real
            contribs.append((i+1, g, float(c)))
        top5 = sorted(contribs, key=lambda x: abs(x[2]), reverse=True)[:5]
        print(f"  n={n}: top contributing zeros:")
        for idx, g, c in top5:
            print(f"    zero #{idx} (gamma={g:.4f}): contribution = {c:+.8f}")

    print("\n" + "=" * 70)
    print("Session 17 — Li coefficients computed and decomposed.")
    print("=" * 70)
