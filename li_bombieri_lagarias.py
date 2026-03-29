"""
Session 18c: Bombieri-Lagarias explicit formula for Li coefficients.

From Bombieri-Lagarias (1999), the Li coefficients decompose as:
  lambda_n = S_0(n) + S_1(n) + S_infty(n)

where:
  S_0(n)     = sum over trivial zeros rho_k = -2k (k=1,2,3,...):
               S_0(n) = sum_{k=1}^inf [1 - (1 + 1/(2k))^n]

  S_1(n)     = sum over non-trivial zeros (= lambda_n by definition)

  S_infty(n) = 'archimedean' contribution involving:
               - n*ln(4*pi)/2  (from pi^{-s/2})
               - n*(1 - gamma)/2  (from Gamma(s/2))
               - 1  (from s(s-1)/2 factor)

Wait, actually the Bombieri-Lagarias formula is more specific.

From their Theorem 1: For the completed xi function,
  lambda_n = sum_rho [1 - (1-1/rho)^n]
where the sum is over ALL zeros of xi (non-trivial only, since xi is entire).

But they also express lambda_n in terms of:
  lambda_n = 1 + n/2 * [ln(4*pi) + gamma - 2] - sum_{k=2}^n C(n,k)*(-1)^k * (1-2^{-k})*zeta(k)/(k-1) + ...

Actually, the clean Bombieri-Lagarias decomposition is:
  lambda_n = S_gamma(n) + S_P(n)
where:
  S_gamma(n) = "Gamma-factor side" = computable from Gamma function
  S_P(n) = "prime side" = sum over prime powers

The key result (their Theorem 2): If we define
  tau_n = lambda_n / n
then under RH:
  tau_n = (1/2)*ln(n/(4*pi*e)) + (1/2)*gamma + 1/n + O(1/n^2)

Let me verify this asymptotic and compute the individual pieces.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, zeta, power, loggamma, euler, fac

mp.dps = 100


def taylor_fft(f, center, n_terms, radius=2.0, n_points=1024):
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


def log_xi(s):
    h = s - 1
    if abs(h) < mpf(10)**(-80):
        return -log(mpf(2))
    sz = h * zeta(s)
    return log(mpf(0.5)) + log(s) + log(sz) - (s / 2) * log(pi) + loggamma(s / 2)


# ─── Trivial zero contribution ───

def S_trivial(n, K_max=1000):
    """
    Sum over trivial zeros of zeta at s = -2k, k=1,2,...

    The trivial zeros contribute to lambda_n through the Hadamard product:
    lambda_n^{trivial} = sum_{k=1}^inf [1 - (1 + 1/(2k))^n]

    Note: 1 + 1/(2k) > 1, so (1+1/(2k))^n grows, making each term negative.
    """
    total = mpf(0)
    for k in range(1, K_max + 1):
        term = 1 - power(1 + mpf(1)/(2*k), n)
        total += term
        # Check convergence
        if k > 10 and abs(term) < mpf(10)**(-30):
            break
    return float(total)


# ─── B-L Asymptotic ───

def tau_n_BL_asymptotic(n):
    """
    Bombieri-Lagarias asymptotic (under RH):
    tau_n = lambda_n / n ~ (1/2)*ln(n) + (1/2)*(gamma - 1 - ln(4*pi)) + 1/n + ...
    """
    c = float(euler) - 1 - np.log(4 * np.pi)
    return (1/2) * np.log(n) + (1/2) * c + 1/n


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 18c: Bombieri-Lagarias Explicit Formula")
    print("=" * 70)

    N_MAX = 100

    # --- Step 1: Compute exact lambda_n ---
    print("\nComputing exact lambda_n via Taylor (100dp, r=2)...")
    xi_coeffs = taylor_fft(log_xi, 1, N_MAX + 5)

    # --- Step 2: Trivial zero contribution ---
    print("\n--- Step 2: Trivial zero contribution S_0(n) ---")
    print(f"  S_0(n) = sum_{{k=1}}^inf [1 - (1+1/(2k))^n]")
    print(f"  Each term is negative (exponentially growing subtraction)")
    print()
    print(f"  {'n':>4s}  {'S_0(n)':>16s}  {'per zero avg':>14s}")
    for n in [1, 2, 5, 10, 20, 50, 100]:
        S0 = S_trivial(n)
        avg = S0 / n  # rough per-zero average
        print(f"  {n:4d}  {S0:+16.6f}  {avg:+14.6f}")

    # --- Step 3: Compare with our Gamma part ---
    print("\n--- Step 3: Trivial zeros vs our Gamma decomposition ---")
    print("  Our Gamma part = smooth_gamma contribution to lambda_n")
    print("  B-L S_0 = trivial zero contribution")
    print("  These should differ by the s(s-1)/2 and pi^{-s/2} factors")
    print()

    def smooth_gamma_part(s):
        return log(mpf(0.5)) + log(s) - (s / 2) * log(pi) + loggamma(s / 2)

    def log_sz(s):
        h = s - 1
        if abs(h) < mpf(10)**(-80):
            return mpf(0)
        return log(h * zeta(s))

    gc = taylor_fft(smooth_gamma_part, 1, N_MAX + 5)
    zc = taylor_fft(log_sz, 1, N_MAX + 5)

    print(f"  {'n':>4s}  {'our Gamma':>14s}  {'S_0(trivial)':>14s}  {'diff':>14s}  {'our Zeta':>14s}")
    for n in [1, 2, 5, 10, 20, 50]:
        g = lambda_from_taylor(n, gc)
        z = lambda_from_taylor(n, zc)
        S0 = S_trivial(n)
        diff = g - S0
        print(f"  {n:4d}  {g:+14.6f}  {S0:+14.6f}  {diff:+14.6f}  {z:+14.6f}")

    # --- Step 4: Asymptotic test ---
    print("\n--- Step 4: B-L asymptotic tau_n = lambda_n/n ---")
    print("  Expected (under RH): (1/2)ln(n) + (1/2)(gamma - 1 - ln(4*pi)) + 1/n")
    c = float(euler) - 1 - np.log(4 * np.pi)
    print(f"  constant c = gamma - 1 - ln(4*pi) = {c:.6f}")
    print()
    print(f"  {'n':>4s}  {'tau_n exact':>14s}  {'B-L asymp':>14s}  {'error':>12s}  {'error/n':>10s}")

    for n in range(1, N_MAX + 1):
        lam = lambda_from_taylor(n, xi_coeffs)
        tau = lam / n
        bl = tau_n_BL_asymptotic(n)
        err = tau - bl
        err_n = err * n
        if n <= 20 or n % 10 == 0:
            print(f"  {n:4d}  {tau:+14.8f}  {bl:+14.8f}  {err:+12.6f}  {err_n:+10.4f}")

    # --- Step 5: The O(1/n) correction ---
    print("\n--- Step 5: Higher-order corrections ---")
    print("  Define R(n) = tau_n - [(1/2)ln(n) + c/2 + 1/n]")
    print("  Under RH, R(n) should be O(1/n^2)")
    print()
    print(f"  {'n':>4s}  {'R(n)':>14s}  {'n*R(n)':>14s}  {'n^2*R(n)':>14s}")
    for n in range(5, N_MAX + 1, 5):
        lam = lambda_from_taylor(n, xi_coeffs)
        tau = lam / n
        bl = tau_n_BL_asymptotic(n)
        R = tau - bl
        print(f"  {n:4d}  {R:+14.8f}  {n*R:+14.6f}  {n*n*R:+14.4f}")

    # --- Step 6: What does this mean for provability? ---
    print("\n--- Step 6: Implications ---")
    print()
    print("  The B-L asymptotic shows that UNDER RH:")
    print("    lambda_n = (n/2)*ln(n) + (n/2)*(gamma-1-ln(4*pi)) + 1 + O(1/n)")
    print()
    print("  The leading term (n/2)*ln(n) matches our Gamma part.")
    print("  The second term (n/2)*c where c = -2.567 is NEGATIVE,")
    print("  pulling lambda_n below (n/2)*ln(n).")
    print()
    print("  Our decomposition: Gamma + Zeta = total")
    print("  B-L decomposition: S_infty + S_0 + S_1 = total")
    print()
    print("  The difference: our 'Gamma' includes the trivial zeros,")
    print("  while B-L separates them into S_0.")
    print()

    # --- Step 7: Can we isolate S_1 (non-trivial zero contribution)? ---
    print("--- Step 7: Non-trivial zero contribution ---")
    print("  S_1(n) = lambda_n - S_0(n) - S_infty(n)")
    print("  = sum_rho [1-(1-1/rho)^n] over non-trivial zeros")
    print()
    print("  This is just our lambda_from_zeros computation from Session 17.")
    print("  Under RH, S_1(n) is the oscillatory part.")
    print()

    # Compute S_1 as total - S_0
    print(f"  {'n':>4s}  {'lambda_n':>14s}  {'S_0':>14s}  {'S_1 = lam - S_0':>16s}")
    for n in [1, 5, 10, 20, 50, 100]:
        lam = lambda_from_taylor(n, xi_coeffs)
        S0 = S_trivial(n)
        S1 = lam - S0
        print(f"  {n:4d}  {lam:+14.6f}  {S0:+14.6f}  {S1:+16.6f}")

    # --- Step 8: Keiper-Li tau_n analysis ---
    print("\n--- Step 8: Keiper-Li oscillation in tau_n ---")
    print("  tau_n = lambda_n/n should be monotonically approaching")
    print("  (1/2)*ln(n) + c/2 from above (under RH).")
    print()
    tau_vals = [lambda_from_taylor(n, xi_coeffs) / n for n in range(1, N_MAX + 1)]
    print(f"  {'n':>4s}  {'tau_n':>14s}  {'tau_{n+1}-tau_n':>16s}  {'monotone?':>10s}")
    for n in range(1, min(30, N_MAX)):
        diff = tau_vals[n] - tau_vals[n-1]
        mono = "YES" if diff > 0 else "no"
        if n <= 25:
            print(f"  {n:4d}  {tau_vals[n-1]:+14.8f}  {diff:+16.8f}  {mono:>10s}")

    n_mono = sum(1 for i in range(1, len(tau_vals)) if tau_vals[i] > tau_vals[i-1])
    print(f"\n  Monotone increases: {n_mono} of {N_MAX-1}")
    print(f"  Non-monotone: tau_n is {'mostly' if n_mono > N_MAX*0.8 else 'NOT'} increasing")

    print("\n" + "=" * 70)
