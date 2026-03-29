"""
Session 18b: Truncated Euler product analysis.

Avenue 4 from Grok assessment (scored 1/10 — least explored).

Questions:
  1. M-function with finite Euler product: does M_K(s) > 0?
  2. Li coefficients with finite Euler product: how do they compare?
  3. At what K does the truncated product start matching the full answer?

The truncated Euler product:
  zeta_K(s) = prod_{p <= K} 1/(1-p^{-s})

Key property: zeta_K has NO zeros (it's a finite product of non-vanishing factors).
So (s-1)*zeta_K(s) has no non-trivial zeros, and log((s-1)*zeta_K(s)) is analytic
in a large domain.

But the M-function uses xi'/xi, which involves zeta'/zeta. For the truncated product:
  zeta_K'/zeta_K = sum_{p<=K} (log p) * p^{-s} / (1-p^{-s})
  = sum_{p<=K} sum_{m>=1} (log p) * p^{-ms}
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, zeta, power, loggamma, euler

mp.dps = 50


def primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


# ─── Truncated Euler product ───

def zeta_K(s, primes):
    """Truncated Euler product: prod_{p in primes} 1/(1-p^{-s})"""
    result = mpf(1)
    for p in primes:
        result /= (1 - power(p, -s))
    return result


def log_zeta_K(s, primes):
    """log(zeta_K(s)) = -sum log(1-p^{-s}) = sum sum p^{-ks}/k"""
    result = mpf(0)
    for p in primes:
        result -= log(1 - power(p, -s))
    return result


def zeta_K_log_deriv(s, primes):
    """(zeta_K'/zeta_K)(s) = sum_{p<=K} (log p)*p^{-s} / (1-p^{-s})"""
    result = mpf(0)
    for p in primes:
        lp = log(mpf(p))
        ps = power(p, -s)
        result += lp * ps / (1 - ps)
    return result


# ─── M-function with truncated product ───

def M_function_exact(sigma, t, dps=50):
    """M(s) using full zeta via mpmath."""
    with mpmath.workdps(dps):
        s = mpc(sigma, t)
        def xi_func(sv):
            return sv * (sv-1)/2 * power(pi, -sv/2) * mpmath.gamma(sv/2) * zeta(sv)
        xi_val = xi_func(s)
        if abs(xi_val) < mpf(10)**(-dps+5):
            return float('nan')
        xi_d = mpmath.diff(xi_func, s)
        f = xi_d / xi_val
        u = s - mpf(0.5)
        return float((u * mpmath.conj(f)).imag)


def xi_K_log_deriv(s, primes):
    """(xi_K'/xi_K)(s) using truncated Euler product.

    xi_K(s) = (s(s-1)/2) * pi^{-s/2} * Gamma(s/2) * zeta_K(s)

    xi_K'/xi_K = 1/s + 1/(s-1) - log(pi)/2 + psi(s/2)/2 + zeta_K'/zeta_K
    """
    # Gamma contribution
    f_gamma = 1/s + 1/(s-1) - log(pi)/2 + mpmath.digamma(s/2)/2
    # Euler product contribution
    f_euler = zeta_K_log_deriv(s, primes)
    return f_gamma + f_euler


def M_function_K(sigma, t, primes):
    """M(s) using truncated Euler product."""
    s = mpc(sigma, t)
    f = xi_K_log_deriv(s, primes)
    u = s - mpf(0.5)
    return float((u * mpmath.conj(f)).imag)


# ─── Li coefficients with truncated product ───

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


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 18b: Truncated Euler Product Analysis")
    print("=" * 70)

    all_primes = primes_up_to(1000)
    print(f"Primes available: {len(all_primes)} (up to {all_primes[-1]})")

    # --- Step 1: M-function comparison ---
    print("\n--- Step 1: M-function: exact vs truncated Euler product ---")
    print("  Test points: sigma=0.8, t=15 (near first zero)")
    print("               sigma=1.5, t=10 (right of critical strip)")

    test_points = [(0.8, 15.0), (0.8, 25.0), (1.5, 10.0), (0.6, 50.0), (0.75, 14.1)]

    for sigma, t in test_points:
        M_exact = M_function_exact(sigma, t)
        print(f"\n  s = {sigma} + {t}i:  M_exact = {M_exact:+.8f}")
        for K in [2, 10, 50, 200, 1000]:
            pk = [p for p in all_primes if p <= K]
            M_K = M_function_K(sigma, t, pk)
            diff = M_K - M_exact
            print(f"    K={K:4d} ({len(pk):3d} primes): M_K = {M_K:+.8f}  diff = {diff:+.2e}")

    # --- Step 2: Does M_K > 0 everywhere? ---
    print("\n--- Step 2: M_K positivity scan ---")
    print("  Since zeta_K has NO zeros, is M_K always positive?")

    for K in [2, 10, 50]:
        pk = [p for p in all_primes if p <= K]
        print(f"\n  K={K} ({len(pk)} primes):")

        # Scan grid
        n_neg = 0
        min_M = float('inf')
        min_pt = None
        for sigma in np.arange(0.55, 2.01, 0.05):
            for t in np.arange(1.0, 100.0, 2.0):
                M = M_function_K(sigma, t, pk)
                if M < min_M:
                    min_M = M
                    min_pt = (sigma, t)
                if M < 0:
                    n_neg += 1

        print(f"    Negative points: {n_neg} of {30*50}")
        print(f"    Minimum M_K: {min_M:+.8f} at s={min_pt[0]:.2f}+{min_pt[1]:.1f}i")

    # --- Step 3: Li coefficients with truncated product ---
    print("\n--- Step 3: Li coefficients lambda_n^zeta from truncated Euler product ---")
    print("  Replacing log((s-1)*zeta(s)) with log((s-1)*zeta_K(s))")

    N_LI = 30

    for K in [2, 10, 50, 200]:
        pk = [p for p in all_primes if p <= K]

        def log_sz_K(s, primes=pk):
            h = s - 1
            if abs(h) < mpf(10)**(-40):
                return mpf(0)
            return log(h * zeta_K(s, primes))

        zk_coeffs = taylor_fft(log_sz_K, 1, N_LI + 3, radius=0.4, n_points=128)

        print(f"\n  K={K} ({len(pk)} primes):")
        all_pos = True
        for n in [1, 5, 10, 20, 30]:
            z = lambda_from_taylor(n, zk_coeffs)
            pos = z > 0
            if not pos:
                all_pos = False
            print(f"    lambda_{n:2d}^zeta(K={K}) = {z:+.8f}  {'OK' if pos else 'NEG'}")
        print(f"    All positive (n=1..{N_LI})? {'YES' if all_pos else 'NO'}")

    # --- Step 4: Compare d_1 coefficient ---
    print("\n--- Step 4: d_1 coefficient (should approach gamma) ---")
    print(f"  Exact: d_1 = gamma = {float(euler):.15f}")
    for K in [2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        pk = [p for p in all_primes if p <= K]

        def log_sz_K(s, primes=pk):
            h = s - 1
            if abs(h) < mpf(10)**(-40):
                return mpf(0)
            return log(h * zeta_K(s, primes))

        zk_coeffs = taylor_fft(log_sz_K, 1, 5, radius=0.4, n_points=64)
        d1 = float(zk_coeffs[1])
        err = abs(d1 - float(euler))
        print(f"  K={K:4d}: d_1 = {d1:.15f}  error = {err:.2e}")

    # --- Step 5: Key insight ---
    print("\n--- Step 5: Interpretation ---")
    print()
    print("  zeta_K(s) = finite product, has NO ZEROS")
    print("  -> (s-1)*zeta_K(s) has zero only at s=1 (removed)")
    print("  -> log((s-1)*zeta_K(s)) is analytic in C")
    print("  -> Its Taylor series converges EVERYWHERE")
    print("  -> Lambda_n^{zeta,K} should be well-behaved for all n")
    print()
    print("  As K -> infinity, zeta_K -> zeta, and the zeros of zeta")
    print("  introduce singularities that create the oscillation in lambda_n^zeta.")
    print()
    print("  The TRANSITION from lambda_n^{zeta,K} (all positive?)")
    print("  to lambda_n^{zeta} (oscillating, sometimes negative)")
    print("  is driven by the EMERGENCE of zeros as K grows.")
    print()
    print("  This connects to the explicit formula: the prime sum converges")
    print("  to the zero sum as more primes are included.")

    print("\n" + "=" * 70)
