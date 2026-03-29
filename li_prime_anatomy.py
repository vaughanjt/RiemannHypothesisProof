"""
Session 17b: Prime anatomy of Li coefficients.

Key finding from li_criterion.py:
  - lambda_n^{zeta} > 0 for ALL n tested (n=1..30)
  - For n=1..7, the zeta part ALONE drives positivity
  - The zeta part encodes the prime distribution

This script investigates:
  1. The Stieltjes constant structure of the zeta part
  2. Per-prime contributions to lambda_n
  3. Whether lambda_n^{zeta} > 0 can be understood from prime structure
  4. Connection to the M-function decomposition from Session 16
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


def taylor_fft(f, center, n_terms, radius=0.4, n_points=None):
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


def lambda_from_taylor(n, coeffs):
    total = mpf(0)
    for j in range(n):
        k = n - j
        if k < len(coeffs):
            total += mpmath.binomial(n - 1, j) * coeffs[k]
    return n * total


# ─── Functions for decomposition ───

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


# ─── Stieltjes constants ───

def compute_stieltjes(n_max):
    """Compute Stieltjes constants gamma_0, gamma_1, ..., gamma_n."""
    stieltjes = []
    for n in range(n_max + 1):
        gn = mpmath.stieltjes(n)
        stieltjes.append(gn)
    return stieltjes


def taylor_of_log_sz_from_stieltjes(n_max):
    """
    Compute Taylor coefficients of log((s-1)*zeta(s)) from Stieltjes constants.

    (s-1)*zeta(s) = 1 + sum_{k=0}^inf gamma_k * h^{k+1} / k!
    where h = s-1.

    Then log((s-1)*zeta(s)) = log(1 + u) where u = sum_{k=0} gamma_k * h^{k+1} / k!

    Using log(1+u) = u - u^2/2 + u^3/3 - ... and expanding powers of u.
    """
    stj = compute_stieltjes(n_max + 5)
    print(f"  Stieltjes constants:")
    for k in range(min(8, len(stj))):
        print(f"    gamma_{k} = {float(stj[k]):+.15f}")

    # Build the Taylor coefficients of u(h) = sum gamma_k * h^{k+1} / k!
    # u has no constant term (u(0) = 0)
    u_coeffs = [mpf(0)]  # u_0 = 0
    for k in range(n_max + 3):
        u_coeffs.append(power(-1, k) * stj[k] / fac(k))  # (-1)^k * gamma_k / k!

    # Now compute log(1 + u) = sum_{m=1} (-1)^{m+1} * u^m / m
    # We need to multiply polynomial powers of u

    # Initialize result with zeros
    d = [mpf(0)] * (n_max + 2)

    # Compute u^m up to order n_max
    u_power = [mpf(0)] * (n_max + 2)
    u_power[0] = mpf(1)  # u^0 = 1

    for m in range(1, n_max + 2):
        # Multiply u_power by u (polynomial multiplication)
        new_power = [mpf(0)] * (n_max + 2)
        for i in range(n_max + 2):
            for j in range(min(len(u_coeffs), n_max + 2 - i)):
                if i + j < n_max + 2:
                    new_power[i + j] += u_power[i] * u_coeffs[j]
        u_power = new_power

        # Add (-1)^{m+1} * u^m / m to result
        sign = mpf((-1)**(m + 1))
        for k in range(n_max + 2):
            d[k] += sign * u_power[k] / m

    return d


# ─── Per-prime contribution ───

def per_prime_lambda_zeta(n_target, prime_max=500):
    """
    Decompose lambda_n^{zeta} into individual prime contributions.

    log((s-1)*zeta(s)) = log(s-1) + log(zeta(s))
                       = log(s-1) - sum_p log(1 - p^{-s})

    Each prime p contributes: -log(1 - p^{-s}) = sum_{k>=1} p^{-ks}/k

    The Taylor coefficients of p^{-ks}/k around s=1:
    p^{-ks} = p^{-k} * exp(-k*log(p)*h)
            = p^{-k} * sum_m (-k*log p)^m * h^m / m!

    So the m-th Taylor coefficient of -log(1-p^{-s}) is:
    sum_{k>=1} p^{-k} * (-k*log p)^m / (k * m!)

    BUT: The log(s-1) part is singular. The cancellation with the zeta pole
    is encoded in the sum over ALL primes. We can't separate it per-prime
    cleanly. Instead, compute the Taylor coefficients of the ENTIRE log_sz
    and compare with partial prime sums.
    """
    primes = primes_up_to(prime_max)

    # For each prime, compute its contribution to Taylor coefficients of
    # -log(1-p^{-s}) = sum_k p^{-ks}/k
    prime_data = {}
    for p in primes:
        lp = log(mpf(p))
        p_taylor = [mpf(0)] * (n_target + 2)  # coefficients of h^m
        for k in range(1, 50):
            pk = power(p, -k)
            if pk < mpf(10)**(-40):
                break
            for m in range(n_target + 2):
                # coefficient of h^m from term p^{-k(1+h)}/k
                p_taylor[m] += pk * power(-k * lp, m) / (k * fac(m))
        prime_data[p] = p_taylor

    # lambda_n from each prime's Taylor contribution
    prime_lambdas = []
    for p in primes:
        lam_p = float(lambda_from_taylor(n_target, prime_data[p]))
        prime_lambdas.append((p, lam_p))

    return prime_lambdas


# ─── Main ───

if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 17b: Prime Anatomy of Li Coefficients")
    print("=" * 70)

    N_MAX = 30

    # --- Step 1: Stieltjes constants structure ---
    print("\n--- Step 1: Stieltjes constants and log((s-1)*zeta(s)) ---")
    d_stieltjes = taylor_of_log_sz_from_stieltjes(N_MAX)
    d_fft = taylor_fft(log_sz, 1, N_MAX + 2, radius=0.4, n_points=256)

    print(f"\n  Taylor coefficients of log((s-1)*zeta(s)):")
    print(f"  {'k':>3s}  {'Stieltjes':>18s}  {'FFT':>18s}  {'match':>8s}")
    for k in range(min(10, len(d_stieltjes))):
        s_val = float(d_stieltjes[k])
        f_val = float(d_fft[k])
        match = "OK" if abs(s_val - f_val) < 1e-8 else f"ERR {abs(s_val-f_val):.2e}"
        print(f"  {k:3d}  {s_val:+18.12f}  {f_val:+18.12f}  {match:>8s}")

    # --- Step 2: Lambda_n^{zeta} from Stieltjes vs FFT ---
    print(f"\n--- Step 2: lambda_n^{{zeta}} comparison ---")
    print(f"  {'n':>3s}  {'Stieltjes':>16s}  {'FFT':>16s}  {'match':>10s}")
    for n in range(1, 21):
        lam_s = float(lambda_from_taylor(n, d_stieltjes))
        lam_f = float(lambda_from_taylor(n, d_fft))
        diff = abs(lam_s - lam_f)
        print(f"  {n:3d}  {lam_s:+16.10f}  {lam_f:+16.10f}  {diff:10.2e}")

    # --- Step 3: Explicit formula for lambda_n^{zeta} from Stieltjes ---
    print(f"\n--- Step 3: Anatomy of lambda_n^{{zeta}} ---")
    print("  lambda_n^{zeta} = n * sum C(n-1,j) * d_{n-j}")
    print("  where d_k are Taylor coefficients of log((s-1)*zeta(s))")
    print()
    for n in [1, 2, 3, 5]:
        terms = []
        for j in range(n):
            k = n - j
            if k < len(d_stieltjes):
                binom = float(mpmath.binomial(n-1, j))
                dk = float(d_stieltjes[k])
                contrib = binom * dk
                terms.append((j, k, binom, dk, contrib))
        total = n * sum(t[4] for t in terms)
        print(f"  lambda_{n}^{{zeta}} = {total:+.10f}")
        for j, k, b, d, c in terms:
            print(f"    j={j}: C({n-1},{j})={b:.0f} * d_{k}={d:+.10f} -> {c:+.10f}")

    # --- Step 4: Per-prime contributions ---
    print(f"\n--- Step 4: Per-prime contributions to lambda_n^{{zeta}} ---")
    print("  Each prime contributes -log(1-p^{-s}) to log zeta(s)")
    print("  (Does NOT include the log(s-1) pole cancellation)")
    print("  So per-prime sums != total, but show DISTRIBUTION")

    for n in [1, 2, 5, 10, 20]:
        prime_lambdas = per_prime_lambda_zeta(n, prime_max=200)
        total_prime = sum(c for _, c in prime_lambdas)
        actual_zeta = float(lambda_from_taylor(n, d_fft))

        print(f"\n  n={n}: lambda_n^{{zeta}} = {actual_zeta:+.8f}")
        print(f"  n={n}: sum of {len(prime_lambdas)} prime contribs = {total_prime:+.8f}")
        print(f"  n={n}: difference (log(s-1) piece) = {actual_zeta - total_prime:+.8f}")

        # Sort by absolute contribution
        sorted_p = sorted(prime_lambdas, key=lambda x: abs(x[1]), reverse=True)
        # Show top 5 and count positive/negative
        n_pos = sum(1 for _, c in prime_lambdas if c > 0)
        n_neg = sum(1 for _, c in prime_lambdas if c < 0)
        print(f"  Positive primes: {n_pos},  Negative primes: {n_neg}")
        print(f"  Top 5 contributors:")
        for p, c in sorted_p[:5]:
            sign = "+" if c > 0 else "-"
            print(f"    p={p:3d}: {c:+.8f}")

    # --- Step 5: Is there a pattern in the sign of prime contributions? ---
    print(f"\n--- Step 5: Sign pattern of per-prime contributions ---")
    print("  For n=1 (lambda_1^{zeta}), each prime contributes positively")
    print("  because -log(1-p^{-1}) > 0 for all p.")
    print()

    for n in [1, 2, 3, 5, 10]:
        prime_lambdas = per_prime_lambda_zeta(n, prime_max=100)
        signs = [(p, "+" if c > 0 else "-") for p, c in prime_lambdas]
        neg_primes = [p for p, c in prime_lambdas if c < 0]
        print(f"  n={n:2d}: negative primes = {neg_primes[:10]}{'...' if len(neg_primes) > 10 else ''}")

    # --- Step 6: The KEY question ---
    print(f"\n--- Step 6: The key question ---")
    print()
    print("  Observation: lambda_n^{zeta} > 0 for all n = 1..30")
    print()
    print("  If we could PROVE lambda_n^{zeta} > 0 for all n,")
    print("  then for n >= 8 (where lambda_n^{Gamma} > 0 too),")
    print("  we'd have lambda_n > 0 automatically.")
    print()
    print("  For n = 1..7, we'd still need |lambda_n^{zeta}| > |lambda_n^{Gamma}|.")
    print()

    # Check how the zeta part relates to n
    print("  Growth of lambda_n^{zeta}:")
    zeta_fft = taylor_fft(log_sz, 1, 55, radius=0.4, n_points=256)
    print(f"  {'n':>3s}  {'lambda_n^zeta':>14s}  {'ln(n)':>10s}  {'ratio':>10s}")
    for n in range(1, 31):
        z_val = float(lambda_from_taylor(n, zeta_fft))
        ln_n = np.log(n) if n > 1 else 0.001
        ratio = z_val / ln_n if ln_n > 0 else 0
        if n <= 10 or n % 5 == 0:
            print(f"  {n:3d}  {z_val:+14.8f}  {ln_n:10.4f}  {ratio:10.4f}")

    # --- Step 7: M-function connection ---
    print(f"\n--- Step 7: M-function vs Li coefficient positivity ---")
    print()
    print("  SESSION 16 M-function: M(s) = Im[(s-1/2)*conj(xi'/xi)] > 0")
    print("  Decomposition: M = M_Gamma + M_zeta")
    print("    M_Gamma > 0 for sigma > 1 (proved)")
    print("    M_Gamma > 0 for large t (proved)")
    print("    M_zeta can be negative (110x gap)")
    print()
    print("  SESSION 17 Li coefficients: lambda_n = lambda_n^{Gamma} + lambda_n^{zeta}")
    print("    lambda_n^{Gamma} > 0 for n >= 8")
    print("    lambda_n^{Gamma} < 0 for n = 1..7")
    print("    lambda_n^{zeta} > 0 for ALL n tested (1..30)")
    print()
    print("  INSIGHT: The two decompositions have OPPOSITE difficulty regions:")
    print("    M-function: Gamma easy (sigma>1, large t), zeta hard (critical strip)")
    print("    Li coeff:   Zeta easy (always positive??), Gamma hard (small n)")
    print()
    print("  QUESTION: Is lambda_n^{zeta} > 0 provable without RH?")
    print("  If yes, this reduces RH to: |lambda_n^{zeta}| > |lambda_n^{Gamma}| for n=1..7")
    print("  — a FINITE verification!")

    print("\n" + "=" * 70)
