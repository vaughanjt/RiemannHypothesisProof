"""
Session 19b: Bounding lambda_n^zeta via Stieltjes constants.

The Taylor coefficients d_k of log((s-1)*zeta(s)) at s=1 satisfy:
  d_k = coefficient of h^k in log(1 + sum_{n>=0} (-1)^n * gamma_n/n! * h^{n+1})

The Stieltjes constants gamma_n grow: |gamma_n| ~ (n!) / (2*pi)^n * C
(from the pole of zeta at s=1 with reflection formula).

Known bounds (Berndt 1972, Matsuoka 1985, Adell 2011):
  |gamma_n| <= A * (n+1)! / (2*pi)^{n+1} * (something)

The question: do the d_k decay fast enough that lambda_n^zeta stays bounded?

From Session 17: d_k alternate in sign with |d_k| ~ R^{-k} where R~3
(distance to nearest trivial zero at s=-2).

But the lambda_n^zeta involves binomial sums of d_k, which can amplify...
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, zeta, power, fac, euler

mp.dps = 100


def compute_stieltjes(n_max):
    return [mpmath.stieltjes(n) for n in range(n_max + 1)]


def d_k_from_stieltjes(k_max):
    """
    Taylor coefficients of log((s-1)*zeta(s)) at s=1.

    (s-1)*zeta(s) = 1 + sum_{n>=0} (-1)^n * gamma_n/n! * h^{n+1}  where h=s-1

    Let u(h) = sum_{n>=0} (-1)^n * gamma_n/n! * h^{n+1}
    Then d_k = [h^k] log(1+u(h))
    """
    stj = compute_stieltjes(k_max + 5)

    # Build u coefficients (u starts at h^1, no constant)
    u = [mpf(0)] * (k_max + 2)  # u_0 = 0
    for n in range(min(k_max + 3, len(stj))):
        u.append(mpf(0))  # extend if needed
    # u_j for j >= 1: coefficient of h^j in u(h)
    # u(h) = sum_{n>=0} (-1)^n * gamma_n/n! * h^{n+1}
    # so u_{j} = (-1)^{j-1} * gamma_{j-1} / (j-1)!  for j >= 1
    u = [mpf(0)]  # u_0 = 0
    for j in range(1, k_max + 2):
        n = j - 1
        if n < len(stj):
            u.append(power(-1, n) * stj[n] / fac(n))
        else:
            u.append(mpf(0))

    # Compute d_k = [h^k] log(1 + u) = [h^k] sum_{m>=1} (-1)^{m+1} u^m / m
    d = [mpf(0)] * (k_max + 2)

    # u^m via polynomial multiplication
    u_power = [mpf(0)] * (k_max + 2)
    u_power[0] = mpf(1)  # u^0

    for m in range(1, k_max + 2):
        # u_power = u_power * u (truncated polynomial multiplication)
        new_power = [mpf(0)] * (k_max + 2)
        for i in range(k_max + 2):
            if abs(u_power[i]) < mpf(10)**(-80):
                continue
            for j in range(1, k_max + 2 - i):  # u starts at j=1
                if j < len(u) and abs(u[j]) > mpf(10)**(-80):
                    new_power[i + j] += u_power[i] * u[j]
        u_power = new_power

        # d += (-1)^{m+1} * u^m / m
        sign = power(-1, m + 1)
        for k in range(k_max + 2):
            d[k] += sign * u_power[k] / m

    return d


def lambda_from_coeffs(n, coeffs):
    total = mpf(0)
    for j in range(n):
        k = n - j
        if k < len(coeffs):
            total += mpmath.binomial(n - 1, j) * coeffs[k]
    return float(n * total)


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 19b: Bounding lambda_n^zeta")
    print("=" * 70)

    K_MAX = 30

    # --- Step 1: Stieltjes constants ---
    print("\n--- Step 1: Stieltjes constants ---")
    stj = compute_stieltjes(K_MAX)
    print(f"  {'n':>3s}  {'gamma_n':>22s}  {'|gamma_n|':>14s}  {'n!/(2pi)^n':>14s}  {'ratio':>10s}")
    for n in range(K_MAX + 1):
        gn = float(stj[n])
        abs_gn = abs(gn)
        nfact_2pi = float(fac(n) / power(2*pi, n))
        ratio = abs_gn / nfact_2pi if nfact_2pi > 1e-20 else 0
        if n <= 20 or n % 5 == 0:
            print(f"  {n:3d}  {gn:+22.10f}  {abs_gn:14.6e}  {nfact_2pi:14.6e}  {ratio:10.4f}")

    # --- Step 2: d_k coefficients ---
    print("\n--- Step 2: Taylor coefficients d_k of log((s-1)*zeta(s)) ---")
    dk = d_k_from_stieltjes(K_MAX)
    print(f"  {'k':>3s}  {'d_k':>22s}  {'|d_k|':>14s}  {'(1/3)^k':>14s}  {'ratio':>10s}")
    for k in range(K_MAX + 1):
        dkv = float(dk[k])
        abs_dk = abs(dkv)
        third_k = (1/3.0)**k
        ratio = abs_dk / third_k if third_k > 1e-20 else 0
        if k <= 15 or k % 5 == 0:
            print(f"  {k:3d}  {dkv:+22.15f}  {abs_dk:14.6e}  {third_k:14.6e}  {ratio:10.6f}")

    # --- Step 3: lambda_n^zeta ---
    print("\n--- Step 3: lambda_n^zeta from Stieltjes ---")
    print(f"  {'n':>3s}  {'lambda_n^zeta':>16s}")
    for n in range(1, K_MAX + 1):
        z = lambda_from_coeffs(n, dk)
        print(f"  {n:3d}  {z:+16.10f}")

    # --- Step 4: Bound analysis ---
    print("\n--- Step 4: Can we bound |lambda_n^zeta|? ---")
    print()
    print("  The d_k decay like R^{-k} where R is the radius of convergence of")
    print("  log((s-1)*zeta(s)) around s=1.")
    print()
    print("  R = distance to nearest singularity = min(dist to trivial zero, dist to non-trivial zero)")
    print("  Trivial zeros: s=-2 -> distance 3")
    print("  Non-trivial zeros: s=1/2+i*14.13 -> distance sqrt(0.25+200) ~ 14.14")
    print("  So R = 3.")
    print()
    print("  This means |d_k| <= M / 3^k for some constant M.")
    print()

    # Compute M from the data
    M_estimates = []
    for k in range(1, K_MAX + 1):
        M_est = abs(float(dk[k])) * 3**k
        M_estimates.append(M_est)
    M_max = max(M_estimates)
    print(f"  Estimated M = max |d_k| * 3^k = {M_max:.6f}")
    print(f"  (From k=1: M ~ |d_1|*3 = {abs(float(dk[1]))*3:.6f})")
    print()

    # Now: lambda_n^zeta = n * sum C(n-1,j) * d_{n-j}
    # |lambda_n^zeta| <= n * sum C(n-1,j) * |d_{n-j}|
    # <= n * sum C(n-1,j) * M / 3^{n-j}
    # = n * M * sum C(n-1,j) / 3^{n-j}
    # = n * M * (1/3)^n * sum C(n-1,j) * 3^j  (setting k = n-j, j goes 0..n-1)
    # Wait: sum_{j=0}^{n-1} C(n-1,j) * 3^j = (1+3)^{n-1} = 4^{n-1}  (if j goes to n-1)
    # Actually need more care: sum_{j=0}^{n-1} C(n-1,j) / 3^{n-j}
    # = (1/3) * sum C(n-1,j) * (1/3)^{n-1-j}
    # = (1/3) * (1 + 1/3)^{n-1} = (1/3) * (4/3)^{n-1} = (4/3)^{n-1} / 3

    print("  Crude bound: |lambda_n^zeta| <= n * M * (4/3)^{n-1} / 3")
    print()
    print("  BUT this grows exponentially! So the crude bound is useless.")
    print()
    print("  The CANCELLATION in the alternating sum is essential.")
    print("  |d_k| alone can't bound lambda_n^zeta — we need sign information.")
    print()

    # --- Step 5: Alternating sign analysis ---
    print("--- Step 5: Sign structure of d_k ---")
    signs = ['+' if float(dk[k]) >= 0 else '-' for k in range(K_MAX + 1)]
    print(f"  Signs: {' '.join(signs)}")
    print()

    # d_k alternates: d_1 > 0, d_2 < 0, d_3 > 0, d_4 < 0, ...
    # This means d_k = (-1)^{k+1} * |d_k|
    # (Same alternation as the g_k from the Gamma part, but different magnitudes)

    alternates = all(float(dk[k]) * float(dk[k+1]) < 0 for k in range(1, min(K_MAX, len(dk)-1)))
    print(f"  Strictly alternating for k=1..{min(K_MAX, len(dk)-2)}? {alternates}")

    # --- Step 6: Comparison with Gamma part ---
    print("\n--- Step 6: d_k vs g_k (Gamma coefficients) ---")
    print("  The g_k also alternate. How do they compare?")
    print()

    # Compute g_k analytically
    g = [mpf(0)] * (K_MAX + 2)
    g[0] = log(mpf(0.5)) + loggamma(mpf(0.5)) - log(pi)/2  # constant
    g[1] = 1 - log(pi)/2 + mpmath.digamma(mpf(0.5))/2
    for k in range(2, K_MAX + 2):
        psi_k = mpmath.polygamma(k-1, mpf(0.5))
        g[k] = power(-1, k+1)/k + psi_k / (fac(k) * power(2, k))

    print(f"  {'k':>3s}  {'d_k (zeta)':>18s}  {'g_k (Gamma)':>18s}  {'d_k + g_k':>18s}  {'|d/g|':>10s}")
    for k in range(1, 16):
        dv = float(dk[k])
        gv = float(g[k])
        sv = dv + gv  # this is the total xi Taylor coefficient c_k
        ratio = abs(dv / gv) if abs(gv) > 1e-20 else float('inf')
        print(f"  {k:3d}  {dv:+18.12f}  {gv:+18.12f}  {sv:+18.12f}  {ratio:10.4f}")

    # --- Step 7: The circularity problem ---
    print("\n--- Step 7: The circularity obstacle ---")
    print()
    print("  The d_k are computable constants (Stieltjes + polynomial algebra).")
    print("  The lambda_n^zeta are KNOWN for each n (finite computation).")
    print()
    print("  But PROVING |lambda_n^zeta| < lambda_n^Gamma for ALL n >= 11")
    print("  requires bounding an INFINITE family of sums.")
    print()
    print("  The crude bound fails (exponential growth).")
    print("  The cancellation in the alternating sum is key but hard to control.")
    print()
    print("  Generating function approach:")
    print("  lambda_n^zeta / n = [x^n] log_sz(1/(1-x))")
    print("  log_sz(1/(1-x)) has singularities at x = 1-1/rho on |x|=1 (under RH)")
    print("  and at x = 3/2 (from trivial zero at s=-2).")
    print()
    print("  The x=3/2 singularity gives [x^n] ~ C*(2/3)^n (exponentially decaying).")
    print("  The unit-circle singularities give [x^n] ~ sum_rho oscillatory/n.")
    print()
    print("  So asymptotically: lambda_n^zeta ~ -sum_rho 2*Re[w_rho^n] / (some function of n)")
    print("  This is a CONDITIONAL result (assumes zeros are on critical line).")
    print()
    print("  WITHOUT assuming RH, the singularities could be INSIDE |x|=1,")
    print("  giving exponentially GROWING Taylor coefficients — which would make")
    print("  lambda_n negative for large n. This is precisely how a zero off")
    print("  the critical line would manifest as lambda_n < 0.")
    print()
    print("  CONCLUSION: Bounding |lambda_n^zeta| unconditionally is equivalent to RH.")
    print("  There is no shortcut through the Stieltjes constants alone.")

    print("\n" + "=" * 70)
