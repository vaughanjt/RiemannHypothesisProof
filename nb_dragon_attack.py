"""
Session 29c: Attack RH directly via Nyman-Beurling distance d_N.

RH <=> d_N -> 0, where d_N^2 = inf ||1 - sum c_j f_j||^2 = 1 - b^T G^{-1} b

The EULER PRODUCT DISCOVERY from session 28 tells us about the geometry
of the Gram matrix. Now let's see if it helps us understand d_N.

QUESTIONS:
1. What is the optimal c vector? (c = G^{-1} b)
2. Does c have multiplicative/Euler product structure?
3. Where does the residual ||1 - sum c_j f_j||^2 concentrate?
4. Can we construct explicit c vectors that make d_N small?

THE DRAGON: If we can construct c_j such that ||1 - sum c_j f_j||^2 -> 0
and PROVE it, that's RH.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, power, nstr, log, euler
from math import gcd
from functools import reduce
import time
import sympy

mp.dps = 30


def build_gram(N, n_grid=500000):
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    return (fp @ fp.T) * dx, fp, x, dx


def distinct_primes(n):
    return sorted(set(sympy.factorint(n).keys())) if n > 1 else []


def euler_product_factor(j):
    return reduce(lambda a, p: a * (1 - 1.0/p**2), distinct_primes(j), 1.0) if j > 1 else 1.0


def mobius(n):
    """Mobius function."""
    return int(sympy.mobius(n))


if __name__ == "__main__":
    print("SESSION 29c: ATTACK THE RH DRAGON VIA d_N")
    print("=" * 70)

    # ================================================================
    # PART 1: Compute d_N and optimal coefficients for large N
    # ================================================================
    print("\nPART 1: NB DISTANCE d_N AND OPTIMAL COEFFICIENTS")
    print("-" * 70)

    results = []

    for N in [10, 20, 30, 50, 75, 100, 150, 200, 300]:
        t0 = time.time()
        n_grid = max(500000, N*5000)
        G, fp, x, dx = build_gram(N, n_grid)

        # b_j = <f_j, 1> = integral_0^1 {1/(jx)} dx
        b = np.sum(fp, axis=1) * dx  # = sum of fractional parts * dx

        # Optimal coefficients: c = G^{-1} b
        try:
            c_opt = np.linalg.solve(G, b)
            d_sq = 1.0 - np.dot(b, c_opt)
            d_N = np.sqrt(max(0, d_sq))
        except:
            c_opt = np.zeros(N)
            d_sq = 1.0
            d_N = 1.0

        # Eigenvalues
        evals = np.linalg.eigvalsh(G)
        sigma_min = evals[0]

        dt = time.time() - t0

        print(f"N={N:>4}: d_N={d_N:.8f}, d_N^2={d_sq:.4e}, "
              f"sigma_min={sigma_min:.4e}, cond={evals[-1]/sigma_min:.1e} ({dt:.1f}s)")

        results.append({'N': N, 'd_N': d_N, 'd_sq': d_sq, 'sigma_min': sigma_min,
                        'c_opt': c_opt[:min(30, N)].copy(), 'b': b[:min(30, N)].copy()})

    # Fit d_N decay rate
    Ns = np.array([r['N'] for r in results])
    dns = np.array([r['d_N'] for r in results])
    dsqs = np.array([r['d_sq'] for r in results])

    # Try power law: d_N ~ N^alpha
    valid = dns > 0
    if np.sum(valid) > 2:
        log_N = np.log(Ns[valid])
        log_d = np.log(dns[valid])
        alpha, log_C = np.polyfit(log_N, log_d, 1)
        print(f"\nPower law fit: d_N ~ {np.exp(log_C):.4f} * N^({alpha:.4f})")

    # Try exp(-c*sqrt(log N))
    log_d = np.log(dns[valid])
    sqrt_log_N = np.sqrt(np.log(Ns[valid]))
    beta, log_D = np.polyfit(sqrt_log_N, log_d, 1)
    print(f"Exp-sqrt fit: d_N ~ {np.exp(log_D):.4f} * exp({beta:.4f} * sqrt(log N))")

    print(f"\n{'N':>5} {'d_N':>12} {'d_N^2':>12} {'N^alpha pred':>12} {'exp pred':>12}")
    print("-" * 60)
    for r in results:
        plaw = np.exp(log_C) * r['N']**alpha
        expf = np.exp(log_D + beta * np.sqrt(np.log(r['N'])))
        print(f"{r['N']:>5} {r['d_N']:>12.8f} {r['d_sq']:>12.4e} {plaw:>12.8f} {expf:>12.8f}")

    # ================================================================
    # PART 2: Structure of optimal coefficients
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: STRUCTURE OF OPTIMAL c = G^{-1}b")
    print("-" * 70)

    # For the largest N computed, examine c_opt
    for N in [50, 100, 200]:
        n_grid = max(500000, N*5000)
        G, fp, x, dx = build_gram(N, n_grid)
        b = np.sum(fp, axis=1) * dx
        c_opt = np.linalg.solve(G, b)

        print(f"\nN={N}:")

        # Sign pattern
        n_pos = np.sum(c_opt > 0)
        n_neg = np.sum(c_opt < 0)
        print(f"  Sign: {n_pos} positive, {n_neg} negative")

        # Where is the weight?
        energy = c_opt**2
        js = np.arange(1, N+1)

        # Energy by region
        q1, q2, q3 = N//4, N//2, 3*N//4
        print(f"  Energy by quartile: "
              f"[1,{q1}]={np.sum(energy[:q1]):.4f}, "
              f"[{q1+1},{q2}]={np.sum(energy[q1:q2]):.4f}, "
              f"[{q2+1},{q3}]={np.sum(energy[q2:q3]):.4f}, "
              f"[{q3+1},{N}]={np.sum(energy[q3:]):.4f}")

        # Are the coefficients multiplicative? Test: c_{mn} vs c_m * c_n for gcd(m,n)=1
        print(f"  Multiplicativity test:")
        print(f"  {'m':>4} {'n':>4} {'mn':>4} {'c_m':>12} {'c_n':>12} {'c_{mn}':>12} "
              f"{'c_m*c_n':>12} {'ratio':>8}")
        for m, n in [(2,3), (2,5), (3,5), (2,7), (3,7), (5,7), (2,11), (2,13)]:
            if m*n <= N:
                cm, cn, cmn = c_opt[m-1], c_opt[n-1], c_opt[m*n-1]
                prod_cn = cm * cn
                ratio = cmn / prod_cn if abs(prod_cn) > 1e-10 else float('nan')
                print(f"  {m:>4} {n:>4} {m*n:>4} {cm:>12.6f} {cn:>12.6f} {cmn:>12.6f} "
                      f"{prod_cn:>12.6f} {ratio:>8.4f}")

        # Mobius-weighted sum: sum_{d|j} mu(j/d) * c_d
        print(f"\n  Mobius transform of c (should it be simple?):")
        print(f"  {'j':>4} {'c_j':>12} {'(Mc)_j':>12} {'j*(Mc)_j':>12} {'mu(j)':>6}")
        for j in range(1, min(31, N+1)):
            divs = [d for d in range(1, j+1) if j % d == 0]
            Mc_j = sum(mobius(j//d) * c_opt[d-1] for d in divs)
            print(f"  {j:>4} {c_opt[j-1]:>12.6f} {Mc_j:>12.6f} {j*Mc_j:>12.6f} {mobius(j):>6}")

    # ================================================================
    # PART 3: Can we construct explicit c vectors that beat random?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: EXPLICIT CONSTRUCTIONS FOR d_N")
    print("-" * 70)

    for N in [50, 100, 200]:
        n_grid = max(500000, N*5000)
        G, fp, x, dx = build_gram(N, n_grid)
        b = np.sum(fp, axis=1) * dx

        # Optimal (baseline)
        c_opt = np.linalg.solve(G, b)
        d_opt_sq = 1.0 - np.dot(b, c_opt)

        # Construction 1: Mobius-based — c_j = mu(j)/j (the "natural" NB choice)
        c_mob = np.array([mobius(j)/j for j in range(1, N+1)], dtype=float)
        r_mob = np.dot(fp.T @ c_mob, np.ones(len(x))) * dx  # = integral sum c_j f_j dx
        d_mob_sq = 1.0 - 2*np.dot(b, c_mob) + c_mob @ G @ c_mob

        # Construction 2: c_j = mu(j) * log(N/j) / j (Beurling's approximation)
        c_beu = np.array([mobius(j) * max(0, np.log(N/j)) / j for j in range(1, N+1)])
        d_beu_sq = 1.0 - 2*np.dot(b, c_beu) + c_beu @ G @ c_beu

        # Construction 3: c_j = 6/pi^2 * mu(j)/j (scaled for sum c_j/j ~ 1)
        scale = 6 / np.pi**2
        c_sc = np.array([scale * mobius(j)/j for j in range(1, N+1)])
        d_sc_sq = 1.0 - 2*np.dot(b, c_sc) + c_sc @ G @ c_sc

        # Construction 4: Truncated Dirichlet series for 1/zeta
        # sum_{n=1}^N mu(n) n^{-s} approximates 1/zeta(s)
        # In NB terms, this is related to the distance problem

        print(f"\nN={N}:")
        print(f"  Optimal:              d_N^2 = {d_opt_sq:.6e}")
        print(f"  Mobius mu(j)/j:       d_N^2 = {d_mob_sq:.6e}")
        print(f"  Beurling mu(j)log(N/j)/j: d^2 = {d_beu_sq:.6e}")
        print(f"  Scaled 6/pi^2 mu(j)/j: d^2 = {d_sc_sq:.6e}")

        # Which construction is closest to optimal?
        for name, c_test in [("Mobius", c_mob), ("Beurling", c_beu), ("Scaled", c_sc)]:
            rel = np.linalg.norm(c_test - c_opt) / np.linalg.norm(c_opt)
            print(f"  ||c_{name} - c_opt|| / ||c_opt|| = {rel:.4f}")

    # ================================================================
    # PART 4: Residual function analysis — WHERE does ||1 - sum c_j f_j||^2 live?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: RESIDUAL FUNCTION r(x) = 1 - sum c_j f_j(x)")
    print("-" * 70)

    N = 100
    n_grid = max(500000, N*5000)
    G, fp, x, dx = build_gram(N, n_grid)
    b = np.sum(fp, axis=1) * dx
    c_opt = np.linalg.solve(G, b)

    # Compute residual function
    approx = fp.T @ c_opt  # sum c_j f_j(x) at each grid point
    residual = 1.0 - approx

    # Residual analysis
    r_sq = residual**2
    d_sq_check = np.sum(r_sq) * dx

    print(f"N={N}")
    print(f"  ||residual||^2 = {d_sq_check:.6e} (should match d_N^2)")

    # Where does the residual concentrate?
    # Split [0,1] into regions
    regions = [
        (0, 1e-4, "x < 1e-4"),
        (1e-4, 1e-3, "1e-4 < x < 1e-3"),
        (1e-3, 0.01, "1e-3 < x < 1e-2"),
        (0.01, 0.1, "1e-2 < x < 0.1"),
        (0.1, 0.5, "0.1 < x < 0.5"),
        (0.5, 1.0, "0.5 < x < 1.0"),
    ]

    print(f"\n  {'Region':>20} {'contrib to d^2':>14} {'%':>7} {'mean |r|':>12} {'max |r|':>10}")
    print("  " + "-" * 70)
    for lo, hi, name in regions:
        mask = (x >= lo) & (x < hi)
        if np.sum(mask) < 1:
            continue
        contrib = np.sum(r_sq[mask]) * dx
        pct = 100 * contrib / d_sq_check
        mean_r = np.mean(np.abs(residual[mask]))
        max_r = np.max(np.abs(residual[mask]))
        print(f"  {name:>20} {contrib:>14.4e} {pct:>6.1f}% {mean_r:>12.4e} {max_r:>10.4f}")

    # Near x ~ 1/j for small j: this is where {1/(jx)} has its discontinuities
    print(f"\n  Residual at special points x = 1/j:")
    for j in [1, 2, 3, 5, 10, 20, 50, 100]:
        idx = np.argmin(np.abs(x - 1.0/j))
        print(f"    x = 1/{j} = {1.0/j:.4f}: r(x) = {residual[idx]:>10.6f}")

    # ================================================================
    # PART 5: Mellin transform of the residual — connection to zeta
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: MELLIN TRANSFORM OF RESIDUAL")
    print("-" * 70)

    # The residual r(x) = 1 - sum c_j {1/(jx)}
    # Its Mellin transform: r_hat(s) = 1/(s-1) - sum c_j f_hat_j(s)
    # = 1/(s-1) - sum c_j [1/(j(s-1)) - j^{-s} zeta(s)/s]
    # = (1/(s-1))(1 - sum c_j/j) + (zeta(s)/s) sum c_j j^{-s}
    # = A(s) + (zeta(s)/s) D(s)
    #
    # where A(s) = (1 - sum c_j/j)/(s-1) is the pole residual
    # and D(s) = sum c_j j^{-s} is the Dirichlet polynomial
    #
    # ||r||^2 = (1/2pi) integral |r_hat(1/2+it)|^2 dt
    # = (1/2pi) integral |A + zeta*D/s|^2 dt
    #
    # If sum c_j/j ≈ 1, the pole residual A is small.
    # Then ||r||^2 ≈ (1/2pi) integral |zeta*D/s|^2 dt

    S = np.dot(c_opt, 1.0/np.arange(1, N+1))
    print(f"N={N}")
    print(f"  sum c_j/j = {S:.10f} (should be close to 1 for good approximation)")
    print(f"  1 - sum c_j/j = {1-S:.6e}")

    # D(s) = sum c_j j^{-s} evaluated at key points
    print(f"\n  D(s) at key points on critical line:")
    for t_val in [0, 1, 5, 14.13, 21.02, 25.01]:
        s = mpc(0.5, t_val)
        D_val = sum(c_opt[j] * power(mpf(j+1), -s) for j in range(N))
        z_val = zeta(s)
        full = (1 - S) / (s - 1) + z_val * D_val / s

        print(f"    t={t_val:>6.2f}: |D|={float(abs(D_val)):>10.4e}, "
              f"|zeta|={float(abs(z_val)):>8.4f}, "
              f"|zeta*D/s|={float(abs(z_val*D_val/s)):>10.4e}, "
              f"|full|={float(abs(full)):>10.4e}")

    # KEY: If RH is true, can we construct D(s) that makes zeta*D/s small?
    # zeta(s) * D(s) = zeta(s) * sum c_j j^{-s}
    # If we could choose c_j = mu(j)/j, then D(s) = (1/zeta(s)) * sum/j = ...
    # Actually, sum mu(j) j^{-s-1} = 1/(zeta(s+1)) for Re(s) > 0
    # So D(s) with c_j = mu(j)/j gives D(s) = sum mu(j)/j * j^{-s} = sum mu(j) j^{-s-1} = 1/zeta(s+1)
    # Then zeta(s) * D(s) = zeta(s)/zeta(s+1)

    print(f"\n  MOBIUS TEST: D(s) = sum mu(j)/j * j^{{-s}} = 1/zeta(s+1)")
    for t_val in [0, 1, 5, 14.13, 25.01]:
        s = mpc(0.5, t_val)
        # Truncated sum
        D_trunc = sum(mobius(j) / j * power(mpf(j), -s) for j in range(1, N+1))
        # Exact
        D_exact = 1 / zeta(s + 1)
        # zeta * D = zeta(s) / zeta(s+1)
        ratio_val = zeta(s) / zeta(s + 1)

        print(f"    t={t_val:>6.2f}: D_trunc={nstr(D_trunc, 8)}, "
              f"1/zeta(s+1)={nstr(D_exact, 8)}, "
              f"zeta/zeta(s+1)={nstr(ratio_val, 8)}")
        print(f"            |D_trunc - exact| = {float(abs(D_trunc - D_exact)):.4e}, "
              f"|zeta*D/s| = {float(abs(ratio_val/s)):.4e}")

    # ================================================================
    # PART 6: The killer question — zeta(s)/zeta(s+1) growth on Re(s)=1/2
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 6: |zeta(1/2+it)/zeta(3/2+it)| — the residual driver")
    print("-" * 70)

    # If c_j = mu(j)/j, then the residual in Mellin space is ~ zeta(s)/zeta(s+1)/s
    # ||r||^2 = (1/2pi) integral |zeta(s)/zeta(s+1)/s|^2 dt  (plus pole correction)
    # This integral CONVERGES because:
    #   |zeta(1/2+it)| ~ t^{1/4} (Lindelof hypothesis) or t^{0.15...} (best known)
    #   |1/zeta(3/2+it)| <= c (zeta has no zeros on Re(s) > 1, so 1/zeta(3/2+it) bounded)
    #   |1/s| = 1/sqrt(1/4+t^2) ~ 1/t
    # So |integrand| ~ t^{1/2} * c / t^2 = c/t^{3/2} — converges!

    print("Behavior of |zeta(1/2+it)/zeta(3/2+it)|^2 / (1/4+t^2):")
    t_vals = np.concatenate([np.arange(1, 10, 0.5), np.arange(10, 50, 2), np.arange(50, 200, 10)])

    integrand_vals = []
    for t in t_vals:
        s = mpc(0.5, t)
        z1 = zeta(s)
        z2 = zeta(s + 1)
        val = float(abs(z1/z2)**2) / (0.25 + t**2)
        integrand_vals.append(val)

    # Compute the integral approximation
    print(f"\n  {'t':>6} {'|zeta(s)|^2':>12} {'|1/zeta(s+1)|^2':>16} {'integrand':>12}")
    for t, val in zip(t_vals[::5], integrand_vals[::5]):
        s = mpc(0.5, t)
        z1sq = float(abs(zeta(s))**2)
        z2sq_inv = float(1/abs(zeta(s+1))**2)
        print(f"  {t:>6.1f} {z1sq:>12.4f} {z2sq_inv:>16.4f} {val:>12.4e}")

    # Approximate the full integral
    # (1/2pi) integral_{-inf}^{inf} |zeta(s)/zeta(s+1)/s|^2 dt
    # = (1/pi) integral_0^inf ... dt (by symmetry)
    t_fine = np.linspace(0.01, 500, 50000)
    dt_fine = t_fine[1] - t_fine[0]
    integral_vals = np.zeros(len(t_fine))
    for i, t in enumerate(t_fine):
        s = mpc(0.5, t)
        z1 = zeta(s)
        z2 = zeta(s + 1)
        integral_vals[i] = float(abs(z1/(z2*s))**2)

    full_integral = np.trapezoid(integral_vals, t_fine) / float(pi)
    print(f"\n  (1/pi) integral_0^500 |zeta(s)/zeta(s+1)/s|^2 dt ≈ {full_integral:.6f}")

    # Pole contribution: (1-S)^2 / (s-1)^2 integrates to 2pi*(1-S)^2
    # Full d_N^2 for Mobius construction:
    c_mob = np.array([mobius(j)/j for j in range(1, N+1)], dtype=float)
    S_mob = np.dot(c_mob, 1.0/np.arange(1, N+1))
    pole_part = (1 - S_mob)**2
    print(f"  Pole part (1-S)^2 = {pole_part:.6e} where S = {S_mob:.8f}")
    print(f"  Theoretical d_mob^2 ~ pole + integral = {pole_part + full_integral:.6f}")
    d_mob_sq_check = 1.0 - 2*np.dot(b, c_mob) + c_mob @ G @ c_mob
    print(f"  Actual d_mob^2 (direct) = {d_mob_sq_check:.6f}")

    # ================================================================
    # PART 7: Improved Beurling construction
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 7: IMPROVED EXPLICIT CONSTRUCTIONS")
    print("-" * 70)

    # The Beurling/Vasyunin approach: instead of c_j = mu(j)/j,
    # use c_j that approximately solves D(s) = 1/(zeta(s) * s) on Re(s) = 1/2
    # This makes the residual r_hat(s) ≈ 0

    # Construction: c_j = (1/N) * sum_{n=1}^N mu(n)/n * [n|j] * j/n
    # i.e., Mobius inversion through the divisor lattice

    for N in [50, 100, 200, 300]:
        n_grid = max(500000, N*5000)
        G, fp, x, dx = build_gram(N, n_grid)
        b = np.sum(fp, axis=1) * dx

        # Optimal
        c_opt = np.linalg.solve(G, b)
        d_opt = np.sqrt(max(0, 1.0 - np.dot(b, c_opt)))

        # Mobius: c_j = mu(j)/j
        c_mob = np.array([mobius(j)/j for j in range(1, N+1)], dtype=float)
        d_mob = np.sqrt(max(0, 1.0 - 2*np.dot(b, c_mob) + c_mob @ G @ c_mob))

        # Smoothed Mobius: c_j = mu(j)/j * (1 - log(j)/log(N)) for j <= N
        c_smooth = np.array([mobius(j)/j * max(0, 1 - np.log(j)/np.log(N))
                             for j in range(1, N+1)])
        d_smooth = np.sqrt(max(0, 1.0 - 2*np.dot(b, c_smooth) + c_smooth @ G @ c_smooth))

        # Vasyunin-inspired: c_j = mu(j) * (log(N/j))^k / (j * k!)
        for k in [1, 2, 3]:
            c_vas = np.array([mobius(j) * max(0, np.log(N/j))**k / (j * np.math.factorial(k))
                              for j in range(1, N+1)])
            d_vas = np.sqrt(max(0, 1.0 - 2*np.dot(b, c_vas) + c_vas @ G @ c_vas))
            if k == 1:
                d_vas1 = d_vas

        # Exponential smoothing: c_j = mu(j)/j * exp(-j/N)
        c_exp = np.array([mobius(j)/j * np.exp(-j/N) for j in range(1, N+1)])
        d_exp = np.sqrt(max(0, 1.0 - 2*np.dot(b, c_exp) + c_exp @ G @ c_exp))

        print(f"N={N:>4}: d_opt={d_opt:.6f}, d_mob={d_mob:.6f}, "
              f"d_smooth={d_smooth:.6f}, d_vas(k=1)={d_vas1:.6f}, d_exp={d_exp:.6f}")

    print(f"\n{'='*70}")
    print("DRAGON STATUS REPORT")
    print("=" * 70)
    print("""
KEY INSIGHT: The residual r(x) = 1 - sum c_j f_j(x) in Mellin space is:
  r_hat(s) = (1-S)/(s-1) + zeta(s)*D(s)/s

For c_j = mu(j)/j:  D(s) -> 1/zeta(s+1), so zeta*D -> zeta(s)/zeta(s+1)

The distance is:
  d_N^2 ≈ pole_part + (1/2pi) integral |zeta(s)/zeta(s+1)/s|^2 dt + truncation_error

The truncation error from sum_1^N vs sum_1^inf controls d_N -> 0.
RH <=> this truncation error -> 0, which is related to the behavior of
the Mobius function M(x) = sum_{n<=x} mu(n).

RH <=> M(x) = O(x^{1/2+eps}) for all eps > 0.

This is EQUIVALENT to RH — we've come full circle to the classical formulation!
The NB approach reformulates RH as: can sum_{n=1}^N mu(n)/n approximate 1/zeta well?
""")
