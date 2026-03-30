"""
Session 29: Attack the Mobius inversion gap for NB sigma_min proof.

THE GAP: We know dist_j^2 ≈ 3 * prod_{p|j}(1-1/p^2) / j^2 numerically.
Need to PROVE dist_j^2 >= c/j^2 for all j.

STRATEGY:
A) Exact Gram entry formula: G_{jk} = <f_j, f_k> where f_j(x) = {1/(jx)}
   Use the Mellin formula: f_hat_j(s) = 1/(j(s-1)) - j^{-s}*zeta(s)/s

B) Schur complement: dist_j^2 = G_{jj} - G_{j,<j} G_{<j,<j}^{-1} G_{<j,j}
   = G_{jj} - (projection of f_j onto span{f_1,...,f_{j-1}})^2

C) Key insight: f_j and f_d are strongly correlated when d|j because
   {1/(jx)} and {1/(dx)} share sawtooth discontinuities at x = 1/(jn)

D) Multiplicative structure: if j = p1^a1 * p2^a2 * ..., the correlation
   with each f_{j/p_i} introduces a factor related to (1-1/p_i^2)

APPROACH 1: Prove via the Mellin representation
   dist_j^2 = (1/2pi) * min_{c: c_j=1} integral |sum c_k f_hat_k(1/2+it)|^2 dt
   The multiplicative structure of j^{-s} makes this a Euler product question.

APPROACH 2: Direct divisor-lattice Schur complement
   Build the exact (restricted) Gram matrix for divisors of j only,
   compute Schur complement on that sublattice.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, power, nstr, log, euler
from math import gcd
from functools import reduce
import time
import sympy

mp.dps = 50


def build_gram(N, n_grid=500000):
    """Build N x N Gram matrix."""
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    return (fp @ fp.T) * dx


def divisors(n):
    """All divisors of n, sorted."""
    divs = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def prime_factors(n):
    """Prime factorization of n."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def distinct_primes(n):
    return sorted(set(prime_factors(n)))


def euler_product_factor(j):
    """prod_{p|j} (1 - 1/p^2)"""
    return reduce(lambda a, p: a * (1 - 1.0/p**2), distinct_primes(j), 1.0) if j > 1 else 1.0


def f_hat(j, s):
    """Mellin transform: f_hat_j(s) = 1/(j(s-1)) - j^{-s} zeta(s)/s"""
    return 1/(mpf(j)*(s-1)) - power(mpf(j),-s)*zeta(s)/s


def gram_mellin(j, k, T_max=300, n_t=30000):
    """Compute G_{jk} via Mellin-Parseval integral."""
    t_grid = np.linspace(-T_max, T_max, n_t)
    dt = t_grid[1] - t_grid[0]
    total = mpf(0)
    for t in t_grid:
        s = mpc(0.5, t)
        fj = f_hat(j, s)
        fk = f_hat(k, s)
        total += (fj * mpmath.conj(fk)).real
    return float(total * dt / (2 * pi))


if __name__ == "__main__":
    print("SESSION 29: MOBIUS INVERSION ATTACK")
    print("=" * 70)

    # ================================================================
    # PART 1: Verify Euler product formula for dist_j^2 on larger range
    # ================================================================
    print("\nPART 1: Euler product formula verification (extended)")
    print("-" * 70)

    N = 200
    t0 = time.time()
    G = build_gram(N, n_grid=max(500000, N*5000))
    print(f"Built G_{N}x{N} in {time.time()-t0:.1f}s")

    # Compute dist_j^2 = Schur complement = G_{jj} - G_{j,<j} G_{<j}^{-1} G_{<j,j}
    print(f"\n{'j':>4} {'dist_j^2':>12} {'j^2*d^2':>10} {'EP factor':>10} "
          f"{'j^2*d^2/EP':>12} {'C_j':>8} {'primes':>15}")
    print("-" * 80)

    dist_sq = np.zeros(N)
    C_values = []

    for j in range(1, N+1):
        if j == 1:
            dist_sq[0] = G[0, 0]
        else:
            G_sub = G[:j-1, :j-1]
            g_cross = G[j-1, :j-1]
            try:
                coeffs = np.linalg.solve(G_sub, g_cross)
                dist_sq[j-1] = G[j-1, j-1] - np.dot(g_cross, coeffs)
            except:
                dist_sq[j-1] = 0

        ep = euler_product_factor(j)
        j2d2 = j**2 * dist_sq[j-1]
        C_j = j2d2 / ep if ep > 0 else float('nan')
        C_values.append(C_j)

        primes_j = distinct_primes(j) if j > 1 else []

        if j <= 30 or j % 10 == 0 or sympy.isprime(j):
            prime_str = '*'.join(map(str, primes_j)) if primes_j else '1'
            print(f"{j:>4} {dist_sq[j-1]:>12.6e} {j2d2:>10.4f} {ep:>10.4f} "
                  f"{C_j:>12.4f} {C_j:>8.3f} {prime_str:>15}")

    # Is C_j really constant or slowly growing?
    C_arr = np.array(C_values[1:])  # skip j=1
    print(f"\nC_j statistics (j=2..{N}):")
    print(f"  mean = {np.mean(C_arr):.4f}")
    print(f"  std  = {np.std(C_arr):.4f}")
    print(f"  min  = {np.min(C_arr):.4f} at j={np.argmin(C_arr)+2}")
    print(f"  max  = {np.max(C_arr):.4f} at j={np.argmax(C_arr)+2}")

    # Check if C_j grows as log(j)
    js = np.arange(2, N+1)
    log_js = np.log(js)
    slope, intercept = np.polyfit(log_js, C_arr, 1)
    print(f"\n  Linear fit C_j ~ {slope:.4f} * ln(j) + {intercept:.4f}")
    print(f"  So dist_j^2 ~ ({slope:.2f}*ln(j) + {intercept:.2f}) * EP(j) / j^2")

    # ================================================================
    # PART 2: Divisor lattice Schur complement — exact structure
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: DIVISOR LATTICE STRUCTURE")
    print("-" * 70)

    # For j with known factorization, compute the Gram matrix restricted
    # to divisors of j and examine the Schur complement

    for j in [6, 12, 30, 60]:
        divs = divisors(j)
        n_div = len(divs)

        # Build restricted Gram matrix
        G_div = np.zeros((n_div, n_div))
        for a in range(n_div):
            for b in range(n_div):
                G_div[a, b] = G[divs[a]-1, divs[b]-1]

        # Schur complement: project f_j onto span of {f_d : d|j, d<j}
        idx_j = divs.index(j)
        proper_divs = [d for d in divs if d < j]
        idx_proper = [divs.index(d) for d in proper_divs]

        G_proper = G_div[np.ix_(idx_proper, idx_proper)]
        g_cross = G_div[idx_j, idx_proper]

        try:
            coeffs = np.linalg.solve(G_proper, g_cross)
            dist_div_sq = G_div[idx_j, idx_j] - np.dot(g_cross, coeffs)
        except:
            dist_div_sq = 0

        # Compare with dist when projecting onto ALL {f_1,...,f_{j-1}}
        ep = euler_product_factor(j)

        print(f"\nj={j}, divisors={divs}")
        print(f"  dist^2 (all <j):     {dist_sq[j-1]:.6e}")
        print(f"  dist^2 (divisors):   {dist_div_sq:.6e}")
        print(f"  ratio (div/all):     {dist_div_sq/dist_sq[j-1]:.4f}")
        print(f"  j^2*dist^2:          {j**2*dist_sq[j-1]:.4f}")
        print(f"  j^2*dist_div^2:      {j**2*dist_div_sq:.4f}")
        print(f"  EP factor:           {ep:.4f}")

        # Projection coefficients: how much of f_j is explained by each divisor?
        print(f"  Projection coefficients (f_j onto divisors):")
        for d, c in zip(proper_divs, coeffs):
            print(f"    f_{d}: coeff = {c:>10.6f}")

    # ================================================================
    # PART 3: The key analytical structure — correlation between f_j and f_d for d|j
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: CORRELATION STRUCTURE f_j vs f_d (d|j)")
    print("-" * 70)

    # Correlation r(d,j) = G_{dj} / sqrt(G_{dd} * G_{jj})
    # For d|j, what's the pattern?

    print(f"\n{'d':>4} {'j':>4} {'j/d':>4} {'r(d,j)':>10} {'r^2':>10} {'1-r^2':>10} {'1-1/(j/d)^2':>12}")
    print("-" * 70)

    for j in [6, 10, 12, 15, 30]:
        for d in divisors(j):
            if d < j:
                r = G[d-1, j-1] / np.sqrt(G[d-1, d-1] * G[j-1, j-1])
                ratio = j // d
                predicted = 1 - 1.0/ratio**2
                print(f"{d:>4} {j:>4} {ratio:>4} {r:>10.6f} {r**2:>10.6f} "
                      f"{1-r**2:>10.6f} {predicted:>12.6f}")
        print()

    # ================================================================
    # PART 4: Multiplicative Mobius decomposition attempt
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: MOBIUS DECOMPOSITION OF dist_j^2")
    print("-" * 70)

    # If dist_j^2 has multiplicative structure, then:
    #   log(j^2 * dist_j^2) = sum_{p|j} g(p) + constant
    # for some function g(p)

    # Test: is j^2 * dist_j^2 a multiplicative function of j?
    # i.e., dist_{mn}^2 * (mn)^2 = dist_m^2 * m^2 * dist_n^2 * n^2 / C  for gcd(m,n)=1?

    print("\nMultiplicativity test: j^2*dist_j^2 for coprime products")
    print(f"{'m':>4} {'n':>4} {'m*n':>4} {'F(m)':>10} {'F(n)':>10} {'F(mn)':>10} "
          f"{'F(m)*F(n)':>12} {'ratio':>8}")
    print("-" * 75)

    F = lambda j: j**2 * dist_sq[j-1]  # F(j) = j^2 * dist_j^2

    for m, n in [(2,3), (2,5), (2,7), (3,5), (3,7), (5,7), (2,9), (3,8),
                 (4,5), (4,7), (2,11), (3,11), (5,11), (7,11), (2,13), (3,13)]:
        if gcd(m, n) == 1 and m*n <= N:
            Fm, Fn, Fmn = F(m), F(n), F(m*n)
            ratio = Fmn / (Fm * Fn) if Fm*Fn > 0 else float('nan')
            print(f"{m:>4} {n:>4} {m*n:>4} {Fm:>10.4f} {Fn:>10.4f} {Fmn:>10.4f} "
                  f"{Fm*Fn:>12.4f} {ratio:>8.4f}")

    # ================================================================
    # PART 5: Mellin-based analytical bound attempt
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: MELLIN REPRESENTATION OF dist_j^2")
    print("-" * 70)

    # dist_j^2 = min over c_{1..j-1} of ||f_j - sum c_k f_k||^2
    # In Mellin space:
    #   = (1/2pi) min integral |f_hat_j(1/2+it) - sum c_k f_hat_k(1/2+it)|^2 dt
    #
    # f_hat_j(s) = A_j(s) + B_j(s) where:
    #   A_j(s) = 1/(j(s-1))  [pole part]
    #   B_j(s) = -j^{-s} zeta(s)/s  [zeta part]
    #
    # Key: B_j(s) = -j^{-s} * zeta(s)/s
    # For d|j with j=d*m: j^{-s} = d^{-s} * m^{-s}
    # So B_j(s) = m^{-s} * B_d(s)  (multiplicative!)
    #
    # This means: the "zeta part" of f_j is a multiplicative scaling of f_d's zeta part.
    # The novelty (Schur complement) in the zeta part is controlled by
    # how much of j^{-s} can be reconstructed from {d^{-s} : d|j, d<j}

    # Test: what fraction of j^{-s} lives in span{d^{-s} : d|j, d<j}?
    print("\nMultiplicative structure of j^{-s} on the critical line:")
    print("Can j^{-1/2-it} be reconstructed from {d^{-1/2-it} : d|j, d<j}?")

    for j in [6, 12, 30]:
        divs_proper = [d for d in divisors(j) if d < j]

        # Sample at many t values and do least squares
        t_vals = np.linspace(0.1, 100, 500)

        # Build matrix: columns = d^{-1/2-it} for d in divs_proper, rows = t values
        A_mat = np.zeros((len(t_vals), len(divs_proper)), dtype=complex)
        b_vec = np.zeros(len(t_vals), dtype=complex)

        for i, t in enumerate(t_vals):
            s = 0.5 + 1j*t
            for k, d in enumerate(divs_proper):
                A_mat[i, k] = d**(-s)
            b_vec[i] = j**(-s)

        # Least squares: find c such that sum c_d d^{-s} ≈ j^{-s}
        result = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        c_opt = result[0]
        residual = b_vec - A_mat @ c_opt

        frac_explained = 1 - np.sum(np.abs(residual)**2) / np.sum(np.abs(b_vec)**2)

        print(f"\n  j={j}, divisors={divs_proper}")
        print(f"  Fraction of j^{{-s}} explained: {frac_explained:.6f}")
        print(f"  Residual ||j^{{-s}} - sum c_d d^{{-s}}||^2 / ||j^{{-s}}||^2 = {1-frac_explained:.6f}")

        ep = euler_product_factor(j)
        print(f"  Euler product factor: {ep:.6f}")
        print(f"  Residual vs EP: {(1-frac_explained)/ep:.4f}")

        # Coefficients
        print(f"  Optimal coefficients:")
        for d, c in zip(divs_proper, c_opt):
            if abs(c) > 0.001:
                print(f"    c_{d} = {c.real:>10.6f} + {c.imag:>10.6f}i  (|c|={abs(c):.6f})")

    # ================================================================
    # PART 6: The Ramanujan sum connection
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 6: RAMANUJAN SUM / MOBIUS CONNECTION")
    print("-" * 70)

    # The key identity we need:
    # For a multiplicative group, the orthogonal complement of {d^{-s} : d|j, d<j}
    # in the span of {d^{-s} : d|j} has a specific structure related to Mobius function.
    #
    # Claim: the Gram-Schmidt residual of j^{-s} w.r.t. {d^{-s} : d|j, d<j}
    # is related to sum_{d|j} mu(j/d) d^{-s} = (j^{-s} * mu)(1)
    #
    # This is the Mobius function acting on the Dirichlet series!

    # Compute the Mobius-weighted combination
    print("\nMobius-weighted combination: M_j(s) = sum_{d|j} mu(j/d) * d^{-s}")
    print("Normalized: M_j(s) / j^{-s}")

    for j in [6, 12, 30, 60]:
        divs_j = divisors(j)

        # M_j(s) at several points on the critical line
        print(f"\n  j={j}:")
        t_vals = [0, 1, 5, 14.13, 25.01]

        for t in t_vals:
            s = mpc(0.5, t)
            M_val = sum(int(sympy.mobius(j//d)) * power(mpf(d), -s) for d in divs_j)
            j_val = power(mpf(j), -s)
            ratio = M_val / j_val if abs(j_val) > 1e-20 else mpc(0)

            # Expected: M_j(s)/j^{-s} = prod_{p|j}(1 - p^{s-2s}) = prod_{p|j}(1 - p^{-s})...
            # Actually: sum_{d|j} mu(j/d) d^{-s} = j^{-s} * prod_{p|j}(1 - p^s/j^s)...
            # Let me think...
            # sum_{d|j} mu(j/d) d^{-s} = sum_{d|j} mu(d) (j/d)^{-s} = j^{-s} sum_{d|j} mu(d) d^s
            # For j = prod p_i^{a_i}: this is j^{-s} * prod_{p|j} (1 - p^s) if all a_i=1 (squarefree)

            print(f"    t={t:>6.2f}: M/j^{{-s}} = {nstr(ratio, 8)}, |M/j^{{-s}}|^2 = {float(abs(ratio)**2):.6f}")

        # The squared norm of M_j on the critical line
        # |M_j(1/2+it)|^2 / |j^{-1/2-it}|^2 = |prod(1-p^{1/2+it})|^2 for squarefree j
        # = prod |1 - p^{1/2+it}|^2 = prod (1 - 2*Re(p^{1/2+it}) + p)
        # At t=0: = prod (1 - 2*sqrt(p) + p) = prod (sqrt(p) - 1)^2
        # Hmm, that doesn't look right for a lower bound...

        # Actually the MEAN SQUARE over t should give us the Euler product:
        # (1/T) integral_0^T |M_j(1/2+it)/j^{-1/2-it}|^2 dt -> prod(1 + 1/p) as T->inf
        # No wait...

        # Let's just compute it numerically
        t_sample = np.linspace(0, 200, 5000)
        M_sq_avg = 0
        for t in t_sample:
            s = mpc(0.5, t)
            M_val = sum(int(sympy.mobius(j//d)) * power(mpf(d), -s) for d in divs_j)
            j_val = power(mpf(j), -s)
            M_sq_avg += float(abs(M_val / j_val)**2) if abs(j_val) > 1e-20 else 0
        M_sq_avg /= len(t_sample)

        ep = euler_product_factor(j)
        print(f"    Mean |M/j^{{-s}}|^2 over [0,200]: {M_sq_avg:.6f}")
        print(f"    Euler product prod(1-1/p^2): {ep:.6f}")
        print(f"    Ratio: {M_sq_avg/ep:.4f}")

    print(f"\n{'='*70}")
    print("SESSION 29 ANALYSIS COMPLETE")
    print("Next: use findings to formalize the bound")
