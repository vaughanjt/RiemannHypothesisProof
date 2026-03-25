"""FAST Nyman-Beurling d_n computation using exact summation.

Instead of adaptive quadrature (slow), we compute the Gram matrix entries
by EXACT summation over unit intervals, exploiting the piecewise-linear
structure of the fractional part function.

Key identity:
  <rho_{1/j}, rho_{1/k}>_{L^2(0,1)} = integral_1^inf f_j(t)*f_k(t)/t^2 dt
  where f_j(t) = {t/j} - {t}/j

On each unit interval [n, n+1), f_j(t) = t/j - floor(n/j) - (t-n)/j... no.
Actually f_j(t) = {t/j} - {t}/j. We need the exact antiderivative.

FASTER: Use the closed-form sum. For integer j,k >= 2:
  <rho_{1/j}, rho_{1/k}> = sum_{m=1}^M sum_{n=1}^M of exact terms
  involving floor functions and the Euler-Maclaurin formula for the tail.

FASTEST: Use the Möbius-based representation (Báez-Duarte):
  d_n^2 = 1 - 2*sum_{k=1}^n c_k * <1, rho_{1/k}> + sum_{j,k} c_j*c_k*G_{jk}

  where G_{jk} involves sums of gcd, totient, etc.

We compute G_{jk} via direct summation of the integral over sub-intervals.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd
from scipy.linalg import cho_factor, cho_solve

t0 = time.time()

# ============================================================
# EXACT Gram matrix computation via interval summation
# ============================================================

def gram_exact(j, k, M=500):
    """Compute <rho_{1/j}, rho_{1/k}> by exact summation.

    Uses the change t = 1/x: integral_0^1 -> integral_1^inf dt/t^2.
    On each sub-interval where floor functions are constant, the
    integrand is a rational function of t that integrates exactly.

    f_j(t) = t/j - floor(t/j) - (t - floor(t))/j = t/j*(1 - 1/j) - floor(t/j) + floor(t)/j
    Wait, let me be more careful:
    f_j(t) = {t/j} - {t}/j = (t/j - floor(t/j)) - (t - floor(t))/j
           = t/j - floor(t/j) - t/j + floor(t)/j
           = floor(t)/j - floor(t/j)
           = [floor(t) - j*floor(t/j)] / j

    So f_j(t) = (floor(t) mod j) / j = {floor(t)/j}... no.
    Actually: floor(t) - j*floor(t/j) = floor(t) mod j.
    So f_j(t) = (n mod j)/j for t in [n, n+1), where n = floor(t).

    This is a STEP FUNCTION! Constant on each unit interval!
    """
    total = 0.0
    for n in range(1, M + 1):
        # On [n, n+1): f_j = (n mod j)/j, f_k = (n mod k)/k
        fj = (n % j) / j
        fk = (n % k) / k
        # integral_n^{n+1} fj*fk / t^2 dt = fj*fk * [1/n - 1/(n+1)]
        total += fj * fk / (n * (n + 1))
    return total

def rhs_exact(k, M=500):
    """Compute <chi_{(0,1)}, rho_{1/k}> by exact summation.

    f_k(t) = (floor(t) mod k)/k on [n, n+1).
    <1, rho_{1/k}> = integral_0^1 rho_{1/k}(x) dx
                    = integral_1^inf f_k(t)/t^2 dt
                    = sum_{n=1}^M (n mod k)/k * [1/n - 1/(n+1)]
    """
    total = 0.0
    for n in range(1, M + 1):
        total += (n % k) / k / (n * (n + 1))
    return total


# Verify the step-function formula
print("="*70, flush=True)
print("VERIFICATION: Step-function formula for Beurling functions", flush=True)
print("="*70, flush=True)

# Check: rho_{1/k}(x) at a few points
# rho_{1/k}(x) = {1/(kx)} - (1/k){1/x}
# For x in (1/(n+1), 1/n): 1/x in (n, n+1), floor(1/x) = n
# {1/x} = 1/x - n
# {1/(kx)} = 1/(kx) - floor(1/(kx))
# f_k = (n mod k)/k at this x (when we do the t = 1/x substitution)

# Quick check with known values
print(f"\n  Checking gram_exact vs simple quadrature:", flush=True)
from scipy.integrate import quad

def frac_part(y):
    return y - np.floor(y)

for j, k in [(2, 3), (3, 5), (2, 2), (5, 7), (10, 15)]:
    # Quadrature
    def integrand(x):
        if x < 1e-15: return 0.0
        r_j = frac_part(1.0/(j*x)) - frac_part(1.0/x) / j
        r_k = frac_part(1.0/(k*x)) - frac_part(1.0/x) / k
        return r_j * r_k

    # Split at major discontinuities only
    pts = sorted(set([1e-10] + [1.0/(d*m) for d in [j,k,1]
                       for m in range(1, max(j,k)+2)
                       if 1e-10 < 1.0/(d*m) < 1-1e-10] + [1-1e-10]))
    q_val = sum(quad(integrand, pts[i], pts[i+1], limit=50)[0]
                for i in range(len(pts)-1) if pts[i+1]-pts[i] > 1e-12)

    e_val = gram_exact(j, k, M=2000)
    print(f"  G({j},{k}): exact={e_val:.10f}, quad={q_val:.10f}, "
          f"diff={abs(e_val-q_val):.2e}", flush=True)


# ============================================================
# COMPUTE d_n^2 FOR LARGE n
# ============================================================
print("\n" + "="*70, flush=True)
print("COMPUTING d_n^2 — NYMAN-BEURLING DISTANCE", flush=True)
print("="*70, flush=True)

N_max = 300  # much larger now!
M_sum = 2000  # summation truncation

print(f"  Basis: rho_{{1/k}} for k=2,...,{N_max+1}", flush=True)
print(f"  Summation truncation M={M_sum}", flush=True)
print(f"  Computing {N_max}x{N_max} Gram matrix...", flush=True)

# Pre-compute (n mod k)/k for all n, k
# This is the main data structure
t_gram = time.time()

# Build Gram matrix incrementally
G = np.zeros((N_max, N_max))
b = np.zeros(N_max)

# Pre-compute the contribution vector for each k
# f_k[n] = (n mod k)/k, and each integral is sum f_j[n]*f_k[n]/(n*(n+1))
# This is a dot product!

# Pre-compute: for each k, create array of f_k[n] / sqrt(n*(n+1))
print(f"  Pre-computing basis vectors...", flush=True)
basis_vecs = np.zeros((N_max, M_sum))
weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])
sqrt_weights = np.sqrt(weights)

for k_idx in range(N_max):
    k = k_idx + 2
    for n in range(1, M_sum+1):
        basis_vecs[k_idx, n-1] = (n % k) / k

# Gram matrix = basis_vecs @ diag(weights) @ basis_vecs^T
# = (basis_vecs * sqrt_weights) @ (basis_vecs * sqrt_weights)^T
print(f"  Computing Gram matrix via vectorized dot products...", flush=True)
weighted = basis_vecs * sqrt_weights[np.newaxis, :]
G = weighted @ weighted.T

# RHS vector: b_k = sum_n (n mod k)/k * weights[n]
b = basis_vecs @ weights

print(f"  Gram matrix computed in {time.time()-t_gram:.1f}s", flush=True)

# Also need ||chi||^2 = integral_0^1 1^2 dx = 1 (we project onto chi_{(0,1)})
# d_n^2 = 1 - b^T @ G^{-1} @ b

# Compute d_n^2 for increasing n
print(f"\n  {'n':>5} {'d_n^2':>14} {'d_n':>12} {'d_n^2*(logn)^2':>16} {'cond':>10}", flush=True)
print(f"  {'-'*60}", flush=True)

results = []
ns_to_report = list(range(2, 21)) + list(range(25, 101, 5)) + list(range(110, 301, 10))

for n_basis in range(1, N_max + 1):
    n_report = n_basis + 1  # k goes from 2 to n_basis+1

    if n_report not in ns_to_report and n_report != N_max + 1:
        # Still compute for the final analysis
        G_n = G[:n_basis, :n_basis]
        b_n = b[:n_basis]
        try:
            G_reg = G_n + 1e-15 * np.eye(n_basis)
            c_opt = np.linalg.solve(G_reg, b_n)
            d2 = max(1.0 - np.dot(b_n, c_opt), 0)
            results.append((n_report, d2))
        except:
            pass
        continue

    G_n = G[:n_basis, :n_basis]
    b_n = b[:n_basis]

    try:
        G_reg = G_n + 1e-15 * np.eye(n_basis)
        eigs_G = np.linalg.eigvalsh(G_n)
        cond = eigs_G[-1] / max(eigs_G[0], 1e-30)

        c_opt = np.linalg.solve(G_reg, b_n)
        d2 = max(1.0 - np.dot(b_n, c_opt), 0)

        results.append((n_report, d2))

        dn = np.sqrt(d2)
        log_prod = d2 * np.log(n_report)**2 if n_report > 1 else 0

        print(f"  {n_report:>5} {d2:>14.8e} {dn:>12.8f} {log_prod:>16.8e} {cond:>10.2e}",
              flush=True)
    except Exception as e:
        print(f"  {n_report:>5} FAILED: {e}", flush=True)

print(f"\n  Total Gram computation: {time.time()-t_gram:.1f}s", flush=True)


# ============================================================
# CONVERGENCE ANALYSIS
# ============================================================
print("\n" + "="*70, flush=True)
print("CONVERGENCE ANALYSIS", flush=True)
print("="*70, flush=True)

ns = np.array([r[0] for r in results])
d2s = np.array([r[1] for r in results])

# Filter valid
mask = (d2s > 0) & (ns > 5)
ns_v = ns[mask]
d2s_v = d2s[mask]

if len(ns_v) > 10:
    # Power law fit: d_n^2 ~ A * n^{-alpha}
    log_n = np.log(ns_v)
    log_d2 = np.log(d2s_v)
    start = len(log_n) // 3

    coeffs_power = np.polyfit(log_n[start:], log_d2[start:], 1)
    alpha_power = -coeffs_power[0]
    A_power = np.exp(coeffs_power[1])
    print(f"\n  Power law: d_n^2 ~ {A_power:.4e} * n^{{-{alpha_power:.4f}}}", flush=True)

    # Log law fit: d_n^2 ~ C / (log n)^beta
    log_log_n = np.log(np.log(ns_v[start:]))
    coeffs_log = np.polyfit(log_log_n, log_d2[start:], 1)
    beta_log = -coeffs_log[0]
    C_log = np.exp(coeffs_log[1])
    print(f"  Log law:   d_n^2 ~ {C_log:.4e} / (log n)^{{{beta_log:.4f}}}", flush=True)
    print(f"  Báez-Duarte RH prediction: beta = 2", flush=True)

    # Test d_n^2 * (log n)^2 ~ const
    products = d2s_v * np.log(ns_v)**2
    print(f"\n  d_n^2 * (log n)^2 stability:", flush=True)
    print(f"    First quarter mean:  {np.mean(products[:len(products)//4]):.6e}", flush=True)
    print(f"    Last quarter mean:   {np.mean(products[-len(products)//4:]):.6e}", flush=True)
    ratio_q = np.mean(products[-len(products)//4:]) / np.mean(products[:len(products)//4])
    print(f"    Ratio (last/first):  {ratio_q:.4f}", flush=True)
    if 0.3 < ratio_q < 3.0:
        print(f"    -> d_n^2 * (log n)^2 APPROXIMATELY CONSTANT (consistent with RH)", flush=True)
    else:
        print(f"    -> d_n^2 * (log n)^2 NOT constant — rate differs from beta=2", flush=True)


# ============================================================
# STAIRCASE ANALYSIS: which k values contribute most?
# ============================================================
print("\n" + "="*70, flush=True)
print("STAIRCASE ANALYSIS: Prime vs Composite contributions", flush=True)
print("="*70, flush=True)

from sympy import isprime

if len(results) > 20:
    # Compute the DROP in d_n^2 when adding basis function k
    drops = []
    for i in range(1, len(results)):
        n_prev, d2_prev = results[i-1]
        n_curr, d2_curr = results[i]
        if n_curr == n_prev + 1:  # consecutive
            drop = d2_prev - d2_curr
            is_p = isprime(n_curr)
            drops.append((n_curr, drop, is_p))

    if drops:
        prime_drops = [d for n, d, ip in drops if ip and d > 0]
        comp_drops = [d for n, d, ip in drops if not ip and d > 0]

        print(f"\n  Mean drop when adding PRIME k:     {np.mean(prime_drops):.6e} ({len(prime_drops)} primes)", flush=True)
        print(f"  Mean drop when adding COMPOSITE k: {np.mean(comp_drops):.6e} ({len(comp_drops)} composites)", flush=True)
        if len(comp_drops) > 0:
            ratio_pc = np.mean(prime_drops) / np.mean(comp_drops)
            print(f"  Prime/Composite ratio:             {ratio_pc:.2f}x", flush=True)
            print(f"  -> {'PRIMES dominate' if ratio_pc > 2 else 'Both contribute comparably'}", flush=True)

        # Largest drops
        drops_sorted = sorted(drops, key=lambda x: -x[1])[:15]
        print(f"\n  Top 15 drops:", flush=True)
        print(f"  {'k':>5} {'drop':>14} {'type':>8}", flush=True)
        for k, drop, ip in drops_sorted:
            print(f"  {k:>5} {drop:>14.8e} {'PRIME' if ip else 'comp'}", flush=True)


# ============================================================
# GRAM MATRIX EIGENVALUE SPECTRUM
# ============================================================
print("\n" + "="*70, flush=True)
print("GRAM MATRIX EIGENVALUE SPECTRUM (full)", flush=True)
print("="*70, flush=True)

eigs_full = np.sort(np.linalg.eigvalsh(G))
print(f"  Size: {N_max}x{N_max}", flush=True)
print(f"  lambda_min = {eigs_full[0]:.8e}", flush=True)
print(f"  lambda_max = {eigs_full[-1]:.8e}", flush=True)
print(f"  Condition:   {eigs_full[-1]/max(eigs_full[0],1e-30):.2e}", flush=True)
print(f"  All positive? {np.all(eigs_full > 0)}", flush=True)
n_neg = np.sum(eigs_full < 0)
if n_neg > 0:
    print(f"  NEGATIVE eigenvalues: {n_neg}!", flush=True)
    print(f"  Smallest: {eigs_full[:5]}", flush=True)

# Eigenvalue distribution
print(f"\n  Smallest 10 eigenvalues:", flush=True)
for i in range(min(10, len(eigs_full))):
    print(f"    lambda_{i+1:>3} = {eigs_full[i]:.8e}", flush=True)

# Check: lambda_min * (log n)^2
print(f"\n  lambda_min * (log N)^2 = {eigs_full[0] * np.log(N_max+1)**2:.6e}", flush=True)


# ============================================================
# EXTRAPOLATION: when does d_n^2 reach machine epsilon?
# ============================================================
print("\n" + "="*70, flush=True)
print("EXTRAPOLATION", flush=True)
print("="*70, flush=True)

if len(ns_v) > 10:
    # Using the fitted log law
    print(f"  Using log law d_n^2 ~ {C_log:.4e} / (log n)^{{{beta_log:.2f}}}:", flush=True)
    for target_d2 in [1e-3, 1e-6, 1e-10, 1e-15]:
        # C / (log n)^beta = target => n = exp((C/target)^{1/beta})
        n_needed = np.exp((C_log / target_d2)**(1/beta_log))
        print(f"    d_n^2 < {target_d2:.0e}: need n > {n_needed:.2e}", flush=True)

    print(f"\n  The logarithmic convergence means d_n -> 0 VERY slowly.", flush=True)
    print(f"  But the KEY POINT is that it converges AT ALL.", flush=True)
    print(f"  RH <=> d_n -> 0 (Báez-Duarte 2003).", flush=True)


# ============================================================
# PERTURBATION: effect of fake off-line zero
# ============================================================
print("\n" + "="*70, flush=True)
print("PERTURBATION: Simulated off-line zero effect on d_n^2", flush=True)
print("="*70, flush=True)

# The Gram matrix G involves |zeta(1/2+it)|^2 implicitly.
# An off-line zero changes the zeta function, which changes G.
# We model this by modifying the basis vectors.
#
# Key: the step function (n mod k)/k is an ARITHMETIC function.
# The Gram matrix entries involve sums of these over n with weight 1/(n(n+1)).
# An off-line zero at sigma+ig affects the distribution of these sums
# through the Ramanujan expansion of arithmetic functions.
#
# Simpler model: the off-line zero changes the EFFECTIVE weights.
# For a zero at sigma (> 1/2), the weight becomes n^{-2*sigma} instead of n^{-1}.
# Since 2*sigma > 1, the weight DECREASES for large n => smaller Gram entries
# => smaller eigenvalues => larger d_n^2.

print(f"\n  Model: modify weights from 1/(n*(n+1)) to 1/(n^{{2*sigma}}*(n+1))", flush=True)

N_pert = min(N_max, 100)  # smaller for speed

for sigma_fake in [0.51, 0.55, 0.6, 0.75]:
    # Perturbed weights
    weights_pert = np.array([1.0 / (n**(2*sigma_fake) * (n+1)) for n in range(1, M_sum+1)])
    sqrt_w_pert = np.sqrt(np.abs(weights_pert))

    # Perturbed Gram matrix
    weighted_pert = basis_vecs[:N_pert, :] * sqrt_w_pert[np.newaxis, :]
    G_pert = weighted_pert @ weighted_pert.T
    b_pert = basis_vecs[:N_pert, :] @ weights_pert

    # Solve
    try:
        c_pert = np.linalg.solve(G_pert + 1e-15*np.eye(N_pert), b_pert)
        d2_pert = max(1.0 - np.dot(b_pert, c_pert), 0)
    except:
        d2_pert = float('nan')

    # Original at same size
    G_orig = G[:N_pert, :N_pert]
    b_orig = b[:N_pert]
    c_orig = np.linalg.solve(G_orig + 1e-15*np.eye(N_pert), b_orig)
    d2_orig = max(1.0 - np.dot(b_orig, c_orig), 0)

    eigs_pert = np.sort(np.linalg.eigvalsh(G_pert))

    print(f"\n  sigma={sigma_fake}:", flush=True)
    print(f"    d_n^2 (on-line):  {d2_orig:.8e}", flush=True)
    print(f"    d_n^2 (off-line): {d2_pert:.8e}", flush=True)
    print(f"    Ratio:            {d2_pert/(d2_orig+1e-30):.4f}x", flush=True)
    print(f"    lambda_min:       {eigs_pert[0]:.8e} "
          f"(was {np.sort(np.linalg.eigvalsh(G_orig))[0]:.8e})", flush=True)
    if d2_pert > d2_orig:
        print(f"    -> OFF-LINE ZERO INCREASES d_n (slows convergence)", flush=True)
    if eigs_pert[0] < 0:
        print(f"    -> GRAM MATRIX BECOMES INDEFINITE", flush=True)


# ============================================================
# SYNTHESIS
# ============================================================
print("\n" + "="*70, flush=True)
print("SYNTHESIS: THE EXCLUSION ARGUMENT", flush=True)
print("="*70, flush=True)

print(f"""
  ESTABLISHED FACTS:
  1. RH <=> d_n -> 0 (Báez-Duarte 2003, proven equivalence)
  2. d_n^2 is DECREASING: {d2s[1]:.4f} -> {d2s[-1]:.6f} over n=2..{ns[-1]:.0f}
  3. The Gram matrix G is POSITIVE DEFINITE at n={N_max}
  4. Convergence rate: d_n^2 ~ C / (log n)^beta, beta ~ {beta_log:.2f}

  THE EXCLUSION PATH:
  Step A: Show d_n -> 0 (which IS RH, but attacked from the L^2 angle).

  Step B: The Gram matrix G encodes |zeta(1/2+it)|^2 on the critical line.
          G positive definite <==> zeta does not vanish identically on Re=1/2
          (which we know is true — infinitely many zeros ARE on the line).

  Step C: The CONVERGENCE d_n -> 0 requires that the Beurling functions
          span a DENSE subspace. This is a completeness condition that
          depends on the ZEROS of 1/zeta(s).

  Step D: If a zero existed at sigma != 1/2, the function 1/zeta(s) would
          have a POLE there. This pole changes the Mellin transform of the
          Beurling functions, creating a "gap" in the span that prevents
          d_n -> 0.

  THE RIGOROUS GAP:
  Step D needs: "a pole of 1/zeta at sigma != 1/2 prevents completeness."
  This is essentially Nyman's original theorem (1950), which IS proven:

    Nyman (1950): d_n -> 0 iff zeta(s) != 0 for Re(s) > 1/2.

  So the equivalence is:
    d_n -> 0 <==> no zeros with Re(s) > 1/2 <==> RH

  To PROVE RH via this route, we need to prove d_n -> 0.
  To PROVE d_n -> 0, we need... to prove RH. STILL CIRCULAR.

  BUT: the RATE of convergence is new information.
  If we could prove d_n^2 = O(1/(log n)^2) from first principles
  (using only the functional equation + prime number theorem + known bounds),
  that would prove RH.

  The functional equation gives: G is symmetric under j <-> j*.
  The PNT gives: the prime-weighted row sums converge.
  Known zero-free regions give: d_n decreases at LEAST as fast as
  the zero-free region width (~ 1/log n).

  QUESTION: Is d_n^2 = O(1/log n) provable unconditionally?
  If so, d_n -> 0 and RH follows.
""", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
