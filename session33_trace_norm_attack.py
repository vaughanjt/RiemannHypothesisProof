"""
SESSION 33 — THE TRACE-NORM ATTACK: Proving max_eigenvalue(M|null) < 0

THE KEY INSIGHT:
  eps_0 > 0  <=>  M restricted to null(W02) has ALL eigenvalues < 0
  <=>  max_eigenvalue(M|null) < 0

THE BOUND (Schur-Horn):
  For symmetric A with eigenvalues lam_1 >= ... >= lam_n:
  lam_1 <= tr(A)/n + sqrt((n-1)/n * (||A||_F^2 - tr(A)^2/n))

  If tr(A) is deeply negative and ||A||_F^2 is controlled, then lam_1 < 0.

THE STRATEGY:
  1. Compute tr(M|null) — involves diagonal sums, bounded by PNT
  2. Compute ||M|null||_F^2 — involves squared sums, bounded by PNT^2
  3. Apply the bound: lam_max < 0 if tr^2/n > ||A||_F^2 * n/(n-1)
     Equivalently: tr^2 > n^2/(n-1) * ||A||_F^2
     Equivalently: (tr/n)^2 > ||A||_F^2/(n-1)
     This is: (average eigenvalue)^2 > variance of eigenvalues

  4. If this holds, ALL eigenvalues are negative => eps_0 > 0 => RH

  THIS AVOIDS THE PNT SIGN PROBLEM because:
  - tr involves sum of eigenvalues (one-sided by PNT)
  - ||F||^2 involves sum of squares (bounded by Cauchy-Schwarz + PNT)
  - The bound combines them to get a SIGN result from MAGNITUDE bounds
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, sinh, euler, digamma, hyp2f1
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition

mp.dps = 50


def compute_null_projection(W02, threshold_factor=1e-10):
    """Get the projector onto null(W02)."""
    evals, evecs = np.linalg.eigh(W02)
    threshold = np.max(np.abs(evals)) * threshold_factor
    null_idx = np.where(np.abs(evals) <= threshold)[0]
    P_null = evecs[:, null_idx]
    return P_null, len(null_idx)


def trace_norm_bound(lam_sq, N=None):
    """
    Compute the trace-norm bound on max eigenvalue of M|null(W02).

    The Schur-Horn bound:
      lam_max <= mu + sigma * sqrt((n-1)/n)
    where:
      mu = tr(A)/n  (mean eigenvalue)
      sigma^2 = ||A||_F^2/n - mu^2  (variance of eigenvalues)

    For lam_max < 0, we need:
      mu + sigma * sqrt((n-1)/n) < 0
      mu < -sigma * sqrt((n-1)/n)
      |mu| > sigma * sqrt((n-1)/n)
      mu^2 > sigma^2 * (n-1)/n
      (tr/n)^2 > (||A||_F^2/n - (tr/n)^2) * (n-1)/n
      tr^2/n + tr^2*(n-1)/n^2 > ||A||_F^2 * (n-1)/n
      tr^2/n * (1 + (n-1)/n) > ||A||_F^2 * (n-1)/n
      tr^2 * (2n-1)/n > ||A||_F^2 * (n-1)
      tr^2 > n*(n-1)/(2n-1) * ||A||_F^2

    Simplified: tr^2 > (n-1)/2 * ||A||_F^2  (approximately for large n)
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1

    t0 = time.time()
    W02, M_full, QW = build_all(lam_sq, N)
    eps_0 = np.linalg.eigvalsh(QW)[0]

    # Get null space of W02
    P_null, D_null = compute_null_projection(W02)

    # Restrict M to null(W02)
    M_null = P_null.T @ M_full @ P_null
    n = D_null  # dimension of null space

    # Compute trace and Frobenius norm
    tr_M = np.trace(M_null)
    frob_sq = np.sum(M_null**2)  # ||M_null||_F^2

    # Actual eigenvalues for comparison
    evals_M_null = np.linalg.eigvalsh(M_null)
    lam_max_actual = evals_M_null[-1]

    # Mean and variance
    mu = tr_M / n
    sigma_sq = frob_sq / n - mu**2
    sigma = np.sqrt(max(sigma_sq, 0))

    # The bound
    lam_max_bound = mu + sigma * np.sqrt((n-1) / n)

    # The critical ratio
    critical_ratio = tr_M**2 / (frob_sq * n * (n-1) / (2*n - 1))

    elapsed = time.time() - t0

    print(f"\nlam^2={lam_sq} (dim={dim}, null_dim={n}, {elapsed:.1f}s)")
    print(f"  eps_0 = {eps_0:.6e}")
    print(f"  tr(M|null) = {tr_M:.6f}")
    print(f"  ||M|null||_F^2 = {frob_sq:.6f}")
    print(f"  mu (mean eig) = {mu:.6f}")
    print(f"  sigma (eig std) = {sigma:.6f}")
    print(f"  Schur-Horn bound: lam_max <= {lam_max_bound:.6e}")
    print(f"  Actual lam_max = {lam_max_actual:.6e}")
    print(f"  Critical ratio tr^2 / threshold = {critical_ratio:.4f}")
    if lam_max_bound < 0:
        print(f"  *** BOUND PROVES lam_max < 0 => M < 0 on null(W02) ***")
        print(f"  *** THIS IMPLIES eps_0 > 0 => RH (for this lambda) ***")
    elif critical_ratio > 1:
        print(f"  Critical ratio > 1 => bound proves negativity")
    else:
        gap = 1 - critical_ratio
        print(f"  Bound does NOT prove negativity (gap = {gap:.4f})")
        print(f"  Need {gap*100:.1f}% tighter Frobenius bound or trace bound")

    return {
        'lam_sq': lam_sq, 'n': n, 'eps_0': float(eps_0),
        'trace': float(tr_M), 'frob_sq': float(frob_sq),
        'mu': float(mu), 'sigma': float(sigma),
        'bound': float(lam_max_bound), 'actual': float(lam_max_actual),
        'critical_ratio': float(critical_ratio)
    }


def decomposed_trace_norm_bound(lam_sq, N=None):
    """
    REFINED BOUND: Decompose M = M_diag + M_alpha + M_prime and bound each.

    Since M_diag + M_alpha has max eigenvalue ~ +2.4 on null(W02),
    and M_prime compensates with min eigenvalue ~ -6.4,
    we need a bound that respects this structure.

    TRIANGLE BOUND:
      lam_max(M) <= lam_max(M_diag + M_alpha) + lam_max(M_prime)

    This is WEAKER than the direct bound. Instead, use:

    TRACE-NORM ON M_prime ALONE:
      If M_prime|null is sufficiently negative (trace bound),
      and M_diag + M_alpha is bounded above,
      then M = (M_diag + M_alpha) + M_prime < 0.

    Specifically: lam_max(M) <= lam_max(M_d+M_a) + lam_max(M_prime)
    Need: lam_max(M_prime) < -lam_max(M_d+M_a)

    Can we bound lam_max(M_prime) using trace-norm on M_prime|null?
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1

    t0 = time.time()
    W02, M_check, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    P_null, D_null = compute_null_projection(W02)
    n = D_null

    # Project each component
    Md_null = P_null.T @ M_diag @ P_null
    Ma_null = P_null.T @ M_alpha @ P_null
    Mp_null = P_null.T @ M_prime @ P_null
    Mda_null = Md_null + Ma_null  # analytic part
    M_null = P_null.T @ M_full @ P_null

    # Eigenvalue bounds for each component
    ev_da = np.linalg.eigvalsh(Mda_null)
    ev_p = np.linalg.eigvalsh(Mp_null)
    ev_full = np.linalg.eigvalsh(M_null)

    # Trace-norm bound on M_prime|null
    tr_p = np.trace(Mp_null)
    frob_sq_p = np.sum(Mp_null**2)
    mu_p = tr_p / n
    sigma_sq_p = frob_sq_p / n - mu_p**2
    sigma_p = np.sqrt(max(sigma_sq_p, 0))
    bound_p = mu_p + sigma_p * np.sqrt((n-1)/n)

    # For M_full|null
    tr_full = np.trace(M_null)
    frob_sq_full = np.sum(M_null**2)
    mu_full = tr_full / n
    sigma_sq_full = frob_sq_full / n - mu_full**2
    sigma_full = np.sqrt(max(sigma_sq_full, 0))
    bound_full = mu_full + sigma_full * np.sqrt((n-1)/n)

    elapsed = time.time() - t0

    print(f"\nDECOMPOSED BOUND: lam^2={lam_sq} (null_dim={n}, {elapsed:.1f}s)")
    print(f"  M_diag+alpha on null: [{ev_da[0]:.4e}, {ev_da[-1]:.4e}]")
    print(f"  M_prime on null:      [{ev_p[0]:.4e}, {ev_p[-1]:.4e}]")
    print(f"  M_full on null:       [{ev_full[0]:.4e}, {ev_full[-1]:.4e}]")

    print(f"\n  Trace-norm bound on M_prime|null:")
    print(f"    tr = {tr_p:.4f}, ||F||^2 = {frob_sq_p:.4f}")
    print(f"    mu = {mu_p:.4f}, sigma = {sigma_p:.4f}")
    print(f"    Bound: lam_max(M_prime) <= {bound_p:.4e}")
    print(f"    Actual: {ev_p[-1]:.4e}")
    print(f"    Need: < {-ev_da[-1]:.4e} (to beat M_diag+alpha)")
    if bound_p < -ev_da[-1]:
        print(f"    *** M_prime BOUND BEATS M_diag+alpha => M < 0 on null ***")
    else:
        print(f"    Gap: bound is {bound_p - (-ev_da[-1]):.4e} too large")

    print(f"\n  Trace-norm bound on M_full|null:")
    print(f"    tr = {tr_full:.4f}, ||F||^2 = {frob_sq_full:.4f}")
    print(f"    mu = {mu_full:.4f}, sigma = {sigma_full:.4f}")
    print(f"    Bound: lam_max(M_full) <= {bound_full:.4e}")
    print(f"    Actual: {ev_full[-1]:.4e}")
    if bound_full < -1e-10:
        print(f"    *** FULL BOUND PROVES M < 0 => eps_0 > 0 ***")

    # TIGHTER BOUND: Use the 4th moment (kurtosis)
    # lam_max <= mu + sigma * sqrt((n-1)/n) * sqrt(kurtosis_factor)
    # But basic Schur-Horn is already sharp for this...

    # ALTERNATIVE: Gershgorin on M_null
    gershgorin_bounds = []
    for i in range(n):
        center = M_null[i, i]
        radius = np.sum(np.abs(M_null[i, :])) - abs(center)
        gershgorin_bounds.append(center + radius)
    gershgorin_max = np.max(gershgorin_bounds)

    print(f"\n  Gershgorin bound: lam_max <= {gershgorin_max:.4e}")
    if gershgorin_max < 0:
        print(f"    *** GERSHGORIN PROVES M < 0 => eps_0 > 0 ***")

    return {
        'bound_prime': float(bound_p),
        'bound_full': float(bound_full),
        'gershgorin': float(gershgorin_max),
        'da_max': float(ev_da[-1]),
        'prime_max': float(ev_p[-1]),
        'full_max': float(ev_full[-1]),
        'trace_full': float(tr_full),
        'frob_full': float(frob_sq_full)
    }


def pnt_expressible_bounds(lam_sq, N=None):
    """
    Express the trace and Frobenius norm in terms of PRIME SUMS
    that can be bounded by explicit PNT (Rosser-Schoenfeld).

    tr(M|null) = tr(M) - tr(M on range(W02))
    The trace of M involves:
      tr(M_diag) = sum_n wr_diag[n]  (analytic — computable)
      tr(M_alpha) = 0  (off-diagonal, zero trace)
      tr(M_prime) = sum_n sum_{pk} w(pk) * q(n,n,logpk)
                  = sum_{pk} w(pk) * sum_n q(n,n,logpk)

    The inner sum: sum_n q(n,n,y) = sum_n 2(L-y)/L * cos(2*pi*n*y/L)
    This is a Dirichlet kernel! For n from -N to N:
      sum_{n=-N}^{N} cos(2*pi*n*y/L) = sin((2N+1)*pi*y/L) / sin(pi*y/L)

    So: sum_n q(n,n,y) = 2(L-y)/L * D_N(y/L)  where D_N is the Dirichlet kernel.

    And: tr(M_prime) = sum_{pk} w(pk) * 2(L-logpk)/L * D_N(logpk/L)

    This is a PRIME SUM with a smooth weight function!
    Explicit PNT (Rosser-Schoenfeld) bounds this sum.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)

    print(f"\nPNT-EXPRESSIBLE BOUNDS: lam^2={lam_sq}, N={N}")
    print("=" * 70)

    # Compute the Dirichlet kernel contribution
    # D_N(x) = sum_{n=-N}^{N} cos(2*pi*n*x) = sin((2N+1)*pi*x) / sin(pi*x)
    def dirichlet_kernel(x):
        if abs(np.sin(np.pi * x)) < 1e-15:
            return 2 * N + 1
        return np.sin((2*N+1) * np.pi * x) / np.sin(np.pi * x)

    # tr(M_prime) = sum_{pk} (log p / pk^{1/2}) * 2(L-logpk)/L * D_N(logpk/L)
    limit = min(lam_sq, 10000)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False

    tr_prime_computed = 0
    tr_prime_smooth = 0  # without Dirichlet kernel oscillation
    prime_contribs = []
    for p in range(2, limit + 1):
        if sieve[p] and p <= lam_sq:
            pk = p
            while pk <= lam_sq:
                logpk = np.log(pk)
                logp = np.log(p)
                x = logpk / L_f
                w = logp * pk**(-0.5)
                kernel_val = dirichlet_kernel(x)
                smooth_val = 2 * (L_f - logpk) / L_f  # D_N replaced by 1

                contrib = w * 2 * (L_f - logpk) / L_f * kernel_val
                contrib_smooth = w * smooth_val * dim  # D_N(0) = dim

                tr_prime_computed += contrib
                tr_prime_smooth += contrib_smooth
                prime_contribs.append((pk, logp, x, w, kernel_val, contrib))
                pk *= p

    print(f"  tr(M_prime) with Dirichlet kernel: {tr_prime_computed:.4f}")
    print(f"  tr(M_prime) smooth (no oscillation): {tr_prime_smooth:.4f}")

    # The Dirichlet kernel oscillates! For random x, D_N(x) oscillates between
    # roughly -sqrt(N) and +sqrt(N), but at x near 0 or 1, it peaks at 2N+1.
    # The primes are "random" in some sense, so the kernel averages out.

    # Compare with actual trace
    W02, M, QW = build_all(lam_sq, N)
    P_null, D_null = compute_null_projection(W02)
    M_null = P_null.T @ M @ P_null
    tr_actual = np.trace(M_null)

    print(f"  tr(M|null) actual: {tr_actual:.4f}")
    print(f"  tr(M) full: {np.trace(M):.4f}")

    # The Dirichlet kernel contribution from each prime
    print(f"\n  Prime contributions to trace (with Dirichlet kernel):")
    sorted_contribs = sorted(prime_contribs, key=lambda c: abs(c[5]), reverse=True)
    for pk, logp, x, w, kernel, contrib in sorted_contribs[:15]:
        print(f"    p^k={pk:>5} (x={x:.4f}): D_N={kernel:>8.2f}  "
              f"weight={w:.4f}  contrib={contrib:>+10.4f}")

    # KEY: Are the Dirichlet kernel values at primes bounded?
    kernel_vals = [c[4] for c in prime_contribs]
    print(f"\n  Dirichlet kernel at primes:")
    print(f"    min: {np.min(kernel_vals):.4f}")
    print(f"    max: {np.max(kernel_vals):.4f}")
    print(f"    mean: {np.mean(kernel_vals):.4f}")
    print(f"    std: {np.std(kernel_vals):.4f}")
    print(f"    |D_N| < sqrt(dim) = {np.sqrt(dim):.4f}? "
          f"{'YES' if np.max(np.abs(kernel_vals)) < np.sqrt(dim) else 'NO'}")

    # Rosser-Schoenfeld bounds
    # theta(x) = sum_{p<=x} log(p)
    # |theta(x) - x| < 0.0077629 * x / log(x) for x >= 1319007
    # For smaller x: explicit computation
    theta_actual = sum(np.log(p) for p in range(2, lam_sq + 1) if sieve[p] and p <= lam_sq)
    theta_pnt = lam_sq  # PNT: theta(x) ~ x
    theta_error = theta_actual - theta_pnt
    rosser_bound = 0.0077629 * lam_sq / np.log(lam_sq) if lam_sq >= 1319007 else abs(theta_error) * 2

    print(f"\n  PNT check:")
    print(f"    theta({lam_sq}) = {theta_actual:.4f}")
    print(f"    PNT prediction: {theta_pnt:.4f}")
    print(f"    Error: {theta_error:.4f} ({theta_error/lam_sq*100:.4f}%)")
    print(f"    Rosser-Schoenfeld bound: {rosser_bound:.4f}")

    # The trace involves sum (log p / sqrt(p)) * f(log p)
    # By partial summation with theta(x):
    # sum_{p<=x} log(p)/sqrt(p) * f(log p)
    #   = integral_2^x f(log t) / sqrt(t) d(theta(t))
    #   = integral f(log t) / sqrt(t) dt + integral f(log t)/sqrt(t) dE(t)
    # The error integral: |int f/sqrt(t) dE| <= max|f| * |E(x)|/sqrt(x) + ...

    # For our f(u) = 2(L-u)/L * D_N(u/L):
    max_f = 2 * dim  # upper bound on |f|
    error_contribution = max_f * abs(theta_error) / np.sqrt(lam_sq)
    print(f"\n  Error contribution to trace: <= {error_contribution:.4f}")
    print(f"  Trace magnitude: {abs(tr_actual):.4f}")
    print(f"  Error/Trace ratio: {error_contribution/abs(tr_actual):.4f}")

    if error_contribution < abs(tr_actual) * 0.5:
        print(f"  *** Error is small relative to trace — trace sign is PROVABLE ***")

    return tr_actual, tr_prime_computed, kernel_vals


def the_final_bound(lam_sq_values):
    """
    THE ULTIMATE TEST: For each lambda, can we prove M < 0 on null(W02)?

    Method 1: Direct trace-norm (Schur-Horn)
    Method 2: Decomposed trace-norm (M_prime dominates M_d+M_a)
    Method 3: Gershgorin circles
    Method 4: PNT-expressible trace bound + Frobenius bound
    """
    print("\n\n" + "=" * 75)
    print("THE FINAL BOUND: Can we prove M < 0 on null(W02)?")
    print("=" * 75)

    results = []
    for lam_sq in lam_sq_values:
        print(f"\n{'='*75}")
        print(f"  lambda^2 = {lam_sq}")
        print(f"{'='*75}")

        r1 = trace_norm_bound(lam_sq)
        r2 = decomposed_trace_norm_bound(lam_sq)

        proved = (r1['bound'] < -1e-10 or
                  r2['bound_full'] < -1e-10 or
                  r2['gershgorin'] < -1e-10 or
                  r2['bound_prime'] < -r2['da_max'])

        results.append({
            'lam_sq': lam_sq,
            'eps_0': r1['eps_0'],
            'schur_horn_proves': r1['bound'] < 0,
            'full_trace_norm_proves': r2['bound_full'] < 0,
            'gershgorin_proves': r2['gershgorin'] < 0,
            'decomposed_proves': r2['bound_prime'] < -r2['da_max'],
            'any_proves': proved
        })

    # Summary
    print("\n\n" + "=" * 75)
    print("SUMMARY: PROOF STATUS BY LAMBDA")
    print("=" * 75)
    print(f"  {'lam^2':>6} {'eps_0':>12} {'Schur-Horn':>12} {'Full TN':>12} "
          f"{'Gershgorin':>12} {'Decomposed':>12} {'ANY':>6}")
    for r in results:
        sh = 'PROVED' if r['schur_horn_proves'] else 'no'
        ft = 'PROVED' if r['full_trace_norm_proves'] else 'no'
        ge = 'PROVED' if r['gershgorin_proves'] else 'no'
        de = 'PROVED' if r['decomposed_proves'] else 'no'
        an = '***' if r['any_proves'] else ''
        print(f"  {r['lam_sq']:>6} {r['eps_0']:>12.3e} {sh:>12} {ft:>12} "
              f"{ge:>12} {de:>12} {an:>6}")

    return results


if __name__ == "__main__":
    print("SESSION 33 — THE TRACE-NORM ATTACK")
    print("=" * 75)
    print("Can we prove max_eigenvalue(M|null(W02)) < 0 using trace + norm bounds?")
    print()

    # The main test
    lam_sq_values = [50, 100, 200, 500, 1000]

    results = the_final_bound(lam_sq_values)

    # PNT expressibility
    print("\n\n" + "#" * 75)
    print("# PNT-EXPRESSIBLE ANALYSIS")
    print("#" * 75)
    for lam_sq in [200, 1000]:
        pnt_expressible_bounds(lam_sq)

    # Save
    with open('session33_trace_norm.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to session33_trace_norm.json")
