"""Trace formula approach: -sum Tr(L^n)/n = log det(I-L) vs log zeta(s).

KEY INSIGHT: Instead of computing det(I-L) directly (overflows),
use the trace formula:
    -log det(I - L) = sum_{n=1}^inf Tr(L^n) / n

If the operator is correct, this should equal log zeta(s).
The trace Tr(L^n) counts closed paths of length n in the
multiplication graph, weighted by (product of p^{-s} along path).

For the SYMMETRIC operator:
    Tr(L^n) = sum over n-step closed walks (j -> p1*j -> p1*p2*j -> ... -> j)
    where each step multiplies or divides by a prime.

For the DIRICHLET operator:
    Tr(L^n) = sum over n-tuples (d1,...,dn) with d1*...*dn having
    specific divisibility structure.

For the FORWARD-ONLY operator:
    Tr(L^n) = 0 for all n (no closed paths — multiplication only goes up)

This explains why forward-only cannot find zeros!

For the BACKWARD-ONLY operator:
    Tr(L^n) counts paths that divide n times and return to start.
    Only possible when j = (product of n primes) * j, impossible for n>0.
    So Tr(L^n) = 0 too. Backward-only also cannot work!

The symmetric operator's traces should decompose as:
    Tr(L^2) = 2 * sum_p p^{-2s} * #{j: pj <= N}  (go forward then back)
    Tr(L^4) = ... includes 2-prime orbits

Odd traces: Tr(L^{2k+1}) = 0 (can't return in odd steps)

So -log det(I-L_sym) = sum_{k=1}^inf Tr(L^{2k}) / (2k)
                      = sum_p f(p,N,s) + sum_{p,q} g(p,q,N,s) + ...

Compare to: log zeta(s) = sum_p sum_m p^{-ms}/m  (Euler product expansion)
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from sympy import primerange
import mpmath

t0 = time.time()
mpmath.mp.dps = 20


def get_primes_up_to(N):
    return list(primerange(2, N + 1))


def build_symmetric(s, N, primes):
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += ps
            if j % p == 0:
                L[j - 1, j // p - 1] += ps
    return L


def build_dirichlet(s, N):
    L = np.zeros((N, N), dtype=complex)
    for n in range(1, N + 1):
        for d in range(2, n + 1):
            if n % d == 0:
                m = n // d
                L[n - 1, m - 1] += complex(mpmath.power(d, -s))
    return L


def build_backward_only(s, N, primes):
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        for j in range(1, N + 1):
            if j % p == 0:
                L[j - 1, j // p - 1] += ps
    return L


# ============================================================
# Trace formula: -sum Tr(L^n)/n
# ============================================================
def trace_expansion(L, max_power=30):
    """Compute -sum_{n=1}^{max_power} Tr(L^n)/n.

    Uses matrix power iteration (more stable than L^n directly).
    Returns cumulative sums at each order.
    """
    N = L.shape[0]
    L_power = np.eye(N, dtype=complex)  # L^0 = I
    cumsum = 0.0 + 0j
    results = []

    for n in range(1, max_power + 1):
        L_power = L_power @ L  # L^n
        tr = np.trace(L_power)
        cumsum += tr / n
        results.append((n, tr, -cumsum))  # -cumsum should equal log det(I-L)

    return results


# ============================================================
# TEST 1: Trace structure — which orders contribute?
# ============================================================
print("=" * 70)
print("TEST 1: TRACE STRUCTURE Tr(L^n) FOR EACH OPERATOR")
print("=" * 70)

N_test = 200
primes = get_primes_up_to(N_test)

for s_val, s_label in [(2.0, "s=2 (real)"),
                         (mpmath.mpc(0.5, 14.134), "s=1/2+14.13i (1st zero)"),
                         (mpmath.mpc(0.5, 20.0), "s=1/2+20i (non-zero)")]:

    print(f"\n  At {s_label}:")
    log_zeta = complex(mpmath.log(mpmath.zeta(s_val)))

    for op_name, builder in [("Symmetric", lambda s: build_symmetric(s, N_test, primes)),
                              ("Dirichlet", lambda s: build_dirichlet(s, N_test)),
                              ("Backward", lambda s: build_backward_only(s, N_test, primes))]:
        L = builder(s_val)
        results = trace_expansion(L, max_power=20)

        print(f"\n    {op_name}:")
        print(f"    {'n':>4} {'Re[Tr(L^n)]':>16} {'Im[Tr(L^n)]':>16} "
              f"{'Re[-cumsum]':>16} {'Im[-cumsum]':>16}")
        for n, tr, cum in results:
            if n <= 10 or n % 5 == 0:
                print(f"    {n:>4} {tr.real:>+16.8f} {tr.imag:>+16.8f} "
                      f"{cum.real:>+16.8f} {cum.imag:>+16.8f}")

        final = results[-1][2]
        print(f"    ---")
        print(f"    -sum Tr(L^n)/n (n=1..20) = {final.real:>+.8f} {final.imag:>+.8f}i")
        print(f"    log(zeta(s))              = {log_zeta.real:>+.8f} {log_zeta.imag:>+.8f}i")
        err = abs(final - log_zeta) / abs(log_zeta) if abs(log_zeta) > 1e-30 else abs(final)
        print(f"    Relative error: {err:.4e}")


# ============================================================
# TEST 2: Critical line scan via trace formula
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: CRITICAL LINE SCAN VIA TRACE FORMULA")
print("=" * 70)
print("  Computing -sum Tr(L^n)/n vs log(zeta) along critical line")

N_op = 200
primes_op = get_primes_up_to(N_op)
t_scan = np.linspace(10, 55, 300)
max_power = 20

known_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
               37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703]

# Compute trace expansion for symmetric operator
print(f"\n  Symmetric operator, N={N_op}, max_power={max_power}...")
trace_vals = np.zeros(len(t_scan), dtype=complex)
zeta_vals = np.zeros(len(t_scan), dtype=complex)

t_start = time.time()
for i, t_val in enumerate(t_scan):
    s = mpmath.mpc(0.5, t_val)
    L = build_symmetric(s, N_op, primes_op)
    results = trace_expansion(L, max_power=max_power)
    trace_vals[i] = results[-1][2]  # -sum Tr(L^n)/n

    # Actual log zeta (handle near-zeros carefully)
    zeta_s = mpmath.zeta(s)
    if abs(zeta_s) > 1e-30:
        zeta_vals[i] = complex(mpmath.log(zeta_s))
    else:
        zeta_vals[i] = complex(mpmath.log(abs(zeta_s))) + 1j * np.pi

    if (i + 1) % 100 == 0:
        print(f"    {i+1}/{len(t_scan)} ({time.time()-t_start:.0f}s)")

print(f"  Done: {time.time()-t_start:.1f}s")

# Compare
error = np.abs(trace_vals - zeta_vals)
rel_error = error / (np.abs(zeta_vals) + 1e-30)

print(f"\n  Error statistics:")
print(f"    Mean |error|:     {np.mean(error):.4e}")
print(f"    Median |error|:   {np.median(error):.4e}")
print(f"    Mean rel error:   {np.mean(rel_error):.4e}")
print(f"    Max |error|:      {np.max(error):.4e}")


# ============================================================
# TEST 3: The spectral determinant via regularized trace
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: SPECTRAL DETERMINANT — REGULARIZED COMPUTATION")
print("=" * 70)
print("  det(I-L) = exp(-sum Tr(L^n)/n)")
print("  Compare |det| at zeros vs non-zeros")

# Use the trace formula to compute det(I-L) without overflow
def regularized_log_det(L, max_power=30):
    """Compute log det(I-L) = -sum Tr(L^n)/n via trace formula."""
    N = L.shape[0]
    L_power = np.eye(N, dtype=complex)
    logdet = 0.0 + 0j
    for n in range(1, max_power + 1):
        L_power = L_power @ L
        logdet -= np.trace(L_power) / n
    return logdet


print(f"\n  N={N_op}, max_power={max_power}")
print(f"  {'t':>8} {'Re[log det]':>14} {'Im[log det]':>14} "
      f"{'|det|':>14} {'Re[log zeta]':>14} {'Zero?':>8}")
print(f"  {'-'*76}")

for t_val in [14.13, 17.0, 21.02, 25.01, 28.0, 30.42, 32.94, 35.0,
              37.59, 40.92, 43.33, 46.0, 48.01, 49.77, 52.97]:
    s = mpmath.mpc(0.5, t_val)
    L = build_symmetric(s, N_op, primes_op)
    ld = regularized_log_det(L, max_power)

    zeta_s = mpmath.zeta(s)
    log_z = complex(mpmath.log(abs(zeta_s)))

    is_zero = "<<<" if min(abs(t_val - z) for z in known_zeros) < 0.5 else ""
    print(f"  {t_val:>8.2f} {ld.real:>+14.6f} {ld.imag:>+14.6f} "
          f"{np.exp(ld.real):>14.4e} {log_z:>+14.6f} {is_zero:>8}")


# ============================================================
# TEST 4: Analytical trace for the symmetric operator
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: ANALYTICAL TRACE — CLOSED FORM FOR Tr(L^2)")
print("=" * 70)
print("  Tr(L^2) = 2 * sum_p p^{-2s} * #{j: 1<=j<=N/p}")
print("  Expected: Tr(L^2) -> 2 * sum_p p^{-2s} * N/p as N -> inf")

for s_val, label in [(mpmath.mpf(2), "s=2"),
                      (mpmath.mpc(0.5, 14.134), "s=1/2+14.13i")]:

    # Analytical
    tr2_analytical = 0
    for p in primes:
        n_valid = N_test // p  # #{j: pj <= N}
        tr2_analytical += 2 * complex(mpmath.power(p, -2 * s_val)) * n_valid

    # Numerical
    L = build_symmetric(s_val, N_test, primes)
    tr2_numerical = np.trace(L @ L)

    # Euler product: sum_p p^{-2s}
    euler2 = sum(complex(mpmath.power(p, -2 * s_val)) for p in primes)

    print(f"\n  {label} (N={N_test}):")
    print(f"    Tr(L^2) numerical:   {tr2_numerical:.8f}")
    print(f"    Tr(L^2) analytical:  {tr2_analytical:.8f}")
    print(f"    Difference:          {abs(tr2_numerical - tr2_analytical):.2e}")
    print(f"    2*sum_p p^{{-2s}}*N/p: {2*sum(complex(mpmath.power(p,-2*s_val))*N_test/p for p in primes):.8f}")
    print(f"    sum_p p^{{-2s}}:       {euler2:.8f}")


# ============================================================
# TEST 5: How many trace terms needed to match log(zeta)?
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: CONVERGENCE OF TRACE EXPANSION TO log(zeta)")
print("=" * 70)

s_test = mpmath.mpf(2)  # Real s for clean comparison
log_zeta_2 = complex(mpmath.log(mpmath.zeta(2)))

N_big = 300
primes_big = get_primes_up_to(N_big)
L_big = build_symmetric(s_test, N_big, primes_big)
results = trace_expansion(L_big, max_power=40)

print(f"\n  s=2, N={N_big}, log(zeta(2)) = {log_zeta_2:.10f}")
print(f"  {'n':>4} {'-sum Tr/n':>20} {'error':>16}")

for n, tr, cum in results:
    err = abs(cum - log_zeta_2)
    if n <= 10 or n % 5 == 0:
        print(f"  {n:>4} {cum.real:>+20.10f} {err:>16.2e}")

# ============================================================
# TEST 6: Does the DIRICHLET operator's trace match log(zeta)?
# ============================================================
print("\n" + "=" * 70)
print("TEST 6: DIRICHLET OPERATOR TRACE EXPANSION")
print("=" * 70)

for s_val, label in [(mpmath.mpf(2), "s=2"),
                      (mpmath.mpf(3), "s=3"),
                      (mpmath.mpc(0.5, 14.134), "s=1/2+14.13i")]:

    log_z = complex(mpmath.log(mpmath.zeta(s_val)))

    for N_val in [100, 200, 300]:
        L_d = build_dirichlet(s_val, N_val)
        results = trace_expansion(L_d, max_power=20)
        final = results[-1][2]
        err = abs(final - log_z) / abs(log_z) if abs(log_z) > 1e-30 else abs(final)
        print(f"  {label:>20} N={N_val:>3}: "
              f"-sum={final.real:>+12.6f}{final.imag:>+10.4f}i  "
              f"log(z)={log_z.real:>+12.6f}{log_z.imag:>+10.4f}i  "
              f"rel_err={err:.2e}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

# Check if trace expansion of symmetric operator matches log(zeta)
# at s=2 (where convergence is easy)
L_s2 = build_symmetric(mpmath.mpf(2), 300, get_primes_up_to(300))
res_s2 = trace_expansion(L_s2, 30)
final_s2 = res_s2[-1][2]
log_z2 = complex(mpmath.log(mpmath.zeta(2)))
err_s2 = abs(final_s2 - log_z2) / abs(log_z2)

if err_s2 < 0.01:
    print(f"\n  MATCH: Symmetric trace expansion matches log(zeta) at s=2")
    print(f"  Relative error: {err_s2:.4e}")
    print(f"  This confirms: det(I - L_sym) = 1/zeta(s) in some form")
elif err_s2 < 0.1:
    print(f"\n  APPROXIMATE: Trace expansion is close but not exact")
    print(f"  Relative error: {err_s2:.4e}")
    print(f"  The operator captures part of the Euler product")
else:
    print(f"\n  MISMATCH: Trace expansion does NOT match log(zeta)")
    print(f"  Relative error: {err_s2:.4e}")
    print(f"  The symmetric operator has a DIFFERENT spectral determinant")

# Check Dirichlet
L_d2 = build_dirichlet(mpmath.mpf(2), 300)
res_d2 = trace_expansion(L_d2, 20)
final_d2 = res_d2[-1][2]
err_d2 = abs(final_d2 - log_z2) / abs(log_z2)

if err_d2 < 0.01:
    print(f"\n  Dirichlet trace matches log(zeta) at s=2: err={err_d2:.4e}")
    print(f"  The Dirichlet operator IS the zeta function in operator form!")
else:
    print(f"\n  Dirichlet trace error at s=2: {err_d2:.4e}")

print(f"\nTotal time: {time.time() - t0:.1f}s")
