"""Asymmetric prime transfer operator with functional equation chi(s) weighting.

The Connes adelic flow in matrix form:
  (L_s f)(j) = sum_{p prime} p^{-s} f(pj)                  [forward shift]
             + chi(s) * sum_{p|j, p prime} p^{-s} f(j/p)    [backward shift]

where chi(s) = 2^s * pi^{s-1} * sin(pi*s/2) * gamma(1-s)
is the functional equation factor: zeta(s) = chi(s) * zeta(1-s).

Hypothesis: The symmetric version (chi=1) finds zeta zeros but also produces
spurious zeros from the backward shift. The functional equation factor chi(s)
should cancel the spurious zeros because:
  - At true zeta zeros, chi(s) adjusts the backward contribution to be
    exactly what's needed for det(I - L_s) = 0
  - At spurious locations, chi(s) destructively interferes with the
    backward contribution

If det(I - L_s) = 0 precisely at zeta zeros (and nowhere else),
this IS the transfer operator whose Fredholm determinant is 1/zeta(s).

Previous result (symmetric, chi=1):
  - Found 5 known zeros: t=14.13, 21.02, 30.42, 40.92, 48.01
  - But also produced spurious minima

This script tests:
  1. Asymmetric operator with exact chi(s) weighting
  2. Comparison: symmetric vs asymmetric on same scan range
  3. Convergence of det(I - L_s) toward 1/zeta(s) at real s
  4. Phase analysis: how chi(s) rotates the backward contribution
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.stats import pearsonr
from sympy import isprime, primerange
import mpmath

t0 = time.time()
mpmath.mp.dps = 30


# ============================================================
# chi(s): the functional equation factor
# ============================================================
def chi(s):
    """Compute chi(s) = 2^{s-1} * pi^s / (cos(pi*s/2) * gamma(s)).

    Equivalent to the standard form 2^s * pi^{s-1} * sin(pi*s/2) * gamma(1-s)
    but avoids gamma poles at positive integers by using the reflection formula.
    Derived via: gamma(1-s) = pi / (sin(pi*s) * gamma(s)) and
    sin(pi*s) = 2*sin(pi*s/2)*cos(pi*s/2).
    """
    return (
        mpmath.power(2, s - 1)
        * mpmath.power(mpmath.pi, s)
        / (mpmath.cos(mpmath.pi * s / 2) * mpmath.gamma(s))
    )


# ============================================================
# Operator builders
# ============================================================
def get_primes_up_to(N):
    """Return list of primes up to N."""
    return list(primerange(2, N + 1))


def build_symmetric_transfer(s, N, primes=None):
    """Symmetric prime transfer: equal weight on forward and backward shifts.

    (L_s f)(j) = sum_p p^{-s} [f(pj) + f(j/p)]
    where f(j/p) only contributes when p divides j.
    """
    if primes is None:
        primes = get_primes_up_to(N)
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        for j in range(1, N + 1):
            # Forward shift: j -> pj (if pj <= N)
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += ps  # row j, column pj
            # Backward shift: j -> j/p (if p divides j)
            if j % p == 0:
                jp = j // p
                L[j - 1, jp - 1] += ps  # row j, column j/p
    return L


def build_connes_transfer(s, N, primes=None):
    """Asymmetric (Connes) transfer: chi(s)-weighted backward shifts.

    (L_s f)(j) = sum_p p^{-s} f(pj)                    [forward]
               + chi(s) * sum_{p|j} p^{-s} f(j/p)      [backward]
    """
    if primes is None:
        primes = get_primes_up_to(N)
    chi_s = complex(chi(s))
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        for j in range(1, N + 1):
            # Forward shift: j -> pj (if pj <= N)
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += ps
            # Backward shift with chi(s) weight: j -> j/p (if p|j)
            if j % p == 0:
                jp = j // p
                L[j - 1, jp - 1] += chi_s * ps
    return L


def build_connes_transfer_v2(s, N, primes=None):
    """Variant: per-prime chi weighting via local Euler factor.

    Instead of global chi(s), weight each backward shift by the
    local Euler factor: (1 - p^{-s}) / (1 - p^{s-1}).
    This is the ratio that makes each prime's contribution satisfy
    the functional equation locally.
    """
    if primes is None:
        primes = get_primes_up_to(N)
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        p1ms = complex(mpmath.power(p, s - 1))
        # Local functional equation weight for backward shift
        local_weight = ps / p1ms if abs(p1ms) > 1e-30 else 0.0
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += ps
            if j % p == 0:
                jp = j // p
                L[j - 1, jp - 1] += local_weight
    return L


# ============================================================
# Fredholm determinant computation
# ============================================================
def fredholm_det(L):
    """Compute det(I - L) using numpy."""
    N = L.shape[0]
    return np.linalg.det(np.eye(N, dtype=complex) - L)


# ============================================================
# TEST 1: Verify chi(s) at known points
# ============================================================
print("=" * 70)
print("TEST 1: VERIFY chi(s) FUNCTIONAL EQUATION FACTOR")
print("=" * 70)

for s_val in [2, 3, 0.5 + 14.134j, 0.5 + 21.022j]:
    s_val = mpmath.mpc(s_val)
    chi_val = chi(s_val)
    zeta_s = mpmath.zeta(s_val)
    zeta_1ms = mpmath.zeta(1 - s_val)
    ratio = zeta_s / (chi_val * zeta_1ms) if abs(chi_val * zeta_1ms) > 1e-30 else "N/A"
    print(f"  s={str(s_val):>20}  |chi|={float(abs(chi_val)):.6f}  "
          f"zeta(s)/(chi*zeta(1-s))={complex(ratio) if ratio != 'N/A' else 'N/A'}")


# ============================================================
# TEST 2: Compare operators at real s (convergence check)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: FREDHOLM DETERMINANT AT REAL s — CONVERGENCE TO 1/zeta(s)")
print("=" * 70)

N_test = 150
primes = get_primes_up_to(N_test)

print(f"\n  N={N_test}, {len(primes)} primes")
print(f"  {'s':>6} {'|det_sym|':>12} {'|det_chi|':>12} {'|det_v2|':>12} "
      f"{'|1/zeta|':>12} {'sym/inv_z':>10} {'chi/inv_z':>10} {'v2/inv_z':>10}")
print(f"  {'-'*96}")

for s_real in [2.0, 2.5, 3.0, 4.0, 1.5]:
    s_val = mpmath.mpf(s_real)
    zeta_val = complex(mpmath.zeta(s_val))
    inv_zeta = 1.0 / zeta_val

    L_sym = build_symmetric_transfer(s_val, N_test, primes)
    L_chi = build_connes_transfer(s_val, N_test, primes)
    L_v2 = build_connes_transfer_v2(s_val, N_test, primes)

    det_sym = fredholm_det(L_sym)
    det_chi = fredholm_det(L_chi)
    det_v2 = fredholm_det(L_v2)

    r_sym = abs(det_sym / inv_zeta) if abs(inv_zeta) > 1e-30 else 0
    r_chi = abs(det_chi / inv_zeta) if abs(inv_zeta) > 1e-30 else 0
    r_v2 = abs(det_v2 / inv_zeta) if abs(inv_zeta) > 1e-30 else 0

    print(f"  {s_real:>6.1f} {abs(det_sym):>12.6f} {abs(det_chi):>12.6f} "
          f"{abs(det_v2):>12.6f} {abs(inv_zeta):>12.6f} "
          f"{r_sym:>10.4f} {r_chi:>10.4f} {r_v2:>10.4f}")


# ============================================================
# TEST 3: CRITICAL LINE SCAN — the main event
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: CRITICAL LINE SCAN — SYMMETRIC vs CONNES vs V2")
print("=" * 70)

N_op = 150
primes = get_primes_up_to(N_op)
t_scan = np.linspace(10, 55, 500)

known_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
               37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703]

det_sym_scan = np.zeros(len(t_scan), dtype=complex)
det_chi_scan = np.zeros(len(t_scan), dtype=complex)
det_v2_scan = np.zeros(len(t_scan), dtype=complex)

print(f"\n  Scanning {len(t_scan)} points on s = 1/2 + it, t in [10, 55], N={N_op}...")
t_start = time.time()

for i, t_val in enumerate(t_scan):
    s = mpmath.mpc(0.5, t_val)

    L_sym = build_symmetric_transfer(s, N_op, primes)
    L_chi = build_connes_transfer(s, N_op, primes)
    L_v2 = build_connes_transfer_v2(s, N_op, primes)

    det_sym_scan[i] = fredholm_det(L_sym)
    det_chi_scan[i] = fredholm_det(L_chi)
    det_v2_scan[i] = fredholm_det(L_v2)

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (len(t_scan) - i - 1) / rate
        print(f"    {i+1}/{len(t_scan)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

print(f"  Scan complete: {time.time() - t_start:.1f}s")


# ============================================================
# Find and compare minima
# ============================================================
def find_minima(det_scan, t_values, depth_cutoff_pct=50):
    """Find local minima in |det| scan."""
    abs_det = np.abs(det_scan)
    minima = []
    for i in range(1, len(abs_det) - 1):
        if abs_det[i] < abs_det[i - 1] and abs_det[i] < abs_det[i + 1]:
            minima.append((t_values[i], abs_det[i]))
    minima.sort(key=lambda x: x[1])
    # Keep top percentile
    if minima:
        cutoff = np.percentile([m[1] for m in minima], depth_cutoff_pct)
        minima = [(t, d) for t, d in minima if d <= cutoff]
    return minima


def score_minima(minima, known, threshold=0.5):
    """Count matches and spurious minima."""
    matched_zeros = set()
    spurious = 0
    for t_min, _ in minima:
        dists = [abs(t_min - z) for z in known]
        best_idx = np.argmin(dists)
        if dists[best_idx] < threshold:
            matched_zeros.add(best_idx)
        else:
            spurious += 1
    return len(matched_zeros), spurious, matched_zeros


print("\n" + "-" * 70)
print("MINIMA ANALYSIS")
print("-" * 70)

for name, det_scan in [("Symmetric (chi=1)", det_sym_scan),
                        ("Connes (chi(s))", det_chi_scan),
                        ("Local Euler (v2)", det_v2_scan)]:
    minima = find_minima(det_scan, t_scan)
    n_match, n_spurious, matched = score_minima(minima, known_zeros)
    total_possible = sum(1 for z in known_zeros if 10 < z < 55)

    print(f"\n  {name}:")
    print(f"    Total minima: {len(minima)}")
    print(f"    Matched zeros: {n_match}/{total_possible}")
    print(f"    Spurious: {n_spurious}")
    print(f"    Precision: {n_match/(n_match+n_spurious)*100:.0f}%" if (n_match + n_spurious) > 0 else "    Precision: N/A")

    print(f"\n    {'t_min':>10} {'|det|':>12} {'Nearest zero':>14} {'Dist':>8} {'Match?':>8}")
    for t_min, d_min in minima[:15]:
        dists = [abs(t_min - z) for z in known_zeros]
        best = np.argmin(dists)
        tag = "YES" if dists[best] < 0.5 else "no"
        print(f"    {t_min:>10.4f} {d_min:>12.4e} {known_zeros[best]:>14.4f} "
              f"{dists[best]:>8.4f} {tag:>8}")


# ============================================================
# TEST 4: Phase analysis — how chi(s) affects the operator
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: chi(s) PHASE AND MAGNITUDE ON THE CRITICAL LINE")
print("=" * 70)

print(f"\n  {'t':>8} {'|chi|':>10} {'arg(chi)/pi':>12} {'|det_sym|':>12} "
      f"{'|det_chi|':>12} {'ratio':>10} {'zeta zero?':>12}")
print(f"  {'-'*80}")

for t_val in [14.13, 21.02, 25.01, 30.42, 32.94, 37.59, 40.92, 43.33, 48.01, 49.77]:
    s = mpmath.mpc(0.5, t_val)
    chi_val = complex(chi(s))
    chi_mag = abs(chi_val)
    chi_phase = np.angle(chi_val) / np.pi

    idx = np.argmin(np.abs(t_scan - t_val))
    ds = abs(det_sym_scan[idx])
    dc = abs(det_chi_scan[idx])
    ratio = dc / ds if ds > 1e-30 else float("inf")

    is_zero = "<<< ZERO" if min(abs(t_val - z) for z in known_zeros) < 0.5 else ""
    print(f"  {t_val:>8.2f} {chi_mag:>10.4f} {chi_phase:>+12.4f} "
          f"{ds:>12.4e} {dc:>12.4e} {ratio:>10.4f} {is_zero:>12}")


# ============================================================
# TEST 5: Convergence in N — does increasing matrix size help?
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: N-CONVERGENCE — HOW OPERATOR IMPROVES WITH SIZE")
print("=" * 70)

# Test at first zeta zero t=14.134
s_test = mpmath.mpc(0.5, 14.134)

print(f"\n  Testing at s = 0.5 + 14.134i (first zeta zero)")
print(f"  {'N':>6} {'|det_sym|':>12} {'|det_chi|':>12} {'|det_v2|':>12} {'primes':>8}")
print(f"  {'-'*54}")

for N_val in [50, 100, 150, 200, 300]:
    p_list = get_primes_up_to(N_val)
    L_s = build_symmetric_transfer(s_test, N_val, p_list)
    L_c = build_connes_transfer(s_test, N_val, p_list)
    L_v2 = build_connes_transfer_v2(s_test, N_val, p_list)

    d_s = abs(fredholm_det(L_s))
    d_c = abs(fredholm_det(L_c))
    d_v2 = abs(fredholm_det(L_v2))

    print(f"  {N_val:>6} {d_s:>12.4e} {d_c:>12.4e} {d_v2:>12.4e} {len(p_list):>8}")

# And at a NON-zero point for comparison
print(f"\n  Testing at s = 0.5 + 20.0i (NOT a zeta zero)")
s_nonzero = mpmath.mpc(0.5, 20.0)

print(f"  {'N':>6} {'|det_sym|':>12} {'|det_chi|':>12} {'|det_v2|':>12}")
print(f"  {'-'*46}")

for N_val in [50, 100, 150, 200, 300]:
    p_list = get_primes_up_to(N_val)
    L_s = build_symmetric_transfer(s_nonzero, N_val, p_list)
    L_c = build_connes_transfer(s_nonzero, N_val, p_list)
    L_v2 = build_connes_transfer_v2(s_nonzero, N_val, p_list)

    d_s = abs(fredholm_det(L_s))
    d_c = abs(fredholm_det(L_c))
    d_v2 = abs(fredholm_det(L_v2))

    print(f"  {N_val:>6} {d_s:>12.4e} {d_c:>12.4e} {d_v2:>12.4e}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

# Score all three
results = {}
for name, det_scan in [("Symmetric", det_sym_scan),
                        ("Connes_chi", det_chi_scan),
                        ("Local_Euler", det_v2_scan)]:
    minima = find_minima(det_scan, t_scan)
    n_match, n_spurious, _ = score_minima(minima, known_zeros)
    precision = n_match / (n_match + n_spurious) if (n_match + n_spurious) > 0 else 0
    results[name] = (n_match, n_spurious, precision)
    print(f"\n  {name:>15}: {n_match} zeros found, {n_spurious} spurious, "
          f"precision={precision:.1%}")

best = max(results, key=lambda k: (results[k][2], results[k][0]))
print(f"\n  >>> Best operator: {best}")

if results[best][2] > 0.8:
    print(f"  >>> HIGH PRECISION — chi(s) weighting works!")
    print(f"  >>> This may be the Connes adelic flow in matrix form.")
elif results[best][0] > results["Symmetric"][0]:
    print(f"  >>> chi(s) finds MORE zeros — operator improving.")
elif results[best][2] > results["Symmetric"][2]:
    print(f"  >>> chi(s) reduces spurious zeros — cleaner spectrum.")
else:
    print(f"  >>> chi(s) weighting needs refinement.")
    print(f"  >>> Consider: normalization, truncation boundary effects,")
    print(f"  >>> or modified weighting (chi(s)^alpha for alpha != 1).")

print(f"\nTotal time: {time.time() - t0:.1f}s")
