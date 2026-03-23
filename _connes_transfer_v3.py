"""Transfer operator v3: Phase winding detection + gauge symmetry breaking.

KEY INSIGHT FROM V2:
Hermitian operators L(t) = sum_p f(p) * [p^{-it} A_p + p^{+it} A_p^T]
are unitarily gauge-equivalent to L(0) via D(t) = diag(n^{-it}).
Result: eigenvalues are t-INDEPENDENT. Cannot find zeros.

The SYMMETRIC operator breaks this gauge symmetry:
L_sym(t) = sum_p p^{-1/2-it} (A_p + A_p^T)
D(t) L_sym(0) D(t)^{-1} gives p^{-it} on forward, p^{+it} on backward,
but L_sym has p^{-it} on BOTH. Gauge is broken → t-dependent eigenvalues.

This script:
1. Uses the symmetric operator (the only one that works)
2. Employs PHASE WINDING NUMBER to detect true zeros vs spurious minima
   - True zero: det(I-L) winds around origin → winding number ±1
   - Spurious minimum: |det| dips but doesn't wind → winding number 0
3. Tests Dirichlet convolution operator (the truncated Euler product itself)
4. Tests at larger N for better convergence
5. Explores gauge-symmetry-breaking variants
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from sympy import isprime, primerange
import mpmath

t0 = time.time()
mpmath.mp.dps = 20


def get_primes_up_to(N):
    return list(primerange(2, N + 1))


# ============================================================
# Operators
# ============================================================

def build_symmetric(s, N, primes):
    """Symmetric prime transfer: p^{-s} on both forward and backward."""
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
    """Dirichlet convolution operator: (L_s f)(n) = sum_{d|n,d>1} d^{-s} f(n/d).
    This IS the truncated Euler product in operator form.
    """
    L = np.zeros((N, N), dtype=complex)
    for n in range(1, N + 1):
        for d in range(2, n + 1):
            if n % d == 0:
                m = n // d
                L[n - 1, m - 1] += complex(mpmath.power(d, -s))
    return L


def build_asymmetric_log(s, N, primes):
    """Gauge-breaking variant: log(p) * p^{-s} forward, p^{-s} backward.

    The log(p) factor on forward only breaks the gauge symmetry differently
    from the symmetric operator (which has equal coefficients on both).
    """
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += np.log(p) * ps  # forward: log(p)*p^{-s}
            if j % p == 0:
                L[j - 1, j // p - 1] += ps  # backward: p^{-s}
    return L


def build_forward_only(s, N, primes):
    """Forward-only: (L f)(j) = sum_p p^{-s} f(pj). No backward shift."""
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += ps
    return L


def build_backward_only(s, N, primes):
    """Backward-only: (L f)(j) = sum_{p|j} p^{-s} f(j/p)."""
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        for j in range(1, N + 1):
            if j % p == 0:
                L[j - 1, j // p - 1] += ps
    return L


# ============================================================
# Phase winding detection
# ============================================================

def compute_winding_numbers(det_values, t_values):
    """Compute winding number of det around the origin.

    For each local minimum of |det|, count how many times det winds
    around the origin in a neighborhood. True zeros have winding ±1.
    """
    phases = np.angle(det_values)
    # Unwrap to get continuous phase
    uphases = np.unwrap(phases)

    # Total winding over the scan
    total_winding = (uphases[-1] - uphases[0]) / (2 * np.pi)

    # Local winding at each point (derivative of unwrapped phase)
    dt = np.diff(t_values)
    dphase = np.diff(uphases)
    winding_rate = dphase / (2 * np.pi)  # winding per unit t

    return uphases, total_winding, winding_rate


def find_zeros_by_winding(det_values, t_values, window=5):
    """Find zeros using phase winding in local windows.

    A true zero of det has a ±2π phase jump in the unwrapped phase.
    """
    uphases = np.unwrap(np.angle(det_values))
    zeros = []

    for i in range(window, len(t_values) - window):
        # Phase change over local window
        dphase = uphases[i + window] - uphases[i - window]
        winding = dphase / (2 * np.pi)

        # A zero corresponds to winding ≈ ±1
        if abs(abs(winding) - 1.0) < 0.3:
            # Refine: find the minimum |det| in this window
            local_abs = np.abs(det_values[i - window:i + window + 1])
            local_min_idx = np.argmin(local_abs) + i - window
            t_zero = t_values[local_min_idx]
            det_min = abs(det_values[local_min_idx])

            # Avoid duplicates
            if not zeros or abs(t_zero - zeros[-1][0]) > 0.5:
                zeros.append((t_zero, det_min, winding))

    return zeros


# ============================================================
# TEST 1: Dense scan with multiple operators
# ============================================================
print("=" * 70)
print("TEST 1: DENSE CRITICAL LINE SCAN (1000 points)")
print("=" * 70)

known_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
               37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703]

# Dense scan for winding number detection
N_op = 200
primes = get_primes_up_to(N_op)
t_scan = np.linspace(10, 55, 1000)

operators = [
    ("Symmetric", lambda s: build_symmetric(s, N_op, primes)),
    ("Dirichlet", lambda s: build_dirichlet(s, N_op)),
    ("AsymLog", lambda s: build_asymmetric_log(s, N_op, primes)),
    ("Forward", lambda s: build_forward_only(s, N_op, primes)),
    ("Backward", lambda s: build_backward_only(s, N_op, primes)),
]

det_scans = {}

for op_name, builder in operators:
    print(f"\n  Scanning {op_name} (N={N_op})...")
    t_start = time.time()
    dets = np.zeros(len(t_scan), dtype=complex)

    for i, t_val in enumerate(t_scan):
        s = mpmath.mpc(0.5, t_val)
        L = builder(s)
        dets[i] = np.linalg.det(np.eye(N_op, dtype=complex) - L)

        if (i + 1) % 250 == 0:
            elapsed = time.time() - t_start
            print(f"    {i+1}/1000 ({elapsed:.0f}s)")

    det_scans[op_name] = dets
    print(f"    Done: {time.time() - t_start:.1f}s")


# ============================================================
# TEST 2: Phase winding analysis
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: PHASE WINDING ZERO DETECTION")
print("=" * 70)

total_possible = sum(1 for z in known_zeros if 10 < z < 55)

for op_name in det_scans:
    dets = det_scans[op_name]
    uphases, total_winding, _ = compute_winding_numbers(dets, t_scan)

    # Find zeros by winding
    winding_zeros = find_zeros_by_winding(dets, t_scan, window=8)

    # Also find by minima (traditional)
    abs_det = np.abs(dets)
    minima = []
    for i in range(1, len(abs_det) - 1):
        if abs_det[i] < abs_det[i - 1] and abs_det[i] < abs_det[i + 1]:
            minima.append((t_scan[i], abs_det[i]))
    minima.sort(key=lambda x: x[1])

    # Score winding zeros
    w_matched = set()
    w_spurious = 0
    for t_z, _, _ in winding_zeros:
        dists = [abs(t_z - z) for z in known_zeros]
        best = np.argmin(dists)
        if dists[best] < 0.5:
            w_matched.add(best)
        else:
            w_spurious += 1

    # Score minima (top 15)
    m_matched = set()
    m_spurious = 0
    for t_m, _ in minima[:15]:
        dists = [abs(t_m - z) for z in known_zeros]
        best = np.argmin(dists)
        if dists[best] < 0.5:
            m_matched.add(best)
        else:
            m_spurious += 1

    print(f"\n  {op_name}:")
    print(f"    Total phase winding: {total_winding:+.1f} (expect ~{total_possible} for all zeros)")
    print(f"    Winding zeros: {len(w_matched)}/{total_possible} found, "
          f"{w_spurious} spurious, "
          f"precision={len(w_matched)/(len(w_matched)+w_spurious):.0%}" if (len(w_matched) + w_spurious) > 0 else "")
    print(f"    Minima zeros:  {len(m_matched)}/{total_possible} found, "
          f"{m_spurious} spurious")

    if winding_zeros:
        print(f"\n    {'t_zero':>10} {'|det|':>12} {'winding':>8} {'Near':>12} {'Dist':>8} {'?':>5}")
        for t_z, d_z, w in winding_zeros:
            dists = [abs(t_z - z) for z in known_zeros]
            best = np.argmin(dists)
            tag = "YES" if dists[best] < 0.5 else ""
            print(f"    {t_z:>10.4f} {d_z:>12.4e} {w:>+8.2f} "
                  f"{known_zeros[best]:>12.4f} {dists[best]:>8.4f} {tag:>5}")


# ============================================================
# TEST 3: Phase portrait at known zeros
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: PHASE PORTRAIT NEAR KNOWN ZEROS")
print("=" * 70)

# For the symmetric operator, zoom into ±1 around each known zero
# and measure the phase winding
op_name = "Symmetric"
dets = det_scans[op_name]
uphases = np.unwrap(np.angle(dets))

print(f"\n  {op_name} operator, N={N_op}:")
print(f"  {'Zero':>8} {'Phase_before':>14} {'Phase_after':>14} "
      f"{'Winding':>10} {'min|det|':>12} {'Detected':>10}")

for z in known_zeros:
    if z < t_scan[0] or z > t_scan[-1]:
        continue
    # Find indices bracketing the zero
    i_before = np.searchsorted(t_scan, z - 1.0)
    i_after = np.searchsorted(t_scan, z + 1.0)
    i_zero = np.searchsorted(t_scan, z)

    if i_before >= len(uphases) or i_after >= len(uphases):
        continue

    phase_before = uphases[i_before]
    phase_after = uphases[min(i_after, len(uphases) - 1)]
    winding = (phase_after - phase_before) / (2 * np.pi)

    # Min |det| near zero
    local_abs = np.abs(dets[max(0, i_zero - 5):min(len(dets), i_zero + 5)])
    min_det = np.min(local_abs) if len(local_abs) > 0 else float("inf")

    detected = "YES" if abs(winding) > 0.5 or min_det < 1e-3 else "no"
    print(f"  {z:>8.4f} {phase_before:>14.4f} {phase_after:>14.4f} "
          f"{winding:>+10.4f} {min_det:>12.4e} {detected:>10}")


# ============================================================
# TEST 4: N-scaling of the symmetric operator
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: N-SCALING — SYMMETRIC OPERATOR ZERO DETECTION vs N")
print("=" * 70)

for N_val in [100, 150, 200, 300]:
    p_list = get_primes_up_to(N_val)
    # Scan at 500 points
    t_dense = np.linspace(10, 55, 500)
    dets_n = np.zeros(len(t_dense), dtype=complex)

    t_start = time.time()
    for i, t_val in enumerate(t_dense):
        s = mpmath.mpc(0.5, t_val)
        L = build_symmetric(s, N_val, p_list)
        dets_n[i] = np.linalg.det(np.eye(N_val, dtype=complex) - L)
    elapsed = time.time() - t_start

    # Find winding zeros
    wz = find_zeros_by_winding(dets_n, t_dense, window=5)
    matched = set()
    spurious = 0
    for t_z, _, _ in wz:
        dists = [abs(t_z - z) for z in known_zeros]
        best = np.argmin(dists)
        if dists[best] < 0.5:
            matched.add(best)
        else:
            spurious += 1

    # Also check minima
    abs_det = np.abs(dets_n)
    minima = []
    for i in range(1, len(abs_det) - 1):
        if abs_det[i] < abs_det[i - 1] and abs_det[i] < abs_det[i + 1]:
            minima.append((t_dense[i], abs_det[i]))
    minima.sort(key=lambda x: x[1])
    m_matched = set()
    for t_m, _ in minima[:15]:
        dists = [abs(t_m - z) for z in known_zeros]
        best = np.argmin(dists)
        if dists[best] < 0.5:
            m_matched.add(best)

    print(f"\n  N={N_val:>3} ({len(p_list)} primes, {elapsed:.0f}s):")
    print(f"    Winding:  {len(matched)}/{total_possible} zeros, {spurious} spurious")
    print(f"    Minima:   {len(m_matched)}/{total_possible} zeros")

    # Show which zeros found
    found = sorted(matched)
    print(f"    Found: {[f'{known_zeros[i]:.2f}' for i in found]}")


# ============================================================
# TEST 5: Combine forward + backward with relative phase
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: FORWARD + BACKWARD WITH PHASE OFFSET")
print("=" * 70)
print("  Testing L(s,phi) = forward(p^{-s}) + e^{i*phi} * backward(p^{-s})")
print("  phi=0 is symmetric, phi=pi is antisymmetric")

N_phi = 150
primes_phi = get_primes_up_to(N_phi)
t_test_vals = np.linspace(10, 55, 300)

for phi in [0, np.pi/4, np.pi/2, np.pi, -np.pi/4]:
    phase = np.exp(1j * phi)
    dets_phi = np.zeros(len(t_test_vals), dtype=complex)

    for i, t_val in enumerate(t_test_vals):
        s = mpmath.mpc(0.5, t_val)
        L_f = build_forward_only(s, N_phi, primes_phi)
        L_b = build_backward_only(s, N_phi, primes_phi)
        L = L_f + phase * L_b
        dets_phi[i] = np.linalg.det(np.eye(N_phi, dtype=complex) - L)

    wz = find_zeros_by_winding(dets_phi, t_test_vals, window=4)
    matched = set()
    spurious = 0
    for t_z, _, _ in wz:
        dists = [abs(t_z - z) for z in known_zeros]
        best = np.argmin(dists)
        if dists[best] < 0.5:
            matched.add(best)
        else:
            spurious += 1

    precision = len(matched) / (len(matched) + spurious) if (len(matched) + spurious) > 0 else 0
    print(f"  phi={phi:>+6.3f}: {len(matched)}/{total_possible} zeros, "
          f"{spurious} spurious, precision={precision:.0%}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

# Summary of best results
print("""
  KEY FINDINGS:

  1. GAUGE SYMMETRY BREAKING IS ESSENTIAL:
     Hermitian operators (conjugate phases on forward/backward)
     are unitarily equivalent ∀t → eigenvalues t-independent.
     Only non-Hermitian operators can encode zeta zeros.

  2. PHASE WINDING vs MINIMA:
     [Results above show which detection method is more reliable]

  3. N-SCALING:
     [Results show convergence behavior with matrix size]
""")

print(f"Total time: {time.time() - t0:.1f}s")
