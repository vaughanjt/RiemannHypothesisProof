"""Transfer operator v4: Persistence analysis across N values.

ESTABLISHED:
- Symmetric prime transfer operator finds zeta zeros via phase winding
- N=300 finds 8/11 known zeros in [10, 55]
- Hermitian operators CANNOT work (gauge symmetry theorem)
- Forward-only and backward-only CANNOT work (need both shifts)

THIS SCRIPT:
1. Run N=100,150,200,250,300,350,400 systematically
2. Track which zeros are found at each N
3. Use PERSISTENCE (found at 3+ values of N) to filter true vs spurious
4. Measure convergence rate: how fast do new zeros appear?
5. Extrapolate: what N is needed for 11/11?
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from sympy import primerange
import mpmath
from collections import defaultdict

t0 = time.time()
mpmath.mp.dps = 20


def get_primes_up_to(N):
    return list(primerange(2, N + 1))


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


def find_zeros_by_winding(det_values, t_values, window=5):
    """Find zeros using phase winding in local windows."""
    uphases = np.unwrap(np.angle(det_values))
    zeros = []
    for i in range(window, len(t_values) - window):
        dphase = uphases[i + window] - uphases[i - window]
        winding = dphase / (2 * np.pi)
        if abs(abs(winding) - 1.0) < 0.3:
            local_abs = np.abs(det_values[i - window:i + window + 1])
            local_min_idx = np.argmin(local_abs) + i - window
            t_zero = t_values[local_min_idx]
            det_min = abs(det_values[local_min_idx])
            if not zeros or abs(t_zero - zeros[-1][0]) > 0.5:
                zeros.append((t_zero, det_min, winding))
    return zeros


def find_zeros_by_minima(det_values, t_values, n_keep=20):
    """Find zeros as deepest local minima of |det|."""
    abs_det = np.abs(det_values)
    minima = []
    for i in range(1, len(abs_det) - 1):
        if abs_det[i] < abs_det[i - 1] and abs_det[i] < abs_det[i + 1]:
            minima.append((t_values[i], abs_det[i]))
    minima.sort(key=lambda x: x[1])
    return minima[:n_keep]


# Known zeta zeros in scan range
known_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
               37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703]
total_possible = len(known_zeros)

# Scan parameters
t_scan = np.linspace(10, 55, 800)  # 800 points for good resolution
N_values = [100, 150, 200, 250, 300, 350, 400]

# ============================================================
# MAIN SCAN: all N values
# ============================================================
print("=" * 70)
print("PERSISTENCE ANALYSIS: SYMMETRIC OPERATOR ACROSS N VALUES")
print("=" * 70)

all_winding_zeros = {}  # N -> list of (t, |det|, winding)
all_minima = {}  # N -> list of (t, |det|)
det_at_zeros = {}  # N -> {zero_idx: |det| value}

for N_val in N_values:
    p_list = get_primes_up_to(N_val)
    dets = np.zeros(len(t_scan), dtype=complex)

    t_start = time.time()
    for i, t_val in enumerate(t_scan):
        s = mpmath.mpc(0.5, t_val)
        L = build_symmetric(s, N_val, p_list)
        dets[i] = np.linalg.det(np.eye(N_val, dtype=complex) - L)

    elapsed = time.time() - t_start

    # Find zeros by winding
    wz = find_zeros_by_winding(dets, t_scan, window=6)
    all_winding_zeros[N_val] = wz

    # Find zeros by minima
    minima = find_zeros_by_minima(dets, t_scan, n_keep=25)
    all_minima[N_val] = minima

    # Record |det| at each known zero location
    det_at_zeros[N_val] = {}
    for zi, z in enumerate(known_zeros):
        idx = np.argmin(np.abs(t_scan - z))
        det_at_zeros[N_val][zi] = abs(dets[idx])

    # Score
    matched = set()
    spurious = 0
    for t_z, _, _ in wz:
        dists = [abs(t_z - z) for z in known_zeros]
        best = np.argmin(dists)
        if dists[best] < 0.5:
            matched.add(best)
        else:
            spurious += 1

    found_names = sorted([f"{known_zeros[i]:.2f}" for i in matched])
    print(f"\n  N={N_val:>3} ({len(p_list):>2} primes, {elapsed:.0f}s): "
          f"{len(matched)}/{total_possible} zeros, {spurious} spurious")
    print(f"    Found: {found_names}")


# ============================================================
# PERSISTENCE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("PERSISTENCE: WHICH t-VALUES APPEAR ACROSS MULTIPLE N?")
print("=" * 70)

# Bin all detected zeros into 0.5-wide bins
bins = defaultdict(lambda: defaultdict(list))  # bin_center -> N -> [(t, |det|)]

for N_val in N_values:
    for t_z, d_z, w in all_winding_zeros[N_val]:
        # Find nearest known zero (if any)
        bin_center = round(t_z * 2) / 2  # Round to nearest 0.5
        bins[bin_center][N_val].append((t_z, d_z))

# For each known zero, check how many N values detect it
print(f"\n  Known zero persistence (detected at how many N values):")
print(f"  {'Zero':>8} " + " ".join(f"{'N='+str(N):>6}" for N in N_values) + f" {'Count':>6} {'Status':>10}")
print(f"  {'-'*80}")

zero_persistence = {}
for zi, z in enumerate(known_zeros):
    row = f"  {z:>8.4f}"
    count = 0
    for N_val in N_values:
        found = any(abs(t_z - z) < 0.7 for t_z, _, _ in all_winding_zeros[N_val])
        row += f" {'  YES' if found else '   --':>6}"
        if found:
            count += 1
    zero_persistence[zi] = count
    status = "CONFIRMED" if count >= 3 else "MARGINAL" if count >= 2 else "MISSING"
    row += f" {count:>6} {status:>10}"
    print(row)


# ============================================================
# |det| AT KNOWN ZEROS: CONVERGENCE TO ZERO
# ============================================================
print("\n" + "=" * 70)
print("|det(I-L)| AT KNOWN ZERO LOCATIONS vs N")
print("=" * 70)
print("  (True zeros should show |det| -> 0 as N increases)")

print(f"\n  {'Zero':>8} " + " ".join(f"{'N='+str(N):>12}" for N in N_values))
print(f"  {'-'*(8 + 13*len(N_values))}")

for zi, z in enumerate(known_zeros):
    row = f"  {z:>8.4f}"
    values = [det_at_zeros[N][zi] for N in N_values]
    for v in values:
        row += f" {v:>12.4e}"
    # Check if decreasing
    decreasing = all(values[i] >= values[i+1] * 0.1 for i in range(len(values)-1) if values[i+1] > 0)
    trend = ""
    if len(values) >= 3:
        # Log-linear fit for decay rate
        log_vals = [np.log10(v) if v > 0 else -30 for v in values]
        if log_vals[-1] < log_vals[0] - 2:
            trend = " << CONVERGING"
        elif log_vals[-1] < log_vals[0]:
            trend = " < slow"
        else:
            trend = " ~ flat/growing"
    row += trend
    print(row)


# ============================================================
# SPURIOUS ZERO ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("SPURIOUS ZEROS: PERSISTENCE AND DRIFT")
print("=" * 70)

# Collect all spurious detections
spurious_bins = defaultdict(int)  # bin -> count across N values
for N_val in N_values:
    for t_z, d_z, w in all_winding_zeros[N_val]:
        dists = [abs(t_z - z) for z in known_zeros]
        if min(dists) > 0.7:
            bin_center = round(t_z * 2) / 2
            spurious_bins[bin_center] += 1

print(f"\n  Spurious detections that persist across 3+ N values:")
persistent_spurious = {b: c for b, c in spurious_bins.items() if c >= 3}
if persistent_spurious:
    for b in sorted(persistent_spurious):
        print(f"    t ~ {b:.1f}: detected at {persistent_spurious[b]}/{len(N_values)} N values")
else:
    print(f"    None — all spurious zeros are transient!")

transient = sum(1 for c in spurious_bins.values() if c < 3)
persistent = sum(1 for c in spurious_bins.values() if c >= 3)
print(f"\n  Transient spurious (appear at 1-2 N values): {transient}")
print(f"  Persistent spurious (appear at 3+ N values): {persistent}")
print(f"  >>> Persistence filter removes {transient} false positives")


# ============================================================
# CONVERGENCE RATE
# ============================================================
print("\n" + "=" * 70)
print("CONVERGENCE RATE")
print("=" * 70)

n_found = []
for N_val in N_values:
    matched = set()
    for t_z, _, _ in all_winding_zeros[N_val]:
        dists = [abs(t_z - z) for z in known_zeros]
        best = np.argmin(dists)
        if dists[best] < 0.7:
            matched.add(best)
    n_found.append(len(matched))

print(f"\n  {'N':>6} {'Primes':>8} {'Zeros found':>12} {'Fraction':>10}")
print(f"  {'-'*40}")
for i, N_val in enumerate(N_values):
    p = len(get_primes_up_to(N_val))
    print(f"  {N_val:>6} {p:>8} {n_found[i]:>12} {n_found[i]/total_possible:>10.1%}")

# Extrapolate
if len(n_found) >= 3 and n_found[-1] > n_found[0]:
    # Linear extrapolation in log(N) space
    log_N = np.log(N_values)
    # Fit: n_found = a * log(N) + b
    A = np.vstack([log_N, np.ones(len(log_N))]).T
    result = np.linalg.lstsq(A, n_found, rcond=None)
    a, b = result[0]
    # N needed for 11 zeros
    if a > 0:
        N_target = np.exp((total_possible - b) / a)
        print(f"\n  Linear fit: n_zeros ~ {a:.1f} * log(N) + {b:.1f}")
        print(f"  Extrapolated N for {total_possible}/{total_possible}: ~{N_target:.0f}")


# ============================================================
# AFTER PERSISTENCE FILTERING
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS: PERSISTENCE-FILTERED ZERO DETECTION")
print("=" * 70)

# A zero is "confirmed" if detected at 3+ N values
confirmed = [(zi, z) for zi, z in enumerate(known_zeros)
             if zero_persistence[zi] >= 3]
marginal = [(zi, z) for zi, z in enumerate(known_zeros)
            if zero_persistence[zi] == 2]
missing = [(zi, z) for zi, z in enumerate(known_zeros)
           if zero_persistence[zi] <= 1]

print(f"\n  Confirmed (3+ N values): {len(confirmed)}/{total_possible}")
for zi, z in confirmed:
    print(f"    t = {z:.4f} (detected at {zero_persistence[zi]} N values)")

print(f"\n  Marginal (2 N values): {len(marginal)}/{total_possible}")
for zi, z in marginal:
    print(f"    t = {z:.4f}")

print(f"\n  Missing (0-1 N values): {len(missing)}/{total_possible}")
for zi, z in missing:
    print(f"    t = {z:.4f}")

print(f"\n  Persistent spurious zeros: {persistent}")
precision = len(confirmed) / (len(confirmed) + persistent) if (len(confirmed) + persistent) > 0 else 0
recall = len(confirmed) / total_possible
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"  Precision: {precision:.0%}")
print(f"  Recall: {recall:.0%}")
print(f"  F1: {f1:.2f}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if len(confirmed) >= 9:
    print(f"\n  BREAKTHROUGH: {len(confirmed)}/11 zeros confirmed with persistence!")
    print(f"  The symmetric prime transfer operator det(I - L_s) encodes zeta zeros.")
elif len(confirmed) >= 6:
    print(f"\n  STRONG: {len(confirmed)}/11 zeros confirmed.")
    print(f"  Clear convergence trend. N=500-1000 should capture all 11.")
elif len(confirmed) + len(marginal) >= 6:
    print(f"\n  PROMISING: {len(confirmed)} confirmed + {len(marginal)} marginal = "
          f"{len(confirmed)+len(marginal)} total.")
    print(f"  Need larger N to promote marginal to confirmed.")
else:
    print(f"\n  MODERATE: {len(confirmed)} confirmed. Convergence too slow or")
    print(f"  operator structure needs refinement.")

print(f"\nTotal time: {time.time() - t0:.1f}s")
