"""Probe r(T) asymptotics using Odlyzko's tables and computed zeros.

Strategy:
1. Compute r at intermediate T (50k, 100k, 200k, 500k) using mpmath.zetazero
2. Load Odlyzko tables at T ~ 2.7e11 and T ~ 1.4e20
3. Compute Z at midpoints (Odlyzko T~2.7e11 — test if mpmath handles it)
4. For T~1.4e20: use gap-based proxy if Z computation is infeasible
"""
import numpy as np
from scipy.stats import pearsonr
import mpmath
import time

# ============================================================
# PART 1: Time test for siegelz at different heights
# ============================================================
print("="*70)
print("TIMING TEST: mpmath.siegelz at various T")
print("="*70)

for T_test in [1000, 10000, 100000, 1e6]:
    mpmath.mp.dps = 15
    t_start = time.time()
    val = mpmath.siegelz(T_test)
    elapsed = time.time() - t_start
    print(f"  T = {T_test:.0e}: Z = {float(val):+.6f}, time = {elapsed:.4f}s")

# Try high T
for T_test in [1e8, 1e9]:
    mpmath.mp.dps = 15
    t_start = time.time()
    try:
        val = mpmath.siegelz(T_test)
        elapsed = time.time() - t_start
        print(f"  T = {T_test:.0e}: Z = {float(val):+.6f}, time = {elapsed:.4f}s")
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  T = {T_test:.0e}: FAILED after {elapsed:.2f}s: {e}")

# ============================================================
# PART 2: Compute r at intermediate heights using zetazero
# ============================================================
print(f"\n{'='*70}")
print("r(T) AT INTERMEDIATE HEIGHTS (using mpmath.zetazero)")
print("="*70)

def compute_r_at_height(n_start, n_count, label):
    """Compute peak-gap correlation for zeros n_start to n_start+n_count."""
    t_start = time.time()
    mpmath.mp.dps = 15

    # Get zeros
    zeros = []
    for n in range(n_start, n_start + n_count + 1):
        z = float(mpmath.im(mpmath.zetazero(n)))
        zeros.append(z)
    zeros = np.array(zeros)

    T_mid = np.mean(zeros)
    T_range = zeros[-1] - zeros[0]

    # Gaps and midpoints
    gaps = np.diff(zeros)
    mids = (zeros[:-1] + zeros[1:]) / 2

    # Peaks
    peaks = np.array([abs(float(mpmath.siegelz(m))) for m in mids])

    # Trim edges
    trim = int(0.1 * len(gaps))
    g_core = gaps[trim:-trim]
    p_core = peaks[trim:-trim]

    # Correlation
    r = pearsonr(g_core, p_core)[0]
    g_avg = np.mean(g_core)

    elapsed = time.time() - t_start
    print(f"  {label:>12}: T_mid = {T_mid:.0f}, r = {r:+.4f}, "
          f"g_avg = {g_avg:.4f}, ({elapsed:.1f}s)", flush=True)

    return T_mid, r, g_avg

results = []

# Lower T — fast
for label, n_start, n_count in [
    ("T~500", 100, 200),
    ("T~2000", 600, 200),
    ("T~5000", 2000, 200),
    ("T~10000", 5000, 200),
    ("T~20000", 10000, 200),
    ("T~50000", 28000, 200),
]:
    T_mid, r, g_avg = compute_r_at_height(n_start, n_count, label)
    results.append((T_mid, r, g_avg))

# Higher T — slower but critical
for label, n_start, n_count in [
    ("T~100000", 60000, 200),
    ("T~200000", 130000, 200),
]:
    T_mid, r, g_avg = compute_r_at_height(n_start, n_count, label)
    results.append((T_mid, r, g_avg))


# ============================================================
# PART 3: Odlyzko zeros at T ~ 2.68e11 (10^12-th zero)
# ============================================================
print(f"\n{'='*70}")
print("ODLYZKO ZEROS: T ~ 2.68e11 (zeros near 10^12-th)")
print("="*70)

# Load zeros
T_offset_3 = 267653395647.0
with open("data/odlyzko/zeros3.txt") as f:
    lines = f.readlines()

# Parse: skip header lines (non-numeric)
oz3 = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        val = float(line)
        oz3.append(T_offset_3 + val)
    except ValueError:
        continue

oz3 = np.array(oz3)
print(f"  Loaded {len(oz3)} zeros")
print(f"  T range: [{oz3[0]:.2f}, {oz3[-1]:.2f}]")

# Gaps
gaps_3 = np.diff(oz3)
g_avg_3 = np.mean(gaps_3)
print(f"  Average gap: {g_avg_3:.6f}")
print(f"  Expected gap (2pi/logT): {2*np.pi/np.log(oz3[0]):.6f}")

# Can we compute Z at these heights?
print(f"\n  Attempting Z computation at T ~ 2.68e11...")
mpmath.mp.dps = 15
t_start = time.time()
try:
    mid_test = (oz3[100] + oz3[101]) / 2
    z_val = mpmath.siegelz(mid_test)
    elapsed = time.time() - t_start
    print(f"  SUCCESS: Z({mid_test:.4f}) = {float(z_val):+.6f}, time = {elapsed:.2f}s")

    # If feasible, compute r for a subset
    if elapsed < 5:
        print(f"  Computing r for subset of {min(500, len(oz3))} zeros...")
        N_use = min(500, len(oz3))
        mids_3 = (oz3[:N_use-1] + oz3[1:N_use]) / 2
        peaks_3 = []
        for i, m in enumerate(mids_3):
            peaks_3.append(abs(float(mpmath.siegelz(m))))
            if (i+1) % 100 == 0:
                print(f"    {i+1}/{len(mids_3)} done...", flush=True)

        peaks_3 = np.array(peaks_3)
        gaps_3_sub = np.diff(oz3[:N_use])
        trim = int(0.1 * len(gaps_3_sub))
        r_odl3 = pearsonr(gaps_3_sub[trim:-trim], peaks_3[trim:-trim])[0]
        print(f"  r at T ~ 2.68e11: {r_odl3:+.4f}")
        results.append((np.mean(oz3[:N_use]), r_odl3, g_avg_3))
    else:
        print(f"  Too slow ({elapsed:.1f}s per point) — skipping Z computation")
        print(f"  Gap statistics only:")
        print(f"    gap std: {np.std(gaps_3):.6f}")
        print(f"    gap CV: {np.std(gaps_3)/np.mean(gaps_3):.4f}")

except Exception as e:
    elapsed = time.time() - t_start
    print(f"  FAILED after {elapsed:.2f}s: {e}")


# ============================================================
# PART 4: Odlyzko zeros at T ~ 1.44e20 (10^21-th zero)
# ============================================================
print(f"\n{'='*70}")
print("ODLYZKO ZEROS: T ~ 1.44e20 (zeros near 10^21-th)")
print("="*70)

T_offset_4 = 144176897509546973000.0
with open("data/odlyzko/zeros4.txt") as f:
    lines = f.readlines()

oz4 = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        val = float(line)
        oz4.append(T_offset_4 + val)
    except ValueError:
        continue

oz4 = np.array(oz4)
print(f"  Loaded {len(oz4)} zeros")
print(f"  T range: [{oz4[0]:.2f}, {oz4[-1]:.2f}]")

gaps_4 = np.diff(oz4)
g_avg_4 = np.mean(gaps_4)
print(f"  Average gap: {g_avg_4:.6f}")
print(f"  Expected gap (2pi/logT): {2*np.pi/np.log(oz4[0]):.6f}")

# Z computation at T ~ 1.44e20 is infeasible (RS sum has ~10^9 terms)
# But we can use a PROXY for r:
# Gap-next-gap correlation: Corr(g_k, g_{k+1})
# This is related to r through the structure of Z oscillations
print(f"\n  Z computation infeasible at T ~ 10^20. Using gap-based proxies:")

# Gap autocorrelation
trim4 = int(0.1 * len(gaps_4))
g4c = gaps_4[trim4:-trim4]
r_gap_gap = pearsonr(g4c[:-1], g4c[1:])[0]
print(f"  Gap-gap autocorrelation: {r_gap_gap:+.4f}")

# Normalized gap distribution statistics
g4_norm = g4c / np.mean(g4c)
print(f"  Normalized gap statistics:")
print(f"    mean = {np.mean(g4_norm):.4f}")
print(f"    std = {np.std(g4_norm):.4f}")
print(f"    min = {np.min(g4_norm):.4f}")
print(f"    max = {np.max(g4_norm):.4f}")

# GUE gap distribution has: mean=1, std≈0.4178 (Wigner surmise)
print(f"  GUE reference: std ≈ 0.4178")

# Same for zeros3
g3c = gaps_3[trim4:-trim4] if len(gaps_3) > 2*trim4 else gaps_3
g3_norm = g3c / np.mean(g3c)
r_gap_gap_3 = pearsonr(g3c[:-1], g3c[1:])[0]
print(f"\n  Zeros3 (T~2.68e11):")
print(f"    Gap-gap autocorrelation: {r_gap_gap_3:+.4f}")
print(f"    Normalized gap std: {np.std(g3_norm):.4f}")


# ============================================================
# PART 5: r(T) trajectory
# ============================================================
print(f"\n{'='*70}")
print("r(T) TRAJECTORY")
print("="*70)

results.sort(key=lambda x: x[0])
print(f"\n  {'T':>12} {'r':>8} {'g_avg':>8} {'log(T)':>8}")
print(f"  {'-'*40}")
for T, r, g in results:
    print(f"  {T:>12.0f} {r:>+8.4f} {g:>8.5f} {np.log(T):>8.3f}")

# Fit r vs log(T)
T_arr = np.array([x[0] for x in results])
r_arr = np.array([x[1] for x in results])
logT = np.log(T_arr)

from scipy.optimize import curve_fit

# Model 1: r = a - b*log(T)
def linear_logT(x, a, b):
    return a - b * np.log(x)

# Model 2: r = r_inf + a/log(T)
def asymptotic(x, r_inf, a):
    return r_inf + a / np.log(x)

# Model 3: r = a - b*log(log(T))
def loglog(x, a, b):
    return a - b * np.log(np.log(x))

print(f"\n  Fitting models:")
for name, func in [("r = a - b*logT", linear_logT),
                     ("r = r_inf + a/logT", asymptotic),
                     ("r = a - b*log(logT)", loglog)]:
    try:
        popt, pcov = curve_fit(func, T_arr, r_arr, maxfev=10000)
        residuals = r_arr - func(T_arr, *popt)
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"    {name}: params = {[f'{p:.6f}' for p in popt]}, RMSE = {rmse:.6f}")

        # Extrapolate
        for T_ext in [1e6, 1e8, 1e11, 1e20]:
            r_ext = func(T_ext, *popt)
            print(f"      T = {T_ext:.0e}: r = {r_ext:+.4f}")
    except Exception as e:
        print(f"    {name}: FAILED ({e})")

print(f"\nTotal time: {time.time() - time.time():.0f}s... ", flush=True)
# Fix: compute total time properly
print(f"\nTotal time: {time.time()-t_start:.1f}s")
