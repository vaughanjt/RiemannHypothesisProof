"""CLOSING GAP 2 (fast version): Use cached zeros, avoid redundant mpmath.

Key fix: use _zeros_500.npy for exact Z zeros, only compute Z_main zeros
with numpy (fast). Only call mpmath for |Z| at midpoints (unavoidable but limited).
"""
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import brentq
import mpmath; mpmath.mp.dps = 15
import time

t0 = time.time()

# Cached exact zeros
zeta_zeros = np.load("_zeros_500.npy")
N_total = len(zeta_zeros)

def hardy_Z(t):
    return float(mpmath.siegelz(t))

def theta_approx(t):
    return t/2 * np.log(t/(2*np.pi)) - t/2 - np.pi/8

def Z_main(t, N_terms=None):
    if N_terms is None:
        N_terms = int(np.sqrt(t / (2*np.pi)))
    if N_terms < 1: N_terms = 1
    th = theta_approx(t)
    return 2 * sum(np.cos(th - t*np.log(n)) / np.sqrt(n)
                   for n in range(1, N_terms+1))

def Z_corrected(t):
    N_t = int(np.sqrt(t / (2*np.pi)))
    if N_t < 1: N_t = 1
    main = Z_main(t, N_t)
    p = np.sqrt(t / (2*np.pi)) - N_t
    denom = np.cos(2*np.pi*p)
    if abs(denom) < 1e-10: return main
    C0 = np.cos(2*np.pi*(p**2 - p - 1.0/16)) / denom
    R0 = (-1)**(N_t - 1) * (t / (2*np.pi))**(-0.25) * C0
    return main + R0

def find_zeros_fast(f, t_start, t_end, dt=0.02):
    """Fast zero-finding for numpy-based functions."""
    ts = np.arange(t_start, t_end, dt)
    vals = np.array([f(t) for t in ts])
    zeros = []
    for i in range(len(vals)-1):
        if vals[i]*vals[i+1] < 0:
            try: zeros.append(brentq(f, ts[i], ts[i+1]))
            except: pass
    return np.array(zeros)

def compute_r_from_zeros(zeros, f_peak, trim_frac=0.1):
    """Compute r using given zeros and a function for peak evaluation."""
    if len(zeros) < 20: return 0., 0
    gaps = np.diff(zeros)
    mids = (zeros[:-1]+zeros[1:])/2
    peaks = np.array([abs(f_peak(m)) for m in mids])
    nt = int(trim_frac * len(gaps))
    if nt > 0 and len(gaps) > 2*nt:
        return pearsonr(gaps[nt:-nt], peaks[nt:-nt])[0], len(gaps)-2*nt
    return pearsonr(gaps, peaks)[0], len(gaps)


# ============================================================
# TEST 1: r_full vs r_main in windows (using cached zeros for full)
# ============================================================
print("="*60)
print("TEST 1: r_full vs r_main vs r_corrected by window")
print("="*60)

# Precompute |Z| at midpoints of cached zeros (expensive but one-time)
print("  Precomputing |Z| at 499 midpoints...", flush=True)
mids_all = (zeta_zeros[:-1] + zeta_zeros[1:])/2
peaks_Z_all = np.array([abs(hardy_Z(m)) for m in mids_all])
print(f"  Done. ({time.time()-t0:.1f}s)", flush=True)

print(f"\n  {'window':>10} {'T_mid':>6} {'r_full':>8} {'r_main':>8} {'r_corr':>8} "
      f"{'gap_m':>8} {'gap_c':>8} {'N(T)':>5}")
print(f"  {'-'*62}")

for i_lo, i_hi in [(0,80), (40,120), (80,160), (120,200),
                    (160,240), (200,280), (240,320), (320,400), (400,499)]:
    if i_hi > N_total: i_hi = N_total
    zeros_w = zeta_zeros[i_lo:i_hi]
    t_lo, t_hi = zeros_w[0]-1, zeros_w[-1]+1
    T_mid = np.mean(zeros_w)
    N_T = int(np.sqrt(T_mid / (2*np.pi)))

    # r_full from cached data
    gaps_w = np.diff(zeros_w)
    peaks_w = peaks_Z_all[i_lo:i_hi-1]
    nt_w = int(0.1*len(gaps_w))
    if len(gaps_w) <= 2*nt_w + 5: continue
    r_full = pearsonr(gaps_w[nt_w:-nt_w], peaks_w[nt_w:-nt_w])[0]

    # r_main: find zeros of Z_main in this range
    z_main = find_zeros_fast(Z_main, t_lo, t_hi, dt=0.015)
    if len(z_main) < 20: continue
    g_main = np.diff(z_main)
    m_main = (z_main[:-1]+z_main[1:])/2
    p_main = np.array([abs(Z_main(m)) for m in m_main])
    nt_m = int(0.1*len(g_main))
    r_main = pearsonr(g_main[nt_m:-nt_m], p_main[nt_m:-nt_m])[0] if len(g_main)>2*nt_m else 0

    # r_corrected: Z_main + R0
    z_corr = find_zeros_fast(Z_corrected, t_lo, t_hi, dt=0.015)
    if len(z_corr) < 20:
        r_corr = 0
    else:
        g_corr = np.diff(z_corr)
        m_corr = (z_corr[:-1]+z_corr[1:])/2
        p_corr = np.array([abs(Z_corrected(m)) for m in m_corr])
        nt_c = int(0.1*len(g_corr))
        r_corr = pearsonr(g_corr[nt_c:-nt_c], p_corr[nt_c:-nt_c])[0] if len(g_corr)>2*nt_c else 0

    gap_m = r_full - r_main
    gap_c = r_full - r_corr

    print(f"  {f'{i_lo}-{i_hi}':>10} {T_mid:>6.0f} {r_full:>+8.4f} {r_main:>+8.4f} "
          f"{r_corr:>+8.4f} {gap_m:>+8.4f} {gap_c:>+8.4f} {N_T:>5}")


# ============================================================
# TEST 2: Remainder size vs T
# ============================================================
print(f"\n{'='*60}")
print("TEST 2: REMAINDER |R(t)| vs T")
print("="*60)

print(f"\n  {'T_range':>12} {'||R||_rms':>10} {'||Z||_rms':>10} {'R/Z':>8} {'T^-1/4':>8}")
print(f"  {'-'*52}")

for t_lo, t_hi in [(50,100), (100,200), (200,300), (300,400),
                    (400,500), (500,600), (600,700), (700,811)]:
    ts = np.linspace(t_lo, t_hi, 200)
    R_vals = np.array([hardy_Z(t) - Z_main(t) for t in ts])
    Z_vals = np.array([hardy_Z(t) for t in ts])
    R_rms = np.sqrt(np.mean(R_vals**2))
    Z_rms = np.sqrt(np.mean(Z_vals**2))
    T_mid = (t_lo+t_hi)/2

    print(f"  {f'{t_lo}-{t_hi}':>12} {R_rms:>10.4f} {Z_rms:>10.4f} "
          f"{R_rms/Z_rms:>8.4f} {T_mid**(-0.25):>8.4f}")


# ============================================================
# TEST 3: Does adding R0 correction close the gap?
# ============================================================
print(f"\n{'='*60}")
print("TEST 3: FULL RANGE r — main vs corrected vs exact")
print("="*60)

# Full range comparison
for label, f_zeros, f_peaks in [
    ("Z_main", lambda: find_zeros_fast(Z_main, 14, 811, dt=0.015), Z_main),
    ("Z_corrected", lambda: find_zeros_fast(Z_corrected, 14, 811, dt=0.015), Z_corrected),
]:
    zeros = f_zeros()
    r_val, n = compute_r_from_zeros(zeros, f_peaks)
    print(f"  {label:>15}: r = {r_val:+.4f} ({len(zeros)} zeros)")

# Exact from cache
gaps_full = np.diff(zeta_zeros)
nt_f = int(0.1*len(gaps_full))
r_exact = pearsonr(gaps_full[nt_f:-nt_f], peaks_Z_all[nt_f:-nt_f])[0]
print(f"  {'Z_exact':>15}: r = {r_exact:+.4f} ({N_total} zeros)")


# ============================================================
# VERDICT
# ============================================================
print(f"\n{'='*60}")
print("VERDICT: GAP 2 CONVERGENCE")
print("="*60)

print(f"""
  The gap between r_main and r_full:
  - Measures whether the RS remainder affects the correlation.
  - If the gap SHRINKS with T, then r_main -> r_full at large T.
  - The remainder R(t) = O(T^{{-1/4}}) vanishes, so the gap must close.

  Adding the first correction term R_0 should recover most of the gap.

  For the density bound: if r_main can be computed analytically (or bounded),
  and |r_full - r_main| < epsilon(T) -> 0, then r_0 is known to arbitrary
  precision at large T, closing Gap 2.
""")

print(f"Total time: {time.time()-t0:.1f}s")
