"""FORMAL BOUND: Prove |r_exact - r_corrected| < epsilon(T) -> 0.

THE ARGUMENT:
  Z(t) = Z_corr(t) + R_1(t)
  where Z_corr = Z_main + R_0 (the corrected RS sum)
  and R_1(t) = O(T^{-3/4}) is the REMAINING error.

  The zeros of Z differ from zeros of Z_corr by:
    delta_gamma_k = -R_1(gamma_k) / Z'(gamma_k) + O(R_1^2)
  (implicit function theorem)

  The peaks differ by:
    delta_P_k = |Z(m_k)| - |Z_corr(m_k)| = O(|R_1(m_k)|) = O(T^{-3/4})

  The gap perturbation:
    delta_g_k = delta_gamma_{k+1} - delta_gamma_k = O(T^{-3/4} / |Z'|)

  The effect on r:
    |delta_r| <= (1/M) * [|delta_g|/s_g * max(|z_P|) + |delta_P|/s_p * max(|z_g|)]
    where the terms involve standardized quantities.

  Since s_g ~ 1/log(T) and s_p ~ sqrt(log T):
    |delta_g|/s_g ~ T^{-3/4} * log(T) / |Z'|
    |delta_P|/s_p ~ T^{-3/4} / sqrt(log T)

  Both -> 0 as T -> infinity. The rate is at LEAST T^{-3/4} * log(T)^{3/2}.

LET'S VERIFY THIS NUMERICALLY and compute the actual convergence rate.
"""
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import brentq
import mpmath; mpmath.mp.dps = 15
import time

t0 = time.time()

zeta_zeros = np.load("_zeros_500.npy")

def hardy_Z(t): return float(mpmath.siegelz(t))

def theta_approx(t):
    return t/2 * np.log(t/(2*np.pi)) - t/2 - np.pi/8

def Z_main(t):
    N_t = int(np.sqrt(t / (2*np.pi)))
    if N_t < 1: N_t = 1
    th = theta_approx(t)
    return 2 * sum(np.cos(th - t*np.log(n)) / np.sqrt(n)
                   for n in range(1, N_t+1))

def Z_corrected(t):
    N_t = int(np.sqrt(t / (2*np.pi)))
    if N_t < 1: N_t = 1
    main = Z_main(t)
    p = np.sqrt(t / (2*np.pi)) - N_t
    denom = np.cos(2*np.pi*p)
    if abs(denom) < 1e-10: return main
    C0 = np.cos(2*np.pi*(p**2 - p - 1.0/16)) / denom
    R0 = (-1)**(N_t - 1) * (t / (2*np.pi))**(-0.25) * C0
    return main + R0

def find_zeros_fast(f, t_start, t_end, dt=0.015):
    ts = np.arange(t_start, t_end, dt)
    vals = np.array([f(t) for t in ts])
    zeros = []
    for i in range(len(vals)-1):
        if vals[i]*vals[i+1] < 0:
            try: zeros.append(brentq(f, ts[i], ts[i+1]))
            except: pass
    return np.array(zeros)


# ============================================================
# 1. Measure R_1(t) = Z_exact - Z_corrected
# ============================================================
print("="*60)
print("1. REMAINDER R_1(t) = Z_exact - Z_corrected")
print("="*60)

print(f"\n  {'T_range':>12} {'||R_1||_rms':>12} {'||Z||_rms':>10} "
      f"{'R_1/Z':>8} {'T^-3/4':>8} {'ratio':>8}")
print(f"  {'-'*62}")

for t_lo, t_hi in [(50,100), (100,200), (200,300), (300,400),
                    (400,600), (600,811)]:
    ts = np.linspace(t_lo, t_hi, 200)
    R1 = np.array([hardy_Z(t) - Z_corrected(t) for t in ts])
    Zv = np.array([hardy_Z(t) for t in ts])
    R1_rms = np.sqrt(np.mean(R1**2))
    Z_rms = np.sqrt(np.mean(Zv**2))
    T_mid = (t_lo+t_hi)/2
    t34 = T_mid**(-0.75)

    print(f"  {f'{t_lo}-{t_hi}':>12} {R1_rms:>12.6f} {Z_rms:>10.4f} "
          f"{R1_rms/Z_rms:>8.5f} {t34:>8.5f} {R1_rms/t34:>8.4f}")


# ============================================================
# 2. Zero displacement: how far do Z_corr zeros deviate from Z_exact zeros?
# ============================================================
print(f"\n{'='*60}")
print("2. ZERO DISPLACEMENT |gamma_exact - gamma_corr|")
print("="*60)

z_corr_all = find_zeros_fast(Z_corrected, 14, 811, dt=0.012)
print(f"  Found {len(z_corr_all)} corrected zeros vs {len(zeta_zeros)} exact")

# Match corrected zeros to exact zeros (nearest neighbor)
displacements = []
for zc in z_corr_all:
    idx = np.argmin(np.abs(zeta_zeros - zc))
    displacements.append(zeta_zeros[idx] - zc)

disp = np.array(displacements)
print(f"  |displacement| stats:")
print(f"    mean = {np.mean(np.abs(disp)):.6f}")
print(f"    max  = {np.max(np.abs(disp)):.6f}")
print(f"    std  = {np.std(disp):.6f}")

# Displacement vs T
print(f"\n  {'T_range':>12} {'mean|disp|':>12} {'max|disp|':>12} {'T^-3/4':>10}")
print(f"  {'-'*50}")
for t_lo, t_hi in [(14,100), (100,200), (200,400), (400,600), (600,811)]:
    mask = [(t_lo <= zeta_zeros[np.argmin(np.abs(zeta_zeros-zc))] < t_hi)
            for zc in z_corr_all]
    d_w = np.abs(disp[mask])
    if len(d_w) > 0:
        T_mid = (t_lo+t_hi)/2
        print(f"  {f'{t_lo}-{t_hi}':>12} {np.mean(d_w):>12.6f} {np.max(d_w):>12.6f} "
              f"{T_mid**(-0.75):>10.6f}")


# ============================================================
# 3. Effect on r: analytic bound
# ============================================================
print(f"\n{'='*60}")
print("3. ANALYTIC BOUND |r_exact - r_corr|")
print("="*60)

# The Pearson r perturbation from small data perturbations:
# If (g_k, P_k) -> (g_k + dg_k, P_k + dP_k), then
# dr = (1/(M*s_g*s_p)) * sum_k [dg_k*(P_k-Pbar)/s_p + dP_k*(g_k-gbar)/s_g
#       - r*(dg_k*(g_k-gbar)/s_g + dP_k*(P_k-Pbar)/s_p)]  +  O(||d||^2)
#
# Bounding: |dr| <= (1/(M*s_g*s_p)) * M * [max|dg|*s_p + max|dP|*s_g + ...]
#         ~ max|dg|/s_g + max|dP|/s_p  (up to constants and r factors)

# From the windowed analysis, compute the ACTUAL dr vs the PREDICTED bound
print(f"\n  {'window':>10} {'|dr|':>10} {'max|dg|/sg':>12} {'max|dP|/sp':>12} "
      f"{'bound':>10} {'ratio':>8}")
print(f"  {'-'*66}")

for i_lo, i_hi in [(0,100), (100,200), (200,300), (300,400), (400,499)]:
    if i_hi > len(zeta_zeros): continue
    zeros_ex = zeta_zeros[i_lo:i_hi]
    t_lo, t_hi_w = zeros_ex[0]-1, zeros_ex[-1]+1

    # Exact r
    gaps_ex = np.diff(zeros_ex)
    mids_ex = (zeros_ex[:-1]+zeros_ex[1:])/2
    peaks_ex = np.array([abs(hardy_Z(m)) for m in mids_ex])
    nt = int(0.1*len(gaps_ex))
    r_ex = pearsonr(gaps_ex[nt:-nt], peaks_ex[nt:-nt])[0]

    # Corrected zeros in this range
    mask_c = (z_corr_all >= t_lo) & (z_corr_all <= t_hi_w)
    z_corr_w = z_corr_all[mask_c]
    if len(z_corr_w) < 20: continue

    gaps_co = np.diff(z_corr_w)
    mids_co = (z_corr_w[:-1]+z_corr_w[1:])/2
    peaks_co = np.array([abs(Z_corrected(m)) for m in mids_co])
    nt_c = int(0.1*len(gaps_co))
    r_co = pearsonr(gaps_co[nt_c:-nt_c], peaks_co[nt_c:-nt_c])[0]

    dr = abs(r_ex - r_co)

    # Perturbation bound ingredients
    s_g = np.std(gaps_ex[nt:-nt], ddof=1)
    s_p = np.std(peaks_ex[nt:-nt], ddof=1)
    M = len(gaps_ex) - 2*nt

    # Max gap perturbation (from zero displacement)
    T_mid = np.mean(zeros_ex)
    max_dg = 2 * np.max(np.abs(disp))  # conservative
    max_dP = np.sqrt(np.mean([(hardy_Z(m)-Z_corrected(m))**2
                               for m in mids_ex[nt:-nt]]))

    bound = (max_dg/s_g + max_dP/s_p) * (1+abs(r_ex)) / M
    ratio = dr / bound if bound > 0 else 0

    print(f"  {f'{i_lo}-{i_hi}':>10} {dr:>10.6f} {max_dg/s_g:>12.6f} "
          f"{max_dP/s_p:>12.6f} {bound:>10.6f} {ratio:>8.4f}")


# ============================================================
# 4. The convergence rate
# ============================================================
print(f"\n{'='*60}")
print("4. CONVERGENCE RATE: |r_exact - r_corr| vs T")
print("="*60)

# From the windowed data, fit |dr| ~ A * T^alpha
# We expect alpha ~ -3/4 (from R_1 = O(T^{-3/4}))

# Collect (T_mid, |dr|) pairs from sliding windows
T_mids = []; dr_vals = []
for start in range(0, len(zeta_zeros)-80, 40):
    end = min(start+80, len(zeta_zeros))
    zeros_w = zeta_zeros[start:end]
    t_lo_w, t_hi_w = zeros_w[0]-1, zeros_w[-1]+1
    T_mid = np.mean(zeros_w)

    gaps_w = np.diff(zeros_w)
    mids_w = (zeros_w[:-1]+zeros_w[1:])/2
    peaks_w = np.array([abs(hardy_Z(m)) for m in mids_w])
    nt_w = int(0.1*len(gaps_w))
    if len(gaps_w) <= 2*nt_w+5: continue
    r_w = pearsonr(gaps_w[nt_w:-nt_w], peaks_w[nt_w:-nt_w])[0]

    mask_c = (z_corr_all >= t_lo_w) & (z_corr_all <= t_hi_w)
    zc_w = z_corr_all[mask_c]
    if len(zc_w) < 20: continue
    gc_w = np.diff(zc_w); mc_w = (zc_w[:-1]+zc_w[1:])/2
    pc_w = np.array([abs(Z_corrected(m)) for m in mc_w])
    nt_c = int(0.1*len(gc_w))
    if len(gc_w) <= 2*nt_c+5: continue
    r_c = pearsonr(gc_w[nt_c:-nt_c], pc_w[nt_c:-nt_c])[0]

    dr_w = abs(r_w - r_c)
    T_mids.append(T_mid)
    dr_vals.append(dr_w)

T_mids = np.array(T_mids); dr_vals = np.array(dr_vals)

print(f"\n  {'T_mid':>8} {'|dr|':>12} {'T^-3/4':>10} {'|dr|/T^-3/4':>12}")
print(f"  {'-'*46}")
for i in range(len(T_mids)):
    t34 = T_mids[i]**(-0.75)
    print(f"  {T_mids[i]:>8.0f} {dr_vals[i]:>12.6f} {t34:>10.6f} "
          f"{dr_vals[i]/t34:>12.4f}")

# Fit power law
if len(T_mids) > 3:
    from scipy.optimize import curve_fit
    def power_law(T, A, alpha):
        return A * T**alpha
    # Only fit where dr > 0
    mask = dr_vals > 1e-8
    if np.sum(mask) > 3:
        try:
            popt, _ = curve_fit(power_law, T_mids[mask], dr_vals[mask],
                                p0=[1, -0.75], maxfev=5000)
            print(f"\n  Power law fit: |dr| ~ {popt[0]:.4f} * T^({popt[1]:.4f})")
            print(f"  Expected: T^(-0.75)")
        except:
            print("  Power law fit failed")


# ============================================================
# 5. THE BOTTOM LINE: at what T does |dr| < C/N?
# ============================================================
print(f"\n{'='*60}")
print("5. WHEN DOES |dr| < C/N ?")
print("="*60)

C = 1.76  # conservative C >= 2r
print(f"\n  C = {C} (conservative)")
print(f"  Required: |r_exact - r_corr| < C/N")
print(f"\n  {'T':>8} {'N(T)':>8} {'C/N':>10} {'|dr| est':>10} {'PASSES?':>8}")
print(f"  {'-'*48}")

for T_test in [100, 200, 400, 811, 1000, 5000, 10000, 100000]:
    N_test = T_test / (2*np.pi) * np.log(T_test/(2*np.pi))  # approx zero count
    CN = C / max(N_test, 1)
    # Estimate |dr| from power law or from data
    dr_est = max(dr_vals) * (400/T_test)**0.75 if T_test > 100 else max(dr_vals)
    passes = "YES" if dr_est < CN else "no"
    print(f"  {T_test:>8} {N_test:>8.0f} {CN:>10.6f} {dr_est:>10.6f} {passes:>8}")


print(f"\nTotal time: {time.time()-t0:.1f}s")
