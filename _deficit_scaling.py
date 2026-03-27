"""Regression deficit scaling with T.

CRITICAL QUESTION: Does the deficit ratio R = |Z'|/(4(aG+b)) grow with T?

If R stays bounded -> deficit can be proved analytically.
If R ~ log(T) -> deficit eventually fails and approach needs modification.

Strategy: compute zeros in windows at different heights T, measure R.
"""
import numpy as np
from scipy.stats import pearsonr, linregress
import mpmath
mpmath.mp.dps = 25
import time

t0 = time.time()

def hardy_Z(t):
    return float(mpmath.siegelz(t))

def zeta_deriv_at_zero(gamma):
    """Compute |zeta'(1/2 + i*gamma)| at a zero."""
    s = mpmath.mpc(0.5, gamma)
    return float(abs(mpmath.diff(mpmath.zeta, s)))


# ============================================================
# 1. Get zeros in windows at different heights
# ============================================================
# Use mpmath.zetazero(n) which is fast even for large n

# Approximate n for given T: N(T) ~ (T/2pi)*log(T/2pie)
def approx_n_for_T(T):
    return int(T / (2*np.pi) * np.log(T / (2*np.pi*np.e))) + 1

# Windows: 200 zeros each, centered at different heights
windows = [
    ("T~400", 50, 250),          # zeros 50-250, T ~ 150-400
    ("T~800", 200, 400),         # zeros 200-400, T ~ 400-700
    ("T~1500", 500, 700),        # zeros 500-700, T ~ 900-1300
    ("T~3000", 1100, 1300),      # zeros 1100-1300
    ("T~5000", 2000, 2200),      # zeros 2000-2200
    ("T~10000", 4500, 4700),     # zeros 4500-4700
    ("T~20000", 10000, 10200),   # zeros 10000-10200
    ("T~50000", 28000, 28200),   # zeros 28000-28200
]

print("="*80)
print("REGRESSION DEFICIT SCALING WITH T")
print("="*80)
print(f"\nComputing zeros in {len(windows)} windows...\n")

results = []
for label, n_start, n_end in windows:
    t_start = time.time()
    N_w = n_end - n_start

    # Compute zeros
    zeros = np.array([float(mpmath.im(mpmath.zetazero(n))) for n in range(n_start, n_end + 1)])
    T_mid = np.mean(zeros)
    T_lo, T_hi = zeros[0], zeros[-1]

    # Gaps, midpoints, peaks
    gaps = np.diff(zeros)
    mids = (zeros[:-1] + zeros[1:]) / 2
    peaks = np.array([abs(hardy_Z(m)) for m in mids])

    # Trim edges (10% each side)
    trim = int(0.1 * len(gaps))
    g_core = gaps[trim:-trim]
    p_core = peaks[trim:-trim]
    z_core = zeros[trim:-trim-1]  # zeros for the core gaps

    # Peak-gap correlation
    r_val = pearsonr(g_core, p_core)[0]

    # Regression: P = a*g + b
    slope, intercept, _, _, _ = linregress(g_core, p_core)
    a, b = slope, intercept

    # Compute |zeta'| at each core zero and the deficit ratio
    print(f"  {label}: computing |zeta'| at {len(z_core)} zeros (T ~ {T_mid:.0f})...",
          flush=True)

    zeta_primes = []
    deficit_ratios = []
    shape_factors = []

    for i in range(1, len(z_core) - 1):
        gamma = z_core[i]

        # |zeta'(1/2 + i*gamma)|
        zp = zeta_deriv_at_zero(gamma)
        zeta_primes.append(zp)

        # Merged gap: G = g_{i-1} + g_i (gap to left + gap to right of this zero)
        # In the core array, the gap g_core[i] is between z_core[i] and z_core[i+1]
        # We need the gaps on BOTH sides of z_core[i]
        G = g_core[i-1] + g_core[i]  # merged gap around this zero

        # Deficit ratio: R = |zeta'|/4 / (a*G + b)
        predicted = a * G + b
        if predicted > 0:
            R = (zp / 4) / predicted
            deficit_ratios.append(R)

        # Shape factor: c = |Z'(gamma)| * g / P
        # P ~ |Z'| * g / c, so c = |Z'| * g / P
        # Use the gap to the right and the corresponding peak
        g_right = g_core[i]
        P_right = p_core[i]
        if P_right > 0:
            c = zp * g_right / (2 * P_right)  # factor of 2 from g/2 being the half-gap
            shape_factors.append(c)

    zp_arr = np.array(zeta_primes)
    R_arr = np.array(deficit_ratios)
    c_arr = np.array(shape_factors)

    # Average gap
    g_avg = np.mean(g_core)

    elapsed = time.time() - t_start

    res = {
        'label': label, 'T_mid': T_mid, 'T_lo': T_lo, 'T_hi': T_hi,
        'r': r_val, 'a': a, 'b': b, 'g_avg': g_avg,
        'R_max': np.max(R_arr), 'R_mean': np.mean(R_arr), 'R_std': np.std(R_arr),
        'zp_mean': np.mean(zp_arr), 'zp_max': np.max(zp_arr),
        'c_mean': np.mean(c_arr), 'c_std': np.std(c_arr),
        'violations': np.sum(R_arr >= 1.0),
        'elapsed': elapsed,
        'N_w': N_w
    }
    results.append(res)

    print(f"    T_range=[{T_lo:.0f}, {T_hi:.0f}], r={r_val:.4f}, "
          f"R_max={np.max(R_arr):.4f}, violations={np.sum(R_arr >= 1.0)}, "
          f"({elapsed:.1f}s)", flush=True)


# ============================================================
# 2. Summary table
# ============================================================
print(f"\n{'='*80}")
print("SUMMARY: DEFICIT RATIO R = |zeta'|/(4(aG+b)) vs T")
print("="*80)

print(f"\n  {'Window':>10} {'T_mid':>8} {'r':>7} {'a':>8} {'g_avg':>7} "
      f"{'R_max':>7} {'R_mean':>7} {'c_mean':>7} {'Viol':>5}")
print(f"  {'-'*72}")

for res in results:
    print(f"  {res['label']:>10} {res['T_mid']:>8.0f} {res['r']:>+7.3f} "
          f"{res['a']:>8.3f} {res['g_avg']:>7.4f} "
          f"{res['R_max']:>7.4f} {res['R_mean']:>7.4f} {res['c_mean']:>7.3f} "
          f"{res['violations']:>5d}")


# ============================================================
# 3. Scaling analysis: how does R_max scale with T?
# ============================================================
print(f"\n{'='*80}")
print("SCALING ANALYSIS")
print("="*80)

T_vals = np.array([r['T_mid'] for r in results])
R_maxs = np.array([r['R_max'] for r in results])
R_means = np.array([r['R_mean'] for r in results])
c_means = np.array([r['c_mean'] for r in results])
g_avgs = np.array([r['g_avg'] for r in results])
a_vals = np.array([r['a'] for r in results])

# Fit R_max vs log(T)
logT = np.log(T_vals)
if len(logT) > 2:
    sl, ic, _, _, _ = linregress(logT, R_maxs)
    print(f"\n  R_max vs log(T): slope={sl:.4f}, intercept={ic:.4f}")
    print(f"  Model: R_max ~ {sl:.4f} * log(T) + ({ic:.4f})")
    if sl > 0:
        T_fail = np.exp((1.0 - ic) / sl)
        print(f"  R_max = 1 (deficit fails) at T ~ {T_fail:.0e}")
    else:
        print(f"  R_max DECREASING — deficit gets safer with T!")

# Shape factor c vs T
sl_c, ic_c, _, _, _ = linregress(logT, c_means)
print(f"\n  c_mean vs log(T): slope={sl_c:.4f}, intercept={ic_c:.4f}")
print(f"  Shape factor {'GROWS' if sl_c > 0 else 'SHRINKS'} with T")

# Regression slope a vs T
sl_a, ic_a, _, _, _ = linregress(logT, np.log(a_vals))
print(f"\n  log(a) vs log(T): slope={sl_a:.4f}")
print(f"  a ~ T^{sl_a:.3f} or a ~ (logT)^{sl_a * np.mean(logT) / np.mean(np.log(logT)):.2f}")

# Fit a vs (logT)^alpha
from scipy.optimize import curve_fit
def power_log(x, A, alpha):
    return A * np.log(x)**alpha
try:
    popt, _ = curve_fit(power_log, T_vals, a_vals, p0=[0.05, 2.2])
    print(f"  Best fit: a = {popt[0]:.4f} * (logT)^{popt[1]:.3f}")
except:
    pass


# ============================================================
# 4. Theoretical prediction
# ============================================================
print(f"\n{'='*80}")
print("THEORETICAL PREDICTION")
print("="*80)

print(f"""
  If |Z'(gamma)| ~ c * P / (g/2), and P ~ a*g + b ~ a*g, then:
    |Z'| ~ 2c * a
    G ~ 2*g
    aG + b ~ 2ag
    R = |Z'|/(4(aG+b)) ~ 2ca/(8ag) = c/(4g)

  Since g ~ 2pi/log(T), R ~ c*log(T)/(8pi).

  This GROWS logarithmically UNLESS c shrinks to compensate.
""")

# Check: does c/(4*g) match R_mean?
for res in results:
    pred = res['c_mean'] / (4 * res['g_avg'])
    print(f"  {res['label']:>10}: c/(4g) = {pred:.4f}  vs  R_mean = {res['R_mean']:.4f}  "
          f"(ratio = {res['R_mean']/pred:.3f})")


# ============================================================
# 5. The key: does |zeta'|/(aG) stay bounded?
# ============================================================
print(f"\n{'='*80}")
print("KEY RATIO: |zeta'| / (a * G)")
print("="*80)

print(f"\n  This ratio = 4R * (1 + b/(aG)) ~ 4R")
print(f"  If this stays bounded < 4, the deficit holds.\n")

for res in results:
    ratio = 4 * res['R_mean']
    ratio_max = 4 * res['R_max']
    print(f"  {res['label']:>10}: mean = {ratio:.4f}, max = {ratio_max:.4f}")


print(f"\nTotal time: {time.time()-t0:.1f}s")
