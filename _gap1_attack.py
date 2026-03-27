"""Gap 1 Attack: Prove E[|zeta'|/4] < P_bar analytically.

Strategy:
1. Measure the actual distributions of |zeta'| and P at zeros
2. Check lognormal fit (RMT prediction)
3. Determine the scaling of E[|zeta'|] and P_bar with T
4. Identify what known results (Gonek, Selberg) suffice for a proof

The key: if |zeta'| is lognormal with variance ~ log log T,
then E[|zeta'|] << sqrt(E[|zeta'|^2]) and the bound might hold.
"""
import numpy as np
from scipy.stats import pearsonr, lognorm, norm, kstest
from scipy.optimize import curve_fit
import mpmath
mpmath.mp.dps = 20
import time

t0 = time.time()

def hardy_Z(t):
    return float(mpmath.siegelz(t))

def zeta_deriv(gamma):
    s = mpmath.mpc(0.5, gamma)
    return float(abs(mpmath.diff(mpmath.zeta, s)))

# ============================================================
# 1. Compute |zeta'|, P, g at multiple heights
# ============================================================
print("="*70)
print("GAP 1 ATTACK: Distribution of |zeta'| and peaks")
print("="*70)

windows = [
    ("T~400", 50, 250),
    ("T~2000", 600, 800),
    ("T~5000", 2000, 2200),
    ("T~10000", 4500, 4700),
    ("T~20000", 10000, 10200),
    ("T~50000", 28000, 28200),
]

all_results = []

for label, n_start, n_end in windows:
    ts = time.time()

    # Get zeros
    zeros = np.array([float(mpmath.im(mpmath.zetazero(n)))
                       for n in range(n_start, n_end + 1)])
    T_mid = np.mean(zeros)

    # Gaps, peaks
    gaps = np.diff(zeros)
    mids = (zeros[:-1] + zeros[1:]) / 2
    peaks = np.array([abs(hardy_Z(m)) for m in mids])

    # Trim edges
    trim = int(0.1 * len(gaps))
    g = gaps[trim:-trim]
    P = peaks[trim:-trim]
    z = zeros[trim:-trim-1]

    # |zeta'| at each zero
    zp = np.array([zeta_deriv(gamma) for gamma in z])

    # Merged gaps
    G = g[:-1] + g[1:]  # for interior zeros
    zp_int = zp[1:-1]   # interior zeros

    g_bar = np.mean(g)
    P_bar = np.mean(P)
    s_P = np.std(P, ddof=1)

    # The KEY quantities
    phantom_mean = np.mean(zp) / 4      # E[|zeta'|/4]
    phantom_median = np.median(zp) / 4   # Median[|zeta'|/4]

    # Gonek check: E[|zeta'|^2] vs (1/3)(log T)^2
    logT = np.log(T_mid)
    zp2_mean = np.mean(zp**2)
    gonek_pred = (1/3) * logT**2
    # Note: Gonek's formula has additional factors, this is the leading term

    # Lognormal fit for |zeta'|
    log_zp = np.log(zp)
    mu_ln = np.mean(log_zp)
    sigma_ln = np.std(log_zp, ddof=1)

    # Under lognormal: E[X] = exp(mu + sigma^2/2)
    E_zp_lognormal = np.exp(mu_ln + sigma_ln**2/2)
    # E[X^2] = exp(2*mu + 2*sigma^2)
    E_zp2_lognormal = np.exp(2*mu_ln + 2*sigma_ln**2)

    # K-S test for lognormality
    ks_stat, ks_p = kstest(log_zp, 'norm', args=(mu_ln, sigma_ln))

    # P_bar vs phantom_mean
    ratio = phantom_mean / P_bar
    margin = P_bar - phantom_mean

    print(f"\n{'='*60}")
    print(f"  {label} (T = {T_mid:.0f}, log T = {logT:.2f})")
    print(f"{'='*60}")
    print(f"  Peaks:   P_bar = {P_bar:.4f}, s_P = {s_P:.4f}")
    print(f"  Gaps:    g_bar = {g_bar:.4f}")
    print(f"  |zeta'|: mean = {np.mean(zp):.4f}, median = {np.median(zp):.4f}")
    print(f"           std  = {np.std(zp):.4f}")
    print(f"")
    print(f"  KEY RATIO: E[|zeta'|/4] / P_bar = {ratio:.4f}")
    print(f"  Margin: P_bar - E[|zeta'|/4] = {margin:.4f}")
    print(f"")
    print(f"  Lognormal fit for |zeta'|:")
    print(f"    mu = {mu_ln:.4f}, sigma = {sigma_ln:.4f}")
    print(f"    E[|zeta'|] from lognormal: {E_zp_lognormal:.4f} vs actual: {np.mean(zp):.4f}")
    print(f"    sigma^2 = {sigma_ln**2:.4f}")
    print(f"    K-S test: stat = {ks_stat:.4f}, p = {ks_p:.4f}")
    print(f"")
    print(f"  Gonek check: E[|zeta'|^2] = {zp2_mean:.4f}")
    print(f"    (1/3)(log T)^2 = {gonek_pred:.4f}")
    print(f"    Ratio: {zp2_mean/gonek_pred:.4f}")
    print(f"")
    print(f"  Cauchy-Schwarz: E[|zeta'|] <= sqrt(E[|zeta'|^2]) = {np.sqrt(zp2_mean):.4f}")
    print(f"    Actual E[|zeta'|] / CS bound = {np.mean(zp)/np.sqrt(zp2_mean):.4f}")
    print(f"    (< 1 always, but how much < 1?)")

    all_results.append({
        'label': label, 'T': T_mid, 'logT': logT,
        'P_bar': P_bar, 'g_bar': g_bar,
        'zp_mean': np.mean(zp), 'zp_median': np.median(zp),
        'zp2_mean': zp2_mean,
        'phantom_mean': phantom_mean, 'ratio': ratio, 'margin': margin,
        'mu_ln': mu_ln, 'sigma_ln': sigma_ln, 'sigma2_ln': sigma_ln**2,
        'ks_p': ks_p,
        'cs_ratio': np.mean(zp)/np.sqrt(zp2_mean),
    })

    elapsed = time.time() - ts
    print(f"  ({elapsed:.1f}s)", flush=True)


# ============================================================
# 2. Scaling analysis
# ============================================================
print(f"\n{'='*70}")
print("SCALING ANALYSIS")
print("="*70)

T_arr = np.array([r['T'] for r in all_results])
logT_arr = np.log(T_arr)
loglogT_arr = np.log(logT_arr)

# E[|zeta'|] vs T
zp_means = np.array([r['zp_mean'] for r in all_results])
P_bars = np.array([r['P_bar'] for r in all_results])
phantoms = np.array([r['phantom_mean'] for r in all_results])
sigmas2 = np.array([r['sigma2_ln'] for r in all_results])
ratios = np.array([r['ratio'] for r in all_results])
cs_ratios = np.array([r['cs_ratio'] for r in all_results])

print(f"\n  {'T':>8} {'logT':>6} {'E|zp|':>8} {'P_bar':>8} {'E|zp|/4':>8} {'ratio':>7} {'sigma^2':>8} {'CS_eff':>7}")
print(f"  {'-'*64}")
for r in all_results:
    print(f"  {r['T']:>8.0f} {r['logT']:>6.2f} {r['zp_mean']:>8.3f} {r['P_bar']:>8.3f} "
          f"{r['phantom_mean']:>8.3f} {r['ratio']:>7.4f} {r['sigma2_ln']:>8.4f} {r['cs_ratio']:>7.4f}")

# Fit E[|zeta'|] vs log(T)
print(f"\n  Scaling fits for E[|zeta'|]:")

def power_logT(T, a, alpha):
    return a * np.log(T)**alpha

try:
    popt, _ = curve_fit(power_logT, T_arr, zp_means, p0=[1, 1])
    print(f"    E[|zeta'|] ~ {popt[0]:.4f} * (logT)^{popt[1]:.4f}")
except: pass

# Fit P_bar vs log(T)
try:
    popt_P, _ = curve_fit(power_logT, T_arr, P_bars, p0=[1, 0.5])
    print(f"    P_bar ~ {popt_P[0]:.4f} * (logT)^{popt_P[1]:.4f}")
except: pass

# Fit sigma^2 vs log(log(T))
from scipy.stats import linregress
sl, ic, _, _, _ = linregress(loglogT_arr, sigmas2)
print(f"\n    sigma^2 vs log(log T): slope = {sl:.4f}, intercept = {ic:.4f}")
print(f"    (RMT predicts sigma^2 ~ c * log log T)")

# Fit ratio E[|zeta'|/4]/P_bar vs log(T)
sl_r, ic_r, _, _, _ = linregress(logT_arr, ratios)
print(f"\n    Ratio E[|zp|/4]/P_bar vs log(T):")
print(f"    slope = {sl_r:.6f}, intercept = {ic_r:.4f}")
if sl_r > 0:
    T_fail = np.exp((1.0 - ic_r) / sl_r)
    print(f"    Ratio = 1 (mean influence = 0) at T ~ {T_fail:.2e}")
else:
    print(f"    Ratio DECREASING — Gap 1 gets easier with T!")

# CS efficiency (how far below Cauchy-Schwarz is the actual mean)
sl_cs, ic_cs, _, _, _ = linregress(logT_arr, cs_ratios)
print(f"\n    Cauchy-Schwarz efficiency E[|zp|]/sqrt(E[|zp|^2]):")
print(f"    slope vs logT = {sl_cs:.6f}")
print(f"    (Decreasing means distribution spreads -> E[X]/sqrt(E[X^2]) shrinks)")

# ============================================================
# 3. The proof sketch
# ============================================================
print(f"\n{'='*70}")
print("PROOF SKETCH FOR GAP 1")
print("="*70)

print(f"""
GOAL: Prove E[|zeta'(1/2+ig)|/4] < P_bar(T) for all T.

KNOWN:
  1. Gonek (1984, under RH): E[|zeta'|^2] ~ (1/3)(logT)^2
  2. |zeta'| at zeros is approximately lognormal with sigma^2 ~ c*log(logT)
  3. For lognormal: E[X] = exp(mu + sigma^2/2)
                    E[X^2] = exp(2mu + 2sigma^2)
     So: E[X] = sqrt(E[X^2]) * exp(-sigma^2/2)
         E[|zeta'|] ~ sqrt((1/3)(logT)^2) * exp(-c*log(logT)/2)
                     = (logT)/sqrt(3) * (logT)^(-c/2)
                     = (logT)^(1-c/2) / sqrt(3)

  If c > 0 (which we measure), then E[|zeta'|] grows SLOWER than logT.
  Specifically: E[|zeta'|] ~ (logT)^(1-c/2) / sqrt(3)

NEEDED:
  P_bar(T) > E[|zeta'|]/4 = (logT)^(1-c/2) / (4*sqrt(3))

  P_bar is the mean |Z(midpoint)| which scales as (logT)^alpha for some alpha.

THE KEY COMPARISON:
  (logT)^(1-c/2) / (4*sqrt(3)) vs (logT)^alpha * const

  This holds for all T if alpha > 1-c/2, OR if alpha = 1-c/2 and the constant
  on the P_bar side is larger than 1/(4*sqrt(3)).
""")

# Compute the effective exponent
if len(all_results) > 2:
    c_est = np.mean(sigmas2) / np.mean(loglogT_arr)  # sigma^2 / log(log T)
    print(f"  From data: sigma^2 / log(logT) ~ {c_est:.4f}")
    print(f"  -> E[|zeta'|] ~ (logT)^(1 - {c_est/2:.4f}) = (logT)^{1-c_est/2:.4f}")
    print(f"  -> E[|zeta'|/4] ~ (logT)^{1-c_est/2:.4f} / {4*np.sqrt(3):.4f}")

    # P_bar exponent
    try:
        print(f"  -> P_bar ~ (logT)^{popt_P[1]:.4f} * {popt_P[0]:.4f}")
        print(f"")
        if popt_P[1] > 1 - c_est/2:
            print(f"  *** P_bar grows FASTER than E[|zeta'|/4] ***")
            print(f"  *** Gap 1 holds for all sufficiently large T ***")
            print(f"  Exponent comparison: {popt_P[1]:.4f} > {1-c_est/2:.4f}")
        else:
            print(f"  P_bar exponent ({popt_P[1]:.4f}) <= zeta' exponent ({1-c_est/2:.4f})")
            print(f"  Need to compare constants.")
    except:
        pass

print(f"\nTotal time: {time.time()-t0:.1f}s")
