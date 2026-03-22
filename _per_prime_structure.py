"""Per-prime amplitude structure: what determines C_p beyond log(p)/p^alpha?

The smooth law captures 84% of the oscillatory signal. The remaining 16%
lives in per-prime deviations. Are these deviations:
  (a) Correlated with known number-theoretic quantities?
  (b) Stable across different observation heights T?
  (c) Predictable from the explicit formula or form factor?

If (a) or (c): we can build a better amplitude model.
If (b): the fluctuations are intrinsic to the primes (not T-dependent).
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.stats import spearmanr, pearsonr
from sympy import primerange, isprime, factorint, mobius
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

MAX_LAG = 400
T_BASE = 267653395647.0
LOG_T = np.log(T_BASE / (2 * np.pi))
k_arr = np.arange(1, MAX_LAG + 1, dtype=float)
ALPHA = 0.787  # best m=1 alpha from harmonic analysis

# ============================================================
# SETUP
# ============================================================
t0 = time.time()
print('Loading data + baseline...')

def gue_eigs(n, rng):
    d = rng.standard_normal(n)
    e = np.sqrt(rng.chisquare(2 * np.arange(n - 1, 0, -1)) / 2)
    return eigvalsh_tridiagonal(d, e) / np.sqrt(n)

rng_bl = np.random.default_rng(42)
bl_acfs = []
for _ in range(100):
    eigs = gue_eigs(1200, rng_bl)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > MAX_LAG + 10:
        bl_acfs.append(spacing_autocorrelation(sp, MAX_LAG))
baseline = np.mean(bl_acfs, axis=0)[1:MAX_LAG + 1]

zeros_raw = []
with open('data/odlyzko/zeros3.txt') as f:
    for line in f:
        try: zeros_raw.append(float(line.strip()))
        except ValueError: pass
zeros_arr = np.array(zeros_raw)
density = LOG_T / (2 * np.pi)
sp_real = np.diff(zeros_arr) * density
sp_real /= np.mean(sp_real)
N_real = len(sp_real)
se = 1.0 / np.sqrt(N_real)
acf_real = spacing_autocorrelation(sp_real, MAX_LAG)[1:MAX_LAG + 1]
excess = acf_real - baseline
ss_tot = np.sum(excess ** 2)
print(f'  Odlyzko: {N_real} spacings, {time.time()-t0:.1f}s')

# ============================================================
# STEP 1: Extract per-prime C_p (first harmonic, cos+sin)
# ============================================================
print('\nStep 1: Extracting per-prime amplitudes...')

# Use only primes with well-separated frequencies (f < 0.25 for safety)
primes_list = list(primerange(2, 300))
freqs = [np.log(p) / LOG_T for p in primes_list]
# Keep only primes where frequency < 0.25 (well within Nyquist)
mask = [f < 0.25 for f in freqs]
primes = [p for p, m in zip(primes_list, mask) if m]
freqs = [f for f, m in zip(freqs, mask) if m]
n_primes = len(primes)

# Build per-prime cos+sin design matrix
short_3 = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]
cols = []
for f in freqs:
    cols.append(np.cos(2 * np.pi * k_arr * f))
    cols.append(np.sin(2 * np.pi * k_arr * f))
cols.extend(short_3)
X = np.column_stack(cols)
amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)

# Extract amplitude and phase
C_p = np.zeros(n_primes)
phi_p = np.zeros(n_primes)
A_cos = np.zeros(n_primes)
A_sin = np.zeros(n_primes)
for i in range(n_primes):
    a_c = amps[2 * i]
    a_s = amps[2 * i + 1]
    C_p[i] = np.sqrt(a_c ** 2 + a_s ** 2)
    phi_p[i] = np.arctan2(-a_s, a_c)
    A_cos[i] = a_c
    A_sin[i] = a_s

# Smooth law prediction
log_p = np.array([np.log(p) for p in primes])
C_smooth = log_p / np.array([p ** ALPHA for p in primes])
# Fit overall scale
scale = np.dot(C_smooth, A_cos) / np.dot(C_smooth, C_smooth)
C_pred = scale * C_smooth

# Residuals (deviations from smooth law)
# Use A_cos directly since phi≈0 for well-resolved primes
delta_C = A_cos - C_pred

print(f'  {n_primes} primes (f < 0.25), scale={scale:.5f}')
print(f'  Smooth law R2 (on A_cos): {1 - np.sum(delta_C**2) / np.sum((A_cos - np.mean(A_cos))**2):.4f}')

# ============================================================
# STEP 2: Number-theoretic correlates of delta_C
# ============================================================
print('\n' + '=' * 70)
print('CORRELATIONS: delta_C vs number-theoretic quantities')
print('=' * 70)

primes_arr = np.array(primes)

# Compute various prime-theoretic features
features = {}

# 1. Prime gaps
gaps = np.diff(primes_arr)
# For each prime, use the gap BEFORE it (except p=2)
gap_before = np.zeros(n_primes)
gap_before[0] = 1  # p=2 has no predecessor
for i in range(1, n_primes):
    gap_before[i] = primes[i] - primes[i - 1]
features['gap_before'] = gap_before

# Gap after
gap_after = np.zeros(n_primes)
for i in range(n_primes - 1):
    gap_after[i] = primes[i + 1] - primes[i]
gap_after[-1] = gap_after[-2]  # extrapolate last
features['gap_after'] = gap_after

# 2. log(p) / p (form factor first derivative correction)
features['1/p'] = 1.0 / primes_arr
features['log(p)/p'] = log_p / primes_arr
features['log(p)^2/p'] = log_p ** 2 / primes_arr

# 3. Prime index (n-th prime)
prime_idx = np.arange(1, n_primes + 1, dtype=float)
features['prime_index'] = prime_idx

# 4. Fractional part of log(p)/log(2) — measures position in prime ladder
features['frac_log_ratio'] = np.array([np.log(p) / np.log(2) % 1 for p in primes])

# 5. Residue classes mod small numbers
for q in [3, 4, 5, 6, 8, 12]:
    features[f'p_mod_{q}'] = np.array([p % q for p in primes], dtype=float)

# 6. Legendre symbol (quadratic residue of 2, 3, 5 mod p)
for a in [2, 3, 5]:
    features[f'legendre_{a}'] = np.array([pow(a, (p-1)//2, p) if p > 2 else 0
                                           for p in primes], dtype=float)
    # Convert p-1 -> -1
    features[f'legendre_{a}'] = np.where(features[f'legendre_{a}'] == primes_arr - 1,
                                          -1, features[f'legendre_{a}'])

# 7. Mobius function of nearby integers
features['mobius_p-1'] = np.array([int(mobius(p - 1)) for p in primes], dtype=float)
features['mobius_p+1'] = np.array([int(mobius(p + 1)) for p in primes], dtype=float)

# 8. Number of prime factors of p-1 and p+1
def omega(n):
    """Number of distinct prime factors."""
    return len(factorint(n))

features['omega_p-1'] = np.array([omega(p - 1) for p in primes], dtype=float)
features['omega_p+1'] = np.array([omega(p + 1) for p in primes], dtype=float)

# 9. Mertens function M(p) = sum_{n<=p} mu(n)
mertens = np.zeros(n_primes)
running_sum = 0
max_p = max(primes)
mu_vals = {}
for n in range(1, max_p + 1):
    mu_vals[n] = int(mobius(n))
    running_sum += mu_vals[n]
    if n in set(primes):
        idx = primes.index(n)
        mertens[idx] = running_sum
features['mertens'] = mertens

# 10. Li(p) - pi(p) (deviation of prime counting from log integral)
# pi(p) = prime_index
features['pi_deviation'] = prime_idx - np.array([p / np.log(p) for p in primes])

# 11. Chebyshev bias: primes ≡ 3 mod 4 vs 1 mod 4 running count
bias_34 = np.cumsum([1 if p % 4 == 3 else (-1 if p % 4 == 1 else 0) for p in primes])
features['chebyshev_bias'] = bias_34.astype(float)

# 12. Frequency-domain: tau = log(p)/log(T/2pi) position in form factor
tau = log_p / LOG_T
features['tau'] = tau
features['1-tau'] = 1.0 - tau  # distance from form factor boundary

# Compute correlations
print(f'\n{"Feature":<20} {"Pearson r":>10} {"p-value":>10} {"Spearman":>10} {"p-value":>10}')
print('-' * 65)

# Use only well-resolved primes (say p <= 100) to avoid multicollinearity
n_good = sum(1 for p in primes if p <= 100)
sig_features = []

for name, feat in sorted(features.items()):
    feat_good = feat[:n_good]
    delta_good = delta_C[:n_good]
    if np.std(feat_good) < 1e-10:
        continue
    r_p, p_p = pearsonr(feat_good, delta_good)
    r_s, p_s = spearmanr(feat_good, delta_good)
    tag = ''
    if p_p < 0.05 or p_s < 0.05:
        tag = ' *'
        sig_features.append((name, r_p, p_p, r_s, p_s))
    if p_p < 0.01 or p_s < 0.01:
        tag = ' **'
    print(f'{name:<20} {r_p:>+10.4f} {p_p:>10.4f} {r_s:>+10.4f} {p_s:>10.4f}{tag}')

if sig_features:
    print(f'\n{len(sig_features)} features with p < 0.05 (expect ~{len(features)*0.05:.1f} by chance)')
else:
    print('\nNo significant correlations found.')

# ============================================================
# STEP 3: Height dependence — compare low-T vs high-T
# ============================================================
print('\n' + '=' * 70)
print('HEIGHT DEPENDENCE: low-T zeros vs Odlyzko high-T')
print('=' * 70)

# Load low zeros
try:
    zeros_low = np.load('_zeros_500.npy')
    T_low = np.mean(zeros_low)
    LOG_T_low = np.log(T_low / (2 * np.pi))
    density_low = LOG_T_low / (2 * np.pi)
    sp_low = np.diff(zeros_low) * density_low
    sp_low /= np.mean(sp_low)
    N_low = len(sp_low)
    max_lag_low = min(100, N_low // 4)  # conservative lag limit

    # GUE baseline for low-T (reuse same baseline — should be OK for normalized spacings)
    acf_low = spacing_autocorrelation(sp_low, max_lag_low)[1:max_lag_low + 1]
    baseline_low = baseline[:max_lag_low]
    excess_low = acf_low - baseline_low
    se_low = 1.0 / np.sqrt(N_low)

    k_low = np.arange(1, max_lag_low + 1, dtype=float)

    # Extract per-prime cos amplitudes at low-T
    # Use fewer primes (only those with f < 0.25 at low-T)
    primes_low_list = [p for p in primes if np.log(p) / LOG_T_low < 0.25]
    freqs_low = [np.log(p) / LOG_T_low for p in primes_low_list]
    n_primes_low = len(primes_low_list)

    cols_low = []
    for f in freqs_low:
        cols_low.append(np.cos(2 * np.pi * k_low * f))
        cols_low.append(np.sin(2 * np.pi * k_low * f))
    short_low = [np.exp(-k_low / 1.0), np.exp(-k_low / 3.0), 1.0 / k_low ** 2]
    cols_low.extend(short_low)

    if len(cols_low) < max_lag_low:
        X_low = np.column_stack(cols_low)
        amps_low, _, _, _ = np.linalg.lstsq(X_low, excess_low, rcond=None)

        A_cos_low = np.array([amps_low[2*i] for i in range(n_primes_low)])
        C_smooth_low = np.array([np.log(p) / p ** ALPHA for p in primes_low_list])
        scale_low = np.dot(C_smooth_low, A_cos_low) / np.dot(C_smooth_low, C_smooth_low)

        print(f'  Low-T: {N_low} spacings at T~{T_low:.0f}, {n_primes_low} primes, {max_lag_low} lags')
        print(f'  Scale: high-T={scale:.5f}, low-T={scale_low:.5f}, ratio={scale_low/scale:.3f}')

        # Compare per-prime amplitudes (normalized by smooth law)
        # For primes in common
        common = [p for p in primes_low_list if p in primes[:n_good]]
        if len(common) >= 5:
            norm_high = []
            norm_low = []
            for p in common:
                idx_h = primes.index(p)
                idx_l = primes_low_list.index(p)
                smooth_h = scale * np.log(p) / p ** ALPHA
                smooth_l = scale_low * np.log(p) / p ** ALPHA
                if abs(smooth_h) > 1e-10 and abs(smooth_l) > 1e-10:
                    norm_high.append(A_cos[idx_h] / smooth_h)
                    norm_low.append(A_cos_low[idx_l] / smooth_l)

            norm_high = np.array(norm_high)
            norm_low = np.array(norm_low)

            r, p_val = pearsonr(norm_high, norm_low)
            print(f'\n  Normalized amplitude correlation (high vs low T):')
            print(f'  Pearson r = {r:+.4f}, p = {p_val:.4f}')
            print(f'  Common primes: {len(common)}')

            if p_val < 0.05:
                print('  -> CORRELATED: per-prime pattern is stable across heights!')
                print('  -> Fluctuations are intrinsic to the primes, not T-dependent')
            else:
                print('  -> UNCORRELATED: per-prime pattern changes with height')
                print('  -> Fluctuations are T-dependent modulation')

            # Show comparison
            print(f'\n  {"p":>5} {"high_norm":>10} {"low_norm":>10} {"diff":>10}')
            for i, p in enumerate(common[:20]):
                print(f'  {p:>5} {norm_high[i]:>+10.4f} {norm_low[i]:>+10.4f} {norm_high[i]-norm_low[i]:>+10.4f}')
        else:
            print(f'  Only {len(common)} common primes — too few for comparison')
    else:
        print(f'  Too many regressors ({len(cols_low)}) for {max_lag_low} lags — skipping')
except FileNotFoundError:
    print('  _zeros_500.npy not found — skipping height comparison')

# ============================================================
# STEP 4: Multivariate model — can we predict delta_C?
# ============================================================
print('\n' + '=' * 70)
print('MULTIVARIATE PREDICTION OF delta_C')
print('=' * 70)

# Use the top features (if any were significant) in a multivariate regression
# Even without significance, test: how much of delta_C is predictable?

# Build feature matrix from the most theoretically motivated features
feat_names = ['tau', '1-tau', 'gap_before', 'gap_after', 'mertens',
              'chebyshev_bias', 'omega_p-1', 'prime_index']
F = np.column_stack([features[n][:n_good] for n in feat_names])
delta_good = delta_C[:n_good]

# Standardize features
F_std = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-10)

# Ridge regression (regularized to prevent overfitting with few data points)
from numpy.linalg import lstsq
amps_mv, _, _, _ = lstsq(F_std, delta_good, rcond=None)
pred_mv = F_std @ amps_mv
R2_mv = 1 - np.sum((delta_good - pred_mv) ** 2) / np.sum(delta_good ** 2)
R2_adj_mv = 1 - (1 - R2_mv) * (n_good - 1) / (n_good - len(feat_names) - 1)

print(f'  {n_good} primes (p<=100), {len(feat_names)} features')
print(f'  R2 = {R2_mv:.4f}, R2_adj = {R2_adj_mv:.4f}')
print(f'\n  Feature importances:')
for i, name in enumerate(feat_names):
    print(f'    {name:<20} coeff={amps_mv[i]:+.6f}')

if R2_adj_mv > 0.10:
    print(f'\n  -> {R2_adj_mv*100:.0f}% of amplitude fluctuations are predictable')
else:
    print(f'\n  -> Amplitude fluctuations are NOT predictable from these features')
    print('  -> They may be truly random or require different correlates')

# ============================================================
# STEP 5: Autocorrelation of delta_C (ordered by prime)
# ============================================================
print('\n' + '=' * 70)
print('SPATIAL STRUCTURE: autocorrelation of delta_C along prime sequence')
print('=' * 70)

delta_centered = delta_C[:n_good] - np.mean(delta_C[:n_good])
acf_delta = np.correlate(delta_centered, delta_centered, 'full')
acf_delta = acf_delta[n_good - 1:] / acf_delta[n_good - 1]
se_acf = 1.0 / np.sqrt(n_good)

print(f'  ACF of delta_C (first 10 lags):')
for lag in range(1, 11):
    sig = '*' if abs(acf_delta[lag]) > 2 * se_acf else ''
    print(f'    lag {lag:>2}: {acf_delta[lag]:+.4f} (se={se_acf:.4f}) {sig}')

n_sig_acf = sum(1 for lag in range(1, n_good // 4) if abs(acf_delta[lag]) > 2 * se_acf)
print(f'  Significant lags (|r| > 2*se): {n_sig_acf} / {n_good // 4}')
print(f'  Expected by chance: ~{n_good // 4 * 0.05:.1f}')

# ============================================================
# STEP 6: Fourier analysis of delta_C — hidden periodicities?
# ============================================================
print('\n' + '=' * 70)
print('FOURIER ANALYSIS: periodicities in delta_C')
print('=' * 70)

fft_delta = np.fft.rfft(delta_C[:n_good])
power = np.abs(fft_delta) ** 2
freqs_fft = np.fft.rfftfreq(n_good)

# Top 5 peaks
peaks = np.argsort(power[1:])[-5:][::-1] + 1
print(f'  Top 5 Fourier peaks in delta_C:')
print(f'  {"Rank":<5} {"Freq":>8} {"Period":>8} {"Power":>10} {"% of total":>10}')
total_power = np.sum(power[1:])
for rank, idx in enumerate(peaks):
    period = 1.0 / freqs_fft[idx] if freqs_fft[idx] > 0 else float('inf')
    pct = power[idx] / total_power * 100
    print(f'  {rank+1:<5} {freqs_fft[idx]:>8.4f} {period:>8.1f} {power[idx]:>10.6f} {pct:>9.1f}%')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: NATURE OF PER-PRIME FLUCTUATIONS')
print('=' * 70)

# Count significant correlations
n_sig_corr = len(sig_features)
n_features = len(features)
expected_by_chance = n_features * 0.05

print(f'\nCorrelation scan: {n_sig_corr} significant / {n_features} tested '
      f'(expected by chance: {expected_by_chance:.1f})')

if n_sig_corr <= expected_by_chance + 1:
    print('-> No excess correlations. Fluctuations appear UNCORRELATED with')
    print('   standard number-theoretic quantities.')
else:
    print('-> EXCESS correlations detected! The following features predict delta_C:')
    for name, rp, pp, rs, ps in sig_features:
        print(f'   {name}: r={rp:+.3f} (p={pp:.4f})')

if n_sig_acf <= n_good // 4 * 0.1:
    print('\nSpatial structure: NONE. Adjacent primes have independent fluctuations.')
else:
    print(f'\nSpatial structure: PRESENT ({n_sig_acf} significant ACF lags).')
    print('Nearby primes have correlated amplitude deviations.')

print(f'\nMultivariate prediction: R2_adj = {R2_adj_mv:.4f}')
if R2_adj_mv < 0.05:
    print('-> Per-prime fluctuations are essentially UNPREDICTABLE')
    print('-> They carry genuine mathematical information not reducible to')
    print('   simple number-theoretic features')
    print('-> This is the irreducible complexity of the prime distribution')
    print('   as encoded in the zeta zero ACF')
elif R2_adj_mv < 0.25:
    print('-> Weakly predictable. Some structure but mostly noise or')
    print('   higher-order correlations not captured by linear model.')
else:
    print(f'-> {R2_adj_mv*100:.0f}% predictable from number-theoretic features.')
    print('   The amplitude profile carries interpretable information.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
