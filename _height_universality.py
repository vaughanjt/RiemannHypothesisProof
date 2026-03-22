"""Height universality test: is the 5-parameter model universal across T?

The model: ACF_excess(k) = scale * Sum_p log(p)/p^alpha * cos(2*pi*k*logp/logT) + short-range
has 5 parameters: alpha, scale, a (exp(-k)), b (exp(-k/3)), c (1/k^2).

If these parameters are UNIVERSAL (same at all heights T), the model is
a fundamental structural result about the zeta function.
If they SHIFT with T, the model is a first-order approximation with
deeper T-dependent structure underneath.

Data:
  - Low T:  ~500 zeros at T ~ 458 (_zeros_500.npy)
  - Mid T:  ~200 zeros at T ~ 270 (_zeros_200.npy)
  - High T: 10,000 zeros at T ~ 2.7e11 (Odlyzko zeros3.txt)
  - Ultra T: zeros at T ~ 1.4e20 (Odlyzko zeros4.txt)
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize_scalar, minimize
from sympy import primerange
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

# ============================================================
# GUE BASELINE (shared)
# ============================================================
t0 = time.time()
print('Computing GUE baseline...')

def gue_eigs(n, rng):
    d = rng.standard_normal(n)
    e = np.sqrt(rng.chisquare(2 * np.arange(n - 1, 0, -1)) / 2)
    return eigvalsh_tridiagonal(d, e) / np.sqrt(n)

rng_bl = np.random.default_rng(42)
bl_acfs_full = []
for _ in range(100):
    eigs = gue_eigs(1200, rng_bl)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > 410:
        bl_acfs_full.append(spacing_autocorrelation(sp, 400))
baseline_full = np.mean(bl_acfs_full, axis=0)
print(f'  Baseline: {len(bl_acfs_full)} matrices, {time.time()-t0:.1f}s')

primes_all = list(primerange(2, 500))

# ============================================================
# LOAD ALL DATASETS
# ============================================================
print('\nLoading datasets...')

datasets = {}

# High T: Odlyzko zeros3 (T ~ 2.7e11)
zeros_high = []
with open('data/odlyzko/zeros3.txt') as f:
    for line in f:
        try: zeros_high.append(float(line.strip()))
        except ValueError: pass
zeros_high = np.array(zeros_high)
T_high = 267653395647.0
datasets['high'] = {'zeros': zeros_high, 'T': T_high, 'label': 'T~2.7e11'}

# Ultra T: Odlyzko zeros4 (T ~ 1.44e20)
# NOTE: zeros4.txt contains RESIDUALS from a known base height, not absolute values
zeros_ultra = []
with open('data/odlyzko/zeros4.txt') as f:
    for line in f:
        try: zeros_ultra.append(float(line.strip()))
        except ValueError: pass
zeros_ultra = np.array(zeros_ultra)
T_ultra = 1.44e20  # known base height for Odlyzko zeros4 block
datasets['ultra'] = {'zeros': zeros_ultra, 'T': T_ultra, 'label': 'T~1.4e20'}

# Low T: computed zeros at T ~ 458
try:
    zeros_low = np.load('_zeros_500.npy')
    T_low = np.mean(zeros_low)
    datasets['low'] = {'zeros': zeros_low, 'T': T_low, 'label': f'T~{T_low:.0f}'}
except FileNotFoundError:
    print('  _zeros_500.npy not found, skipping low-T')

# Mid T: computed zeros at T ~ 270
try:
    zeros_mid = np.load('_zeros_200.npy')
    T_mid = np.mean(zeros_mid)
    datasets['mid'] = {'zeros': zeros_mid, 'T': T_mid, 'label': f'T~{T_mid:.0f}'}
except FileNotFoundError:
    print('  _zeros_200.npy not found, skipping mid-T')

for name, ds in datasets.items():
    print(f'  {name}: {len(ds["zeros"])} zeros, {ds["label"]}')

# ============================================================
# FIT THE 5-PARAMETER MODEL AT EACH HEIGHT
# ============================================================
print('\n' + '=' * 70)
print('5-PARAMETER FIT AT EACH HEIGHT')
print('=' * 70)


def fit_5param(zeros, T_base, max_lag, baseline_arr):
    """Fit the 5-parameter model at a given height.
    Returns: alpha, scale, a, b, c, R2, R2_adj, se, excess."""
    log_T = np.log(T_base / (2 * np.pi))
    density = log_T / (2 * np.pi)
    sp = np.diff(zeros) * density
    sp = sp / np.mean(sp)
    N = len(sp)
    se = 1.0 / np.sqrt(N)

    actual_max_lag = min(max_lag, N // 4)
    acf = spacing_autocorrelation(sp, actual_max_lag)
    bl = baseline_arr[1:actual_max_lag + 1]
    excess = acf[1:actual_max_lag + 1] - bl
    ss_tot = np.sum(excess ** 2)

    k_arr = np.arange(1, actual_max_lag + 1, dtype=float)
    short_cols = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]

    def build_model(alpha, primes, log_T):
        model = np.zeros(actual_max_lag)
        for p in primes:
            freq = np.log(p) / log_T
            if freq >= 0.45:
                continue
            amp = np.log(p) / p ** alpha
            model += amp * np.cos(2 * np.pi * k_arr * freq)
        return model

    def neg_R2(alpha):
        model = build_model(alpha, primes_all, log_T)
        X = np.column_stack([model] + short_cols)
        a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
        return np.sum((excess - X @ a) ** 2) / ss_tot

    res = minimize_scalar(neg_R2, bounds=(0.3, 1.5), method='bounded')
    opt_alpha = res.x
    R2 = 1 - res.fun

    model = build_model(opt_alpha, primes_all, log_T)
    X = np.column_stack([model] + short_cols)
    amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    opt_scale = amps[0]
    opt_a, opt_b, opt_c = amps[1], amps[2], amps[3]
    n_params = 5  # alpha (pre-optimized) + scale + 3 short-range
    R2_adj = 1 - (1 - R2) * (actual_max_lag - 1) / (actual_max_lag - n_params - 1)

    pred = X @ amps
    residual = excess - pred
    chi2 = np.sum((residual / se) ** 2)
    dof = actual_max_lag - n_params
    chi2_z = (chi2 - dof) / np.sqrt(2 * dof) if dof > 0 else 0

    return {
        'alpha': opt_alpha, 'scale': opt_scale,
        'a': opt_a, 'b': opt_b, 'c': opt_c,
        'R2': R2, 'R2_adj': R2_adj,
        'chi2_z': chi2_z,
        'N': N, 'max_lag': actual_max_lag,
        'se': se, 'excess': excess, 'log_T': log_T,
    }


results = {}
print(f'\n{"Dataset":<10} {"N":>6} {"lags":>5} {"alpha":>7} {"scale":>9} '
      f'{"a(exp-k)":>9} {"b(exp-k/3)":>10} {"c(1/k2)":>9} {"R2_adj":>7} {"chi2_z":>7}')
print('-' * 95)

for name in ['mid', 'low', 'high', 'ultra']:
    if name not in datasets:
        continue
    ds = datasets[name]
    max_lag = 400 if name in ('high', 'ultra') else min(100, len(ds['zeros']) // 5)
    try:
        r = fit_5param(ds['zeros'], ds['T'], max_lag, baseline_full)
        results[name] = r
        print(f'{ds["label"]:<10} {r["N"]:>6} {r["max_lag"]:>5} {r["alpha"]:>7.4f} '
              f'{r["scale"]:>+9.5f} {r["a"]:>+9.4f} {r["b"]:>+9.4f} {r["c"]:>+9.5f} '
              f'{r["R2_adj"]:>7.4f} {r["chi2_z"]:>+7.2f}')
    except Exception as e:
        print(f'{ds["label"]:<10} FAILED: {e}')

# ============================================================
# UNIVERSALITY ANALYSIS
# ============================================================
print('\n' + '=' * 70)
print('UNIVERSALITY ANALYSIS')
print('=' * 70)

# Compare alpha across heights
if len(results) >= 2:
    alphas = {name: r['alpha'] for name, r in results.items()}
    scales = {name: r['scale'] for name, r in results.items()}
    a_vals = {name: r['a'] for name, r in results.items()}
    b_vals = {name: r['b'] for name, r in results.items()}
    c_vals = {name: r['c'] for name, r in results.items()}
    log_Ts = {name: r['log_T'] for name, r in results.items()}

    print('\n--- Alpha (decay exponent) ---')
    for name, alpha in sorted(alphas.items(), key=lambda x: datasets[x[0]]['T']):
        T = datasets[name]['T']
        print(f'  {datasets[name]["label"]:<15} alpha = {alpha:.4f}  (log(T/2pi) = {log_Ts[name]:.2f})')

    alpha_vals = list(alphas.values())
    alpha_spread = max(alpha_vals) - min(alpha_vals)
    alpha_mean = np.mean(alpha_vals)
    print(f'  Range: {min(alpha_vals):.4f} to {max(alpha_vals):.4f} (spread = {alpha_spread:.4f})')
    print(f'  Mean: {alpha_mean:.4f}')

    if alpha_spread < 0.1:
        print('  -> STABLE: alpha varies by less than 0.1 across heights')
    elif alpha_spread < 0.3:
        print('  -> MODERATE VARIATION: alpha shifts by ~0.1-0.3')
    else:
        print('  -> LARGE VARIATION: alpha is strongly T-dependent')

    print('\n--- Scale (overall amplitude) ---')
    for name, sc in sorted(scales.items(), key=lambda x: datasets[x[0]]['T']):
        T = datasets[name]['T']
        # Theoretical prediction: scale should go as 1/log(T/2pi) or similar
        log_T = log_Ts[name]
        print(f'  {datasets[name]["label"]:<15} scale = {sc:+.5f}  '
              f'(scale * log_T = {sc * log_T:+.4f})')

    print('\n--- Short-range coefficients ---')
    print(f'  {"Dataset":<15} {"a(exp-k)":>10} {"b(exp-k/3)":>10} {"c(1/k2)":>10}')
    for name in sorted(results.keys(), key=lambda x: datasets[x]['T']):
        r = results[name]
        print(f'  {datasets[name]["label"]:<15} {r["a"]:>+10.4f} {r["b"]:>+10.4f} {r["c"]:>+10.5f}')

    # Test: does alpha correlate with log(T)?
    if len(results) >= 3:
        log_T_arr = np.array([log_Ts[n] for n in sorted(results.keys(),
                              key=lambda x: datasets[x]['T'])])
        alpha_arr = np.array([alphas[n] for n in sorted(results.keys(),
                              key=lambda x: datasets[x]['T'])])
        if len(log_T_arr) >= 3:
            from scipy.stats import pearsonr
            r, p = pearsonr(log_T_arr, alpha_arr)
            print(f'\n--- Alpha vs log(T) correlation ---')
            print(f'  Pearson r = {r:+.4f}, p = {p:.4f}')
            if p < 0.05:
                print('  -> SIGNIFICANT: alpha depends on T')
                # Fit alpha = a + b * log(log(T))
                log_log_T = np.log(log_T_arr)
                slope = np.polyfit(log_log_T, alpha_arr, 1)
                print(f'  Fit: alpha = {slope[1]:.4f} + {slope[0]:+.4f} * log(log(T/2pi))')
            else:
                print('  -> NOT SIGNIFICANT: alpha is consistent with universal')

# ============================================================
# EXCESS SHAPE COMPARISON
# ============================================================
print('\n' + '=' * 70)
print('EXCESS SHAPE: normalized comparison across heights')
print('=' * 70)

# Compare the shape of ACF excess at different heights
# Normalize each by its L2 norm so we compare shapes, not amplitudes
if len(results) >= 2:
    # Use the shortest common lag range
    common_lags = min(r['max_lag'] for r in results.values())
    common_lags = min(common_lags, 50)  # focus on first 50 lags for shape

    print(f'  Comparing first {common_lags} lags (normalized by L2 norm)')
    print(f'\n  Pairwise correlation of ACF excess shapes:')

    names = sorted(results.keys(), key=lambda x: datasets[x]['T'])
    shape_matrix = {}
    for name in names:
        exc = results[name]['excess'][:common_lags]
        norm = np.sqrt(np.sum(exc ** 2))
        shape_matrix[name] = exc / norm if norm > 1e-10 else exc

    print(f'  {"":>15}', end='')
    for n2 in names:
        print(f'  {datasets[n2]["label"]:>12}', end='')
    print()
    for n1 in names:
        print(f'  {datasets[n1]["label"]:>15}', end='')
        for n2 in names:
            corr = np.dot(shape_matrix[n1], shape_matrix[n2])
            print(f'  {corr:>+12.4f}', end='')
        print()

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: HEIGHT UNIVERSALITY')
print('=' * 70)

if len(results) < 2:
    print('\nInsufficient datasets for comparison.')
else:
    # Primary question: is alpha universal?
    alpha_vals = [results[n]['alpha'] for n in results]
    alpha_spread = max(alpha_vals) - min(alpha_vals)

    # Secondary: is the shape universal?
    if len(results) >= 2:
        names = sorted(results.keys(), key=lambda x: datasets[x]['T'])
        # Get pairwise correlations
        corrs = []
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                exc1 = results[n1]['excess'][:common_lags]
                exc2 = results[n2]['excess'][:common_lags]
                norm1, norm2 = np.linalg.norm(exc1), np.linalg.norm(exc2)
                if norm1 > 1e-10 and norm2 > 1e-10:
                    corrs.append(np.dot(exc1, exc2) / (norm1 * norm2))

    print(f'\n  Alpha spread across heights: {alpha_spread:.4f}')
    if corrs:
        print(f'  Shape correlations: {[f"{c:+.3f}" for c in corrs]}')

    if alpha_spread < 0.15 and (not corrs or min(corrs) > 0.5):
        print(f'\n>>> UNIVERSAL: The 5-parameter model is height-independent.')
        print(f'>>> alpha = {np.mean(alpha_vals):.3f} +/- {np.std(alpha_vals):.3f} across all heights.')
        print(f'>>> This is a fundamental structural property of the zeta function,')
        print(f'>>> not an artifact of the observation height.')
    elif alpha_spread < 0.3:
        print(f'\n>>> APPROXIMATELY UNIVERSAL: alpha varies modestly ({alpha_spread:.3f}).')
        print(f'>>> The 5-parameter model captures the dominant structure at all heights')
        print(f'>>> but the parameters have weak T-dependence.')
        print(f'>>> This may reflect higher-order corrections to the pair correlation.')
    else:
        print(f'\n>>> NOT UNIVERSAL: alpha varies by {alpha_spread:.3f} across heights.')
        print(f'>>> The 5-parameter model is a first-order approximation.')
        print(f'>>> The amplitude law A(p) ~ log(p)/p^alpha has a T-dependent exponent,')
        print(f'>>> likely reflecting the pair correlation form factor evaluated at T.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
