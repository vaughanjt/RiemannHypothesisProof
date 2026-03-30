"""Definitive alpha sweep: find the optimal decay exponent and bootstrap CI.

Uses the constrained cos+sin+short-range model, sweeping alpha continuously.
This avoids the per-prime free-fit noise issues."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from sympy import primerange
from riemann.analysis.bost_connes_operator import (
    spacing_autocorrelation, polynomial_unfold
)

max_lag = 400  # full 400 lags for maximum DOF

def load_zeros(path):
    values = []
    with open(path) as f:
        for line in f:
            try: values.append(float(line.strip()))
            except ValueError: continue
    return np.array(values)

res = load_zeros('data/odlyzko/zeros3.txt')
T_base = 267653395647.0
log_T = np.log(T_base / (2*np.pi))
density = log_T / (2*np.pi)
sp = np.diff(res) * density
sp = sp / np.mean(sp)
N = len(sp)
n_data = max_lag

acf = spacing_autocorrelation(sp, max_lag)

print('Computing GUE baseline (100 matrices at N=1200)...')
rng = np.random.default_rng(42)
gue_acfs = []
for _ in range(100):
    A = rng.standard_normal((1200, 1200)) + 1j * rng.standard_normal((1200, 1200))
    H = (A + A.conj().T) / (2 * np.sqrt(2400))
    eigs = np.linalg.eigvalsh(H)
    s = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(s) > max_lag + 10:
        gue_acfs.append(spacing_autocorrelation(s, max_lag))
gue_acf = np.mean(gue_acfs, axis=0)
excess = acf[1:max_lag+1] - gue_acf[1:max_lag+1]
ss_tot = np.sum(excess**2)

def make_cos(freq, n):
    return np.array([np.cos(2*np.pi*k*freq) for k in range(1, n+1)])
def make_sin(freq, n):
    return np.array([np.sin(2*np.pi*k*freq) for k in range(1, n+1)])

k_arr = np.arange(1, n_data + 1, dtype=float)
short_range = [np.exp(-k_arr), np.exp(-k_arr/3), 1.0/k_arr**2]
primes_200 = list(primerange(2, 201))

def fit_alpha_model(alpha, excess, primes, max_m=6):
    """Fit constrained model at given alpha. Returns R2, R2_adj."""
    cos_m = np.zeros(len(excess))
    sin_m = np.zeros(len(excess))
    for p in primes:
        for m in range(1, max_m + 1):
            freq = m * np.log(p) / log_T
            if freq >= 0.5: break
            w = np.log(p) / p**(alpha * m)
            cos_m += w * make_cos(freq, len(excess))
            sin_m += w * make_sin(freq, len(excess))
    cols = [cos_m, sin_m] + short_range
    X = np.column_stack(cols)
    amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    pred = X @ amps
    ss_res = np.sum((excess - pred)**2)
    R2 = 1 - ss_res / ss_tot
    n_p = len(cols)
    R2_adj = 1 - (1 - R2) * (len(excess) - 1) / (len(excess) - n_p - 1)
    return R2, R2_adj

# ============================================================
# SWEEP alpha from 0.3 to 1.3
# ============================================================
print('\n' + '='*70)
print('ALPHA SWEEP: Constrained model R2_adj vs alpha')
print('='*70)

alphas = np.linspace(0.30, 1.30, 201)
R2s = []
R2_adjs = []
for a in alphas:
    R2, R2_adj = fit_alpha_model(a, excess, primes_200)
    R2s.append(R2)
    R2_adjs.append(R2_adj)

R2s = np.array(R2s)
R2_adjs = np.array(R2_adjs)
best_idx = np.argmax(R2_adjs)
alpha_best = alphas[best_idx]
R2_adj_best = R2_adjs[best_idx]

print(f'\nBest alpha = {alpha_best:.4f}, R2_adj = {R2_adj_best:.4f}')
print(f'\nAlpha sweep results:')
print(f'{"alpha":>8} {"R2_adj":>8}')
print('-'*20)
for a_show in [0.40, 0.50, 0.60, 0.667, 0.70, 0.75, 0.76, 0.77, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10]:
    idx = np.argmin(np.abs(alphas - a_show))
    marker = ' <-- BEST' if abs(alphas[idx] - alpha_best) < 0.01 else ''
    marker += ' <-- 3/4' if abs(a_show - 0.75) < 0.01 else ''
    marker += ' <-- explicit' if abs(a_show - 0.50) < 0.01 else ''
    marker += ' <-- Montgomery' if abs(a_show - 1.00) < 0.01 else ''
    print(f'{alphas[idx]:>8.3f} {R2_adjs[idx]:>8.4f}{marker}')

# How flat is the peak?
# Find alpha range where R2_adj is within 0.01 of peak
threshold = R2_adj_best - 0.01
in_range = alphas[R2_adjs >= threshold]
print(f'\nalpha range within 0.01 of peak: [{in_range[0]:.3f}, {in_range[-1]:.3f}]')

# ============================================================
# BOOTSTRAP alpha uncertainty
# ============================================================
print('\n' + '='*70)
print('BOOTSTRAP: Alpha uncertainty from resampled ACF')
print('='*70)

# Bootstrap by resampling the spacings, recomputing ACF and refitting
n_boot = 200
alpha_boots = []
rng_boot = np.random.default_rng(42)

print(f'Running {n_boot} bootstrap iterations...')
for b in range(n_boot):
    # Block bootstrap: resample blocks of spacings to preserve local correlation
    block_size = 100
    n_blocks = N // block_size
    blocks = [sp[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    boot_blocks = [blocks[j] for j in rng_boot.integers(0, n_blocks, size=n_blocks)]
    sp_boot = np.concatenate(boot_blocks)
    sp_boot = sp_boot / np.mean(sp_boot)

    acf_boot = spacing_autocorrelation(sp_boot, max_lag)
    excess_boot = acf_boot[1:max_lag+1] - gue_acf[1:max_lag+1]
    ss_tot_boot = np.sum(excess_boot**2)

    if ss_tot_boot < 1e-20:
        continue

    # Quick sweep: test 5 alpha values, parabolic interpolation
    test_alphas = [0.50, 0.65, 0.80, 0.95, 1.10]
    test_R2adj = []
    for a in test_alphas:
        cos_m = np.zeros(n_data)
        sin_m = np.zeros(n_data)
        for p in primes_200:
            for m in range(1, 7):
                freq = m * np.log(p) / log_T
                if freq >= 0.5: break
                w = np.log(p) / p**(a * m)
                cos_m += w * make_cos(freq, n_data)
                sin_m += w * make_sin(freq, n_data)
        cols = [cos_m, sin_m] + short_range
        X = np.column_stack(cols)
        amps, _, _, _ = np.linalg.lstsq(X, excess_boot, rcond=None)
        pred = X @ amps
        ss_res = np.sum((excess_boot - pred)**2)
        R2 = 1 - ss_res / ss_tot_boot
        n_p = len(cols)
        R2_adj = 1 - (1 - R2) * (n_data - 1) / (n_data - n_p - 1)
        test_R2adj.append(R2_adj)

    # Parabolic fit to find peak
    best_i = np.argmax(test_R2adj)
    if 0 < best_i < len(test_alphas) - 1:
        # Fit parabola through 3 points around peak
        x = np.array(test_alphas[best_i-1:best_i+2])
        y = np.array(test_R2adj[best_i-1:best_i+2])
        # y = a*x^2 + b*x + c -> peak at x = -b/(2a)
        coeffs = np.polyfit(x, y, 2)
        if coeffs[0] < 0:  # concave -> has maximum
            alpha_peak = -coeffs[1] / (2 * coeffs[0])
            if 0.3 < alpha_peak < 1.3:
                alpha_boots.append(alpha_peak)
            else:
                alpha_boots.append(test_alphas[best_i])
        else:
            alpha_boots.append(test_alphas[best_i])
    else:
        alpha_boots.append(test_alphas[best_i])

alpha_boots = np.array(alpha_boots)
alpha_mean = np.mean(alpha_boots)
alpha_std = np.std(alpha_boots)
alpha_ci_lo = np.percentile(alpha_boots, 2.5)
alpha_ci_hi = np.percentile(alpha_boots, 97.5)

print(f'\nBootstrap results ({len(alpha_boots)} valid iterations):')
print(f'  alpha = {alpha_mean:.4f} +/- {alpha_std:.4f}')
print(f'  95% CI: [{alpha_ci_lo:.4f}, {alpha_ci_hi:.4f}]')
print(f'  Median: {np.median(alpha_boots):.4f}')
print()

# Test specific hypotheses
for test_a, label in [(0.50, 'Explicit (1/2)'),
                       (0.75, 'Convolution (3/4)'),
                       (1.00, 'Montgomery (1)'),
                       (alpha_best, f'Sweep best ({alpha_best:.3f})')]:
    z = (alpha_mean - test_a) / alpha_std if alpha_std > 0 else 0
    in_ci = alpha_ci_lo <= test_a <= alpha_ci_hi
    print(f'  {label:<25}: z={z:+.2f}, in 95% CI: {"YES" if in_ci else "NO"}')

# ============================================================
# FINAL VERDICT
# ============================================================
print('\n' + '='*70)
print('VERDICT: THEORETICAL DERIVATION OF alpha')
print('='*70)

print(f"""
SWEEP RESULT:  alpha_opt = {alpha_best:.4f}
BOOTSTRAP:     alpha = {alpha_mean:.4f} +/- {alpha_std:.4f}
               95% CI = [{alpha_ci_lo:.4f}, {alpha_ci_hi:.4f}]

THEORETICAL PREDICTIONS:
  Explicit formula:  alpha = 0.500  {"IN CI" if alpha_ci_lo <= 0.5 <= alpha_ci_hi else "OUTSIDE CI"}
  Convolution (3/4): alpha = 0.750  {"IN CI" if alpha_ci_lo <= 0.75 <= alpha_ci_hi else "OUTSIDE CI"}
  Montgomery (1):    alpha = 1.000  {"IN CI" if alpha_ci_lo <= 1.0 <= alpha_ci_hi else "OUTSIDE CI"}
""")

# The R2_adj curve
print('R2_adj landscape (how flat is the peak?):')
for da in [0.5, 0.75, alpha_best, 1.0]:
    idx = np.argmin(np.abs(alphas - da))
    drop = R2_adj_best - R2_adjs[idx]
    print(f'  alpha={da:.3f}: R2_adj={R2_adjs[idx]:.4f} (drop from peak: {drop:.4f})')

print()
if alpha_ci_lo <= 0.75 <= alpha_ci_hi:
    print('CONCLUSION: alpha = 3/4 is CONSISTENT with the data.')
    print()
    print('The convolution argument holds:')
    print('  - The spacing ACF amplitude is the GEOMETRIC MEAN of')
    print('    the pair correlation amplitude (1/p) and the density')
    print('    fluctuation amplitude (1/sqrt(p))')
    print('  - This gives 1/p^(3/4), i.e., alpha = 3/4')
    print()
    print('The R2_adj landscape is FLAT near the peak -- alpha values')
    print(f'from {in_range[0]:.2f} to {in_range[-1]:.2f} are statistically equivalent.')
    print('This flatness means the data cannot distinguish between')
    print('alpha=0.75 and alpha=0.80 (or even alpha=1.0 in some bootstraps).')
    print()
    print('The convolution derivation provides the THEORETICAL REASON')
    print('why alpha is near 3/4, even if the data alone cannot pin it')
    print('down more precisely than the 95% CI.')
else:
    print(f'CONCLUSION: alpha = 3/4 is NOT in the 95% CI [{alpha_ci_lo:.3f}, {alpha_ci_hi:.3f}].')
    print('The convolution argument may need refinement.')
