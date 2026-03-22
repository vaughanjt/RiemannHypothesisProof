"""Ridge regression for stable per-prime amplitude extraction.

The free cos+sin regression gives unstable amplitudes due to multicollinearity
(adjacent primes have similar frequencies). Ridge regression shrinks coefficients
toward zero, preferring the smooth solution while allowing per-prime deviations
only when the data strongly supports them.

Key question: do the gap/Chebyshev correlations survive regularization?
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.stats import spearmanr, pearsonr
from sympy import primerange, mobius, factorint
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

MAX_LAG = 400
T_BASE = 267653395647.0
LOG_T = np.log(T_BASE / (2 * np.pi))
k_arr = np.arange(1, MAX_LAG + 1, dtype=float)
ALPHA = 0.787

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
print(f'  {N_real} spacings, {time.time()-t0:.1f}s')

# ============================================================
# BUILD DESIGN MATRIX: per-prime cos + 3 short-range
# (cos only — phases confirmed as noise)
# ============================================================
primes = list(primerange(2, 550))
freqs = [np.log(p) / LOG_T for p in primes]
# Keep primes with f < 0.5 (Nyquist)
mask = [f < 0.45 for f in freqs]
primes = [p for p, m in zip(primes, mask) if m]
freqs = [f for f, m in zip(freqs, mask) if m]
n_primes = len(primes)

short_cols = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]
n_short = len(short_cols)

# Design matrix: one cos column per prime + short-range
X_prime = np.column_stack([np.cos(2 * np.pi * k_arr * f) for f in freqs])
X_short = np.column_stack(short_cols)
X_full = np.column_stack([X_prime, X_short])
n_total = X_full.shape[1]

print(f'{n_primes} primes (f < 0.45), {n_total} total columns')

# ============================================================
# STEP 1: Ridge regression with cross-validated lambda
# ============================================================
print('\n' + '=' * 70)
print('RIDGE REGRESSION: cross-validated regularization')
print('=' * 70)

# Ridge: minimize ||y - Xb||^2 + lambda * ||b_prime||^2
# Only regularize the prime coefficients, not the short-range
# Solution: b = (X^T X + Lambda)^{-1} X^T y
# where Lambda = diag(lambda, ..., lambda, 0, 0, 0)

def ridge_fit(X_prime, X_short, y, lam):
    """Ridge regression: regularize prime cols, leave short-range free."""
    X = np.column_stack([X_prime, X_short])
    n_p = X_prime.shape[1]
    n_s = X_short.shape[1]
    # Penalty matrix: lambda on prime coeffs, 0 on short-range
    Lambda = np.zeros(n_p + n_s)
    Lambda[:n_p] = lam
    XtX = X.T @ X + np.diag(Lambda)
    Xty = X.T @ y
    b = np.linalg.solve(XtX, Xty)
    pred = X @ b
    return b[:n_p], b[n_p:], pred


def ridge_gcv(X_prime, X_short, y, lam):
    """Generalized cross-validation score for ridge."""
    X = np.column_stack([X_prime, X_short])
    n = len(y)
    n_p = X_prime.shape[1]
    n_s = X_short.shape[1]
    Lambda = np.zeros(n_p + n_s)
    Lambda[:n_p] = lam
    XtX = X.T @ X + np.diag(Lambda)
    H = X @ np.linalg.solve(XtX, X.T)  # hat matrix
    b = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ b
    trace_I_minus_H = n - np.trace(H)
    gcv = np.sum(resid ** 2) / (trace_I_minus_H / n) ** 2
    eff_df = np.trace(H)
    return gcv, eff_df, np.sum(resid ** 2)


# Sweep lambda
lambdas = np.logspace(-4, 4, 50)
gcv_scores = []
eff_dfs = []
rss_vals = []

for lam in lambdas:
    gcv, edf, rss = ridge_gcv(X_prime, X_short, excess, lam)
    gcv_scores.append(gcv)
    eff_dfs.append(edf)
    rss_vals.append(rss)

best_idx = np.argmin(gcv_scores)
best_lambda = lambdas[best_idx]
best_edf = eff_dfs[best_idx]

print(f'GCV-optimal lambda: {best_lambda:.4f}')
print(f'Effective DOF: {best_edf:.1f} (out of {n_total})')

# Also show a few key lambda values
print(f'\n{"lambda":>10} {"eff_DOF":>8} {"R2":>7} {"R2_adj":>7}')
print('-' * 40)
for lam in [0, 0.01, 0.1, best_lambda, 1.0, 10.0, 100.0]:
    b_p, b_s, pred = ridge_fit(X_prime, X_short, excess, lam)
    rss = np.sum((excess - pred) ** 2)
    R2 = 1 - rss / ss_tot
    _, edf, _ = ridge_gcv(X_prime, X_short, excess, lam)
    R2_adj = 1 - (1 - R2) * (MAX_LAG - 1) / (MAX_LAG - edf - 1) if edf < MAX_LAG - 1 else R2
    tag = ' <-- GCV optimal' if abs(lam - best_lambda) < 0.001 else ''
    print(f'{lam:>10.4f} {edf:>8.1f} {R2:>7.4f} {R2_adj:>7.4f}{tag}')

# ============================================================
# STEP 2: Extract stable per-prime amplitudes at optimal lambda
# ============================================================
print('\n' + '=' * 70)
print('STABLE PER-PRIME AMPLITUDES (GCV-optimal ridge)')
print('=' * 70)

b_prime, b_short, pred_ridge = ridge_fit(X_prime, X_short, excess, best_lambda)
R2_ridge = 1 - np.sum((excess - pred_ridge) ** 2) / ss_tot

# Smooth law prediction
log_p = np.array([np.log(p) for p in primes])
C_smooth = log_p / np.array([p ** ALPHA for p in primes])
scale_smooth = np.dot(C_smooth, b_prime) / np.dot(C_smooth, C_smooth)
C_pred = scale_smooth * C_smooth
delta_C = b_prime - C_pred

# How well does smooth law fit the ridge amplitudes?
R2_smooth = 1 - np.sum(delta_C ** 2) / np.sum((b_prime - np.mean(b_prime)) ** 2)

print(f'Ridge R2: {R2_ridge:.4f}')
print(f'Smooth law fit to ridge amplitudes: R2 = {R2_smooth:.4f}')
print(f'Scale: {scale_smooth:.5f}')

# Show amplitudes for first 40 primes
print(f'\n{"p":>5} {"freq":>8} {"ridge":>10} {"smooth":>10} {"delta":>10} {"delta/C":>10}')
print('-' * 60)
for i in range(min(40, n_primes)):
    p = primes[i]
    ratio = delta_C[i] / C_pred[i] if abs(C_pred[i]) > 1e-12 else 0
    print(f'{p:>5} {freqs[i]:>8.5f} {b_prime[i]:>+10.6f} {C_pred[i]:>+10.6f} '
          f'{delta_C[i]:>+10.6f} {ratio:>+10.3f}')

# Amplitude stability check: compare OLS vs ridge for small primes
print('\n--- OLS vs Ridge comparison (first 20 primes) ---')
b_ols, _, _ = ridge_fit(X_prime, X_short, excess, 0)
print(f'{"p":>5} {"OLS":>10} {"Ridge":>10} {"Ratio":>8}')
for i in range(20):
    ratio = b_ols[i] / b_prime[i] if abs(b_prime[i]) > 1e-12 else 0
    print(f'{primes[i]:>5} {b_ols[i]:>+10.6f} {b_prime[i]:>+10.6f} {ratio:>8.3f}')

# ============================================================
# STEP 3: Correlations with number-theoretic features (STABLE)
# ============================================================
print('\n' + '=' * 70)
print('CORRELATIONS WITH RIDGE-STABILIZED AMPLITUDES')
print('=' * 70)

primes_arr = np.array(primes)

# Compute features for ALL primes (not just p<=100)
features = {}
gaps_before = np.zeros(n_primes)
gaps_after = np.zeros(n_primes)
for i in range(n_primes):
    if i > 0:
        gaps_before[i] = primes[i] - primes[i - 1]
    if i < n_primes - 1:
        gaps_after[i] = primes[i + 1] - primes[i]
gaps_before[0] = 1
gaps_after[-1] = gaps_after[-2]
features['gap_before'] = gaps_before
features['gap_after'] = gaps_after
features['gap_mean'] = (gaps_before + gaps_after) / 2

# Normalized gap (gap / expected gap ~ log(p))
features['gap_norm'] = (gaps_before + gaps_after) / (2 * log_p)

features['prime_index'] = np.arange(1, n_primes + 1, dtype=float)
features['1/p'] = 1.0 / primes_arr
features['tau'] = np.array(freqs)

# Chebyshev bias (running count 3mod4 - 1mod4)
bias = np.cumsum([1 if p % 4 == 3 else (-1 if p % 4 == 1 else 0) for p in primes])
features['chebyshev_bias'] = bias.astype(float)

# pi-deviation
features['pi_deviation'] = np.arange(1, n_primes + 1) - primes_arr / log_p

# Mertens
mertens = np.zeros(n_primes)
running = 0
for n in range(1, max(primes) + 1):
    running += int(mobius(n))
    if n in set(primes):
        mertens[primes.index(n)] = running
features['mertens'] = mertens

# omega(p-1), omega(p+1)
features['omega_p-1'] = np.array([len(factorint(p - 1)) for p in primes], dtype=float)
features['omega_p+1'] = np.array([len(factorint(p + 1)) for p in primes], dtype=float)

# Residue classes
for q in [4, 8, 12]:
    features[f'p_mod_{q}'] = np.array([p % q for p in primes], dtype=float)

# Legendre symbols
for a in [2, 3, 5]:
    leg = np.array([pow(a, (p-1)//2, p) if p > 2 else 0 for p in primes], dtype=float)
    leg = np.where(leg == primes_arr - 1, -1, leg)
    features[f'legendre_{a}'] = leg

# Use multiple prime ranges to test stability of correlations
print(f'\n--- Correlations at different prime ranges ---')
for max_p_test, label in [(50, 'p<=50'), (100, 'p<=100'), (200, 'p<=200'), (max(primes), 'all')]:
    n_test = sum(1 for p in primes if p <= max_p_test)
    if n_test < 10:
        continue
    delta_test = delta_C[:n_test]

    print(f'\n  {label} ({n_test} primes):')
    print(f'  {"Feature":<20} {"Pearson r":>10} {"p-val":>8} {"Spearman":>10} {"p-val":>8}')
    print(f'  {"-"*60}')

    for name in ['gap_norm', 'chebyshev_bias', 'mertens', 'pi_deviation',
                  'omega_p-1', 'legendre_2', 'legendre_3', 'p_mod_4']:
        feat = features[name][:n_test]
        if np.std(feat) < 1e-10:
            continue
        rp, pp = pearsonr(feat, delta_test)
        rs, ps = spearmanr(feat, delta_test)
        tag = ' *' if pp < 0.05 else ''
        if pp < 0.01:
            tag = ' **'
        print(f'  {name:<20} {rp:>+10.4f} {pp:>8.4f} {rs:>+10.4f} {ps:>8.4f}{tag}')

# ============================================================
# STEP 4: The critical test — constrained model with gap correction
# ============================================================
print('\n' + '=' * 70)
print('GAP-CORRECTED AMPLITUDE MODEL')
print('=' * 70)
print('If gaps predict delta_C, a model C_p ~ log(p)/p^alpha * (1 + beta*gap)')
print('should improve on the smooth law.')

# Test: does adding gap information improve the constrained model?
def build_gap_model(primes, alpha, beta_gap, k_arr, log_T):
    """Model with gap-corrected amplitudes."""
    model = np.zeros(len(k_arr))
    for i, p in enumerate(primes):
        f = np.log(p) / log_T
        if f >= 0.45:
            continue
        # Gap-corrected amplitude
        if i > 0 and i < len(primes) - 1:
            gap = (primes[i] - primes[i-1] + primes[i+1] - primes[i]) / 2
        elif i == 0:
            gap = primes[1] - primes[0]
        else:
            gap = primes[-1] - primes[-2]
        expected_gap = np.log(p)
        gap_ratio = gap / expected_gap
        amp = np.log(p) / p ** alpha * (1 + beta_gap * (gap_ratio - 1))
        model += amp * np.cos(2 * np.pi * k_arr * f)
    return model

from scipy.optimize import minimize

def neg_R2_gap(params):
    alpha, beta = params
    m = build_gap_model(primes, alpha, beta, k_arr, LOG_T)
    X = np.column_stack([m] + short_cols)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    return np.sum((excess - X @ a) ** 2) / ss_tot

# Baseline (no gap correction)
res_base = minimize(lambda p: neg_R2_gap([p[0], 0]), [0.79], method='Nelder-Mead')
R2_base = 1 - neg_R2_gap([res_base.x[0], 0])

# With gap correction
res_gap = minimize(neg_R2_gap, [0.79, 0.5], method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-8})
R2_gap = 1 - res_gap.fun
opt_alpha, opt_beta = res_gap.x

# True DOF comparison
R2_adj_base = 1 - (1 - R2_base) * (MAX_LAG - 1) / (MAX_LAG - 5)
R2_adj_gap = 1 - (1 - R2_gap) * (MAX_LAG - 1) / (MAX_LAG - 6)

print(f'Smooth only: alpha={res_base.x[0]:.4f}, R2={R2_base:.4f}, R2_adj={R2_adj_base:.4f}')
print(f'Gap-corrected: alpha={opt_alpha:.4f}, beta={opt_beta:+.4f}, R2={R2_gap:.4f}, R2_adj={R2_adj_gap:.4f}')
print(f'Improvement: delta_R2_adj = {R2_adj_gap - R2_adj_base:+.4f}')

if R2_adj_gap > R2_adj_base + 0.005:
    print('-> Gap correction IMPROVES the model (survives DOF penalty)')
else:
    print('-> Gap correction does NOT improve after DOF penalty')

# Also test Chebyshev-corrected model
def build_cheb_model(primes, alpha, beta_cheb, k_arr, log_T):
    """Model with Chebyshev bias correction."""
    model = np.zeros(len(k_arr))
    running_bias = 0
    for i, p in enumerate(primes):
        f = np.log(p) / log_T
        if f >= 0.45:
            continue
        if p % 4 == 3:
            running_bias += 1
        elif p % 4 == 1:
            running_bias -= 1
        amp = np.log(p) / p ** alpha * (1 + beta_cheb * running_bias / (i + 1))
        model += amp * np.cos(2 * np.pi * k_arr * f)
    return model

def neg_R2_cheb(params):
    alpha, beta = params
    m = build_cheb_model(primes, alpha, beta, k_arr, LOG_T)
    X = np.column_stack([m] + short_cols)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    return np.sum((excess - X @ a) ** 2) / ss_tot

res_cheb = minimize(neg_R2_cheb, [0.79, 0.1], method='Nelder-Mead',
                    options={'xatol': 1e-5, 'fatol': 1e-8})
R2_cheb = 1 - res_cheb.fun
R2_adj_cheb = 1 - (1 - R2_cheb) * (MAX_LAG - 1) / (MAX_LAG - 6)
print(f'\nChebyshev-corrected: alpha={res_cheb.x[0]:.4f}, beta={res_cheb.x[1]:+.4f}, '
      f'R2={R2_cheb:.4f}, R2_adj={R2_adj_cheb:.4f}')
print(f'Improvement: delta_R2_adj = {R2_adj_cheb - R2_adj_base:+.4f}')

# Combined: gap + Chebyshev
def neg_R2_combined(params):
    alpha, bg, bc = params
    model = np.zeros(MAX_LAG)
    running_bias = 0
    for i, p in enumerate(primes):
        f = np.log(p) / LOG_T
        if f >= 0.45: continue
        if p % 4 == 3: running_bias += 1
        elif p % 4 == 1: running_bias -= 1
        if i > 0 and i < len(primes) - 1:
            gap = (primes[i]-primes[i-1]+primes[i+1]-primes[i])/2
        elif i == 0: gap = primes[1]-primes[0]
        else: gap = primes[-1]-primes[-2]
        gap_ratio = gap / np.log(p)
        amp = np.log(p)/p**alpha * (1 + bg*(gap_ratio-1) + bc*running_bias/(i+1))
        model += amp * np.cos(2*np.pi*k_arr*f)
    X = np.column_stack([model]+short_cols)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    return np.sum((excess - X @ a)**2) / ss_tot

res_comb = minimize(neg_R2_combined, [0.79, 0.5, 0.1], method='Nelder-Mead',
                    options={'xatol': 1e-5, 'fatol': 1e-8})
R2_comb = 1 - res_comb.fun
R2_adj_comb = 1 - (1 - R2_comb) * (MAX_LAG - 1) / (MAX_LAG - 7)
print(f'\nCombined (gap+Cheb): alpha={res_comb.x[0]:.4f}, '
      f'beta_gap={res_comb.x[1]:+.4f}, beta_cheb={res_comb.x[2]:+.4f}')
print(f'R2={R2_comb:.4f}, R2_adj={R2_adj_comb:.4f}')
print(f'Improvement: delta_R2_adj = {R2_adj_comb - R2_adj_base:+.4f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

print(f'\nRidge regression:')
print(f'  Smooth law R2 on ridge amplitudes: {R2_smooth:.4f}')
print(f'  GCV-optimal lambda: {best_lambda:.4f}, eff. DOF: {best_edf:.1f}')

improvements = {
    'gap': R2_adj_gap - R2_adj_base,
    'chebyshev': R2_adj_cheb - R2_adj_base,
    'combined': R2_adj_comb - R2_adj_base,
}
best_correction = max(improvements, key=improvements.get)
best_improvement = improvements[best_correction]

print(f'\nConstrained model improvements (R2_adj over smooth-only):')
print(f'  Gap correction:      {improvements["gap"]:+.4f}')
print(f'  Chebyshev correction: {improvements["chebyshev"]:+.4f}')
print(f'  Combined:            {improvements["combined"]:+.4f}')

if best_improvement > 0.01:
    print(f'\n>>> {best_correction.upper()} correction provides genuine improvement')
    print(f'>>> The per-prime fluctuations are PARTIALLY predicted by {best_correction}')
    print(f'>>> The amplitude law should be:')
    if best_correction == 'combined':
        print(f'>>>   C_p ~ log(p)/p^{res_comb.x[0]:.3f} '
              f'* (1 + {res_comb.x[1]:+.3f}*(gap_ratio-1) '
              f'+ {res_comb.x[2]:+.3f}*cheb_bias)')
    elif best_correction == 'gap':
        print(f'>>>   C_p ~ log(p)/p^{opt_alpha:.3f} * (1 + {opt_beta:+.3f}*(gap_ratio-1))')
    else:
        print(f'>>>   C_p ~ log(p)/p^{res_cheb.x[0]:.3f} * (1 + {res_cheb.x[1]:+.3f}*cheb_bias)')
elif best_improvement > 0.002:
    print(f'\n>>> MARGINAL improvement from {best_correction} ({best_improvement:+.4f})')
    print('>>> Suggestive but not conclusive with current data')
else:
    print(f'\n>>> NO improvement from any correction')
    print('>>> Per-prime fluctuations are NOT predicted by gaps or Chebyshev bias')
    print('>>> They are either genuinely random or require different correlates')

print(f'\nTotal time: {time.time() - t0:.1f}s')
