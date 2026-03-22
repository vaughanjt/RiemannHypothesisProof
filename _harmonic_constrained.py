"""Constrained model with higher harmonics (prime powers m>1).

The 6-param first-harmonic model achieves R2_adj=0.625. The free 206-param
model achieves 0.73. The gap should be from prime powers (m=2,3,4,...).

This script tests: with harmonics m=1..M and constrained amplitudes
A(p,m) = scale * log(p) / p^(alpha*m), how few parameters close the gap?

If a ~10 parameter model reaches R2_adj~0.73, the non-GUE ACF is a
fully predicted function of known mathematics with essentially zero
free parameters beyond normalization.
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize
from sympy import primerange
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

MAX_LAG = 400
T_BASE = 267653395647.0
LOG_T = np.log(T_BASE / (2 * np.pi))
k_arr = np.arange(1, MAX_LAG + 1, dtype=float)

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

zeros = []
with open('data/odlyzko/zeros3.txt') as f:
    for line in f:
        try: zeros.append(float(line.strip()))
        except ValueError: pass
zeros = np.array(zeros)
sp_real = np.diff(zeros) * (LOG_T / (2 * np.pi))
sp_real /= np.mean(sp_real)
N_real = len(sp_real)
se = 1.0 / np.sqrt(N_real)
acf_real = spacing_autocorrelation(sp_real, MAX_LAG)[1:MAX_LAG + 1]
excess = acf_real - baseline
ss_tot = np.sum(excess ** 2)
print(f'  {N_real} spacings, {time.time()-t0:.1f}s')

# Short-range basis
short_3 = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]
short_6 = short_3 + [np.exp(-k_arr / 10.0), 1.0 / k_arr, (-1) ** k_arr / k_arr]

primes_all = list(primerange(2, 1000))

# ============================================================
# STEP 1: Sweep max_harmonic (M) with constrained amplitudes
# ============================================================
print('\n' + '=' * 70)
print('HARMONIC SWEEP: constrained A(p,m) = scale * log(p) / p^(alpha*m)')
print('=' * 70)


def build_harmonic_model(primes, max_m, alpha):
    """Build oscillatory model summing over primes and harmonics.
    Single cosine vector (phase=0) weighted by constrained amplitude."""
    model = np.zeros(MAX_LAG)
    n_terms = 0
    for p in primes:
        for m in range(1, max_m + 1):
            freq = m * np.log(p) / LOG_T
            if freq >= 0.5:
                break
            amp = np.log(p) / p ** (alpha * m)
            model += amp * np.cos(2 * np.pi * k_arr * freq)
            n_terms += 1
    return model, n_terms


def fit_with_short(model_osc, short_cols, excess, ss_tot):
    """Fit: scale * model_osc + short-range terms. Return metrics."""
    X = np.column_stack([model_osc] + short_cols)
    amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    pred = X @ amps
    ss_res = np.sum((excess - pred) ** 2)
    n_p = X.shape[1]
    R2 = 1 - ss_res / ss_tot
    R2_adj = 1 - (1 - R2) * (MAX_LAG - 1) / (MAX_LAG - n_p - 1)
    residual = excess - pred
    chi2 = np.sum((residual / se) ** 2)
    dof = MAX_LAG - n_p
    chi2_z = (chi2 - dof) / np.sqrt(2 * dof)
    return R2, R2_adj, chi2_z, n_p, amps


# Sweep alpha and max_m
print(f'\n{"max_p":>6} {"M":>3} {"alpha":>6} {"short":>5} {"terms":>6} '
      f'{"params":>6} {"R2":>7} {"R2_adj":>7} {"chi2_z":>7}')
print('-' * 70)

results = []
for max_p in [30, 100, 200, 500]:
    primes = [p for p in primes_all if p <= max_p]
    for max_m in [1, 2, 3, 4, 6, 8]:
        for alpha in [0.75, 0.80, 0.834, 0.90, 1.0]:
            for short_cols, short_label in [(short_3, '3'), (short_6, '6')]:
                model, n_terms = build_harmonic_model(primes, max_m, alpha)
                R2, R2_adj, chi2_z, n_p, amps = fit_with_short(
                    model, short_cols, excess, ss_tot)
                results.append((max_p, max_m, alpha, short_label, n_terms,
                                n_p, R2, R2_adj, chi2_z))

# Sort by R2_adj and show top 20
results.sort(key=lambda x: -x[7])
for r in results[:25]:
    mp, mm, al, sl, nt, np_, r2, r2a, cz = r
    print(f'{mp:>6} {mm:>3} {al:>6.3f} {sl:>5} {nt:>6} '
          f'{np_:>6} {r2:>7.4f} {r2a:>7.4f} {cz:>+7.2f}')

best = results[0]
print(f'\nBest: p<={best[0]}, M={best[1]}, alpha={best[2]:.3f}, '
      f'short={best[3]}, R2_adj={best[7]:.4f}')

# ============================================================
# STEP 2: Joint optimization of alpha for each (max_p, max_m)
# ============================================================
print('\n' + '=' * 70)
print('JOINT ALPHA OPTIMIZATION')
print('=' * 70)


def neg_R2_alpha(alpha, primes, max_m, short_cols):
    model, _ = build_harmonic_model(primes, max_m, alpha)
    X = np.column_stack([model] + short_cols)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    return np.sum((excess - X @ a) ** 2) / ss_tot


print(f'\n{"max_p":>6} {"M":>3} {"short":>5} {"opt_alpha":>10} '
      f'{"R2":>7} {"R2_adj":>7} {"chi2_z":>7} {"terms":>6}')
print('-' * 72)

best_configs = []
for max_p in [30, 100, 200, 500]:
    primes = [p for p in primes_all if p <= max_p]
    for max_m in [1, 2, 4, 6, 8]:
        for short_cols, sl, n_short in [(short_3, '3', 3), (short_6, '6', 6)]:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(neg_R2_alpha, bounds=(0.3, 1.5),
                                  args=(primes, max_m, short_cols), method='bounded')
            opt_alpha = res.x
            model, n_terms = build_harmonic_model(primes, max_m, opt_alpha)
            R2, R2_adj, chi2_z, n_p, amps = fit_with_short(
                model, short_cols, excess, ss_tot)
            # True DOF: alpha (optimized) + scale + n_short = 2 + n_short
            true_params = 2 + n_short
            R2_adj_true = 1 - (1 - R2) * (MAX_LAG - 1) / (MAX_LAG - true_params - 1)
            best_configs.append((max_p, max_m, sl, opt_alpha, R2, R2_adj_true,
                                 chi2_z, n_terms, true_params))
            print(f'{max_p:>6} {max_m:>3} {sl:>5} {opt_alpha:>10.4f} '
                  f'{R2:>7.4f} {R2_adj_true:>7.4f} {chi2_z:>+7.2f} {n_terms:>6}')

best_configs.sort(key=lambda x: -x[5])
bc = best_configs[0]
print(f'\nBest joint: p<={bc[0]}, M={bc[1]}, short={bc[2]}, alpha={bc[3]:.4f}, '
      f'R2_adj={bc[5]:.4f} ({bc[8]} true DOF)')

# ============================================================
# STEP 3: Separate alpha per harmonic order
# ============================================================
print('\n' + '=' * 70)
print('SEPARATE ALPHA PER HARMONIC: A(p,m) = scale_m * log(p) / p^(alpha_m)')
print('=' * 70)
print('Do different harmonics prefer different decay rates?')


def build_separate_harmonics(primes, max_m, alphas):
    """Build separate model vector for each harmonic order."""
    models = []
    for m in range(1, max_m + 1):
        model_m = np.zeros(MAX_LAG)
        for p in primes:
            freq = m * np.log(p) / LOG_T
            if freq >= 0.5:
                continue
            amp = np.log(p) / p ** (alphas[m - 1] * m)
            model_m += amp * np.cos(2 * np.pi * k_arr * freq)
        models.append(model_m)
    return models


# Joint optimize alpha_1, ..., alpha_M
primes_200 = [p for p in primes_all if p <= 200]
for max_m in [2, 4, 6]:
    def neg_R2_multi(alpha_vec, max_m=max_m):
        models = build_separate_harmonics(primes_200, max_m, alpha_vec)
        X = np.column_stack(models + short_6)
        a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
        return np.sum((excess - X @ a) ** 2) / ss_tot

    x0 = [0.83] * max_m
    res = minimize(neg_R2_multi, x0, method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 5000})
    opt_alphas = res.x
    R2_multi = 1 - res.fun
    n_params = max_m + 6  # max_m scales + 6 short-range (alphas pre-fit)
    true_dof = max_m * 2 + 6  # alphas + scales + short-range
    R2_adj_multi = 1 - (1 - R2_multi) * (MAX_LAG - 1) / (MAX_LAG - true_dof - 1)

    alpha_str = ', '.join(f'{a:.3f}' for a in opt_alphas)
    print(f'\nM={max_m}: alphas = [{alpha_str}]')
    print(f'  R2={R2_multi:.4f}, R2_adj({true_dof} DOF)={R2_adj_multi:.4f}')

    # Show per-harmonic contribution
    models = build_separate_harmonics(primes_200, max_m, opt_alphas)
    X = np.column_stack(models + short_6)
    amps_multi, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    for m_idx in range(max_m):
        contrib = amps_multi[m_idx] * models[m_idx]
        var_frac = np.sum(contrib * excess) / ss_tot
        print(f'  m={m_idx+1}: alpha={opt_alphas[m_idx]:.3f}, '
              f'scale={amps_multi[m_idx]:+.5f}, var_frac={var_frac:.3f}')
    sr_start = max_m
    sr_labels = ['exp(-k/1)', 'exp(-k/3)', 'exp(-k/10)', '1/k', '1/k^2', '(-1)^k/k']
    for i, label in enumerate(sr_labels):
        contrib = amps_multi[sr_start + i] * short_6[i]
        var_frac = np.sum(contrib * excess) / ss_tot
        print(f'  {label}: coeff={amps_multi[sr_start+i]:+.5f}, var_frac={var_frac:.3f}')

# ============================================================
# STEP 4: Comparison table — all model families
# ============================================================
print('\n' + '=' * 70)
print('COMPARISON TABLE: All model families')
print('=' * 70)

print(f'\n{"Model":<58} {"DOF":>4} {"R2":>7} {"R2_adj":>7}')
print('-' * 80)

# Recompute key models for clean comparison
# A. Original Selberg 5-param (from _selberg_convergence.py)
cos_sel = np.zeros(MAX_LAG)
sin_sel = np.zeros(MAX_LAG)
for p in primes_200:
    for m in range(1, 9):
        f = m * np.log(p) / LOG_T
        if f >= 0.5: break
        w = np.log(p) / p ** (0.758 * m)
        cos_sel += w * np.cos(2 * np.pi * k_arr * f)
        sin_sel += w * np.sin(2 * np.pi * k_arr * f)
X_sel = np.column_stack([cos_sel, sin_sel] + short_3)
a_sel, _, _, _ = np.linalg.lstsq(X_sel, excess, rcond=None)
R2_sel = 1 - np.sum((excess - X_sel @ a_sel) ** 2) / ss_tot
R2a_sel = 1 - (1 - R2_sel) * (MAX_LAG - 1) / (MAX_LAG - 6)
print(f'{"Selberg a=0.758 cos+sin M<=8 p<=200 + 3short":<58} {5:>4} {R2_sel:>7.4f} {R2a_sel:>7.4f}')

# B. First harmonic only, alpha=0.834
model_1h, _ = build_harmonic_model(primes_200, 1, 0.834)
R2_1h, R2a_1h, _, _, _ = fit_with_short(model_1h, short_3, excess, ss_tot)
R2a_1h_true = 1 - (1 - R2_1h) * (MAX_LAG - 1) / (MAX_LAG - 6)
print(f'{"Constrained a=0.834 M=1 p<=200 + 3short":<58} {5:>4} {R2_1h:>7.4f} {R2a_1h_true:>7.4f}')

# C. Best single-alpha harmonic model
model_best, n_t = build_harmonic_model(
    [p for p in primes_all if p <= bc[0]], bc[1], bc[3])
short_best = short_3 if bc[2] == '3' else short_6
R2_best, R2a_best, cz_best, _, _ = fit_with_short(model_best, short_best, excess, ss_tot)
n_short_best = 3 if bc[2] == '3' else 6
R2a_best_true = 1 - (1 - R2_best) * (MAX_LAG - 1) / (MAX_LAG - (2 + n_short_best) - 1)
print(f'{"Best single-alpha: p<="+str(bc[0])+" M="+str(bc[1])+" a="+f"{bc[3]:.3f}"+" +"+bc[2]+"short":<58} '
      f'{2+n_short_best:>4} {R2_best:>7.4f} {R2a_best_true:>7.4f}')

# D. Separate alpha per harmonic (M=4)
def neg_R2_m4(alpha_vec):
    models = build_separate_harmonics(primes_200, 4, alpha_vec)
    X = np.column_stack(models + short_6)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    return np.sum((excess - X @ a) ** 2) / ss_tot

res_m4 = minimize(neg_R2_m4, [0.83]*4, method='Nelder-Mead',
                  options={'xatol': 1e-4, 'fatol': 1e-8})
R2_m4 = 1 - res_m4.fun
R2a_m4 = 1 - (1 - R2_m4) * (MAX_LAG - 1) / (MAX_LAG - 15)  # 4 alphas + 4 scales + 6 short + 1 (generous)
print(f'{"Sep. alpha M=4 p<=200 + 6short":<58} {14:>4} {R2_m4:>7.4f} {R2a_m4:>7.4f}')

# E. Free cos+sin p<=200 m<=6 + 6 short (original 206-param)
freqs = []
for p in primes_200:
    for m in range(1, 7):
        f = m * np.log(p) / LOG_T
        if f < 0.5: freqs.append(f)
uq = []
for f in freqs:
    if not any(abs(f - u) < 1e-6 for u in uq): uq.append(f)
if len(uq) > MAX_LAG // 4:
    uq = uq[:MAX_LAG // 4]
cols_free = []
for f in uq:
    cols_free.append(np.cos(2 * np.pi * k_arr * f))
    cols_free.append(np.sin(2 * np.pi * k_arr * f))
cols_free.extend(short_6)
X_free = np.column_stack(cols_free)
a_free, _, _, _ = np.linalg.lstsq(X_free, excess, rcond=None)
R2_free = 1 - np.sum((excess - X_free @ a_free) ** 2) / ss_tot
R2a_free = 1 - (1 - R2_free) * (MAX_LAG - 1) / (MAX_LAG - X_free.shape[1] - 1)
print(f'{"Free cos+sin p<=200 M<=6 + 6short (206 params)":<58} {X_free.shape[1]:>4} {R2_free:>7.4f} {R2a_free:>7.4f}')

# ============================================================
# STEP 5: The definitive constrained model
# ============================================================
print('\n' + '=' * 70)
print('DEFINITIVE MODEL: best constrained configuration')
print('=' * 70)

# Use the separate-alpha M=4 model for detailed analysis
opt_alphas_4 = res_m4.x
models_4 = build_separate_harmonics(primes_200, 4, opt_alphas_4)
X_def = np.column_stack(models_4 + short_6)
amps_def, _, _, _ = np.linalg.lstsq(X_def, excess, rcond=None)
pred_def = X_def @ amps_def
residual_def = excess - pred_def
chi2_def = np.sum((residual_def / se) ** 2)
dof_def = MAX_LAG - 14  # generous DOF count
chi2_z_def = (chi2_def - dof_def) / np.sqrt(2 * dof_def)

print(f'Harmonics: m=1..4, primes p<=200')
print(f'Alphas: {", ".join(f"m{i+1}={a:.4f}" for i, a in enumerate(opt_alphas_4))}')
print(f'Scales: {", ".join(f"m{i+1}={amps_def[i]:+.5f}" for i in range(4))}')
print(f'Short-range: {", ".join(f"{amps_def[4+i]:+.5f}" for i in range(6))}')
print(f'R2={R2_m4:.4f}, R2_adj(14 DOF)={R2a_m4:.4f}')
print(f'Chi2 z-score: {chi2_z_def:+.2f}')
print(f'Residual max |z|: {np.max(np.abs(residual_def / se)):.2f}')
n_sig = np.sum(np.abs(residual_def / se) > 2.5)
print(f'Significant residual lags: {n_sig}/{MAX_LAG}')

# Variance decomposition
print(f'\nVariance decomposition:')
total_explained = 0
for i in range(4):
    contrib = amps_def[i] * models_4[i]
    var_abs = np.sum(contrib * excess)
    var_frac = var_abs / ss_tot
    total_explained += var_frac
    print(f'  Harmonic m={i+1} (alpha={opt_alphas_4[i]:.3f}): {var_frac*100:>+6.1f}%')
sr_labels = ['exp(-k/1)', 'exp(-k/3)', 'exp(-k/10)', '1/k', '1/k^2', '(-1)^k/k']
sr_total = 0
for i in range(6):
    contrib = amps_def[4 + i] * short_6[i]
    var_frac = np.sum(contrib * excess) / ss_tot
    sr_total += var_frac
    total_explained += var_frac
print(f'  Short-range total: {sr_total*100:>+6.1f}%')
print(f'  Total explained: {total_explained*100:.1f}%')
print(f'  Residual: {(1-R2_m4)*100:.1f}%')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

gap_closed = (R2a_m4 - R2a_sel) / (R2a_free - R2a_sel) * 100 if R2a_free > R2a_sel else 0
print(f'\nOriginal Selberg (5 DOF):     R2_adj = {R2a_sel:.4f}')
print(f'Best constrained ({14} DOF):    R2_adj = {R2a_m4:.4f}')
print(f'Free model (206 DOF):         R2_adj = {R2a_free:.4f}')
print(f'\nGap closed: {gap_closed:.0f}%')

if R2a_m4 >= R2a_free - 0.01:
    print(f'\n>>> CONVERGED: {14}-parameter constrained model matches the free model')
    print('>>> The non-GUE ACF is a PREDICTED function with ~14 effective parameters:')
    print('>>>   4 harmonic decay rates + 4 harmonic scales + 6 short-range')
    print('>>>   All from the Selberg/Montgomery trace formula framework')
elif R2a_m4 >= 0.70:
    print(f'\n>>> NEAR-COMPLETE: constrained model explains {R2a_m4*100:.0f}% (adjusted)')
    print(f'>>> Closes {gap_closed:.0f}% of the gap to the free model')
    print('>>> Remaining gap likely from per-prime amplitude fluctuations')
else:
    print(f'\n>>> PARTIAL: constrained model at {R2a_m4*100:.0f}% (adjusted)')
    print(f'>>> Higher harmonics help but per-prime structure remains')

# Is the alpha pattern meaningful?
print(f'\nAlpha pattern across harmonics:')
for i, a in enumerate(opt_alphas_4):
    expected_if_uniform = opt_alphas_4[0]  # all same
    print(f'  m={i+1}: alpha = {a:.4f}')

alpha_spread = np.std(opt_alphas_4)
print(f'  Spread (std): {alpha_spread:.4f}')
if alpha_spread < 0.05:
    print('  -> Consistent with uniform alpha (single decay law)')
else:
    print('  -> Different harmonics prefer different decay rates')
    print('  -> The amplitude law may have m-dependent structure')

print(f'\nTotal time: {time.time() - t0:.1f}s')
