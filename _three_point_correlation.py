"""Three-point correlation of zeta zero spacings.
Tests whether the diffuse ~30% remainder in the trace formula decomposition
is captured by higher-order (3-point) correlation structure."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from sympy import primerange
from riemann.analysis.bost_connes_operator import (
    spacing_autocorrelation, polynomial_unfold
)

max_lag = 200  # enough for 3-point structure

# ============================================================
# LOAD DATA
# ============================================================
def load_zeros(path):
    values = []
    with open(path) as f:
        for line in f:
            try:
                values.append(float(line.strip()))
            except ValueError:
                continue
    return np.array(values)

res = load_zeros('data/odlyzko/zeros3.txt')
T_base = 267653395647.0
density = np.log(T_base / (2*np.pi)) / (2*np.pi)
sp = np.diff(res) * density
sp = sp / np.mean(sp)
N = len(sp)
se = 1.0 / np.sqrt(N)

print(f'Data: {N} spacings at T~2.7e11')

# ============================================================
# 2-POINT ACF (baseline)
# ============================================================
acf = spacing_autocorrelation(sp, max_lag)

# GUE baseline
gue_N = 1200
rng = np.random.default_rng(42)
gue_acfs_2pt = []
gue_acfs_3pt = {}  # will be filled below
n_gue = 80

print(f'Computing GUE baseline ({n_gue} matrices at N={gue_N})...')
gue_spacings_list = []
for i in range(n_gue):
    A = rng.standard_normal((gue_N, gue_N)) + 1j * rng.standard_normal((gue_N, gue_N))
    H = (A + A.conj().T) / (2 * np.sqrt(2 * gue_N))
    eigs = np.linalg.eigvalsh(H)
    s = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(s) > max_lag + 10:
        gue_acfs_2pt.append(spacing_autocorrelation(s, max_lag))
        gue_spacings_list.append(s)

gue_acf = np.mean(gue_acfs_2pt, axis=0)
excess_2pt = acf[1:max_lag+1] - gue_acf[1:max_lag+1]
print(f'2-point ACF excess L2: {np.sqrt(np.sum(excess_2pt**2)):.4f}')

# ============================================================
# 3-POINT CORRELATION FUNCTION
# ============================================================
print('\n' + '='*70)
print('3-POINT SPACING CORRELATION')
print('='*70)

def three_point_acf(spacings, max_lag1, max_lag2):
    """Compute C3(k1, k2) = <(s_i - 1)(s_{i+k1} - 1)(s_{i+k2} - 1)>.
    Returns matrix of shape (max_lag1, max_lag2).
    Only computes for k2 >= k1 (upper triangle) for efficiency."""
    n = len(spacings)
    centered = spacings - 1.0  # center around mean=1
    c3 = np.zeros((max_lag1, max_lag2))
    for k1 in range(1, max_lag1 + 1):
        for k2 in range(k1, max_lag2 + 1):
            # <c_i * c_{i+k1} * c_{i+k2}>
            end = n - k2
            if end <= 0:
                continue
            c3[k1-1, k2-1] = np.mean(centered[:end] * centered[k1:end+k1] * centered[k2:end+k2])
    return c3

# Compute 3-point ACF for small lags first (computationally cheap)
max_lag3 = 40  # 40x40 = 1600 entries, manageable
print(f'\nComputing 3-point ACF for zeta zeros (lags 1-{max_lag3})...')
c3_zeta = three_point_acf(sp, max_lag3, max_lag3)

# GUE baseline 3-point
print(f'Computing GUE 3-point baseline ({len(gue_spacings_list)} matrices)...')
c3_gue_list = []
for s in gue_spacings_list:
    c3_gue_list.append(three_point_acf(s, max_lag3, max_lag3))
c3_gue = np.mean(c3_gue_list, axis=0)
c3_gue_std = np.std(c3_gue_list, axis=0) / np.sqrt(len(c3_gue_list))

# 3-point excess
c3_excess = c3_zeta - c3_gue

# Standard error for 3-point
se3 = 1.0 / np.sqrt(N)  # rough estimate, actual SE depends on 6th moment

print(f'\n--- 3-point ACF excess statistics ---')
# Only upper triangle (k2 >= k1)
upper = np.triu(np.ones((max_lag3, max_lag3), dtype=bool))
excess_vals = c3_excess[upper]
n_entries = len(excess_vals)
z_vals = excess_vals / se3

print(f'Entries (upper triangle): {n_entries}')
print(f'Max |excess|: {np.max(np.abs(excess_vals)):.6f}')
print(f'Max |z|: {np.max(np.abs(z_vals)):.2f}')
n_sig = np.sum(np.abs(z_vals) > 2.5)
print(f'Entries with |z| > 2.5: {n_sig} (expected: {n_entries * 0.012:.1f})')
print(f'Entries with |z| > 3.0: {np.sum(np.abs(z_vals) > 3.0)} (expected: {n_entries * 0.003:.1f})')

# ============================================================
# KEY QUESTION: Does 3-point excess correlate with 2-point residual?
# ============================================================
print('\n' + '='*70)
print('KEY TEST: Does 3-point structure explain 2-point residual?')
print('='*70)

# Build the best 2-point model (30-prime cosine, BIC-optimal)
log_T = np.log(T_base / (2*np.pi))

def make_cosine_column(freq, max_lag):
    return np.array([np.cos(2*np.pi*k*freq) for k in range(1, max_lag+1)])

primes_30 = list(primerange(2, 128))[:30]
X_2pt = np.column_stack([make_cosine_column(np.log(p)/log_T, max_lag) for p in primes_30])
amps_2pt, _, _, _ = np.linalg.lstsq(X_2pt, excess_2pt, rcond=None)
pred_2pt = X_2pt @ amps_2pt
residual_2pt = excess_2pt - pred_2pt

print(f'2-point model: 30 primes, R2={1 - np.sum(residual_2pt**2)/np.sum(excess_2pt**2):.4f}')

# Hypothesis: the 2-point residual at lag k is related to
# the "marginal" 3-point structure: sum over k2 of C3_excess(k, k2)
# This is the projection of 3-point onto 2-point via integration
print('\nComputing 3-point marginal (sum C3_excess over second lag)...')
c3_marginal = np.zeros(max_lag3)
for k1 in range(max_lag3):
    # Sum over k2 >= k1 and k2 < k1 (use symmetry)
    c3_marginal[k1] = np.sum(c3_excess[k1, k1:]) + np.sum(c3_excess[:k1, k1])

# Compare marginal to 2-point residual
residual_short = residual_2pt[:max_lag3]
corr = np.corrcoef(c3_marginal, residual_short)[0, 1]
print(f'Correlation between 3-point marginal and 2-point residual: {corr:.4f}')

# Regression: can 3-point marginal predict 2-point residual?
X_3pt = c3_marginal.reshape(-1, 1)
from numpy.linalg import lstsq
beta, _, _, _ = lstsq(X_3pt, residual_short, rcond=None)
pred_3pt = (X_3pt @ beta).flatten()
ss_res_3pt = np.sum((residual_short - pred_3pt)**2)
ss_tot_3pt = np.sum(residual_short**2)
R2_3pt = 1 - ss_res_3pt / ss_tot_3pt
R2_adj_3pt = 1 - (1 - R2_3pt) * (max_lag3 - 1) / (max_lag3 - 2)
print(f'R2 of 3-point marginal predicting 2-point residual: {R2_3pt:.4f} (adj: {R2_adj_3pt:.4f})')

# ============================================================
# 3-POINT STRUCTURE: TOP EXCESSES
# ============================================================
print('\n' + '='*70)
print('STRONGEST 3-POINT EXCESSES')
print('='*70)

# Find top (k1, k2) pairs by |z|
indices = np.argwhere(upper)
z_flat = z_vals
top_idx = np.argsort(np.abs(z_flat))[-20:][::-1]

print(f'{"Rank":<5} {"k1":>4} {"k2":>4} {"C3_zeta":>10} {"C3_gue":>10} {"Excess":>10} {"z":>7}')
print('-'*55)
for rank, idx in enumerate(top_idx):
    k1, k2 = indices[idx]
    print(f'{rank+1:<5} {k1+1:>4} {k2+1:>4} {c3_zeta[k1,k2]:>+10.6f} {c3_gue[k1,k2]:>+10.6f} {c3_excess[k1,k2]:>+10.6f} {z_flat[idx]:>+7.2f}')

# ============================================================
# PRIME STRUCTURE IN 3-POINT CORRELATIONS
# ============================================================
print('\n' + '='*70)
print('PRIME STRUCTURE IN 3-POINT ACF')
print('='*70)

# Test: at (k1, k2) where k2-k1 or k1 or k2 relate to prime beats,
# is the 3-point excess larger?
prime_beat_lags = set()
for p1 in [2, 3, 5, 7, 11]:
    period = 2 * np.pi / (np.log(p1) / log_T) / (2 * np.pi)
    nearest = round(period)
    if 1 <= nearest <= max_lag3:
        prime_beat_lags.add(nearest)
    for p2 in [2, 3, 5, 7, 11]:
        if p2 > p1:
            beat_period = 1.0 / abs(np.log(p2)/log_T - np.log(p1)/log_T)
            nearest = round(beat_period)
            if 1 <= nearest <= max_lag3:
                prime_beat_lags.add(nearest)

print(f'Prime-related lags (periods of log(p)/logT and beats): {sorted(prime_beat_lags)}')

# Mean |z| at prime-related (k1,k2) vs non-prime
prime_z = []
nonprime_z = []
for k1 in range(max_lag3):
    for k2 in range(k1, max_lag3):
        z = c3_excess[k1, k2] / se3
        if (k1+1) in prime_beat_lags or (k2+1) in prime_beat_lags or (k2-k1) in prime_beat_lags:
            prime_z.append(abs(z))
        else:
            nonprime_z.append(abs(z))

print(f'Mean |z| at prime-related lags: {np.mean(prime_z):.3f} (n={len(prime_z)})')
print(f'Mean |z| at non-prime lags:     {np.mean(nonprime_z):.3f} (n={len(nonprime_z)})')
# Significance of the difference
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(prime_z, nonprime_z, alternative='greater')
print(f'Mann-Whitney U test (prime > non-prime): p={pval:.4f}')

# ============================================================
# DIAGONAL STRUCTURE: C3(k, k+d) for small d
# ============================================================
print('\n' + '='*70)
print('DIAGONAL STRUCTURE: C3(k, k+d) for d=0,1,2,3')
print('='*70)

for d in range(4):
    diagonal = []
    for k in range(max_lag3 - d):
        diagonal.append(c3_excess[k, k + d])
    diagonal = np.array(diagonal)
    z_diag = diagonal / se3

    # Is this diagonal correlated with 2-point residual?
    if len(diagonal) >= len(residual_short):
        diag_short = diagonal[:len(residual_short)]
    else:
        diag_short = diagonal
        res_short = residual_short[:len(diagonal)]

    corr_d = np.corrcoef(diag_short, residual_short[:len(diag_short)])[0, 1]
    print(f'd={d}: mean_excess={np.mean(diagonal):+.6f}, '
          f'max|z|={np.max(np.abs(z_diag)):.2f}, '
          f'n_sig(2.5)={np.sum(np.abs(z_diag)>2.5)}, '
          f'corr_with_2pt_resid={corr_d:+.4f}')

# ============================================================
# COMBINED MODEL: 2-point trace + 3-point correction
# ============================================================
print('\n' + '='*70)
print('COMBINED MODEL: 30-prime cosine + 3-point diagonal corrections')
print('='*70)

# Use diagonals d=0,1,2,3 of C3_excess as additional regressors for the 2-point ACF
extra_cols = []
for d in range(4):
    col = np.zeros(max_lag3)
    for k in range(max_lag3 - d):
        col[k] = c3_excess[k, k + d]
    extra_cols.append(col)

# Add short-range terms
k_arr = np.arange(1, max_lag3 + 1, dtype=float)
extra_cols.append(np.exp(-k_arr / 1.0))
extra_cols.append(1.0 / k_arr**2)

# Build combined model for first max_lag3 lags
X_combined = np.column_stack(
    [make_cosine_column(np.log(p)/log_T, max_lag3) for p in primes_30]
    + extra_cols
)
n_params_comb = X_combined.shape[1]
excess_short = excess_2pt[:max_lag3]
ss_tot_short = np.sum(excess_short**2)

amps_comb, _, _, _ = np.linalg.lstsq(X_combined, excess_short, rcond=None)
pred_comb = X_combined @ amps_comb
ss_res_comb = np.sum((excess_short - pred_comb)**2)
R2_comb = 1 - ss_res_comb / ss_tot_short
R2_adj_comb = 1 - (1 - R2_comb) * (max_lag3 - 1) / (max_lag3 - n_params_comb - 1)

print(f'30-prime cosine only (first {max_lag3} lags): R2={1-np.sum(residual_short**2)/ss_tot_short:.4f}')
print(f'+ 4 C3 diagonals + 2 short-range:  R2={R2_comb:.4f}, R2_adj={R2_adj_comb:.4f}')
print(f'Parameters: {n_params_comb} ({30} prime + {len(extra_cols)} extra)')

# Final residual check
resid_comb = excess_short - pred_comb
resid_z_comb = resid_comb / se
n_sig_comb = np.sum(np.abs(resid_z_comb) > 2.5)
chi2_comb = np.sum(resid_z_comb**2)
dof_comb = max_lag3 - n_params_comb
chi2_z_comb = (chi2_comb - dof_comb) / np.sqrt(2 * dof_comb) if dof_comb > 0 else float('inf')

print(f'\nResidual: max|z|={np.max(np.abs(resid_z_comb)):.2f}, '
      f'sig lags={n_sig_comb}/{max_lag3}, '
      f'chi2 z={chi2_z_comb:+.2f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '='*70)
print('3-POINT CORRELATION VERDICT')
print('='*70)

print(f'\n1. 3-point excess exists: {n_sig}/{n_entries} entries > 2.5sigma '
      f'(expected {n_entries*0.012:.0f})')
if n_sig > 2 * n_entries * 0.012:
    print('   => YES, significant 3-point non-GUE structure detected')
else:
    print('   => Marginal or absent')

print(f'\n2. 3-point marginal predicts 2-point residual: R2={R2_3pt:.4f} '
      f'(corr={corr:+.4f})')
if abs(corr) > 0.3:
    print('   => YES, the diffuse 2-point remainder has 3-point origin')
elif abs(corr) > 0.15:
    print('   => PARTIAL connection between 2-point and 3-point structure')
else:
    print('   => NO clear connection — the diffuse remainder is NOT simply 3-point')

print(f'\n3. Prime structure in 3-point ACF: p={pval:.4f}')
if pval < 0.05:
    print('   => YES, prime-related lags show stronger 3-point excess')
else:
    print('   => NO preferential prime structure in 3-point correlations')

print(f'\n4. Combined model (trace + 3-point + short-range): R2_adj={R2_adj_comb:.4f}')
print(f'   vs trace-only: R2_adj~0.53')
if R2_adj_comb > 0.65:
    print('   => SUBSTANTIAL improvement — 3-point terms capture the diffuse remainder')
elif R2_adj_comb > 0.55:
    print('   => MODEST improvement — 3-point partially explains remainder')
else:
    print('   => MINIMAL improvement — diffuse remainder has different origin')
