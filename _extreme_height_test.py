"""Extreme height test: prime beat model predictions vs Odlyzko zeros.
Fixed: use residuals directly for float precision, proper GUE baseline."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from scipy import stats
from riemann.analysis.bost_connes_operator import (
    spacing_autocorrelation, gue_reference_autocorrelation
)

max_lag_short = 15
max_lag_long = 80

# ============================================================
# LOAD DATA (use RESIDUALS for high-T to avoid float64 overflow)
# ============================================================
def load_residuals(path):
    """Load numerical values from Odlyzko file."""
    values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            try:
                val = float(line)
                values.append(val)
            except ValueError:
                continue
    return np.array(values)

# For spacing computation, we only need DIFFERENCES between consecutive zeros.
# At high T, the local density normalization uses the FULL height, but since
# density varies slowly over 10k zeros at T~10^11, we can use a constant.

# zeros3: residuals from 267653395647, full T ~ 2.676e11
res_e12 = load_residuals('data/odlyzko/zeros3.txt')
T_base_e12 = 267653395647.0
T_mean_e12 = T_base_e12 + np.mean(res_e12)
print(f'zeros3: {len(res_e12)} zeros, residuals {res_e12[0]:.4f} to {res_e12[-1]:.4f}')
print(f'  Full T ~ {T_mean_e12:.3e}')

# zeros4: residuals from 144176897509546973000
res_e21 = load_residuals('data/odlyzko/zeros4.txt')
T_base_e21 = 144176897509546973000.0
T_mean_e21 = T_base_e21 + np.mean(res_e21)
print(f'zeros4: {len(res_e21)} zeros, residuals {res_e21[0]:.4f} to {res_e21[-1]:.4f}')
print(f'  Full T ~ {T_mean_e21:.3e}')

# Our zeros
zeros_200 = np.load('_zeros_200.npy')
T_mean_200 = np.mean(zeros_200)
print(f'Our: {len(zeros_200)} zeros, T ~ {T_mean_200:.1f}')

# ============================================================
# NORMALIZE SPACINGS
# ============================================================
def normalize_high_T(residuals, T_base):
    """Normalize spacings for high-T zeros using constant local density."""
    sp_raw = np.diff(residuals)
    # At height T, density ~ log(T/(2*pi)) / (2*pi)
    # For 10k zeros spanning ~2500 at T~2.7e11, density varies by ~1e-8 — essentially constant
    density = np.log(T_base / (2*np.pi)) / (2*np.pi)
    sp = sp_raw * density
    return sp / np.mean(sp)

def normalize_low_T(zeros):
    sp_raw = np.diff(np.sort(zeros))
    t_mid = (zeros[:-1] + zeros[1:]) / 2
    ld = np.log(t_mid / (2*np.pi)) / (2*np.pi)
    sp = sp_raw * ld
    return sp / np.mean(sp)

sp_200 = normalize_low_T(zeros_200)
sp_e12 = normalize_high_T(res_e12, T_base_e12)
sp_e21 = normalize_high_T(res_e21, T_base_e21)

print(f'\nSpacings: ours={len(sp_200)}, e12={len(sp_e12)}, e21={len(sp_e21)}')
print(f'Mean: ours={np.mean(sp_200):.4f}, e12={np.mean(sp_e12):.4f}, e21={np.mean(sp_e21):.4f}')
print(f'Std:  ours={np.std(sp_200):.4f}, e12={np.std(sp_e12):.4f}, e21={np.std(sp_e21):.4f}')
print(f'SE:   ours={1/np.sqrt(len(sp_200)):.4f}, e12={1/np.sqrt(len(sp_e12)):.4f}')

# ============================================================
# GUE BASELINE
# ============================================================
print('\nComputing GUE baseline ACF (500 matrices, N=200)...')
gue_acf_short = gue_reference_autocorrelation(200, 500, max_lag_short, seed=42)

# Extended GUE ACF — for k >= ~5, GUE ACF decays like -1/(pi*k)^2
# Compute from matrices for lags 1-80
print('Computing extended GUE ACF (200 matrices, N=500)...')
rng = np.random.default_rng(42)
gue_acfs_long = []
for _ in range(200):
    A = rng.standard_normal((500, 500)) + 1j * rng.standard_normal((500, 500))
    H = (A + A.conj().T) / (2 * np.sqrt(1000))
    eigs = np.linalg.eigvalsh(H)
    from riemann.analysis.bost_connes_operator import polynomial_unfold
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > max_lag_long:
        gue_acfs_long.append(spacing_autocorrelation(sp, max_lag_long))
gue_acf_long = np.mean(gue_acfs_long, axis=0)
print(f'GUE ACF at lag 1: {gue_acf_long[1]:.4f}, lag 5: {gue_acf_long[5]:.4f}, lag 20: {gue_acf_long[20]:.6f}')

# ============================================================
# PREDICTIONS
# ============================================================
print('\n' + '='*70)
print('PRIME BEAT PREDICTIONS')
print('='*70)

small_pairs = [(2,3), (2,5), (2,7), (3,5), (3,7), (5,7)]

for label, T_mean in [('T~230', T_mean_200), ('T~2.7e11', T_mean_e12), ('T~1.4e20', T_mean_e21)]:
    log_T = np.log(T_mean / (2*np.pi))
    beats = {}
    for p1, p2 in small_pairs:
        period = log_T / abs(np.log(p1) - np.log(p2))
        beats[(p1,p2)] = round(period)
    in_range = {k for k in beats.values() if k <= max_lag_long}
    print(f'{label}: beats at lags {sorted(in_range)} (of {sorted(beats.values())})')

# ============================================================
# TEST: ACF at all three heights with proper GUE reference
# ============================================================
print('\n' + '='*70)
print('ACF EXCESS OVER GUE (with proper GUE baseline)')
print('='*70)

acf_200 = spacing_autocorrelation(sp_200, max_lag_short)
acf_e12 = spacing_autocorrelation(sp_e12, max_lag_long)
acf_e21 = spacing_autocorrelation(sp_e21, max_lag_long)

se_200 = 1.0 / np.sqrt(len(sp_200))
se_e12 = 1.0 / np.sqrt(len(sp_e12))
se_e21 = 1.0 / np.sqrt(len(sp_e21))

# Short range comparison (lags 1-15)
print(f'\n--- Lags 1-15 ---')
print(f'{"Lag":<5} {"Ours(excess)":>12} {"z_ours":>7} {"e12(excess)":>12} {"z_e12":>7} {"e21(excess)":>12} {"z_e21":>7}')
print('-'*70)

for k in range(1, max_lag_short + 1):
    ex_200 = acf_200[k] - gue_acf_short[k]
    z_200 = ex_200 / se_200
    ex_e12 = acf_e12[k] - gue_acf_long[k]
    z_e12 = ex_e12 / se_e12
    ex_e21 = acf_e21[k] - gue_acf_long[k]
    z_e21 = ex_e21 / se_e21
    flag = ' ***' if k in {4, 7, 10, 11} else ''
    print(f'{k:<5} {ex_200:>+12.4f} {z_200:>+7.2f} {ex_e12:>+12.4f} {z_e12:>+7.2f} {ex_e21:>+12.4f} {z_e21:>+7.2f}{flag}')

# Extended range for high-T (lags 15-80)
print(f'\n--- Lags 15-80 (high-T only, showing |z|>2 or predicted) ---')

log_T_e12 = np.log(T_mean_e12 / (2*np.pi))
predicted_e12 = {}
for p1, p2 in small_pairs:
    period = log_T_e12 / abs(np.log(p1) - np.log(p2))
    nearest = round(period)
    if nearest <= max_lag_long:
        predicted_e12[(p1,p2)] = nearest
pred_lags = set(predicted_e12.values())

print(f'Predicted beat lags: {sorted(pred_lags)}')
print(f'{"Lag":<5} {"e12 ACF":>10} {"e12 excess":>10} {"z_e12":>7} {"Predicted":>12}')
print('-'*50)

sig_at_pred = 0
sig_total_e12 = 0
for k in range(15, max_lag_long + 1):
    ex = acf_e12[k] - gue_acf_long[k]
    z = ex / se_e12
    is_pred = k in pred_lags
    is_sig = abs(z) > 2.5

    if is_pred or is_sig:
        pred_str = ','.join(f'({p1},{p2})' for (p1,p2), lag in predicted_e12.items() if lag == k)
        if not pred_str:
            pred_str = '-'
        print(f'{k:<5} {acf_e12[k]:>+10.4f} {ex:>+10.4f} {z:>+7.2f} {pred_str:>12}')

    if is_sig:
        sig_total_e12 += 1
        if is_pred:
            sig_at_pred += 1

# ============================================================
# VERDICT
# ============================================================
print('\n' + '='*70)
print('VERDICT')
print('='*70)

# Test 1: Did our anomalous lags disappear at high T?
print('\n1. ANOMALOUS LAG SHIFT:')
for k in [4, 7, 10, 11]:
    ex_200 = acf_200[k] - gue_acf_short[k]
    z_200 = ex_200 / se_200
    ex_e12 = acf_e12[k] - gue_acf_long[k]
    z_e12 = ex_e12 / se_e12
    sign_200 = '+' if ex_200 > 0 else '-'
    sign_e12 = '+' if ex_e12 > 0 else '-'
    changed = 'SHIFTED' if sign_200 != sign_e12 or abs(z_e12) < 1.5 else 'PERSISTS'
    print(f'  lag {k}: T~230 z={z_200:+.2f}({sign_200}), T~2.7e11 z={z_e12:+.2f}({sign_e12}) -> {changed}')

# Test 2: Predicted lags
print(f'\n2. PREDICTED BEAT LAGS:')
print(f'  {sig_at_pred}/{len(pred_lags)} predicted lags are significant at T~2.7e11')

# Test 3: Distribution universality
ks, p = stats.ks_2samp(sp_e12, sp_200)
print(f'\n3. DISTRIBUTION UNIVERSALITY:')
print(f'  KS test (e12 vs ours): p={p:.4f} {"(universal)" if p > 0.05 else "(different)"}')

ks21, p21 = stats.ks_2samp(sp_e21, sp_e12)
print(f'  KS test (e21 vs e12):  p={p21:.4f} {"(universal)" if p21 > 0.05 else "(different)"}')
