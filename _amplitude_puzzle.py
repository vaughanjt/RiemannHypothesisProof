"""Amplitude decay puzzle: why do prime beat amplitudes drop 3x faster than 1/log(T)?
Hypothesis: prime powers (p^2, p^3) and cross-terms (p1*p2) in the explicit formula
create destructive interference that increases with T."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from scipy import stats
from riemann.analysis.bost_connes_operator import (
    spacing_autocorrelation, polynomial_unfold
)

max_lag_low = 15
max_lag_high = 80

# ============================================================
# LOAD ALL DATA
# ============================================================
# T~230 (our zeros)
zeros_200 = np.load('_zeros_200.npy')
sp_raw = np.diff(np.sort(zeros_200))
t_mid = (zeros_200[:-1] + zeros_200[1:]) / 2
ld = np.log(t_mid / (2*np.pi)) / (2*np.pi)
sp_200 = sp_raw * ld / np.mean(sp_raw * ld)
acf_200 = spacing_autocorrelation(sp_200, max_lag_low)

# T~2.7e11 (Odlyzko zeros3)
def load_residuals(path):
    values = []
    with open(path) as f:
        for line in f:
            try:
                values.append(float(line.strip()))
            except ValueError:
                continue
    return np.array(values)

res_e12 = load_residuals('data/odlyzko/zeros3.txt')
T_base_e12 = 267653395647.0
density_e12 = np.log(T_base_e12 / (2*np.pi)) / (2*np.pi)
sp_e12 = np.diff(res_e12) * density_e12
sp_e12 = sp_e12 / np.mean(sp_e12)
acf_e12 = spacing_autocorrelation(sp_e12, max_lag_high)

# T~500 (our zeros, high half)
zeros_500 = np.load('_zeros_500.npy')
high_zeros = zeros_500[250:]
sp_raw_h = np.diff(np.sort(high_zeros))
t_mid_h = (high_zeros[:-1] + high_zeros[1:]) / 2
ld_h = np.log(t_mid_h / (2*np.pi)) / (2*np.pi)
sp_500h = sp_raw_h * ld_h / np.mean(sp_raw_h * ld_h)
acf_500h = spacing_autocorrelation(sp_500h, max_lag_low)

# GUE baselines
rng = np.random.default_rng(42)
gue_acfs_short = []
gue_acfs_long = []
for _ in range(300):
    A = rng.standard_normal((200, 200)) + 1j * rng.standard_normal((200, 200))
    H = (A + A.conj().T) / (2 * np.sqrt(400))
    eigs = np.linalg.eigvalsh(H)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > max_lag_low:
        gue_acfs_short.append(spacing_autocorrelation(sp, max_lag_low))
for _ in range(200):
    A = rng.standard_normal((500, 500)) + 1j * rng.standard_normal((500, 500))
    H = (A + A.conj().T) / (2 * np.sqrt(1000))
    eigs = np.linalg.eigvalsh(H)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > max_lag_high:
        gue_acfs_long.append(spacing_autocorrelation(sp, max_lag_high))
gue_short = np.mean(gue_acfs_short, axis=0)
gue_long = np.mean(gue_acfs_long, axis=0)

excess_200 = acf_200[1:max_lag_low+1] - gue_short[1:max_lag_low+1]
excess_e12 = acf_e12[1:max_lag_high+1] - gue_long[1:max_lag_high+1]
excess_500h = acf_500h[1:max_lag_low+1] - gue_short[1:max_lag_low+1]

print('Data loaded.')
print(f'  T~230: {len(sp_200)} spacings')
print(f'  T~645: {len(sp_500h)} spacings')
print(f'  T~2.7e11: {len(sp_e12)} spacings')

# ============================================================
# MODEL 1: Single primes only (original model)
# ============================================================
primes = [2, 3, 5, 7, 11]

def build_design_matrix(T, max_lag, terms):
    """Build design matrix for given terms at height T.
    terms: list of (base, power) tuples, e.g., (2,1)=log(2), (2,2)=log(4)
    """
    log_T = np.log(T / (2*np.pi))
    X = np.zeros((max_lag, len(terms)))
    for j, (base, power) in enumerate(terms):
        freq = power * np.log(base) / log_T
        for k in range(1, max_lag + 1):
            X[k-1, j] = np.cos(2*np.pi * k * freq)
    return X

# Single primes: (p, 1) for p in primes
single_terms = [(p, 1) for p in primes]

print('\n' + '='*70)
print('MODEL 1: Single primes only (5 terms)')
print('='*70)

for label, T, excess, ml in [('T~230', 229.3, excess_200, max_lag_low),
                               ('T~645', 645.0, excess_500h, max_lag_low),
                               ('T~2.7e11', T_base_e12, excess_e12, max_lag_high)]:
    X = build_design_matrix(T, ml, single_terms)
    amps, res, rank, sv = np.linalg.lstsq(X, excess, rcond=None)
    predicted = X @ amps
    ss_res = np.sum((excess - predicted)**2)
    ss_tot = np.sum(excess**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    print(f'  {label}: R2={R2:.4f}, amps={[f"{a:+.5f}" for a in amps]}')

# ============================================================
# MODEL 2: Primes + prime powers (p, p^2, p^3)
# ============================================================
print('\n' + '='*70)
print('MODEL 2: Primes + prime powers (p, p^2, p^3 for p=2,3,5)')
print('='*70)

power_terms = []
for p in [2, 3, 5]:
    for m in [1, 2, 3]:
        power_terms.append((p, m))
for p in [7, 11]:
    power_terms.append((p, 1))
# Total: 3*3 + 2 = 11 terms

print(f'Terms: {[(f"log({b}^{m})" if m>1 else f"log({b})") for b,m in power_terms]}')

for label, T, excess, ml in [('T~230', 229.3, excess_200, max_lag_low),
                               ('T~645', 645.0, excess_500h, max_lag_low),
                               ('T~2.7e11', T_base_e12, excess_e12, max_lag_high)]:
    X = build_design_matrix(T, ml, power_terms)
    amps, res, rank, sv = np.linalg.lstsq(X, excess, rcond=None)
    predicted = X @ amps
    ss_res = np.sum((excess - predicted)**2)
    ss_tot = np.sum(excess**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

    # Show prime power amplitudes
    amp_str = []
    for j, (b, m) in enumerate(power_terms):
        if abs(amps[j]) > 0.001 or m == 1:
            amp_str.append(f'{b}^{m}:{amps[j]:+.5f}')
    print(f'  {label}: R2={R2:.4f}')
    print(f'    {", ".join(amp_str)}')

# ============================================================
# MODEL 3: Cross-terms (products p1*p2)
# ============================================================
print('\n' + '='*70)
print('MODEL 3: Primes + cross-terms log(p1*p2)')
print('='*70)

cross_terms = [(p, 1) for p in primes]  # single primes
# Add cross products: frequency = log(p1*p2) = log(p1) + log(p2)
# In the explicit formula, these arise from the pair correlation
from itertools import combinations
for p1, p2 in combinations([2, 3, 5, 7], 2):
    # Product term: frequency = (log(p1) + log(p2)) / log(T/2pi)
    # Represent as (p1*p2, 1)
    cross_terms.append((p1*p2, 1))
# Also ratio terms: frequency = |log(p1) - log(p2)| / log(T/2pi)
# These are the BEAT frequencies themselves
# Represent differently: we need log(p1/p2) which isn't an integer base
# Use a special encoding
for p1, p2 in combinations([2, 3, 5, 7], 2):
    cross_terms.append((p1/p2, 1))  # ratio

print(f'Terms ({len(cross_terms)}):')
for b, m in cross_terms:
    if b == int(b):
        print(f'  log({int(b)})', end='')
    else:
        print(f'  log({b:.4f})', end='')
print()

def build_general_matrix(T, max_lag, terms):
    """Like build_design_matrix but handles non-integer bases."""
    log_T = np.log(T / (2*np.pi))
    X = np.zeros((max_lag, len(terms)))
    for j, (base, power) in enumerate(terms):
        freq = power * np.log(base) / log_T
        for k in range(1, max_lag + 1):
            X[k-1, j] = np.cos(2*np.pi * k * freq)
    return X

for label, T, excess, ml in [('T~230', 229.3, excess_200, max_lag_low),
                               ('T~645', 645.0, excess_500h, max_lag_low),
                               ('T~2.7e11', T_base_e12, excess_e12, max_lag_high)]:
    X = build_general_matrix(T, ml, cross_terms)
    amps, res, rank, sv = np.linalg.lstsq(X, excess, rcond=None)
    predicted = X @ amps
    ss_res = np.sum((excess - predicted)**2)
    ss_tot = np.sum(excess**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    print(f'  {label}: R2={R2:.4f}')

# ============================================================
# MODEL 4: Full explicit formula model
# ============================================================
print('\n' + '='*70)
print('MODEL 4: Full explicit formula (primes + powers + cross + ratios)')
print('='*70)

full_terms = []
# Single prime powers: log(p^m) for p <= 11, m <= 4
for p in [2, 3, 5, 7, 11]:
    for m in range(1, 5):
        full_terms.append((p**m, 1))
# Cross products: log(p1 * p2) for small primes
for p1, p2 in combinations([2, 3, 5, 7], 2):
    full_terms.append((p1*p2, 1))
# Ratios: log(p1/p2)
for p1, p2 in combinations([2, 3, 5, 7], 2):
    if p1 < p2:
        full_terms.append((p2/p1, 1))

print(f'Total terms: {len(full_terms)}')

for label, T, excess, ml in [('T~230', 229.3, excess_200, max_lag_low),
                               ('T~645', 645.0, excess_500h, max_lag_low),
                               ('T~2.7e11', T_base_e12, excess_e12, max_lag_high)]:
    X = build_general_matrix(T, ml, full_terms)
    amps, res, rank, sv = np.linalg.lstsq(X, excess, rcond=None)
    predicted = X @ amps
    ss_res = np.sum((excess - predicted)**2)
    ss_tot = np.sum(excess**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

    # Adjusted R2 (penalize for many parameters)
    n = ml
    p = len(full_terms)
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - p - 1) if n > p + 1 else R2
    print(f'  {label}: R2={R2:.4f}, R2_adj={R2_adj:.4f} (n={n}, p={p})')

# ============================================================
# KEY TEST: Does the prime power model explain the amplitude decay?
# ============================================================
print('\n' + '='*70)
print('KEY TEST: Amplitude decay across heights')
print('='*70)

print('\nFitting single-prime model at each height independently:')
single = [(p, 1) for p in [2, 3, 5, 7, 11]]

heights_data = [
    ('T~230', 229.3, excess_200, max_lag_low),
    ('T~645', 645.0, excess_500h, max_lag_low),
    ('T~2.7e11', T_base_e12, excess_e12, max_lag_high),
]

fitted_amps = {}
for label, T, excess, ml in heights_data:
    X = build_design_matrix(T, ml, single)
    amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    fitted_amps[label] = amps
    log_T = np.log(T / (2*np.pi))
    print(f'\n{label} (log(T/2pi)={log_T:.2f}):')
    for j, p in enumerate(primes):
        print(f'  p={p}: A={amps[j]:+.6f}')

# Check scaling law
print('\nAmplitude scaling: A(T) = A0 * f(T)')
print(f'{"Prime":<8} {"A(T~230)":>10} {"A(T~645)":>10} {"A(T~2.7e11)":>12} {"Ratio 645/230":>14} {"Ratio e11/230":>14}')
print('-'*70)
a230 = fitted_amps['T~230']
a645 = fitted_amps['T~645']
ae11 = fitted_amps['T~2.7e11']

log230 = np.log(229.3 / (2*np.pi))
log645 = np.log(645.0 / (2*np.pi))
loge11 = np.log(T_base_e12 / (2*np.pi))

for j, p in enumerate(primes):
    r645 = a645[j] / a230[j] if abs(a230[j]) > 1e-8 else float('nan')
    re11 = ae11[j] / a230[j] if abs(a230[j]) > 1e-8 else float('nan')
    print(f'  p={p:<4} {a230[j]:>+10.5f} {a645[j]:>+10.5f} {ae11[j]:>+12.6f} {r645:>14.4f} {re11:>14.4f}')

print(f'\nExpected ratio if A ~ 1/log(T):')
print(f'  645/230: {log230/log645:.4f}')
print(f'  e11/230: {log230/loge11:.4f}')

print(f'\nExpected ratio if A ~ 1/log(T)^2:')
print(f'  645/230: {(log230/log645)**2:.4f}')
print(f'  e11/230: {(log230/loge11)**2:.4f}')

print(f'\nExpected ratio if A ~ 1/log(T)^3:')
print(f'  645/230: {(log230/log645)**3:.4f}')
print(f'  e11/230: {(log230/loge11)**3:.4f}')

# ============================================================
# PRIME POWER INTERFERENCE TEST
# ============================================================
print('\n' + '='*70)
print('PRIME POWER INTERFERENCE: Do p^2, p^3 terms cancel p terms?')
print('='*70)

# At T~2.7e11, the explicit formula contribution from prime p at power m is:
# A_{p,m} ~ 1/(m * p^{m/2}) * cos(2*pi*k * m*log(p)/log(T/2pi))
# The total from prime p is: sum_m A_{p,m} cos(...)
# If the higher powers partially cancel the first, that explains amplitude decay

for T_label, T in [('T~230', 229.3), ('T~2.7e11', T_base_e12)]:
    log_T = np.log(T / (2*np.pi))
    print(f'\n{T_label}:')
    for p in [2, 3, 5]:
        print(f'  Prime p={p}:')
        total_contribution = 0
        for m in range(1, 6):
            weight = 1.0 / (m * p**(m/2))
            freq = m * np.log(p) / log_T
            # Average |cos| over all lags = contribution to ACF variance
            # More useful: contribution at a specific lag. Use lag=10 as example.
            k_test = 10
            contrib = weight * np.cos(2*np.pi * k_test * freq)
            total_contribution += contrib
            print(f'    m={m}: weight={weight:.6f}, freq={freq:.4f}, cos(k=10)={np.cos(2*np.pi*k_test*freq):+.4f}, contrib={contrib:+.8f}')
        print(f'    Total at lag 10: {total_contribution:+.8f}')
        print(f'    m=1 only:        {1.0/(1*p**0.5)*np.cos(2*np.pi*10*np.log(p)/log_T):+.8f}')
        cancel_pct = (1 - abs(total_contribution) / abs(1.0/(p**0.5)*np.cos(2*np.pi*10*np.log(p)/log_T))) * 100 if abs(1.0/(p**0.5)*np.cos(2*np.pi*10*np.log(p)/log_T)) > 1e-10 else 0
        print(f'    Cancellation: {cancel_pct:.1f}%')
