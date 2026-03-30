"""Fourier-domain spectral surgery on GUE spacing sequences."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from scipy import stats
from riemann.analysis.bost_connes_operator import (
    polynomial_unfold, spacing_autocorrelation, gue_reference_autocorrelation
)

N = 200
T = 229.3
max_lag = 15
n_trials = 500
primes = [2, 3, 5, 7, 11]
thetas = {p: np.log(p) / np.log(T / (2*np.pi)) for p in primes}
anomalous = [4, 7, 10, 11]

# Load zeta zero spacings
zeros_200 = np.load('_zeros_200.npy')
zero_sp_raw = np.diff(np.sort(zeros_200))
t_mid = (zeros_200[:-1] + zeros_200[1:]) / 2
ld = np.log(t_mid / (2*np.pi)) / (2*np.pi)
zero_sp = zero_sp_raw * ld / np.mean(zero_sp_raw * ld)
zero_acf = spacing_autocorrelation(zero_sp, max_lag)

gue_acf = gue_reference_autocorrelation(n_matrix=200, n_matrices=500, max_lag=max_lag, seed=42)
target_excess = zero_acf[1:max_lag+1] - gue_acf[1:max_lag+1]
baseline_L2 = np.sqrt(np.sum(target_excess**2))

print('='*70)
print('FOURIER-DOMAIN SPECTRAL SURGERY')
print('='*70)

# Generate GUE spacing sequences
rng = np.random.default_rng(42)
gue_spacing_sets = []
for _ in range(n_trials):
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (A + A.conj().T) / (2 * np.sqrt(2 * N))
    eigs = np.linalg.eigvalsh(H)
    sp = polynomial_unfold(eigs)
    gue_spacing_sets.append(sp)
print(f'Generated {n_trials} GUE spacing sets, ~{len(gue_spacing_sets[0])} spacings each')


def fourier_surgery(spacings, boost_dict):
    """Modify power spectrum at specific frequencies."""
    n = len(spacings)
    S = np.fft.rfft(spacings - np.mean(spacings))
    freqs = np.fft.rfftfreq(n)
    for target_freq, factor in boost_dict.items():
        idx = np.argmin(np.abs(freqs - target_freq))
        S[idx] *= factor
        if idx > 0:
            S[idx-1] *= (1 + (factor-1)*0.3)
        if idx < len(S)-1:
            S[idx+1] *= (1 + (factor-1)*0.3)
    modified = np.fft.irfft(S, n=n) + np.mean(spacings)
    modified = np.maximum(modified, 0.01)
    modified = modified / np.mean(modified)
    return modified


def test_surgery(boost_dict):
    """Evaluate a surgery prescription."""
    acfs = []
    all_sp = []
    for sp in gue_spacing_sets:
        sp_mod = fourier_surgery(sp, boost_dict)
        acfs.append(spacing_autocorrelation(sp_mod, max_lag))
        all_sp.extend(sp_mod)
    mean_acf = np.mean(acfs, axis=0)
    excess = mean_acf[1:max_lag+1] - gue_acf[1:max_lag+1]
    L2 = np.sqrt(np.sum((excess - target_excess)**2))
    imp = 100*(1 - L2/baseline_L2)
    ks_stat, ks_p = stats.ks_2samp(np.array(all_sp), zero_sp)
    signs = sum(1 for k in anomalous if (excess[k-1] > 0) == (target_excess[k-1] > 0))
    return L2, imp, signs, ks_p, excess


# Uniform boost
print('\n--- Uniform boost at all prime frequencies ---')
for boost in [1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]:
    bd = {thetas[p]: boost for p in primes}
    L2, imp, signs, ks_p, exc = test_surgery(bd)
    print(f'  boost={boost:.2f}: L2={L2:.4f} ({imp:+.1f}%), signs={signs}/4, KS_p={ks_p:.4f}')

# Weighted boost (explicit formula weights)
print('\n--- Explicit-formula weighted boost ---')
ef_w = {p: 1.0/(np.sqrt(p)*np.log(p)) for p in primes}
mx = max(ef_w.values())
ef_w = {p: w/mx for p, w in ef_w.items()}
for base in [1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
    bd = {thetas[p]: 1.0 + (base-1.0)*ef_w[p] for p in primes}
    L2, imp, signs, ks_p, exc = test_surgery(bd)
    print(f'  base={base:.1f}: L2={L2:.4f} ({imp:+.1f}%), signs={signs}/4, KS_p={ks_p:.4f}')

# Mixed boost/attenuate
print('\n--- Mixed boost/attenuate (best combos) ---')
best_L2 = baseline_L2
best_config = None
best_exc = None
for b2 in [1.5, 2.0, 3.0, 5.0, 8.0]:
    for b3 in [1.0, 1.5, 2.0, 3.0]:
        for b5 in [0.3, 0.5, 0.8, 1.0]:
            bd = {thetas[2]: b2, thetas[3]: b3, thetas[5]: b5,
                  thetas[7]: 1.2, thetas[11]: 0.8}
            L2, imp, signs, ks_p, exc = test_surgery(bd)
            if L2 < best_L2:
                best_L2 = L2
                best_config = (b2, b3, b5)
                best_exc = exc
                best_ks = ks_p
                best_signs = signs
                print(f'  NEW BEST: b2={b2},b3={b3},b5={b5}: L2={L2:.4f} ({imp:+.1f}%), signs={signs}/4, KS_p={ks_p:.4f}')

print(f'\nBest config: b2={best_config[0]}, b3={best_config[1]}, b5={best_config[2]}')
print(f'L2={best_L2:.4f} (improvement: {100*(1-best_L2/baseline_L2):+.1f}%)')
print(f'Signs: {best_signs}/4, KS_p: {best_ks:.4f}')

# Detailed ACF
print(f'\n{"Lag":<5} {"Target":>8} {"Surgery":>8} {"Residual":>8} Flag')
print('-'*40)
for k in range(1, max_lag+1):
    flag = ' ***' if k in anomalous else ''
    t = target_excess[k-1]
    m = best_exc[k-1]
    print(f'{k:<5} {t:>+8.4f} {m:>+8.4f} {m-t:>+8.4f}{flag}')
