"""Spectral surgery: inject prime beat frequencies into GUE eigenvalues.
Fast version: 100 trials, targeted sweep instead of Nelder-Mead."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy import linalg, stats
from riemann.analysis.bost_connes_operator import (
    construct_hecke_prime_adjacency, polynomial_unfold, spacing_autocorrelation
)

N = 200
T = 229.3
max_lag = 15
n_trials = 100
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

# Pre-generate GUE matrices
rng = np.random.default_rng(42)
print('Pre-generating GUE matrices...')
t0 = time.time()
gue_matrices = []
for i in range(n_trials):
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (A + A.conj().T) / (2 * np.sqrt(2 * N))
    gue_matrices.append(H)
print(f'  Done in {time.time()-t0:.1f}s')

# GUE baseline ACF
print('Computing GUE baseline...')
gue_acfs = []
for H in gue_matrices:
    eigs = np.linalg.eigvalsh(H)
    sp = polynomial_unfold(eigs)
    gue_acfs.append(spacing_autocorrelation(sp, max_lag))
gue_acf = np.mean(gue_acfs, axis=0)
target_excess = zero_acf[1:max_lag+1] - gue_acf[1:max_lag+1]
baseline_L2 = np.sqrt(np.sum(target_excess**2))
print(f'Baseline L2: {baseline_L2:.4f}')

# Helper: evaluate a perturbation
def evaluate(perturbation_fn, label=''):
    """Apply perturbation_fn(H_gue) -> H, compute mean ACF."""
    acfs = []
    all_sp = []
    for H_gue in gue_matrices:
        H = perturbation_fn(H_gue)
        eigs = np.linalg.eigvalsh(H)
        sp = polynomial_unfold(eigs)
        if len(sp) > max_lag:
            acfs.append(spacing_autocorrelation(sp, max_lag))
            all_sp.extend(sp)
    mean_acf = np.mean(acfs, axis=0)
    excess = mean_acf[1:max_lag+1] - gue_acf[1:max_lag+1]
    L2 = np.sqrt(np.sum((excess - target_excess)**2))
    imp = 100*(1 - L2/baseline_L2)
    ks_stat, ks_p = stats.ks_2samp(np.array(all_sp), zero_sp)
    signs = sum(1 for k in anomalous if (excess[k-1] > 0) == (target_excess[k-1] > 0))
    return {'acf': mean_acf, 'excess': excess, 'L2': L2, 'imp': imp,
            'ks_p': ks_p, 'signs': signs, 'spacings': np.array(all_sp)}

H_hpa = construct_hecke_prime_adjacency(N)
hpa_norm = np.linalg.norm(H_hpa, 'fro')

# ============================================================
# APPROACH 1: Diagonal prime-frequency potential
# ============================================================
print('\n' + '='*70)
print('APPROACH 1: H_GUE + eps * diag(V_prime)')
print('='*70)

results_diag = {}
for eps in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
    V = np.zeros(N)
    for j, p in enumerate(primes):
        w = 1.0 / (np.sqrt(p) * np.log(p))
        V += eps * w * np.cos(2*np.pi * np.arange(N) * thetas[p])
    D = np.diag(V)
    r = evaluate(lambda H, D=D: H + D, f'diag eps={eps}')
    results_diag[eps] = r
    print(f'  eps={eps:.2f}: L2={r["L2"]:.4f} ({r["imp"]:+.1f}%), signs={r["signs"]}/4, KS_p={r["ks_p"]:.4f}')

best_diag = min(results_diag.items(), key=lambda x: x[1]['L2'])
print(f'  Best: eps={best_diag[0]}')

# ============================================================
# APPROACH 2: Ultra-weak HPA
# ============================================================
print('\n' + '='*70)
print('APPROACH 2: H_GUE + eps * H_HPA')
print('='*70)

results_hpa = {}
for eps in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    Ha = eps * H_hpa / hpa_norm
    r = evaluate(lambda H, Ha=Ha: H + Ha, f'hpa eps={eps}')
    results_hpa[eps] = r
    print(f'  eps={eps:.3f}: L2={r["L2"]:.4f} ({r["imp"]:+.1f}%), signs={r["signs"]}/4, KS_p={r["ks_p"]:.4f}')

best_hpa = min(results_hpa.items(), key=lambda x: x[1]['L2'])
print(f'  Best: eps={best_hpa[0]}')

# ============================================================
# APPROACH 3: Combined (best diag + best hpa)
# ============================================================
print('\n' + '='*70)
print('APPROACH 3: Combined diagonal + HPA sweep')
print('='*70)

# Fine grid around best values
eps_d_best = best_diag[0]
eps_h_best = best_hpa[0]

results_combined = {}
best_combined = None
best_combined_L2 = float('inf')

for eps_d_mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
    for eps_h_mult in [0.0, 0.5, 1.0, 1.5, 2.0]:
        eps_d = eps_d_best * eps_d_mult
        eps_h = eps_h_best * eps_h_mult

        V = np.zeros(N)
        for j, p in enumerate(primes):
            w = 1.0 / (np.sqrt(p) * np.log(p))
            V += eps_d * w * np.cos(2*np.pi * np.arange(N) * thetas[p])
        D = np.diag(V)
        Ha = eps_h * H_hpa / hpa_norm

        r = evaluate(lambda H, D=D, Ha=Ha: H + D + Ha)
        key = (eps_d, eps_h)
        results_combined[key] = r

        if r['L2'] < best_combined_L2:
            best_combined_L2 = r['L2']
            best_combined = key

bc = results_combined[best_combined]
print(f'Best combined: eps_d={best_combined[0]:.3f}, eps_h={best_combined[1]:.4f}')
print(f'  L2={bc["L2"]:.4f} ({bc["imp"]:+.1f}%), signs={bc["signs"]}/4, KS_p={bc["ks_p"]:.4f}')

# ============================================================
# APPROACH 4: Rank-1 prime perturbation
# ============================================================
print('\n' + '='*70)
print('APPROACH 4: H_GUE + eps * v*v^T (prime-weighted vector)')
print('='*70)

# v_i = sum_p log(p)/p if p|i, else 0
# This is Mangoldt-like: weights integers by their prime factors
for rank_type in ['mangoldt', 'prime_cos']:
    results_rank1 = {}
    for eps in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        if rank_type == 'mangoldt':
            v = np.zeros(N)
            from sympy import factorint
            for i in range(1, N+1):
                factors = factorint(i)
                for p, k in factors.items():
                    v[i-1] += np.log(p) / p
            v = v / np.linalg.norm(v)
        else:
            v = np.zeros(N)
            for j, p in enumerate(primes):
                w = 1.0 / (np.sqrt(p) * np.log(p))
                v += w * np.cos(2*np.pi * np.arange(N) * thetas[p])
            v = v / np.linalg.norm(v)

        R1 = eps * np.outer(v, v)
        r = evaluate(lambda H, R1=R1: H + R1)
        results_rank1[eps] = r
        print(f'  {rank_type} eps={eps:.3f}: L2={r["L2"]:.4f} ({r["imp"]:+.1f}%), signs={r["signs"]}/4, KS_p={r["ks_p"]:.4f}')
    print()

# ============================================================
# FINAL COMPARISON
# ============================================================
print('='*70)
print('FINAL COMPARISON')
print('='*70)

bd = best_diag[1]
bh = best_hpa[1]

print(f'{"Model":<30} {"L2":>7} {"Imp%":>7} {"Signs":>6} {"KS_p":>7} {"Dist+ACF?":>10}')
print('-'*70)
print(f'{"Pure GUE":<30} {baseline_L2:>7.4f} {"---":>7} {"---":>6} {"---":>7} {"---":>10}')
print(f'{"Diag potential (eps="+str(best_diag[0])+")":<30} {bd["L2"]:>7.4f} {bd["imp"]:>+6.1f}% {bd["signs"]:>4}/4 {bd["ks_p"]:>7.4f} {"YES" if bd["ks_p"]>0.05 and bd["imp"]>10 else "no":>10}')
print(f'{"Ultra-weak HPA (eps="+str(best_hpa[0])+")":<30} {bh["L2"]:>7.4f} {bh["imp"]:>+6.1f}% {bh["signs"]:>4}/4 {bh["ks_p"]:>7.4f} {"YES" if bh["ks_p"]>0.05 and bh["imp"]>10 else "no":>10}')
print(f'{"Combined":<30} {bc["L2"]:>7.4f} {bc["imp"]:>+6.1f}% {bc["signs"]:>4}/4 {bc["ks_p"]:>7.4f} {"YES" if bc["ks_p"]>0.05 and bc["imp"]>10 else "no":>10}')

# Detailed ACF for best model
best_all = min([(bd, 'Diagonal'), (bh, 'HPA'), (bc, 'Combined')], key=lambda x: x[0]['L2'])
best_r, best_name = best_all

print(f'\nBest model: {best_name}')
print(f'{"Lag":<5} {"Target":>8} {"Model":>8} {"Residual":>8} Flag')
print('-'*40)
for k in range(1, max_lag+1):
    flag = ' ***' if k in anomalous else ''
    t = target_excess[k-1]
    m = best_r['excess'][k-1]
    print(f'{k:<5} {t:>+8.4f} {m:>+8.4f} {m-t:>+8.4f}{flag}')
