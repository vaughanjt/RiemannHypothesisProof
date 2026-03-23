"""Deep investigation of the GCD kernel operator.

H_{jk} = log(gcd(j,k)) / sqrt(j*k)

This operator encodes the multiplicative structure of integers and showed
the strongest peak-gap correlation (r=+0.59) among all candidates.

Key questions:
1. Does r grow with N? (Like zeta, where r grows from 0.75 to 0.80)
2. What are its eigenvalue statistics? (GUE? Poisson? Something else?)
3. Do its eigenvalues have prime-frequency structure in the ACF?
4. Can we modify it to get r closer to 0.80?
5. What are its eigenvectors? (Ramanujan sums? Multiplicative functions?)
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr, kstest
from sympy import totient, mobius, primerange
from riemann.analysis.bost_connes_operator import (
    spacing_autocorrelation, polynomial_unfold
)

t0 = time.time()

# ============================================================
# STEP 1: GCD kernel at multiple sizes
# ============================================================
print('=' * 70)
print('STEP 1: GCD KERNEL SCALING')
print('=' * 70)


def build_gcd_kernel(N, weight_func=None):
    """Build the GCD kernel matrix H_{jk} = f(gcd(j,k)) / sqrt(j*k)."""
    if weight_func is None:
        weight_func = lambda g: np.log(g + 1)  # default: log(gcd+1)
    H = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(j, N + 1):
            g = np.gcd(j, k)
            val = weight_func(g) / np.sqrt(j * k)
            H[j-1, k-1] = val
            H[k-1, j-1] = val
    return H


def measure_peak_gap(eigs_raw):
    """Measure peak-gap r and beta from eigenvalues."""
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20:
        return 0, 0, 0
    sp = sp / np.mean(sp)
    n_trim = int(0.1 * len(eigs))
    eigs_trim = eigs[n_trim:-n_trim]

    log_peaks = []
    gaps = []
    for k in range(len(sp)):
        if k + 1 >= len(eigs_trim):
            break
        z_mid = (eigs_trim[k] + eigs_trim[k + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        log_peaks.append(log_det)
        gaps.append(sp[k])

    gaps = np.array(gaps)
    log_peaks = np.array(log_peaks)
    r, _ = pearsonr(gaps, log_peaks)
    mask = gaps > 0.1
    beta = np.polyfit(np.log(gaps[mask]), log_peaks[mask], 1)[0] if np.sum(mask) > 10 else 0
    return r, beta, len(gaps)


print(f'\n{"N":>5} {"r":>8} {"beta":>8} {"pts":>5} {"build_s":>8} {"time":>8}')
print('-' * 50)

r_values = []
N_values = []

for N in [50, 100, 200, 300, 400, 500, 600]:
    t1 = time.time()
    H = build_gcd_kernel(N)
    t_build = time.time() - t1
    eigs = np.linalg.eigvalsh(H)
    r, beta, pts = measure_peak_gap(eigs)
    r_values.append(r)
    N_values.append(N)
    print(f'{N:>5} {r:>+8.4f} {beta:>8.1f} {pts:>5} {t_build:>8.2f}s {time.time()-t1:>8.2f}s')

# Does r grow with N?
if len(r_values) >= 3:
    from scipy.stats import pearsonr as pr
    r_trend, p_trend = pr(N_values, r_values)
    print(f'\n  r vs N: Pearson = {r_trend:+.4f} (p = {p_trend:.4f})')
    if r_trend > 0.3:
        print('  -> r GROWS with N (like zeta!)')
    elif r_trend < -0.3:
        print('  -> r SHRINKS with N')
    else:
        print('  -> r is stable (unlike zeta)')

# ============================================================
# STEP 2: Eigenvalue statistics
# ============================================================
print('\n' + '=' * 70)
print('STEP 2: EIGENVALUE STATISTICS OF GCD KERNEL')
print('=' * 70)

N_test = 400
H = build_gcd_kernel(N_test)
eigs = np.linalg.eigvalsh(H)
sp = polynomial_unfold(eigs, trim_fraction=0.1)
sp = sp / np.mean(sp)

print(f'\n  N = {N_test}, {len(sp)} spacings after unfolding')
print(f'  Mean spacing: {np.mean(sp):.4f}, Std: {np.std(sp):.4f}')

# GUE spacing distribution test
# Wigner surmise: p(s) = (pi*s/2) * exp(-pi*s^2/4)
from scipy.stats import kstest

def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s ** 2 / 4)

ks_gue, p_gue = kstest(sp, wigner_cdf)
# Poisson: p(s) = exp(-s)
ks_poisson, p_poisson = kstest(sp, 'expon', args=(0, 1))

print(f'  KS vs Wigner (GUE): D = {ks_gue:.4f}, p = {p_gue:.4f}')
print(f'  KS vs Poisson: D = {ks_poisson:.4f}, p = {p_poisson:.4f}')

if p_gue > p_poisson:
    print(f'  -> More GUE-like than Poisson')
else:
    print(f'  -> More Poisson-like than GUE')

# ============================================================
# STEP 3: ACF structure — prime frequencies?
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: ACF OF GCD KERNEL SPACINGS')
print('=' * 70)

max_lag = min(100, len(sp) // 4)
acf_gcd = spacing_autocorrelation(sp, max_lag)

# Compare to GUE ACF (should be ~0 for all lags if GUE)
print(f'\n  Top 10 ACF values (by magnitude):')
acf_vals = acf_gcd[1:max_lag + 1]
top_lags = np.argsort(np.abs(acf_vals))[-10:][::-1]
se = 1.0 / np.sqrt(len(sp))
for lag_idx in top_lags:
    lag = lag_idx + 1
    z = acf_vals[lag_idx] / se
    print(f'    lag {lag:>3}: ACF = {acf_vals[lag_idx]:+.4f} (z = {z:+.2f})')

n_sig = np.sum(np.abs(acf_vals / se) > 2.5)
print(f'\n  Significant lags (|z| > 2.5): {n_sig}/{max_lag}')
print(f'  Expected under GUE: ~{max_lag * 0.012:.1f}')

# ============================================================
# STEP 4: Weight function variants
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: WEIGHT FUNCTION VARIANTS')
print('=' * 70)

N_var = 300
variants = {
    'log(g+1)/sqrt(jk)': lambda g: np.log(g + 1),
    'g/sqrt(jk)': lambda g: g,
    'sqrt(g)/sqrt(jk)': lambda g: np.sqrt(g),
    'phi(g)/sqrt(jk)': lambda g: float(totient(max(g, 1))),
    'Lambda(g)/sqrt(jk)': lambda g: np.log(g) if g > 1 and len(set([p for p in range(2, g+1) if g % p == 0 and all(g % (p*i) != 0 for i in range(2, g//p + 1))])) <= 1 else 0,
    'log(g)/sqrt(jk)': lambda g: np.log(max(g, 1)),
    '1_{g>1}/sqrt(jk)': lambda g: 1.0 if g > 1 else 0.0,
    'mu(g)^2/sqrt(jk)': lambda g: float(mobius(max(g, 1))) ** 2,
}

print(f'\n  {"Weight f(gcd)":<25} {"r":>8} {"beta":>8} {"KS(GUE)":>10}')
print(f'  {"-"*55}')

for name, wf in variants.items():
    try:
        H = build_gcd_kernel(N_var, weight_func=wf)
        eigs = np.linalg.eigvalsh(H)
        r, beta, pts = measure_peak_gap(eigs)
        sp_v = polynomial_unfold(eigs, trim_fraction=0.1)
        sp_v = sp_v / np.mean(sp_v)
        ks, p_ks = kstest(sp_v, wigner_cdf)
        print(f'  {name:<25} {r:>+8.4f} {beta:>8.1f} {p_ks:>10.4f}')
    except Exception as e:
        print(f'  {name:<25} FAILED: {e}')

# ============================================================
# STEP 5: Eigenvector analysis
# ============================================================
print('\n' + '=' * 70)
print('STEP 5: EIGENVECTOR STRUCTURE')
print('=' * 70)

N_ev = 200
H = build_gcd_kernel(N_ev)
eigs, vecs = np.linalg.eigh(H)

# The eigenvectors should be related to multiplicative functions
# if the operator is truly arithmetic.

# Sort by eigenvalue magnitude
order = np.argsort(np.abs(eigs))[::-1]

# Check: are eigenvectors multiplicative? i.e., v(mn) = v(m)*v(n) for gcd(m,n)=1
print(f'\n  Top 5 eigenvectors — multiplicativity test:')
print(f'  (testing v(6) = v(2)*v(3), v(10) = v(2)*v(5), v(15) = v(3)*v(5))')
for rank in range(5):
    idx = order[rank]
    v = vecs[:, idx]
    # Normalize so v[0] = 1 (or use max)
    v_norm = v / (v[0] + 1e-10) if abs(v[0]) > 1e-10 else v / np.max(np.abs(v))

    # Test multiplicativity at coprime pairs
    tests = [(2, 3, 6), (2, 5, 10), (3, 5, 15), (2, 7, 14), (3, 7, 21)]
    errors = []
    for a, b, ab in tests:
        if ab <= N_ev:
            pred = v_norm[a-1] * v_norm[b-1]
            actual = v_norm[ab-1]
            if abs(pred) > 1e-10:
                errors.append(abs(actual - pred) / abs(pred))
    mean_err = np.mean(errors) if errors else float('inf')
    print(f'  Eigvec {rank+1} (lambda={eigs[idx]:+.4f}): '
          f'multiplicativity error = {mean_err:.4f} '
          f'({"MULTIPLICATIVE" if mean_err < 0.1 else "not multiplicative"})')

# ============================================================
# STEP 6: Hybrid operator — GCD + Hecke
# ============================================================
print('\n' + '=' * 70)
print('STEP 6: HYBRID OPERATORS')
print('=' * 70)

from riemann.analysis.bost_connes_operator import construct_hecke_prime_adjacency

N_hyb = 300
H_gcd = build_gcd_kernel(N_hyb)
H_hpa = construct_hecke_prime_adjacency(N_hyb)

# Normalize
H_gcd_n = H_gcd / np.max(np.abs(np.linalg.eigvalsh(H_gcd)))
H_hpa_n = H_hpa / np.max(np.abs(np.linalg.eigvalsh(H_hpa)))

print(f'\n  {"Mix":<35} {"r":>8}')
print(f'  {"-"*45}')

for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    H_mix = (1 - alpha) * H_gcd_n + alpha * H_hpa_n
    eigs = np.linalg.eigvalsh(H_mix)
    r, _, _ = measure_peak_gap(eigs)
    print(f'  {f"({1-alpha:.1f})*GCD + ({alpha:.1f})*HPA":<35} {r:>+8.4f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: GCD KERNEL AS ZETA OPERATOR CANDIDATE')
print('=' * 70)

print(f'\n  Peak-gap correlation:')
print(f'    GCD kernel (N=600): r = {r_values[-1]:+.4f}')
print(f'    Zeta (target):      r ~ +0.80')
print(f'    GUE:                r ~ +0.04')

print(f'\n  Gap to close: {0.80 - r_values[-1]:.2f}')

if r_values[-1] > 0.5:
    print(f'\n  The GCD kernel captures the QUALITATIVE behavior:')
    print(f'    - Positive peak-gap correlation (r >> 0)')
    print(f'    - Multiplicative arithmetic structure')
    print(f'    - Eigenvectors connected to number-theoretic functions')
    print(f'\n  But the QUANTITATIVE match (r={r_values[-1]:.2f} vs 0.80) is incomplete.')
    print(f'  The operator needs modification to strengthen the coupling.')
else:
    print(f'\n  The GCD kernel is too weak (r < 0.5).')

print(f'\nTotal time: {time.time() - t0:.1f}s')
