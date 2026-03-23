"""Fit the off-diagonal weight function from inverse spectral reconstruction.

Strategy:
1. Reconstruct H using Lowdin symmetric orthogonalization (preserves max RS structure)
2. Extract off-diagonal H_{jk} for all pairs
3. Regress on arithmetic features: gcd, lcm, Lambda, phi, sigma, etc.
4. The best-fitting function IS the weight for the zeta operator
5. Build H = diag(log k) + eps * f_best and verify r + p(GUE)
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr, kstest
from scipy.linalg import sqrtm, inv
from sympy import factorint, totient, mobius, isprime, divisor_sigma, primerange
import mpmath

t0 = time.time()
mpmath.mp.dps = 20

# ============================================================
# STEP 1: Lowdin orthogonalization of RS eigenvectors
# ============================================================
print('Step 1: Lowdin symmetric orthogonalization...')

N = 50
zeros_gamma = np.array([float(mpmath.zetazero(i + 1).imag) for i in range(N)])
print(f'  {N} zeros loaded')

# Build raw RS eigenvector matrix
def build_rs_raw(gammas, N_basis):
    U = np.zeros((len(gammas), N_basis), dtype=complex)
    for j in range(len(gammas)):
        theta = float(mpmath.siegeltheta(gammas[j]))
        for k in range(1, N_basis + 1):
            U[j, k-1] = np.exp(1j * (theta - gammas[j] * np.log(k))) / np.sqrt(k)
        U[j] /= np.linalg.norm(U[j])
    return U

U_raw = build_rs_raw(zeros_gamma, N)

# Lowdin: U_orth = U_raw * (U_raw^H * U_raw)^{-1/2}
S = U_raw @ U_raw.conj().T  # overlap matrix (N x N)
# S^{-1/2} via eigendecomposition (more stable than sqrtm)
eigvals_S, eigvecs_S = np.linalg.eigh(S)
eigvals_S = np.maximum(eigvals_S, 1e-15)  # regularize
S_inv_sqrt = eigvecs_S @ np.diag(eigvals_S ** (-0.5)) @ eigvecs_S.conj().T
U_lowdin = S_inv_sqrt @ U_raw

# Check orthogonality
UUd = U_lowdin @ U_lowdin.conj().T
orth_err = np.max(np.abs(UUd - np.eye(N)))
print(f'  Orthogonality error: {orth_err:.2e}')

# How much RS structure preserved?
r_struct, _ = pearsonr(np.abs(U_raw).flatten(), np.abs(U_lowdin).flatten())
print(f'  RS structure preserved: r = {r_struct:.4f}')
print(f'  (QR gave r = -0.028; Lowdin should be much better)')

# ============================================================
# STEP 2: Reconstruct H and extract off-diagonal
# ============================================================
print('\nStep 2: Reconstructing H...')

D = np.diag(zeros_gamma)
H = U_lowdin.conj().T @ D @ U_lowdin
H = (H + H.conj().T) / 2  # force Hermitian

# Verify eigenvalues
eigs_check = np.sort(np.linalg.eigvalsh(H))
err = np.max(np.abs(eigs_check - np.sort(zeros_gamma)))
print(f'  Eigenvalue error: {err:.2e}')

# Extract diagonal and off-diagonal
diag_H = np.real(np.diag(H))
log_k = np.log(np.arange(1, N + 1))
r_diag, _ = pearsonr(log_k, diag_H)
print(f'  Diagonal ~ log(k): r = {r_diag:+.4f}')

# ============================================================
# STEP 3: Fit the off-diagonal to arithmetic functions
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: FITTING THE OFF-DIAGONAL')
print('=' * 70)

# Collect all off-diagonal entries with their arithmetic context
j_vals, k_vals, h_vals = [], [], []
for j in range(N):
    for k in range(N):
        if j != k:
            j_vals.append(j + 1)
            k_vals.append(k + 1)
            h_vals.append(np.abs(H[j, k]))

j_arr = np.array(j_vals)
k_arr = np.array(k_vals)
h_arr = np.array(h_vals)

# Compute arithmetic features for each (j, k) pair
print(f'  Computing arithmetic features for {len(h_arr)} pairs...')

gcd_arr = np.array([np.gcd(j, k) for j, k in zip(j_vals, k_vals)])
lcm_arr = np.array([j * k // np.gcd(j, k) for j, k in zip(j_vals, k_vals)])

features = {}
features['1/sqrt(jk)'] = 1.0 / np.sqrt(j_arr * k_arr)
features['log(gcd+1)/sqrt(jk)'] = np.log(gcd_arr + 1) / np.sqrt(j_arr * k_arr)
features['gcd/sqrt(jk)'] = gcd_arr / np.sqrt(j_arr * k_arr)
features['sqrt(gcd)/sqrt(jk)'] = np.sqrt(gcd_arr) / np.sqrt(j_arr * k_arr)
features['log(gcd)/sqrt(jk)'] = np.log(np.maximum(gcd_arr, 1)) / np.sqrt(j_arr * k_arr)

# Von Mangoldt of gcd
def lambda_val(n):
    if n <= 1: return 0.0
    f = factorint(n)
    return np.log(list(f.keys())[0]) if len(f) == 1 else 0.0

features['Lambda(gcd)/sqrt(jk)'] = np.array([lambda_val(g) for g in gcd_arr]) / np.sqrt(j_arr * k_arr)

# Euler totient of gcd
features['phi(gcd)/sqrt(jk)'] = np.array([float(totient(max(g, 1))) for g in gcd_arr]) / np.sqrt(j_arr * k_arr)

# Divisor sigma of gcd
features['sigma(gcd)/sqrt(jk)'] = np.array([float(divisor_sigma(max(g, 1))) for g in gcd_arr]) / np.sqrt(j_arr * k_arr)

# Mobius squared of gcd (squarefree indicator)
features['mu^2(gcd)/sqrt(jk)'] = np.array([float(mobius(max(g, 1)))**2 for g in gcd_arr]) / np.sqrt(j_arr * k_arr)

# Divisibility indicator
features['div_indicator/sqrt(jk)'] = np.array([1.0 if j % k == 0 or k % j == 0 else 0.0
                                                 for j, k in zip(j_vals, k_vals)]) / np.sqrt(j_arr * k_arr)

# Number of common prime factors
def n_common_primes(a, b):
    g = np.gcd(a, b)
    if g <= 1: return 0
    return len(factorint(g))

features['n_common_primes/sqrt(jk)'] = np.array([n_common_primes(j, k)
                                                   for j, k in zip(j_vals, k_vals)]) / np.sqrt(j_arr * k_arr)

# log(j)*log(k)/sqrt(jk) — the "number operator squared" off-diagonal
features['log(j)*log(k)/sqrt(jk)'] = np.log(j_arr) * np.log(k_arr) / np.sqrt(j_arr * k_arr)

# Ramanujan sum: c_k(j) related
features['phi(gcd)/gcd/sqrt(jk)'] = np.array([float(totient(max(g, 1))) / max(g, 1)
                                                for g in gcd_arr]) / np.sqrt(j_arr * k_arr)

print(f'\n  {"Feature":<30} {"Pearson r":>10} {"p-value":>12} {"R^2":>8}')
print(f'  {"-"*65}')

results = []
for name, feat in sorted(features.items(), key=lambda x: -abs(pearsonr(x[1], h_arr)[0])):
    r, p = pearsonr(feat, h_arr)
    results.append((name, r, p, r**2))
    print(f'  {name:<30} {r:>+10.4f} {p:>12.2e} {r**2:>8.4f}')

# ============================================================
# STEP 4: Multivariate fit
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: MULTIVARIATE REGRESSION')
print('=' * 70)

# Use the top features in a linear regression
top_features = [name for name, r, p, r2 in results if abs(r) > 0.1][:6]
print(f'  Using top features: {top_features}')

X = np.column_stack([features[name] for name in top_features])
# Standardize
X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

from numpy.linalg import lstsq
beta, _, _, _ = lstsq(X_std, h_arr, rcond=None)
pred = X_std @ beta
R2_multi = 1 - np.sum((h_arr - pred)**2) / np.sum((h_arr - np.mean(h_arr))**2)
R2_adj = 1 - (1 - R2_multi) * (len(h_arr) - 1) / (len(h_arr) - len(top_features) - 1)

print(f'  R^2 = {R2_multi:.4f}, R^2_adj = {R2_adj:.4f}')
print(f'\n  Coefficients:')
for name, b in zip(top_features, beta):
    print(f'    {name:<30} {b:>+10.4f}')

# ============================================================
# STEP 5: Build the best operator and test
# ============================================================
print('\n' + '=' * 70)
print('STEP 5: BUILD AND TEST THE BEST OPERATOR')
print('=' * 70)

# Use the single best feature as the weight function
best_name, best_r, _, _ = results[0]
print(f'  Best single feature: {best_name} (r = {best_r:+.4f})')

def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s ** 2 / 4)

def measure_all(eigs_raw):
    from riemann.analysis.bost_connes_operator import polynomial_unfold
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20: return 0, 0
    sp = sp / np.mean(sp)
    n_trim = int(0.1 * len(eigs))
    eigs_trim = eigs[n_trim:-n_trim]
    log_peaks, gaps = [], []
    for i in range(min(len(sp), len(eigs_trim) - 1)):
        z_mid = (eigs_trim[i] + eigs_trim[i+1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        log_peaks.append(log_det)
        gaps.append(sp[i])
    gaps = np.array(gaps)
    log_peaks = np.array(log_peaks)
    if len(gaps) < 10: return 0, 0
    r, _ = pearsonr(gaps, log_peaks)
    ks, p_gue = kstest(sp, wigner_cdf)
    return r, p_gue

# Build operators with each top feature as weight, sweep epsilon
N_op = 400
D_op = np.diag(np.log(np.arange(1, N_op + 1, dtype=float) + 1))
D_norm = np.linalg.norm(D_op, 'fro')

print(f'\n  Testing top weight functions at N={N_op}:')
print(f'  {"Weight":<30} {"best_eps":>8} {"r":>8} {"p(GUE)":>8}')
print(f'  {"-"*58}')

# Test the top 5 features
for feat_name, feat_r, _, _ in results[:5]:
    # Build the weight matrix
    W = np.zeros((N_op, N_op))
    for j in range(1, N_op + 1):
        for k in range(j + 1, N_op + 1):
            g = np.gcd(j, k)
            if 'log(gcd+1)' in feat_name:
                val = np.log(g + 1) / np.sqrt(j * k)
            elif 'sqrt(gcd)' in feat_name:
                val = np.sqrt(g) / np.sqrt(j * k)
            elif 'phi(gcd)/sqrt' == feat_name[:14]:
                val = float(totient(max(g, 1))) / np.sqrt(j * k)
            elif 'Lambda(gcd)' in feat_name:
                val = lambda_val(g) / np.sqrt(j * k)
            elif 'sigma(gcd)' in feat_name:
                val = float(divisor_sigma(max(g, 1))) / np.sqrt(j * k)
            elif 'gcd/sqrt' in feat_name and 'log' not in feat_name:
                val = g / np.sqrt(j * k)
            elif 'log(j)*log(k)' in feat_name:
                val = np.log(j) * np.log(k) / np.sqrt(j * k)
            elif 'div_indicator' in feat_name:
                val = (1.0 if j % k == 0 or k % j == 0 else 0.0) / np.sqrt(j * k)
            else:
                val = np.log(g + 1) / np.sqrt(j * k)  # fallback
            W[j-1, k-1] = val
            W[k-1, j-1] = val

    W_norm = np.linalg.norm(W, 'fro')
    W_sc = W * (D_norm / W_norm)

    # Sweep epsilon
    best_eps, best_score = 0, -1
    best_r_val, best_p_val = 0, 0
    for eps in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
        H_test = D_op + eps * W_sc
        eigs_test = np.linalg.eigvalsh(H_test)
        r_test, p_test = measure_all(eigs_test)
        score = r_test * min(p_test, 0.5) if p_test > 0.01 else 0
        if score > best_score:
            best_score = score
            best_eps = eps
            best_r_val = r_test
            best_p_val = p_test

    tag = ' <-- BOTH' if best_r_val > 0.3 and best_p_val > 0.05 else ''
    print(f'  {feat_name:<30} {best_eps:>8.2f} {best_r_val:>+8.4f} {best_p_val:>8.4f}{tag}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

print(f'\n  Inverse spectral reconstruction reveals:')
print(f'  - Diagonal: H_kk ~ log(k) (r = {r_diag:+.3f})')
print(f'  - Off-diagonal best correlate: {results[0][0]} (r = {results[0][1]:+.3f})')
print(f'  - Multivariate R^2 = {R2_multi:.3f} from top {len(top_features)} features')

print(f'\nTotal time: {time.time() - t0:.1f}s')
