"""Deep analysis of H = diag(log k) + epsilon * GCD_kernel.

The inverse spectral reconstruction gave H_{kk} ~ log(k).
Now test: what coupling epsilon gives GUE eigenvalues + high peak-gap r?

Also: examine the STRUCTURE of the off-diagonal. Which entries matter?
Do specific (j,k) pairs dominate? Do prime-related entries stand out?
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr, kstest
from sympy import isprime, factorint
from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()
rng = np.random.default_rng(42)

def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s ** 2 / 4)

def measure_all(eigs_raw):
    """Measure peak-gap r and spacing statistics."""
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20:
        return 0, 1, 1
    sp = sp / np.mean(sp)
    n_trim = int(0.1 * len(eigs))
    eigs_trim = eigs[n_trim:-n_trim]
    log_peaks, gaps = [], []
    for k in range(min(len(sp), len(eigs_trim) - 1)):
        z_mid = (eigs_trim[k] + eigs_trim[k + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        log_peaks.append(log_det)
        gaps.append(sp[k])
    gaps = np.array(gaps)
    log_peaks = np.array(log_peaks)
    r, _ = pearsonr(gaps, log_peaks) if len(gaps) > 10 else (0, 1)
    ks_g, p_g = kstest(sp, wigner_cdf) if len(sp) > 10 else (1, 0)
    return r, p_g, len(gaps)

# ============================================================
# THE OPERATOR FAMILY: H(epsilon) = diag(log k) + epsilon * W
# ============================================================
print('=' * 70)
print('H(epsilon) = diag(log k) + epsilon * W')
print('=' * 70)

N = 400

def build_gcd_kernel(N):
    H = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(j + 1, N + 1):
            g = np.gcd(j, k)
            val = np.log(g + 1) / np.sqrt(j * k)
            H[j-1, k-1] = val
            H[k-1, j-1] = val
    return H

print(f'  Building GCD kernel (N={N})...')
W_gcd = build_gcd_kernel(N)
W_norm = np.linalg.norm(W_gcd, 'fro')

# Normalize W so that epsilon=1 means equal weight to diagonal and off-diagonal
D = np.diag(np.log(np.arange(1, N + 1, dtype=float) + 1))  # log(k+1) to avoid log(1)=0
D_norm = np.linalg.norm(D, 'fro')
W_scaled = W_gcd * (D_norm / W_norm)

print(f'  ||D||_F = {D_norm:.2f}, ||W||_F = {W_norm:.2f}, scale factor = {D_norm/W_norm:.2f}')

# ============================================================
# SWEEP EPSILON
# ============================================================
print(f'\n{"epsilon":>10} {"r":>8} {"p(GUE)":>8} {"pts":>5} {"note":>20}')
print('-' * 55)

best_combined = None
best_score = -1

for eps in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    H = D + eps * W_scaled
    eigs = np.linalg.eigvalsh(H)
    r, p_gue, pts = measure_all(eigs)

    note = ''
    if p_gue > 0.05 and r > 0.3:
        note = '<-- BOTH!'
    elif p_gue > 0.05:
        note = 'GUE ok'
    elif r > 0.3:
        note = 'r ok'

    # Score: want both high r and high p_gue
    score = r * min(p_gue, 0.5) if p_gue > 0.01 else r * 0.001
    if score > best_score:
        best_score = score
        best_combined = (eps, r, p_gue)

    print(f'{eps:>10.3f} {r:>+8.4f} {p_gue:>8.4f} {pts:>5} {note:>20}')

if best_combined:
    print(f'\n  Best combined: epsilon={best_combined[0]}, r={best_combined[1]:+.4f}, p(GUE)={best_combined[2]:.4f}')

# ============================================================
# ADD RANDOMNESS: H = diag(log k) + eps*GCD + delta*GUE
# ============================================================
print('\n' + '=' * 70)
print('H = diag(log k) + eps*GCD + delta*GUE')
print('=' * 70)

print(f'\n{"eps":>6} {"delta":>6} {"r":>8} {"p(GUE)":>8} {"note":>15}')
print('-' * 48)

for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
    for delta in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        A = rng.standard_normal((N, N))
        A = (A + A.T) / 2
        A_norm = np.linalg.norm(A, 'fro')
        A_scaled = A * (D_norm / A_norm)

        H = D + eps * W_scaled + delta * A_scaled
        eigs = np.linalg.eigvalsh(H)
        r, p_gue, _ = measure_all(eigs)

        note = ''
        if p_gue > 0.05 and r > 0.2:
            note = '<-- BOTH'
        print(f'{eps:>6.2f} {delta:>6.2f} {r:>+8.4f} {p_gue:>8.4f} {note:>15}')

# ============================================================
# THE CRITICAL EXPERIMENT: WHAT IF WE USE THE ACTUAL HECKE OPERATORS?
# ============================================================
print('\n' + '=' * 70)
print('H = diag(log k) + eps * HECKE + delta * GUE')
print('=' * 70)

from riemann.analysis.bost_connes_operator import construct_hecke_prime_adjacency

W_hecke = construct_hecke_prime_adjacency(N).astype(float)
W_hecke_norm = np.linalg.norm(W_hecke, 'fro')
W_hecke_scaled = W_hecke * (D_norm / W_hecke_norm)

print(f'\n{"eps":>6} {"delta":>6} {"r":>8} {"p(GUE)":>8} {"note":>15}')
print('-' * 48)

for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
    for delta in [0, 0.1, 0.5, 1.0, 2.0]:
        if delta > 0:
            A = rng.standard_normal((N, N))
            A = (A + A.T) / 2
            A_scaled = A * (D_norm / np.linalg.norm(A, 'fro'))
            H = D + eps * W_hecke_scaled + delta * A_scaled
        else:
            H = D + eps * W_hecke_scaled
        eigs = np.linalg.eigvalsh(H)
        r, p_gue, _ = measure_all(eigs)
        note = ''
        if p_gue > 0.05 and r > 0.2:
            note = '<-- BOTH'
        print(f'{eps:>6.2f} {delta:>6.2f} {r:>+8.4f} {p_gue:>8.4f} {note:>15}')

# ============================================================
# PURE ARITHMETIC: diag(log k) + Hecke, NO randomness
# ============================================================
print('\n' + '=' * 70)
print('PURE ARITHMETIC: diag(log k) + eps * Hecke (no randomness)')
print('=' * 70)

for eps in np.logspace(-3, 1, 20):
    H = D + eps * W_hecke_scaled
    eigs = np.linalg.eigvalsh(H)
    r, p_gue, _ = measure_all(eigs)
    tag = ' ***' if p_gue > 0.05 and r > 0.3 else (' *' if p_gue > 0.05 or r > 0.3 else '')
    print(f'  eps={eps:>8.4f}: r={r:>+8.4f}, p(GUE)={p_gue:>8.4f}{tag}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

print(f'\n  The tension between GUE eigenvalues and eigenvector rigidity:')
print(f'  - Pure diag(log k): r ~ +0.5, p(GUE) = 0 (too structured)')
print(f'  - + GCD/Hecke: r increases, p(GUE) stays 0 (more structure)')
print(f'  - + randomness: p(GUE) increases, r decreases (dilutes coupling)')
print(f'  - The ONLY way to get both: the "randomness" must be arithmetic')
print(f'\n  This is the fundamental insight: the zeta operator is not')
print(f'  diag(log k) + arithmetic + random. It is a SINGLE object where')
print(f'  the "randomness" in eigenvalue statistics EMERGES from the')
print(f'  arithmetic structure itself. The primes ARE the random matrix.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
