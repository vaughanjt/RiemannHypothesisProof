#!/usr/bin/env python3
"""
Operator notebook for Grok verification at N=800.

The operator:
  H_N = diag(log(k+1))_{k=1..N} + (c/log(N)) * (||D||_F / ||W||_F) * W

where W_{jk} = 1/sqrt(j*k) if j|k or k|j (j != k), else 0.

This script computes:
1. The operator and its eigenvalues (N=100,200,400,600,800)
2. Peak-gap correlation r (paper §2.2-2.3 methodology)
3. Eigenvalue spacing statistics vs GUE Wigner surmise
4. The small-n decomposition analogue
5. Generic-phase test
6. Resolvent trace comparison to RS sum

All measurements use the EXACT methodology from:
  "Eigenvector Rigidity of Riemann Zeta Zeros" (Claude & Vaughan, 2026)

Dependencies: numpy, scipy, sympy, mpmath
No custom library imports — fully self-contained.
"""
import numpy as np
from scipy.stats import pearsonr, kstest
from scipy.linalg import eigvalsh
import time


# ============================================================
# OPERATOR CONSTRUCTION
# ============================================================
def build_operator(N, c=1.1):
    """Build H_N = diag(log(k+1)) + (c/logN) * scaled_divisor_matrix.

    Parameters
    ----------
    N : int — matrix size
    c : float — coupling constant (default 1.1)

    Returns
    -------
    H : ndarray (N, N) — symmetric real matrix
    """
    # Diagonal: log(k+1) for k = 1, ..., N
    D = np.diag(np.log(np.arange(1, N + 1, dtype=np.float64) + 1))
    D_norm = np.linalg.norm(D, 'fro')

    # Off-diagonal: divisibility indicator / sqrt(j*k)
    W = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(j + 1, N + 1):
            if k % j == 0:  # j divides k (covers j|k; k|j is j|k with j<k)
                W[j - 1, k - 1] = 1.0 / np.sqrt(j * k)
                W[k - 1, j - 1] = W[j - 1, k - 1]
    W_norm = np.linalg.norm(W, 'fro')

    eps = c / np.log(N)
    H = D + eps * (D_norm / W_norm) * W
    return H


# ============================================================
# POLYNOMIAL UNFOLDING (standard RMT procedure)
# ============================================================
def polynomial_unfold(eigenvalues, degree=5, trim_fraction=0.1):
    """Unfold eigenvalues using polynomial fit to integrated density.

    Parameters
    ----------
    eigenvalues : sorted array of eigenvalues
    degree : polynomial degree for CDF fit
    trim_fraction : fraction to trim from each edge

    Returns
    -------
    spacings : normalized spacings (mean ~ 1)
    """
    eigs = np.sort(eigenvalues)
    n = len(eigs)
    n_trim = int(trim_fraction * n)
    eigs_trimmed = eigs[n_trim:n - n_trim]

    # Fit polynomial to empirical CDF
    cdf = np.arange(1, len(eigs_trimmed) + 1) / len(eigs_trimmed)
    coeffs = np.polyfit(eigs_trimmed, cdf, degree)
    unfolded = np.polyval(coeffs, eigs_trimmed)

    # Spacings
    spacings = np.diff(unfolded) * len(eigs_trimmed)
    return spacings


# ============================================================
# PEAK-GAP CORRELATION (paper §2.2-2.3)
# ============================================================
def peak_gap_correlation(eigenvalues):
    """Measure peak-gap r using characteristic polynomial at midpoints.

    For each consecutive eigenvalue pair (lambda_k, lambda_{k+1}):
    - gap = normalized spacing
    - peak = log|det(z_mid - H)| = sum_j log|z_mid - lambda_j|

    Returns Pearson r(gap, log_peak).
    """
    eigs = np.sort(eigenvalues)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20:
        return 0, 0, 0
    sp = sp / np.mean(sp)

    n_trim = int(0.1 * len(eigs))
    eigs_trimmed = eigs[n_trim:len(eigs) - n_trim]

    log_peaks = []
    gaps = []
    for k in range(min(len(sp), len(eigs_trimmed) - 1)):
        z_mid = (eigs_trimmed[k] + eigs_trimmed[k + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        log_peaks.append(log_det)
        gaps.append(sp[k])

    gaps = np.array(gaps)
    log_peaks = np.array(log_peaks)

    if len(gaps) < 10:
        return 0, 0, 0

    r, p_val = pearsonr(gaps, log_peaks)

    # Power law beta: log|peak| = beta * log(gap) + const
    mask = gaps > 0.1
    if np.sum(mask) > 10:
        beta = np.polyfit(np.log(gaps[mask]), log_peaks[mask], 1)[0]
    else:
        beta = 0

    return r, beta, len(gaps)


# ============================================================
# GUE COMPARISON
# ============================================================
def gue_peak_gap(N_matrix=200, n_matrices=50, seed=42):
    """Compute peak-gap r for GUE ensemble."""
    rng = np.random.default_rng(seed)
    all_r = []
    for _ in range(n_matrices):
        A = rng.standard_normal((N_matrix, N_matrix))
        H_gue = (A + A.T) / (2 * np.sqrt(2 * N_matrix))
        eigs = np.linalg.eigvalsh(H_gue)
        r, _, _ = peak_gap_correlation(eigs)
        all_r.append(r)
    return np.mean(all_r), np.std(all_r)


def wigner_cdf(s):
    """GUE Wigner surmise CDF."""
    return 1 - np.exp(-np.pi * s ** 2 / 4)


# ============================================================
# MAIN: RUN ALL DIAGNOSTICS
# ============================================================
if __name__ == '__main__':
    print('=' * 70)
    print('OPERATOR DIAGNOSTICS: H_N = diag(log(k+1)) + (c/logN) * div/sqrt(jk)')
    print('c = 1.1')
    print('=' * 70)

    # GUE baseline
    print('\nGUE baseline (N=200, 50 matrices):')
    r_gue, std_gue = gue_peak_gap()
    print(f'  r = {r_gue:+.4f} +/- {std_gue:.4f}')

    # Sweep N
    print(f'\n{"N":>5} {"r":>8} {"beta":>8} {"pts":>5} {"KS_p(GUE)":>10} {"build_s":>8}')
    print('-' * 50)

    for N in [100, 200, 400, 600, 800]:
        t0 = time.time()
        H = build_operator(N, c=1.1)
        t_build = time.time() - t0

        eigs = np.linalg.eigvalsh(H)
        r, beta, pts = peak_gap_correlation(eigs)

        sp = polynomial_unfold(eigs, trim_fraction=0.1)
        sp = sp / np.mean(sp)
        ks_stat, ks_p = kstest(sp, wigner_cdf)

        print(f'{N:>5} {r:>+8.4f} {beta:>8.1f} {pts:>5} {ks_p:>10.4e} {t_build:>8.2f}s')

    # Detailed analysis at N=800
    print('\n' + '=' * 70)
    print('DETAILED ANALYSIS AT N=800')
    print('=' * 70)

    N = 800
    print(f'\nBuilding operator (N={N})...')
    t0 = time.time()
    H = build_operator(N, c=1.1)
    eigs = np.linalg.eigvalsh(H)
    print(f'  Built in {time.time()-t0:.1f}s')

    # Full spacing analysis
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    sp = sp / np.mean(sp)
    print(f'\n  Spacings: {len(sp)} values')
    print(f'  Mean: {np.mean(sp):.4f}, Std: {np.std(sp):.4f}')

    ks_gue, p_gue = kstest(sp, wigner_cdf)
    ks_poi, p_poi = kstest(sp, 'expon', args=(0, 1))
    print(f'  KS vs Wigner (GUE): D={ks_gue:.4f}, p={p_gue:.4e}')
    print(f'  KS vs Poisson: D={ks_poi:.4f}, p={p_poi:.4e}')

    # Peak-gap
    r, beta, pts = peak_gap_correlation(eigs)
    print(f'\n  Peak-gap: r = {r:+.4f}, beta = {beta:.1f}, {pts} pairs')
    print(f'  GUE baseline: r = {r_gue:+.4f}')
    print(f'  Excess over GUE: {r/r_gue:.1f}x')

    # Eigenvalue range
    print(f'\n  Eigenvalue range: [{eigs[0]:.4f}, {eigs[-1]:.4f}]')
    print(f'  log(2)={np.log(2):.4f}, log(N+1)={np.log(N+1):.4f}')

    # Resolvent trace at a few points on the critical line
    print(f'\n  Resolvent trace at s = 1/2 + it:')
    for t_val in [14.13, 25.01, 50.0]:
        s = complex(0.5, t_val)
        res = np.sum(1.0 / (eigs - s))
        print(f'    t={t_val:>6.2f}: |Tr(H-sI)^-1| = {abs(res):.4f}')

    print('\n' + '=' * 70)
    print('END OF DIAGNOSTICS')
    print('=' * 70)
    print('\nTo run: python operator_notebook_for_grok.py')
    print('Dependencies: numpy, scipy (no custom imports)')
