"""GUE comparison for the peak-gap correlation.

Critical test: is the r=+0.70 peak-gap correlation expected from GUE?
And does the p=3 prime modulation survive after controlling for gap size?
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal, eigvalsh
from scipy.stats import pearsonr
from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()

# ============================================================
# STEP 1: GUE characteristic polynomial heights at midpoints
# ============================================================
print('Computing GUE peak-gap correlation...')
print('  For each GUE matrix: eigenvalues + |det(z - H)| at midpoints')

def gue_peak_gap_correlation(n_matrix, n_matrices, rng):
    """Compute peak heights (|char poly at midpoint|) and gaps for GUE."""
    all_gaps = []
    all_peaks = []

    for trial in range(n_matrices):
        # Generate GUE matrix
        A = rng.standard_normal((n_matrix, n_matrix)) + 1j * rng.standard_normal((n_matrix, n_matrix))
        H = (A + A.conj().T) / (2 * np.sqrt(2 * n_matrix))
        eigs = np.sort(np.linalg.eigvalsh(H))

        # Unfold and normalize
        sp = polynomial_unfold(eigs, trim_fraction=0.1)
        if len(sp) < 10:
            continue
        sp = sp / np.mean(sp)

        # Trim eigenvalues to match unfolded spacings
        n_trim = int(0.1 * len(eigs))
        eigs_trimmed = eigs[n_trim:-n_trim]

        # For the characteristic polynomial |det(z - H)|:
        # At midpoint z_mid between eig_k and eig_{k+1}:
        # |det(z_mid - H)| = prod_j |z_mid - eig_j|
        # This is expensive for large N, but we can use log:
        # log|det| = sum_j log|z_mid - eig_j|
        for k in range(len(sp)):
            if k + 1 >= len(eigs_trimmed):
                break
            z_mid = (eigs_trimmed[k] + eigs_trimmed[k + 1]) / 2
            log_det = np.sum(np.log(np.abs(z_mid - eigs)))
            all_gaps.append(sp[k])
            all_peaks.append(np.exp(log_det))  # |det(z_mid - H)|

    return np.array(all_gaps), np.array(all_peaks)


rng = np.random.default_rng(42)
# Use moderate matrix size for speed
gue_gaps, gue_peaks = gue_peak_gap_correlation(200, 100, rng)
print(f'  {len(gue_gaps)} GUE gap-peak pairs from 100 matrices at N=200')

# Normalize peaks (they can span huge range due to det product)
# Use log peaks for correlation
gue_log_peaks = np.log(gue_peaks + 1e-300)

r_gue, p_gue = pearsonr(gue_gaps, gue_log_peaks)
print(f'  GUE: corr(gap, log|det|) = {r_gue:+.4f} (p = {p_gue:.2e})')

# Also rank correlation (more robust)
from scipy.stats import spearmanr
rs_gue, ps_gue = spearmanr(gue_gaps, gue_log_peaks)
print(f'  GUE: Spearman(gap, log|det|) = {rs_gue:+.4f} (p = {ps_gue:.2e})')

# ============================================================
# STEP 2: Same for zeta zeros (already computed)
# ============================================================
print('\nLoading zeta zero peak-gap data...')
import mpmath
mpmath.mp.dps = 20

zeros = np.load('_zeros_500.npy')
N_zeros = len(zeros)
LOG_T = np.log(np.mean(zeros) / (2 * np.pi))
mean_gap = 2 * np.pi / LOG_T

gaps = np.diff(zeros) / mean_gap  # normalized
N = len(gaps)

# Compute Z(midpoints)
Z_mid = np.zeros(N)
for i in range(N):
    t_mid = (zeros[i] + zeros[i + 1]) / 2
    Z_mid[i] = float(mpmath.siegelz(t_mid))

peak_heights = np.abs(Z_mid)
log_peaks_zeta = np.log(peak_heights + 1e-10)

r_zeta, p_zeta = pearsonr(gaps, log_peaks_zeta)
rs_zeta, ps_zeta = spearmanr(gaps, log_peaks_zeta)

print(f'  Zeta: corr(gap, log|Z|) = {r_zeta:+.4f} (p = {p_zeta:.2e})')
print(f'  Zeta: Spearman(gap, log|Z|) = {rs_zeta:+.4f} (p = {ps_zeta:.2e})')

# Direct comparison
print(f'\n  {"Metric":<30} {"Zeta":>10} {"GUE":>10} {"Excess":>10}')
print(f'  {"-"*65}')
print(f'  {"Pearson(gap, log|peak|)":<30} {r_zeta:>+10.4f} {r_gue:>+10.4f} {r_zeta - r_gue:>+10.4f}')
print(f'  {"Spearman(gap, log|peak|)":<30} {rs_zeta:>+10.4f} {rs_gue:>+10.4f} {rs_zeta - rs_gue:>+10.4f}')

# ============================================================
# STEP 3: Prime modulation after controlling for gap
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: PRIME MODULATION AFTER CONTROLLING FOR GAP')
print('=' * 70)

# Residualize |Z| on gap: remove the peak-gap correlation
# Then test if the residual |Z| correlates with prime phases.
# This isolates DIRECT prime modulation of the wave function
# from INDIRECT modulation via gaps.

# Linear regression: log|Z| = a + b * gap + residual
from numpy.polynomial.polynomial import polyfit
coeffs = np.polyfit(gaps, log_peaks_zeta, 1)
predicted_log_peak = np.polyval(coeffs, gaps)
residual_log_peak = log_peaks_zeta - predicted_log_peak

midpoints = (zeros[:-1] + zeros[1:]) / 2

primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
print(f'\n  Prime modulation of RESIDUAL log|Z| (gap-controlled):')
print(f'  {"Prime":>6} {"r(resid, cos)":>15} {"p-value":>10} {"r(raw, cos)":>15} {"p-value":>10}')
print(f'  {"-"*60}')

sig_resid = 0
for p in primes:
    phase = np.cos(2 * np.pi * midpoints * np.log(p) / LOG_T)
    r_raw, p_raw = pearsonr(log_peaks_zeta, phase)
    r_res, p_res = pearsonr(residual_log_peak, phase)
    tag = ' **' if p_res < 0.01 else (' *' if p_res < 0.05 else '')
    if p_res < 0.05:
        sig_resid += 1
    print(f'  {p:>6} {r_res:>+15.6f} {p_res:>10.4f} {r_raw:>+15.6f} {p_raw:>10.4f}{tag}')

print(f'\n  Significant after gap control: {sig_resid}/{len(primes)} (expect {len(primes)*0.05:.1f})')

# ============================================================
# STEP 4: Quintile analysis — peak height conditional on gap
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: PEAK HEIGHT DISTRIBUTION BY GAP QUINTILE')
print('=' * 70)

quintile_edges = np.percentile(gaps, [0, 20, 40, 60, 80, 100])
labels = ['Q1(tiny)', 'Q2(small)', 'Q3(mid)', 'Q4(large)', 'Q5(huge)']

print(f'\n  {"Quintile":<12} {"Mean gap":>10} {"Mean |Z|":>10} {"Mean log|Z|":>12} {"Std log|Z|":>12}')
for q in range(5):
    mask = (gaps >= quintile_edges[q]) & (gaps < quintile_edges[q + 1])
    g = gaps[mask]
    lp = log_peaks_zeta[mask]
    pk = peak_heights[mask]
    print(f'  {labels[q]:<12} {np.mean(g):>10.4f} {np.mean(pk):>10.4f} '
          f'{np.mean(lp):>12.4f} {np.std(lp):>12.4f}')

# Compare std(log|Z|) across quintiles — is the VARIANCE of peak heights
# gap-dependent? (GUE predicts it should be roughly constant)
q1_std = np.std(log_peaks_zeta[(gaps >= quintile_edges[0]) & (gaps < quintile_edges[1])])
q5_std = np.std(log_peaks_zeta[(gaps >= quintile_edges[4]) & (gaps < quintile_edges[5])])
print(f'\n  Std(log|Z|) ratio Q5/Q1: {q5_std / q1_std:.3f}')
print(f'  (1.0 = gap-independent variance, != 1.0 = heteroscedastic)')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

peak_gap_excess = r_zeta - r_gue
print(f'\nPeak-gap correlation:')
print(f'  Zeta:  r = {r_zeta:+.4f}')
print(f'  GUE:   r = {r_gue:+.4f}')
print(f'  Excess: {peak_gap_excess:+.4f}')

if abs(peak_gap_excess) < 0.05:
    print('  -> Consistent with GUE. The peak-gap link is UNIVERSAL.')
elif peak_gap_excess > 0.05:
    print('  -> Zeta has STRONGER peak-gap correlation than GUE.')
    print('  -> The wave function is more tightly coupled to eigenvalue gaps.')
else:
    print('  -> Zeta has WEAKER peak-gap correlation than GUE.')

print(f'\nPrime modulation after gap control:')
if sig_resid > len(primes) * 0.1 + 1:
    print(f'  -> {sig_resid} primes significant: DIRECT wave function modulation')
    print('  -> This is NOT mediated by gaps — it is genuinely new.')
    print('  -> The arithmetic modulation extends to the wave function,')
    print('  -> not just the eigenvalue spectrum.')
elif sig_resid > 0:
    print(f'  -> {sig_resid} primes marginally significant.')
    print('  -> Suggestive but not definitive with N={N} intervals.')
else:
    print('  -> No residual prime modulation. All prime effects are mediated by gaps.')
    print('  -> The wave function carries no arithmetic information beyond what')
    print('  -> the eigenvalue gaps already encode.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
