"""Value distribution of Z(t) between consecutive zeta zeros.

The spacing statistics tell us WHERE the zeros are. This script examines
HOW the zeta function behaves BETWEEN zeros — the amplitudes, shapes,
and correlations of the oscillation peaks.

This is analogous to eigenvector statistics in RMT — far less explored
than eigenvalue statistics. Keating-Snaith (2000) predict the moments
of |zeta(1/2+it)| but the fine structure (peak shapes, peak-gap
correlations, valley depths) is largely uncharted.

Key questions:
1. Does the peak height between zeros correlate with the surrounding gaps?
2. Is the oscillation shape (width, asymmetry) universal or arithmetically modulated?
3. Do the valleys (local minima of |Z(t)|) carry structure beyond GUE?
4. Is there a "wave function" analog of the pair-correlation exclusivity?
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr, spearmanr, kstest
import mpmath

t0 = time.time()

# ============================================================
# COMPUTE Z(t) VALUES BETWEEN ZEROS
# ============================================================
print('Computing Z(t) between Odlyzko zeros...')

# Load zeros — use the low zeros (T~458) where siegelz is fast
# siegelz at T~100: 2ms; at T~2.7e11: 781ms (355x slower)
zeros_abs = np.load('_zeros_500.npy')  # absolute imaginary parts
T_BASE = 0  # zeros are already absolute heights
N_zeros = len(zeros_abs)
zeros = zeros_abs  # use directly as heights

# The Riemann-Siegel Z function: Z(t) = exp(i*theta(t)) * zeta(1/2 + it)
# Z(t) is real-valued and its sign changes at zeros.
# Between consecutive zeros, Z(t) has exactly one extremum (peak or valley).

# Compute Z(t) at midpoints and quarter-points between consecutive zeros
# Use mpmath for high precision at T ~ 2.7e11
mpmath.mp.dps = 20  # 20 digits sufficient for Z values

T_MID = np.mean(zeros)
LOG_T = np.log(T_MID / (2 * np.pi))
print(f'  {N_zeros} zeros, T range: {zeros[0]:.2f} to {zeros[-1]:.2f} (mean T ~ {T_MID:.0f})')
print(f'  Computing Z(t) at midpoints + quarter-points...')

# Sample ALL intervals (fast at low T)
N_SAMPLE = N_zeros - 1
midpoints = np.zeros(N_SAMPLE)
quarter1 = np.zeros(N_SAMPLE)
quarter3 = np.zeros(N_SAMPLE)
Z_mid = np.zeros(N_SAMPLE)
Z_q1 = np.zeros(N_SAMPLE)
Z_q3 = np.zeros(N_SAMPLE)
gap_sizes = np.zeros(N_SAMPLE)

t_compute = time.time()
for i in range(N_SAMPLE):
    t0_zero = zeros[i]
    t1_zero = zeros[i + 1]
    gap = t1_zero - t0_zero
    gap_sizes[i] = gap

    t_mid = (t0_zero + t1_zero) / 2
    t_q1 = t0_zero + gap / 4
    t_q3 = t0_zero + 3 * gap / 4

    midpoints[i] = t_mid
    quarter1[i] = t_q1
    quarter3[i] = t_q3

    # Compute Z(t) using mpmath
    Z_mid[i] = float(mpmath.siegelz(t_mid))
    Z_q1[i] = float(mpmath.siegelz(t_q1))
    Z_q3[i] = float(mpmath.siegelz(t_q3))

    if (i + 1) % 200 == 0:
        elapsed = time.time() - t_compute
        eta = elapsed / (i + 1) * (N_SAMPLE - i - 1)
        print(f'    {i+1}/{N_SAMPLE} ({elapsed:.0f}s, ETA {eta:.0f}s)')

print(f'  Done: {N_SAMPLE} intervals, {time.time()-t_compute:.1f}s')

# Normalize gaps by local mean spacing
mean_gap = 2 * np.pi / LOG_T
norm_gaps = gap_sizes / mean_gap

# Peak heights: |Z(midpoint)| (absolute value at the extremum)
peak_heights = np.abs(Z_mid)
# Sign of Z at midpoint (alternates +/- between consecutive intervals)
Z_sign = np.sign(Z_mid)

# ============================================================
# STEP 1: Peak height distribution
# ============================================================
print('\n' + '=' * 70)
print('STEP 1: PEAK HEIGHT DISTRIBUTION')
print('=' * 70)

print(f'  N intervals: {N_SAMPLE}')
print(f'  Peak |Z(mid)|: mean={np.mean(peak_heights):.4f}, std={np.std(peak_heights):.4f}')
print(f'  Peak |Z(mid)|: median={np.median(peak_heights):.4f}')
print(f'  Peak |Z(mid)|: min={np.min(peak_heights):.6f}, max={np.max(peak_heights):.4f}')

# Keating-Snaith predict: moments of |zeta(1/2+it)|
# E[|zeta|^{2k}] ~ c_k * (log T)^{k^2}
# For k=1: E[|zeta|^2] ~ c_1 * log T
# The Z function has the same distribution as zeta on the critical line
print(f'\n  Keating-Snaith prediction: E[|Z|^2] ~ c * log(T/2pi)')
print(f'  log(T/2pi) = {LOG_T:.2f}')
print(f'  E[Z^2] (data) = {np.mean(Z_mid**2):.4f}')
print(f'  E[|Z|] (data) = {np.mean(peak_heights):.4f}')

# Percentile distribution
print(f'\n  Percentiles of |Z(mid)|:')
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f'    {pct:>3}th: {np.percentile(peak_heights, pct):.4f}')

# ============================================================
# STEP 2: Peak-gap correlation (THE KEY TEST)
# ============================================================
print('\n' + '=' * 70)
print('STEP 2: PEAK-GAP CORRELATION')
print('=' * 70)

# Does the height of Z(t) at the midpoint correlate with the gap size?
# If yes: the wave function encodes spacing information (non-trivial)
# If no: peak heights are independent of the eigenvalue gaps (GUE universal)

rp, pp = pearsonr(norm_gaps, peak_heights)
rs, ps = spearmanr(norm_gaps, peak_heights)
print(f'  |Z(mid)| vs normalized gap:')
print(f'    Pearson  r = {rp:+.4f}, p = {pp:.2e}')
print(f'    Spearman r = {rs:+.4f}, p = {ps:.2e}')

# Also test Z^2 (related to local energy)
rp2, pp2 = pearsonr(norm_gaps, Z_mid ** 2)
print(f'  Z(mid)^2 vs normalized gap:')
print(f'    Pearson  r = {rp2:+.4f}, p = {pp2:.2e}')

# Log peak height vs gap
rp_log, pp_log = pearsonr(norm_gaps, np.log(peak_heights + 1e-10))
print(f'  log|Z(mid)| vs normalized gap:')
print(f'    Pearson  r = {rp_log:+.4f}, p = {pp_log:.2e}')

if pp < 0.01:
    print('\n  >>> SIGNIFICANT: peak heights correlate with gap sizes!')
    print('  >>> The wave function encodes eigenvalue gap information.')
else:
    print('\n  -> No significant peak-gap correlation.')

# ============================================================
# STEP 3: Peak height vs NEIGHBORING gaps
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: PEAK vs NEIGHBORING GAPS')
print('=' * 70)

# Does the peak in interval n correlate with gap n-1 or gap n+1?
print(f'  {"Comparison":<35} {"Pearson r":>10} {"p-value":>12}')
print(f'  {"-"*60}')

for label, x, y in [
    ('gap_n vs |Z_n|', norm_gaps, peak_heights),
    ('gap_n vs |Z_{n+1}|', norm_gaps[:-1], peak_heights[1:]),
    ('gap_n vs |Z_{n-1}|', norm_gaps[1:], peak_heights[:-1]),
    ('gap_n vs gap_{n+1}', norm_gaps[:-1], norm_gaps[1:]),
    ('|Z_n| vs |Z_{n+1}|', peak_heights[:-1], peak_heights[1:]),
    ('gap_n * |Z_n| vs 1', norm_gaps, peak_heights),
]:
    r, p = pearsonr(x, y)
    tag = ' **' if p < 0.01 else (' *' if p < 0.05 else '')
    print(f'  {label:<35} {r:>+10.4f} {p:>12.2e}{tag}')

# ============================================================
# STEP 4: Oscillation shape (asymmetry)
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: OSCILLATION SHAPE — ASYMMETRY')
print('=' * 70)

# The oscillation between zeros can be asymmetric: the peak may be
# closer to one zero than the other. This "shape" information goes
# beyond the spacing statistics.

# Asymmetry: compare |Z(quarter1)| and |Z(quarter3)|
# If symmetric: |Z(q1)| ≈ |Z(q3)| on average
# If asymmetric: systematic bias

asymmetry = (np.abs(Z_q3) - np.abs(Z_q1)) / (peak_heights + 1e-10)
print(f'  Asymmetry = (|Z(3/4)| - |Z(1/4)|) / |Z(mid)|')
print(f'  Mean asymmetry: {np.mean(asymmetry):+.4f}')
print(f'  Std asymmetry: {np.std(asymmetry):.4f}')
print(f'  t-statistic: {np.mean(asymmetry) / (np.std(asymmetry) / np.sqrt(N_SAMPLE)):.2f}')

# Does asymmetry correlate with gap size or peak height?
r_asym_gap, p_asym_gap = pearsonr(norm_gaps, asymmetry)
r_asym_peak, p_asym_peak = pearsonr(peak_heights, asymmetry)
print(f'\n  Asymmetry vs gap: r = {r_asym_gap:+.4f}, p = {p_asym_gap:.4f}')
print(f'  Asymmetry vs peak: r = {r_asym_peak:+.4f}, p = {p_asym_peak:.4f}')

# Shape ratio: how "peaked" is the oscillation?
# peaked = |Z(mid)| / (|Z(q1)| + |Z(q3)|)
shape_ratio = peak_heights / (np.abs(Z_q1) + np.abs(Z_q3) + 1e-10)
print(f'\n  Shape ratio |Z(mid)| / (|Z(1/4)| + |Z(3/4)|):')
print(f'    Mean: {np.mean(shape_ratio):.4f}, Std: {np.std(shape_ratio):.4f}')

r_shape_gap, p_shape_gap = pearsonr(norm_gaps, shape_ratio)
print(f'    Shape vs gap: r = {r_shape_gap:+.4f}, p = {p_shape_gap:.4f}')

# ============================================================
# STEP 5: Valley analysis — near-misses
# ============================================================
print('\n' + '=' * 70)
print('STEP 5: NEAR-ZERO VALUES (VALLEY ANALYSIS)')
print('=' * 70)

# The values of Z(t) at the quarter-points tell us how close Z gets
# to zero between its actual zeros. Near-misses (small |Z(q)|) indicate
# "almost-zeros" — places where the zeta function nearly has an extra zero.

# Find near-zero values at quarter-points
all_quarter_vals = np.concatenate([np.abs(Z_q1), np.abs(Z_q3)])
print(f'  Quarter-point |Z| values: {len(all_quarter_vals)} samples')
print(f'    Mean: {np.mean(all_quarter_vals):.4f}')
print(f'    Min: {np.min(all_quarter_vals):.6f}')
print(f'    N with |Z| < 0.1: {np.sum(all_quarter_vals < 0.1)}')
print(f'    N with |Z| < 0.01: {np.sum(all_quarter_vals < 0.01)}')

# Near-miss rate: how often does |Z(quarter)| < threshold?
# Compare to random (if Z were Gaussian at each point)
for thresh in [0.1, 0.5, 1.0]:
    frac = np.mean(all_quarter_vals < thresh)
    print(f'    P(|Z(quarter)| < {thresh}): {frac:.4f}')

# ============================================================
# STEP 6: Prime-phase modulation of peak heights
# ============================================================
print('\n' + '=' * 70)
print('STEP 6: PRIME MODULATION OF PEAK HEIGHTS')
print('=' * 70)

# The big test: do peak heights oscillate with prime-frequency modulation?
# If the BK pair correlation comes from primes, does the WAVE FUNCTION
# also show prime structure?

# For each interval, compute the prime phase
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

print(f'  {"Prime":>6} {"r(|Z|,cos)":>12} {"p-value":>10} {"r(Z^2,cos)":>12} {"p-value":>10}')
print(f'  {"-"*55}')

sig_count = 0
for p in primes:
    phase = np.cos(2 * np.pi * midpoints * np.log(p) / LOG_T)
    r1, p1 = pearsonr(peak_heights, phase)
    r2, p2 = pearsonr(Z_mid ** 2, phase)
    tag = ' **' if p1 < 0.01 or p2 < 0.01 else (' *' if p1 < 0.05 or p2 < 0.05 else '')
    if p1 < 0.05 or p2 < 0.05:
        sig_count += 1
    print(f'  {p:>6} {r1:>+12.6f} {p1:>10.4f} {r2:>+12.6f} {p2:>10.4f}{tag}')

print(f'\n  Significant (p<0.05): {sig_count}/{len(primes)} (expect {len(primes)*0.05:.1f} by chance)')

# ============================================================
# STEP 7: Successive peak height ratios
# ============================================================
print('\n' + '=' * 70)
print('STEP 7: SUCCESSIVE PEAK RATIOS')
print('=' * 70)

# The ratio |Z_{n+1}| / |Z_n| — how do successive peaks relate?
ratios = peak_heights[1:] / (peak_heights[:-1] + 1e-10)
log_ratios = np.log(ratios + 1e-10)

print(f'  |Z_{{n+1}}| / |Z_n| ratio:')
print(f'    Mean: {np.mean(ratios):.4f}')
print(f'    Geometric mean: {np.exp(np.mean(log_ratios)):.4f}')
print(f'    Std: {np.std(ratios):.4f}')
print(f'    ACF(1) of log-ratios: {np.corrcoef(log_ratios[:-1], log_ratios[1:])[0,1]:+.4f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: VALUE DISTRIBUTION OF Z(t)')
print('=' * 70)

findings = []

# Peak-gap correlation
if pp < 0.01:
    findings.append(f'Peak-gap correlation: r={rp:+.4f} (p={pp:.2e}) — '
                     'peak heights encode gap information')

# Asymmetry
t_asym = np.mean(asymmetry) / (np.std(asymmetry) / np.sqrt(N_SAMPLE))
if abs(t_asym) > 2.5:
    findings.append(f'Systematic asymmetry: t={t_asym:.2f} — oscillation shape is biased')

# Prime modulation
if sig_count > len(primes) * 0.05 + 2:
    findings.append(f'{sig_count}/{len(primes)} primes modulate peak heights')

# Shape-gap correlation
if p_shape_gap < 0.01:
    findings.append(f'Shape-gap correlation: r={r_shape_gap:+.4f} (p={p_shape_gap:.4f})')

if not findings:
    print('\n>>> NO NOVEL STRUCTURE in the value distribution.')
    print('>>> Peak heights, shapes, and prime modulations are all consistent')
    print('>>> with what GUE + BK predicts for the wave function.')
else:
    print(f'\n>>> {len(findings)} FINDINGS:')
    for f in findings:
        print(f'>>>   {f}')
    print('\n>>> These go BEYOND spacing statistics into wave function territory.')
    print('>>> If confirmed, they represent structure not captured by k-point correlations.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
