"""Extreme gap statistics of Riemann zeta zeros.

Do unusually large or small spacings carry information beyond GUE?
If R_k = GUE for all k >= 3, the extreme gap distribution should
match GUE exactly. Any deviation is genuinely new.

Analysis:
1. Block maxima/minima distributions vs GUE
2. Conditional clustering: do extreme gaps attract each other?
3. Gap-prime correlations: do extreme gaps occur at specific phases
   of the prime oscillation?
4. Champion gaps: what's special about the largest/smallest spacings?
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.stats import kstest, pearsonr, spearmanr
from sympy import primerange
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

t0 = time.time()

# ============================================================
# LOAD DATA
# ============================================================
print('Loading data...')

def gue_eigs(n, rng):
    d = rng.standard_normal(n)
    e = np.sqrt(rng.chisquare(2 * np.arange(n - 1, 0, -1)) / 2)
    return eigvalsh_tridiagonal(d, e) / np.sqrt(n)

# Odlyzko zeros
zeros = []
with open('data/odlyzko/zeros3.txt') as f:
    for line in f:
        try: zeros.append(float(line.strip()))
        except ValueError: pass
zeros = np.array(zeros)
T_BASE = 267653395647.0
LOG_T = np.log(T_BASE / (2 * np.pi))
density = LOG_T / (2 * np.pi)
spacings = np.diff(zeros) * density
spacings = spacings / np.mean(spacings)  # normalize to mean 1
N = len(spacings)
print(f'  {N} spacings at T~2.7e11')

# Generate GUE comparison spacings (large ensemble)
print('  Generating GUE reference ensemble...')
rng = np.random.default_rng(42)
gue_spacings_all = []
for _ in range(200):
    eigs = gue_eigs(1200, rng)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    sp = sp / np.mean(sp)
    gue_spacings_all.append(sp)
gue_spacings = np.concatenate(gue_spacings_all)
print(f'  GUE reference: {len(gue_spacings)} spacings from 200 matrices')

# ============================================================
# STEP 1: Basic extreme statistics
# ============================================================
print('\n' + '=' * 70)
print('STEP 1: BASIC EXTREME STATISTICS')
print('=' * 70)

print(f'\n{"Statistic":<30} {"Zeta zeros":>12} {"GUE":>12}')
print('-' * 58)
for label, func in [
    ('Mean', np.mean), ('Std', np.std), ('Min', np.min), ('Max', np.max),
    ('1st percentile', lambda x: np.percentile(x, 1)),
    ('5th percentile', lambda x: np.percentile(x, 5)),
    ('95th percentile', lambda x: np.percentile(x, 95)),
    ('99th percentile', lambda x: np.percentile(x, 99)),
    ('Skewness', lambda x: np.mean(((x - np.mean(x))/np.std(x))**3)),
    ('Kurtosis (excess)', lambda x: np.mean(((x - np.mean(x))/np.std(x))**4) - 3),
]:
    v_zeta = func(spacings)
    v_gue = func(gue_spacings)
    print(f'{label:<30} {v_zeta:>12.6f} {v_gue:>12.6f}')

# KS test: overall distribution
ks_stat, ks_p = kstest(spacings, gue_spacings)
print(f'\nKS test (zeta vs GUE): D = {ks_stat:.4f}, p = {ks_p:.4f}')

# ============================================================
# STEP 2: Block maxima and minima
# ============================================================
print('\n' + '=' * 70)
print('STEP 2: BLOCK MAXIMA AND MINIMA')
print('=' * 70)

for block_size in [10, 20, 50, 100]:
    n_blocks = N // block_size
    # Zeta block maxima/minima
    zeta_maxima = np.array([np.max(spacings[i*block_size:(i+1)*block_size])
                            for i in range(n_blocks)])
    zeta_minima = np.array([np.min(spacings[i*block_size:(i+1)*block_size])
                            for i in range(n_blocks)])

    # GUE block maxima/minima (same block size, many blocks)
    n_gue_blocks = len(gue_spacings) // block_size
    gue_maxima = np.array([np.max(gue_spacings[i*block_size:(i+1)*block_size])
                           for i in range(n_gue_blocks)])
    gue_minima = np.array([np.min(gue_spacings[i*block_size:(i+1)*block_size])
                           for i in range(n_gue_blocks)])

    ks_max, p_max = kstest(zeta_maxima, gue_maxima)
    ks_min, p_min = kstest(zeta_minima, gue_minima)

    print(f'\n  Block size = {block_size} ({n_blocks} blocks):')
    print(f'    Maxima: zeta mean={np.mean(zeta_maxima):.4f} vs GUE {np.mean(gue_maxima):.4f}'
          f'  KS p={p_max:.4f}')
    print(f'    Minima: zeta mean={np.mean(zeta_minima):.4f} vs GUE {np.mean(gue_minima):.4f}'
          f'  KS p={p_min:.4f}')

# ============================================================
# STEP 3: Extreme gap clustering
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: EXTREME GAP CLUSTERING')
print('=' * 70)

# Do extreme gaps attract each other? If gap n is extreme, is gap n+1
# more likely to also be extreme?

# Define "extreme" as top/bottom 5%
threshold_high = np.percentile(spacings, 95)
threshold_low = np.percentile(spacings, 5)

is_high = spacings > threshold_high
is_low = spacings < threshold_low
is_extreme = is_high | is_low

# Conditional probability: P(gap_{n+1} extreme | gap_n extreme)
extreme_idx = np.where(is_extreme)[0]
next_also_extreme = sum(1 for i in extreme_idx if i + 1 < N and is_extreme[i + 1])
p_conditional = next_also_extreme / max(len(extreme_idx), 1)
p_unconditional = np.mean(is_extreme)

print(f'  Extreme threshold: top/bottom 5% (>{threshold_high:.4f} or <{threshold_low:.4f})')
print(f'  P(extreme): {p_unconditional:.4f}')
print(f'  P(next extreme | current extreme): {p_conditional:.4f}')
print(f'  Clustering ratio: {p_conditional / p_unconditional:.3f}x')

# Same for GUE
gue_th_high = np.percentile(gue_spacings, 95)
gue_th_low = np.percentile(gue_spacings, 5)
gue_extreme = (gue_spacings > gue_th_high) | (gue_spacings < gue_th_low)
gue_ext_idx = np.where(gue_extreme)[0]
gue_next_ext = sum(1 for i in gue_ext_idx if i + 1 < len(gue_spacings) and gue_extreme[i + 1])
gue_p_cond = gue_next_ext / max(len(gue_ext_idx), 1)
gue_p_uncond = np.mean(gue_extreme)

print(f'\n  GUE P(extreme): {gue_p_uncond:.4f}')
print(f'  GUE P(next extreme | current extreme): {gue_p_cond:.4f}')
print(f'  GUE clustering ratio: {gue_p_cond / gue_p_uncond:.3f}x')

clustering_excess = (p_conditional / p_unconditional) - (gue_p_cond / gue_p_uncond)
print(f'\n  Zeta excess clustering vs GUE: {clustering_excess:+.3f}')
if abs(clustering_excess) > 0.1:
    print('  -> ANOMALOUS: extreme gaps cluster differently from GUE')
else:
    print('  -> Consistent with GUE clustering')

# Lag-dependent clustering
print(f'\n  Clustering ratio at different lags:')
print(f'  {"Lag":>5} {"Zeta":>8} {"GUE":>8} {"Excess":>8}')
for lag in [1, 2, 3, 5, 10, 20, 50]:
    z_next = sum(1 for i in extreme_idx if i + lag < N and is_extreme[i + lag])
    z_ratio = (z_next / max(len(extreme_idx), 1)) / p_unconditional
    g_next = sum(1 for i in gue_ext_idx if i + lag < len(gue_spacings) and gue_extreme[i + lag])
    g_ratio = (g_next / max(len(gue_ext_idx), 1)) / gue_p_uncond
    print(f'  {lag:>5} {z_ratio:>8.3f} {g_ratio:>8.3f} {z_ratio - g_ratio:>+8.3f}')

# ============================================================
# STEP 4: Gap-prime phase correlations
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: GAP-PRIME PHASE CORRELATIONS')
print('=' * 70)

# For each spacing s_n at height t_n, compute the "prime phase":
#   phi_p(n) = 2*pi * t_n * log(p) / log(T/2pi)  mod 2*pi
# If extreme gaps occur at specific phases, that's new.

# Use the zero heights directly
zero_heights = zeros[:-1]  # height of each spacing's left endpoint

primes_test = [2, 3, 5, 7, 11, 13]

print(f'  Testing: do extreme gaps prefer specific prime phases?')
print(f'\n  {"Prime":>6} {"Kuiper V (all)":>15} {"p-value":>10} '
      f'{"V (extreme)":>15} {"p-value":>10}')
print(f'  {"-"*60}')

for p in primes_test:
    # Phase for each spacing
    phase = (2 * np.pi * zero_heights * np.log(p) / LOG_T) % (2 * np.pi)

    # Kuiper test for uniformity (two-sided KS for circular data)
    # Simplified: use KS against uniform
    phase_norm = phase / (2 * np.pi)  # [0, 1)
    ks_all, p_all = kstest(phase_norm, 'uniform')

    # Same for extreme gaps only
    phase_extreme = phase_norm[is_extreme]
    if len(phase_extreme) > 10:
        ks_ext, p_ext = kstest(phase_extreme, 'uniform')
    else:
        ks_ext, p_ext = 0, 1

    print(f'  {p:>6} {ks_all:>15.4f} {p_all:>10.4f} '
          f'{ks_ext:>15.4f} {p_ext:>10.4f}')

# More sensitive: correlation between spacing size and phase
print(f'\n  Correlation between spacing size and prime phase:')
print(f'  {"Prime":>6} {"Pearson r":>10} {"p-value":>10} {"Spearman r":>10} {"p-value":>10}')
for p in primes_test:
    phase = np.cos(2 * np.pi * zero_heights * np.log(p) / LOG_T)
    rp, pp = pearsonr(spacings, phase)
    rs, ps = spearmanr(spacings, phase)
    tag = ' *' if pp < 0.05 else ''
    if pp < 0.01: tag = ' **'
    print(f'  {p:>6} {rp:>+10.6f} {pp:>10.4f} {rs:>+10.6f} {ps:>10.4f}{tag}')

# ============================================================
# STEP 5: Champion gaps — the extremes
# ============================================================
print('\n' + '=' * 70)
print('STEP 5: CHAMPION GAPS')
print('=' * 70)

# Top 10 largest gaps
top_idx = np.argsort(spacings)[-10:][::-1]
print(f'\n  10 largest normalized spacings:')
print(f'  {"Rank":>5} {"Index":>7} {"Spacing":>10} {"z-score":>8} {"Height":>15} {"Prev gap":>10} {"Next gap":>10}')
spacing_mean = np.mean(spacings)
spacing_std = np.std(spacings)
for rank, idx in enumerate(top_idx):
    z = (spacings[idx] - spacing_mean) / spacing_std
    height = zero_heights[idx]
    prev_gap = spacings[idx - 1] if idx > 0 else 0
    next_gap = spacings[idx + 1] if idx < N - 1 else 0
    print(f'  {rank+1:>5} {idx:>7} {spacings[idx]:>10.6f} {z:>+8.3f} {height:>15.4f} '
          f'{prev_gap:>10.6f} {next_gap:>10.6f}')

# Bottom 10 smallest gaps
bot_idx = np.argsort(spacings)[:10]
print(f'\n  10 smallest normalized spacings:')
print(f'  {"Rank":>5} {"Index":>7} {"Spacing":>10} {"z-score":>8} {"Prev gap":>10} {"Next gap":>10}')
for rank, idx in enumerate(bot_idx):
    z = (spacings[idx] - spacing_mean) / spacing_std
    prev_gap = spacings[idx - 1] if idx > 0 else 0
    next_gap = spacings[idx + 1] if idx < N - 1 else 0
    print(f'  {rank+1:>5} {idx:>7} {spacings[idx]:>10.6f} {z:>+8.3f} '
          f'{prev_gap:>10.6f} {next_gap:>10.6f}')

# ============================================================
# STEP 6: Nearest-neighbor gap correlations (beyond ACF lag 1)
# ============================================================
print('\n' + '=' * 70)
print('STEP 6: GAP-GAP CONDITIONAL DISTRIBUTIONS')
print('=' * 70)

# Given s_n in quintile Q, what's the distribution of s_{n+1}?
quintile_edges = np.percentile(spacings, [0, 20, 40, 60, 80, 100])
labels = ['Q1 (tiny)', 'Q2 (small)', 'Q3 (mid)', 'Q4 (large)', 'Q5 (huge)']

print(f'\n  Mean of s_{{n+1}} given s_n in quintile:')
print(f'  {"Quintile":<12} {"Zeta next":>10} {"GUE next":>10} {"Excess":>10}')

gue_quintile_edges = np.percentile(gue_spacings, [0, 20, 40, 60, 80, 100])

for q in range(5):
    # Zeta
    mask_z = (spacings[:-1] >= quintile_edges[q]) & (spacings[:-1] < quintile_edges[q+1])
    next_z = spacings[1:][mask_z]
    mean_z = np.mean(next_z) if len(next_z) > 0 else 0

    # GUE
    mask_g = (gue_spacings[:-1] >= gue_quintile_edges[q]) & (gue_spacings[:-1] < gue_quintile_edges[q+1])
    next_g = gue_spacings[1:][mask_g]
    mean_g = np.mean(next_g) if len(next_g) > 0 else 0

    excess = mean_z - mean_g
    print(f'  {labels[q]:<12} {mean_z:>10.6f} {mean_g:>10.6f} {excess:>+10.6f}')

# ============================================================
# STEP 7: Return time statistics
# ============================================================
print('\n' + '=' * 70)
print('STEP 7: RETURN TIME TO EXTREME GAPS')
print('=' * 70)

# How many spacings until the next extreme gap?
# This probes the long-range structure of extreme events.
extreme_positions = np.where(is_extreme)[0]
return_times = np.diff(extreme_positions)

gue_extreme_pos = np.where(gue_extreme)[0]
gue_return_times = np.diff(gue_extreme_pos)

print(f'  Return time (spacings until next extreme):')
print(f'  {"Statistic":<20} {"Zeta":>10} {"GUE":>10}')
print(f'  {"-"*42}')
print(f'  {"Mean":<20} {np.mean(return_times):>10.2f} {np.mean(gue_return_times):>10.2f}')
print(f'  {"Std":<20} {np.std(return_times):>10.2f} {np.std(gue_return_times):>10.2f}')
print(f'  {"Max":<20} {np.max(return_times):>10} {np.max(gue_return_times):>10}')
print(f'  {"Coefficient of var":<20} {np.std(return_times)/np.mean(return_times):>10.3f} '
      f'{np.std(gue_return_times)/np.mean(gue_return_times):>10.3f}')

# If return times are Poisson (memoryless), CoV = 1.
# If return times are clustered, CoV > 1.
# If return times are regular, CoV < 1.
zeta_cov = np.std(return_times) / np.mean(return_times)
gue_cov = np.std(gue_return_times) / np.mean(gue_return_times)
print(f'\n  Zeta CoV = {zeta_cov:.3f}, GUE CoV = {gue_cov:.3f}')
if zeta_cov > gue_cov + 0.05:
    print('  -> Zeta extreme gaps are MORE CLUSTERED than GUE')
elif zeta_cov < gue_cov - 0.05:
    print('  -> Zeta extreme gaps are MORE REGULAR than GUE')
else:
    print('  -> Consistent with GUE')

ks_rt, p_rt = kstest(return_times, gue_return_times)
print(f'  KS test on return times: D = {ks_rt:.4f}, p = {p_rt:.4f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: EXTREME GAP STATISTICS')
print('=' * 70)

# Count anomalies
anomalies = []

if ks_p < 0.05:
    anomalies.append(f'Overall spacing distribution differs from GUE (KS p={ks_p:.4f})')
if abs(clustering_excess) > 0.1:
    anomalies.append(f'Extreme gap clustering differs by {clustering_excess:+.3f}')
if p_rt < 0.05:
    anomalies.append(f'Return time distribution differs from GUE (KS p={p_rt:.4f})')

# Check phase correlations
for p_prime in primes_test:
    phase = np.cos(2 * np.pi * zero_heights * np.log(p_prime) / LOG_T)
    rp, pp = pearsonr(spacings, phase)
    if pp < 0.01:
        anomalies.append(f'Spacing correlates with prime {p_prime} phase (r={rp:+.4f}, p={pp:.4f})')

if not anomalies:
    print('\n>>> NO ANOMALIES DETECTED.')
    print('>>> Extreme gap statistics are fully consistent with GUE.')
    print('>>> The pair-correlation exclusivity (R_k = GUE for k >= 3) extends')
    print('>>> to the extreme tails of the spacing distribution.')
    print('>>> There is no hidden structure in the extreme gaps.')
else:
    print(f'\n>>> {len(anomalies)} ANOMALIES DETECTED:')
    for a in anomalies:
        print(f'>>>   - {a}')
    print('>>> These warrant further investigation.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
