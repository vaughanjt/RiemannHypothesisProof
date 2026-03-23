"""Generate figures and bootstrap statistics for the eigenvector rigidity paper."""
import sys
sys.path.insert(0, '../src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import pearsonr
from scipy.linalg import eigvalsh_tridiagonal
import mpmath
from riemann.analysis.bost_connes_operator import polynomial_unfold

mpmath.mp.dps = 20
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# ============================================================
# LOAD / COMPUTE DATA
# ============================================================
print('Loading zeta zeros...')
zeros = np.load('../_zeros_500.npy')
gaps = np.diff(zeros)
LOG_T = np.log(np.mean(zeros) / (2 * np.pi))
mean_gap = 2 * np.pi / LOG_T
norm_gaps = gaps / mean_gap

Z_mid = np.zeros(len(gaps))
for i in range(len(gaps)):
    t = (zeros[i] + zeros[i + 1]) / 2
    Z_mid[i] = float(mpmath.siegelz(t))
peak = np.abs(Z_mid)
log_peak = np.log(peak + 1e-10)

print('Computing GUE comparison...')
rng = np.random.default_rng(42)
gue_gaps_all, gue_logpeak_all = [], []
for _ in range(100):
    A = rng.standard_normal((200, 200)) + 1j * rng.standard_normal((200, 200))
    H = (A + A.conj().T) / (2 * np.sqrt(400))
    eigs = np.sort(np.linalg.eigvalsh(H))
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20:
        continue
    sp = sp / np.mean(sp)
    n_trim = int(0.1 * len(eigs))
    eigs_trim = eigs[n_trim:-n_trim]
    for k in range(min(len(sp), len(eigs_trim) - 1)):
        z_mid = (eigs_trim[k] + eigs_trim[k + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        gue_gaps_all.append(sp[k])
        gue_logpeak_all.append(log_det)

gue_gaps = np.array(gue_gaps_all)
gue_logpeak = np.array(gue_logpeak_all)
# Normalize GUE log-peaks to zero mean for visual comparison
gue_logpeak_centered = gue_logpeak - np.mean(gue_logpeak)
gue_logpeak_norm = gue_logpeak_centered / np.std(gue_logpeak_centered)
log_peak_norm = (log_peak - np.mean(log_peak)) / np.std(log_peak)
norm_gaps_zeta_norm = (norm_gaps - np.mean(norm_gaps)) / np.std(norm_gaps)
gue_gaps_norm = (gue_gaps - np.mean(gue_gaps)) / np.std(gue_gaps)

# ============================================================
# FIGURE 1: Scatter plot — gap vs log|Z| for zeta and GUE
# ============================================================
print('Generating Figure 1: scatter plot...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=False)

# Left: Zeta zeros
ax1.scatter(norm_gaps, log_peak, s=4, alpha=0.5, c='#2166ac', edgecolors='none',
            rasterized=True)
# Fit line
m, b = np.polyfit(norm_gaps, log_peak, 1)
x_fit = np.linspace(0.2, 2.5, 100)
ax1.plot(x_fit, m * x_fit + b, 'k-', linewidth=1.2, alpha=0.8)
r_z, _ = pearsonr(norm_gaps, log_peak)
ax1.set_xlabel('Normalized gap $\\Delta_k$')
ax1.set_ylabel('$\\log|Z(m_k)|$')
ax1.set_title(f'Zeta zeros ($T \\sim 458$, $r = {r_z:+.3f}$)', fontsize=10)
ax1.set_xlim(0, 2.8)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# Right: GUE
# Subsample GUE for visual clarity
idx = rng.choice(len(gue_gaps), size=2000, replace=False)
ax2.scatter(gue_gaps[idx], gue_logpeak[idx], s=2, alpha=0.3, c='#b2182b',
            edgecolors='none', rasterized=True)
m_g, b_g = np.polyfit(gue_gaps, gue_logpeak, 1)
x_fit_g = np.linspace(0, 4, 100)
ax2.plot(x_fit_g, m_g * x_fit_g + b_g, 'k-', linewidth=1.2, alpha=0.8)
r_g, _ = pearsonr(gue_gaps, gue_logpeak)
ax2.set_xlabel('Normalized gap')
ax2.set_ylabel('$\\log|\\det(z_{\\mathrm{mid}} - H)|$')
ax2.set_title(f'GUE ($N = 200$, $r = {r_g:+.3f}$)', fontsize=10)
ax2.set_xlim(0, 4)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

plt.tight_layout(w_pad=1.5)
plt.savefig('fig1_scatter.pdf')
plt.close()
print('  Saved fig1_scatter.pdf')

# ============================================================
# FIGURE 2: Phase hierarchy bar chart
# ============================================================
print('Generating Figure 2: phase hierarchy...')
fig, ax = plt.subplots(figsize=(4.5, 2.5))
labels = ['Level 0\n$\\frac{t}{2}\\log\\frac{t}{2\\pi e}$',
          'Level 1\n$+ (-\\pi/8)$',
          'Level 2\n$+ \\mathrm{Stirling}$',
          'Level 3\nfull $\\theta(t)$']
values = [0.670, 0.699, 0.699, 0.699]
colors = ['#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
bars = ax.bar(range(4), values, color=colors, edgecolor='#333333', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('$r(|Z|, \\mathrm{gap})$')
ax.set_ylim(0, 0.8)
ax.axhline(y=0.04, color='#b2182b', linestyle='--', linewidth=0.8, label='GUE ($r = 0.04$)')
ax.legend(loc='upper left', fontsize=8)
# Annotate the jump
ax.annotate('$+0.029$\nfrom $-\\pi/8$', xy=(0.5, 0.685), xytext=(1.5, 0.55),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8))
plt.tight_layout()
plt.savefig('fig2_phase_hierarchy.pdf')
plt.close()
print('  Saved fig2_phase_hierarchy.pdf')

# ============================================================
# BOOTSTRAP: confidence intervals on beta
# ============================================================
print('Bootstrap on beta...')
n_boot = 10000

# Low-T beta
log_gaps = np.log(norm_gaps[norm_gaps > 0.1])
log_peaks_filt = log_peak[norm_gaps > 0.1]
n_data = len(log_gaps)

betas_low = np.zeros(n_boot)
for b in range(n_boot):
    idx_b = rng.choice(n_data, size=n_data, replace=True)
    betas_low[b] = np.polyfit(log_gaps[idx_b], log_peaks_filt[idx_b], 1)[0]

beta_low_med = np.median(betas_low)
beta_low_lo = np.percentile(betas_low, 2.5)
beta_low_hi = np.percentile(betas_low, 97.5)

print(f'  Low-T beta: {beta_low_med:.3f} [{beta_low_lo:.3f}, {beta_low_hi:.3f}] (95% CI)')

# Also bootstrap r
rs_low = np.zeros(n_boot)
for b in range(n_boot):
    idx_b = rng.choice(len(norm_gaps), size=len(norm_gaps), replace=True)
    rs_low[b], _ = pearsonr(norm_gaps[idx_b], log_peak[idx_b])
r_low_lo = np.percentile(rs_low, 2.5)
r_low_hi = np.percentile(rs_low, 97.5)
print(f'  Low-T r: {np.median(rs_low):.3f} [{r_low_lo:.3f}, {r_low_hi:.3f}] (95% CI)')

print('\nDone. Insert these into the LaTeX:')
print(f'  beta (low-T) = {beta_low_med:.2f}, 95% bootstrap CI [{beta_low_lo:.2f}, {beta_low_hi:.2f}]')
print(f'  r (low-T) = {np.median(rs_low):.3f}, 95% bootstrap CI [{r_low_lo:.3f}, {r_low_hi:.3f}]')
