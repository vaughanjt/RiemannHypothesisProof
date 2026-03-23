"""Generate figures for Paper 1: Finite-T Pair Correlation."""
import sys
sys.path.insert(0, '../src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize_scalar
from sympy import primerange
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# Load data
print('Loading data...')
def gue_eigs(n, rng):
    d = rng.standard_normal(n)
    e = np.sqrt(rng.chisquare(2 * np.arange(n-1, 0, -1)) / 2)
    return eigvalsh_tridiagonal(d, e) / np.sqrt(n)

rng = np.random.default_rng(42)
bl = []
for _ in range(100):
    eigs = gue_eigs(1200, rng)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > 410:
        bl.append(spacing_autocorrelation(sp, 400))
baseline = np.mean(bl, axis=0)[1:401]

zeros = []
with open('../data/odlyzko/zeros3.txt') as f:
    for line in f:
        try: zeros.append(float(line.strip()))
        except: pass
zeros = np.array(zeros)
T = 267653395647.0
LOG_T = np.log(T / (2*np.pi))
density = LOG_T / (2*np.pi)
sp = np.diff(zeros) * density
sp /= np.mean(sp)
acf = spacing_autocorrelation(sp, 400)[1:401]
excess = acf - baseline
k_arr = np.arange(1, 401, dtype=float)

# ---- FIGURE 1: ACF excess with two-component model ----
print('Figure 1: ACF excess...')

primes = list(primerange(2, 500))
def build_model(alpha):
    m = np.zeros(400)
    for p in primes:
        f = np.log(p) / LOG_T
        if f >= 0.45: continue
        m += np.log(p) / p**alpha * np.cos(2*np.pi*k_arr*f)
    return m

short = [np.exp(-k_arr/1), np.exp(-k_arr/3), 1/k_arr**2]
res = minimize_scalar(lambda a: np.sum((excess - np.column_stack([build_model(a)]+short) @ np.linalg.lstsq(np.column_stack([build_model(a)]+short), excess, rcond=None)[0])**2), bounds=(0.3,1.5), method='bounded')
alpha_opt = res.x
model = build_model(alpha_opt)
X = np.column_stack([model] + short)
amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
pred = X @ amps
osc = amps[0] * model
sr = sum(amps[i+1]*short[i] for i in range(3))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
se = 1/np.sqrt(len(sp))
ax1.fill_between(k_arr, -2*se, 2*se, color='#f0f0f0', label='$\\pm 2\\sigma$ (GUE)')
ax1.plot(k_arr[:100], excess[:100], 'k-', linewidth=0.5, alpha=0.7, label='ACF excess')
ax1.plot(k_arr[:100], pred[:100], '#d62728', linewidth=1.2, label=f'Model ($\\alpha={alpha_opt:.3f}$)')
ax1.plot(k_arr[:100], osc[:100], '#1f77b4', linewidth=0.8, alpha=0.6, label='Oscillatory')
ax1.plot(k_arr[:100], sr[:100], '#2ca02c', linewidth=0.8, alpha=0.6, label='Short-range')
ax1.set_ylabel('ACF excess')
ax1.legend(loc='upper right', fontsize=8)
ax1.set_title('Non-GUE ACF excess at $T \\sim 2.7 \\times 10^{11}$', fontsize=11)
ax1.xaxis.set_minor_locator(AutoMinorLocator())

resid = excess - pred
ax2.plot(k_arr[:100], resid[:100]/se, 'k-', linewidth=0.5)
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axhline(2.5, color='#d62728', linewidth=0.5, linestyle='--')
ax2.axhline(-2.5, color='#d62728', linewidth=0.5, linestyle='--')
ax2.set_xlabel('Lag $k$')
ax2.set_ylabel('Residual ($z$-score)')
ax2.set_xlim(0, 100)
plt.tight_layout()
plt.savefig('paper1_fig1_acf.pdf')
plt.close()
print('  Saved paper1_fig1_acf.pdf')

# ---- FIGURE 2: Amplitude law comparison ----
print('Figure 2: Amplitude laws...')

laws = {
    'Explicit $\\log(p)/\\sqrt{p}$': 0.5,
    'Selberg $\\log(p)/p$': 1.0,
    f'Fitted $\\log(p)/p^{{{alpha_opt:.3f}}}$': alpha_opt,
}
colors_law = ['#ff7f0e', '#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(5, 3.2))
ss_tot = np.sum(excess**2)
for (label, alpha), color in zip(laws.items(), colors_law):
    m = build_model(alpha)
    X_l = np.column_stack([m] + short)
    a_l, _, _, _ = np.linalg.lstsq(X_l, excess, rcond=None)
    R2 = 1 - np.sum((excess - X_l @ a_l)**2) / ss_tot
    R2a = 1 - (1-R2)*399/(399-5)
    ax.bar(label, R2a, color=color, edgecolor='#333', linewidth=0.5, width=0.6)
    ax.text(label, R2a + 0.01, f'{R2a:.3f}', ha='center', fontsize=9)

# Add BK
m_bk = np.zeros(400)
for p in primes:
    f = np.log(p)/LOG_T
    if f >= 0.45: continue
    m_bk += np.log(p)**2/p * np.cos(2*np.pi*k_arr*f)
X_bk = np.column_stack([m_bk]+short)
a_bk, _, _, _ = np.linalg.lstsq(X_bk, excess, rcond=None)
R2_bk = 1 - np.sum((excess-X_bk@a_bk)**2)/ss_tot
R2a_bk = 1-(1-R2_bk)*399/(399-5)
ax.bar('BK $\\log^2(p)/p$', R2a_bk, color='#9467bd', edgecolor='#333', linewidth=0.5, width=0.6)
ax.text('BK $\\log^2(p)/p$', R2a_bk+0.01, f'{R2a_bk:.3f}', ha='center', fontsize=9)

ax.set_ylabel('$R^2_{\\mathrm{adj}}$')
ax.set_ylim(0, 0.7)
ax.set_title('Amplitude law comparison', fontsize=11)
plt.xticks(rotation=15, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('paper1_fig2_amplitude.pdf')
plt.close()
print('  Saved paper1_fig2_amplitude.pdf')

# ---- FIGURE 3: Harmonic variance decomposition ----
print('Figure 3: Harmonic decomposition...')
fig, ax = plt.subplots(figsize=(4, 2.8))
harmonics = ['$m=1$', '$m=2$', '$m \\geq 3$', 'Short-range']
fracs = [62.2, 1.2, 0.8, 1.5]
colors_h = ['#2166ac', '#92c5de', '#d1e5f0', '#f4a582']
bars = ax.bar(harmonics, fracs, color=colors_h, edgecolor='#333', linewidth=0.5)
ax.set_ylabel('\\% of variance explained')
ax.set_title('Variance decomposition by harmonic order', fontsize=10)
for b, f in zip(bars, fracs):
    ax.text(b.get_x()+b.get_width()/2, f+0.5, f'{f:.1f}\\%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('paper1_fig3_harmonics.pdf')
plt.close()
print('  Saved paper1_fig3_harmonics.pdf')

print('Done.')
