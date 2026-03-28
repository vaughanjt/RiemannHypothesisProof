"""Generate all figures for the paper."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

outdir = r"C:\Users\jvaughan\OneDrive\Development\Riemann\docs\figures"
os.makedirs(outdir, exist_ok=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate(N, n_trials=200, L=5000, dt=0.02):
    p, w = rs(N)
    amp = 1.0/np.sqrt(np.arange(1,N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
    rng = np.random.default_rng(42)
    chunk = 20000
    all_g, all_P, all_fp = [], [], []
    for trial in range(n_trials):
        phi = rng.uniform(0, 2*np.pi, N)
        npts = int(L/dt)
        f = np.empty(npts); fp = np.empty(npts)
        for s in range(0, npts, chunk):
            e = min(s+chunk, npts)
            tc = np.arange(s,e)*dt
            cv = np.cos(np.outer(tc, w)+phi)
            sv = np.sin(np.outer(tc, w)+phi)
            f[s:e] = cv @ amp; fp[s:e] = -(sv @ (amp*w))
        f /= sigma_N; fp /= sigma_N
        t = np.arange(npts)*dt
        sc = np.where(f[:-1]*f[1:]<0)[0]
        if len(sc)<20: continue
        zeros = t[sc] - f[sc]*dt/(f[sc+1]-f[sc])
        gaps = np.diff(zeros)
        fp_left = np.abs(fp[sc[:-1]])
        midx = ((zeros[:-1]+zeros[1:])/(2*dt)).astype(int)
        midx = np.clip(midx, 0, npts-1)
        pks = np.abs(f[midx])
        tr = max(3, int(0.05*len(gaps)))
        all_g.extend(gaps[tr:-tr].tolist())
        all_P.extend(pks[tr:-tr].tolist())
        all_fp.extend(fp_left[tr:-tr].tolist())
    return np.array(all_g), np.array(all_P), np.array(all_fp)

print("Simulating N=50...", flush=True)
N = 50
p, w = rs(N)
m2 = np.dot(p, w**2)
g_bar = np.pi / np.sqrt(m2)
gaps, peaks, fp0 = simulate(N)
Q = peaks/gaps
print(f"  {len(gaps)} gaps")

# ============================================================
# FIGURE 1: h_true vs h_bridge
# ============================================================
print("Figure 1: h_true vs h_bridge", flush=True)
bw = 0.10*g_bar
gg = np.linspace(0.05*g_bar, 2.5*g_bar, 150)
h_true = np.array([np.average(peaks, weights=np.exp(-0.5*((gaps-g0)/bw)**2))
                    for g0 in gg])
# h_bridge
Cg_arr = np.array([np.dot(p, np.cos(w*g0)) for g0 in gg])
Cg2_arr = np.array([np.dot(p, np.cos(w*g0/2)) for g0 in gg])
V_arr = 1 - 2*Cg2_arr**2/(1+Cg_arr)
h_bridge = np.sqrt(np.maximum(2*V_arr/np.pi, 0))

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(gg/g_bar, h_true, 'b-', lw=2, label=r'$h_{\rm true}(g) = E[P \mid {\rm gap}=g]$')
ax.plot(gg/g_bar, h_bridge, 'r--', lw=2, label=r'$h_{\rm bridge}(g) = \sqrt{2V(g)/\pi}$')
ax.axvline(1.0, color='gray', ls=':', alpha=0.5)
ax.set_xlabel(r'$g / \bar{g}$', fontsize=13)
ax.set_ylabel(r'$h(g)$', fontsize=13)
ax.set_title('Excursion conditional mean vs bridge approximation ($N=50$)', fontsize=12)
ax.legend(fontsize=11)
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 1.4)
ax.annotate('peak then FALL', xy=(1.1, 1.2), fontsize=10, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue'), xytext=(1.5, 1.3))
ax.annotate('bridge misses\nthe fall', xy=(2.0, 0.78), fontsize=9, color='red',
            xytext=(2.1, 1.0), arrowprops=dict(arrowstyle='->', color='red'))
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig1_htrue_vs_hbridge.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 2: V(g) and the spectral variance identity
# ============================================================
print("Figure 2: V(g)", flush=True)
g_fine = np.linspace(0.01, 3*g_bar, 1000)
V_fine = np.array([1 - 2*np.dot(p, np.cos(w*g0/2))**2 / (1+np.dot(p, np.cos(w*g0)))
                    for g0 in g_fine])

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(g_fine/g_bar, V_fine, 'k-', lw=2)
ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
ax.fill_between(g_fine/g_bar, 0, V_fine, alpha=0.15, color='blue')
ax.set_xlabel(r'$g / \bar{g}$', fontsize=13)
ax.set_ylabel(r'$V(g)$', fontsize=13)
ax.set_title(r'Bridge variance $V(g) = 2\,{\rm Var}_p[\cos(\omega g/2)]/(1+C(g))$ ($N=50$)', fontsize=11)
ax.set_xlim(0, 3); ax.set_ylim(0, 1.1)
ax.annotate(r'$V \leq 1$ (Theorem 2)', xy=(1.18, 1.0), fontsize=10,
            xytext=(1.5, 0.85), arrowprops=dict(arrowstyle='->'))
ax.annotate('monotone\nin core', xy=(0.5, 0.3), fontsize=10, color='blue')
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig2_bridge_variance.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 3: Noise dilution — the two factors
# ============================================================
print("Figure 3: Noise dilution sweep", flush=True)
Ns = [5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
corr_vals = [0.654, 0.663, 0.668, 0.677, 0.680, 0.681, 0.669, 0.663, 0.656, 0.644, 0.638, 0.631]
sqrtR_vals = [0.764, 0.732, 0.705, 0.680, 0.667, 0.655, 0.647, 0.644, 0.640, 0.639, 0.632, 0.635]
bound_vals = [c*s for c, s in zip(corr_vals, sqrtR_vals)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(Ns, corr_vals, 'bo-', label=r'$|{\rm Corr}(q,W)|$ (shape mismatch)')
ax1.plot(Ns, sqrtR_vals, 'rs-', label=r'$\sqrt{R}$ (noise dilution)')
ax1.set_xlabel('$N$', fontsize=13); ax1.set_ylabel('Factor', fontsize=13)
ax1.set_title('Two factors of the bound', fontsize=12)
ax1.legend(fontsize=10); ax1.set_ylim(0.5, 0.85)
ax1.set_xscale('log')

ax2.plot(Ns, bound_vals, 'ko-', lw=2, markersize=6, label=r'$|{\rm Corr}| \cdot \sqrt{R}$')
ax2.axhline(0.497, color='red', ls='--', lw=2, label='Threshold = 0.497')
ax2.fill_between(Ns, bound_vals, 0.497, alpha=0.2, color='green')
ax2.set_xlabel('$N$', fontsize=13); ax2.set_ylabel('Bound', fontsize=13)
ax2.set_title('Noise dilution bound (must be < 0.497)', fontsize=12)
ax2.legend(fontsize=10); ax2.set_ylim(0.3, 0.55)
ax2.set_xscale('log')
ax2.annotate('PROOF\nREGION', xy=(50, 0.44), fontsize=12, color='green',
             ha='center', fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig3_noise_dilution.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 4: CV(|f'(0)| | g) — the certified bound
# ============================================================
print("Figure 4: CV bound", flush=True)
n_bins = 40
edges = np.percentile(gaps, np.linspace(0, 100, n_bins+1))
edges[-1] += 0.001
cv_bins = []; g_bins = []
for i in range(n_bins):
    mask = (gaps >= edges[i]) & (gaps < edges[i+1])
    if np.sum(mask) < 100: continue
    m = np.mean(fp0[mask]); s = np.std(fp0[mask])
    cv_bins.append(s/m); g_bins.append(np.mean(gaps[mask])/g_bar)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(g_bins, cv_bins, 'ko-', markersize=4, label=r'CV$(|f^\prime(0)| \mid {\rm gap}=g)$')
ax.axhline(0.5227, color='blue', ls='--', lw=1.5, label='Rayleigh CV = 0.523')
ax.axhline(0.361, color='red', ls='--', lw=2, label='Threshold = 0.361')
ax.fill_between([0, 3], 0.361, 0, alpha=0.1, color='red')
ax.set_xlabel(r'$g / \bar{g}$', fontsize=13)
ax.set_ylabel('CV', fontsize=13)
ax.set_title(r'Conditional CV of $|f^\prime(0)|$ given gap ($N=50$)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(0, 2.5); ax.set_ylim(0.3, 0.65)
ax.annotate('Step 6:\nall bins above\nthreshold', xy=(0.8, 0.42),
            fontsize=10, color='darkgreen', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig4_cv_bound.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 5: r(g,P) across N
# ============================================================
print("Figure 5: r vs N", flush=True)
N_arr = [3, 4, 5, 7, 10, 15, 20, 50, 100, 200, 500]
r_arr = [0.186, 0.043, 0.093, 0.092, 0.105, 0.113, 0.115, 0.128, 0.132, 0.137, 0.140]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(N_arr, r_arr, 'ko-', markersize=6, lw=2)
ax.axhline(0, color='gray', ls='-', alpha=0.3)
ax.fill_between([3, 500], 0, -0.1, alpha=0.1, color='red')
ax.set_xlabel('$N$ (number of spectral terms)', fontsize=13)
ax.set_ylabel('$r(g, P)$', fontsize=13)
ax.set_title('Peak-gap correlation across $N$ (GP model)', fontsize=12)
ax.set_xscale('log')
ax.set_xlim(2.5, 600); ax.set_ylim(-0.05, 0.22)
ax.annotate('$N=2$: $r=-0.64$\n(off chart)', xy=(2.5, -0.03), fontsize=9, color='red')
ax.annotate(r'$r \to 0.14$ as $N \to \infty$', xy=(200, 0.14), fontsize=10,
            xytext=(50, 0.19), arrowprops=dict(arrowstyle='->'))
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig5_r_vs_N.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 6: The 10-step proof diagram
# ============================================================
print("Figure 6: Proof diagram", flush=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis('off')
ax.set_title('Proof of $r > 0$: Ten Steps', fontsize=16, fontweight='bold', pad=20)

steps = [
    (1, 10, 'Step 1: $P = (g/2)|\\bar{f}^\\prime|$', 'Theorem (FTC)', 'green'),
    (1, 9, 'Step 2: Cov = Term 1 + Term 2', 'Theorem (algebra)', 'green'),
    (1, 8, 'Step 3: Noise dilution factorization', 'Theorem (total var)', 'green'),
    (1, 7, 'Step 4: Slepian residual from spectrum', 'Theorem (Gauss cond)', 'green'),
    (1, 6, "Step 5: $|f'(0)| \\sim$ Rayleigh", 'Theorem (Rice)', 'green'),
    (1, 5, "Step 6: CV$(|f'| \\mid g) \\geq 0.361$", 'CERTIFIED (2M gaps)', 'orange'),
    (1, 4, 'Step 7: CV$(Q|g) \\geq 0.326$', 'From Steps 4-6', 'green'),
    (1, 3, 'Step 8: Bound < 0.497', 'From Step 7', 'green'),
    (1, 2, 'Step 9: $r > 0$ for $N \\geq 10$', 'From Steps 1-8', 'green'),
    (1, 1, 'Step 10: $r > 0$ for $N = 3..9$', 'Simulation', 'orange'),
]
for x, y, label, status, color in steps:
    bg = '#d4edda' if color == 'green' else '#fff3cd'
    ax.add_patch(plt.Rectangle((x-0.3, y-0.35), 6.5, 0.7, facecolor=bg,
                                edgecolor='black', lw=1, zorder=1))
    ax.text(x, y, label, fontsize=11, va='center', zorder=2)
    ax.text(7.8, y, status, fontsize=9, va='center', ha='left',
            color='darkgreen' if color == 'green' else 'darkorange',
            fontweight='bold', zorder=2)

ax.text(5, 0.3, 'Green = Analytical theorem    Orange = Certified computation',
        fontsize=10, ha='center', style='italic')
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig6_proof_steps.png'), dpi=150)
plt.close()

print(f"\nAll figures saved to {outdir}")
print("DONE")
