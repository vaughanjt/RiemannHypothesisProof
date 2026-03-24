"""THE HYBRID OPERATOR: explicit formula diagonal + GCD off-diagonal.

Architecture (Grok-concurred):
  Diagonal: alpha_k = weyl(k) + S(weyl_k) / N'(weyl_k)  [locks eigenvalues]
  Off-diagonal: H_{jk} = eps * log(gcd(j,k)) / sqrt(j*k)  [locks eigenvectors]

The GCD kernel from session 5 achieved r=+0.55 with GUE-compatible spacings.
The explicit formula diagonal gives eigenvalue accuracy to <1 spacing.
Combined: eigenvalues from the explicit formula, eigenvectors from GCD.

Single free parameter: eps (the GCD coupling strength).
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd
from scipy.linalg import eigh
from scipy.stats import pearsonr, kstest
from scipy.optimize import minimize_scalar
import mpmath
mpmath.mp.dps = 20

t0 = time.time()
from riemann.analysis.bost_connes_operator import polynomial_unfold

N = 200
print(f"Computing {N} zeta zeros...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N+1)])

from sympy import primerange
primes = list(primerange(2, 3000))[:303]

trim = int(0.1*N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

def N_smooth(T):
    if T < 2: return 0.
    return T/(2*np.pi)*np.log(T/(2*np.pi)) - T/(2*np.pi) + 7./8.
def N_deriv(T):
    if T < 2: return .001
    return np.log(T/(2*np.pi)) / (2*np.pi)
def weyl_zero(k):
    t = 2*np.pi*k/np.log(max(k,2)+2)
    for _ in range(30):
        if t<1: t=10.
        t -= (N_smooth(t)-k)/N_deriv(t)
    return t

def wigner_cdf(s):
    return 1 - np.exp(-np.pi*s**2/4)

def measure_peak_gap(eigs_raw):
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20: return 0., 0
    sp = sp / np.mean(sp)
    nt = int(0.1*len(eigs)); et = eigs[nt:-nt]
    lp, ga = [], []
    for k in range(min(len(sp), len(et)-1)):
        z = (et[k]+et[k+1])/2
        lp.append(np.sum(np.log(np.abs(z-eigs)+1e-30)))
        ga.append(sp[k])
    if len(ga)<10: return 0., 0
    return pearsonr(np.array(ga), np.array(lp))[0], len(ga)

def score_full(eigs):
    """All metrics."""
    errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    r, _ = measure_peak_gap(eigs)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp)>20:
        sp = sp/np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)
    else:
        p_gue = 0.
    return np.mean(errs), np.mean(errs<ms/2), r, p_gue


# ============================================================
# Build explicit formula diagonal
# ============================================================
print("Building explicit formula diagonal...", flush=True)

alpha = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k); dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes for m in range(1,6)) / np.pi
    alpha[k-1] = Tw + s / dN


# ============================================================
# Build GCD kernel
# ============================================================
print("Building GCD kernel...", flush=True)

def build_gcd_kernel(N_size):
    """H_{jk} = log(gcd(j,k)+1) / sqrt(j*k)"""
    G = np.zeros((N_size, N_size))
    for j in range(1, N_size+1):
        for k in range(j, N_size+1):
            g = gcd(j, k)
            val = np.log(g + 1) / np.sqrt(j * k)
            G[j-1, k-1] = val
            G[k-1, j-1] = val
    return G

GCD = build_gcd_kernel(N)
print(f"  GCD kernel built: {N}x{N}, ||GCD||_2 = {np.linalg.norm(GCD, ord=2):.4f}", flush=True)


# ============================================================
# THE HYBRID: explicit diagonal + eps * GCD off-diagonal
# ============================================================

def build_hybrid(alpha_diag, gcd_matrix, eps):
    """H = diag(alpha) + eps * (GCD - diag(GCD))"""
    G_offdiag = gcd_matrix - np.diag(np.diag(gcd_matrix))
    return np.diag(alpha_diag) + eps * G_offdiag


# ============================================================
# SWEEP: eps from 0 to 100
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP: HYBRID OPERATOR eps", flush=True)
print("="*70, flush=True)

print(f"\n  {'eps':>8} {'mean_err':>10} {'r':>8} {'p(GUE)':>8} "
      f"{'<half':>8} {'<gap':>8}", flush=True)
print(f"  {'-'*54}", flush=True)

best_r = 0
best_eps_r = 0
best_combined = 1e10
best_eps_combined = 0

for eps in [0, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 500]:
    H = build_hybrid(alpha, GCD, eps)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err, pct_h, r, p_gue = score_full(eigs)

    marker = ""
    if r > best_r:
        best_r = r; best_eps_r = eps; marker = " <-- best r"
    # Combined: want low error AND high r AND p_gue > 0.01
    penalty = max(0, 0.01 - p_gue) * 100 + max(0, 0.5 - r) * 2
    combined = err + penalty
    if combined < best_combined:
        best_combined = combined; best_eps_combined = eps

    print(f"  {eps:>8.1f} {err:>10.4f} {r:>+8.4f} {p_gue:>8.4f} "
          f"{pct_h*100:>7.1f}% {(np.mean(np.abs(np.sort(np.linalg.eigvalsh(H))-zeta_zeros[:N])[trim:-trim]<ms))*100:>7.1f}%{marker}", flush=True)


# ============================================================
# FINE SWEEP around the best region
# ============================================================
print("\n" + "="*70, flush=True)
print(f"FINE SWEEP around eps ~ {best_eps_r}", flush=True)
print("="*70, flush=True)

low = max(0.1, best_eps_r * 0.3)
high = best_eps_r * 3 + 1
fine_eps = np.linspace(low, high, 40)

print(f"\n  {'eps':>8} {'mean_err':>10} {'r':>8} {'p(GUE)':>8} {'<half':>8}", flush=True)
print(f"  {'-'*46}", flush=True)

best_r_fine = 0
best_eps_fine = 0

for eps in fine_eps:
    H = build_hybrid(alpha, GCD, eps)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err, pct_h, r, p_gue = score_full(eigs)

    if r > best_r_fine:
        best_r_fine = r
        best_eps_fine = eps
        print(f"  {eps:>8.2f} {err:>10.4f} {r:>+8.4f} {p_gue:>8.4f} "
              f"{pct_h*100:>7.1f}%  <-- NEW BEST r", flush=True)

print(f"\n  Best: eps={best_eps_fine:.2f}, r={best_r_fine:+.4f}", flush=True)


# ============================================================
# DETAILED ANALYSIS at best eps
# ============================================================
print("\n" + "="*70, flush=True)
print(f"DETAILED ANALYSIS at eps = {best_eps_fine:.2f}", flush=True)
print("="*70, flush=True)

H_best = build_hybrid(alpha, GCD, best_eps_fine)
eigs_best = np.sort(np.linalg.eigvalsh(H_best))
err, pct_h, r, p_gue = score_full(eigs_best)

print(f"\n  METRICS:", flush=True)
print(f"    Peak-gap r:   {r:+.4f}  (target: > 0.65)", flush=True)
print(f"    KS p(GUE):    {p_gue:.4f}  (target: > 0.01)", flush=True)
print(f"    Mean error:    {err:.4f}", flush=True)
print(f"    < half gap:    {pct_h*100:.1f}%", flush=True)

# Eigenvector IPR
_, vecs = np.linalg.eigh(H_best)
ipr = np.sum(vecs**4, axis=0)
print(f"    Mean IPR:      {np.mean(ipr):.6f} (GUE: {3./N:.6f})", flush=True)
print(f"    IPR/GUE:       {np.mean(ipr)/(3./N):.2f}x", flush=True)

# First 15 eigenvalues
print(f"\n  {'k':>4} {'Eigenvalue':>12} {'Zero':>12} {'Error':>10}", flush=True)
for i in range(15):
    e = abs(eigs_best[i] - zeta_zeros[i])
    tag = " <<<" if e < 0.3 else ""
    print(f"  {i+1:>4} {eigs_best[i]:>12.4f} {zeta_zeros[i]:>12.4f} {e:>10.4f}{tag}", flush=True)


# ============================================================
# Also test: GCD diagonal (log(k)) instead of explicit formula
# ============================================================
print("\n" + "="*70, flush=True)
print("COMPARISON: explicit diagonal vs log(k) diagonal", flush=True)
print("="*70, flush=True)

# The inverse spectral result from session 5: diagonal = log(k)
alpha_logk = np.log(np.arange(1, N+1))

# Scale log(k) to match zero range
alpha_logk_scaled = alpha_logk / alpha_logk[-1] * zeta_zeros[-1]

for diag_name, diag_vals in [("explicit formula", alpha),
                               ("log(k) scaled", alpha_logk_scaled),
                               ("zeros (oracle)", zeta_zeros[:N])]:
    # Find best eps for each diagonal
    def obj_d(log_eps, dv=diag_vals):
        H = build_hybrid(dv, GCD, np.exp(log_eps))
        eigs = np.sort(np.linalg.eigvalsh(H))
        err, _, r, p = score_full(eigs)
        return err - 0.5*max(r, 0)  # balance error and r

    res = minimize_scalar(obj_d, bounds=(-2, 7), method='bounded')
    eps_opt = np.exp(res.x)

    H = build_hybrid(diag_vals, GCD, eps_opt)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err, pct_h, r, p_gue = score_full(eigs)

    print(f"\n  {diag_name:>20}: eps={eps_opt:.2f}, err={err:.4f}, "
          f"r={r:+.4f}, p(GUE)={p_gue:.4f}, <half={pct_h*100:.1f}%", flush=True)


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("VERDICT: THE HYBRID OPERATOR", flush=True)
print("="*70, flush=True)

print(f"""
  H = diag(alpha) + eps * GCD_offdiag

  alpha_k = weyl(k) + S(weyl_k, 303 primes) / N'(weyl_k)
  GCD_{jk} = log(gcd(j,k)+1) / sqrt(j*k)   (j != k)
  eps = {best_eps_fine:.2f}

  Results at N={N}:
    Peak-gap r:   {r:+.4f}
    KS p(GUE):    {p_gue:.4f}
    Mean error:    {err:.4f}
    < half gap:    {pct_h*100:.1f}%
""", flush=True)

print(f"Total time: {time.time()-t0:.1f}s", flush=True)
