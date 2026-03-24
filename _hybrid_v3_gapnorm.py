"""Hybrid v3: gap-normalized value-space GCD kernel.

H_{jk} = C * log(gcd(j_val, k_val)) / (sqrt(j_val * k_val) * |alpha_j - alpha_k|)

The gap normalization forces the coupling to respect the local density
of states. Strong coupling between nearby eigenvalues (where GCD
creates rigidity), weak between distant ones (where the diagonal
dominates for eigenvalue accuracy).
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
    errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    r, _ = measure_peak_gap(eigs)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    p_gue = 0.
    if len(sp)>20:
        sp = sp/np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)
    return np.mean(errs), np.mean(errs<ms/2), r, p_gue

# Diagonal
print("Building explicit formula diagonal...", flush=True)
alpha = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k); dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes for m in range(1,6)) / np.pi
    alpha[k-1] = Tw + s / dN

round_alpha = np.maximum(np.round(alpha).astype(int), 1)


def build_gapnorm(alpha_vals, C, W=None, weight="log"):
    """Gap-normalized value-space GCD.

    H_{jk} = C * w(gcd) / (sqrt(j_val * k_val) * |alpha_j - alpha_k|)
    """
    ra = np.maximum(np.round(alpha_vals).astype(int), 1)
    n = len(alpha_vals)
    H = np.diag(alpha_vals.copy())

    for j in range(n):
        jv = ra[j]
        k_range = range(j+1, min(j+W+1, n)) if W else range(j+1, n)
        for k in k_range:
            kv = ra[k]
            g = gcd(jv, kv)
            if g <= 1:
                continue
            gap = abs(alpha_vals[j] - alpha_vals[k])
            if gap < 1e-8:
                gap = 1e-8

            if weight == "log":
                w = np.log(g)
            elif weight == "log1p":
                w = np.log(g + 1)
            elif weight == "sqrt":
                w = np.sqrt(g)
            elif weight == "linear":
                w = float(g)
            else:
                w = np.log(g)

            val = C * w / (np.sqrt(max(jv,1) * max(kv,1)) * gap)
            H[j, k] = val
            H[k, j] = val

    return H


# ============================================================
# SWEEP: C and W for gap-normalized kernel
# ============================================================
print("\n" + "="*70, flush=True)
print("MAIN SWEEP: GAP-NORMALIZED VALUE-SPACE GCD", flush=True)
print("="*70, flush=True)

print(f"\n  {'W':>5} {'C':>8} {'err':>8} {'r':>8} {'p_gue':>8} "
      f"{'<half':>8} {'TARGET?':>10}", flush=True)
print(f"  {'-'*56}", flush=True)

candidates = []

for W in [3, 5, 10, 20, 50, None]:
    W_label = str(W) if W else "full"
    for C in [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]:
        H = build_gapnorm(alpha, C, W)
        eigs = np.sort(np.linalg.eigvalsh(H))
        err, ph, r, pg = score_full(eigs)

        target = ""
        if r >= 0.60 and pg >= 0.01:
            target = "*** YES ***"
            candidates.append((W_label, C, err, r, pg, ph))
        elif r >= 0.40 and pg >= 0.01:
            target = "close"
        elif r >= 0.60:
            target = "r ok"

        if r > 0.05 or C <= 1:
            print(f"  {W_label:>5} {C:>8.2f} {err:>8.4f} {r:>+8.4f} "
                  f"{pg:>8.4f} {ph*100:>7.1f}% {target:>10}", flush=True)


# ============================================================
# FINE SWEEP around any candidates
# ============================================================
if candidates:
    print("\n" + "="*70, flush=True)
    print("CANDIDATES FOUND! Fine-tuning...", flush=True)
    print("="*70, flush=True)

    for W_label, C_base, _, _, _, _ in candidates[:3]:
        W_val = int(W_label) if W_label != "full" else None
        for C in np.linspace(C_base*0.3, C_base*3, 30):
            H = build_gapnorm(alpha, C, W_val)
            eigs = np.sort(np.linalg.eigvalsh(H))
            err, ph, r, pg = score_full(eigs)
            if r >= 0.60 and pg >= 0.01:
                print(f"  W={W_label:>5} C={C:>8.3f} err={err:.4f} r={r:+.4f} "
                      f"p(GUE)={pg:.4f} <half={ph*100:.1f}%", flush=True)
else:
    print("\n  No candidates hit r>=0.60 AND p(GUE)>=0.01 in main sweep.", flush=True)
    print("  Showing best compromises:", flush=True)

    # Find Pareto frontier
    all_results = []
    for W in [5, 10, 20, None]:
        W_label = str(W) if W else "full"
        for C in np.logspace(-2, 2, 50):
            H = build_gapnorm(alpha, C, W)
            eigs = np.sort(np.linalg.eigvalsh(H))
            err, ph, r, pg = score_full(eigs)
            all_results.append((W_label, C, err, r, pg, ph))

    # Sort by r * min(pg, 0.05) — rewards both
    all_results.sort(key=lambda x: -x[3] * min(x[4], 0.05))
    print(f"\n  {'W':>5} {'C':>8} {'err':>8} {'r':>8} {'p_gue':>8} {'<half':>8}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for row in all_results[:20]:
        print(f"  {row[0]:>5} {row[1]:>8.3f} {row[2]:>8.4f} {row[3]:>+8.4f} "
              f"{row[4]:>8.4f} {row[5]*100:>7.1f}%", flush=True)


# ============================================================
# WEIGHT FUNCTION COMPARISON at best C
# ============================================================
print("\n" + "="*70, flush=True)
print("WEIGHT FUNCTION COMPARISON", flush=True)
print("="*70, flush=True)

for C_test in [1, 5, 20]:
    print(f"\n  C={C_test}:", flush=True)
    for wt in ["log", "log1p", "sqrt", "linear"]:
        H = build_gapnorm(alpha, C_test, W=10, weight=wt)
        eigs = np.sort(np.linalg.eigvalsh(H))
        err, ph, r, pg = score_full(eigs)
        print(f"    {wt:>8}: err={err:.4f} r={r:+.4f} p(GUE)={pg:.4f} <half={ph*100:.1f}%", flush=True)


print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
