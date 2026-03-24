"""Hybrid v2: value-space GCD kernel — the basis-aligned operator.

THE FIX: Compute GCD on the ZERO HEIGHTS, not the matrix indices.
  GCD(j,k) in index space -> gcd(round(alpha_j), round(alpha_k)) in value space.

This aligns the multiplicative kernel with the explicit formula diagonal
so both operate in the same coordinate system.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd
from scipy.linalg import eigh
from scipy.stats import pearsonr, kstest
from scipy.optimize import minimize_scalar, minimize
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


# ============================================================
# Explicit formula diagonal
# ============================================================
print("Building explicit formula diagonal...", flush=True)
alpha = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k); dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes for m in range(1,6)) / np.pi
    alpha[k-1] = Tw + s / dN

round_alpha = np.maximum(np.round(alpha).astype(int), 1)
print(f"  Diagonal range: [{alpha[0]:.1f}, {alpha[-1]:.1f}]", flush=True)
print(f"  Rounded range:  [{round_alpha[0]}, {round_alpha[-1]}]", flush=True)


# ============================================================
# VALUE-SPACE GCD KERNEL
# ============================================================
print("Building value-space GCD kernel...", flush=True)

def build_valuespace_gcd(alpha_vals, W, C, weight_type="log"):
    """GCD kernel computed on rounded zero heights, not indices.

    H_{k,k+d} = C * w(gcd(round(alpha_k), round(alpha_{k+d}))) / sqrt(alpha_k * alpha_{k+d})
    """
    r_alpha = np.maximum(np.round(alpha_vals).astype(int), 1)
    n = len(alpha_vals)
    H = np.diag(alpha_vals)

    for k in range(n):
        jv = r_alpha[k]
        for d in range(1, W+1):
            if k+d >= n: continue
            kv = r_alpha[k+d]
            g = gcd(jv, kv)
            if weight_type == "log":
                w = np.log(g + 1) / np.sqrt(max(jv, 1) * max(kv, 1))
            elif weight_type == "raw":
                w = float(g) / np.sqrt(max(jv, 1) * max(kv, 1))
            elif weight_type == "indicator":
                w = 1.0 / np.sqrt(max(jv, 1) * max(kv, 1)) if g > 1 else 0.0
            elif weight_type == "vonmangoldt":
                # Lambda(gcd): log(p) if gcd is a prime power, else 0
                w = 0.0
                if g > 1:
                    temp = g
                    for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
                        if temp == 1: break
                        if temp % p == 0:
                            count = 0
                            while temp % p == 0:
                                temp //= p; count += 1
                            if temp == 1:  # g = p^count
                                w = np.log(p) / np.sqrt(max(jv, 1) * max(kv, 1))
                            break
            else:
                w = np.log(g + 1) / np.sqrt(max(jv, 1) * max(kv, 1))

            H[k, k+d] = C * w
            H[k+d, k] = C * w

    return H


# Also build FULL value-space GCD (not just banded)
def build_valuespace_gcd_full(alpha_vals, C, weight_type="log"):
    """Full (non-banded) value-space GCD matrix."""
    r_alpha = np.maximum(np.round(alpha_vals).astype(int), 1)
    n = len(alpha_vals)
    H = np.diag(alpha_vals)

    for j in range(n):
        jv = r_alpha[j]
        for k in range(j+1, n):
            kv = r_alpha[k]
            g = gcd(jv, kv)
            if g <= 1: continue  # skip trivial gcd=1
            w = np.log(g + 1) / np.sqrt(max(jv, 1) * max(kv, 1))
            H[j, k] = C * w
            H[k, j] = C * w

    return H


# ============================================================
# SWEEP 1: eps for banded value-space GCD
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 1: BANDED VALUE-SPACE GCD (W=3,5,10)", flush=True)
print("="*70, flush=True)

print(f"\n  {'W':>4} {'C':>8} {'wt':>6} {'err':>8} {'r':>8} {'p_gue':>8} {'<half':>8}", flush=True)
print(f"  {'-'*54}", flush=True)

for W in [3, 5, 10, 20, N-1]:
    W_label = W if W < N else "full"
    for C in [1, 5, 10, 50, 100, 500, 1000, 5000]:
        for wt in ["log", "indicator"]:
            if W < N:
                H = build_valuespace_gcd(alpha, W, C, wt)
            else:
                H = build_valuespace_gcd_full(alpha, C, wt)

            eigs = np.sort(np.linalg.eigvalsh(H))
            err, ph, r, pg = score_full(eigs)

            if r > 0.05 or (C <= 10 and W == 3):
                print(f"  {str(W_label):>4} {C:>8} {wt:>6} {err:>8.4f} {r:>+8.4f} "
                      f"{pg:>8.4f} {ph*100:>7.1f}%", flush=True)


# ============================================================
# SWEEP 2: Full value-space GCD (all pairs, not just banded)
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 2: FULL VALUE-SPACE GCD (all j,k with gcd > 1)", flush=True)
print("="*70, flush=True)

print(f"\n  {'C':>8} {'err':>8} {'r':>8} {'p_gue':>8} {'<half':>8}", flush=True)
print(f"  {'-'*42}", flush=True)

for C in [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
    H = build_valuespace_gcd_full(alpha, C)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err, ph, r, pg = score_full(eigs)
    mark = " ***" if r > 0.3 and pg > 0.001 else ""
    print(f"  {C:>8.1f} {err:>8.4f} {r:>+8.4f} {pg:>8.4f} {ph*100:>7.1f}%{mark}", flush=True)


# ============================================================
# SWEEP 3: Different weight functions
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 3: WEIGHT FUNCTION COMPARISON (full matrix, C=100)", flush=True)
print("="*70, flush=True)

C_test = 100
for wt in ["log", "raw", "indicator", "vonmangoldt"]:
    H = build_valuespace_gcd(alpha, N-1, C_test, wt)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err, ph, r, pg = score_full(eigs)
    print(f"  {wt:>15}: err={err:.4f}, r={r:+.4f}, p(GUE)={pg:.4f}, <half={ph*100:.1f}%", flush=True)


# ============================================================
# SWEEP 4: Joint optimization (C for best r*p_gue product)
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 4: OPTIMIZE C FOR JOINT r AND p(GUE)", flush=True)
print("="*70, flush=True)

def joint_obj(log_C):
    C = np.exp(log_C)
    H = build_valuespace_gcd_full(alpha, C)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err, ph, r, pg = score_full(eigs)
    # Want high r AND high p_gue AND low error
    cost = -r  # maximize r
    if pg < 0.01: cost += 1.0  # penalize non-GUE
    if err > 2.0: cost += (err - 2.0)  # penalize large error
    return cost

res = minimize_scalar(joint_obj, bounds=(-2, 10), method='bounded')
C_opt = np.exp(res.x)

H_opt = build_valuespace_gcd_full(alpha, C_opt)
eigs_opt = np.sort(np.linalg.eigvalsh(H_opt))
err, ph, r, pg = score_full(eigs_opt)

print(f"\n  Optimal C = {C_opt:.2f}", flush=True)
print(f"  r = {r:+.4f}", flush=True)
print(f"  p(GUE) = {pg:.4f}", flush=True)
print(f"  mean_err = {err:.4f}", flush=True)
print(f"  <half = {ph*100:.1f}%", flush=True)

# Eigenvector stats
_, vecs = np.linalg.eigh(H_opt)
ipr = np.mean(np.sum(vecs**4, axis=0))
print(f"  IPR = {ipr:.6f} ({ipr/(3./N):.1f}x GUE)", flush=True)


# ============================================================
# BEST RESULT DETAILS
# ============================================================
print("\n" + "="*70, flush=True)
print("BEST RESULT DETAILS", flush=True)
print("="*70, flush=True)

print(f"\n  {'k':>4} {'Eigenvalue':>12} {'Zero':>12} {'Error':>10}", flush=True)
for i in range(min(20, len(eigs_opt))):
    e = abs(eigs_opt[i] - zeta_zeros[i])
    tag = " <<<" if e < 0.3 else ""
    print(f"  {i+1:>4} {eigs_opt[i]:>12.4f} {zeta_zeros[i]:>12.4f} {e:>10.4f}{tag}", flush=True)


# ============================================================
# SPARSITY: how many GCD > 1 pairs exist?
# ============================================================
n_pairs = 0
n_total = N*(N-1)//2
for j in range(N):
    for k in range(j+1, N):
        if gcd(round_alpha[j], round_alpha[k]) > 1:
            n_pairs += 1

print(f"\n  GCD > 1 pairs: {n_pairs}/{n_total} ({100*n_pairs/n_total:.1f}%)", flush=True)
print(f"  Sparsity: {100*(1-n_pairs/n_total):.1f}% zeros", flush=True)


print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
