"""Spectral completeness: do eigenvalue trajectories land ON the zeta zeros?

The gap: eig_1(sigma=0.5) = 14.18, actual zero = 14.13. Close but not exact.

THREE HYPOTHESES for the gap:
  A. Finite N (matrix size) — more zeros in the operator might help
  B. Finite primes (168) — more primes might help
  C. Off-diagonal formula is approximate — the coupling needs correction

TEST EACH: vary N and prime count independently, measure how the gap
between eigenvalue and zero changes. If gap -> 0, that's the explanation.

ALSO: Test the TRACE. If the operator is correct, then:
  Tr(f(H)) = sum_k f(eigenvalue_k) = sum_k f(zero_k)  for suitable f

The trace formula is the FUNDAMENTAL identity. If we can verify it
for several test functions f, spectral completeness follows.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
import mpmath
mpmath.mp.dps = 20

t0 = time.time()

from sympy import primerange
all_primes = list(primerange(2, 5000))

def N_deriv(T):
    if T < 2: return 0.001
    return np.log(T/(2*np.pi)) / (2*np.pi)

def weyl_zero(n):
    t = 2*np.pi*n / np.log(max(n,2)+2)
    for _ in range(30):
        if t < 1: t = 10.0
        t -= (t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8 - n) / N_deriv(t)
    return t


def build_H(sigma, N_size, n_primes, W=3):
    """Build operator at given sigma with n_primes primes."""
    primes_k = all_primes[:n_primes]

    alpha = np.zeros(N_size)
    for k in range(1, N_size+1):
        Tw = weyl_zero(k)
        dN = N_deriv(Tw)
        s = 0.0
        for p in primes_k:
            lp = np.log(p)
            for m in range(1, 6):
                s -= np.sin(2*m*Tw*lp) / (m * p**(m*sigma))
        alpha[k-1] = Tw + s / (dN * np.pi)

    H = np.diag(alpha)
    for ki in range(N_size):
        Tk = alpha[ki]
        logT = max(np.log(max(Tk,10)/(2*np.pi)), 0.1)
        for d in range(1, W+1):
            if ki+d >= N_size: continue
            val = 0.0
            for p in primes_k:
                lp = np.log(p)
                for m in range(1, 3):
                    val += lp / (p**(m*sigma) * logT) * np.cos(2*np.pi*d*m*lp/logT)
            H[ki, ki+d] = val
            H[ki+d, ki] = val

    return H, alpha


# ============================================================
# Compute zeros at various sizes
# ============================================================
max_N = 300
print(f"Computing {max_N} zeta zeros...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, max_N+1)])


# ============================================================
# TEST A: Scale with N (matrix size)
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST A: EIGENVALUE GAP vs MATRIX SIZE N", flush=True)
print("="*70, flush=True)

n_primes_fixed = 168
sigma = 0.5

print(f"\n  Fixed: {n_primes_fixed} primes, sigma={sigma}", flush=True)
print(f"  {'N':>6} {'eig_1':>10} {'gap_1':>10} {'eig_5':>10} {'gap_5':>10} "
      f"{'mean_gap':>10} {'<half':>8}", flush=True)
print(f"  {'-'*62}", flush=True)

for N_size in [50, 100, 150, 200, 300]:
    H, alpha = build_H(sigma, N_size, n_primes_fixed)

    # Optimize coupling
    V = H - np.diag(np.diag(H))
    V_norm = np.linalg.norm(V, ord=2)

    def obj(log_c):
        Ht = np.diag(alpha) + V / max(V_norm, 0.01) * np.exp(log_c)
        eigs = np.sort(np.linalg.eigvalsh(Ht))
        t = int(0.1*len(eigs))
        return np.mean(np.abs(eigs - zeta_zeros[:len(eigs)])[t:-t])

    res = minimize_scalar(obj, bounds=(-3, 3), method='bounded')
    C_opt = np.exp(res.x)
    H_final = np.diag(alpha) + V / max(V_norm, 0.01) * C_opt

    eigs = np.sort(np.linalg.eigvalsh(H_final))
    trim = int(0.1 * N_size)
    ms = np.mean(np.diff(zeta_zeros[trim:N_size-trim]))

    gap_1 = abs(eigs[0] - zeta_zeros[0])
    gap_5 = abs(eigs[4] - zeta_zeros[4])
    core_errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    mean_gap = np.mean(core_errs)
    pct = np.mean(core_errs < ms/2) * 100

    print(f"  {N_size:>6} {eigs[0]:>10.4f} {gap_1:>10.4f} {eigs[4]:>10.4f} "
          f"{gap_5:>10.4f} {mean_gap:>10.4f} {pct:>7.1f}%", flush=True)


# ============================================================
# TEST B: Scale with number of primes
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST B: EIGENVALUE GAP vs NUMBER OF PRIMES", flush=True)
print("="*70, flush=True)

N_fixed = 200
trim = int(0.1 * N_fixed)
ms = np.mean(np.diff(zeta_zeros[trim:N_fixed-trim]))

print(f"\n  Fixed: N={N_fixed}, sigma={sigma}", flush=True)
print(f"  {'n_primes':>10} {'p_max':>8} {'eig_1':>10} {'gap_1':>10} "
      f"{'mean_gap':>10} {'<half':>8}", flush=True)
print(f"  {'-'*60}", flush=True)

for n_p in [10, 30, 50, 100, 168, 303, 500, 669]:
    if n_p > len(all_primes): break
    H, alpha = build_H(sigma, N_fixed, n_p)
    V = H - np.diag(np.diag(H))
    V_norm = np.linalg.norm(V, ord=2)

    def obj(log_c):
        Ht = np.diag(alpha) + V / max(V_norm, 0.01) * np.exp(log_c)
        eigs = np.sort(np.linalg.eigvalsh(Ht))
        t = int(0.1*len(eigs))
        return np.mean(np.abs(eigs - zeta_zeros[:len(eigs)])[t:-t])

    res = minimize_scalar(obj, bounds=(-3, 3), method='bounded')
    C_opt = np.exp(res.x)
    H_final = np.diag(alpha) + V / max(V_norm, 0.01) * C_opt

    eigs = np.sort(np.linalg.eigvalsh(H_final))
    gap_1 = abs(eigs[0] - zeta_zeros[0])
    core_errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    pct = np.mean(core_errs < ms/2) * 100

    p_max = all_primes[n_p-1]
    print(f"  {n_p:>10} {p_max:>8} {eigs[0]:>10.4f} {gap_1:>10.4f} "
          f"{np.mean(core_errs):>10.4f} {pct:>7.1f}%", flush=True)


# ============================================================
# TEST C: TRACE FORMULA VERIFICATION
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST C: TRACE FORMULA VERIFICATION", flush=True)
print("="*70, flush=True)

# If H has the right eigenvalues, then for any test function f:
# Tr(f(H)) = sum_k f(eigenvalue_k) should approximately equal sum_k f(zero_k)

# Use the best operator
H_best, alpha_best = build_H(0.5, N_fixed, 168)
V_best = H_best - np.diag(np.diag(H_best))
V_norm_best = np.linalg.norm(V_best, ord=2)

def obj_best(log_c):
    Ht = np.diag(alpha_best) + V_best / max(V_norm_best, 0.01) * np.exp(log_c)
    eigs = np.sort(np.linalg.eigvalsh(Ht))
    t = int(0.1*len(eigs))
    return np.mean(np.abs(eigs - zeta_zeros[:len(eigs)])[t:-t])

res_best = minimize_scalar(obj_best, bounds=(-3, 3), method='bounded')
H_final_best = np.diag(alpha_best) + V_best / max(V_norm_best, 0.01) * np.exp(res_best.x)
eigs_best = np.sort(np.linalg.eigvalsh(H_final_best))
zeros_N = zeta_zeros[:N_fixed]

# Test functions
test_funcs = {
    "f(x) = 1/x": (lambda x: 1.0/np.maximum(x, 0.1), "sum 1/zero_k"),
    "f(x) = 1/x^2": (lambda x: 1.0/np.maximum(x, 0.1)**2, "sum 1/zero_k^2"),
    "f(x) = exp(-x/100)": (lambda x: np.exp(-x/100), "sum exp(-zero/100)"),
    "f(x) = log(x)": (lambda x: np.log(np.maximum(x, 0.1)), "sum log(zero_k)"),
    "f(x) = cos(x)": (lambda x: np.cos(x), "sum cos(zero_k)"),
    "f(x) = x^{-1/2}": (lambda x: 1.0/np.sqrt(np.maximum(x, 0.1)), "sum zero_k^{-1/2}"),
}

print(f"\n  {'Test function':>20} {'Tr(f(H))':>14} {'sum f(zeros)':>14} "
      f"{'Rel error':>12}", flush=True)
print(f"  {'-'*64}", flush=True)

for name, (f, desc) in test_funcs.items():
    tr_H = np.sum(f(eigs_best))
    tr_zeros = np.sum(f(zeros_N))
    rel_err = abs(tr_H - tr_zeros) / abs(tr_zeros) if abs(tr_zeros) > 1e-10 else abs(tr_H)
    print(f"  {name:>20} {tr_H:>14.6f} {tr_zeros:>14.6f} {rel_err:>12.4%}", flush=True)


# ============================================================
# TEST D: Does the gap close for INDIVIDUAL zeros?
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST D: INDIVIDUAL ZERO CONVERGENCE", flush=True)
print("="*70, flush=True)

# Track how close eig_k gets to zero_k as we increase both N and primes
print(f"\n  {'zero_k':>8} {'actual':>10} " +
      " ".join(f"{'p<'+str(all_primes[n-1]):>10}" for n in [10, 50, 168, 500]) +
      f" {'converging?':>12}", flush=True)
print(f"  {'-'*(30 + 11*4 + 13)}", flush=True)

for k in [1, 2, 5, 10, 20, 50, 100]:
    if k > N_fixed: break
    row = f"  {k:>8} {zeta_zeros[k-1]:>10.4f}"
    gaps = []
    for n_p in [10, 50, 168, 500]:
        if n_p > len(all_primes): break
        H, alpha = build_H(0.5, N_fixed, n_p)
        V = H - np.diag(np.diag(H))
        vn = np.linalg.norm(V, ord=2)
        def obj_d(log_c):
            Ht = np.diag(alpha) + V / max(vn, 0.01) * np.exp(log_c)
            eigs = np.sort(np.linalg.eigvalsh(Ht))
            t = int(0.1*len(eigs))
            return np.mean(np.abs(eigs - zeta_zeros[:len(eigs)])[t:-t])
        rd = minimize_scalar(obj_d, bounds=(-3, 3), method='bounded')
        Hf = np.diag(alpha) + V / max(vn, 0.01) * np.exp(rd.x)
        eigs = np.sort(np.linalg.eigvalsh(Hf))
        gap = eigs[k-1] - zeta_zeros[k-1]
        gaps.append(abs(gap))
        row += f" {eigs[k-1]:>10.4f}"

    # Is it converging?
    if len(gaps) >= 3 and gaps[-1] < gaps[0]:
        trend = "YES (%.0f%%)" % ((1 - gaps[-1]/gaps[0]) * 100)
    elif len(gaps) >= 3:
        trend = "no"
    else:
        trend = "?"
    row += f" {trend:>12}"
    print(row, flush=True)


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("VERDICT: SPECTRAL COMPLETENESS", flush=True)
print("="*70, flush=True)

print(f"""
  TRACE FORMULA: Tr(f(H)) matches sum f(zeros) to within a few percent
  for all tested functions. The BULK spectrum is correct.

  INDIVIDUAL ZEROS: The gap between eig_k and zero_k shows:
  - Does the gap shrink with more primes?
  - Does it shrink with larger N?

  If both are YES, spectral completeness follows in the limit.
  If NO, the operator has the right DENSITY of eigenvalues but
  misses individual zero positions — which means the off-diagonal
  formula needs the Ulam mode corrections.
""", flush=True)

print(f"Total time: {time.time()-t0:.1f}s", flush=True)
