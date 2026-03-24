"""Banded operator: pair correlation kernel from the explicit formula.

WHY TRIDIAGONAL FAILS:
  beta_k couples zero k to k±1 only. But the pair correlation R2(x)
  extends to all separations. The Montgomery explicit formula gives:

  R2(x) = 1 - sin^2(pi*x)/(pi*x)^2 + sum_p 2*log^2(p)/(p*log^2(T)) * cos(2*pi*x*log(p)/log(T))

  The prime terms oscillate at ALL separations, not just nearest-neighbor.

THE BANDED APPROACH:
  Build a symmetric matrix H with:
    H_{kk} = alpha_k (diagonal, from explicit formula)
    H_{k,k+d} = V_d(T_k) for |d| <= W (off-diagonal, from pair correlation)

  where V_d(T) = sum_p A_p * cos(2*pi*d*log(p)/log(T)) / p^{something}

  This is the explicit formula applied at FINITE SEPARATION d,
  creating a banded matrix of bandwidth W.

  W=0: diagonal only (alpha predicts zeros)
  W=1: tridiagonal (previous result)
  W=5-20: captures short-range pair correlation
  W=N: full matrix (overfitting risk)

ALSO TEST:
  - GUE sine kernel: K(d) = sin(pi*d)/(pi*d) [the universal part]
  - Explicit formula kernel: K(d) = sine + prime corrections [the full thing]
  - Empirical kernel from actual zero pair correlation
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.stats import pearsonr
import mpmath

t0 = time.time()
mpmath.mp.dps = 20

# ============================================================
# Setup
# ============================================================
print("Computing 500 zeta zeros...")
t_start = time.time()
N = 500
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N + 1)])
print(f"  Done ({time.time()-t_start:.1f}s)")

from sympy import primerange
all_primes = list(primerange(2, 2000))

trim = int(0.1 * N)
actual = np.sort(zeta_zeros)
mean_spacing = np.mean(np.diff(actual[trim:-trim]))


def N_deriv(T):
    if T < 2: return 0.001
    return np.log(T / (2*np.pi)) / (2*np.pi)

def weyl_zero(n):
    t = 2*np.pi*n / np.log(max(n,2)+2)
    for _ in range(30):
        if t < 1: t = 10.0
        Nt = t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
        dNt = N_deriv(t)
        if abs(dNt) < 1e-30: break
        t -= (Nt - n) / dNt
    return t

def S_func(T, primes, max_m=3):
    s = 0.0
    for p in primes:
        lp = np.log(p)
        for m in range(1, max_m+1):
            s -= np.sin(2*m*T*lp) / (m * p**(m/2))
    return s / np.pi

def build_alpha(N_size, primes, max_m=3):
    alpha = np.zeros(N_size)
    for k in range(1, N_size+1):
        T_w = weyl_zero(k)
        dN = N_deriv(T_w)
        S_T = S_func(T_w, primes, max_m)
        alpha[k-1] = T_w + S_T / dN
    return alpha


# Build alpha
print("Building alpha from explicit formula (168 primes)...")
alpha = build_alpha(N, all_primes[:168], max_m=5)


# ============================================================
# Pair correlation kernels
# ============================================================

def kernel_smooth(d, T):
    """Smooth off-diagonal: just 1/N'(T) decay with distance."""
    rho = N_deriv(T)
    return np.exp(-abs(d) * rho) / rho

def kernel_sine(d):
    """GUE sine kernel: K(d) = sin(pi*d)/(pi*d)."""
    if abs(d) < 1e-10:
        return 1.0
    return np.sin(np.pi * d) / (np.pi * d)

def kernel_explicit(d, T, primes, max_m=2):
    """Explicit formula kernel at separation d.

    From the pair correlation function:
    K(d) = -sin(pi*d)/(pi*d) + sum_p A_p cos(2*pi*d*log(p)/log(T))

    The sine kernel gives GUE repulsion.
    The prime sum gives the arithmetic correction.
    """
    # GUE part (level repulsion)
    if abs(d) < 1e-10:
        gue = 1.0
    else:
        gue = -np.sin(np.pi * d) / (np.pi * d)

    # Prime part
    log_T = np.log(T / (2*np.pi))
    prime_corr = 0.0
    for p in primes:
        lp = np.log(p)
        for m in range(1, max_m + 1):
            amp = lp / (p**(m/2) * log_T)
            prime_corr += amp * np.cos(2 * np.pi * d * m * lp / log_T)

    return gue + prime_corr

def kernel_empirical(d, zero_spacings, max_d=30):
    """Empirical pair correlation from actual zero spacings."""
    if abs(d) > max_d or abs(d) < 1:
        return 0.0
    # Count pairs at separation d (in units of mean spacing)
    n_pairs = 0
    total = 0
    cumsum = np.cumsum(zero_spacings / np.mean(zero_spacings))
    for i in range(len(cumsum)):
        for j in range(i+1, min(i+max_d+1, len(cumsum))):
            sep = cumsum[j] - cumsum[i]
            if abs(sep - abs(d)) < 0.5:
                n_pairs += 1
            total += 1
    return n_pairs / max(total, 1) if total > 0 else 0


# ============================================================
# Build banded matrices
# ============================================================

def build_banded(alpha_vals, kernel_func, bandwidth, scale=1.0):
    """Build symmetric banded matrix with given kernel.

    H_{k,k+d} = scale * kernel_func(d, T_k) for |d| <= bandwidth
    H_{kk} = alpha_k
    """
    n = len(alpha_vals)
    H = np.diag(alpha_vals)

    for k in range(n):
        T_k = alpha_vals[k]  # Use alpha as proxy for T
        for d in range(1, bandwidth + 1):
            if k + d < n:
                val = scale * kernel_func(d, T_k)
                H[k, k+d] = val
                H[k+d, k] = val

    return H


def score_banded(alpha_vals, kernel_func, bandwidth, scale, actual_zeros):
    """Score a banded matrix by eigenvalue error."""
    H = build_banded(alpha_vals, kernel_func, bandwidth, scale)
    eigs = np.sort(np.linalg.eigvalsh(H))
    t = int(0.1 * len(eigs))
    return np.mean(np.abs(eigs - actual_zeros[:len(eigs)])[t:-t])


# ============================================================
# TEST 1: Bandwidth sweep with sine kernel
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: BANDWIDTH SWEEP — SINE KERNEL (GUE)")
print("=" * 70)

print(f"\n  {'W':>4} {'C_opt':>8} {'mean_err':>10} {'%<half':>8} {'%<full':>8} {'improvement':>12}")
print(f"  {'-'*56}")

baseline_err = np.mean(np.abs(alpha - zeta_zeros)[trim:-trim])

for W in [0, 1, 2, 3, 5, 10, 15, 20, 30, 50]:
    if W == 0:
        # Diagonal only
        errs = np.abs(alpha - zeta_zeros)[trim:-trim]
        print(f"  {W:>4} {'N/A':>8} {np.mean(errs):>10.4f} "
              f"{np.mean(errs<mean_spacing/2)*100:>7.1f}% "
              f"{np.mean(errs<mean_spacing)*100:>7.1f}% {'baseline':>12}")
        continue

    # Optimize scale
    from scipy.optimize import minimize_scalar

    def obj(log_c):
        c = np.exp(log_c)
        return score_banded(alpha, lambda d, T: kernel_sine(d), W, c, zeta_zeros)

    res = minimize_scalar(obj, bounds=(-5, 5), method='bounded')
    C_opt = np.exp(res.x)

    H = build_banded(alpha, lambda d, T: kernel_sine(d), W, C_opt)
    eigs = np.sort(np.linalg.eigvalsh(H))
    errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]

    improv = (1 - np.mean(errs)/baseline_err) * 100
    print(f"  {W:>4} {C_opt:>8.4f} {np.mean(errs):>10.4f} "
          f"{np.mean(errs<mean_spacing/2)*100:>7.1f}% "
          f"{np.mean(errs<mean_spacing)*100:>7.1f}% {improv:>+11.1f}%")


# ============================================================
# TEST 2: Bandwidth sweep with explicit formula kernel
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: BANDWIDTH SWEEP — EXPLICIT FORMULA KERNEL (PRIMES)")
print("=" * 70)

primes_kernel = all_primes[:50]

print(f"\n  {'W':>4} {'C_opt':>8} {'mean_err':>10} {'%<half':>8} {'%<full':>8} {'improvement':>12}")
print(f"  {'-'*56}")

for W in [0, 1, 2, 3, 5, 10, 15, 20]:
    if W == 0:
        errs = np.abs(alpha - zeta_zeros)[trim:-trim]
        print(f"  {W:>4} {'N/A':>8} {np.mean(errs):>10.4f} "
              f"{np.mean(errs<mean_spacing/2)*100:>7.1f}% "
              f"{np.mean(errs<mean_spacing)*100:>7.1f}% {'baseline':>12}")
        continue

    def obj_ef(log_c):
        c = np.exp(log_c)
        return score_banded(
            alpha,
            lambda d, T: kernel_explicit(d, T, primes_kernel, max_m=2),
            W, c, zeta_zeros)

    res = minimize_scalar(obj_ef, bounds=(-5, 5), method='bounded')
    C_opt = np.exp(res.x)

    H = build_banded(
        alpha,
        lambda d, T: kernel_explicit(d, T, primes_kernel, max_m=2),
        W, C_opt)
    eigs = np.sort(np.linalg.eigvalsh(H))
    errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]

    improv = (1 - np.mean(errs)/baseline_err) * 100
    print(f"  {W:>4} {C_opt:>8.4f} {np.mean(errs):>10.4f} "
          f"{np.mean(errs<mean_spacing/2)*100:>7.1f}% "
          f"{np.mean(errs<mean_spacing)*100:>7.1f}% {improv:>+11.1f}%")


# ============================================================
# TEST 3: Pure sine vs sine + primes at optimal bandwidth
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: SINE vs SINE+PRIMES AT FIXED BANDWIDTH")
print("=" * 70)

for W in [5, 10, 20]:
    # Sine only
    def obj_s(log_c):
        return score_banded(alpha, lambda d, T: kernel_sine(d), W, np.exp(log_c), zeta_zeros)
    res_s = minimize_scalar(obj_s, bounds=(-5, 5), method='bounded')
    C_s = np.exp(res_s.x)
    err_s = res_s.fun

    # Explicit formula
    def obj_e(log_c):
        return score_banded(alpha, lambda d, T: kernel_explicit(d, T, primes_kernel, 2),
                           W, np.exp(log_c), zeta_zeros)
    res_e = minimize_scalar(obj_e, bounds=(-5, 5), method='bounded')
    C_e = np.exp(res_e.x)
    err_e = res_e.fun

    print(f"  W={W:>2}: Sine={err_s:.4f}, Explicit={err_e:.4f}, "
          f"{'EXPLICIT WINS' if err_e < err_s else 'SINE WINS'} "
          f"({abs(err_s-err_e)/err_s*100:.1f}% diff)")


# ============================================================
# TEST 4: Detailed results at best bandwidth
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: DETAILED RESULTS AT OPTIMAL SETTINGS")
print("=" * 70)

# Find best overall
best_W = 0
best_err = baseline_err
best_type = "diagonal"

for W in [1, 2, 3, 5, 10, 15, 20]:
    for ktype, kfunc in [
        ("sine", lambda d, T: kernel_sine(d)),
        ("explicit", lambda d, T: kernel_explicit(d, T, primes_kernel, 2))
    ]:
        def obj(log_c):
            return score_banded(alpha, kfunc, W, np.exp(log_c), zeta_zeros)
        res = minimize_scalar(obj, bounds=(-5, 5), method='bounded')
        if res.fun < best_err:
            best_err = res.fun
            best_W = W
            best_type = ktype
            best_C = np.exp(res.x)

print(f"\n  Best: W={best_W}, kernel={best_type}, C={best_C:.4f}")
print(f"  Mean error: {best_err:.4f} (baseline: {baseline_err:.4f})")
print(f"  Improvement: {(1-best_err/baseline_err)*100:.1f}%")

# Build and show detailed comparison
if best_type == "sine":
    kf = lambda d, T: kernel_sine(d)
else:
    kf = lambda d, T: kernel_explicit(d, T, primes_kernel, 2)

H_best = build_banded(alpha, kf, best_W, best_C)
eigs_best = np.sort(np.linalg.eigvalsh(H_best))

print(f"\n  {'k':>4} {'Predicted':>12} {'Actual':>12} {'Error':>10} {'Rel%':>8}")
print(f"  {'-'*48}")

good_count = 0
for i in range(min(25, len(eigs_best))):
    err = abs(eigs_best[i] - zeta_zeros[i])
    rel = err / zeta_zeros[i] * 100
    tag = " <<<" if err < 0.3 else ""
    if err < 0.3:
        good_count += 1
    print(f"  {i+1:>4} {eigs_best[i]:>12.4f} {zeta_zeros[i]:>12.4f} "
          f"{err:>10.4f} {rel:>7.3f}%{tag}")

errs_core = np.abs(eigs_best - zeta_zeros[:len(eigs_best)])[trim:-trim]
print(f"\n  Core statistics:")
print(f"    Mean error:       {np.mean(errs_core):.4f}")
print(f"    Median error:     {np.median(errs_core):.4f}")
print(f"    % < half gap:     {np.mean(errs_core < mean_spacing/2)*100:.1f}%")
print(f"    % < full gap:     {np.mean(errs_core < mean_spacing)*100:.1f}%")
print(f"    % < 0.1 spacing:  {np.mean(errs_core < mean_spacing*0.1)*100:.1f}%")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

print(f"""
  BANDED OPERATOR RESULTS:
    Diagonal only (W=0):     {baseline_err:.4f} mean error, {np.mean(np.abs(alpha-zeta_zeros)[trim:-trim]<mean_spacing/2)*100:.1f}% < half gap
    Best banded (W={best_W}):     {best_err:.4f} mean error, {np.mean(errs_core<mean_spacing/2)*100:.1f}% < half gap
    Improvement:              {(1-best_err/baseline_err)*100:.1f}%
    Kernel type:              {best_type}

  The off-diagonal kernel encodes:
    - Level repulsion (sin(pi*d)/(pi*d)) — the GUE part
    - Prime oscillations (sum_p cos(d*log(p)/log(T))) — the arithmetic part
""")

print(f"Total time: {time.time() - t0:.1f}s")
