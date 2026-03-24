"""Beta correction: encode local density fluctuations from primes.

THE INSIGHT:
  alpha_k encodes WHERE zeros are (positions)
  beta_k  must encode how CLOSELY SPACED they are (local density)

Both come from the SAME explicit formula:
  S(T) = -(1/pi) sum_p sum_m sin(2mT*log(p)) / (m*p^{m/2})

  alpha_k = weyl(k) + S(weyl_k) / N'(weyl_k)         [zero position]
  beta_k  = C / (N'(weyl_k) + S'(weyl_k))             [inverse local density]

where S'(T) = dS/dT = -(2/pi) sum_p sum_m log(p)*cos(2mT*log(p)) / p^{m/2}

When S' > 0: local density is higher → beta smaller → eigenvalues cluster
When S' < 0: local density is lower → beta larger → eigenvalues spread

The constant C is the ONLY free parameter, determined by matching
the eigenvalue scale.

ALSO TEST:
  - beta_k = predicted_spacing_k * C  (spacing = 1/density)
  - beta_k = C / rho + D * rho'  (include density gradient)
  - Direct formula: beta_k = (alpha_{k+1} - alpha_k) * C  (spacing of alphas)
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize_scalar
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
actual_spacing = np.mean(np.diff(zeta_zeros[trim:-trim]))


def N_smooth(T):
    if T < 2: return 0
    return T/(2*np.pi) * np.log(T/(2*np.pi)) - T/(2*np.pi) + 7/8

def N_deriv(T):
    if T < 2: return 0.001
    return np.log(T/(2*np.pi)) / (2*np.pi)

def weyl_zero(n):
    t = 2*np.pi*n / np.log(max(n,2)+2)
    for _ in range(30):
        if t < 1: t = 10.0
        Nt = N_smooth(t)
        dNt = N_deriv(t)
        if abs(dNt) < 1e-30: break
        t -= (Nt - n) / dNt
    return t

def S_func(T, primes, max_m=3):
    """S(T) = -(1/pi) sum_p sum_m sin(2mT*log(p)) / (m*p^{m/2})"""
    s = 0.0
    for p in primes:
        lp = np.log(p)
        for m in range(1, max_m+1):
            s -= np.sin(2*m*T*lp) / (m * p**(m/2))
    return s / np.pi

def S_deriv(T, primes, max_m=3):
    """S'(T) = -(2/pi) sum_p sum_m log(p)*cos(2mT*log(p)) / p^{m/2}"""
    s = 0.0
    for p in primes:
        lp = np.log(p)
        for m in range(1, max_m+1):
            s -= 2*lp * np.cos(2*m*T*lp) / p**(m/2)
    return s / np.pi


# ============================================================
# Build alpha (same as before — explicit formula)
# ============================================================
print("Building alpha from explicit formula...")

def build_alpha(N_size, primes, max_m=3):
    alpha = np.zeros(N_size)
    for k in range(1, N_size+1):
        T_w = weyl_zero(k)
        dN = N_deriv(T_w)
        S_T = S_func(T_w, primes, max_m)
        alpha[k-1] = T_w + S_T / dN
    return alpha

alpha = build_alpha(N, all_primes[:168], max_m=5)

# How good is alpha alone?
alpha_errs = np.abs(alpha - zeta_zeros)[trim:-trim]
print(f"  Alpha alone: mean err = {np.mean(alpha_errs):.4f}, "
      f"median = {np.median(alpha_errs):.4f}")


# ============================================================
# Beta models
# ============================================================

def beta_smooth_only(N_size):
    """Beta = C / N'(weyl_k). Smooth density, no primes."""
    beta = np.zeros(N_size - 1)
    for k in range(1, N_size):
        T_w = (weyl_zero(k) + weyl_zero(k+1)) / 2
        beta[k-1] = 1.0 / N_deriv(T_w)
    return beta

def beta_density_corrected(N_size, primes, max_m=3):
    """Beta = C / (N'(T) + S'(T)). Local density with prime correction."""
    beta = np.zeros(N_size - 1)
    for k in range(1, N_size):
        T_w = (weyl_zero(k) + weyl_zero(k+1)) / 2
        rho = N_deriv(T_w) + S_deriv(T_w, primes, max_m)
        if rho > 0.01:
            beta[k-1] = 1.0 / rho
        else:
            beta[k-1] = 1.0 / N_deriv(T_w)
    return beta

def beta_from_alpha_spacing(alpha_vals):
    """Beta = C * (alpha_{k+1} - alpha_k). Direct from alpha spacings."""
    return np.diff(alpha_vals)

def beta_pair_corr(N_size, primes, max_m=3):
    """Beta including pair correlation correction.

    The pair correlation for zeta zeros:
    R2(x) = 1 - sin^2(pi*x)/(pi*x)^2 + arithmetic_correction

    The arithmetic correction modulates the local spacing.
    The beta should reflect: spacing_k = (1/rho_k) * (1 + corr_k)
    where corr_k comes from the two-point function.

    Approximate: corr_k ~ S'(T_k) * (local spacing) / (pi * rho)
    """
    beta = np.zeros(N_size - 1)
    for k in range(1, N_size):
        T_w = (weyl_zero(k) + weyl_zero(k+1)) / 2
        rho = N_deriv(T_w)
        Sp = S_deriv(T_w, primes, max_m)

        # Base spacing
        base_spacing = 1.0 / rho

        # Pair correlation correction
        # When S' > 0: zeros cluster → spacing decreases
        # When S' < 0: zeros spread → spacing increases
        pair_corr = 1.0 - Sp / rho  # Relative correction

        beta[k-1] = base_spacing * pair_corr
    return beta


# ============================================================
# Optimization: find the best constant C for each model
# ============================================================

def score_jacobi(alpha_vals, beta_raw, C, actual_zeros):
    """Score: mean error of eigenvalues vs actual zeros."""
    beta_scaled = np.abs(beta_raw * C)
    beta_scaled = np.maximum(beta_scaled, 1e-10)
    try:
        eigs = np.sort(eigh_tridiagonal(alpha_vals, beta_scaled, eigvals_only=True))
        actual = np.sort(actual_zeros[:len(eigs)])
        t = int(0.1 * len(eigs))
        return np.mean(np.abs(eigs - actual)[t:-t])
    except:
        return 1e10


def optimize_C(alpha_vals, beta_raw, actual_zeros, C_range=(0.01, 10)):
    """Find the C that minimizes eigenvalue error."""
    result = minimize_scalar(
        lambda c: score_jacobi(alpha_vals, beta_raw, c, actual_zeros),
        bounds=C_range, method='bounded'
    )
    return result.x, result.fun


# ============================================================
# TEST ALL BETA MODELS
# ============================================================
print("\n" + "=" * 70)
print("TESTING BETA MODELS (with optimized C)")
print("=" * 70)

models = {}

# Model 1: Smooth only
beta_raw_1 = beta_smooth_only(N)
C1, err1 = optimize_C(alpha, beta_raw_1, zeta_zeros)
models["Smooth (no primes)"] = (beta_raw_1 * C1, err1, C1)

# Model 2: Density-corrected (5 primes)
beta_raw_2a = beta_density_corrected(N, all_primes[:5], max_m=3)
C2a, err2a = optimize_C(alpha, beta_raw_2a, zeta_zeros)
models["Density (5 primes)"] = (beta_raw_2a * C2a, err2a, C2a)

# Model 2b: Density-corrected (30 primes)
beta_raw_2b = beta_density_corrected(N, all_primes[:30], max_m=3)
C2b, err2b = optimize_C(alpha, beta_raw_2b, zeta_zeros)
models["Density (30 primes)"] = (beta_raw_2b * C2b, err2b, C2b)

# Model 2c: Density-corrected (168 primes)
beta_raw_2c = beta_density_corrected(N, all_primes[:168], max_m=5)
C2c, err2c = optimize_C(alpha, beta_raw_2c, zeta_zeros)
models["Density (168 primes)"] = (beta_raw_2c * C2c, err2c, C2c)

# Model 3: Alpha spacing
beta_raw_3 = beta_from_alpha_spacing(alpha)
C3, err3 = optimize_C(alpha, beta_raw_3, zeta_zeros)
models["Alpha spacing"] = (beta_raw_3 * C3, err3, C3)

# Model 4: Pair correlation
beta_raw_4 = beta_pair_corr(N, all_primes[:168], max_m=5)
C4, err4 = optimize_C(alpha, beta_raw_4, zeta_zeros)
models["Pair corr (168p)"] = (beta_raw_4 * C4, err4, C4)

# Model 5: Density (500 primes, more terms)
beta_raw_5 = beta_density_corrected(N, all_primes[:303], max_m=5)
C5, err5 = optimize_C(alpha, beta_raw_5, zeta_zeros)
models["Density (303p, m=5)"] = (beta_raw_5 * C5, err5, C5)

print(f"\n  {'Model':<25} {'C':>8} {'mean_err':>10} {'%<half':>8} {'%<full':>8}")
print(f"  {'-'*64}")

for name, (beta_final, err, C) in sorted(models.items(), key=lambda x: x[1][1]):
    beta_v = np.maximum(np.abs(beta_final), 1e-10)
    eigs = np.sort(eigh_tridiagonal(alpha, beta_v, eigvals_only=True))
    actual = np.sort(zeta_zeros[:len(eigs)])
    errs = np.abs(eigs - actual)[trim:-trim]
    ms = np.mean(np.diff(actual[trim:-trim]))
    pct_h = np.mean(errs < ms/2) * 100
    pct_f = np.mean(errs < ms) * 100
    print(f"  {name:<25} {C:>8.4f} {np.mean(errs):>10.4f} {pct_h:>7.1f}% {pct_f:>7.1f}%")


# ============================================================
# DEEP DIVE: Best model analysis
# ============================================================
print("\n" + "=" * 70)
print("DEEP DIVE: BEST MODEL")
print("=" * 70)

# Find the best
best_name = min(models, key=lambda k: models[k][1])
best_beta, best_err, best_C = models[best_name]
print(f"\n  Best model: {best_name} (C={best_C:.4f})")

beta_v = np.maximum(np.abs(best_beta), 1e-10)
eigs_best = np.sort(eigh_tridiagonal(alpha, beta_v, eigvals_only=True))
actual = np.sort(zeta_zeros[:len(eigs_best)])

# Show first 20 and middle
print(f"\n  {'k':>4} {'Predicted':>12} {'Actual':>12} {'Error':>10} {'Rel%':>8}")
print(f"  {'-'*48}")
for i in list(range(20)) + [249, 250, 251]:
    if i >= len(eigs_best): break
    err = abs(eigs_best[i] - actual[i])
    rel = err/actual[i]*100
    tag = " <<<" if err < 0.3 else ""
    print(f"  {i+1:>4} {eigs_best[i]:>12.4f} {actual[i]:>12.4f} "
          f"{err:>10.4f} {rel:>7.3f}%{tag}")


# ============================================================
# SCALING: how does accuracy improve with more primes?
# ============================================================
print("\n" + "=" * 70)
print("SCALING: ACCURACY vs PRIME COUNT (best beta model type)")
print("=" * 70)

# Determine which model type won
# Rebuild with varying prime counts
print(f"\n  {'Primes':>8} {'p_max':>8} {'C':>8} {'mean_err':>10} {'%<half':>8} {'%<full':>8}")
print(f"  {'-'*54}")

for n_p in [0, 3, 5, 10, 20, 50, 100, 168, 303]:
    p_list = all_primes[:n_p] if n_p > 0 else []
    max_m = min(5, max(1, n_p // 10))

    # Build alpha and beta with this prime set
    a = build_alpha(N, p_list, max_m) if n_p > 0 else \
        np.array([weyl_zero(k) for k in range(1, N+1)])

    # Use the best-performing beta model type
    if "Density" in best_name or "Pair" in best_name:
        b_raw = beta_density_corrected(N, p_list, max_m) if n_p > 0 else beta_smooth_only(N)
    elif "Alpha" in best_name:
        b_raw = beta_from_alpha_spacing(a)
    else:
        b_raw = beta_smooth_only(N)

    C_opt, _ = optimize_C(a, b_raw, zeta_zeros)
    b_final = np.maximum(np.abs(b_raw * C_opt), 1e-10)

    try:
        eigs = np.sort(eigh_tridiagonal(a, b_final, eigvals_only=True))
        errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
        ms = np.mean(np.diff(zeta_zeros[trim:-trim]))
        p_max = all_primes[n_p-1] if n_p > 0 else 0
        print(f"  {n_p:>8} {p_max:>8} {C_opt:>8.4f} {np.mean(errs):>10.4f} "
              f"{np.mean(errs<ms/2)*100:>7.1f}% {np.mean(errs<ms)*100:>7.1f}%")
    except:
        print(f"  {n_p:>8} FAILED")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

# Compare alpha-only to alpha+beta_corrected
eigs_alpha_only = np.sort(alpha)
errs_alpha = np.abs(eigs_alpha_only - zeta_zeros)[trim:-trim]

eigs_full = np.sort(eigh_tridiagonal(alpha, np.maximum(np.abs(best_beta), 1e-10),
                                      eigvals_only=True))
errs_full = np.abs(eigs_full - zeta_zeros[:len(eigs_full)])[trim:-trim]

ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

print(f"""
  COMPARISON:
    Alpha only (diagonal):     mean_err = {np.mean(errs_alpha):.4f}, <half_gap = {np.mean(errs_alpha<ms/2)*100:.1f}%
    Alpha + beta (tridiag):    mean_err = {np.mean(errs_full):.4f}, <half_gap = {np.mean(errs_full<ms/2)*100:.1f}%
    Improvement from beta:     {(1 - np.mean(errs_full)/np.mean(errs_alpha))*100:.1f}%

  The operator:
    alpha_k = weyl(k) + S(weyl_k)/N'(weyl_k)
    beta_k  = C / (N'(weyl_k) + S'(weyl_k))     [C = {best_C:.4f}]

  where S and S' are the explicit formula prime sums.
  Both alpha and beta derive from the SAME arithmetic function S(T).
  The single free parameter C sets the overall coupling scale.
""")

print(f"Total time: {time.time() - t0:.1f}s")
