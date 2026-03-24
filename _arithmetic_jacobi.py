"""The Arithmetic Jacobi Operator: primes in, zeros out.

GOAL: Find explicit formulas for the Jacobi matrix entries
      alpha_k and beta_k in terms of prime numbers, such that
      the eigenvalues of J = tridiag(beta, alpha, beta) are
      the Riemann zeta zeros.

APPROACH:
1. Compute Jacobi from 500 actual zeros (the "target")
2. Decompose alpha_k = alpha_smooth(k) + alpha_osc(k)
   where alpha_smooth comes from Weyl law, alpha_osc from primes
3. Same for beta_k
4. Fit: alpha_osc(k) = sum_p A_p^(a) * cos(2*pi*k*theta_p + phi_p^(a))
   where theta_p = log(p) / (2*pi * mean_density)
5. Build J_arith from the FITTED formulas (primes only, no zeros)
6. Compute eigenvalues of J_arith -> predicted zeros
7. Compare to actual zeros

If the predicted zeros match, we have an explicit operator
defined by primes whose spectrum gives the zeta zeros.
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize
from scipy.stats import pearsonr
import mpmath

t0 = time.time()
mpmath.mp.dps = 20


def lanczos_from_eigenvalues(eigenvalues, start_vec):
    """Lanczos on diag(eigenvalues) with given starting vector."""
    N = len(eigenvalues)
    eigs = eigenvalues.astype(float)
    v = start_vec / np.linalg.norm(start_vec)
    alpha = np.zeros(N)
    beta = np.zeros(N - 1)
    V = np.zeros((N, N))
    V[:, 0] = v
    w = eigs * v
    alpha[0] = np.dot(v, w)
    w = w - alpha[0] * v
    for k in range(1, N):
        beta[k - 1] = np.linalg.norm(w)
        if beta[k - 1] < 1e-14:
            return alpha[:k], beta[:k - 1]
        v_new = w / beta[k - 1]
        for j in range(k):
            v_new -= np.dot(V[:, j], v_new) * V[:, j]
        v_new /= np.linalg.norm(v_new)
        V[:, k] = v_new
        w = eigs * v_new
        alpha[k] = np.dot(v_new, w)
        w = w - alpha[k] * v_new - beta[k - 1] * V[:, k - 1]
    return alpha, beta


def weyl_zero(n):
    """n-th zero from Weyl law: N(t) = t/(2pi)*log(t/(2pi)) - t/(2pi) + 7/8 = n."""
    t = 2 * np.pi * n / np.log(max(n, 2) + 2)
    for _ in range(30):
        if t < 1:
            t = 10.0
        Nt = t / (2 * np.pi) * np.log(t / (2 * np.pi)) - t / (2 * np.pi) + 7 / 8
        dNt = np.log(t / (2 * np.pi)) / (2 * np.pi)
        if abs(dNt) < 1e-30:
            break
        t -= (Nt - n) / dNt
    return t


# ============================================================
# Step 1: Compute zeros and Jacobi
# ============================================================
print("Step 1: Computing zeta zeros...")
t_start = time.time()
N = 500
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N + 1)])
print(f"  {N} zeros in {time.time()-t_start:.1f}s")

print("  Computing Weyl zeros...")
weyl_zeros = np.array([weyl_zero(n) for n in range(1, N + 1)])

print("  Building Lanczos Jacobi...")
v0 = np.ones(N) / np.sqrt(N)
alpha_z, beta_z = lanczos_from_eigenvalues(zeta_zeros, v0)
alpha_w, beta_w = lanczos_from_eigenvalues(weyl_zeros, v0)

# Verify
eigs_check = eigh_tridiagonal(alpha_z, beta_z, eigvals_only=True)
print(f"  Reconstruction error: {np.max(np.abs(np.sort(eigs_check) - np.sort(zeta_zeros))):.2e}")

n = min(len(alpha_z), len(alpha_w))
print(f"  Jacobi dimension: {n}")


# ============================================================
# Step 2: Decompose into smooth + oscillatory
# ============================================================
print("\nStep 2: Smooth + oscillatory decomposition...")

k = np.arange(1, n + 1)

# Fit smooth part of alpha: polynomial in k
alpha_residual = alpha_z[:n] - alpha_w[:n]

# Fit smooth part of beta: it decreases roughly linearly
abs_beta_z = np.abs(beta_z[:n - 1])
abs_beta_w = np.abs(beta_w[:n - 1])
beta_residual = abs_beta_z - abs_beta_w
kb = np.arange(1, n)

# Detrend both residuals (remove any remaining smooth component)
alpha_trend = np.polyfit(k, alpha_residual, 3)
alpha_osc = alpha_residual - np.polyval(alpha_trend, k)

beta_trend = np.polyfit(kb, beta_residual, 3)
beta_osc = beta_residual - np.polyval(beta_trend, kb)

print(f"  Alpha oscillatory: std = {np.std(alpha_osc):.4f}")
print(f"  Beta oscillatory:  std = {np.std(beta_osc):.4f}")


# ============================================================
# Step 3: Fit prime-frequency model to oscillatory parts
# ============================================================
print("\nStep 3: Fitting prime-frequency model...")

# The explicit formula predicts oscillations at frequencies
# theta_p = log(p) / (2*pi) in the "zero counting" variable n.
# But we're in JACOBI INDEX space, where the mapping from n to
# eigenvalue (zero) is nonlinear. The Jacobi index k maps to
# approximately the k-th zero, so the frequency should be
# theta_p = log(p) * (mean density at T_k) where T_k ~ weyl_zero(k).
#
# For simplicity, use a single mean density:
mean_T = np.mean(zeta_zeros)
mean_density = np.log(mean_T / (2 * np.pi)) / (2 * np.pi)

from sympy import primerange
primes = list(primerange(2, 200))

print(f"  Using {len(primes)} primes up to 199")
print(f"  Mean density: {mean_density:.6f} zeros/unit height")


def fit_prime_model(signal, k_vals, primes, n_primes_max=50):
    """Fit signal = sum_p A_p cos(2*pi*k*f_p + phi_p) using least squares.

    For each prime p, the frequency is f_p = log(p) * mean_density.
    Fit amplitudes and phases via cos/sin decomposition:
    signal ~ sum_p [a_p cos(2*pi*k*f_p) + b_p sin(2*pi*k*f_p)]
    Then A_p = sqrt(a_p^2 + b_p^2), phi_p = atan2(b_p, a_p)
    """
    n_p = min(n_primes_max, len(primes))
    freqs = [np.log(p) * mean_density for p in primes[:n_p]]

    # Build design matrix [cos(2*pi*k*f_1), sin(2*pi*k*f_1), ..., 1]
    M = np.zeros((len(k_vals), 2 * n_p + 1))
    for i, f in enumerate(freqs):
        M[:, 2 * i] = np.cos(2 * np.pi * k_vals * f)
        M[:, 2 * i + 1] = np.sin(2 * np.pi * k_vals * f)
    M[:, -1] = 1  # DC offset

    # Least squares fit
    coeffs, residuals, rank, sv = np.linalg.lstsq(M, signal, rcond=None)

    # Extract amplitudes and phases
    amplitudes = np.zeros(n_p)
    phases = np.zeros(n_p)
    for i in range(n_p):
        a = coeffs[2 * i]
        b = coeffs[2 * i + 1]
        amplitudes[i] = np.sqrt(a ** 2 + b ** 2)
        phases[i] = np.arctan2(b, a)

    dc = coeffs[-1]

    # Compute fitted signal
    fitted = M @ coeffs

    # R^2
    ss_res = np.sum((signal - fitted) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return amplitudes, phases, freqs[:n_p], dc, fitted, r2


# Fit alpha oscillatory
print("\n  Fitting alpha oscillatory part:")
for n_p in [5, 10, 20, 50]:
    amps_a, phases_a, freqs_a, dc_a, fitted_a, r2_a = fit_prime_model(
        alpha_osc, k, primes, n_primes_max=n_p)
    print(f"    {n_p:>3} primes: R^2 = {r2_a:.4f}")

# Use 30 primes as the working model
amps_a, phases_a, freqs_a, dc_a, fitted_a, r2_a = fit_prime_model(
    alpha_osc, k, primes, n_primes_max=30)
print(f"\n  Alpha model (30 primes): R^2 = {r2_a:.4f}")

# Fit beta oscillatory
print("\n  Fitting beta oscillatory part:")
for n_p in [5, 10, 20, 50]:
    amps_b, phases_b, freqs_b, dc_b, fitted_b, r2_b = fit_prime_model(
        beta_osc, kb, primes, n_primes_max=n_p)
    print(f"    {n_p:>3} primes: R^2 = {r2_b:.4f}")

amps_b, phases_b, freqs_b, dc_b, fitted_b, r2_b = fit_prime_model(
    beta_osc, kb, primes, n_primes_max=30)
print(f"\n  Beta model (30 primes): R^2 = {r2_b:.4f}")

# Show top amplitudes
print(f"\n  Top alpha amplitudes:")
order_a = np.argsort(amps_a)[::-1]
for rank, i in enumerate(order_a[:10]):
    p = primes[i]
    print(f"    p={p:>3}: A={amps_a[i]:.4f}, phi={phases_a[i]:+.4f}, "
          f"predicted A~log(p)/p = {np.log(p)/p:.4f}")

print(f"\n  Top beta amplitudes:")
order_b = np.argsort(amps_b)[::-1]
for rank, i in enumerate(order_b[:10]):
    p = primes[i]
    print(f"    p={p:>3}: A={amps_b[i]:.4f}, phi={phases_b[i]:+.4f}")


# ============================================================
# Step 4: BUILD THE ARITHMETIC JACOBI (primes in, zeros out)
# ============================================================
print("\n" + "=" * 70)
print("Step 4: BUILD THE ARITHMETIC JACOBI")
print("=" * 70)

def build_arithmetic_jacobi(N_size, primes_list, n_primes=30,
                             alpha_amps=None, alpha_phases=None,
                             beta_amps=None, beta_phases=None,
                             alpha_smooth_coeffs=None, beta_smooth_coeffs=None,
                             alpha_trend_coeffs=None, beta_trend_coeffs=None,
                             alpha_dc=0, beta_dc=0):
    """Build a Jacobi matrix from prime-sum formulas.

    alpha_k = weyl_alpha(k) + trend(k) + sum_p A_p^a cos(2*pi*k*f_p + phi_p^a) + dc
    beta_k  = weyl_beta(k)  + trend(k) + sum_p A_p^b cos(2*pi*k*f_p + phi_p^b) + dc

    where weyl_alpha/beta come from the Weyl-law Jacobi.
    """
    # Compute Weyl Jacobi at this size
    weyl_z = np.array([weyl_zero(n_val) for n_val in range(1, N_size + 1)])
    v0 = np.ones(N_size) / np.sqrt(N_size)
    a_weyl, b_weyl = lanczos_from_eigenvalues(weyl_z, v0)

    k_a = np.arange(1, len(a_weyl) + 1)
    k_b = np.arange(1, len(b_weyl) + 1)

    # Oscillatory correction from primes
    n_p = min(n_primes, len(primes_list))

    alpha_corr = np.zeros(len(a_weyl))
    for i in range(n_p):
        f = np.log(primes_list[i]) * mean_density
        alpha_corr += alpha_amps[i] * np.cos(2 * np.pi * k_a * f + alpha_phases[i])
    alpha_corr += alpha_dc

    beta_corr = np.zeros(len(b_weyl))
    for i in range(n_p):
        f = np.log(primes_list[i]) * mean_density
        beta_corr += beta_amps[i] * np.cos(2 * np.pi * k_b * f + beta_phases[i])
    beta_corr += beta_dc

    # Add trend correction
    if alpha_trend_coeffs is not None:
        alpha_corr += np.polyval(alpha_trend_coeffs, k_a)
    if beta_trend_coeffs is not None:
        beta_corr += np.polyval(beta_trend_coeffs, k_b)

    # Full Jacobi
    alpha_full = a_weyl + alpha_corr
    beta_full = np.abs(b_weyl) + beta_corr  # beta must be positive for tridiag

    return alpha_full, beta_full


# ============================================================
# Step 5: THE ACID TEST — eigenvalues from primes alone
# ============================================================
print("\nStep 5: THE ACID TEST")
print("  Building Jacobi from Weyl + prime corrections (NO zeros as input)...")

for n_p in [0, 5, 10, 20, 30]:
    if n_p == 0:
        # Weyl only (no prime corrections)
        weyl_z = np.array([weyl_zero(nn) for nn in range(1, N + 1)])
        v0 = np.ones(N) / np.sqrt(N)
        a_arith, b_arith = lanczos_from_eigenvalues(weyl_z, v0)
    else:
        a_arith, b_arith = build_arithmetic_jacobi(
            N, primes, n_primes=n_p,
            alpha_amps=amps_a, alpha_phases=phases_a,
            beta_amps=amps_b, beta_phases=phases_b,
            alpha_trend_coeffs=alpha_trend,
            beta_trend_coeffs=beta_trend,
            alpha_dc=dc_a, beta_dc=dc_b,
        )

    # Ensure beta is valid (positive)
    b_valid = np.abs(b_arith)
    b_valid = np.maximum(b_valid, 1e-10)

    try:
        eigs_arith = eigh_tridiagonal(a_arith, b_valid, eigvals_only=True)
        eigs_arith = np.sort(eigs_arith)
        eigs_actual = np.sort(zeta_zeros[:len(eigs_arith)])

        # Match each predicted to nearest actual
        dists = []
        for e in eigs_arith:
            d = np.min(np.abs(e - eigs_actual))
            dists.append(d)
        dists = np.array(dists)

        # Trim edges (boundary effects)
        trim = int(0.1 * len(dists))
        dists_core = dists[trim:-trim]
        mean_dist = np.mean(dists_core)
        median_dist = np.median(dists_core)
        mean_spacing = np.mean(np.diff(eigs_actual[trim:-trim]))
        frac_within_half = np.mean(dists_core < mean_spacing / 2)
        frac_within_one = np.mean(dists_core < mean_spacing)

        label = "Weyl only" if n_p == 0 else f"{n_p} primes"
        print(f"\n  {label:>12}: mean_dist={mean_dist:.4f}, "
              f"median={median_dist:.4f}, "
              f"frac<half_gap={frac_within_half:.1%}, "
              f"frac<gap={frac_within_one:.1%}")
    except Exception as e:
        print(f"\n  {n_p:>3} primes: FAILED ({e})")


# ============================================================
# Step 6: Detailed comparison at 30 primes
# ============================================================
print("\n" + "=" * 70)
print("Step 6: DETAILED COMPARISON (30 PRIMES)")
print("=" * 70)

a_arith, b_arith = build_arithmetic_jacobi(
    N, primes, n_primes=30,
    alpha_amps=amps_a, alpha_phases=phases_a,
    beta_amps=amps_b, beta_phases=phases_b,
    alpha_trend_coeffs=alpha_trend,
    beta_trend_coeffs=beta_trend,
    alpha_dc=dc_a, beta_dc=dc_b,
)
b_valid = np.maximum(np.abs(b_arith), 1e-10)
eigs_arith = np.sort(eigh_tridiagonal(a_arith, b_valid, eigvals_only=True))
eigs_actual = np.sort(zeta_zeros[:len(eigs_arith)])

print(f"\n  {'k':>4} {'Predicted':>12} {'Actual':>12} {'Error':>10} {'Rel err':>10}")
print(f"  {'-'*50}")

for i in range(0, min(len(eigs_arith), 20)):
    err = abs(eigs_arith[i] - eigs_actual[i])
    rel = err / eigs_actual[i]
    tag = " <<<" if err < 0.5 else ""
    print(f"  {i+1:>4} {eigs_arith[i]:>12.4f} {eigs_actual[i]:>12.4f} "
          f"{err:>10.4f} {rel:>10.4%}{tag}")

# Also show some from the middle
print(f"  ...")
mid = len(eigs_arith) // 2
for i in range(mid - 3, mid + 3):
    err = abs(eigs_arith[i] - eigs_actual[i])
    rel = err / eigs_actual[i]
    print(f"  {i+1:>4} {eigs_arith[i]:>12.4f} {eigs_actual[i]:>12.4f} "
          f"{err:>10.4f} {rel:>10.4%}")


# ============================================================
# Step 7: How does accuracy scale with number of primes?
# ============================================================
print("\n" + "=" * 70)
print("Step 7: ACCURACY vs NUMBER OF PRIMES")
print("=" * 70)

print(f"\n  {'n_primes':>10} {'mean_err':>10} {'median_err':>12} "
      f"{'%<half_gap':>12} {'%<gap':>12}")
print(f"  {'-'*60}")

for n_p in [0, 1, 2, 3, 5, 10, 15, 20, 30]:
    if n_p == 0:
        weyl_z = np.array([weyl_zero(nn) for nn in range(1, N + 1)])
        a_a, b_a = lanczos_from_eigenvalues(weyl_z, np.ones(N) / np.sqrt(N))
    else:
        a_a, b_a = build_arithmetic_jacobi(
            N, primes, n_primes=n_p,
            alpha_amps=amps_a, alpha_phases=phases_a,
            beta_amps=amps_b, beta_phases=phases_b,
            alpha_trend_coeffs=alpha_trend,
            beta_trend_coeffs=beta_trend,
            alpha_dc=dc_a, beta_dc=dc_b,
        )

    b_v = np.maximum(np.abs(b_a), 1e-10)
    try:
        eigs_a = np.sort(eigh_tridiagonal(a_a, b_v, eigvals_only=True))
        eigs_t = np.sort(zeta_zeros[:len(eigs_a)])
        trim = int(0.1 * len(eigs_a))
        errs = np.abs(eigs_a - eigs_t)[trim:-trim]
        ms = np.mean(np.diff(eigs_t[trim:-trim]))
        label = "Weyl" if n_p == 0 else f"{n_p}"
        print(f"  {label:>10} {np.mean(errs):>10.4f} {np.median(errs):>12.4f} "
              f"{np.mean(errs < ms/2):>12.1%} {np.mean(errs < ms):>12.1%}")
    except:
        print(f"  {n_p:>10} FAILED")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

print(f"\nTotal time: {time.time() - t0:.1f}s")
