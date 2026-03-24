"""Explicit formula Jacobi: define alpha_k and beta_k FROM PRIMES.

TWO INDEPENDENT APPROACHES:

APPROACH A: Canonical starting vectors for Lanczos
  The Lanczos Jacobi depends on the starting vector. Test vectors from physics:
  - Uniform: 1/sqrt(N) (current)
  - 1/gamma: weights from explicit formula (1/|rho|)
  - Z'-weighted: |Z'(t_k)| (zero crossing speed)
  - Density: sqrt(log(t_k/(2*pi))) (local zero density)
  Compare: which gives Jacobi entries with MOST prime structure (highest R^2)?

APPROACH B: Direct construction from the explicit formula (NO Lanczos)
  The Riemann-von Mangoldt formula gives:
    t_k = N^{-1}(k) where N(T) = smooth(T) + S(T)
    S(T) = -(1/pi) * sum_p sum_m sin(2*m*T*log(p)) / (m * p^{m/2})

  So: alpha_k ≈ t_k = weyl_k + S_correction(k)

  For beta_k: use the relation between the Jacobi off-diagonal
  and the local spacing. For a Jacobi matrix with smooth eigenvalues,
  beta_k ≈ (t_{k+1} - t_k) * pi / 4  (Christoffel-Darboux relation)

  This gives BOTH alpha and beta as EXPLICIT functions of primes.
  No fitting, no Lanczos, no zeros as input.
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.stats import pearsonr
import mpmath

t0 = time.time()
mpmath.mp.dps = 20


def lanczos_from_eigenvalues(eigenvalues, start_vec):
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
    for k_idx in range(1, N):
        beta[k_idx - 1] = np.linalg.norm(w)
        if beta[k_idx - 1] < 1e-14:
            return alpha[:k_idx], beta[:k_idx - 1]
        v_new = w / beta[k_idx - 1]
        for j in range(k_idx):
            v_new -= np.dot(V[:, j], v_new) * V[:, j]
        v_new /= np.linalg.norm(v_new)
        V[:, k_idx] = v_new
        w = eigs * v_new
        alpha[k_idx] = np.dot(v_new, w)
        w = w - alpha[k_idx] * v_new - beta[k_idx - 1] * V[:, k_idx - 1]
    return alpha, beta


# ============================================================
# Compute zeros and auxiliary quantities
# ============================================================
print("Computing 500 zeta zeros + Z'(t_k)...")
t_start = time.time()
N = 500
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N + 1)])

# Z'(t_k) at each zero
Z_prime = np.zeros(N)
for i, t in enumerate(zeta_zeros):
    Z_prime[i] = abs(float(mpmath.diff(lambda s: mpmath.siegelz(s), t)))
print(f"  Done ({time.time()-t_start:.1f}s)")


# ============================================================
# Weyl law functions
# ============================================================
def N_weyl(T):
    """Smooth zero-counting function."""
    if T < 2:
        return 0
    return T / (2 * np.pi) * np.log(T / (2 * np.pi)) - T / (2 * np.pi) + 7 / 8


def N_weyl_deriv(T):
    """dN/dT = log(T/(2*pi)) / (2*pi)."""
    if T < 2:
        return 0.001
    return np.log(T / (2 * np.pi)) / (2 * np.pi)


def weyl_zero(n):
    """Invert N(t) = n."""
    t = 2 * np.pi * n / np.log(max(n, 2) + 2)
    for _ in range(30):
        if t < 1:
            t = 10.0
        Nt = N_weyl(t)
        dNt = N_weyl_deriv(t)
        if abs(dNt) < 1e-30:
            break
        t -= (Nt - n) / dNt
    return t


# ============================================================
# The S(T) function from primes
# ============================================================
from sympy import primerange
all_primes = list(primerange(2, 1000))


def S_prime_sum(T, primes, max_m=3):
    """S(T) = -(1/pi) * sum_p sum_m sin(2*m*T*log(p)) / (m * p^{m/2}).

    This is the fluctuating part of the zero-counting function.
    """
    s = 0.0
    for p in primes:
        log_p = np.log(p)
        for m in range(1, max_m + 1):
            s -= np.sin(2 * m * T * log_p) / (m * p ** (m / 2))
    return s / np.pi


# ============================================================
# APPROACH A: Canonical starting vectors
# ============================================================
print("\n" + "=" * 70)
print("APPROACH A: CANONICAL STARTING VECTORS")
print("=" * 70)

# Define starting vectors
start_vectors = {
    "uniform": np.ones(N) / np.sqrt(N),
    "1/gamma": 1.0 / zeta_zeros,
    "1/sqrt(gamma)": 1.0 / np.sqrt(zeta_zeros),
    "Z'(t_k)": Z_prime,
    "1/Z'(t_k)": 1.0 / (Z_prime + 1e-10),
    "sqrt(density)": np.sqrt(np.array([N_weyl_deriv(t) for t in zeta_zeros])),
    "log(k)": np.log(np.arange(1, N + 1) + 1),
}

# For each, build Lanczos Jacobi and measure prime structure
mean_density = np.mean([N_weyl_deriv(t) for t in zeta_zeros])
primes_fit = list(primerange(2, 200))


def measure_prime_R2(signal, k_vals, primes, n_primes=30):
    """Fit prime-frequency model, return R^2."""
    n_p = min(n_primes, len(primes))
    M = np.zeros((len(k_vals), 2 * n_p + 1))
    for i in range(n_p):
        f = np.log(primes[i]) * mean_density
        M[:, 2 * i] = np.cos(2 * np.pi * k_vals * f)
        M[:, 2 * i + 1] = np.sin(2 * np.pi * k_vals * f)
    M[:, -1] = 1
    coeffs = np.linalg.lstsq(M, signal, rcond=None)[0]
    fitted = M @ coeffs
    ss_res = np.sum((signal - fitted) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


print(f"\n  {'Start vector':<18} {'alpha R^2':>10} {'beta R^2':>10} "
      f"{'alpha_std':>10} {'beta_std':>10} {'r(a,zeros)':>12}")
print(f"  {'-'*74}")

weyl_z = np.array([weyl_zero(nn) for nn in range(1, N + 1)])
alpha_weyl, beta_weyl = lanczos_from_eigenvalues(weyl_z, np.ones(N) / np.sqrt(N))

best_name, best_r2 = "", 0

for name, v0 in start_vectors.items():
    alpha, beta = lanczos_from_eigenvalues(zeta_zeros, v0)
    n = min(len(alpha), len(alpha_weyl))

    # Build Weyl Jacobi with SAME starting vector
    a_w, b_w = lanczos_from_eigenvalues(weyl_z, v0)
    n = min(n, len(a_w))

    # Residuals
    alpha_res = alpha[:n] - a_w[:n]
    beta_res = np.abs(beta[:n - 1]) - np.abs(b_w[:n - 1])

    # Detrend
    k_a = np.arange(1, n + 1)
    k_b = np.arange(1, n)
    alpha_det = alpha_res - np.polyval(np.polyfit(k_a, alpha_res, 3), k_a)
    beta_det = beta_res - np.polyval(np.polyfit(k_b, beta_res, 3), k_b)

    # Prime R^2
    r2_a = measure_prime_R2(alpha_det, k_a, primes_fit, 30)
    r2_b = measure_prime_R2(beta_det, k_b, primes_fit, 30)

    r_az, _ = pearsonr(alpha[:n], zeta_zeros[:n])

    print(f"  {name:<18} {r2_a:>10.4f} {r2_b:>10.4f} "
          f"{np.std(alpha_det):>10.4f} {np.std(beta_det):>10.4f} {r_az:>12.6f}")

    if r2_a + r2_b > best_r2:
        best_r2 = r2_a + r2_b
        best_name = name

print(f"\n  Best starting vector: {best_name} (combined R^2 = {best_r2:.4f})")


# ============================================================
# APPROACH B: EXPLICIT FORMULA JACOBI (no Lanczos)
# ============================================================
print("\n" + "=" * 70)
print("APPROACH B: EXPLICIT FORMULA JACOBI — DIRECT FROM PRIMES")
print("=" * 70)

# alpha_k = weyl_zero(k) + S(weyl_zero(k)) / N'(weyl_zero(k))
# beta_k = predicted spacing * normalization

def build_explicit_formula_jacobi(N_size, primes, max_m=3):
    """Build Jacobi matrix directly from the explicit formula.

    alpha_k = t_k^{predicted} = weyl_k + S(weyl_k) / N'(weyl_k)
    beta_k = pi/(4 * N'(weyl_k))  (from Christoffel-Darboux asymptotic)

    S(T) = -(1/pi) sum_p sum_m sin(2mT*log(p)) / (m * p^{m/2})
    """
    alpha = np.zeros(N_size)
    beta = np.zeros(N_size - 1)

    for k in range(1, N_size + 1):
        T_weyl = weyl_zero(k)
        dN = N_weyl_deriv(T_weyl)

        # S(T) correction from primes
        S_T = S_prime_sum(T_weyl, primes, max_m)

        # Predicted zero position
        alpha[k - 1] = T_weyl + S_T / dN

    # beta: use the predicted spacing between consecutive alpha values
    for k in range(N_size - 1):
        # Local spacing from the density
        T_mid = (alpha[k] + alpha[k + 1]) / 2
        dN = N_weyl_deriv(T_mid)
        # The mean spacing is 1/dN (in zero-counting units)
        # For a Jacobi matrix with "uniform" spectral measure,
        # beta_k ~ spacing * pi / 4
        local_spacing = 1.0 / dN
        beta[k] = local_spacing * np.pi / 4

    return alpha, beta


print("\nTesting explicit formula with varying prime sets...")

for n_primes in [0, 5, 10, 30, 50, 100, 168]:
    p_list = all_primes[:n_primes] if n_primes > 0 else []

    alpha_ef, beta_ef = build_explicit_formula_jacobi(N, p_list, max_m=3)

    # Ensure beta is positive
    beta_valid = np.maximum(np.abs(beta_ef), 1e-10)

    try:
        eigs_ef = np.sort(eigh_tridiagonal(alpha_ef, beta_valid, eigvals_only=True))
        eigs_actual = np.sort(zeta_zeros[:len(eigs_ef)])

        # Trim edges
        trim = int(0.1 * len(eigs_ef))
        errs = np.abs(eigs_ef - eigs_actual)[trim:-trim]
        ms = np.mean(np.diff(eigs_actual[trim:-trim]))

        label = "Weyl only" if n_primes == 0 else f"p<={all_primes[n_primes-1] if n_primes <= len(all_primes) else '1000'}"
        pct_half = np.mean(errs < ms / 2) * 100
        pct_full = np.mean(errs < ms) * 100

        print(f"  {n_primes:>4} primes ({label:>12}): "
              f"mean_err={np.mean(errs):.4f}, "
              f"median={np.median(errs):.4f}, "
              f"<half_gap={pct_half:.1f}%, "
              f"<gap={pct_full:.1f}%")
    except Exception as e:
        print(f"  {n_primes:>4} primes: FAILED ({e})")


# ============================================================
# Detailed comparison: explicit formula with 168 primes
# ============================================================
print("\n" + "=" * 70)
print("DETAILED: EXPLICIT FORMULA WITH 168 PRIMES (p < 1000)")
print("=" * 70)

alpha_ef, beta_ef = build_explicit_formula_jacobi(N, all_primes, max_m=5)
beta_valid = np.maximum(np.abs(beta_ef), 1e-10)
eigs_ef = np.sort(eigh_tridiagonal(alpha_ef, beta_valid, eigvals_only=True))

print(f"\n  {'k':>4} {'Predicted':>12} {'Actual':>12} {'Error':>10} {'Rel%':>8}")
print(f"  {'-'*48}")

for i in list(range(20)) + list(range(245, 255)):
    if i >= len(eigs_ef):
        break
    err = abs(eigs_ef[i] - zeta_zeros[i])
    rel = err / zeta_zeros[i] * 100
    tag = " <<<" if err < 0.3 else ""
    print(f"  {i+1:>4} {eigs_ef[i]:>12.4f} {zeta_zeros[i]:>12.4f} "
          f"{err:>10.4f} {rel:>7.3f}%{tag}")


# ============================================================
# Compare: alpha vs actual zeros (are they close?)
# ============================================================
print("\n" + "=" * 70)
print("ALPHA vs ACTUAL ZEROS (before tridiagonalization)")
print("=" * 70)

alpha_ef_168, _ = build_explicit_formula_jacobi(N, all_primes, max_m=5)
alpha_errs = np.abs(alpha_ef_168 - zeta_zeros)
trim = int(0.1 * N)
alpha_errs_core = alpha_errs[trim:-trim]
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

print(f"\n  The alpha_k from the explicit formula ARE the predicted zeros.")
print(f"  How good are they BEFORE forming the Jacobi matrix?")
print(f"\n  Mean |alpha_k - t_k|: {np.mean(alpha_errs_core):.4f}")
print(f"  Median:               {np.median(alpha_errs_core):.4f}")
print(f"  Mean spacing:         {ms:.4f}")
print(f"  % within half gap:    {np.mean(alpha_errs_core < ms/2)*100:.1f}%")
print(f"  % within full gap:    {np.mean(alpha_errs_core < ms)*100:.1f}%")

# Breakdown by number of primes
print(f"\n  {'n_primes':>10} {'mean|alpha-t|':>15} {'median':>10} {'%<half_gap':>12}")
print(f"  {'-'*50}")

for n_p in [0, 5, 10, 30, 50, 100, 168]:
    p_list = all_primes[:n_p] if n_p > 0 else []
    a_ef, _ = build_explicit_formula_jacobi(N, p_list, max_m=3)
    errs = np.abs(a_ef - zeta_zeros)[trim:-trim]
    label = "Weyl" if n_p == 0 else str(n_p)
    print(f"  {label:>10} {np.mean(errs):>15.4f} {np.median(errs):>10.4f} "
          f"{np.mean(errs < ms/2)*100:>11.1f}%")


# ============================================================
# The REAL acid test: eigenvalues from ONLY primes + Weyl
# ============================================================
print("\n" + "=" * 70)
print("THE REAL ACID TEST: EIGENVALUES FROM PRIMES + WEYL ALONE")
print("=" * 70)

# Use alpha from explicit formula (primes determine the diagonal)
# Use beta from local density (Weyl determines the off-diagonal)
# NO zeros used at any point

for n_p, max_m in [(0, 1), (10, 2), (50, 3), (100, 3), (168, 3), (168, 5)]:
    p_list = all_primes[:n_p] if n_p > 0 else []
    a_ef, b_ef = build_explicit_formula_jacobi(N, p_list, max_m)
    b_v = np.maximum(np.abs(b_ef), 1e-10)

    eigs_pred = np.sort(eigh_tridiagonal(a_ef, b_v, eigvals_only=True))

    errs = np.abs(eigs_pred - zeta_zeros)[trim:-trim]
    ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

    label = f"p={n_p},m={max_m}"
    print(f"  {label:>12}: mean_err={np.mean(errs):.4f}, "
          f"<half_gap={np.mean(errs<ms/2)*100:.1f}%, "
          f"<gap={np.mean(errs<ms)*100:.1f}%")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

# Final comparison
a_ef_best, b_ef_best = build_explicit_formula_jacobi(N, all_primes, max_m=5)
b_v = np.maximum(np.abs(b_ef_best), 1e-10)
eigs_best = np.sort(eigh_tridiagonal(a_ef_best, b_v, eigvals_only=True))

errs_best = np.abs(eigs_best - zeta_zeros)[trim:-trim]
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

print(f"""
  EXPLICIT FORMULA JACOBI (168 primes, max_m=5):
    alpha_k = weyl_zero(k) + S(weyl_k)/N'(weyl_k)
    beta_k  = pi / (4 * N'(weyl_k))
    where S(T) = -(1/pi) sum_{{p<1000}} sum_{{m=1}}^5 sin(2mT*log(p)) / (m*p^{{m/2}})

  Eigenvalues vs actual zeta zeros (500 zeros, core 80%):
    Mean error:      {np.mean(errs_best):.4f}
    Median error:    {np.median(errs_best):.4f}
    Mean spacing:    {ms:.4f}
    Within half gap: {np.mean(errs_best < ms/2)*100:.1f}%
    Within full gap: {np.mean(errs_best < ms)*100:.1f}%

  This operator is defined ENTIRELY by:
    1. The Weyl law (smooth zero density)
    2. The prime numbers (via S(T))
    3. Two universal constants (pi, 7/8)

  NO zeros were used in the construction.
""")

print(f"Total time: {time.time() - t0:.1f}s")
