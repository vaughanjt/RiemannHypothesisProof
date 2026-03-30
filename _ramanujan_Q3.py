"""Ramanujan Q hunt — Part 3: corrected r metric + 7/8 fix.

FIXES FROM GROK'S REVIEW:
1. r was computed using |det(z_mid - eigs)| (wrong proxy).
   Must use |Z(m_k)| — the Hardy Z-function at gap midpoints.
   Correct r for 200 zeros is +0.73, not +0.04.

2. The +0.887 DC offset is the 7/8 constant from N(T) counting formula.
   Check if it's being double-counted or misapplied in the explicit formula.

With the correct r metric, the Q hunt is back on:
- GCD r=+0.59 was measuring something real (if different metric)
- Explicit formula r=+0.03 IS a genuine deficiency
- Exact zeros r=+0.73 is the real target
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd
from scipy.linalg import eigh, svd, qr
from scipy.stats import pearsonr, kstest
from scipy.optimize import minimize_scalar, least_squares
from sympy import totient, mobius, primerange
import mpmath
mpmath.mp.dps = 30

t0 = time.time()

N = 200
print(f"Computing {N} zeta zeros at 30-digit precision...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N+1)])

primes_all = list(primerange(2, 3000))[:303]
trim = int(0.1 * N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

from riemann.analysis.bost_connes_operator import polynomial_unfold


# ============================================================
# CORRECTED r METRIC: Hardy Z-function at midpoints
# ============================================================

def hardy_Z(t):
    """Compute the Hardy Z-function: Z(t) = exp(i*theta(t)) * zeta(1/2 + it).
    Z(t) is real-valued and |Z(t)| gives the true peak amplitude."""
    s = mpmath.mpc(0.5, t)
    z_val = mpmath.zeta(s)
    # theta(t) = arg(gamma(1/4 + it/2)) - (t/2)*log(pi)
    theta = mpmath.siegeltheta(t)
    Z = float(mpmath.re(mpmath.exp(1j * theta) * z_val))
    return Z


def measure_peak_gap_hardy(zeros, max_zeros=None):
    """Correct peak-gap r using |Z(m_k)| at midpoints.

    r = Pearson correlation between:
      - gaps: g_k = gamma_{k+1} - gamma_k (unnormalized spacing)
      - peaks: |Z(m_k)| where m_k = (gamma_k + gamma_{k+1}) / 2
    """
    if max_zeros is not None:
        zeros = zeros[:max_zeros]
    n = len(zeros)
    gaps = np.diff(zeros)  # g_k = gamma_{k+1} - gamma_k

    # Compute |Z(m_k)| at each midpoint
    print(f"    Computing |Z(m_k)| at {n-1} midpoints...", flush=True)
    peaks = np.zeros(n - 1)
    for k in range(n - 1):
        m_k = (zeros[k] + zeros[k+1]) / 2
        peaks[k] = abs(hardy_Z(m_k))

    # Trim edges (10%)
    nt = int(0.1 * len(gaps))
    if nt > 0:
        gaps_t = gaps[nt:-nt]
        peaks_t = peaks[nt:-nt]
    else:
        gaps_t = gaps
        peaks_t = peaks

    r, p = pearsonr(gaps_t, peaks_t)
    return r, p, len(gaps_t)


def measure_peak_gap_old(eigs_raw):
    """OLD metric (|det| proxy) for comparison."""
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20: return 0., 0
    sp = sp / np.mean(sp)
    nt = int(0.1 * len(eigs)); et = eigs[nt:-nt]
    lp, ga = [], []
    for k in range(min(len(sp), len(et)-1)):
        z = (et[k] + et[k+1]) / 2
        lp.append(np.sum(np.log(np.abs(z - eigs) + 1e-30)))
        ga.append(sp[k])
    if len(ga) < 10: return 0., 0
    return pearsonr(np.array(ga), np.array(lp))[0], len(ga)


def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s**2 / 4)


# ============================================================
# STEP 1: Correct r for exact zeros
# ============================================================
print("\n" + "="*70)
print("STEP 1: CORRECT r FOR EXACT ZETA ZEROS")
print("="*70)

r_hardy, p_hardy, n_pts = measure_peak_gap_hardy(zeta_zeros)
r_old, _ = measure_peak_gap_old(zeta_zeros)

print(f"\n  Exact zeros (N={N}):")
print(f"    Hardy Z r:  {r_hardy:+.4f} (p={p_hardy:.2e}, n={n_pts})")
print(f"    Old det r:  {r_old:+.4f}")
print(f"    Gap: {abs(r_hardy - r_old):.4f}")
print(f"\n  Grok reports: r = +0.73 for N=200. {'MATCHES' if abs(r_hardy - 0.73) < 0.05 else 'DIFFERS'}")


# ============================================================
# STEP 2: Fix the 7/8 offset in explicit formula
# ============================================================
print("\n" + "="*70)
print("STEP 2: THE 7/8 OFFSET FIX")
print("="*70)

def N_smooth(T):
    """Riemann-von Mangoldt counting function N(T)."""
    if T < 2: return 0.
    return T/(2*np.pi)*np.log(T/(2*np.pi)) - T/(2*np.pi) + 7./8.

def N_deriv(T):
    if T < 2: return .001
    return np.log(T/(2*np.pi)) / (2*np.pi)

def weyl_zero(k):
    """Solve N(t) = k for t (the k-th Weyl-law zero approximation)."""
    t = 2*np.pi*k / np.log(max(k, 2)+2)
    for _ in range(50):
        if t < 1: t = 10.
        t -= (N_smooth(t) - k) / N_deriv(t)
    return t

# Check: does the Weyl approximation already include 7/8?
print("  Weyl zero approximations vs actual zeros:")
for k in [1, 5, 10, 50, 100, 200]:
    tw = weyl_zero(k)
    actual = zeta_zeros[k-1]
    print(f"    k={k:>3}: Weyl={tw:.4f}, actual={actual:.4f}, diff={tw-actual:+.4f}")

# Build explicit formula diagonal — CURRENT version
alpha_current = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k)
    dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes_all for m in range(1, 6)) / np.pi
    alpha_current[k-1] = Tw + s / dN

err_current = alpha_current - zeta_zeros[:N]
print(f"\n  Current explicit formula: mean|err|={np.mean(np.abs(err_current)):.4f}")
print(f"  Mean error (signed): {np.mean(err_current):+.4f}")
print(f"  The systematic offset is {np.mean(err_current):+.4f}")

# The 7/8 = 0.875. If the offset is +0.887, the difference is 0.012.
# This small residual could be from arg(Gamma(1/4)) in theta(t).

# Try: subtract the mean offset
alpha_centered = alpha_current - np.mean(err_current)
err_centered = alpha_centered - zeta_zeros[:N]
print(f"\n  After centering: mean|err|={np.mean(np.abs(err_centered)):.4f}")

# Now compute the CORRECT r for each version
# For operator eigenvalues, we can't use the Hardy Z directly
# (the operator doesn't have a "Z function").
# But we CAN check: if eigenvalues = exact zeros, r = 0.73.
# So how close do eigenvalues need to be to preserve r = 0.73?

# Sensitivity test with Hardy Z metric
print("\n  Sensitivity of Hardy-Z r to eigenvalue noise:")
rng = np.random.default_rng(42)
for noise_std in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    perturbed = zeta_zeros[:N] + rng.normal(0, noise_std, N)
    perturbed = np.sort(perturbed)  # must stay sorted
    # For perturbed eigenvalues, compute Z at their midpoints
    gaps_p = np.diff(perturbed)
    peaks_p = np.zeros(N-1)
    for k in range(N-1):
        m = (perturbed[k] + perturbed[k+1]) / 2
        peaks_p[k] = abs(hardy_Z(m))
    nt = int(0.1 * len(gaps_p))
    r_p, _ = pearsonr(gaps_p[nt:-nt], peaks_p[nt:-nt])
    mean_err = np.mean(np.abs(perturbed - zeta_zeros[:N]))
    print(f"    noise={noise_std:.2f}: r={r_p:+.4f}, mean_err={mean_err:.4f}")


# ============================================================
# STEP 3: Ramanujan correction with correct metric
# ============================================================
print("\n" + "="*70)
print("STEP 3: RAMANUJAN ERROR CORRECTION + CORRECT r")
print("="*70)

# Build Ramanujan matrix
def ramanujan_sum(n, q):
    g = gcd(n, q)
    result = 0
    for d in range(1, g+1):
        if g % d == 0:
            result += d * int(mobius(q // d))
    return result

print("Building Ramanujan matrix...", flush=True)
R = np.zeros((N, N))
for n in range(1, N+1):
    for q in range(1, N+1):
        R[n-1, q-1] = ramanujan_sum(n, q)

phi_vals = np.array([float(totient(q)) for q in range(1, N+1)])

# Ramanujan coefficients of the error
errors = alpha_current - zeta_zeros[:N]
a_q_err = np.zeros(N)
for q in range(N):
    a_q_err[q] = np.sum(errors * R[:, q]) / (N * phi_vals[q])

# Progressive correction: add top-k Ramanujan error terms
print("\n  Progressive Ramanujan correction (correct Hardy-Z r):")
print(f"  {'terms':>6} {'mean_err':>10} {'max_err':>10} {'r_hardy':>10}")
print(f"  {'-'*40}")

sorted_q = np.argsort(np.abs(a_q_err))[::-1]  # largest first

for n_terms in [0, 1, 2, 3, 5, 10, 20, 50, 100]:
    correction = np.zeros(N)
    for i in range(min(n_terms, N)):
        q_idx = sorted_q[i]
        correction += a_q_err[q_idx] * R[:, q_idx]

    alpha_fixed = alpha_current - correction
    fix_err = np.abs(alpha_fixed - zeta_zeros[:N])

    # Compute Hardy-Z r for the corrected eigenvalues
    alpha_sorted = np.sort(alpha_fixed)
    gaps_f = np.diff(alpha_sorted)
    peaks_f = np.zeros(N-1)
    for k in range(N-1):
        m = (alpha_sorted[k] + alpha_sorted[k+1]) / 2
        peaks_f[k] = abs(hardy_Z(m))
    nt = int(0.1 * len(gaps_f))
    r_f, _ = pearsonr(gaps_f[nt:-nt], peaks_f[nt:-nt])

    label = f"{n_terms}"
    if n_terms <= 5 and n_terms > 0:
        qs = [sorted_q[i]+1 for i in range(n_terms)]
        label += f" (q={qs})"
    print(f"  {label:>30} {np.mean(fix_err):>10.4f} {np.max(fix_err):>10.4f} {r_f:>+10.4f}")


# ============================================================
# STEP 4: Check what q=1 correction IS
# ============================================================
print("\n" + "="*70)
print("STEP 4: ANATOMY OF THE q=1 CORRECTION")
print("="*70)

q1_coeff = a_q_err[0]
print(f"  a_1 (q=1 Ramanujan coefficient of error) = {q1_coeff:+.6f}")
print(f"  c_1(n) = 1 for all n, so the q=1 term is a CONSTANT OFFSET")
print(f"  Predicted offset: {q1_coeff:+.6f}")
print(f"  Actual mean error: {np.mean(errors):+.6f}")
print(f"  7/8 = {7/8:.6f}")
print(f"  Difference from 7/8: {abs(q1_coeff) - 7/8:+.6f}")

# The Weyl formula includes 7/8. The explicit formula correction S(T)
# adds oscillatory terms. The question: is the 7/8 being properly
# absorbed into the Weyl zeros, or is there a mismatch?

# Check: Weyl(k) vs actual zero, without the S correction
for k in [1, 50, 100, 200]:
    tw = weyl_zero(k)
    actual = zeta_zeros[k-1]
    print(f"  k={k:>3}: Weyl(k)={tw:.4f}, zero={actual:.4f}, "
          f"Weyl-zero={tw-actual:+.4f}, S correction={alpha_current[k-1]-tw:+.4f}")


# ============================================================
# STEP 5: GCD kernel with corrected metric
# ============================================================
print("\n" + "="*70)
print("STEP 5: GCD KERNEL — CORRECTED r METRIC")
print("="*70)

def build_gcd_kernel(N_size, weight_func=None):
    if weight_func is None:
        weight_func = lambda g: np.log(g + 1)
    H = np.zeros((N_size, N_size))
    for j in range(1, N_size+1):
        for k in range(j, N_size+1):
            g = gcd(j, k)
            val = weight_func(g) / np.sqrt(j * k)
            H[j-1, k-1] = val
            H[k-1, j-1] = val
    return H

H_gcd = build_gcd_kernel(N)
eigs_gcd = np.sort(np.linalg.eigvalsh(H_gcd))

# For GCD eigenvalues, the "midpoint Z" doesn't make physical sense
# because the eigenvalues aren't zeta zeros. But we can still compute
# the gap-peak correlation using the OLD metric for comparison.
r_gcd_old, _ = measure_peak_gap_old(eigs_gcd)
print(f"  GCD kernel r (old det metric): {r_gcd_old:+.4f}")

# For the Hardy-Z metric, we need eigenvalues that ARE zero approximations.
# The GCD eigenvalues are NOT zeros, so Hardy-Z r doesn't apply to them.
# The meaningful comparison is:
# - Exact zeros: Hardy-Z r = ? (computed above)
# - Explicit formula eigenvalues: Hardy-Z r = ?
# - Corrected eigenvalues: Hardy-Z r = ?

# Compute Hardy-Z r for the explicit formula eigenvalues
alpha_sorted = np.sort(alpha_current)
gaps_a = np.diff(alpha_sorted)
peaks_a = np.zeros(N-1)
for k in range(N-1):
    m = (alpha_sorted[k] + alpha_sorted[k+1]) / 2
    peaks_a[k] = abs(hardy_Z(m))
nt = int(0.1 * len(gaps_a))
r_alpha, _ = pearsonr(gaps_a[nt:-nt], peaks_a[nt:-nt])

print(f"\n  Hardy-Z r comparison:")
print(f"    Exact zeros:     r = {r_hardy:+.4f}")
print(f"    Explicit formula: r = {r_alpha:+.4f}")
print(f"    Gap: {r_hardy - r_alpha:+.4f}")


# ============================================================
# STEP 6: Hybrid operator with corrected metric
# ============================================================
print("\n" + "="*70)
print("STEP 6: HYBRID OPERATOR — EXPLICIT DIAG + GCD OFF-DIAG")
print("="*70)
print("  Now with the CORRECT r metric")

GCD_offdiag = H_gcd - np.diag(np.diag(H_gcd))

# Use the Ramanujan-corrected diagonal (top 5 terms)
correction_5 = np.zeros(N)
for i in range(5):
    q_idx = sorted_q[i]
    correction_5 += a_q_err[q_idx] * R[:, q_idx]
alpha_fixed5 = alpha_current - correction_5

print(f"\n  Diagonal: Ramanujan-corrected explicit formula (5 terms)")
print(f"  Off-diagonal: GCD kernel")
print(f"  Sweeping coupling strength eps...")

print(f"\n  {'eps':>8} {'mean_err':>10} {'r_hardy':>10} {'p(GUE)':>8}")
print(f"  {'-'*40}")

best_r_hardy = -1
best_eps_hardy = 0

for eps in [0, 0.5, 1, 2, 5, 10, 20, 50, 100]:
    H = np.diag(alpha_fixed5) + eps * GCD_offdiag
    eigs = np.sort(np.linalg.eigvalsh(H))

    # Hardy-Z r
    gaps_h = np.diff(eigs)
    peaks_h = np.zeros(N-1)
    for k in range(N-1):
        m = (eigs[k] + eigs[k+1]) / 2
        peaks_h[k] = abs(hardy_Z(m))
    nt = int(0.1 * len(gaps_h))
    r_h, _ = pearsonr(gaps_h[nt:-nt], peaks_h[nt:-nt])

    # Error
    errs = np.abs(eigs - zeta_zeros[:N])[trim:-trim]

    # GUE test
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    p_gue = 0.
    if len(sp) > 20:
        sp = sp / np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)

    marker = ""
    if r_h > best_r_hardy:
        best_r_hardy = r_h; best_eps_hardy = eps; marker = " <--"
    print(f"  {eps:>8.1f} {np.mean(errs):>10.4f} {r_h:>+10.4f} {p_gue:>8.4f}{marker}")

print(f"\n  Best: eps={best_eps_hardy}, r_hardy={best_r_hardy:+.4f}")
print(f"  Target: r_hardy = {r_hardy:+.4f} (exact zeros)")


# ============================================================
# STEP 7: Ramanujan-structured hybrid
# ============================================================
print("\n" + "="*70)
print("STEP 7: RAMANUJAN-STRUCTURED OFF-DIAGONAL")
print("="*70)
print("  Instead of raw GCD kernel, use Ramanujan-weighted coupling:")
print("  H_{jk} = alpha_k * delta_{jk} + eps * sum_q w_q c_q(j) c_q(k) / phi(q)")

# von Mangoldt Ramanujan weights: w_q = -mu(q)/phi(q)
w_vm = np.array([-float(mobius(q))/float(totient(q)) if q > 0 else 0
                  for q in range(1, N+1)])

# Build Ramanujan off-diagonal
H_ram_offdiag = np.zeros((N, N))
for q in range(N):
    if abs(w_vm[q]) > 1e-10:
        H_ram_offdiag += w_vm[q] * np.outer(R[:, q], R[:, q]) / phi_vals[q]
np.fill_diagonal(H_ram_offdiag, 0)

print(f"  ||GCD off-diag||_F = {np.linalg.norm(GCD_offdiag, 'fro'):.4f}")
print(f"  ||Ramanujan off-diag||_F = {np.linalg.norm(H_ram_offdiag, 'fro'):.4f}")

print(f"\n  {'eps':>8} {'mean_err':>10} {'r_hardy':>10} {'p(GUE)':>8}")
print(f"  {'-'*40}")

best_r_ram = -1
for eps in [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    vn = np.linalg.norm(H_ram_offdiag, ord=2)
    H = np.diag(alpha_fixed5) + eps / max(vn, 1e-10) * H_ram_offdiag
    eigs = np.sort(np.linalg.eigvalsh(H))

    gaps_h = np.diff(eigs)
    peaks_h = np.zeros(N-1)
    for k in range(N-1):
        m = (eigs[k] + eigs[k+1]) / 2
        peaks_h[k] = abs(hardy_Z(m))
    nt = int(0.1 * len(gaps_h))
    r_h, _ = pearsonr(gaps_h[nt:-nt], peaks_h[nt:-nt])

    errs = np.abs(eigs - zeta_zeros[:N])[trim:-trim]
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    p_gue = 0.
    if len(sp) > 20:
        sp = sp / np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)

    marker = ""
    if r_h > best_r_ram:
        best_r_ram = r_h; marker = " <--"
    print(f"  {eps:>8.3f} {np.mean(errs):>10.4f} {r_h:>+10.4f} {p_gue:>8.4f}{marker}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70)
print("VERDICT: RAMANUJAN Q HUNT — PART 3 (CORRECTED)")
print("="*70)

print(f"""
  METRIC FIX:
    Hardy-Z r for exact zeros (N={N}): {r_hardy:+.4f}
    Old det r for exact zeros:         {r_old:+.4f}
    Grok's value:                      +0.73

  OFFSET FIX:
    q=1 Ramanujan coefficient: {q1_coeff:+.6f}
    7/8 = 0.875, diff = {abs(q1_coeff) - 0.875:+.6f}

  BEST RESULTS:
    GCD hybrid:      eps={best_eps_hardy}, r_hardy={best_r_hardy:+.4f}
    Ramanujan hybrid: best r_hardy={best_r_ram:+.4f}
    Target:          r_hardy={r_hardy:+.4f}
""")

print(f"Total time: {time.time()-t0:.1f}s")
