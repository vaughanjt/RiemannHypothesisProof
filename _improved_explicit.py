"""Improved explicit formula: push eigenvalue error below 0.2.

ATTACK PLAN:
1. Fix the +0.887 DC offset (the 7/8 constant)
2. Increase primes from 303 to 1000+
3. Increase harmonics from M=5 to M=10+
4. Add Riemann-Siegel correction terms for S(T)
5. Test: does the improved S(T) give err < 0.2 and thus r > 0.80?

THEORY:
  The k-th zero approximation is:
    gamma_k ~ t_k + S(t_k) / N'(t_k)

  where t_k solves N(t) = k (Weyl zero) and
    S(T) = (1/pi) * arg(zeta(1/2 + iT))
         = -(1/pi) * sum_p sum_m sin(2mT log p) / (m p^{m*sigma})

  At sigma=1/2, this converges CONDITIONALLY.
  The error has two sources:
    a) Truncation of prime sum (currently 303 primes)
    b) Truncation of harmonic sum (currently M=5)
    c) The first-order approximation S/N' itself (higher-order terms exist)

  The Riemann-Siegel formula for S(T) converges much faster:
    S(T) = (1/pi) * Im[log(zeta(1/2+iT))]
  which we can compute directly via mpmath for comparison.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy.stats import pearsonr, kstest
import mpmath
mpmath.mp.dps = 30

t0 = time.time()

N = 200
print(f"Computing {N} zeta zeros at 30-digit precision...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N+1)])

from sympy import primerange
from riemann.analysis.bost_connes_operator import polynomial_unfold

trim = int(0.1 * N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))


# ============================================================
# Hardy Z and corrected r metric
# ============================================================
def hardy_Z(t):
    """Hardy Z-function via mpmath."""
    return float(mpmath.siegelz(t))

def measure_r_hardy(eigs):
    """Peak-gap r using |Z(m_k)| at midpoints."""
    eigs = np.sort(eigs)
    gaps = np.diff(eigs)
    peaks = np.array([abs(hardy_Z((eigs[k] + eigs[k+1]) / 2))
                       for k in range(len(eigs)-1)])
    nt = int(0.1 * len(gaps))
    if nt > 0:
        return pearsonr(gaps[nt:-nt], peaks[nt:-nt])[0]
    return pearsonr(gaps, peaks)[0]

def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s**2 / 4)

def score(eigs, label=""):
    """Full score with Hardy-Z r."""
    eigs = np.sort(eigs)
    errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    r = measure_r_hardy(eigs)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    p_gue = 0.
    if len(sp) > 20:
        sp = sp / np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)
    print(f"  {label:>45}: err={np.mean(errs):.4f}, max={np.max(errs):.4f}, "
          f"r={r:+.4f}, p(GUE)={p_gue:.4f}")
    return np.mean(errs), r


# ============================================================
# Weyl zero solver
# ============================================================
def N_smooth(T):
    if T < 2: return 0.
    return T/(2*np.pi)*np.log(T/(2*np.pi)) - T/(2*np.pi) + 7./8.

def N_deriv(T):
    if T < 2: return .001
    return np.log(T/(2*np.pi)) / (2*np.pi)

def weyl_zero(k):
    t = 2*np.pi*k / np.log(max(k, 2)+2)
    for _ in range(50):
        if t < 1: t = 10.
        t -= (N_smooth(t) - k) / N_deriv(t)
    return t


# ============================================================
# BASELINE: current explicit formula (303 primes, M=5)
# ============================================================
print("\n" + "="*70)
print("BASELINE: CURRENT EXPLICIT FORMULA")
print("="*70)

primes_303 = list(primerange(2, 3000))[:303]

def build_explicit_diagonal(n_primes, M_harmonics, sigma=0.5, offset=0.0):
    """Build explicit formula diagonal with configurable parameters."""
    primes = list(primerange(2, 100000))[:n_primes]
    alpha = np.zeros(N)
    for k in range(1, N+1):
        Tw = weyl_zero(k)
        dN = N_deriv(Tw)
        s = 0.0
        for p in primes:
            lp = np.log(p)
            for m in range(1, M_harmonics+1):
                s -= np.sin(2*m*Tw*lp) / (m * p**(m*sigma))
        s /= np.pi
        alpha[k-1] = Tw + s / dN - offset
    return alpha

alpha_baseline = build_explicit_diagonal(303, 5)
score(alpha_baseline, "Baseline (303 primes, M=5)")

# With offset correction
alpha_offset = alpha_baseline - np.mean(alpha_baseline - zeta_zeros[:N])
score(alpha_offset, "Baseline + offset correction")


# ============================================================
# STEP 1: More primes
# ============================================================
print("\n" + "="*70)
print("STEP 1: MORE PRIMES (M=5 fixed)")
print("="*70)

for n_p in [303, 500, 1000, 2000, 5000]:
    t1 = time.time()
    alpha = build_explicit_diagonal(n_p, 5)
    offset = np.mean(alpha - zeta_zeros[:N])
    alpha_c = alpha - offset
    dt = time.time() - t1
    err, r = score(alpha_c, f"{n_p} primes (offset={offset:+.4f}, {dt:.0f}s)")


# ============================================================
# STEP 2: More harmonics
# ============================================================
print("\n" + "="*70)
print("STEP 2: MORE HARMONICS (1000 primes fixed)")
print("="*70)

for M in [1, 2, 3, 5, 10, 15, 20]:
    t1 = time.time()
    alpha = build_explicit_diagonal(1000, M)
    offset = np.mean(alpha - zeta_zeros[:N])
    alpha_c = alpha - offset
    dt = time.time() - t1
    err, r = score(alpha_c, f"M={M} harmonics (offset={offset:+.4f}, {dt:.0f}s)")


# ============================================================
# STEP 3: Joint optimization (primes x harmonics)
# ============================================================
print("\n" + "="*70)
print("STEP 3: BEST COMBINATION")
print("="*70)

# From steps 1-2, pick the best and run it
t1 = time.time()
alpha_best_trunc = build_explicit_diagonal(2000, 10)
offset_best = np.mean(alpha_best_trunc - zeta_zeros[:N])
alpha_best_c = alpha_best_trunc - offset_best
score(alpha_best_c, f"2000 primes, M=10 ({time.time()-t1:.0f}s)")


# ============================================================
# STEP 4: Direct S(T) from mpmath (the gold standard)
# ============================================================
print("\n" + "="*70)
print("STEP 4: EXACT S(T) FROM MPMATH")
print("="*70)
print("  S(T) = (1/pi) * Im[log(zeta(1/2+iT))]")
print("  This uses mpmath's internal evaluation — no truncation.")

def exact_S(T):
    """Compute S(T) = (1/pi) * arg(zeta(1/2+iT)) exactly."""
    s = mpmath.mpc(0.5, T)
    # arg(zeta(s)) = Im(log(zeta(s)))
    z = mpmath.zeta(s)
    return float(mpmath.im(mpmath.log(z))) / float(mpmath.pi)

print("  Computing exact S(T) for 200 zeros...", flush=True)
t1 = time.time()

alpha_exact_S = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k)
    dN = N_deriv(Tw)
    S_exact = exact_S(Tw)
    alpha_exact_S[k-1] = Tw + S_exact / dN

print(f"  Done in {time.time()-t1:.1f}s")

offset_exact = np.mean(alpha_exact_S - zeta_zeros[:N])
print(f"  Offset before correction: {offset_exact:+.6f}")

score(alpha_exact_S, "Exact S(T), no offset correction")
alpha_exact_c = alpha_exact_S - offset_exact
score(alpha_exact_c, "Exact S(T), offset corrected")


# ============================================================
# STEP 5: Higher-order correction terms
# ============================================================
print("\n" + "="*70)
print("STEP 5: HIGHER-ORDER CORRECTIONS")
print("="*70)
print("  The first-order formula: gamma_k ~ t_k + S(t_k)/N'(t_k)")
print("  Second-order: add S(t_k)^2 * N''(t_k) / (2 * N'(t_k)^3)")
print("  This is Newton's method refinement.")

def N_second_deriv(T):
    """N''(T) = 1/(2*pi*T)."""
    if T < 2: return 0.
    return 1.0 / (2 * np.pi * T)

alpha_second = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k)
    dN = N_deriv(Tw)
    d2N = N_second_deriv(Tw)
    S = exact_S(Tw)

    # First order correction
    delta1 = S / dN

    # Second order: Newton refinement
    # t_1 = Tw + delta1
    # t_2 = t_1 - (N(t_1) - k) / N'(t_1)
    t1_approx = Tw + delta1
    if t1_approx > 2:
        N_t1 = N_smooth(t1_approx) + exact_S(t1_approx) * 0  # smooth part only for stability
        delta2 = -(N_smooth(t1_approx) - k) / N_deriv(t1_approx)
        alpha_second[k-1] = t1_approx + delta2
    else:
        alpha_second[k-1] = Tw + delta1

offset_second = np.mean(alpha_second - zeta_zeros[:N])
score(alpha_second, "Second-order, no offset")
alpha_second_c = alpha_second - offset_second
score(alpha_second_c, "Second-order, offset corrected")


# ============================================================
# STEP 6: Direct Newton refinement from Weyl zeros
# ============================================================
print("\n" + "="*70)
print("STEP 6: NEWTON REFINEMENT USING Z(t)")
print("="*70)
print("  Start from Weyl zeros, refine using Z(t) = 0 via Newton.")
print("  Z'(t) ~ -theta'(t) * Z_approx -- use numerical derivative.")

def newton_refine_zero(t_start, max_iter=5, dt=0.001):
    """Refine a zero approximation using Newton's method on Z(t)."""
    t = t_start
    for _ in range(max_iter):
        Z_val = hardy_Z(t)
        # Numerical derivative
        Z_deriv = (hardy_Z(t + dt) - hardy_Z(t - dt)) / (2 * dt)
        if abs(Z_deriv) < 1e-10:
            break
        t -= Z_val / Z_deriv
    return t

# Start from the BEST explicit formula and refine
print("  Refining from exact-S diagonal...", flush=True)
t1 = time.time()

alpha_newton = np.zeros(N)
for k in range(N):
    # Start from exact S(T) approximation
    t_start = alpha_exact_S[k]
    alpha_newton[k] = newton_refine_zero(t_start, max_iter=10)

print(f"  Done in {time.time()-t1:.1f}s")
score(alpha_newton, "Newton-refined from exact S(T)")


# ============================================================
# STEP 7: Newton from WEYL zeros directly (no explicit formula)
# ============================================================
print("\n" + "="*70)
print("STEP 7: NEWTON FROM WEYL ZEROS (skip explicit formula)")
print("="*70)

t1 = time.time()
alpha_newton_weyl = np.zeros(N)
for k in range(1, N+1):
    t_start = weyl_zero(k)
    alpha_newton_weyl[k-1] = newton_refine_zero(t_start, max_iter=20)

print(f"  Done in {time.time()-t1:.1f}s")
score(alpha_newton_weyl, "Newton-refined from Weyl zeros")

# Check: did we get the RIGHT zeros? (no skips or duplicates)
diffs = np.diff(alpha_newton_weyl)
print(f"  Min gap: {np.min(diffs):.6f} (should be > 0)")
print(f"  Any duplicates: {np.any(diffs < 0.1)}")

# Compare to actual zeros
max_err = np.max(np.abs(alpha_newton_weyl - zeta_zeros[:N]))
print(f"  Max error vs actual zeros: {max_err:.2e}")


# ============================================================
# STEP 8: Build the OPERATOR with Newton-refined diagonal
# ============================================================
print("\n" + "="*70)
print("STEP 8: OPERATOR WITH IMPROVED DIAGONAL")
print("="*70)
print("  Now that we have accurate eigenvalues, add structured off-diagonal.")
print("  The Ramanujan/GCD off-diagonal preserves arithmetic structure")
print("  while Gershgorin bounds keep eigenvalues close to the diagonal.")

from math import gcd

# Build GCD off-diagonal
H_gcd_offdiag = np.zeros((N, N))
for j in range(1, N+1):
    for k in range(j+1, N+1):
        g = gcd(j, k)
        val = np.log(g + 1) / np.sqrt(j * k)
        H_gcd_offdiag[j-1, k-1] = val
        H_gcd_offdiag[k-1, j-1] = val

# Use Newton-refined zeros as diagonal (or best approximation without Newton)
best_diag = alpha_newton.copy()  # Newton from exact S(T)

# Gershgorin: max off-diagonal row sum
max_row_sum = np.max(np.sum(np.abs(H_gcd_offdiag), axis=1))
min_gap = np.min(np.diff(zeta_zeros))

print(f"  Max GCD row sum: {max_row_sum:.4f}")
print(f"  Min zero gap: {min_gap:.4f}")
print(f"  Safe coupling: eps < {min_gap / max_row_sum:.4f}")

print(f"\n  {'eps':>8} {'mean_err':>10} {'r_hardy':>10} {'p(GUE)':>8}")
print(f"  {'-'*42}")

for eps in [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    H = np.diag(best_diag) + eps * H_gcd_offdiag
    eigs = np.sort(np.linalg.eigvalsh(H))
    errs = np.abs(eigs - zeta_zeros[:N])[trim:-trim]
    r = measure_r_hardy(eigs)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    p_gue = 0.
    if len(sp) > 20:
        sp = sp / np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)
    print(f"  {eps:>8.3f} {np.mean(errs):>10.6f} {r:>+10.4f} {p_gue:>8.4f}")


# ============================================================
# STEP 9: The "honest" operator — explicit formula without zeros
# ============================================================
print("\n" + "="*70)
print("STEP 9: HONEST OPERATOR (no zero oracle)")
print("="*70)
print("  Newton refinement uses Z(t)=0 which is circular.")
print("  The HONEST operator uses only: Weyl + S(T) + off-diagonal.")
print("  What's the best r we can get without knowing the zeros?")

# Best honest diagonal: 2000 primes, M=10, offset corrected
# But offset correction also uses zeros! What if we use theoretical offset?
theoretical_offset = 7./8.  # the known constant
alpha_honest = build_explicit_diagonal(2000, 10) - theoretical_offset
err_h, r_h = score(alpha_honest, "Honest: 2000p M=10 - 7/8")

# Try the exact S(T) without offset (it's computable without zeros)
alpha_honest_S = alpha_exact_S.copy()  # exact S is computable without zeros
err_hS, r_hS = score(alpha_honest_S, "Honest: exact S(T), no offset fix")

# The exact S with theoretical 7/8 offset
alpha_honest_S2 = alpha_exact_S - 7./8.
err_hS2, r_hS2 = score(alpha_honest_S2, "Honest: exact S(T) - 7/8")

# Add GCD off-diagonal to the best honest diagonal
print("\n  Adding GCD off-diagonal to best honest diagonal:")
best_honest = alpha_honest_S2 if err_hS2 < err_h else alpha_honest
print(f"\n  {'eps':>8} {'mean_err':>10} {'r_hardy':>10}")
print(f"  {'-'*32}")

for eps in [0, 0.01, 0.05, 0.1, 0.5]:
    H = np.diag(best_honest) + eps * H_gcd_offdiag
    eigs = np.sort(np.linalg.eigvalsh(H))
    errs = np.abs(eigs - zeta_zeros[:N])[trim:-trim]
    r = measure_r_hardy(eigs)
    print(f"  {eps:>8.3f} {np.mean(errs):>10.4f} {r:>+10.4f}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70)
print("VERDICT")
print("="*70)

# Collect all results
print(f"""
  EIGENVALUE ACCURACY LADDER:
    Baseline (303p, M=5):        err ~ 0.93
    + offset correction:         err ~ 0.51
    + more primes (2000):        (see above)
    + more harmonics (M=10):     (see above)
    + exact S(T) from mpmath:    (see above)
    + Newton refinement:         (see above)

  TARGET: err < 0.20 -> r > 0.80

  The key question: can the HONEST operator (no zero oracle)
  achieve err < 0.20 with purely computable corrections?
""")

print(f"Total time: {time.time()-t0:.1f}s")
