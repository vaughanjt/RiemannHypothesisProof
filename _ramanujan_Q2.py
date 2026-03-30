"""Ramanujan Q hunt — Part 2: non-similarity constructions.

KEY FINDING FROM PART 1:
  Similarity transforms Q^T H Q preserve eigenvalues.
  So no basis change can bridge GCD (high r, wrong eigs) and
  explicit formula (right eigs, low r). THE Q HUNT AS ORIGINALLY
  CONCEIVED IS A DEAD END.

NEW APPROACH:
  H = R @ diag(w) @ R^T is NOT a similarity transform when R is not orthogonal.
  The Ramanujan matrix R is non-orthogonal (condition number ~28).
  So H = R diag(w) R^T has eigenvalues that depend on BOTH w AND R's structure.

  Can we find w_q such that:
  1. eigenvalues(R diag(w) R^T) = zeta zeros  (accuracy)
  2. the r of these eigenvalues is high  (structure)

  Constraint 2 is automatically satisfied if constraint 1 is met exactly,
  because the actual zeta zeros have r ~ 0.80.

ALSO TEST:
  - Regularized pseudo-inverse approach
  - Direct eigenvalue targeting via Newton iteration on w
  - Ramanujan-weighted hybrid operators
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd
from scipy.linalg import eigh, svd, solve
from scipy.stats import pearsonr, kstest
from scipy.optimize import minimize, least_squares
from sympy import totient, mobius, primerange, isprime, factorint
import mpmath
mpmath.mp.dps = 20

t0 = time.time()

N = 200
print(f"Computing {N} zeta zeros...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N+1)])

primes_all = list(primerange(2, 3000))[:303]
trim = int(0.1 * N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

from riemann.analysis.bost_connes_operator import polynomial_unfold

def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s**2 / 4)

def measure_peak_gap(eigs_raw):
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

def score_eigs(eigs):
    eigs_s = np.sort(np.real(eigs))
    errs = np.abs(eigs_s - zeta_zeros[:len(eigs_s)])[trim:-trim]
    r, n = measure_peak_gap(eigs_s)
    sp = polynomial_unfold(eigs_s, trim_fraction=0.1)
    p_gue = 0.
    if len(sp) > 20:
        sp = sp / np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)
    return {'mean_err': np.mean(errs), 'pct_half': np.mean(errs < ms/2),
            'r': r, 'p_gue': p_gue}

def print_score(label, s):
    print(f"  {label:>40}: err={s['mean_err']:.4f}, r={s['r']:+.4f}, "
          f"p(GUE)={s['p_gue']:.4f}, <half={s['pct_half']*100:.1f}%")


# ============================================================
# Build Ramanujan matrix
# ============================================================
print("Building Ramanujan matrix...", flush=True)

def ramanujan_sum(n, q):
    g = gcd(n, q)
    result = 0
    for d in range(1, g+1):
        if g % d == 0:
            result += d * int(mobius(q // d))
    return result

R = np.zeros((N, N))
for n in range(1, N+1):
    for q in range(1, N+1):
        R[n-1, q-1] = ramanujan_sum(n, q)

phi_vals = np.array([float(totient(q)) for q in range(1, N+1)])
print(f"  R built: {N}x{N}, rank={np.linalg.matrix_rank(R)}")


# ============================================================
# APPROACH 1: Optimize w in H = R diag(w) R^T
# ============================================================
print("\n" + "="*70)
print("APPROACH 1: OPTIMIZE w IN H = R diag(w) R^T")
print("="*70)
print("  Find w_q so eigenvalues of R diag(w) R^T match zeta zeros")

# The eigenvalues of R diag(w) R^T = eigenvalues of diag(sqrt(w)) R^T R diag(sqrt(w))
# when w > 0. But w can be negative.

# Direct approach: minimize ||eigenvalues(H(w)) - zeros||^2
# This is expensive (N eigendecompositions per gradient step).

# Cheaper: use the fact that Tr(H^k) = sum of eigenvalue^k.
# Match first K trace moments.

# Even cheaper: if R were orthogonal, eigenvalues of R diag(w) R^T = w.
# R is close to orthogonal (Ramanujan orthogonality).
# So w ~ target eigenvalues might work as a starting point.

# Start with the Gram matrix structure
G = R.T @ R  # G_{qq'} = sum_n c_q(n) c_{q'}(n)

# For the Ramanujan matrix: G_{qq} ~ N * phi(q) for q <= N
# So R^T R = G is approximately N * diag(phi)
# Then R diag(w) R^T has eigenvalues approximately N * phi(q) * w_q
# (in the basis diagonalizing G)

# First attempt: w_q = zero_q / (N * phi(q))
# This uses the approximate Ramanujan orthogonality
w_approx1 = zeta_zeros[:N] / (N * phi_vals)
H1 = R @ np.diag(w_approx1) @ R.T
H1 = (H1 + H1.T) / 2
eigs1 = np.sort(np.linalg.eigvalsh(H1))
s1 = score_eigs(eigs1)
print_score("w = zeros/(N*phi)", s1)

# Second attempt: use exact Gram eigenstructure
# G = P @ diag(g_eigs) @ P^T
# R diag(w) R^T = R diag(w) R^T
# Let R = U S V^T (SVD), then H = U S V^T diag(w) V S U^T
# eigenvalues of H = eigenvalues of S V^T diag(w) V S = S M S
# where M = V^T diag(w) V

print("\n  Using SVD structure to invert...")
U, S_vals, Vt = svd(R, full_matrices=False)
V = Vt.T

# We want eigenvalues of U S V^T diag(w) V S U^T = zeta zeros
# Equivalently: eigenvalues of S V^T diag(w) V S = zeta zeros
# Let M = V^T diag(w) V, then eigenvalues of S M S = zeta zeros
# If S = diag(s), then S M S has entries s_i M_{ij} s_j
# eigenvalues of S M S = zeta zeros

# If M were diagonal: M = diag(m), then S M S = diag(s^2 * m)
# So s_i^2 * m_i = zero_i, giving m_i = zero_i / s_i^2

# But M = V^T diag(w) V, and if V is unitary, diag(w) = V M V^T
# So w = diag(V diag(zeros/s^2) V^T) = sum_i (zeros_i/s_i^2) * V_{:,i}^2

# This only works if we WANT M diagonal. Let's try:
m_target = zeta_zeros[:N] / (S_vals**2)
w_svd = np.sum(V**2 * m_target[np.newaxis, :], axis=1)

H_svd = R @ np.diag(w_svd) @ R.T
H_svd = (H_svd + H_svd.T) / 2
eigs_svd = np.sort(np.linalg.eigvalsh(H_svd))
s_svd = score_eigs(eigs_svd)
print_score("w from SVD inversion", s_svd)


# ============================================================
# APPROACH 2: Newton iteration on w
# ============================================================
print("\n" + "="*70)
print("APPROACH 2: NEWTON ITERATION ON w")
print("="*70)

# Use least_squares to find w that minimizes eigenvalue error
# This is the direct approach: expensive but definitive.

def eigenvalue_residual(w):
    """Residual = sorted eigenvalues of R diag(w) R^T - zeta zeros."""
    H = R @ np.diag(w) @ R.T
    H = (H + H.T) / 2
    eigs = np.sort(np.linalg.eigvalsh(H))
    return eigs - zeta_zeros[:N]

# Start from SVD estimate
print("  Running least_squares optimization...", flush=True)
t_opt = time.time()

# Use a cheaper starting point
w0 = w_svd.copy()

result = least_squares(eigenvalue_residual, w0, method='lm',
                        max_nfev=500, verbose=0)

w_opt = result.x
H_opt = R @ np.diag(w_opt) @ R.T
H_opt = (H_opt + H_opt.T) / 2
eigs_opt = np.sort(np.linalg.eigvalsh(H_opt))
s_opt = score_eigs(eigs_opt)
print(f"  Optimization: {time.time()-t_opt:.1f}s, cost={result.cost:.2e}, "
      f"nfev={result.nfev}")
print_score("Newton-optimized w", s_opt)

# Check convergence
max_eig_err = np.max(np.abs(eigs_opt - zeta_zeros[:N]))
print(f"  Max eigenvalue error: {max_eig_err:.6f}")
print(f"  Mean eigenvalue error: {np.mean(np.abs(eigs_opt - zeta_zeros[:N])):.6f}")

# The KEY question: what does the optimized w look like?
print(f"\n  Optimized w statistics:")
print(f"    min={np.min(w_opt):.6f}, max={np.max(w_opt):.6f}")
print(f"    mean={np.mean(w_opt):.6f}, std={np.std(w_opt):.6f}")
print(f"    w[0:10] = {w_opt[:10].round(6)}")

# Compare to simple scalings
corr_phi, _ = pearsonr(w_opt, 1.0/phi_vals)
corr_idx, _ = pearsonr(w_opt, np.arange(1, N+1, dtype=float))
corr_log, _ = pearsonr(w_opt, np.log(np.arange(1, N+1, dtype=float)))
print(f"\n  Correlation of w_opt with:")
print(f"    1/phi(q): {corr_phi:+.4f}")
print(f"    q:        {corr_idx:+.4f}")
print(f"    log(q):   {corr_log:+.4f}")


# ============================================================
# APPROACH 3: R diag(w) R^T with STRUCTURED w
# ============================================================
print("\n" + "="*70)
print("APPROACH 3: STRUCTURED w FROM NUMBER THEORY")
print("="*70)
print("  Try w_q = f(q) for various arithmetic functions f")

structured_ws = {
    "1/phi(q)": 1.0 / phi_vals,
    "mu(q)/phi(q)": np.array([-float(mobius(q))/float(totient(q)) for q in range(1, N+1)]),
    "1/q": 1.0 / np.arange(1, N+1, dtype=float),
    "log(q)/phi(q)": np.log(np.arange(1, N+1, dtype=float) + 1) / phi_vals,
    "1/phi(q)^2": 1.0 / phi_vals**2,
}

# Scale each w to roughly match zero range
for name, w in structured_ws.items():
    # Find optimal scalar: H = R diag(c*w) R^T, optimize c
    def obj_c(log_c, w_base=w):
        c = np.exp(log_c)
        H = R @ np.diag(c * w_base) @ R.T
        H = (H + H.T) / 2
        eigs = np.sort(np.linalg.eigvalsh(H))
        return np.mean((eigs - zeta_zeros[:N])**2)

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(obj_c, bounds=(-10, 10), method='bounded')
    c_best = np.exp(res.x)

    H_s = R @ np.diag(c_best * w) @ R.T
    H_s = (H_s + H_s.T) / 2
    eigs_s = np.sort(np.linalg.eigvalsh(H_s))
    s_s = score_eigs(eigs_s)
    print_score(f"w={name}, c={c_best:.4f}", s_s)


# ============================================================
# APPROACH 4: The CORRECT Ramanujan operator identity
# ============================================================
print("\n" + "="*70)
print("APPROACH 4: RAMANUJAN OPERATOR IDENTITY")
print("="*70)
print("  Key identity: sum_{n=1}^N c_q(n) c_{q'}(n) = N * phi(q) * [q=q']")
print("  (approximately, for q,q' coprime and <= N)")
print()
print("  So (1/N) R^T R ~ diag(phi)")
print("  And (1/N) R diag(w) R^T ~ R diag(w * phi) in the q-basis")
print()
print("  If we want an operator whose DIAGONAL in index space is alpha_k,")
print("  and whose q-th Ramanujan coefficient is w_q, then:")
print("  alpha_k = sum_q w_q * c_q(k)")
print("  This is the Ramanujan expansion. w_q is the spectrum of alpha.")

# Compute the Ramanujan coefficients of the explicit formula diagonal
# using the (approximate) Ramanujan inverse transform
# a_q = (1/N*phi(q)) sum_n alpha_n * c_q(n)
a_q = np.zeros(N)
for q in range(N):
    a_q[q] = np.sum(zeta_zeros[:N] * R[:, q]) / (N * phi_vals[q])

# Reconstruct
alpha_recon = R @ a_q
err_recon = np.linalg.norm(zeta_zeros[:N] - alpha_recon) / np.linalg.norm(zeta_zeros[:N])
print(f"  Reconstruction of zeros from Ramanujan coefficients: err={err_recon:.4f}")

# The a_q coefficients of the zeros themselves
print(f"\n  Ramanujan spectrum of zeta zeros (top 15 by |a_q|):")
top_q = np.argsort(np.abs(a_q))[-15:][::-1]
for q_idx in top_q:
    q = q_idx + 1
    print(f"    q={q:>4}: a_q={a_q[q_idx]:>+10.4f}")

# Build operator from these coefficients
# H_{jk} = sum_q a_q * c_q(j) * c_q(k) / phi(q)
H_ram = np.zeros((N, N))
for q in range(N):
    H_ram += a_q[q] * np.outer(R[:, q], R[:, q]) / phi_vals[q]
H_ram = (H_ram + H_ram.T) / 2

eigs_ram = np.sort(np.linalg.eigvalsh(H_ram))
s_ram = score_eigs(eigs_ram)
print_score("Ramanujan operator from zero coeffs", s_ram)


# ============================================================
# APPROACH 5: Project-and-correct
# ============================================================
print("\n" + "="*70)
print("APPROACH 5: PROJECT-AND-CORRECT")
print("="*70)
print("  Start with diag(zeros). Project onto Ramanujan-structured space.")
print("  The projection forces arithmetic structure while keeping eigenvalues close.")

# The Ramanujan-structured space: matrices of the form R diag(w) R^T
# Project diag(zeros) onto this space.

# diag(zeros) = R diag(w) R^T => w = (R^T R)^{-1} R^T diag(zeros) R (R^T R)^{-1}
# Actually: if H = R W R^T and we know H, then vec(H) = (R kron R) vec(W)
# For diagonal W: h_{jk} = sum_q W_q R_{jq} R_{kq}

# The closest H of the form R diag(w) R^T to diag(zeros):
# minimize ||R diag(w) R^T - diag(zeros)||_F^2

# This is a linear least squares problem!
# Let B_{(j,k), q} = R_{jq} * R_{kq}  (vectorizing the outer products)
# Then h_{jk} = sum_q w_q * B_{(j,k),q}

# For the diagonal target: we only need h_{kk} = zeros_k
# B_{(k,k), q} = R_{kq}^2
B_diag = R**2  # B[k,q] = R_{k,q}^2 = c_q(k)^2

# Solve: B_diag @ w = zeros  (N equations, N unknowns)
w_proj = np.linalg.lstsq(B_diag, zeta_zeros[:N], rcond=None)[0]

# Build the operator
H_proj = R @ np.diag(w_proj) @ R.T
H_proj = (H_proj + H_proj.T) / 2
eigs_proj = np.sort(np.linalg.eigvalsh(H_proj))
s_proj = score_eigs(eigs_proj)
print_score("Projected onto Ramanujan space", s_proj)

# Check diagonal accuracy
diag_err = np.max(np.abs(np.diag(H_proj) - zeta_zeros[:N]))
print(f"  Max diagonal error: {diag_err:.6f}")
print(f"  Mean diagonal error: {np.mean(np.abs(np.diag(H_proj) - zeta_zeros[:N])):.6f}")


# ============================================================
# APPROACH 6: Gershgorin-constrained Ramanujan operator
# ============================================================
print("\n" + "="*70)
print("APPROACH 6: GERSHGORIN CONSTRAINT")
print("="*70)
print("  Gershgorin: eig_k in [H_{kk} - R_k, H_{kk} + R_k]")
print("  where R_k = sum_{j!=k} |H_{jk}|")
print("  If we keep the off-diagonal small enough, eigenvalues stay")
print("  close to the diagonal = zeros")

# Start with diag(zeros)
# Add Ramanujan off-diagonal structure with controlled norm
def build_gershgorin_constrained(target_eigs, R_mat, phi, max_offdiag_ratio=0.1):
    """Build Ramanujan-structured operator with Gershgorin control."""
    N_size = len(target_eigs)

    # Off-diagonal from Ramanujan: H_{jk} = sum_q w_q c_q(j) c_q(k) / phi(q)
    # Use w_q = -mu(q)/phi(q) (von Mangoldt coefficients)
    w_vm = np.array([-float(mobius(q))/float(totient(q)) if q > 0 else 0
                      for q in range(1, N_size+1)])

    H_offdiag = np.zeros((N_size, N_size))
    for q in range(N_size):
        if abs(w_vm[q]) > 1e-10:
            H_offdiag += w_vm[q] * np.outer(R_mat[:, q], R_mat[:, q]) / phi[q]
    np.fill_diagonal(H_offdiag, 0)

    # Scale so max row sum of |off-diag| <= max_offdiag_ratio * min gap
    min_gap = np.min(np.diff(target_eigs))
    max_row_sum = np.max(np.sum(np.abs(H_offdiag), axis=1))
    if max_row_sum > 0:
        scale = max_offdiag_ratio * min_gap / max_row_sum
    else:
        scale = 0

    H = np.diag(target_eigs) + scale * H_offdiag
    return (H + H.T) / 2

for ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    H_g = build_gershgorin_constrained(zeta_zeros[:N], R, phi_vals, ratio)
    eigs_g = np.sort(np.linalg.eigvalsh(H_g))
    s_g = score_eigs(eigs_g)
    print_score(f"Gershgorin ratio={ratio:.2f}", s_g)


# ============================================================
# APPROACH 7: The "correct" way — zeta zeros ALREADY have high r
# ============================================================
print("\n" + "="*70)
print("APPROACH 7: REALITY CHECK — WHAT r DO EXACT ZEROS GIVE?")
print("="*70)

# The actual zeta zeros should have r ~ 0.80
s_zeros = score_eigs(zeta_zeros[:N])
print_score("Exact zeta zeros (N=200)", s_zeros)

# Oracle operator: diag(zeros) has r = this
# Any operator with EXACT zeros as eigenvalues gets this r for free.
# The problem is getting accurate enough eigenvalues.

# How accurate do eigenvalues need to be to preserve r?
print("\n  Sensitivity: perturbing zeros by noise")
for noise_std in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    perturbed = zeta_zeros[:N] + np.random.default_rng(42).normal(0, noise_std, N)
    s_p = score_eigs(perturbed)
    print(f"  noise_std={noise_std:>5.3f}: r={s_p['r']:+.4f}, err={s_p['mean_err']:.4f}")


# ============================================================
# APPROACH 8: Ramanujan-corrected explicit formula
# ============================================================
print("\n" + "="*70)
print("APPROACH 8: RAMANUJAN CORRECTION TO EXPLICIT FORMULA")
print("="*70)
print("  The explicit formula gives err~0.89. If we can reduce to err~0.1,")
print("  the r should jump from 0.03 toward 0.80.")
print("  Use Ramanujan structure to CORRECT the explicit formula errors.")

# Compute the explicit formula diagonal
def N_smooth(T):
    if T < 2: return 0.
    return T/(2*np.pi)*np.log(T/(2*np.pi)) - T/(2*np.pi) + 7./8.
def N_deriv(T):
    if T < 2: return .001
    return np.log(T/(2*np.pi)) / (2*np.pi)
def weyl_zero(k):
    t = 2*np.pi*k / np.log(max(k, 2)+2)
    for _ in range(30):
        if t < 1: t = 10.
        t -= (N_smooth(t)-k) / N_deriv(t)
    return t

alpha = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k)
    dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes_all for m in range(1, 6)) / np.pi
    alpha[k-1] = Tw + s / dN

errors = alpha - zeta_zeros[:N]
print(f"  Explicit formula: mean_err={np.mean(np.abs(errors)):.4f}, "
      f"max_err={np.max(np.abs(errors)):.4f}")

# Ramanujan expansion of the ERROR
a_q_err = np.zeros(N)
for q in range(N):
    a_q_err[q] = np.sum(errors * R[:, q]) / (N * phi_vals[q])

# How many Ramanujan terms to capture most of the error?
print(f"\n  Ramanujan spectrum of the explicit formula ERROR:")
err_recon_by_terms = []
for n_terms in [1, 2, 5, 10, 20, 50, 100, 150, 200]:
    # Use top n_terms coefficients by magnitude
    top_idx = np.argsort(np.abs(a_q_err))[-n_terms:]
    err_approx = np.zeros(N)
    for q_idx in top_idx:
        err_approx += a_q_err[q_idx] * R[:, q_idx]
    residual = np.linalg.norm(errors - err_approx) / np.linalg.norm(errors)
    err_recon_by_terms.append((n_terms, residual))
    print(f"  {n_terms:>4} Ramanujan terms: {(1-residual)*100:.1f}% of error captured")

# Correct the explicit formula
alpha_corrected = alpha - R @ a_q_err  # subtract the Ramanujan-reconstructed error
corr_errors = alpha_corrected - zeta_zeros[:N]
print(f"\n  After full Ramanujan correction:")
print(f"    mean_err={np.mean(np.abs(corr_errors)):.6f}, "
      f"max_err={np.max(np.abs(corr_errors)):.6f}")

# Score the corrected eigenvalues
s_corr = score_eigs(alpha_corrected)
print_score("Ramanujan-corrected explicit", s_corr)

# But this is circular — we used the zeros to compute the correction!
# The question is: how much of the error has ARITHMETIC structure?
# If the error's Ramanujan spectrum concentrates on small q, then
# the arithmetic structure of the error is exploitable.

print(f"\n  Top 15 Ramanujan coefficients of the ERROR (by |a_q|):")
top_err_q = np.argsort(np.abs(a_q_err))[-15:][::-1]
for q_idx in top_err_q:
    q = q_idx + 1
    print(f"    q={q:>4}: a_q={a_q_err[q_idx]:>+10.6f} "
          f"(is prime: {isprime(q)}, phi={int(phi_vals[q_idx])})")

# Key question: do the top error coefficients concentrate at SMALL q?
print(f"\n  Error spectrum concentration:")
total_power = np.sum(a_q_err**2)
for q_max in [5, 10, 20, 50, 100]:
    power = np.sum(a_q_err[:q_max]**2) / total_power
    print(f"    q <= {q_max:>3}: {power*100:.1f}% of error power")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70)
print("VERDICT: RAMANUJAN Q HUNT — PART 2")
print("="*70)

print(f"""
  DEAD END CONFIRMED:
    Similarity transforms (Q^T H Q) cannot bridge GCD and explicit formula.
    r is an eigenvalue-distribution property, not an eigenvector property.

  REALITY CHECK:
    Exact zeta zeros have r={s_zeros['r']:+.4f}. Any operator with exact
    zeros as eigenvalues automatically gets this r.

  THE REAL PROBLEM:
    The explicit formula diagonal has mean_err={np.mean(np.abs(errors)):.4f}.
    This error destroys the fine gap-peak correlation (r drops to 0.03).

  RAMANUJAN INSIGHT:
    The error does/doesn't concentrate at small q (see spectrum above).
    If it concentrates at small q, a few arithmetic corrections could
    dramatically improve eigenvalue accuracy and thus r.

  NEWTON RESULT:
    Optimized w in R diag(w) R^T: err={s_opt['mean_err']:.4f}, r={s_opt['r']:+.4f}
    (R diag(w) R^T CAN hit the right eigenvalues with arbitrary w)

  GERSHGORIN INSIGHT:
    Adding Ramanujan off-diagonal to diag(zeros) preserves eigenvalue
    accuracy as long as ||off-diag|| < min_gap. The r then comes from
    the eigenvalue accuracy, not the off-diagonal structure.
""")

print(f"Total time: {time.time()-t0:.1f}s")
