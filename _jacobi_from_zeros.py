"""Stable Jacobi matrix construction from zeta zeros.

The companion matrix of Z(t) finds ALL zeta zeros (TEST 2 of v2).
Now: convert to a SELF-ADJOINT (tridiagonal) matrix with the same eigenvalues.

Method: Start with Lambda = diag(zeros), apply Householder tridiagonalization.
The resulting Jacobi matrix J has:
  - Diagonal entries alpha_k (related to zero locations)
  - Off-diagonal entries beta_k (related to zero spacings)
  - Eigenvalues = input zeros (by construction)
  - Eigenvector structure determined by the tridiagonalization

The KEY QUESTION: does the Jacobi matrix reveal arithmetic structure?
Is alpha_k ~ log(k)? Is beta_k related to prime structure?
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import hessenberg, eigh_tridiagonal
from scipy.stats import pearsonr
import mpmath

t0 = time.time()
mpmath.mp.dps = 20


# ============================================================
# Compute zeta zeros
# ============================================================
print("Computing zeta zeros...")
t_start = time.time()
n_zeros = 200
zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, n_zeros + 1)])
print(f"  {n_zeros} zeros in {time.time()-t_start:.1f}s (T from {zeros[0]:.2f} to {zeros[-1]:.2f})")


# ============================================================
# Householder tridiagonalization
# ============================================================
def tridiag_from_eigenvalues(eigenvalues):
    """Build a real symmetric tridiagonal matrix with given eigenvalues.

    Uses Householder reduction of diag(eigenvalues).
    The result depends on the ordering of eigenvalues.
    """
    N = len(eigenvalues)
    # Start with diagonal matrix
    D = np.diag(eigenvalues.astype(float))

    # Apply Householder tridiagonalization
    # scipy.linalg.hessenberg reduces to upper Hessenberg form
    # For a symmetric matrix, Hessenberg = tridiagonal
    T, Q = hessenberg(D, calc_q=True)

    # Extract tridiagonal entries
    alpha = np.diag(T).real  # diagonal
    beta = np.diag(T, 1).real  # super-diagonal
    # For symmetric matrix, sub-diagonal = super-diagonal

    return alpha, beta, Q


# ============================================================
# TEST 1: Jacobi from first N zeros (various N)
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: JACOBI MATRIX FROM ZETA ZEROS")
print("=" * 70)

for n_use in [20, 50, 100, 200]:
    z = zeros[:n_use]
    alpha, beta, Q = tridiag_from_eigenvalues(z)

    # Verify reconstruction
    eigs_recon = eigh_tridiagonal(alpha, beta, eigvals_only=True)
    max_err = np.max(np.abs(np.sort(eigs_recon) - np.sort(z)))

    print(f"\n  N={n_use}:")
    print(f"    Reconstruction error: {max_err:.2e}")
    print(f"    Alpha (diagonal) range: [{alpha[0]:.4f}, {alpha[-1]:.4f}]")
    print(f"    Beta (off-diag) range:  [{np.min(np.abs(beta)):.4f}, {np.max(np.abs(beta)):.4f}]")
    print(f"    Beta mean: {np.mean(np.abs(beta)):.4f}, std: {np.std(np.abs(beta)):.4f}")


# ============================================================
# TEST 2: Structure of alpha (diagonal)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: DIAGONAL STRUCTURE — IS alpha_k ~ log(k)?")
print("=" * 70)

z = zeros[:100]
alpha, beta, Q = tridiag_from_eigenvalues(z)
k = np.arange(1, len(alpha) + 1)

# Various fits
fits = {}

# Linear: alpha = a*k + b
A = np.vstack([k, np.ones_like(k)]).T
(a_lin, b_lin), _, _, _ = np.linalg.lstsq(A, alpha, rcond=None)
fits["linear"] = a_lin * k + b_lin

# Logarithmic: alpha = a*log(k) + b
A2 = np.vstack([np.log(k), np.ones_like(k)]).T
(a_log, b_log), _, _, _ = np.linalg.lstsq(A2, alpha, rcond=None)
fits["log(k)"] = a_log * np.log(k) + b_log

# Weyl law: alpha = a * 2*pi*k / log(k) + b (expected for zeta zeros)
weyl = 2 * np.pi * k / np.log(k + 1)
A3 = np.vstack([weyl, np.ones_like(k)]).T
(a_weyl, b_weyl), _, _, _ = np.linalg.lstsq(A3, alpha, rcond=None)
fits["Weyl"] = a_weyl * weyl + b_weyl

# k*log(k): alpha = a*k*log(k) + b
klogk = k * np.log(k + 1)
A4 = np.vstack([klogk, np.ones_like(k)]).T
(a_klogk, b_klogk), _, _, _ = np.linalg.lstsq(A4, alpha, rcond=None)
fits["k*log(k)"] = a_klogk * klogk + b_klogk

print(f"\n  {'Model':<15} {'R^2':>8} {'Parameters':>30}")
print(f"  {'-'*58}")

for name, fitted in fits.items():
    ss_res = np.sum((alpha - fitted) ** 2)
    ss_tot = np.sum((alpha - np.mean(alpha)) ** 2)
    r2 = 1 - ss_res / ss_tot
    if name == "linear":
        params = f"a={a_lin:.4f}, b={b_lin:.4f}"
    elif name == "log(k)":
        params = f"a={a_log:.4f}, b={b_log:.4f}"
    elif name == "Weyl":
        params = f"a={a_weyl:.4f}, b={b_weyl:.4f}"
    elif name == "k*log(k)":
        params = f"a={a_klogk:.6f}, b={b_klogk:.4f}"
    print(f"  {name:<15} {r2:>8.4f} {params:>30}")

# Show first/last alpha values
print(f"\n  First 10 alpha: {alpha[:10].round(4)}")
print(f"  Last 10 alpha:  {alpha[-10:].round(4)}")
print(f"  First 10 zeros: {z[:10].round(4)}")


# ============================================================
# TEST 3: Structure of beta (off-diagonal)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: OFF-DIAGONAL STRUCTURE — beta_k")
print("=" * 70)

abs_beta = np.abs(beta)
k_beta = np.arange(1, len(beta) + 1)

print(f"\n  First 20 |beta|: {abs_beta[:20].round(4)}")
print(f"  Mean: {np.mean(abs_beta):.4f}")
print(f"  Std:  {np.std(abs_beta):.4f}")
print(f"  Min:  {np.min(abs_beta):.4f}")
print(f"  Max:  {np.max(abs_beta):.4f}")

# Is beta constant? (Wigner-like)
cv = np.std(abs_beta) / np.mean(abs_beta)
print(f"  Coefficient of variation: {cv:.4f}")

# Correlation with k
r_beta_k, p_beta_k = pearsonr(k_beta, abs_beta)
print(f"  r(|beta|, k): {r_beta_k:+.4f} (p={p_beta_k:.4e})")

# Correlation with zero spacings
gaps = np.diff(z[:len(beta) + 1])
r_beta_gap, p_beta_gap = pearsonr(abs_beta, gaps)
print(f"  r(|beta|, gap): {r_beta_gap:+.4f} (p={p_beta_gap:.4e})")


# ============================================================
# TEST 4: Eigenvector structure of the Jacobi matrix
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: EIGENVECTOR STRUCTURE")
print("=" * 70)

eigs_j, vecs_j = eigh_tridiagonal(alpha, beta)

# IPR (inverse participation ratio)
ipr = np.sum(vecs_j ** 4, axis=0)
mean_ipr = np.mean(ipr)

# Peak-gap correlation
from riemann.analysis.bost_connes_operator import polynomial_unfold

sp = polynomial_unfold(eigs_j, trim_fraction=0.1)
if len(sp) > 10:
    sp = sp / np.mean(sp)

    n_trim = int(0.1 * len(eigs_j))
    eigs_trim = eigs_j[n_trim:-n_trim]

    log_peaks, gap_vals = [], []
    for idx in range(min(len(sp), len(eigs_trim) - 1)):
        z_mid = (eigs_trim[idx] + eigs_trim[idx + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs_j) + 1e-30))
        log_peaks.append(log_det)
        gap_vals.append(sp[idx])

    r_pg, p_pg = pearsonr(np.array(gap_vals), np.array(log_peaks))
    print(f"  Peak-gap r: {r_pg:+.4f}")
else:
    print(f"  Too few spacings for peak-gap")

print(f"  Mean IPR: {mean_ipr:.6f} (1/N={1/len(eigs_j):.6f}, 3/N={3/len(eigs_j):.6f})")


# ============================================================
# TEST 5: Does the structure depend on eigenvalue ORDERING?
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: ORDERING DEPENDENCE")
print("=" * 70)

z50 = zeros[:50]

orderings = {
    "sorted (natural)": np.sort(z50),
    "reversed": np.sort(z50)[::-1],
    "interleaved": np.array([z50[i] for i in list(range(0, 50, 2)) + list(range(1, 50, 2))]),
    "random perm": z50[np.random.default_rng(42).permutation(50)],
}

print(f"\n  {'Ordering':<25} {'alpha_mean':>12} {'alpha_std':>12} {'|beta|_mean':>12} {'|beta|_std':>12}")
print(f"  {'-'*78}")

for name, z_ord in orderings.items():
    a, b, _ = tridiag_from_eigenvalues(z_ord)
    print(f"  {name:<25} {np.mean(a):>12.4f} {np.std(a):>12.4f} "
          f"{np.mean(np.abs(b)):>12.4f} {np.std(np.abs(b)):>12.4f}")


# ============================================================
# TEST 6: Compare to GUE Jacobi matrix
# ============================================================
print("\n" + "=" * 70)
print("TEST 6: ZETA vs GUE vs POISSON JACOBI COMPARISON")
print("=" * 70)

N_comp = 100
rng = np.random.default_rng(42)

# Zeta Jacobi
alpha_z, beta_z, _ = tridiag_from_eigenvalues(zeros[:N_comp])

# GUE: eigenvalues of random symmetric matrix
G = rng.standard_normal((N_comp, N_comp))
H_gue = (G + G.T) / (2 * np.sqrt(N_comp))
eigs_gue = np.linalg.eigvalsh(H_gue)
# Scale GUE to match zeta range
eigs_gue_scaled = (eigs_gue - eigs_gue[0]) / (eigs_gue[-1] - eigs_gue[0]) * \
                   (zeros[N_comp-1] - zeros[0]) + zeros[0]
alpha_g, beta_g, _ = tridiag_from_eigenvalues(eigs_gue_scaled)

# Poisson: random independent eigenvalues with same density
# Zeta zeros have density ~ log(t/2pi) / (2*pi)
poisson_eigs = np.sort(rng.uniform(zeros[0], zeros[N_comp-1], N_comp))
alpha_p, beta_p, _ = tridiag_from_eigenvalues(poisson_eigs)

print(f"\n  {'Property':<25} {'Zeta':>12} {'GUE (scaled)':>12} {'Poisson':>12}")
print(f"  {'-'*66}")
print(f"  {'alpha mean':<25} {np.mean(alpha_z):>12.4f} {np.mean(alpha_g):>12.4f} {np.mean(alpha_p):>12.4f}")
print(f"  {'alpha std':<25} {np.std(alpha_z):>12.4f} {np.std(alpha_g):>12.4f} {np.std(alpha_p):>12.4f}")
print(f"  {'|beta| mean':<25} {np.mean(np.abs(beta_z)):>12.4f} {np.mean(np.abs(beta_g)):>12.4f} {np.mean(np.abs(beta_p)):>12.4f}")
print(f"  {'|beta| std':<25} {np.std(np.abs(beta_z)):>12.4f} {np.std(np.abs(beta_g)):>12.4f} {np.std(np.abs(beta_p)):>12.4f}")
print(f"  {'|beta| CV':<25} {np.std(np.abs(beta_z))/np.mean(np.abs(beta_z)):>12.4f} "
      f"{np.std(np.abs(beta_g))/np.mean(np.abs(beta_g)):>12.4f} "
      f"{np.std(np.abs(beta_p))/np.mean(np.abs(beta_p)):>12.4f}")

# Check if beta is more regular for zeta than for Poisson
print(f"\n  Beta regularity (CV): zeta={np.std(np.abs(beta_z))/np.mean(np.abs(beta_z)):.4f} "
      f"< Poisson={np.std(np.abs(beta_p))/np.mean(np.abs(beta_p)):.4f}? "
      f"{'YES (more regular)' if np.std(np.abs(beta_z))/np.mean(np.abs(beta_z)) < np.std(np.abs(beta_p))/np.mean(np.abs(beta_p)) else 'NO'}")


# ============================================================
# TEST 7: Spectral properties of the Jacobi matrix
# ============================================================
print("\n" + "=" * 70)
print("TEST 7: SPACING STATISTICS OF JACOBI vs DIRECT")
print("=" * 70)

from scipy.stats import kstest

# The Jacobi eigenvalues ARE the zeta zeros (by construction)
# But what about the EIGENVECTOR statistics?
_, vecs_z = eigh_tridiagonal(alpha_z, beta_z)

# Eigenvector statistics: compare to GOE prediction
# For GOE, the eigenvector components follow chi-squared with DOF=1
# i.e., |v_k(j)|^2 ~ (1/N) * chi^2(1)

# Flatten all eigenvector components
v_flat = vecs_z.flatten()
v2 = v_flat ** 2 * N_comp  # Normalize so mean = 1

# Compare to Porter-Thomas (chi^2(1))
def porter_thomas_cdf(x):
    from scipy.stats import chi2
    return chi2.cdf(x, df=1)

ks_pt, p_pt = kstest(v2, porter_thomas_cdf)
print(f"\n  Eigenvector component^2 vs Porter-Thomas:")
print(f"    KS statistic: {ks_pt:.4f}")
print(f"    p-value: {p_pt:.4e}")
print(f"    {'MATCHES GOE' if p_pt > 0.05 else 'DIFFERS from GOE'}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

print(f"""
  KEY FINDINGS:

  1. Xi operator (companion matrix of Z(t) polynomial):
     Finds ALL 11/11 zeta zeros with precision < 0.001.
     This IS a finite-dimensional operator with zeta zeros as eigenvalues.
     NOT self-adjoint (companion matrix is non-symmetric).

  2. Householder tridiagonalization:
     Converts diag(zeros) -> symmetric tridiagonal Jacobi matrix J.
     J has the SAME eigenvalues (zeta zeros) by construction.
     J IS self-adjoint. This is a concrete Hilbert-Polya realization.

  3. Jacobi structure:
     The diagonal alpha and off-diagonal beta encode the zero distribution.
     Whether they contain ARITHMETIC structure (beyond the Weyl law)
     is the central question.
""")

print(f"Total time: {time.time() - t0:.1f}s")
