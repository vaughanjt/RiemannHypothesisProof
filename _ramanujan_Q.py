"""Hunt for Q: Ramanujan sum basis transformation.

THE IDEA:
  Two operators that each capture HALF the zeta zero structure:
    1. GCD operator: H_{jk} = log(gcd(j,k))/sqrt(jk) -> r=0.68, wrong eigenvalues
    2. Explicit formula: H_{kk} = weyl(k) + S(k) -> right eigenvalues, r=0.03

  If Q is the basis transformation between them:
    H_spectral = Q^T H_GCD Q

  Then Q simultaneously diagonalizes the arithmetic structure (GCD)
  into the spectral structure (zeros).

WHY RAMANUJAN SUMS:
  c_q(n) = sum_{(a,q)=1} e^{2*pi*i*a*n/q} = sum_{d|gcd(n,q)} d * mu(q/d)

  Key identity: gcd(j,k) = sum_q phi(q) * c_q(j) * c_q(k) / phi(q)^2
  (Ramanujan expansion of the GCD function!)

  Ramanujan showed: Lambda(n) = -sum_q mu(q)/phi(q) * c_q(n)
  (von Mangoldt function in Ramanujan basis!)

  So Ramanujan sums ARE the bridge between GCD and the explicit formula.
  The matrix Q_{nq} = c_q(n) / ||c_q|| transforms between index space
  (where GCD is natural) and frequency space (where primes are natural).

PLAN:
  1. Build the Ramanujan sum matrix R_{nq} = c_q(n) for n,q = 1..N
  2. Orthogonalize R to get Q (QR or SVD)
  3. Transform H_GCD into Q basis: H' = Q^T H_GCD Q
  4. Compare eigenvalues of H' to zeta zeros
  5. Measure peak-gap r of H'
  6. Test: does Q^T (explicit formula) Q give better r?
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd
from scipy.linalg import eigh, svd, qr
from scipy.stats import pearsonr, kstest
from sympy import totient, mobius, primerange
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
    """Score eigenvalues against zeta zeros."""
    eigs_s = np.sort(np.real(eigs))
    errs = np.abs(eigs_s - zeta_zeros[:len(eigs_s)])[trim:-trim]
    r, n = measure_peak_gap(eigs_s)
    sp = polynomial_unfold(eigs_s, trim_fraction=0.1)
    p_gue = 0.
    if len(sp) > 20:
        sp = sp / np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)
    return {
        'mean_err': np.mean(errs),
        'pct_half': np.mean(errs < ms/2),
        'r': r,
        'p_gue': p_gue,
    }

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

def print_score(label, s):
    print(f"  {label:>35}: err={s['mean_err']:.4f}, r={s['r']:+.4f}, "
          f"p(GUE)={s['p_gue']:.4f}, <half={s['pct_half']*100:.1f}%")


# ============================================================
# STEP 1: Build Ramanujan sum matrix
# ============================================================
print("\n" + "="*70)
print("STEP 1: RAMANUJAN SUM MATRIX c_q(n)")
print("="*70)

def ramanujan_sum(n, q):
    """Compute c_q(n) = sum_{d|gcd(n,q)} d * mu(q/d)."""
    g = gcd(n, q)
    result = 0
    for d in range(1, g+1):
        if g % d == 0:
            result += d * int(mobius(q // d))
    return result

print(f"Building {N}x{N} Ramanujan matrix...", flush=True)
t1 = time.time()

# R[n-1, q-1] = c_q(n)
R = np.zeros((N, N))
for n in range(1, N+1):
    for q in range(1, N+1):
        R[n-1, q-1] = ramanujan_sum(n, q)

print(f"  Built in {time.time()-t1:.1f}s")
print(f"  ||R||_F = {np.linalg.norm(R, 'fro'):.2f}")
print(f"  Rank = {np.linalg.matrix_rank(R)}")
print(f"  R[0,:5] = {R[0,:5]}  (c_q(1) = mu(q))")
print(f"  R[:,0] = all {np.unique(R[:,0])}  (c_1(n) = 1 for all n)")

# Verify: c_q(n) = sum_{d|gcd(n,q)} d*mu(q/d)
# Spot check: c_6(3) = sum_{d|gcd(3,6)=3} d*mu(6/d)
#   d=1: 1*mu(6)=1*1=1, d=3: 3*mu(2)=3*(-1)=-3 -> c_6(3) = -2
print(f"  Spot check: c_6(3) = {R[2,5]} (expected: -2)")


# ============================================================
# STEP 2: Build GCD operator
# ============================================================
print("\n" + "="*70)
print("STEP 2: GCD OPERATOR")
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
s_gcd = score_eigs(eigs_gcd)
print_score("Pure GCD", s_gcd)


# ============================================================
# STEP 3: Build explicit formula diagonal
# ============================================================
print("\n" + "="*70)
print("STEP 3: EXPLICIT FORMULA DIAGONAL")
print("="*70)

alpha = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k)
    dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes_all for m in range(1, 6)) / np.pi
    alpha[k-1] = Tw + s / dN

H_diag = np.diag(alpha)
eigs_diag = np.sort(np.linalg.eigvalsh(H_diag))
s_diag = score_eigs(eigs_diag)
print_score("Pure diagonal (explicit)", s_diag)


# ============================================================
# STEP 4: Ramanujan basis transformation Q
# ============================================================
print("\n" + "="*70)
print("STEP 4: RAMANUJAN BASIS TRANSFORMATION Q")
print("="*70)

# Method A: Direct QR factorization of R
# R = Q_A * upper_triangular
# Q_A is orthonormal and preserves the column ordering (q = 1,2,3,...)
Q_qr, _ = qr(R, mode='economic')
print(f"  Q from QR: {Q_qr.shape}, orthonormality check: "
      f"||Q^T Q - I||_F = {np.linalg.norm(Q_qr.T @ Q_qr - np.eye(N), 'fro'):.2e}")

# Method B: SVD of R -> orthonormal basis that captures most variance
U, S_vals, Vt = svd(R, full_matrices=False)
print(f"  SVD: top 5 singular values = {S_vals[:5].round(2)}")
print(f"  SVD: condition number = {S_vals[0]/S_vals[-1]:.2e}")
# U gives left singular vectors (indexed by n)
# V gives right singular vectors (indexed by q)
Q_svd = U  # orthonormal columns, ordered by importance

# Method C: Normalized Ramanujan (each column c_q normalized)
col_norms = np.linalg.norm(R, axis=0)
Q_norm = R / (col_norms + 1e-30)
print(f"  Normalized R: max col norm deviation from 1 = "
      f"{np.max(np.abs(np.linalg.norm(Q_norm, axis=0) - 1)):.2e}")

# Method D: Ramanujan-Fourier: weight columns by 1/phi(q)
# The "correct" Ramanujan expansion uses a_q = (1/phi(q)) sum_n f(n) c_q(n)
phi_vals = np.array([float(totient(q)) for q in range(1, N+1)])
R_weighted = R / phi_vals[np.newaxis, :]  # R_{nq} / phi(q)
Q_weighted_qr, _ = qr(R_weighted, mode='economic')
print(f"  Weighted R/phi(q) QR: done")


# ============================================================
# STEP 5: Transform GCD operator through each Q
# ============================================================
print("\n" + "="*70)
print("STEP 5: GCD IN RAMANUJAN BASIS — Q^T H_GCD Q")
print("="*70)

transforms = {
    'Q_QR (raw QR of R)': Q_qr,
    'Q_SVD (left singular vectors)': Q_svd,
    'Q_weighted (R/phi QR)': Q_weighted_qr,
}

for name, Q in transforms.items():
    H_transformed = Q.T @ H_gcd @ Q
    eigs_t = np.sort(np.linalg.eigvalsh(H_transformed))
    s_t = score_eigs(eigs_t)
    print_score(f"GCD in {name}", s_t)

    # Check: does the transformation change eigenvalues?
    # (It shouldn't if Q is orthonormal — similarity transform preserves spectrum)
    eig_diff = np.max(np.abs(np.sort(eigs_gcd) - np.sort(eigs_t)))
    print(f"  {'':>35}  eigenvalue change: {eig_diff:.2e}")


# ============================================================
# STEP 6: THE KEY TEST — mix GCD eigenvectors with explicit diagonal
# ============================================================
print("\n" + "="*70)
print("STEP 6: KEY TEST — RAMANUJAN EIGENVECTORS + EXPLICIT DIAGONAL")
print("="*70)
print("  If Q is right, then Q^T diag(alpha) Q should have both")
print("  correct eigenvalues AND high r")

# Get GCD eigenvectors
eigs_gcd_full, vecs_gcd = np.linalg.eigh(H_gcd)

# Test 1: Use GCD eigenvectors as Q
# H' = V_gcd^T diag(alpha) V_gcd
H_test1 = vecs_gcd.T @ H_diag @ vecs_gcd
eigs_test1 = np.sort(np.linalg.eigvalsh(H_test1))
s_test1 = score_eigs(eigs_test1)
print_score("V_gcd^T diag(alpha) V_gcd", s_test1)

# Test 2: Use Ramanujan Q
for name, Q in transforms.items():
    H_test = Q.T @ H_diag @ Q
    eigs_test = np.sort(np.linalg.eigvalsh(H_test))
    s_test = score_eigs(eigs_test)
    print_score(f"Q^T diag(alpha) Q [{name[:10]}]", s_test)


# ============================================================
# STEP 7: Ramanujan expansion of the von Mangoldt function
# ============================================================
print("\n" + "="*70)
print("STEP 7: RAMANUJAN EXPANSION OF von MANGOLDT")
print("="*70)
print("  Lambda(n) = -sum_q mu(q)/phi(q) * c_q(n)")
print("  This gives us the OPERATOR in Ramanujan space")

# Build the von Mangoldt diagonal in standard basis
from sympy import isprime, factorint

def vonmangoldt(n):
    """Lambda(n) = log(p) if n = p^k, else 0."""
    if n < 2: return 0.
    f = factorint(n)
    if len(f) == 1:
        p = list(f.keys())[0]
        return float(np.log(p))
    return 0.

Lambda_diag = np.array([vonmangoldt(n) for n in range(1, N+1)])

# Ramanujan coefficients: a_q = -mu(q)/phi(q) (Ramanujan's theorem!)
a_q_theory = np.array([-float(mobius(q))/float(totient(q)) if q > 0 else 0
                        for q in range(1, N+1)])

# Verify: reconstruct Lambda from Ramanujan expansion
Lambda_reconstructed = R @ a_q_theory
err_recon = np.max(np.abs(Lambda_diag - Lambda_reconstructed))
print(f"  ||Lambda - R @ a_q||_inf = {err_recon:.4f}")
print(f"  ||Lambda - R @ a_q||_2 / ||Lambda||_2 = "
      f"{np.linalg.norm(Lambda_diag - Lambda_reconstructed) / np.linalg.norm(Lambda_diag):.4f}")

# The a_q coefficients
print(f"\n  First 20 Ramanujan coefficients a_q = -mu(q)/phi(q):")
print(f"  q:  ", end="")
for q in range(1, 21):
    print(f"{q:>6}", end="")
print(f"\n  a_q:", end="")
for q in range(1, 21):
    print(f"{a_q_theory[q-1]:>+6.3f}", end="")
print()

# How many terms needed for good approximation?
for n_terms in [5, 10, 20, 50, 100, N]:
    a_trunc = np.zeros(N)
    a_trunc[:n_terms] = a_q_theory[:n_terms]
    Lambda_approx = R @ a_trunc
    err = np.linalg.norm(Lambda_diag - Lambda_approx) / np.linalg.norm(Lambda_diag)
    print(f"  {n_terms:>4} terms: relative error = {err:.4f}")


# ============================================================
# STEP 8: Build operator IN Ramanujan space
# ============================================================
print("\n" + "="*70)
print("STEP 8: OPERATOR IN RAMANUJAN SPACE")
print("="*70)
print("  In index space: H = diag(alpha_k) + eps * GCD")
print("  In Ramanujan space: H_R = R^{-1} H R (or Q^T H Q)")
print()
print("  KEY INSIGHT: In Ramanujan space, the GCD kernel DIAGONALIZES!")
print("  Because gcd(j,k) = sum_q (1/phi(q)) c_q(j) c_q(k)")
print("  So H_GCD in Ramanujan space is diagonal with entries ~ 1/phi(q)")

# The GCD kernel in terms of Ramanujan sums:
# sum_q c_q(j)*c_q(k)/phi(q) gives a function of gcd(j,k)
# More precisely: gcd(j,k) = sum_{q=1}^inf c_q(j)*c_q(k)/phi(q)^2 * phi(q)
# Let's verify this numerically

print("\n  Verifying GCD = R diag(1/phi) R^T / N ...")
# The exact identity (Cohen, 1949):
#   sum_{q=1}^inf c_q(m)*c_q(n) / phi(q) = 0 if m != n (in some average sense)
# and the Ramanujan expansion of f(gcd(m,n)) involves sum a_q c_q(m) c_q(n)

# For f(g) = log(g+1)/sqrt(mn):
# We have f(gcd(m,n)) = sum_q b_q * c_q(m) * c_q(n) / (phi(q) * sqrt(mn))
# where b_q are determined by the Ramanujan expansion of f

# Let's just compute R^T H_GCD R and see if it's diagonal-ish
D_gcd_ramanujan = Q_qr.T @ H_gcd @ Q_qr
print(f"  ||Q^T H_GCD Q||_F = {np.linalg.norm(D_gcd_ramanujan, 'fro'):.4f}")
print(f"  ||off-diag(Q^T H_GCD Q)||_F = "
      f"{np.linalg.norm(D_gcd_ramanujan - np.diag(np.diag(D_gcd_ramanujan)), 'fro'):.4f}")

diag_ratio = (np.linalg.norm(np.diag(np.diag(D_gcd_ramanujan)), 'fro') /
              np.linalg.norm(D_gcd_ramanujan, 'fro'))
print(f"  Diag energy fraction: {diag_ratio:.4f}")

if diag_ratio > 0.9:
    print("  -> GCD is NEARLY DIAGONAL in Ramanujan basis!")
elif diag_ratio > 0.7:
    print("  -> GCD is substantially diagonal in Ramanujan basis")
else:
    print("  -> GCD is NOT diagonal in this Ramanujan basis")


# ============================================================
# STEP 9: The Ramanujan operator — diagonal GCD + explicit coupling
# ============================================================
print("\n" + "="*70)
print("STEP 9: THE RAMANUJAN OPERATOR")
print("="*70)
print("  In Ramanujan space:")
print("    Diagonal: d_q = eigenvalue of GCD in Ramanujan basis")
print("    Off-diagonal: explicit formula coupling in Ramanujan space")
print()
print("  FLIP THE PERSPECTIVE:")
print("    Old: diag(explicit) + GCD_offdiag  ->  right eigs, no r")
print("    New: diag(GCD_ramanujan) + explicit_offdiag_ramanujan  ->  r + eigs?")

# The GCD eigenvalues in Ramanujan basis (diagonal of Q^T H_GCD Q)
gcd_diag_R = np.diag(D_gcd_ramanujan)

# Scale to match zero range
gcd_diag_scaled = gcd_diag_R - np.min(gcd_diag_R)
if np.max(gcd_diag_scaled) > 0:
    gcd_diag_scaled = gcd_diag_scaled / np.max(gcd_diag_scaled) * (zeta_zeros[-1] - zeta_zeros[0]) + zeta_zeros[0]

# The explicit formula in Ramanujan basis
D_alpha_R = Q_qr.T @ H_diag @ Q_qr
alpha_offdiag_R = D_alpha_R - np.diag(np.diag(D_alpha_R))

print(f"  ||explicit_offdiag in R-space||_F = {np.linalg.norm(alpha_offdiag_R, 'fro'):.4f}")
print(f"  ||explicit_diag in R-space||_F = {np.linalg.norm(np.diag(D_alpha_R)):.4f}")

# Build: diag(gcd_scaled) + eps * alpha_offdiag_R
print(f"\n  {'eps':>8} {'mean_err':>10} {'r':>8} {'p(GUE)':>8} {'<half':>8}")
print(f"  {'-'*48}")

best_r = -1
best_eps = 0
for eps in [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    H_R = np.diag(gcd_diag_scaled) + eps * alpha_offdiag_R
    H_R = (H_R + H_R.T) / 2  # ensure symmetric
    eigs = np.sort(np.linalg.eigvalsh(H_R))
    s = score_eigs(eigs)
    marker = ""
    if s['r'] > best_r:
        best_r = s['r']; best_eps = eps; marker = " <--"
    print(f"  {eps:>8.3f} {s['mean_err']:>10.4f} {s['r']:>+8.4f} "
          f"{s['p_gue']:>8.4f} {s['pct_half']*100:>7.1f}%{marker}")


# ============================================================
# STEP 10: Direct Ramanujan-weighted operator
# ============================================================
print("\n" + "="*70)
print("STEP 10: DIRECT RAMANUJAN-WEIGHTED OPERATOR")
print("="*70)
print("  H_{jk} = sum_q a_q * c_q(j) * c_q(k) / (phi(q) * sqrt(j*k))")
print("  where a_q encodes BOTH GCD structure AND explicit formula")

# The key: use a_q that encodes the explicit formula in Ramanujan language
# The diagonal (explicit formula) in standard basis becomes:
#   alpha_k = sum_q a_q^{(diag)} * c_q(k)
# where a_q^{(diag)} can be computed by Ramanujan inverse transform

# Ramanujan inverse: a_q = (1/N) sum_n f(n) * c_q(n) / phi(q)
# (This is the finite Ramanujan-Fourier transform)
a_q_explicit = np.zeros(N)
for q in range(1, N+1):
    phi_q = float(totient(q))
    a_q_explicit[q-1] = np.sum(alpha * R[:, q-1]) / (N * phi_q)

print(f"  Top Ramanujan coefficients of explicit formula diagonal:")
top_q = np.argsort(np.abs(a_q_explicit))[-15:][::-1]
for q_idx in top_q:
    q = q_idx + 1
    print(f"    q={q:>4}: a_q = {a_q_explicit[q_idx]:>+10.4f} "
          f"(|a_q|*phi(q) = {abs(a_q_explicit[q_idx]) * float(totient(q)):>8.2f})")

# Reconstruct alpha from Ramanujan expansion
alpha_recon = R @ a_q_explicit
err_alpha = np.linalg.norm(alpha - alpha_recon) / np.linalg.norm(alpha)
print(f"\n  Alpha reconstruction error: {err_alpha:.4f}")

# Now build the COMBINED operator in Ramanujan space:
# Diagonal part: from GCD eigenvalues in Ramanujan basis
# Plus: explicit formula structure encoded in a_q coefficients
# The full operator: H_q = diag-from-GCD + coupling-from-explicit

# Actually the cleanest approach: the Ramanujan coefficients tell us
# what the operator SHOULD look like in q-space
# H_{q,q'} = delta_{qq'} * (GCD eigenvalue_q scaled to zero range)
#           + coupling * (explicit formula matrix element in q-basis)


# ============================================================
# STEP 11: The MULTIPLICATIVE Fourier transform
# ============================================================
print("\n" + "="*70)
print("STEP 11: MULTIPLICATIVE FOURIER TRANSFORM")
print("="*70)
print("  The 'correct' Q might not be Ramanujan sums directly,")
print("  but the MULTIPLICATIVE characters: chi(n) = n^{-it}")
print("  This is the Mellin transform basis.")
print("  Q_{n,k} = n^{-i*t_k} / sqrt(N) where t_k = Weyl zeros")

# Build Mellin basis: columns are n^{-it_k} for Weyl approximations
t_weyl = np.array([weyl_zero(k) for k in range(1, N+1)])
n_vals = np.arange(1, N+1, dtype=float)

# Q_Mellin[n-1, k-1] = n^{-i*t_k} / sqrt(N)
Q_mellin = np.zeros((N, N), dtype=complex)
for k in range(N):
    Q_mellin[:, k] = n_vals ** (-1j * t_weyl[k]) / np.sqrt(N)

# This is unitary if t_k are chosen right (they're not exactly, but close)
print(f"  ||Q_M^H Q_M - I||_F = {np.linalg.norm(Q_mellin.conj().T @ Q_mellin - np.eye(N), 'fro'):.4f}")

# Transform GCD through Mellin
D_gcd_mellin = Q_mellin.conj().T @ H_gcd @ Q_mellin
diag_frac_mellin = (np.linalg.norm(np.diag(np.diag(D_gcd_mellin)), 'fro') /
                    np.linalg.norm(D_gcd_mellin, 'fro'))
print(f"  GCD diagonal fraction in Mellin basis: {diag_frac_mellin:.4f}")

# Transform explicit diagonal through Mellin
D_alpha_mellin = Q_mellin.conj().T @ H_diag @ Q_mellin
diag_frac_alpha = (np.linalg.norm(np.diag(np.diag(D_alpha_mellin)), 'fro') /
                    np.linalg.norm(D_alpha_mellin, 'fro'))
print(f"  Explicit diag fraction in Mellin basis: {diag_frac_alpha:.4f}")

# The eigenvalues of the Mellin-transformed GCD
eigs_gcd_mellin = np.sort(np.real(np.linalg.eigvals(D_gcd_mellin)))
s_mellin = score_eigs(eigs_gcd_mellin)
print_score("GCD in Mellin basis", s_mellin)


# ============================================================
# STEP 12: Hybrid in Ramanujan basis with von Mangoldt diagonal
# ============================================================
print("\n" + "="*70)
print("STEP 12: von MANGOLDT DIAGONAL + GCD OFF-DIAGONAL")
print("="*70)
print("  Lambda(n) is the 'natural' diagonal for the GCD kernel")
print("  because both live in the same multiplicative world")

# Scale Lambda to zero range
Lambda_scaled = Lambda_diag.copy()
# Use cumulative sum / normalization to map to zero heights
cumsum_L = np.cumsum(Lambda_diag)
if cumsum_L[-1] > 0:
    Lambda_scaled = cumsum_L / cumsum_L[-1] * (zeta_zeros[-1] - zeta_zeros[0]) + zeta_zeros[0]

# Build hybrid: diag(Lambda_scaled) + eps * GCD_offdiag
GCD_offdiag = H_gcd - np.diag(np.diag(H_gcd))

print(f"\n  {'eps':>8} {'mean_err':>10} {'r':>8} {'p(GUE)':>8} {'<half':>8}")
print(f"  {'-'*48}")

best_r_vm = -1
for eps in [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]:
    H_vm = np.diag(Lambda_scaled) + eps * GCD_offdiag
    eigs = np.sort(np.linalg.eigvalsh(H_vm))
    s = score_eigs(eigs)
    marker = ""
    if s['r'] > best_r_vm:
        best_r_vm = s['r']; marker = " <--"
    print(f"  {eps:>8.1f} {s['mean_err']:>10.4f} {s['r']:>+8.4f} "
          f"{s['p_gue']:>8.4f} {s['pct_half']*100:>7.1f}%{marker}")


# ============================================================
# STEP 13: The Ramanujan-Dirichlet operator
# ============================================================
print("\n" + "="*70)
print("STEP 13: RAMANUJAN-DIRICHLET OPERATOR")
print("="*70)
print("  H_{jk} = sum_{q|gcd(j,k)} mu(j/q)*mu(k/q) * alpha_q")
print("  where alpha_q comes from the explicit formula in q-space")
print("  This builds the operator DIRECTLY from Ramanujan arithmetic")

# The idea: instead of GCD + explicit as separate pieces,
# build a SINGLE operator whose matrix elements are determined by
# the Ramanujan structure of both the zeros and the primes.

# For each divisor q of gcd(j,k), the Ramanujan sum c_q contributes.
# Weight each contribution by how much that q-mode matters for the zeros.

# Effective operator: H_{jk} = sum_q w_q * c_q(j) * c_q(k) / (phi(q)^2)
# where w_q are chosen so eigenvalues match zeros

# In matrix form: H = R @ diag(w / phi^2) @ R^T
# Eigenvalues of H = eigenvalues of diag(w/phi^2) * R^T R
# Since R is NOT orthogonal, R^T R != I

# Gram matrix
G = R.T @ R  # G_{qq'} = sum_n c_q(n) * c_{q'}(n)
print(f"  Gram matrix G = R^T R: shape {G.shape}")
print(f"  G diagonal (first 10): {np.diag(G)[:10].round(1)}")

# The known identity: sum_{n=1}^N c_q(n) c_{q'}(n) ~ N * phi(q) * delta_{qq'}
# (for q, q' <= N, approximately)
gram_diag = np.diag(G)
gram_predicted = N * phi_vals
gram_ratio = gram_diag / gram_predicted
print(f"  G_qq/[N*phi(q)] ratio (first 10): {gram_ratio[:10].round(3)}")
print(f"  Mean ratio: {np.mean(gram_ratio):.4f} (should be ~1.0)")

# So if G ~ N * diag(phi), then
# H = R diag(w/phi^2) R^T has eigenvalues ~ N * phi(q) * w_q / phi(q)^2 = N * w_q / phi(q)
# We want these to be the zeta zeros!
# So: w_q = phi(q) * gamma_q / N where gamma_q = q-th zero?

# But Q and eigenvalue ordering are nontrivial...
# Let's try: choose w_q so that the eigenvalues of H match zeros

# Simple approach: use the Ramanujan-orthogonality to set
# H = (1/N) * R @ diag(gamma) @ R^T where gamma are zero targets
# Then eigenvalues of H are eigenvalues of (1/N) * diag(gamma) * G

# Compute eigenvalues of (1/N) diag(zeros) @ G
# This is NOT symmetric, but (1/N) diag(sqrt(gamma)) G diag(sqrt(gamma)) is
gamma = np.zeros(N)
gamma[:] = zeta_zeros[:N]  # target eigenvalues

# Symmetrize: H_sym = (1/N) R diag(gamma/phi) R^T
w_q = gamma / phi_vals
H_ram_dir = (1.0/N) * R @ np.diag(w_q) @ R.T
H_ram_dir = (H_ram_dir + H_ram_dir.T) / 2

eigs_rd = np.sort(np.linalg.eigvalsh(H_ram_dir))
s_rd = score_eigs(eigs_rd)
print_score("Ramanujan-Dirichlet (gamma/phi)", s_rd)

# Try with phi^2
w_q2 = gamma / phi_vals**2
H_ram_dir2 = R @ np.diag(w_q2) @ R.T
H_ram_dir2 = (H_ram_dir2 + H_ram_dir2.T) / 2
eigs_rd2 = np.sort(np.linalg.eigvalsh(H_ram_dir2))
s_rd2 = score_eigs(eigs_rd2)
print_score("Ramanujan-Dirichlet (gamma/phi^2)", s_rd2)


# ============================================================
# STEP 14: PROCRUSTES — find the BEST Q
# ============================================================
print("\n" + "="*70)
print("STEP 14: PROCRUSTES — OPTIMAL ORTHOGONAL Q")
print("="*70)
print("  Find Q that minimizes ||Q @ diag(zeros) @ Q^T - H_GCD||_F")
print("  This is the orthogonal Procrustes problem.")

# We want: Q diag(zeros) Q^T ~ H_GCD
# Equivalently: H_GCD's eigenvectors should map to diag(zeros) eigenvectors
# via the basis change Q.

# H_GCD = V_gcd @ diag(eigs_gcd) @ V_gcd^T
# diag(zeros) = I @ diag(zeros) @ I
# So Q ~ V_gcd @ P where P is a permutation/sign matrix

# The Procrustes Q is V_gcd itself (with columns reordered to match zeros)
eigs_gcd_sorted, vecs_gcd_sorted = np.linalg.eigh(H_gcd)

# Match GCD eigenvalues to zeros: find the assignment that minimizes total error
from scipy.optimize import linear_sum_assignment

# Cost matrix: |eig_gcd_i - zero_j|
cost = np.abs(eigs_gcd_sorted[:, np.newaxis] - zeta_zeros[np.newaxis, :N])
row_ind, col_ind = linear_sum_assignment(cost)

# The Procrustes Q reorders GCD eigenvectors to match zeros
Q_proc = vecs_gcd_sorted[:, col_ind]  # reordered columns

# Build the operator in Procrustes basis
H_proc = Q_proc.T @ np.diag(zeta_zeros[:N]) @ Q_proc
H_proc = (H_proc + H_proc.T) / 2
eigs_proc = np.sort(np.linalg.eigvalsh(H_proc))
s_proc = score_eigs(eigs_proc)
print_score("Procrustes (Q_proc^T diag(zeros) Q_proc)", s_proc)

# How close is this to GCD?
proc_err = np.linalg.norm(H_proc - H_gcd, 'fro') / np.linalg.norm(H_gcd, 'fro')
print(f"  ||H_proc - H_GCD||_F / ||H_GCD||_F = {proc_err:.4f}")

# The Procrustes operator is the BEST we can do with orthogonal Q
# Its r tells us the upper bound on what basis transformation can achieve
print(f"\n  PROCRUSTES gives the THEORETICAL UPPER BOUND on what Q can do:")
print(f"  r = {s_proc['r']:+.4f}, p(GUE) = {s_proc['p_gue']:.4f}")

# Now the question: is there a STRUCTURED Q (Ramanujan, Mellin, etc.)
# that comes close to Procrustes Q?

# Compare Q_proc to Ramanujan Q
for name, Q in transforms.items():
    # Measure alignment: how well does Q approximate Q_proc?
    # Use Frobenius inner product of the projectors
    alignment = np.linalg.norm(Q_proc.T @ Q, 'fro') / np.sqrt(N)
    print(f"  ||Q_proc^T @ {name[:15]:>15}|| / sqrt(N) = {alignment:.4f}")

# Compare to identity (trivial Q)
alignment_id = np.linalg.norm(Q_proc.T @ np.eye(N), 'fro') / np.sqrt(N)
print(f"  ||Q_proc^T @ {'Identity':>15}|| / sqrt(N) = {alignment_id:.4f}")


# ============================================================
# STEP 15: Analyze Q_proc structure
# ============================================================
print("\n" + "="*70)
print("STEP 15: STRUCTURE OF THE OPTIMAL Q")
print("="*70)
print("  What does Q_proc look like? Is it close to any known transform?")

# Check if Q_proc columns are multiplicative
print("\n  Multiplicativity test on Q_proc columns (first 5):")
for col in range(5):
    v = Q_proc[:, col]
    v_norm = v / (v[0] + 1e-10) if abs(v[0]) > 1e-10 else v / np.max(np.abs(v))
    tests = [(2,3,6), (2,5,10), (3,5,15), (2,7,14), (3,7,21)]
    errs = []
    for a, b, ab in tests:
        if ab <= N:
            pred = v_norm[a-1] * v_norm[b-1]
            actual = v_norm[ab-1]
            if abs(pred) > 1e-10:
                errs.append(abs(actual - pred) / abs(pred))
    mean_e = np.mean(errs) if errs else float('inf')
    print(f"  Col {col}: mult_error = {mean_e:.4f} "
          f"({'MULTIPLICATIVE' if mean_e < 0.1 else 'not multiplicative'})")

# Check if Q_proc has Ramanujan-like structure
# Each column of Q_proc should correlate with some c_q
print("\n  Best Ramanujan correlation for each Q_proc column (first 10):")
for col in range(10):
    corrs = np.abs(R.T @ Q_proc[:, col])
    best_q = np.argmax(corrs) + 1
    best_corr = corrs[best_q - 1] / (np.linalg.norm(R[:, best_q-1]) * np.linalg.norm(Q_proc[:, col]) + 1e-30)
    print(f"  Col {col}: best q={best_q}, corr={best_corr:.4f}")

# IPR of Q_proc columns (localization)
ipr_proc = np.sum(Q_proc**4, axis=0)
print(f"\n  Q_proc IPR: mean={np.mean(ipr_proc):.6f}, "
      f"min={np.min(ipr_proc):.6f}, max={np.max(ipr_proc):.6f}")
print(f"  Delocalized (GUE) IPR: {3.0/N:.6f}")
print(f"  Localized (delta) IPR: 1.0")
print(f"  Mean IPR / GUE = {np.mean(ipr_proc) / (3.0/N):.2f}x")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70)
print("VERDICT: RAMANUJAN Q HUNT")
print("="*70)

print(f"""
  BASELINES:
    Pure GCD:             r={s_gcd['r']:+.4f}, err={s_gcd['mean_err']:.4f}
    Pure explicit diag:   r={s_diag['r']:+.4f}, err={s_diag['mean_err']:.4f}

  RAMANUJAN TRANSFORMS:
    Q_QR^T diag(alpha) Q_QR:    tested above
    Q_SVD^T diag(alpha) Q_SVD:  tested above

  THEORETICAL UPPER BOUND:
    Procrustes Q:         r={s_proc['r']:+.4f}, err={s_proc['mean_err']:.4f}

  The Procrustes Q tells us what r is achievable in principle.
  The gap between Procrustes and Ramanujan Q tells us how far
  the arithmetic structure is from the optimal basis.
""")

print(f"Total time: {time.time()-t0:.1f}s")
