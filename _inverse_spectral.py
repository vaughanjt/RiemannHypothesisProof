"""Inverse spectral reconstruction of the zeta operator.

Given: the first N zeta zeros {gamma_1, ..., gamma_N}
Constraint: eigenvectors must be Riemann-Siegel-type vectors
Reconstruct: H = U * diag(gamma) * U^{dagger}

If U is chosen to have RS structure:
  U_{jk} = (1/sqrt(N)) * a_k * exp(i * phi_k(gamma_j))
where a_k = k^{-1/2} and phi_k(t) = theta(t) - t*log(k),

then H = U * D * U^{dagger} is a Hermitian matrix whose:
  - Eigenvalues are the zeta zeros (by construction)
  - Eigenvectors have RS structure (by construction)

The KEY QUESTION: does the resulting H have RECOGNIZABLE STRUCTURE?
If H turns out to be sparse, Toeplitz, or have a pattern related to
known arithmetic operators, that's the breakthrough.
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
import mpmath

t0 = time.time()
mpmath.mp.dps = 20

# ============================================================
# STEP 1: Load zeta zeros
# ============================================================
print('Loading zeta zeros...')
N = 50  # start small to see structure
zeros_gamma = np.zeros(N)
for i in range(N):
    zeros_gamma[i] = float(mpmath.zetazero(i + 1).imag)

print(f'  {N} zeros: gamma_1={zeros_gamma[0]:.4f} to gamma_{N}={zeros_gamma[-1]:.4f}')

# ============================================================
# STEP 2: Build the RS eigenvector matrix U
# ============================================================
print('\nBuilding RS eigenvector matrix...')

# U_{jk} = (1/C_j) * k^{-1/2} * exp(i*(theta(gamma_j) - gamma_j*log(k)))
# where C_j normalizes each row to unit length.
# j indexes eigenvalues (zeros), k indexes the "integer basis" (1..N)

def build_rs_eigenvectors(gammas, N_basis):
    """Build the RS-type eigenvector matrix."""
    N_eig = len(gammas)
    U = np.zeros((N_eig, N_basis), dtype=complex)

    for j in range(N_eig):
        gamma = gammas[j]
        theta = float(mpmath.siegeltheta(gamma))
        for k in range(1, N_basis + 1):
            phase = theta - gamma * np.log(k)
            U[j, k - 1] = np.exp(1j * phase) / np.sqrt(k)

        # Normalize
        norm = np.linalg.norm(U[j, :])
        if norm > 1e-10:
            U[j, :] /= norm

    return U

U_raw = build_rs_eigenvectors(zeros_gamma, N)
print(f'  U_raw is {U_raw.shape[0]} x {U_raw.shape[1]}, cond: {np.linalg.cond(U_raw):.2e}')

# The RS vectors are NOT orthogonal. Orthogonalize via QR decomposition
# while tracking how much structure is preserved.
Q, R = np.linalg.qr(U_raw.conj().T)  # QR of N_basis x N_eig
U = Q.conj().T  # N_eig x N_basis, now orthonormal rows

print(f'  After QR orthogonalization:')
UUd = U @ U.conj().T
off_diag = np.abs(UUd - np.eye(N))
print(f'  U*U^dagger off-diagonal max: {np.max(off_diag):.6f}')

# How much RS structure survives? Correlate |U_orth[j,k]| with |U_raw[j,k]|
from scipy.stats import pearsonr as pr
r_struct, _ = pr(np.abs(U_raw).flatten(), np.abs(U).flatten())
print(f'  RS structure preserved (|U_orth| vs |U_raw|): r = {r_struct:.4f}')

# ============================================================
# STEP 3: Reconstruct H = U^dagger * diag(gamma) * U
# ============================================================
print('\nReconstructing H = U^dagger * diag(gamma) * U...')

D = np.diag(zeros_gamma)
H = U.conj().T @ D @ U  # N_basis x N_basis Hermitian matrix

# Force exact Hermiticity
H = (H + H.conj().T) / 2

# Verify eigenvalues match
eigs_check = np.sort(np.linalg.eigvalsh(H))
zeros_sorted = np.sort(zeros_gamma)
max_err = np.max(np.abs(eigs_check - zeros_sorted))
print(f'  Eigenvalue reconstruction error: {max_err:.2e}')

# ============================================================
# STEP 4: Analyze the structure of H
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: STRUCTURE OF THE RECONSTRUCTED OPERATOR H')
print('=' * 70)

H_real = np.real(H)
H_imag = np.imag(H)

print(f'\n  H is {N}x{N} Hermitian')
print(f'  Max |Re(H)|: {np.max(np.abs(H_real)):.4f}')
print(f'  Max |Im(H)|: {np.max(np.abs(H_imag)):.4f}')
print(f'  Trace: {np.trace(H).real:.4f} (sum of zeros: {np.sum(zeros_gamma):.4f})')
print(f'  Frobenius norm: {np.linalg.norm(H, "fro"):.4f}')

# Is H sparse?
threshold = 0.01 * np.max(np.abs(H))
n_nonzero = np.sum(np.abs(H) > threshold)
print(f'  Non-zero entries (>1% of max): {n_nonzero}/{N*N} ({100*n_nonzero/(N*N):.1f}%)')

# Is H diagonally dominant?
diag_dom = np.array([np.abs(H[i, i]) / np.sum(np.abs(H[i, :])) for i in range(N)])
print(f'  Diagonal dominance: mean={np.mean(diag_dom):.4f}, min={np.min(diag_dom):.4f}')

# Is H Toeplitz-like? Check if H_{j,k} depends mainly on |j-k|
print(f'\n  Toeplitz test: mean |H_jk| by diagonal offset:')
for d in range(min(10, N)):
    vals = [np.abs(H[i, i + d]) for i in range(N - d)]
    if vals:
        print(f'    |j-k|={d}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, '
              f'CV={np.std(vals)/(np.mean(vals)+1e-10):.3f}')

# Is H related to a known arithmetic matrix?
# Check: does H_{jk} correlate with log(gcd(j,k))/sqrt(j*k)?
gcd_vals = []
h_vals = []
for j in range(N):
    for k in range(N):
        if j != k:
            g = np.gcd(j + 1, k + 1)
            gcd_vals.append(np.log(g + 1) / np.sqrt((j + 1) * (k + 1)))
            h_vals.append(np.abs(H[j, k]))

from scipy.stats import pearsonr, spearmanr
r_gcd, p_gcd = pearsonr(gcd_vals, h_vals)
rs_gcd, ps_gcd = spearmanr(gcd_vals, h_vals)
print(f'\n  Correlation |H_jk| vs log(gcd+1)/sqrt(jk):')
print(f'    Pearson  r = {r_gcd:+.4f} (p = {p_gcd:.2e})')
print(f'    Spearman r = {rs_gcd:+.4f} (p = {ps_gcd:.2e})')

# Check against divisibility indicator
div_vals = []
for j in range(N):
    for k in range(N):
        if j != k:
            div_vals.append(1.0 if (j + 1) % (k + 1) == 0 or (k + 1) % (j + 1) == 0 else 0.0)

r_div, p_div = pearsonr(div_vals, h_vals)
print(f'  Correlation |H_jk| vs divisibility indicator:')
print(f'    Pearson  r = {r_div:+.4f} (p = {p_div:.2e})')

# Check against 1/sqrt(j*k) (the natural arithmetic weight)
inv_sqrt_vals = [1.0 / np.sqrt((j + 1) * (k + 1)) for j in range(N) for k in range(N) if j != k]
r_inv, p_inv = pearsonr(inv_sqrt_vals, h_vals)
print(f'  Correlation |H_jk| vs 1/sqrt(jk):')
print(f'    Pearson  r = {r_inv:+.4f} (p = {p_inv:.2e})')

# ============================================================
# STEP 5: The diagonal structure
# ============================================================
print('\n' + '=' * 70)
print('STEP 5: DIAGONAL OF H')
print('=' * 70)

diag_H = np.real(np.diag(H))
print(f'\n  H_{{kk}} for k = 1..20:')
print(f'  {"k":>5} {"H_kk":>12} {"log(k)":>10} {"H_kk/log(k)":>12}')
for k in range(min(20, N)):
    logk = np.log(k + 1)
    ratio = diag_H[k] / logk if logk > 0 else 0
    print(f'  {k+1:>5} {diag_H[k]:>+12.4f} {logk:>10.4f} {ratio:>12.4f}')

# Does H_{kk} ~ c * log(k)?
from scipy.stats import pearsonr
log_k = np.log(np.arange(1, N + 1))
r_diag, p_diag = pearsonr(log_k, diag_H)
print(f'\n  Correlation H_{{kk}} vs log(k): r = {r_diag:+.4f} (p = {p_diag:.2e})')

# ============================================================
# STEP 6: Scaling — does the structure persist at larger N?
# ============================================================
print('\n' + '=' * 70)
print('STEP 6: SCALING CHECK')
print('=' * 70)

for N_test in [20, 50, 100]:
    gammas = np.array([float(mpmath.zetazero(i + 1).imag) for i in range(N_test)])
    U_t = build_rs_eigenvectors(gammas, N_test)
    D_t = np.diag(gammas)
    H_t = U_t.conj().T @ D_t @ U_t
    H_t = (H_t + H_t.conj().T) / 2

    # Check eigenvalue reconstruction
    eigs_t = np.sort(np.linalg.eigvalsh(H_t))
    err_t = np.max(np.abs(eigs_t - np.sort(gammas)))

    # Sparsity
    thresh_t = 0.01 * np.max(np.abs(H_t))
    sparse_t = np.sum(np.abs(H_t) > thresh_t) / (N_test ** 2)

    # Diagonal correlation with log(k)
    diag_t = np.real(np.diag(H_t))
    log_k_t = np.log(np.arange(1, N_test + 1))
    r_d_t, _ = pearsonr(log_k_t, diag_t)

    # GCD correlation
    gcd_v, h_v = [], []
    for j in range(N_test):
        for k in range(N_test):
            if j != k:
                gcd_v.append(np.log(np.gcd(j+1, k+1) + 1) / np.sqrt((j+1)*(k+1)))
                h_v.append(np.abs(H_t[j, k]))
    r_g_t, _ = pearsonr(gcd_v, h_v)

    print(f'  N={N_test:>3}: err={err_t:.1e}, dense={sparse_t*100:.0f}%, '
          f'diag~logk r={r_d_t:+.3f}, gcd r={r_g_t:+.3f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: STRUCTURE OF THE ZETA OPERATOR')
print('=' * 70)

print(f'\n  The operator H reconstructed from zeta zeros + RS eigenvectors:')
if abs(r_gcd) > 0.3:
    print(f'  - CORRELATES with gcd structure (r = {r_gcd:+.3f})')
if abs(r_diag) > 0.3:
    print(f'  - Diagonal ~ log(k) (r = {r_diag:+.3f})')
if abs(r_div) > 0.1:
    print(f'  - Correlates with divisibility (r = {r_div:+.3f})')

n_dense = np.sum(np.abs(H) > threshold) / (N * N)
if n_dense > 0.5:
    print(f'  - Is DENSE ({n_dense*100:.0f}% non-zero) — not a sparse arithmetic matrix')
else:
    print(f'  - Has sparsity pattern ({(1-n_dense)*100:.0f}% zero)')

print(f'\n  This is the operator whose eigenvalues are zeta zeros and whose')
print(f'  eigenvectors have Riemann-Siegel structure. Its matrix elements')
print(f'  in the integer basis reveal how arithmetic encodes the spectrum.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
