"""Berry-Keating xp operator v2: log-grid discretization with boundary conditions.

KEY INSIGHT: On a logarithmic grid u = log(x), the operator xp = -i d/du
is just the momentum operator on a UNIFORM grid. Different boundary conditions
select different spectra:
  - Periodic BC:  eigenvalues = 2*pi*n/(N*h)  (harmonic oscillator levels)
  - Dirichlet BC: eigenvalues = pi*n/(N*h)
  - FUNCTIONAL EQUATION BC: eigenvalues = zeta zeros (the dream)

The functional equation xi(s) = xi(1-s) constrains the wave functions.
In the Mellin transform picture:
  F(s) = integral f(x) x^{s-1} dx = integral g(u) e^{su} du  (u = log x)

The constraint F(1/2+it) = F(1/2-it) means g(u) must be "self-dual"
under the involution that maps s -> 1-s.

APPROACH 1: Log-grid xp with various BCs
APPROACH 2: Sierra-Townsend via regularized 1/x
APPROACH 3: The "xi operator" — build directly from xi(s)
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigvals, eigh
import mpmath

t0 = time.time()
mpmath.mp.dps = 20

known_zeros_t = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
                  37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703]


# ============================================================
# APPROACH 1: xp on log-grid
# ============================================================

def build_xp_loggrid(N, L=None):
    """Build xp = -i d/du on uniform grid u_j = j*h, h = L/N.

    In log-coordinates u = log(x), xp = -i d/du (exactly).
    Central differences for d/du.
    Returns the matrix with PERIODIC boundary conditions.
    """
    if L is None:
        L = 2 * np.pi * N / 50  # Scale so first ~50 eigenvalues are in range

    h = L / N
    # Central difference: (d/du f)_j ≈ (f_{j+1} - f_{j-1}) / (2h)
    D = np.zeros((N, N))
    for j in range(N):
        D[j, (j + 1) % N] = 1 / (2 * h)
        D[j, (j - 1) % N] = -1 / (2 * h)

    H = -1j * D  # xp = -i d/du
    return H, h, L


def build_xp_dirichlet(N, L=None):
    """xp on log-grid with Dirichlet BC (f(0) = f(L) = 0)."""
    if L is None:
        L = 2 * np.pi * N / 50

    h = L / N
    D = np.zeros((N, N))
    for j in range(N):
        if j + 1 < N:
            D[j, j + 1] = 1 / (2 * h)
        if j - 1 >= 0:
            D[j, j - 1] = -1 / (2 * h)

    H = -1j * D
    return H, h, L


# ============================================================
# APPROACH 2: Berry-Keating with ABSORBING boundary
# ============================================================

def build_xp_absorbing(N, L=None, absorption=0.1):
    """xp with absorbing boundaries that mimic the functional equation.

    Add a complex potential at the boundaries:
    V(u) = -i * absorption * [delta(u~0) + delta(u~L)]

    This creates a non-Hermitian operator whose eigenvalues
    may acquire imaginary parts related to the zero width.
    """
    if L is None:
        L = 2 * np.pi * N / 50

    h = L / N
    D = np.zeros((N, N), dtype=complex)
    for j in range(N):
        if j + 1 < N:
            D[j, j + 1] = 1 / (2 * h)
        if j - 1 >= 0:
            D[j, j - 1] = -1 / (2 * h)

    H = -1j * D

    # Add absorbing potential near boundaries
    for j in range(N):
        u = j * h
        # Smooth absorption near u=0 and u=L
        V_abs = absorption * (np.exp(-u / h) + np.exp(-(L - u) / h))
        H[j, j] -= 1j * V_abs

    return H, h, L


# ============================================================
# APPROACH 3: The xi operator (direct)
# ============================================================

def build_xi_operator(N, T_center=30, T_span=80):
    """Build an operator whose characteristic polynomial approximates xi(s).

    Strategy: evaluate xi(1/2 + it) at N equally spaced points in [T-span/2, T+span/2].
    Build the COMPANION MATRIX of the interpolating polynomial.
    The eigenvalues of the companion matrix are the roots of the polynomial,
    which approximate the zeros of xi in the window.
    """
    t_points = np.linspace(T_center - T_span / 2, T_center + T_span / 2, N)

    # Evaluate xi(1/2 + it) = Z(t) * exp(-i*theta(t)) * ...
    # Actually, use the real-valued Z(t) = exp(i*theta(t)) * zeta(1/2+it)
    Z_vals = np.array([float(mpmath.siegelz(t)) for t in t_points])

    # Build the polynomial that interpolates Z at these points
    # Using Chebyshev interpolation for numerical stability
    from numpy.polynomial import chebyshev

    # Map t_points to [-1, 1]
    t_min, t_max = t_points[0], t_points[-1]
    t_mapped = 2 * (t_points - t_min) / (t_max - t_min) - 1

    # Fit Chebyshev polynomial
    coeffs = chebyshev.chebfit(t_mapped, Z_vals, N - 1)

    # Convert to standard polynomial for companion matrix
    poly_coeffs = chebyshev.cheb2poly(coeffs)

    # Companion matrix: eigenvalues are roots of the polynomial
    # (in mapped coordinates)
    if abs(poly_coeffs[-1]) > 1e-30:
        companion = np.zeros((N - 1, N - 1))
        for i in range(N - 2):
            companion[i + 1, i] = 1.0
        for i in range(N - 1):
            companion[i, N - 2] = -poly_coeffs[i] / poly_coeffs[-1]
        return companion, t_min, t_max
    else:
        return None, t_min, t_max


# ============================================================
# APPROACH 4: Functional equation projection
# ============================================================

def build_xp_functional_eq(N, L=None):
    """xp projected onto functional-equation-symmetric subspace.

    The functional equation xi(s) = xi(1-s) means on the critical line:
    Z(t) = Z(-t) (Z is even in t, since xi(1/2+it) = xi(1/2-it))

    In log-grid coordinates, this is a PARITY constraint:
    g(u) = g(L - u) (reflection symmetry about L/2)

    Project xp onto the symmetric subspace: f(u) = f(L-u).
    The basis for this subspace: cos(n*pi*u/L), n=0,1,...,N/2

    On this symmetric subspace, xp = -i d/du maps symmetric -> antisymmetric.
    So we need xp^2 = -d^2/du^2, which maps symmetric -> symmetric.
    The eigenvalues of xp^2 are (n*pi/L)^2, so xp eigenvalues are ±n*pi/L.
    """
    if L is None:
        L = 2 * np.pi * N / 50

    h = L / N
    M = N // 2  # symmetric subspace dimension

    # Build xp^2 = -d^2/du^2 on symmetric subspace (cosine basis)
    # Eigenvalues: (n*pi/L)^2 for n = 0, 1, ..., M-1
    # In grid representation with symmetry:
    H2 = np.zeros((M, M))
    for j in range(M):
        H2[j, j] = 2 / h ** 2
        if j > 0:
            H2[j, j - 1] = -1 / h ** 2
            H2[j - 1, j] = -1 / h ** 2
    # Boundary: use Neumann at u=0 (symmetric)
    H2[0, 0] = 1 / h ** 2

    return H2, h, L


# ============================================================
# TEST 1: Log-grid xp with different BCs
# ============================================================
print("=" * 70)
print("TEST 1: xp ON LOG-GRID WITH DIFFERENT BOUNDARY CONDITIONS")
print("=" * 70)

N = 200

for name, builder in [("Periodic", build_xp_loggrid),
                       ("Dirichlet", build_xp_dirichlet),
                       ("Absorbing", lambda N: build_xp_absorbing(N, absorption=0.5))]:
    H, h, L = builder(N)
    eigs = eigvals(H)

    # Sort by real part
    eigs_real = np.sort(eigs.real)
    eigs_imag = np.sort(eigs.imag)

    print(f"\n  {name} (N={N}, L={L:.1f}, h={h:.4f}):")
    print(f"    Real parts: [{eigs_real[0]:.4f}, ..., {eigs_real[-1]:.4f}]")
    print(f"    Imag parts: [{eigs_imag[0]:.4f}, ..., {eigs_imag[-1]:.4f}]")
    print(f"    Spacing (real): {np.mean(np.diff(eigs_real[N//4:3*N//4])):.6f}")
    print(f"    Expected 2pi/L: {2*np.pi/L:.6f}")


# ============================================================
# TEST 2: Xi operator (companion matrix from Z(t) interpolation)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: XI OPERATOR — COMPANION MATRIX OF Z(t)")
print("=" * 70)

for N_xi, T_center, T_span in [(30, 30, 50), (50, 35, 60), (80, 40, 80)]:
    result = build_xi_operator(N_xi, T_center, T_span)
    if result[0] is None:
        print(f"\n  N={N_xi}: Failed to build companion matrix")
        continue

    C, t_min, t_max = result
    eigs_c = eigvals(C)

    # Map back to t-coordinates
    eigs_t = (eigs_c.real + 1) / 2 * (t_max - t_min) + t_min

    # Keep only real eigenvalues (imaginary = outside interpolation range)
    real_eigs = eigs_t[np.abs(eigs_c.imag) < 0.1]
    real_eigs = np.sort(real_eigs.real)

    # Filter to scan range
    in_range = real_eigs[(real_eigs > t_min + 2) & (real_eigs < t_max - 2)]

    print(f"\n  N={N_xi}, T in [{t_min:.1f}, {t_max:.1f}]:")
    print(f"    Total eigenvalues: {len(eigs_c)}")
    print(f"    Real eigenvalues in range: {len(in_range)}")

    if len(in_range) > 0:
        # Match to known zeros
        n_match = 0
        print(f"    {'Eigenvalue':>12} {'Nearest zero':>14} {'Dist':>8} {'Match?':>8}")
        for e in in_range[:15]:
            dists = [abs(e - z) for z in known_zeros_t]
            best_idx = np.argmin(dists)
            match = dists[best_idx] < 0.5
            if match:
                n_match += 1
            print(f"    {e:>12.4f} {known_zeros_t[best_idx]:>14.4f} "
                  f"{dists[best_idx]:>8.4f} {'YES' if match else '':>8}")

        zeros_in_range = sum(1 for z in known_zeros_t if t_min + 2 < z < t_max - 2)
        print(f"    Matched: {n_match}/{zeros_in_range}")


# ============================================================
# TEST 3: Absorbing boundary sweep
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: ABSORBING BOUNDARY STRENGTH SWEEP")
print("=" * 70)

N_abs = 300
L_abs = 2 * np.pi * N_abs / 50  # = 37.7

print(f"\n  N={N_abs}, L={L_abs:.1f}")
print(f"  Expected eigenvalue spacing = 2*pi/L = {2*np.pi/L_abs:.6f}")
print(f"  Zeta zero mean spacing near T=20: {2*np.pi/np.log(20/(2*np.pi)):.4f}")

for absorption in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]:
    H, h, L = build_xp_absorbing(N_abs, L=L_abs, absorption=absorption)
    eigs = eigvals(H)

    # Look for eigenvalues near the critical strip
    # The zeta zeros are at s = 1/2 + it, so the eigenvalues of xp
    # (which is the "log-energy" operator) should be near the t values.
    eigs_real = np.sort(eigs.real)

    # Check if any eigenvalues are near known zeta zeros
    n_near = 0
    for z in known_zeros_t[:5]:
        if any(abs(eigs.real - z) < 1.0):
            n_near += 1

    # Spectral gap
    central_eigs = eigs_real[N_abs // 4:3 * N_abs // 4]
    mean_spacing = np.mean(np.diff(central_eigs))

    print(f"  abs={absorption:>5.2f}: spacing={mean_spacing:.6f}, "
          f"max|Im|={np.max(np.abs(eigs.imag)):.4f}, "
          f"near_zeros={n_near}/5")


# ============================================================
# TEST 4: Can we build the operator BACKWARDS from known zeros?
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: INVERSE APPROACH — JACOBI MATRIX FROM ZETA ZEROS")
print("=" * 70)

# Given eigenvalues {t_1,...,t_N}, build the unique Jacobi (tridiagonal)
# matrix with those eigenvalues AND first eigenvector (1,1,...,1)/sqrt(N).
# This is the Lanczos algorithm / moment problem approach.

from scipy.linalg import eigh_tridiagonal

def jacobi_from_eigenvalues(eigenvalues, first_vec=None):
    """Build the Jacobi matrix with given eigenvalues.

    Uses the Lanczos construction: given eigenvalues and a starting vector,
    the Jacobi matrix is uniquely determined.
    """
    N = len(eigenvalues)
    if first_vec is None:
        first_vec = np.ones(N) / np.sqrt(N)

    # Build the eigenvector matrix from the eigenvalues
    # We need eigenvectors. For arbitrary eigenvalues with uniform
    # first components, use the Gaussian quadrature connection:
    # The Jacobi matrix J with eigenvalues {lambda_k} and
    # first row of eigenvector matrix = {w_k} is:
    # alpha_j (diagonal) and beta_j (off-diagonal) from the
    # three-term recurrence.

    # Use the fact that the eigenvalues ARE the nodes of a
    # Gaussian quadrature, and the weights are |first_vec_k|^2.
    lambdas = np.sort(eigenvalues)
    weights = np.abs(first_vec) ** 2

    # Lanczos-like construction via modified Chebyshev algorithm
    # Compute the modified moments: mu_k = sum_i w_i * lambda_i^k
    moments = np.zeros(2 * N)
    for k in range(2 * N):
        moments[k] = np.sum(weights * lambdas ** k)

    # From moments, extract alpha and beta via the Chebyshev algorithm
    alpha = np.zeros(N)
    beta = np.zeros(N - 1)

    # Simple recursion (numerically fragile for large N but OK for N~50)
    sigma = np.zeros((2 * N, 2 * N))
    sigma[0, :] = moments

    alpha[0] = moments[1] / moments[0]
    if N > 1:
        sigma[1, :2 * N - 1] = 0  # Will be filled
        for k in range(2 * N - 1):
            sigma[1, k] = sigma[0, k + 1] - alpha[0] * sigma[0, k]

        for j in range(1, N):
            beta_sq = sigma[j, 0] / sigma[j - 1, 0] if abs(sigma[j - 1, 0]) > 1e-30 else 0
            if beta_sq <= 0:
                break
            beta[j - 1] = np.sqrt(abs(beta_sq))
            alpha[j] = sigma[j, 1] / sigma[j, 0] - sigma[j - 1, 1] / sigma[j - 1, 0] \
                if abs(sigma[j, 0]) > 1e-30 and abs(sigma[j - 1, 0]) > 1e-30 else 0

            if j < N - 1:
                for k in range(2 * N - 2 * j - 1):
                    sigma[j + 1, k] = sigma[j, k + 1] - alpha[j] * sigma[j, k] - \
                                       beta_sq * sigma[j - 1, k + 1] if j > 0 else 0

    return alpha, beta


# Build Jacobi from first 20 zeta zeros
for n_zeros in [10, 20, 30]:
    zeros_use = np.array(known_zeros_t[:n_zeros]) if n_zeros <= len(known_zeros_t) else \
        np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, n_zeros + 1)])

    alpha, beta = jacobi_from_eigenvalues(zeros_use)

    # Reconstruct and verify
    valid_beta = beta[np.isfinite(beta) & (beta > 0)]
    n_valid = len(valid_beta)

    if n_valid > 2:
        eigs_check = eigh_tridiagonal(alpha[:n_valid + 1], valid_beta[:n_valid],
                                       eigvals_only=True)
        errors = np.sort(np.abs(eigs_check - np.sort(zeros_use[:n_valid + 1])))
        max_err = errors[-1] if len(errors) > 0 else float("inf")

        print(f"\n  {n_zeros} zeros -> Jacobi ({n_valid+1} valid):")
        print(f"    Diagonal (alpha):     [{alpha[0]:.4f}, {alpha[1]:.4f}, "
              f"{alpha[2]:.4f}, ...]")
        print(f"    Off-diagonal (beta):  [{valid_beta[0]:.4f}, "
              f"{valid_beta[1]:.4f if len(valid_beta)>1 else 0:.4f}, ...]")
        print(f"    Reconstruction error: {max_err:.2e}")

        # Is there structure in alpha and beta?
        if n_valid > 5:
            # Check if alpha grows linearly (like log(k))
            k_vals = np.arange(1, len(alpha[:n_valid + 1]) + 1)
            from scipy.stats import pearsonr
            r_lin, _ = pearsonr(k_vals, alpha[:n_valid + 1])
            r_log, _ = pearsonr(np.log(k_vals), alpha[:n_valid + 1])
            print(f"    alpha vs k:     r = {r_lin:+.4f}")
            print(f"    alpha vs log(k): r = {r_log:+.4f}")
    else:
        print(f"\n  {n_zeros} zeros -> Jacobi: only {n_valid} valid betas (unstable)")


# ============================================================
# TEST 5: Fetch more zeros and do larger Jacobi
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: JACOBI MATRIX FROM 50-100 ZETA ZEROS")
print("=" * 70)

# Compute first 100 zeta zeros
print("  Computing 100 zeta zeros...")
t_start = time.time()
all_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, 101)])
print(f"  Done ({time.time()-t_start:.1f}s)")

# Build Jacobi matrix from these zeros
for n_use in [20, 50, 100]:
    zeros_use = all_zeros[:n_use]
    alpha, beta = jacobi_from_eigenvalues(zeros_use)
    valid_beta = beta[np.isfinite(beta) & (np.abs(beta) > 1e-30) & (np.abs(beta) < 1e10)]
    n_valid = min(len(valid_beta), n_use - 1)

    if n_valid > 5:
        eigs_check = eigh_tridiagonal(alpha[:n_valid + 1], valid_beta[:n_valid],
                                       eigvals_only=True)
        # Error
        target = np.sort(zeros_use[:n_valid + 1])
        recon = np.sort(eigs_check)
        max_err = np.max(np.abs(target - recon)) if len(target) == len(recon) else float("inf")

        print(f"\n  {n_use} zeros -> {n_valid+1} Jacobi entries:")
        print(f"    Max reconstruction error: {max_err:.2e}")

        # Analyze alpha structure
        a = alpha[:n_valid + 1]
        k = np.arange(1, len(a) + 1)

        # Fit: alpha_k = c1 * k + c2
        A_mat = np.vstack([k, np.ones_like(k)]).T
        (c1, c2), residuals, _, _ = np.linalg.lstsq(A_mat, a, rcond=None)
        print(f"    Best linear fit: alpha ~ {c1:.4f}*k + {c2:.4f}")

        # Fit: alpha_k = c1 * log(k) + c2
        A_mat2 = np.vstack([np.log(k), np.ones_like(k)]).T
        (c1_log, c2_log), _, _, _ = np.linalg.lstsq(A_mat2, a, rcond=None)
        print(f"    Best log fit:    alpha ~ {c1_log:.4f}*log(k) + {c2_log:.4f}")

        # Analyze beta structure
        b = valid_beta[:n_valid]
        kb = np.arange(1, len(b) + 1)
        if len(b) > 3:
            print(f"    Beta range: [{np.min(b):.4f}, {np.max(b):.4f}]")
            print(f"    Beta mean: {np.mean(b):.4f}")
            print(f"    Beta std:  {np.std(b):.4f}")
    else:
        print(f"\n  {n_use} zeros -> only {n_valid} valid (numerically unstable)")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

print(f"\nTotal time: {time.time() - t0:.1f}s")
