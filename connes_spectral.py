"""
Session 19d: Connes spectral operator — direct from zeros.

Strategy: Build QW directly from the explicit formula (sum over zeros),
then construct the rank-one perturbation and verify eigenvalue matching.

This is CIRCULAR (uses zeros to build the operator whose eigenvalues should
match the zeros), but it verifies the mathematical framework and gives us
a template for the non-circular version.

QW(V_n, V_m) = sum_rho V_n_hat(rho) * conj(V_m_hat(rho))

where V_n_hat(s) = 2*sinh((s + i*n*pi/L)*log(lambda)) / (s + i*n*pi/L)
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, power

mp.dps = 30


def mellin_Vn(s, n, L, log_lambda):
    """Mellin transform of V_n at s: 2*sinh((s+i*n*pi/L)*log(lambda))/(s+i*n*pi/L)"""
    alpha = s + mpc(0, n * float(pi) / L)
    if abs(alpha) < 1e-15:
        return 2 * log_lambda  # limit
    return 2 * mpmath.sinh(alpha * log_lambda) / alpha


def build_QW_from_zeros(N, L, gammas, n_zeros=100):
    """
    QW(V_n, V_m) = sum_rho V_n_hat(rho) * conj(V_m_hat(rho))

    Under RH: rho = 1/2 + i*gamma, so:
    QW(n,m) = sum_{gamma>0} 2*Re[V_n_hat(1/2+i*gamma) * conj(V_m_hat(1/2+i*gamma))]
    """
    dim = 2 * N + 1
    log_lam = mpf(L) / 2
    QW = np.zeros((dim, dim))

    for gamma in gammas[:n_zeros]:
        rho = mpc(0.5, gamma)
        # Compute V_n_hat(rho) for each n
        M = []
        for idx in range(dim):
            n = idx - N
            M.append(complex(mellin_Vn(rho, n, L, log_lam)))
        M = np.array(M)

        # QW += 2*Re[M * conj(M)^T] (outer product)
        outer = np.outer(M, np.conj(M))
        QW += 2 * np.real(outer)

    return QW


def scaling_operator(N, L):
    dim = 2 * N + 1
    D = np.zeros((dim, dim))
    for idx in range(dim):
        n = idx - N
        D[idx, idx] = n * np.pi / L
    return D


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 19d: Connes Spectral Operator — Zero-Based Verification")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # Parameters
    lam_sq = 14
    L = np.log(lam_sq)
    N = 60  # matrix dimension = 2*60+1 = 121
    n_zeros = 200

    print(f"\n  lambda^2 = {lam_sq}, L = {L:.6f}")
    print(f"  N = {N}, dim = {2*N+1}")
    print(f"  Using {n_zeros} zeros")

    # --- Step 1: Build QW from zeros ---
    print("\n--- Step 1: QW from zeros ---")
    QW = build_QW_from_zeros(N, L, gammas, n_zeros)
    QW = (QW + QW.T) / 2  # symmetrize

    eigvals_QW, eigvecs_QW = np.linalg.eigh(QW)
    print(f"  Smallest eigenvalue: {eigvals_QW[0]:+.10f}")
    print(f"  Largest eigenvalue:  {eigvals_QW[-1]:+.10f}")
    print(f"  Positive eigenvalues: {np.sum(eigvals_QW > 0)}")
    print(f"  Negative eigenvalues: {np.sum(eigvals_QW < 0)}")

    # Minimum eigenvector
    xi = eigvecs_QW[:, 0]

    # --- Step 2: Scaling operator ---
    print("\n--- Step 2: Scaling operator D_log ---")
    D = scaling_operator(N, L)
    print(f"  Eigenvalues: n*pi/L, range [{-N*np.pi/L:.4f}, {N*np.pi/L:.4f}]")
    print(f"  Spacing: pi/L = {np.pi/L:.6f}")

    # --- Step 3: Rank-one perturbation ---
    print("\n--- Step 3: Rank-one perturbation ---")
    print("  D_perturbed = D - |D*xi><xi| / <xi|xi>")
    print("  (Simple projection form — paper's form is more specific)")

    # Normalize xi
    xi_norm = xi / np.linalg.norm(xi)
    Dxi = D @ xi_norm

    # rank-one perturbation: D - outer(Dxi, xi_norm)
    D_pert = D - np.outer(Dxi, xi_norm)

    eigs = np.linalg.eigvals(D_pert)
    eigs_real = np.sort(np.real(eigs))
    eigs_imag = np.abs(np.imag(eigs))

    print(f"  Max imaginary part: {np.max(eigs_imag):.6e}")

    # --- Step 4: Compare with zeros ---
    print("\n--- Step 4: Eigenvalues vs zeta zeros ---")

    pos_eigs = sorted([e for e in eigs_real if e > 1])

    print(f"  {'#':>3s}  {'eigenvalue':>14s}  {'zero':>14s}  {'diff':>12s}  {'rel err':>10s}")
    for i in range(min(20, len(pos_eigs), len(gammas))):
        e = pos_eigs[i]
        z = gammas[i]
        diff = e - z
        rel = abs(diff / z) * 100
        print(f"  {i+1:3d}  {e:14.6f}  {z:14.6f}  {diff:+12.6f}  {rel:9.4f}%")

    # --- Step 5: Try different perturbation forms ---
    print("\n--- Step 5: Alternative perturbation: eigenvalue equation ---")
    print("  det(D - z*I - outer(Dxi, xi)) = 0")
    print("  equiv: 1 = xi^T * (D-zI)^{-1} * Dxi  (matrix det lemma)")
    print()
    print("  Solving for z where 1 = sum_n xi_n * (n*pi/L) * xi_n / (n*pi/L - z)")

    # The equation: sum_n (n*pi/L * xi_n^2) / (n*pi/L - z) = 1
    # This is a rational function in z with poles at n*pi/L

    def secular_eq(z, xi_vec, N, L):
        """1 - sum (n*pi/L * xi_n^2) / (n*pi/L - z)"""
        total = 0.0
        for idx in range(2 * N + 1):
            n = idx - N
            ev = n * np.pi / L
            total += ev * xi_vec[idx]**2 / (ev - z)
        return 1 - total

    # Find roots by scanning
    print("  Scanning for sign changes in secular equation...")
    z_range = np.linspace(1, 60, 10000)
    roots = []
    prev_val = secular_eq(z_range[0], xi_norm, N, L)
    for i in range(1, len(z_range)):
        val = secular_eq(z_range[i], xi_norm, N, L)
        if prev_val * val < 0:
            # Bisect
            lo, hi = z_range[i-1], z_range[i]
            for _ in range(60):
                mid = (lo + hi) / 2
                if secular_eq(mid, xi_norm, N, L) * secular_eq(lo, xi_norm, N, L) < 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        prev_val = val

    print(f"\n  Found {len(roots)} roots in [1, 60]")
    print(f"  {'#':>3s}  {'secular root':>14s}  {'zero':>14s}  {'diff':>12s}")
    for i in range(min(15, len(roots), len(gammas))):
        r = roots[i]
        z = gammas[i]
        print(f"  {i+1:3d}  {r:14.6f}  {z:14.6f}  {r - z:+12.6f}")

    # --- Step 6: Quality assessment ---
    print("\n--- Step 6: Assessment ---")
    if len(roots) > 0:
        errors = [abs(roots[i] - gammas[i]) for i in range(min(len(roots), len(gammas)))]
        print(f"  Mean absolute error: {np.mean(errors):.6f}")
        print(f"  Max absolute error:  {np.max(errors):.6f}")
        print(f"  Mean relative error: {np.mean([e/gammas[i] for i, e in enumerate(errors)]):.6f}")
    print()
    print("  Note: This uses QW built FROM zeros (circular).")
    print("  The non-circular version needs proper W_R implementation.")
    print("  But this verifies the STRUCTURAL correctness of the framework.")

    print("\n" + "=" * 70)
