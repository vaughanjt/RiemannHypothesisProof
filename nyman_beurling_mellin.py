"""
Session 27b: Verify the Mellin formula for the NB Gram matrix.

Grok's formula:
  c* G c = (1/2pi) integral |D(1/2+it)|^2 |zeta(1/2+it)|^2 / (1/4+t^2) dt
where D(s) = sum c_j j^{-s}.

Test: compute both sides for specific c vectors and verify agreement.
Then check the Montgomery-Vaughan lower bound.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, nstr, fabs, log
import time

mp.dps = 30


def gram_matrix_numpy(N, n_grid=500000):
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    return (fp @ fp.T) * dx


def mellin_quadratic_form(c_vec, N, T_max=100, n_t=10000):
    """Compute (1/2pi) integral |D(1/2+it)|^2 |zeta(1/2+it)|^2 / (1/4+t^2) dt.

    D(s) = sum_{j=1}^N c_j j^{-s}
    """
    t_grid = np.linspace(-T_max, T_max, n_t)
    dt = t_grid[1] - t_grid[0]

    total = 0.0
    for i_t in range(n_t):
        t = t_grid[i_t]
        s = mpc(0.5, t)

        # D(s) = sum c_j j^{-s}
        D_val = sum(c_vec[j] * mpmath.power(j+1, -s) for j in range(N))

        # zeta(s)
        z_val = zeta(s)

        # weight
        w = 1.0 / (0.25 + t**2)

        # integrand
        integrand = float(abs(D_val)**2 * abs(z_val)**2 * w)
        total += integrand

    return total * dt / (2 * float(pi))


if __name__ == "__main__":
    print("MELLIN FORMULA VERIFICATION FOR NB GRAM MATRIX")
    print("=" * 70)

    N = 10  # Small N for speed

    # Build Gram matrix
    G = gram_matrix_numpy(N, n_grid=500000)
    evals, evecs = np.linalg.eigh(G)

    print(f"N = {N}")
    print(f"sigma_min = {evals[0]:.6e}")
    print(f"sigma_max = {evals[-1]:.6e}")

    # Test with several c vectors
    test_vectors = [
        ("e_1 (unit)", np.eye(N)[0]),
        ("e_N (unit)", np.eye(N)[-1]),
        ("uniform", np.ones(N) / np.sqrt(N)),
        ("min eigvec", evecs[:, 0]),
        ("max eigvec", evecs[:, -1]),
    ]

    print(f"\n{'Test vector':>15} {'c*Gc (matrix)':>15} {'Mellin integral':>15} {'ratio':>10}")
    print("-" * 60)

    for name, c in test_vectors:
        # Matrix quadratic form
        qf_matrix = c @ G @ c

        # Mellin integral
        t0 = time.time()
        qf_mellin = mellin_quadratic_form(c, N, T_max=50, n_t=5000)
        dt = time.time() - t0

        ratio = qf_mellin / qf_matrix if qf_matrix > 1e-15 else float('nan')
        print(f"{name:>15} {qf_matrix:>15.6e} {qf_mellin:>15.6e} {ratio:>10.4f}  ({dt:.1f}s)")

    # Now test the Montgomery-Vaughan piece: integral |D|^2 dt ~ sum |c_j|^2 * T
    print(f"\nMONTGOMERY-VAUGHAN CHECK:")
    print(f"integral_0^T |D(1/2+it)|^2 dt vs T * sum |c_j|^2")

    c_min = evecs[:, 0]  # minimum eigenvector
    for T in [10, 20, 50, 100]:
        t_grid = np.linspace(0, T, 2000)
        dt = t_grid[1] - t_grid[0]
        D_sq_integral = 0.0
        for t in t_grid:
            s = mpc(0.5, float(t))
            D_val = sum(c_min[j] * mpmath.power(j+1, -s) for j in range(N))
            D_sq_integral += float(abs(D_val)**2)
        D_sq_integral *= dt

        expected = T * np.sum(c_min**2)
        print(f"  T={T:>4}: integral = {D_sq_integral:.4f}, T*||c||^2 = {expected:.4f}, "
              f"ratio = {D_sq_integral/expected:.4f}")
