"""
Session 26l: Fast Nyman-Beurling Gram matrix via numpy vectorization.

G_{jk} = integral_0^1 {1/(jx)} {1/(kx)} dx

Use numpy with fine grid (100K points) for speed, then verify
key entries with mpmath for precision.
"""

import numpy as np
from scipy.linalg import svd as scipy_svd
import time


def build_gram_numpy(N, n_grid=200000):
    """Build N x N Gram matrix using numpy quadrature."""
    # Grid on (0, 1] — avoid x=0
    x = np.linspace(1.0 / n_grid, 1.0, n_grid)
    dx = x[1] - x[0]

    # Precompute fractional parts {1/(kx)} for k=1..N
    frac_parts = np.zeros((N, n_grid))
    for k in range(1, N + 1):
        vals = 1.0 / (k * x)
        frac_parts[k - 1] = vals - np.floor(vals)

    # Gram matrix: G_{jk} = integral frac_j * frac_k dx
    G = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            G[j, k] = np.sum(frac_parts[j] * frac_parts[k]) * dx
            G[k, j] = G[j, k]

    # b vector: b_k = integral {1/(kx)} dx
    b = np.zeros(N)
    for k in range(N):
        b[k] = np.sum(frac_parts[k]) * dx

    return G, b


if __name__ == "__main__":
    print("NYMAN-BEURLING GRAM MATRIX (FAST)")
    print("=" * 70)

    for N in [20, 50]:
        t0 = time.time()
        print(f"\nN = {N}, building Gram matrix...", end="", flush=True)
        G, b = build_gram_numpy(N, n_grid=200000)
        print(f" ({time.time()-t0:.1f}s)")

        # Eigenvalues
        evals = np.linalg.eigvalsh(G)
        print(f"\n  Eigenvalue spectrum:")
        print(f"    sigma_1 (min) = {evals[0]:.10e}")
        for i in range(1, min(5, N)):
            print(f"    sigma_{i+1}       = {evals[i]:.10e}")
        print(f"    ...")
        print(f"    sigma_{N} (max) = {evals[-1]:.10e}")
        print(f"  Condition number: {evals[-1]/evals[0]:.4e}")

        # Distance
        try:
            c_opt = np.linalg.solve(G, b)
            d_sq = 1.0 - np.dot(b, c_opt)
            print(f"\n  d_N^2 = {d_sq:.10e}")
            print(f"  d_N   = {np.sqrt(max(0, d_sq)):.10e}")
        except:
            print(f"  Singular matrix")

        # DISPLACEMENT RANK CHECK
        D = np.diag(np.arange(1, N + 1, dtype=float))
        disp = D @ G - G @ D
        _, S_disp, _ = scipy_svd(disp)
        print(f"\n  DISPLACEMENT D*G - G*D:")
        print(f"    Top 10 singular values: {', '.join(f'{s:.4e}' for s in S_disp[:10])}")
        eff_rank = np.sum(S_disp > S_disp[0] * 1e-10)
        print(f"    Effective rank: {eff_rank}")

        # Check Cauchy-like structure: is G_{jk} = (a_j - a_k)/(j-k) for j != k?
        # Extract a_j from the first column: a_j - a_1 = (j-1) * G_{j,1} for j != 1
        a_test = np.zeros(N)
        a_test[0] = 0  # gauge choice
        for j in range(1, N):
            a_test[j] = (j + 1 - 1) * G[j, 0]  # a_{j+1} - a_1 = j * G[j+1-1, 1-1]
            # Wait, indices: G[j,0] = G_{j+1,1} in 1-indexed = integral {1/((j+1)x)}{1/x} dx
            # Cauchy-like: G_{j+1,1} = (a_{j+1} - a_1)/(j+1 - 1) = (a_{j+1} - a_1)/j
            # So a_{j+1} = a_1 + j * G[j,0]
            a_test[j] = j * G[j, 0]  # with a_1 = 0

        # Verify: (a_j - a_k)/(j-k) = G_{jk}?
        max_err = 0
        for j in range(N):
            for k in range(N):
                if j != k:
                    predicted = (a_test[j] - a_test[k]) / ((j + 1) - (k + 1))
                    err = abs(predicted - G[j, k])
                    max_err = max(max_err, err)
        print(f"\n  CAUCHY-LIKE TEST: max |(a_j-a_k)/(j-k) - G_{j,k}| = {max_err:.6e}")
        if max_err < 0.01:
            print(f"    -> CAUCHY-LIKE STRUCTURE DETECTED!")
        else:
            print(f"    -> NOT Cauchy-like (error too large)")

        # Toeplitz check: is G_{j,k} ~ f(j-k)?
        diag_vals = {}
        for d in range(-min(5, N - 1), min(6, N)):
            vals = [G[i, i + d] for i in range(N) if 0 <= i + d < N]
            if vals:
                diag_vals[d] = (np.mean(vals), np.std(vals))
        print(f"\n  TOEPLITZ CHECK (mean, std along diagonals):")
        for d in sorted(diag_vals.keys()):
            m, s = diag_vals[d]
            print(f"    diag {d:>3}: mean={m:>12.6f}  std={s:>12.6f}  {'~Toeplitz' if s < 0.01*abs(m) else 'NOT Toeplitz'}")

        # Overall structure
        print(f"\n  G[1:5, 1:5]:")
        for i in range(min(5, N)):
            row = "  ".join(f"{G[i,j]:10.6f}" for j in range(min(5, N)))
            print(f"    {row}")

    print(f"\n{'='*70}")
    print("KEY FINDINGS:")
    print("  1. Displacement rank of G?")
    print("  2. Does sigma_min decrease as N grows? (need it -> 0 for RH)")
    print("  3. Is G Cauchy-like or Toeplitz?")
    print("  4. How fast does d_N -> 0?")
