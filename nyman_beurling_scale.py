"""
Session 27: Scale Nyman-Beurling Gram matrix to N=200.

Track sigma_min(N) and d_N as functions of N to determine:
1. Rate of sigma_min decay (polynomial? exponential?)
2. Rate of d_N -> 0 (need this for RH)
3. Structure of the eigenvalue spectrum
4. Whether the smallest eigenvector has interpretable structure
"""

import numpy as np
from scipy.linalg import svd as scipy_svd
import time


def build_gram(N, n_grid=200000):
    """Build N x N Gram matrix. G_{jk} = integral_0^1 {1/(jx)}{1/(kx)} dx."""
    x = np.linspace(1.0 / n_grid, 1.0, n_grid)
    dx = x[1] - x[0]

    # Precompute fractional parts
    frac_parts = np.zeros((N, n_grid))
    for k in range(1, N + 1):
        vals = 1.0 / (k * x)
        frac_parts[k - 1] = vals - np.floor(vals)

    # Gram matrix via dot products
    G = (frac_parts @ frac_parts.T) * dx

    # b vector
    b = np.sum(frac_parts, axis=1) * dx

    return G, b


if __name__ == "__main__":
    print("NYMAN-BEURLING SCALING STUDY")
    print("=" * 70)

    results = []

    for N in [10, 20, 30, 50, 75, 100, 150, 200]:
        t0 = time.time()
        n_grid = max(200000, N * 5000)  # Scale grid with N
        print(f"N={N:>4}, grid={n_grid}...", end="", flush=True)

        G, b = build_gram(N, n_grid)
        dt_build = time.time() - t0

        # Eigenvalues
        evals = np.linalg.eigvalsh(G)
        sigma_min = evals[0]
        sigma_max = evals[-1]
        cond = sigma_max / sigma_min if sigma_min > 0 else float('inf')

        # Distance
        try:
            c_opt = np.linalg.solve(G, b)
            d_sq = 1.0 - np.dot(b, c_opt)
            d_N = np.sqrt(max(0, d_sq))
        except:
            d_sq = float('nan')
            d_N = float('nan')

        # Smallest eigenvector
        _, evecs = np.linalg.eigh(G)
        v_min = evecs[:, 0]  # eigenvector for smallest eigenvalue

        dt = time.time() - t0
        print(f" sig_min={sigma_min:.4e}, d_N={d_N:.6f}, cond={cond:.1e} ({dt:.1f}s)")

        results.append({
            'N': N, 'sigma_min': sigma_min, 'sigma_max': sigma_max,
            'd_N': d_N, 'cond': cond, 'v_min': v_min[:10]  # first 10 components
        })

    # Fit sigma_min(N) ~ N^{-alpha}
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS")
    print("-" * 70)

    Ns = np.array([r['N'] for r in results])
    sigs = np.array([r['sigma_min'] for r in results])
    ds = np.array([r['d_N'] for r in results])

    # Log-log fit for sigma_min
    log_N = np.log(Ns)
    log_sig = np.log(sigs)
    alpha, log_C = np.polyfit(log_N, log_sig, 1)
    print(f"\nsigma_min ~ C * N^alpha:")
    print(f"  alpha = {alpha:.4f}")
    print(f"  C = {np.exp(log_C):.6f}")
    print(f"  Fit: sigma_min ~ {np.exp(log_C):.4f} * N^({alpha:.3f})")

    # Log-log fit for d_N
    log_d = np.log(ds)
    beta, log_D = np.polyfit(log_N, log_d, 1)
    print(f"\nd_N ~ D * N^beta:")
    print(f"  beta = {beta:.4f}")
    print(f"  D = {np.exp(log_D):.6f}")
    print(f"  Fit: d_N ~ {np.exp(log_D):.4f} * N^({beta:.3f})")

    # Table
    print(f"\n{'N':>6} {'sigma_min':>12} {'d_N':>12} {'cond':>12} {'sig_fit':>12} {'d_fit':>12}")
    print("-" * 70)
    for r in results:
        sf = np.exp(log_C) * r['N']**alpha
        df = np.exp(log_D) * r['N']**beta
        print(f"{r['N']:>6} {r['sigma_min']:>12.4e} {r['d_N']:>12.6f} {r['cond']:>12.1e} {sf:>12.4e} {df:>12.6f}")

    # Eigenvector structure
    print(f"\nSmallest eigenvector (first 10 components):")
    for r in results[-3:]:
        v = r['v_min']
        print(f"  N={r['N']:>4}: [{', '.join(f'{c:.4f}' for c in v)}]")

    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"  If alpha < 0 (sigma_min decreasing): consistent with RH")
    print(f"  If beta < 0 (d_N decreasing): consistent with RH")
    print(f"  Rate matters: known results predict sigma_min ~ exp(-c*sqrt(N))")
    print(f"  Our fit gives sigma_min ~ N^({alpha:.2f})")
