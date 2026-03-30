"""
Session 27a: Detailed structure of the NB Gram matrix for the sigma_min proof.

Need to establish:
1. Diagonal behavior: G_{jj} ~ ? as j grows
2. Off-diagonal decay: G_{jk} ~ ? as |j-k| grows
3. Row sums: R_j = sum_{k!=j} |G_{jk}| vs G_{jj} (diagonal dominance?)
4. Schur complement structure
5. Exact formula for G_{jk} in terms of arithmetic functions
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, quad, floor, nstr, fabs, log, euler
import time

mp.dps = 30


def gram_entry_precise(j, k):
    """Compute G_{jk} using mpmath tanh-sinh quadrature."""
    j_mp = mpf(j); k_mp = mpf(k)
    def integrand(x):
        if x < mpf(10)**(-14): return mpf(0)
        return (1/(j_mp*x) - floor(1/(j_mp*x))) * (1/(k_mp*x) - floor(1/(k_mp*x)))
    return float(quad(integrand, [mpf(10)**(-14), mpf(1)], method='tanh-sinh'))


def build_gram_numpy(N, n_grid=500000):
    x = np.linspace(1.0 / n_grid, 1.0, n_grid)
    dx = x[1] - x[0]
    frac_parts = np.zeros((N, n_grid))
    for k in range(1, N + 1):
        vals = 1.0 / (k * x)
        frac_parts[k - 1] = vals - np.floor(vals)
    G = (frac_parts @ frac_parts.T) * dx
    return G


if __name__ == "__main__":
    print("NB GRAM MATRIX STRUCTURE ANALYSIS")
    print("=" * 70)

    N = 100
    print(f"Building G_{N}x{N} (numpy, 500K grid)...", end="", flush=True)
    t0 = time.time()
    G = build_gram_numpy(N, n_grid=500000)
    print(f" ({time.time()-t0:.1f}s)")

    # 1. DIAGONAL BEHAVIOR
    print("\n1. DIAGONAL BEHAVIOR: G_{jj} vs j")
    print(f"   {'j':>5} {'G_{jj}':>12} {'j*G_{jj}':>12} {'j^2*G_{jj}':>12}")
    for j in [1, 2, 3, 5, 10, 20, 50, 100]:
        gjj = G[j-1, j-1]
        print(f"   {j:>5} {gjj:>12.6f} {j*gjj:>12.6f} {j**2*gjj:>12.4f}")

    # Fit G_{jj} ~ A / j^alpha
    js = np.arange(1, N+1)
    diag = np.array([G[j-1,j-1] for j in range(1, N+1)])
    log_j = np.log(js[9:])  # skip first few for fit
    log_d = np.log(diag[9:])
    alpha_d, log_A = np.polyfit(log_j, log_d, 1)
    print(f"\n   Fit: G_{{jj}} ~ {np.exp(log_A):.4f} * j^({alpha_d:.4f})")

    # 2. OFF-DIAGONAL DECAY
    print("\n2. OFF-DIAGONAL DECAY: G_{1,k} and G_{j,j+1}")
    print(f"   {'k':>5} {'G_{1,k}':>12} {'k*G_{1,k}':>12} {'G_{k,k+1}':>12} {'k*G_{k,k+1}':>12}")
    for k in [2, 3, 5, 10, 20, 50]:
        g1k = G[0, k-1]
        if k < N:
            gkk1 = G[k-1, k]
        else:
            gkk1 = 0
        print(f"   {k:>5} {g1k:>12.6f} {k*g1k:>12.6f} {gkk1:>12.6f} {k*gkk1:>12.6f}")

    # General off-diagonal: G_{j,k} as function of j,k
    print(f"\n   G_{{j,k}} * j * k for various j,k:")
    print(f"   {'':>5}", end="")
    for k in [1,2,5,10,20,50]:
        print(f" {'k='+str(k):>10}", end="")
    print()
    for j in [1,2,5,10,20,50]:
        print(f"   j={j:<3}", end="")
        for k in [1,2,5,10,20,50]:
            if j <= N and k <= N:
                print(f" {j*k*G[j-1,k-1]:>10.4f}", end="")
            else:
                print(f" {'---':>10}", end="")
        print()

    # 3. ROW SUMS (Gershgorin)
    print("\n3. ROW SUMS (Gershgorin bounds)")
    print(f"   {'j':>5} {'G_{jj}':>12} {'R_j':>12} {'G_{jj}-R_j':>12} {'dom?':>6}")
    for j in [1, 2, 5, 10, 20, 50, 100]:
        gjj = G[j-1, j-1]
        Rj = np.sum(np.abs(G[j-1, :])) - gjj
        diff = gjj - Rj
        dom = "YES" if diff > 0 else "NO"
        print(f"   {j:>5} {gjj:>12.6f} {Rj:>12.6f} {diff:>12.6f} {dom:>6}")

    # 4. EIGENVALUE SPACING near sigma_min
    evals = np.linalg.eigvalsh(G)
    print(f"\n4. EIGENVALUE SPACING (bottom 10)")
    print(f"   {'i':>5} {'sigma_i':>14} {'ratio':>10}")
    for i in range(min(10, N)):
        ratio = evals[i+1]/evals[i] if i < N-1 else 0
        print(f"   {i+1:>5} {evals[i]:>14.6e} {ratio:>10.4f}")

    # 5. EXACT DIAGONAL FORMULA
    # G_{jj} = integral_0^1 {1/(jx)}^2 dx
    # Known: G_{jj} = 1/(2j) - 1/(12j^2) + O(1/j^3)  (from Euler-Maclaurin)
    # Actually: integral {u}^2 du from 0 to n = n/6 for integer n
    # G_{jj} = sum_{a=1}^{infty} integral_{1/(j(a+1))}^{1/(ja)} (1/(jx)-a)^2 dx
    print(f"\n5. DIAGONAL FORMULA CHECK: G_{{jj}} vs 1/(2j) - 1/(12j^2)")
    print(f"   {'j':>5} {'G_{jj}':>12} {'approx':>12} {'error':>12}")
    for j in [1, 2, 5, 10, 20, 50, 100]:
        gjj = G[j-1, j-1]
        approx = 1/(2*j) - 1/(12*j**2)
        # Actually for j=1: integral_0^1 {1/x}^2 dx.
        # For x in (1/(a+1), 1/a]: {1/x} = 1/x - a
        # integral (1/x-a)^2 dx = [-1/x - 2a*ln(x) + a^2*x]
        # Sum over a=1 to infinity
        print(f"   {j:>5} {gjj:>12.6f} {approx:>12.6f} {gjj-approx:>12.6f}")

    # Better approximation using Euler-Maclaurin
    # G_{jj} = (ln(j) + gamma + 1)/(2j) + O(1/j^2)  ??
    # Let me check: j*G_{jj} for various j
    print(f"\n   j*G_{{jj}}: {', '.join(f'{j*G[j-1,j-1]:.4f}' for j in [1,2,5,10,20,50,100])}")
    print(f"   ln(j)/2+c: {', '.join(f'{np.log(j)/2+0.26:.4f}' for j in [1,2,5,10,20,50,100])}")
