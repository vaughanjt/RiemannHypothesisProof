"""
Session 28: Derive and verify the EXACT Mellin formula for the (0,1) Gram matrix.

Derivation (substitution u = 1/(jx)):
  f_hat_j(s) = integral_0^1 {1/(jx)} x^{s-1} dx = j^{-s} integral_{1/j}^inf {u} u^{-s-1} du

After computing the integral:
  f_hat_j(s) = 1/(j(s-1)) - j^{-s} zeta(s)/s

Parseval for L^2(0,1) with Mellin on Re(s)=1/2:
  G_{jk} = (1/2pi) integral |f_hat_j(1/2+it)|^2 dt ... wait, that's not right.
  G_{jk} = (1/2pi) integral f_hat_j(1/2+it) conj(f_hat_k(1/2+it)) dt

Verify: compute both sides for the N=10 Gram matrix.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, log, exp, nstr, fabs, gamma, power
import time

mp.dps = 30


def f_hat(j, s):
    """Mellin transform of {1/(jx)} on (0,1) at s.

    Derived formula: f_hat_j(s) = 1/(j*(s-1)) - j^{-s} * zeta(s) / s
    """
    j_mp = mpf(j)
    return 1 / (j_mp * (s - 1)) - power(j_mp, -s) * zeta(s) / s


def f_hat_numerical(j, s, n_quad=10000):
    """Numerical Mellin transform for verification."""
    j_mp = mpf(j)
    def integrand(x):
        if x < mpf(10)**(-14): return mpf(0)
        u = 1 / (j_mp * x)
        frac_u = u - mpmath.floor(u)
        return frac_u * power(x, s - 1)
    return mpmath.quad(integrand, [mpf(10)**(-14), mpf(1)], method='tanh-sinh')


if __name__ == "__main__":
    print("EXACT MELLIN TRANSFORM VERIFICATION")
    print("f_hat_j(s) = 1/(j(s-1)) - j^{-s} zeta(s)/s")
    print("=" * 70)

    # Step 1: Verify the formula against numerical integration
    print("\nStep 1: Formula vs numerical integration")
    print(f"{'j':>3} {'s':>15} {'formula':>20} {'numerical':>20} {'ratio':>10}")
    print("-" * 70)

    for j in [1, 2, 3, 5, 10]:
        for s_val in [mpc(2, 0), mpc(1.5, 3), mpc(0.5, 5), mpc(0.5, 14.13)]:
            form = f_hat(j, s_val)
            numer = f_hat_numerical(j, s_val)
            ratio = abs(form) / abs(numer) if abs(numer) > 1e-20 else float('nan')
            print(f"{j:>3} {nstr(s_val, 8):>15} {nstr(form, 12):>20} {nstr(numer, 12):>20} {float(ratio):>10.6f}")

    # Step 2: Verify Parseval — G_{jk} = (1/2pi) integral f_hat_j(1/2+it) conj(f_hat_k(1/2+it)) dt
    print(f"\nStep 2: Parseval verification for G_{{jk}}")
    print("G_{jk} = (1/2pi) integral f_hat_j(1/2+it) * conj(f_hat_k(1/2+it)) dt")

    # Build exact G from numpy
    N = 10
    n_grid = 500000
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    G_numpy = (fp @ fp.T) * dx

    # Compute G via Mellin integral
    T_max = 200; n_t = 20000
    t_grid = np.linspace(-T_max, T_max, n_t)
    dt_val = t_grid[1] - t_grid[0]

    G_mellin = np.zeros((N, N))
    print(f"\nComputing Mellin integral (T_max={T_max}, n_t={n_t})...", flush=True)
    t0 = time.time()

    for i_t in range(n_t):
        t = t_grid[i_t]
        s = mpc(0.5, t)

        # Compute f_hat for all j
        fh = [f_hat(j, s) for j in range(1, N+1)]

        for j in range(N):
            for k in range(j, N):
                val = float((fh[j] * mpmath.conj(fh[k])).real)
                G_mellin[j, k] += val * dt_val / (2 * float(pi))
                if k != j:
                    G_mellin[k, j] = G_mellin[j, k]

    print(f"  Done ({time.time()-t0:.1f}s)")

    # Compare
    print(f"\n{'j':>3} {'k':>3} {'G_numpy':>14} {'G_mellin':>14} {'ratio':>10}")
    print("-" * 50)
    for j in range(1, min(6, N+1)):
        for k in range(j, min(6, N+1)):
            gn = G_numpy[j-1, k-1]
            gm = G_mellin[j-1, k-1]
            ratio = gm / gn if abs(gn) > 1e-15 else float('nan')
            print(f"{j:>3} {k:>3} {gn:>14.8f} {gm:>14.8f} {ratio:>10.6f}")

    # Eigenvalue comparison
    evals_numpy = np.linalg.eigvalsh(G_numpy)
    evals_mellin = np.linalg.eigvalsh(G_mellin)
    print(f"\nsigma_min: numpy={evals_numpy[0]:.6e}, mellin={evals_mellin[0]:.6e}, "
          f"ratio={evals_mellin[0]/evals_numpy[0]:.6f}")
    print(f"sigma_max: numpy={evals_numpy[-1]:.6e}, mellin={evals_mellin[-1]:.6e}, "
          f"ratio={evals_mellin[-1]/evals_numpy[-1]:.6f}")
