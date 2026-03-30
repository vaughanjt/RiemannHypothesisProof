"""
Session 29q: THE KEY ESTIMATE — does |xi_hat(gamma_k)| -> 0?

If |xi_hat(gamma_k)| <= C * eps_0 for each zeta zero gamma_k,
then H_{lambda,N} -> Xi uniformly on compacts, and RH follows.

This estimate comes from the variational equation, NOT from
eigenvector freezing. We only need eps_0 -> 0 (H3) and the
spectral gap (H2) for the T-metric.

VERIFY: compute xi_hat at zeta zeros for varying lambda with fixed N.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, nstr
import time
import sys

mp.dps = 30

sys.path.insert(0, '.')
from connes_h1h2_correct import build_QW


def xi_hat(z, xi, N, L_f):
    """Compute xi_hat(z) = multiplicative Fourier transform."""
    s = np.sin(z * L_f / 2)
    if abs(s) < 1e-60:
        return 0
    total = sum(xi[j+N] / (z - 2*np.pi*j/L_f)
                for j in range(-N, N+1)
                if abs(z - 2*np.pi*j/L_f) > 1e-12)
    return 2 * L_f**(-0.5) * s * total


if __name__ == "__main__":
    print("THE KEY ESTIMATE: |xi_hat(gamma_k)| vs eps_0")
    print("=" * 70)

    # Load zeta zeros
    gammas = np.load("_zeros_500.npy")

    N = 15  # Fixed

    print(f"\nFixed N={N}")
    print(f"\n{'lam^2':>6} {'eps_0':>12} {'|xi_hat(g1)|':>14} {'|xi_hat(g2)|':>14} "
          f"{'|xi_hat(g3)|':>14} {'ratio1':>10}")
    print("-" * 75)

    for lam_sq in [10, 14, 20, 30, 40, 50, 60]:
        t0 = time.time()
        L_f = np.log(lam_sq)
        QW = build_QW(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi = evecs[:, 0]
        eps_0 = evals[0]

        # Normalize: sum(xi) ~ sqrt(L)
        xs = np.sum(xi)
        if abs(xs) > 1e-30:
            xi = xi * np.sqrt(L_f) / xs

        # Compute xi_hat at first 3 zeta zeros
        vals = []
        for k in range(3):
            if gammas[k] < np.pi * N / L_f:  # within bandwidth
                v = abs(xi_hat(gammas[k], xi, N, L_f))
                vals.append(v)
            else:
                vals.append(float('nan'))

        ratio = vals[0] / abs(eps_0) if abs(eps_0) > 1e-20 and not np.isnan(vals[0]) else float('nan')

        print(f"{lam_sq:>6} {eps_0:>12.4e} {vals[0]:>14.4e} {vals[1]:>14.4e} "
              f"{vals[2]:>14.4e} {ratio:>10.2f}")

    # Now with larger N
    print(f"\n\nWith N=30:")
    N = 30
    print(f"\n{'lam^2':>6} {'eps_0':>12} {'|xi_hat(g1)|':>14} {'|xi_hat(g2)|':>14} "
          f"{'|xi_hat(g3)|':>14} {'ratio1':>10}")
    print("-" * 75)

    for lam_sq in [14, 30, 50]:
        t0 = time.time()
        L_f = np.log(lam_sq)
        QW = build_QW(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi = evecs[:, 0]
        eps_0 = evals[0]

        xs = np.sum(xi)
        if abs(xs) > 1e-30:
            xi = xi * np.sqrt(L_f) / xs

        vals = []
        for k in range(min(5, len(gammas))):
            v = abs(xi_hat(gammas[k], xi, N, L_f))
            vals.append(v)

        ratio = vals[0] / abs(eps_0) if abs(eps_0) > 1e-20 else float('nan')
        print(f"{lam_sq:>6} {eps_0:>12.4e} {vals[0]:>14.4e} {vals[1]:>14.4e} "
              f"{vals[2]:>14.4e} {ratio:>10.2f}")

        # Show all 5 zeros
        print(f"         xi_hat at gamma_1..5: {', '.join(f'{v:.4e}' for v in vals)}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If |xi_hat(gamma_k)| ~ C * eps_0 (ratio bounded):
  => The proof skeleton's key estimate HOLDS
  => H_{lambda,N}(z) -> Xi(z) on compacts as lambda -> inf
  => By Rouche: all zeros of Xi are real
  => RH

The ratio |xi_hat(gamma_k)| / eps_0 should be BOUNDED (not growing)
as lambda -> infinity. If it is, the proof is essentially complete:
  H2 (spectral gap) + H3 (eps_0 -> 0) => convergence => RH
""")
