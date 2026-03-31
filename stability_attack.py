"""
STABILITY ATTACK — Is y=0 a stable equilibrium for each zeta zero?

For zero gamma_k on the real line (in the z-variable where Xi(z) has zeros):

Destabilizing curvature from other zeros:
  kappa_repel = -sum_{j!=k} 1/(gamma_k - gamma_j)^2  (always < 0)

Stabilizing curvature from the Gamma/pi external potential:
  kappa_confine = d^2/dy^2 [log|Xi_smooth(x+iy)|] at y=0, x=gamma_k

NET curvature: kappa = kappa_confine + kappa_repel
If kappa > 0 for ALL k: the real-line configuration is stable => RH

This is NON-CIRCULAR: uses Gamma factor (known) + zero positions (computed).
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, pi, gamma, zeta, log, exp, nstr, diff
import time

mp.dps = 30


def xi_smooth(z):
    """The 'smooth' part of Xi: everything except zeta.
    Xi(z) = (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)*zeta(s) where s = 1/2+iz

    The smooth part (no zeros): (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)
    """
    s = mpf('0.5') + mpc(0, 1) * mpc(z)
    try:
        val = mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s/2) * gamma(s/2)
        return val
    except:
        return mpc(0)


def log_xi_smooth_modulus(x, y):
    """log|Xi_smooth(x + iy)| — the confining potential."""
    z = mpc(x, y)
    val = xi_smooth(z)
    if abs(val) > 0:
        return float(mpmath.log(abs(val)))
    return -1000.0


if __name__ == "__main__":
    print("STABILITY ANALYSIS: Is y=0 stable for each zero?")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # ================================================================
    # PART 1: Destabilizing curvature from zero repulsion
    # ================================================================
    print("\nPART 1: DESTABILIZING CURVATURE (zero repulsion)")
    print("-" * 70)

    N_use = 200  # use 200 zeros for the sum

    print(f"  {'k':>4} {'gamma_k':>12} {'kappa_repel':>14} {'sum 1/(g-g_j)^2':>16}")
    print("  " + "-" * 50)

    kappa_repels = []
    for k in range(min(20, len(gammas))):
        kappa = -sum(1.0 / (gammas[k] - gammas[j])**2
                     for j in range(min(N_use, len(gammas))) if j != k)
        kappa_repels.append(kappa)
        print(f"  {k+1:>4} {gammas[k]:>12.6f} {kappa:>14.6f} {-kappa:>16.6f}")

    # ================================================================
    # PART 2: Stabilizing curvature from Gamma factor
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: STABILIZING CURVATURE (Gamma confinement)")
    print("-" * 70)

    # Compute d^2/dy^2 [log|Xi_smooth(x+iy)|] at y=0 numerically
    dy = 0.001  # finite difference step

    print(f"  {'k':>4} {'gamma_k':>12} {'kappa_confine':>14} {'method':>10}")
    print("  " + "-" * 45)

    kappa_confines = []
    for k in range(min(20, len(gammas))):
        x = gammas[k]

        # Second derivative by finite differences
        f_plus = log_xi_smooth_modulus(x, dy)
        f_zero = log_xi_smooth_modulus(x, 0)
        f_minus = log_xi_smooth_modulus(x, -dy)

        kappa = (f_plus - 2*f_zero + f_minus) / dy**2
        kappa_confines.append(kappa)
        print(f"  {k+1:>4} {gammas[k]:>12.6f} {kappa:>14.6f} {'fin_diff':>10}")

    # ================================================================
    # PART 3: NET CURVATURE — the stability test
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: NET CURVATURE = CONFINEMENT + REPULSION")
    print("-" * 70)

    print(f"  {'k':>4} {'gamma_k':>12} {'confine':>12} {'repel':>12} "
          f"{'NET':>12} {'stable?':>8}")
    print("  " + "-" * 60)

    for k in range(min(20, len(kappa_confines), len(kappa_repels))):
        net = kappa_confines[k] + kappa_repels[k]
        stable = "YES" if net > 0 else "NO"
        print(f"  {k+1:>4} {gammas[k]:>12.6f} {kappa_confines[k]:>12.4f} "
              f"{kappa_repels[k]:>12.4f} {net:>12.4f} {stable:>8}")

    # ================================================================
    # PART 4: How does net curvature scale with k?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: NET CURVATURE vs ZERO INDEX k")
    print("-" * 70)

    print(f"  {'k':>4} {'gamma_k':>12} {'confine':>12} {'repel':>12} {'net':>12} {'net/repel':>10}")
    print("  " + "-" * 65)

    for k in [0, 4, 9, 19, 49, 99]:
        if k >= len(gammas): break
        x = gammas[k]
        kappa_r = -sum(1.0 / (gammas[k] - gammas[j])**2
                       for j in range(min(N_use, len(gammas))) if j != k)

        f_plus = log_xi_smooth_modulus(x, dy)
        f_zero = log_xi_smooth_modulus(x, 0)
        f_minus = log_xi_smooth_modulus(x, -dy)
        kappa_c = (f_plus - 2*f_zero + f_minus) / dy**2

        net = kappa_c + kappa_r
        ratio = net / kappa_r if abs(kappa_r) > 0 else 0
        print(f"  {k+1:>4} {gammas[k]:>12.6f} {kappa_c:>12.4f} {kappa_r:>12.4f} "
              f"{net:>12.4f} {ratio:>10.4f}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)
    print("""
If NET > 0 for ALL zeros: the real-line configuration is a LOCAL MINIMUM
of the energy landscape => zeros are stably on the critical line => RH.

If NET < 0 for some zero: that zero could potentially escape the real line
(but might be prevented by higher-order terms or global topology).

This is a NON-CIRCULAR test: it uses the Gamma factor (analytic, known)
and the zero positions (numerically computed, first 10^13 verified).

The PROOF would be: show kappa_confine > |kappa_repel| for all k.
This requires: (a) a lower bound on kappa_confine from Stirling's formula,
(b) an upper bound on |kappa_repel| from the zero spacing statistics.
""")
