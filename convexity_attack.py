"""
CONVEXITY ATTACK — Promote local stability to global.

We proved: eps=0 is a LOCAL minimum of the energy (infinite restoring force).
Now: is E(eps) CONVEX in eps for all eps > 0?

If yes: local min = global min => RH.

Energy when zero k is displaced to (gamma_k + i*eps, gamma_k - i*eps):

  E(eps) = E_pair(eps) + E_others(eps) + E_ext(eps)

  E_pair   = -log(2*eps)                            [conjugate pair interaction]
  E_others = -sum_{j!=k} log[(g_j - g_k)^2 + eps^2] [interaction with other zeros]
  E_ext    = U(g_k + i*eps) + U(g_k - i*eps)        [Gamma factor confinement]

Convexity test: d^2E/d(eps^2) > 0 for ALL eps > 0.

  d^2/d(eps^2) E_pair   = +1/eps^2              (always positive, always stabilizing)
  d^2/d(eps^2) E_others = sum_j [eps^2 - d_j^2] / [d_j^2 + eps^2]^2
                           where d_j = g_j - g_k  (sign-changing: destabilizing)
  d^2/d(eps^2) E_ext    = numerical (from Gamma factor, expect positive)

The race: does +1/eps^2 + E_ext'' beat E_others'' for ALL eps?
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, pi, gamma, log, exp, nstr
import time

mp.dps = 30


def xi_smooth(z):
    """Smooth part of Xi (no zeta zeros): (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)
    where s = 1/2 + iz.
    """
    z_mp = mpc(z)
    s = mpf('0.5') + mpc(0, 1) * z_mp
    try:
        return mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s / 2) * gamma(s / 2)
    except:
        return mpc(0)


def U_ext(x, y):
    """External potential: -log|Xi_smooth(x+iy)|."""
    z = mpc(x, y)
    val = xi_smooth(z)
    if abs(val) > 0:
        return -float(mpmath.log(abs(val)))
    return 1000.0


def E_pair(eps):
    """Conjugate pair self-interaction energy."""
    if eps <= 0:
        return float('inf')
    return -np.log(2 * eps)


def E_others(eps, gamma_k, gammas, k_idx):
    """Interaction energy with all other zeros (assumed on real line)."""
    total = 0.0
    for j, gj in enumerate(gammas):
        if j == k_idx:
            continue
        d = gj - gamma_k
        # -log[(d^2 + eps^2)] replaces -2*log|d| when zero k splits
        total -= np.log(d**2 + eps**2)
    return total


def E_ext_pair(eps, gamma_k):
    """External potential for the displaced pair."""
    return U_ext(gamma_k, eps) + U_ext(gamma_k, -eps)


def E_total(eps, gamma_k, gammas, k_idx):
    """Total energy of configuration with zero k displaced by eps."""
    return E_pair(eps) + E_others(eps, gamma_k, gammas, k_idx) + E_ext_pair(eps, gamma_k)


def d2_pair(eps):
    """Second derivative of pair energy: +1/eps^2."""
    return 1.0 / eps**2


def d2_others(eps, gamma_k, gammas, k_idx):
    """Analytical second derivative of other-zero interaction.

    d^2/d(eps^2) [-log(d^2 + eps^2)] = (eps^2 - d^2) / (d^2 + eps^2)^2

    Positive when eps > |d| (zero has moved PAST the neighbor),
    Negative when eps < |d| (zero is BETWEEN neighbors).
    """
    total = 0.0
    for j, gj in enumerate(gammas):
        if j == k_idx:
            continue
        d = gj - gamma_k
        d2 = d**2
        denom = (d2 + eps**2)**2
        total += (eps**2 - d2) / denom
    return total


def d2_ext_numerical(eps, gamma_k, h=0.001):
    """Numerical second derivative of external potential."""
    if eps < 2 * h:
        h = eps / 3.0
    f_plus = E_ext_pair(eps + h, gamma_k)
    f_zero = E_ext_pair(eps, gamma_k)
    f_minus = E_ext_pair(eps - h, gamma_k)
    return (f_plus - 2 * f_zero + f_minus) / h**2


if __name__ == "__main__":
    gammas = np.load("_zeros_500.npy")
    N_zeros = min(200, len(gammas))

    print("CONVEXITY ATTACK: Is E(eps) globally convex?")
    print("=" * 75)

    # ================================================================
    # TEST 1: Energy landscape E(eps) for first few zeros
    # ================================================================
    print("\nTEST 1: ENERGY LANDSCAPE E(eps) for selected zeros")
    print("-" * 75)

    eps_values = np.concatenate([
        np.linspace(0.01, 0.5, 50),
        np.linspace(0.5, 5.0, 50),
        np.linspace(5.0, 20.0, 30),
    ])

    for k in [0, 1, 4, 9, 19]:
        if k >= len(gammas):
            break
        gk = gammas[k]
        print(f"\n  Zero k={k+1}, gamma_k = {gk:.6f}")
        print(f"  {'eps':>8} {'E_pair':>12} {'E_others':>12} {'E_ext':>12} {'E_total':>12}")
        print("  " + "-" * 52)

        e_vals = []
        for eps in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            ep = E_pair(eps)
            eo = E_others(eps, gk, gammas[:N_zeros], k)
            ee = E_ext_pair(eps, gk)
            et = ep + eo + ee
            e_vals.append(et)
            print(f"  {eps:>8.3f} {ep:>12.4f} {eo:>12.4f} {ee:>12.4f} {et:>12.4f}")

    # ================================================================
    # TEST 2: Second derivative decomposition
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 2: SECOND DERIVATIVE d^2E/d(eps^2) DECOMPOSITION")
    print("-" * 75)
    print("Convexity requires d^2E/d(eps^2) > 0 for ALL eps > 0.")

    for k in [0, 1, 4, 9, 19]:
        if k >= len(gammas):
            break
        gk = gammas[k]

        # Find the nearest neighbor distance (critical scale)
        dists = sorted([abs(gammas[j] - gk) for j in range(N_zeros) if j != k])
        d_min = dists[0]

        print(f"\n  Zero k={k+1}, gamma_k = {gk:.6f}, nearest neighbor dist = {d_min:.4f}")
        print(f"  {'eps':>8} {'d2_pair':>12} {'d2_others':>12} {'d2_ext':>12} "
              f"{'d2_total':>12} {'convex?':>8}")
        print("  " + "-" * 62)

        min_d2 = float('inf')
        min_eps = 0

        for eps in np.concatenate([
            np.linspace(0.01, d_min, 30),
            np.linspace(d_min, 3 * d_min, 30),
            np.linspace(3 * d_min, 10 * d_min, 20),
            np.linspace(10 * d_min, 50.0, 20),
        ]):
            dp = d2_pair(eps)
            do = d2_others(eps, gk, gammas[:N_zeros], k)
            de = d2_ext_numerical(eps, gk)
            dt = dp + do + de

            if dt < min_d2:
                min_d2 = dt
                min_eps = eps

        # Print at selected eps values
        for eps in [0.01, 0.1, d_min / 2, d_min, 2 * d_min, 5.0, 10.0, 20.0]:
            dp = d2_pair(eps)
            do = d2_others(eps, gk, gammas[:N_zeros], k)
            de = d2_ext_numerical(eps, gk)
            dt = dp + do + de
            cvx = "YES" if dt > 0 else "**NO**"
            print(f"  {eps:>8.3f} {dp:>12.4f} {do:>12.4f} {de:>12.4f} "
                  f"{dt:>12.4f} {cvx:>8}")

        print(f"  >>> Min d2_total = {min_d2:.6f} at eps = {min_eps:.4f}", end="")
        if min_d2 > 0:
            print("  [CONVEX]")
        else:
            print("  [NOT CONVEX — local min is NOT global]")

    # ================================================================
    # TEST 3: Critical eps where d2_others is most negative
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 3: WORST-CASE ANALYSIS — Where is convexity most threatened?")
    print("-" * 75)
    print("d2_others is most negative when eps is near a neighbor distance d_j.")
    print("At eps = d_j/sqrt(3): the contribution from zero j is most negative.\n")

    for k in [0, 1, 4, 9, 19, 49, 99]:
        if k >= N_zeros:
            break
        gk = gammas[k]
        dists = sorted([abs(gammas[j] - gk) for j in range(N_zeros) if j != k])

        # Scan for the minimum d2_total
        min_d2 = float('inf')
        min_eps = 0
        scan_eps = np.concatenate([
            np.linspace(0.005, 1.0, 200),
            np.linspace(1.0, 10.0, 100),
            np.linspace(10.0, max(50.0, dists[-1]), 100),
        ])

        for eps in scan_eps:
            dp = d2_pair(eps)
            do = d2_others(eps, gk, gammas[:N_zeros], k)
            de = d2_ext_numerical(eps, gk, h=max(0.0001, eps * 0.01))
            dt = dp + do + de

            if dt < min_d2:
                min_d2 = dt
                min_eps = eps

        status = "CONVEX" if min_d2 > 0 else "NOT CONVEX"
        print(f"  k={k+1:>3}, gamma={gk:>10.4f}, "
              f"min(d2) = {min_d2:>12.6f} at eps={min_eps:>8.4f}  [{status}]")

    # ================================================================
    # TEST 4: Analytical bound — can we prove convexity?
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 4: ANALYTICAL BOUND ATTEMPT")
    print("-" * 75)
    print("""
The worst case for d2_others comes from the nearest neighbor.
For zero k with nearest neighbor at distance d:

  d2_others >= -(1/4) * N / d^4   (crude upper bound on destabilizing curvature)

For convexity, we need:
  1/eps^2 + d2_ext > |d2_others|

The 1/eps^2 term dominates for eps < d/sqrt(N) approximately.
For eps ~ d: need d2_ext to carry the load.

Let's check if the Gamma confinement grows fast enough:
""")

    for k in [0, 9, 49, 99]:
        if k >= N_zeros:
            break
        gk = gammas[k]

        # Compute d2_ext at various eps
        print(f"  k={k+1}, gamma_k = {gk:.4f}")
        print(f"    {'eps':>8} {'d2_ext':>12} {'Stirling':>12} {'ratio':>8}")
        for eps in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            de = d2_ext_numerical(eps, gk, h=max(0.0001, eps * 0.01))
            # Stirling approximation for large gamma:
            # d^2/dy^2 log|Gamma((1/4-y/2) + i*gamma/2)| ~ pi/4 (from Stirling)
            stirling = np.pi / 4 + 0.5 / (0.25 + (gk / 2)**2)  # leading Stirling + correction
            ratio = de / stirling if stirling > 0 else 0
            print(f"    {eps:>8.2f} {de:>12.6f} {stirling:>12.6f} {ratio:>8.4f}")
        print()

    # ================================================================
    # TEST 5: The critical question — asymptotic analysis
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 5: ASYMPTOTIC ANALYSIS — Does convexity hold for ALL k?")
    print("-" * 75)
    print("""
For large k (gamma_k -> infinity):
  - Zero density ~ (1/2pi) * log(gamma_k/(2pi))
  - Average spacing ~ 2*pi / log(gamma_k/(2pi))
  - d2_others ~ -density^2 * (pi^2/3)   [from sum 1/n^2 with density n/d]
  - d2_ext ~ pi/4  (Stirling, independent of gamma_k)

The race:
  d2_pair = 1/eps^2  (wins for small eps)
  d2_ext ~ pi/4      (constant)
  d2_others ~ -(log gamma)^2 / (4*pi^2) * something  (grows with k)

If d2_others grows faster than d2_ext: convexity eventually FAILS.
Let's check numerically.
""")

    print(f"  {'k':>5} {'gamma':>10} {'density':>10} {'min_d2':>12} {'status':>12}")
    print("  " + "-" * 50)

    for k in range(0, N_zeros, 10):
        gk = gammas[k]
        density = np.log(gk / (2 * np.pi)) / (2 * np.pi) if gk > 2 * np.pi else 0.1

        # Quick scan for min d2_total
        min_d2 = float('inf')
        scan = np.concatenate([
            np.linspace(0.01, 2.0, 100),
            np.linspace(2.0, 20.0, 50),
        ])
        for eps in scan:
            dp = d2_pair(eps)
            do = d2_others(eps, gk, gammas[:N_zeros], k)
            de = d2_ext_numerical(eps, gk, h=max(0.0001, eps * 0.01))
            dt = dp + do + de
            if dt < min_d2:
                min_d2 = dt

        status = "CONVEX" if min_d2 > 0 else "BROKEN"
        print(f"  {k+1:>5} {gk:>10.4f} {density:>10.4f} {min_d2:>12.6f} {status:>12}")

    print(f"\n{'='*75}")
    print("VERDICT")
    print("=" * 75)
