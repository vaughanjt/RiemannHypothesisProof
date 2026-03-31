"""
CODIMENSION GAP SCALING — How close does Xi come to zero off the real line?

For each zero gamma_k, compute min|Xi(gamma_k + iy)| for 0.1 <= |y| <= 2.
This is the "RH safety margin" — how far Xi is from having an off-line zero.

If this margin shrinks to 0 as gamma -> inf, RH becomes increasingly fragile.
If it's bounded below, RH has breathing room.

FAST VERSION: Use hardy_Z / mpmath at a few y values per zero, every 5th zero.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, pi, gamma, zeta, log, exp
import time

mp.dps = 15  # Low precision for speed — we need trends, not exact values


def xi_at(x, y):
    """Compute |Xi(x+iy)| quickly."""
    z = mpc(x, y)
    s = mpf('0.5') + mpc(0, 1) * z
    try:
        val = mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s/2) * gamma(s/2) * zeta(s)
        return float(abs(val))
    except:
        return 1e10


if __name__ == "__main__":
    gammas = np.load("_zeros_500.npy")
    N = len(gammas)

    print("CODIMENSION GAP SCALING — RH Safety Margin vs Height")
    print("=" * 75)

    # ================================================================
    # PART 1: Minimum |Xi| off the real line near each zero
    # ================================================================
    print("\nPART 1: min|Xi(gamma_k + iy)| for |y| in {0.1, 0.5, 1.0}")
    print("-" * 75)
    print(f"  {'k':>4} {'gamma_k':>10} {'|Xi(+0.1i)|':>14} {'|Xi(+0.5i)|':>14} "
          f"{'|Xi(+1.0i)|':>14} {'min':>14}")
    print("  " + "-" * 72)

    gaps_01 = []  # |Xi| at y=0.1
    gaps_05 = []  # |Xi| at y=0.5
    gaps_10 = []  # |Xi| at y=1.0
    gap_min = []  # minimum across y values
    gap_gammas = []

    t0 = time.time()

    for k in range(0, N, 5):  # Every 5th zero
        gk = gammas[k]

        v01 = xi_at(gk, 0.1)
        v05 = xi_at(gk, 0.5)
        v10 = xi_at(gk, 1.0)
        vmin = min(v01, v05, v10)

        gaps_01.append(v01)
        gaps_05.append(v05)
        gaps_10.append(v10)
        gap_min.append(vmin)
        gap_gammas.append(gk)

        if k < 30 or k % 50 == 0:
            print(f"  {k+1:>4} {gk:>10.4f} {v01:>14.6e} {v05:>14.6e} "
                  f"{v10:>14.6e} {vmin:>14.6e}")

    dt = time.time() - t0
    print(f"\n  Computed {len(gap_gammas)} zeros in {dt:.1f}s")

    gaps_01 = np.array(gaps_01)
    gaps_05 = np.array(gaps_05)
    gaps_10 = np.array(gaps_10)
    gap_min = np.array(gap_min)
    gap_gammas = np.array(gap_gammas)

    # ================================================================
    # PART 2: Scaling analysis
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 2: SCALING OF |Xi(gamma_k + 0.1*i)| WITH HEIGHT")
    print("-" * 75)

    # The key question: does |Xi(gamma+0.1i)| grow, shrink, or stay constant?
    # If Xi has a zero at gamma (on the line), then near the zero:
    # Xi(gamma + iy) ~ Xi'(gamma) * iy = |Xi'(gamma)| * |y|
    # So |Xi(gamma + 0.1i)| ~ 0.1 * |Xi'(gamma)|
    # And |Xi'(gamma)| depends on gamma.

    # For the Z function: |Z(t)| = |zeta(1/2+it)|
    # Near a zero: Z(t) ~ Z'(gamma_k) * (t - gamma_k)
    # So |Xi(gamma_k + iy)| is related to |Z'| * |y| * (Gamma factor correction)

    # The Gamma factor at s = 1/2 + i*(gamma+iy) = 1/2 - y + i*gamma
    # |Gamma(s/2)| ~ |gamma/2|^{Re(s/2)-1/2} * exp(-pi*gamma/4)  [Stirling]
    # The exponential decay means |Xi| DECREASES with gamma at fixed y.

    # Log scale analysis
    log_gaps = np.log10(gaps_01)
    log_gammas = np.log10(gap_gammas)

    # Fit: log|Xi| = a + b * log(gamma) + c * gamma
    # The dominant term should be exponential (from Gamma factor)
    # log|Xi| ~ -pi*gamma/4 + polynomial corrections

    # Linear fit in gamma (expecting negative slope from exponential decay)
    valid = np.isfinite(log_gaps)
    if valid.sum() > 5:
        coeffs = np.polyfit(gap_gammas[valid], log_gaps[valid], 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        print(f"  Linear fit: log10|Xi(g+0.1i)| = {slope:.6f}*gamma + {intercept:.4f}")
        print(f"  Slope = {slope:.6f} per unit gamma")
        print(f"  Decay rate = {-slope * np.log(10):.6f} in natural units")
        print(f"  Expected (Stirling): -pi/(4*ln10) = {-np.pi/(4*np.log(10)):.6f}")
        print(f"  Ratio actual/Stirling: {slope / (-np.pi/(4*np.log(10))):.4f}")

    # ================================================================
    # PART 3: NORMALIZED gap — remove the Gamma factor decay
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 3: NORMALIZED GAP (remove Gamma factor decay)")
    print("-" * 75)
    print("The raw |Xi| decays exponentially due to the Gamma factor.")
    print("Normalize by the expected Gamma factor to see the INTRINSIC gap.\n")

    # Xi(gamma+iy) = (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)*zeta(s)
    # where s = 1/2 - y + i*gamma
    # The "smooth part" (Gamma factor) decays as exp(-pi*gamma/4)
    # The "zero part" (zeta) oscillates and vanishes at zeros
    # Near a zero: zeta(s) ~ zeta'(rho) * (s - rho) = zeta'(rho) * (-y + 0*i)
    # (for displacement purely in y at fixed gamma)

    # Normalize by the smooth envelope:
    # |Xi_smooth(gamma+iy)| = |(1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)|
    # This is the value Xi would have if zeta were replaced by 1.

    def xi_smooth_at(x, y):
        """Smooth part of Xi (everything except zeta)."""
        z = mpc(x, y)
        s = mpf('0.5') + mpc(0, 1) * z
        try:
            val = mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s/2) * gamma(s/2)
            return float(abs(val))
        except:
            return 1e10

    # Compute normalized gap for a subset
    print(f"  {'k':>4} {'gamma':>10} {'|Xi(+0.1i)|':>14} {'|smooth|':>14} "
          f"{'ratio':>12} {'~|zeta|':>12}")
    print("  " + "-" * 68)

    ratios = []
    ratio_gammas = []

    for idx in range(0, len(gap_gammas), max(1, len(gap_gammas)//30)):
        gk = gap_gammas[idx]
        xi_val = gaps_01[idx]
        sm_val = xi_smooth_at(gk, 0.1)
        ratio = xi_val / sm_val if sm_val > 1e-300 else 0
        ratios.append(ratio)
        ratio_gammas.append(gk)

        if idx < 10 or idx % 10 == 0:
            # The ratio ~ |zeta(1/2 - 0.1 + i*gamma)| * 0.1 (approximately)
            print(f"  {int(idx*5)+1:>4} {gk:>10.4f} {xi_val:>14.6e} {sm_val:>14.6e} "
                  f"{ratio:>12.6e} {ratio:>12.6e}")

    ratios = np.array(ratios)
    ratio_gammas = np.array(ratio_gammas)

    # The normalized ratio should tell us about |zeta(0.4 + i*gamma)|
    # (zeta slightly off the critical line at each zero height)
    print(f"\n  Normalized gap statistics:")
    print(f"    Mean: {ratios.mean():.6e}")
    print(f"    Std:  {ratios.std():.6e}")
    print(f"    Min:  {ratios.min():.6e}")
    print(f"    Max:  {ratios.max():.6e}")

    # Trend
    if len(ratio_gammas) > 5:
        coeffs_r = np.polyfit(ratio_gammas, np.log10(ratios + 1e-30), 1)
        print(f"    Trend: log10(ratio) = {coeffs_r[0]:.6f}*gamma + {coeffs_r[1]:.4f}")
        print(f"    Slope: {coeffs_r[0]:.6f} per unit gamma")
        print(f"    {'GROWING' if coeffs_r[0] > 0.001 else 'SHRINKING' if coeffs_r[0] < -0.001 else 'STABLE'} "
              f"with height")

    # ================================================================
    # PART 4: The tightest point — where is RH most fragile?
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 4: WHERE IS RH MOST FRAGILE?")
    print("-" * 75)

    # Find the zero with smallest normalized gap
    min_idx = np.argmin(ratios)
    print(f"  Tightest point: zero near gamma = {ratio_gammas[min_idx]:.4f}")
    print(f"  Normalized gap: {ratios[min_idx]:.6e}")
    print(f"  This means |zeta(0.4 + i*{ratio_gammas[min_idx]:.0f})| ~ {ratios[min_idx]:.6e}")

    # Also check: where is the RAW gap smallest?
    min_raw_idx = np.argmin(gap_min)
    print(f"\n  Smallest raw |Xi| off-line: zero at gamma = {gap_gammas[min_raw_idx]:.4f}")
    print(f"  |Xi(gamma + 0.1i)| = {gaps_01[min_raw_idx]:.6e}")

    # ================================================================
    # PART 5: The |zeta(sigma+it)| profile at zero heights
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 5: |zeta(sigma+it)| AT ZERO HEIGHTS — THE RH MARGIN")
    print("-" * 75)
    print("At each zero height t = gamma_k, compute |zeta(sigma + i*gamma_k)|")
    print("for sigma slightly off 1/2. This gives the direct RH margin.\n")

    mp.dps = 15

    print(f"  {'k':>4} {'gamma':>10} {'|z(0.501)|':>12} {'|z(0.51)|':>12} "
          f"{'|z(0.6)|':>12} {'dlog/dsig':>12}")
    print("  " + "-" * 62)

    margins = []
    margin_gammas = []

    for k in range(0, N, 10):
        gk = gammas[k]
        s1 = mpc(0.501, gk)
        s2 = mpc(0.51, gk)
        s3 = mpc(0.6, gk)

        try:
            z1 = float(abs(zeta(s1)))
            z2 = float(abs(zeta(s2)))
            z3 = float(abs(zeta(s3)))

            # Rate of growth: d(log|zeta|)/d(sigma) at sigma=1/2
            # Approximate: (log|z(0.51)| - log|z(0.501)|) / 0.009
            if z1 > 0 and z2 > 0:
                dlog = (np.log(z2) - np.log(z1)) / 0.009
            else:
                dlog = 0

            margins.append(z1)
            margin_gammas.append(gk)

            if k < 40 or k % 50 == 0:
                print(f"  {k+1:>4} {gk:>10.4f} {z1:>12.6e} {z2:>12.6e} "
                      f"{z3:>12.6e} {dlog:>12.4f}")
        except:
            pass

    margins = np.array(margins)
    margin_gammas = np.array(margin_gammas)

    print(f"\n  |zeta(0.501+i*gamma)| statistics:")
    print(f"    Mean:   {margins.mean():.6e}")
    print(f"    Median: {np.median(margins):.6e}")
    print(f"    Min:    {margins.min():.6e}")
    print(f"    Max:    {margins.max():.6e}")

    # Trend
    if len(margins) > 5:
        log_margins = np.log10(margins + 1e-30)
        coeffs_m = np.polyfit(margin_gammas, log_margins, 1)
        print(f"    Trend: log10|zeta(0.501+ig)| = {coeffs_m[0]:.6f}*gamma + {coeffs_m[1]:.4f}")
        print(f"    {'GROWING' if coeffs_m[0] > 0.001 else 'SHRINKING' if coeffs_m[0] < -0.001 else 'STABLE'}")

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print(f"\n{'='*75}")
    print("SYNTHESIS: THE RH MARGIN AS A FUNCTION OF HEIGHT")
    print("=" * 75)
    print(f"""
RESULTS:

1. RAW |Xi(gamma+0.1i)|: Decays exponentially with gamma (Gamma factor).
   Slope: {slope:.6f} per unit gamma (vs Stirling prediction {-np.pi/(4*np.log(10)):.6f}).
   The raw gap gets SMALLER with height — expected from Gamma decay.

2. NORMALIZED gap (remove Gamma factor):
   The ratio |Xi|/|Xi_smooth| measures |zeta| slightly off the critical line.
   This tells us how "hard" zeta works to be zero at each zero height.

3. |zeta(0.501 + i*gamma)|:
   The direct RH margin — how big is zeta just 0.001 off the critical line?
   If this grows with gamma: RH becomes MORE stable at higher zeros.
   If this shrinks: RH becomes MORE fragile.

The trend of |zeta(0.501+ig)| determines the DIFFICULTY of RH at height gamma:
  - Growing: the zero is "sharper" -> harder to move off-line
  - Stable: constant difficulty
  - Shrinking: the zero is "flatter" -> easier to move off-line

This connects to the LINDELOF HYPOTHESIS:
  |zeta(1/2+it)| << t^epsilon for all epsilon > 0

If Lindelof holds, then |zeta(0.501+it)| is also bounded polynomially,
meaning the RH margin doesn't collapse exponentially.
Lindelof is WEAKER than RH but still unproven.
""")
