"""
MOMENT-SIGN CHANGE CONNECTION — Can zeta moments force all sign changes?

KNOWN THEOREMS (proved using Euler product):
  - 2nd moment: integral |zeta(1/2+it)|^2 dt ~ T*log(T)   [Hardy-Littlewood]
  - 4th moment: integral |zeta(1/2+it)|^4 dt ~ T*(log T)^4/(2*pi^2) [Ingham]

IDEA: These moments bound the AVERAGE size of |zeta| on the critical line.
Between consecutive zeros, |zeta| must rise from 0 to some maximum and
return to 0. The moments constrain this maximum.

If the maximum between zeros is bounded, then Xi MUST change sign at every
zero — because it can't "skip" a sign change without |Xi| growing too large.

This connects the Euler product (which determines the moments) to the
sign-change count (which determines the proportion on the critical line).
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, pi, zeta, log, exp, gamma
import time

mp.dps = 20


def hardy_Z(t):
    """Hardy Z-function: real-valued, |Z(t)| = |zeta(1/2+it)|."""
    try:
        return float(mpmath.siegelz(mpf(t)))
    except:
        return 0.0


def xi_function(z):
    """Xi(z) where s = 1/2 + iz."""
    z_mp = mpc(z)
    s = mpf('0.5') + mpc(0, 1) * z_mp
    try:
        return complex(mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s / 2) * gamma(s / 2) * zeta(s))
    except:
        return 0.0


if __name__ == "__main__":
    gammas = np.load("_zeros_500.npy")
    N = len(gammas)

    print("MOMENT-SIGN CHANGE CONNECTION")
    print("=" * 75)

    # ================================================================
    # PART 1: Second moment of zeta on the critical line
    # ================================================================
    print("\nPART 1: SECOND MOMENT OF ZETA ON THE CRITICAL LINE")
    print("-" * 75)
    print("integral_0^T |zeta(1/2+it)|^2 dt ~ T*log(T)  [Hardy-Littlewood]\n")

    for T_idx in [50, 100, 200]:
        T = gammas[min(T_idx, N-1)]
        # Numerical integration
        n_pts = 2000
        t_grid = np.linspace(1.0, T, n_pts)
        dt = t_grid[1] - t_grid[0]

        Z_vals = np.array([hardy_Z(t) for t in t_grid])
        moment2 = np.sum(Z_vals**2) * dt
        predicted = T * np.log(T)

        print(f"  T = {T:.2f} ({T_idx} zeros):")
        print(f"    Computed: {moment2:.4f}")
        print(f"    T*log(T): {predicted:.4f}")
        print(f"    Ratio:    {moment2/predicted:.4f}")

    # ================================================================
    # PART 2: |Z(t)| between consecutive zeros — the peak values
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 2: PEAK |Z(t)| BETWEEN CONSECUTIVE ZEROS")
    print("-" * 75)
    print("Between gamma_k and gamma_{k+1}, Z(t) rises from 0 to a peak.")
    print("The peak height determines how 'aggressively' Z changes sign.\n")

    N_test = min(200, N - 1)
    peaks = []
    peak_data = []

    for k in range(N_test):
        g1, g2 = gammas[k], gammas[k + 1]
        spacing = g2 - g1

        # Sample Z between zeros
        n_sample = max(20, int(spacing * 10))
        t_sample = np.linspace(g1 + spacing * 0.05, g2 - spacing * 0.05, n_sample)
        Z_sample = np.array([hardy_Z(t) for t in t_sample])

        peak = np.max(np.abs(Z_sample))
        peaks.append(peak)
        peak_data.append((k, g1, spacing, peak))

    peaks = np.array(peaks)

    print(f"  Statistics over first {N_test} inter-zero intervals:")
    print(f"    Mean |Z|_max:   {peaks.mean():.4f}")
    print(f"    Median |Z|_max: {np.median(peaks):.4f}")
    print(f"    Min |Z|_max:    {peaks.min():.6f}")
    print(f"    Max |Z|_max:    {peaks.max():.4f}")
    print(f"    Std:            {peaks.std():.4f}")

    # The smallest peaks are the most interesting — these are where
    # Z barely changes sign
    print(f"\n  10 smallest peaks (narrowest sign changes):")
    sorted_peaks = sorted(peak_data, key=lambda x: x[3])
    print(f"  {'k':>4} {'gamma_k':>10} {'spacing':>10} {'|Z|_max':>12} {'normalized':>12}")
    print("  " + "-" * 50)
    for k, g, sp, pk in sorted_peaks[:10]:
        # Normalize by sqrt(log(g)) which is the expected scale
        norm = pk / np.sqrt(np.log(g)) if g > 1 else pk
        print(f"  {k+1:>4} {g:>10.4f} {sp:>10.4f} {pk:>12.6f} {norm:>12.6f}")

    # ================================================================
    # PART 3: The sign-change argument via moments
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 3: THE SIGN-CHANGE ARGUMENT VIA MOMENTS")
    print("-" * 75)
    print("""
THE ARGUMENT:
  1. The 2nd moment gives: avg(|zeta|^2) ~ log(T)
  2. Between consecutive zeros, |Z(t)| rises to peak P_k, then falls back to 0
  3. The integral of |Z|^2 over interval [gamma_k, gamma_{k+1}] is ~ P_k^2 * delta_k / 3
     (treating the peak as a triangle)
  4. Summing: sum P_k^2 * delta_k / 3 ~ T * log(T)
  5. Average: <P_k^2 * delta_k> ~ T*log(T) / N(T) ~ 2*pi (using N(T) ~ T*logT/(2pi))
  6. So: <P_k^2> ~ 2*pi / <delta_k> ~ 2*pi * logT / (2*pi) = log(T)
  7. Typical peak: P_k ~ sqrt(log T)

KEY: If a zero is OFF the line, Z(t) does NOT change sign there.
The "missing" sign change means Z stays positive (or negative) over
a DOUBLE interval. The peak would be roughly DOUBLED, contributing
~4 times more to the moment integral.

The moment constraint: sum P_k^2 * delta_k = T*log(T) is FIXED.
If one interval has 4x the contribution, something else must be SMALLER.
This creates tension — the moments can't accommodate too many missing
sign changes.
""")

    # Compute the actual integral contribution from each interval
    contributions = []
    for k in range(min(100, N_test)):
        g1, g2 = gammas[k], gammas[k + 1]
        sp = g2 - g1
        contrib = peaks[k]**2 * sp / 3  # triangle approximation
        contributions.append(contrib)

    contributions = np.array(contributions)
    total = contributions.sum()
    mean_contrib = contributions.mean()

    print(f"  Actual interval contributions (triangle approx):")
    print(f"    Total (100 intervals): {total:.4f}")
    print(f"    Mean per interval:     {mean_contrib:.4f}")
    print(f"    Std:                   {contributions.std():.4f}")
    print(f"    Max/Mean ratio:        {contributions.max()/mean_contrib:.4f}")

    # What if one interval is doubled (missing sign change)?
    # The doubled interval would have contribution ~ 4 * mean_contrib
    # The new total would be: total - mean_contrib + 4*mean_contrib = total + 3*mean_contrib
    # But the moment is FIXED at T*log(T), so this excess must be compensated.
    print(f"\n  If one sign change is missing:")
    print(f"    Doubled contribution: {4 * mean_contrib:.4f}")
    print(f"    Excess: {3 * mean_contrib:.4f}")
    print(f"    Fraction of total: {3 * mean_contrib / total:.6f}")
    print(f"    One missing sign change is easily hidden in the variance.")
    print(f"    Need M missing sign changes where 3*M*mean > total*fraction")

    # ================================================================
    # PART 4: Higher moments for sharper constraints
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 4: FOURTH MOMENT — SHARPER CONSTRAINT")
    print("-" * 75)
    print("The 4th moment: integral |zeta|^4 dt ~ T*(logT)^4/(2*pi^2)")
    print("This constrains the PEAK distribution more tightly.\n")

    # Compute 4th moment contribution
    T_val = gammas[99]
    n_pts = 3000
    t_grid = np.linspace(1.0, T_val, n_pts)
    dt = t_grid[1] - t_grid[0]

    Z_all = np.array([hardy_Z(t) for t in t_grid])
    moment2_actual = np.sum(Z_all**2) * dt
    moment4_actual = np.sum(Z_all**4) * dt

    moment2_pred = T_val * np.log(T_val)
    moment4_pred = T_val * np.log(T_val)**4 / (2 * np.pi**2)

    print(f"  T = {T_val:.2f}:")
    print(f"  2nd moment: computed={moment2_actual:.2f}, "
          f"predicted={moment2_pred:.2f}, ratio={moment2_actual/moment2_pred:.4f}")
    print(f"  4th moment: computed={moment4_actual:.2f}, "
          f"predicted={moment4_pred:.2f}, ratio={moment4_actual/moment4_pred:.4f}")

    # The ratio moment4/moment2^2 measures the "peakiness"
    kurtosis = moment4_actual * T_val / moment2_actual**2
    print(f"  Kurtosis proxy (M4*T/M2^2): {kurtosis:.4f}")
    print(f"  For uniform: 1.0, for peaked: > 1.0")

    # ================================================================
    # PART 5: The QUANTITATIVE sign-change bound
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 5: QUANTITATIVE SIGN-CHANGE BOUND FROM MOMENTS")
    print("-" * 75)
    print("""
Let N_on = number of zeros on the critical line, N_off = off-line zeros.
N_on + N_off = N(T).  Sign changes = N_on.

Each on-line interval contributes ~P_k^2 * delta_k to the 2nd moment.
Each off-line gap (doubled interval) contributes ~4*P_k^2 * 2*delta_k.

The 2nd moment is FIXED at T*log(T). So:

  sum_{on-line} P_k^2*d_k + sum_{off-line} 8*P_k^2*d_k ~ T*logT

If all are on-line: N_on * <P^2*d> ~ T*logT -> <P^2*d> ~ 2*pi
If M are off-line: (N-M)*2*pi + M*8*2*pi ~ T*logT
  -> (N-M)*2*pi + 16*pi*M ~ T*logT
  -> N*2*pi + 14*pi*M ~ T*logT

But N*2*pi ~ T*logT, so: 14*pi*M ~ 0 -> M ~ 0!

Wait, that's too simple. The issue is that the "8x" factor for off-line
intervals assumes the peak doubles, which isn't exactly right.
""")

    # More careful computation: what happens to Z(t) in a double interval?
    # Take the interval [gamma_24, gamma_26] (skipping gamma_25)
    k_skip = 25
    g_prev = gammas[k_skip - 1]
    g_next = gammas[k_skip + 1]
    double_sp = g_next - g_prev

    # Compute |Z| over the double interval
    n_sample = 200
    t_sample = np.linspace(g_prev + 0.01, g_next - 0.01, n_sample)
    Z_double = np.array([hardy_Z(t) for t in t_sample])

    # Original two intervals
    g_mid = gammas[k_skip]
    Z_left = np.array([hardy_Z(t) for t in np.linspace(g_prev + 0.01, g_mid - 0.01, 100)])
    Z_right = np.array([hardy_Z(t) for t in np.linspace(g_mid + 0.01, g_next - 0.01, 100)])

    # The double interval has Z changing sign at g_mid (the "removed" zero)
    # In reality, if the zero is off-line, Z does NOT change sign there
    # Instead, Z stays on one side. Let's measure what that looks like.

    # |Z| at the "removed" zero location:
    Z_at_removed = hardy_Z(g_mid)
    Z_max_left = np.max(np.abs(Z_left))
    Z_max_right = np.max(np.abs(Z_right))
    Z_max_double = np.max(np.abs(Z_double))

    print(f"  Concrete example: skipping zero #{k_skip+1} at gamma = {g_mid:.4f}")
    print(f"  Left interval:  [{g_prev:.4f}, {g_mid:.4f}], |Z|_max = {Z_max_left:.4f}")
    print(f"  Right interval: [{g_mid:.4f}, {g_next:.4f}], |Z|_max = {Z_max_right:.4f}")
    print(f"  Double interval: [{g_prev:.4f}, {g_next:.4f}], |Z|_max = {Z_max_double:.4f}")
    print(f"  Z at removed zero: {Z_at_removed:.6f}")

    # Moment contributions
    m2_left = np.sum(Z_left**2) * (g_mid - g_prev) / len(Z_left)
    m2_right = np.sum(Z_right**2) * (g_next - g_mid) / len(Z_right)
    m2_double = np.sum(Z_double**2) * double_sp / len(Z_double)

    print(f"\n  2nd moment contributions:")
    print(f"    Left interval:  {m2_left:.4f}")
    print(f"    Right interval: {m2_right:.4f}")
    print(f"    Sum of two:     {m2_left + m2_right:.4f}")
    print(f"    Double interval:{m2_double:.4f}")
    print(f"    Ratio (double / sum of two): {m2_double / (m2_left + m2_right):.4f}")

    # ================================================================
    # PART 6: The Selberg moment bound
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 6: SELBERG'S BOUND AND THE SIGN-CHANGE PROPORTION")
    print("-" * 75)
    print("""
Selberg (1942) proved: N_0(T) > c*T*log(T) for some c > 0,
where N_0(T) = number of zeros on the critical line up to height T.

His method: compute integral of |Z(t)|^2 * w(t) for a weight function w
that is positive at zeros and negative between zeros. If the integral
is positive, there must be zeros on the critical line.

The Euler product enters through the Mellin transform of w,
which relates to the Dirichlet series coefficients.

KEY INSIGHT: The Euler product structure means that the mollified
moments have SPECIFIC values (determined by the prime distribution).
These values FORCE enough sign changes.

The gap from 40% to 100% requires:
  - Better mollifiers (longer Dirichlet polynomials)
  - Better mean-value theorems (bounding the mollified 2nd moment)
  - Or a completely different argument
""")

    # ================================================================
    # PART 7: A new approach — ENERGY of Z(t) between zeros
    # ================================================================
    print(f"{'='*75}")
    print("PART 7: THE ENERGY APPROACH — Z(t) AS A VIBRATING STRING")
    print("-" * 75)
    print("""
Think of Z(t) as a vibrating string with fixed endpoints (zeros).
Between consecutive zeros, Z(t) satisfies an equation like:
  Z''(t) + V(t)*Z(t) = 0
where V(t) comes from the functional equation.

The "energy" of each oscillation: E_k = integral |Z'(t)|^2 + V|Z|^2 dt
The total energy: sum E_k ~ function of T (constrained by moments).

If a zero is missing: the string vibrates over a LONGER interval,
which requires MORE energy for the same amplitude. But the total
energy is fixed -> the missing oscillation must steal from others.

This is the Sturm-Liouville perspective on sign changes.
""")

    # Compute Z'(t) at zeros — this gives the "spring constant"
    h = 0.001
    print(f"  Z'(t) at first 20 zeros (spring strength):")
    hdr_zd = "Z'(gamma_k)"
    print(f"  {'k':>4} {'gamma_k':>10} {hdr_zd:>14} {'|Z_peak|':>12} {'E_approx':>12}")
    print("  " + "-" * 55)

    z_derivs = []
    for k in range(min(20, N_test)):
        gk = gammas[k]
        Z_deriv = (hardy_Z(gk + h) - hardy_Z(gk - h)) / (2 * h)
        pk = peaks[k]
        sp = gammas[k + 1] - gammas[k] if k + 1 < N else 1.0
        E_approx = pk**2 * sp  # rough energy estimate
        z_derivs.append(abs(Z_deriv))

        print(f"  {k+1:>4} {gk:>10.4f} {Z_deriv:>14.6f} {pk:>12.6f} {E_approx:>12.4f}")

    # Correlation between Z' at zeros and peak heights
    z_derivs = np.array(z_derivs)
    peaks_20 = peaks[:20]
    corr = np.corrcoef(z_derivs, peaks_20)[0, 1]
    print(f"\n  Correlation(|Z'(zero)|, |Z|_peak): {corr:.4f}")
    print(f"  (Strong positive correlation means Z' at zeros predicts peak height)")

    # ================================================================
    # PART 8: SYNTHESIS — How moments constrain sign changes
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 8: SYNTHESIS")
    print("=" * 75)
    print(f"""
WHAT WE LEARNED:

1. PEAK HEIGHTS: The smallest peak between consecutive zeros is
   {peaks.min():.6f}. This is the "closest call" for a sign change.
   If this zero were off-line, Z would need to maintain this value
   across the gap without changing sign.

2. MOMENT CONSTRAINT: The 2nd moment integral is fixed at T*log(T).
   Each interval contributes P_k^2 * delta_k ~ 2*pi on average.
   A doubled interval (missing sign change) contributes ~{m2_double/(m2_left+m2_right):.2f}x more.

3. THE CRITICAL RATIO:
   If M zeros are off-line, the moment excess is ~M * extra_contribution.
   For this to be consistent with T*log(T), we need:
   M * extra / (T*logT) << 1

   From our computation: double/sum ratio = {m2_double/(m2_left+m2_right):.4f}
   This is NOT dramatically different from 1.0, meaning the moment
   constraint alone can't rule out a small number of off-line zeros.

4. WHAT'S NEEDED: The 2nd moment is too weak. We need:
   a. Higher moments (4th, 6th) which amplify the difference between
      single and double intervals
   b. TWISTED moments (involving a Dirichlet polynomial) which are
      more sensitive to the Euler product structure
   c. Or the Selberg-Conrey mollifier approach with better mean values

5. THE EULER PRODUCT CONNECTION:
   The moments are DETERMINED by the Euler product:
     integral |zeta|^2 dt ~ T*logT (from sum d(n)^2/n)
     integral |zeta|^4 dt ~ T*(logT)^4/(2*pi^2) (from sum d_3(n)^2/n)
   These are UNCONDITIONAL theorems. They constrain the sign-change
   distribution, but not tightly enough for 100%.

THE BOTTOM LINE: Moments give a SOFT constraint. The sign-change
proportion >40% (Conrey) comes from optimizing over mollifiers.
Reaching 100% requires either a HARD constraint or a completely
different proof strategy.
""")
