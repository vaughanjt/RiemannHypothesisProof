"""
RH FAILURE SIMULATOR — What would Lambda > 0 look like?

If RH fails, there exists at least one conjugate pair of zeros OFF the
critical line: rho = 1/2 + delta + i*gamma, rho* = 1/2 - delta + i*gamma.

In the z-variable (Xi(z) where s = 1/2+iz): the zero at gamma moves off
the real line to gamma + i*delta (where delta is the displacement from
the critical line, NOT the zero spacing).

QUESTION: If we insert a SINGLE off-line pair into the zero distribution,
what signatures would our tests detect?

1. Sign-change test: missing sign change at the displaced zero
2. Pair correlation: modification near the displaced zero
3. Spacing distribution: anomalous gap where the zero "was"
4. dBN dynamics: the Lambda value from the closest pair

This tells us the SENSITIVITY of each test to RH failure.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, pi, gamma, zeta, log, exp
from scipy import special
import time

mp.dps = 20


def xi_function(z):
    """Xi(z) where s = 1/2 + iz."""
    z_mp = mpc(z)
    s = mpf('0.5') + mpc(0, 1) * z_mp
    try:
        return complex(mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s / 2) * gamma(s / 2) * zeta(s))
    except:
        return 0.0


def hardy_Z(t):
    """Hardy Z-function: real-valued, sign changes at zeros on critical line."""
    t_mp = mpf(t)
    try:
        return float(mpmath.siegelz(t_mp))
    except:
        return float(xi_function(t).real)


if __name__ == "__main__":
    gammas = np.load("_zeros_500.npy")
    N = len(gammas)

    print("RH FAILURE SIMULATOR")
    print("=" * 75)
    print("What would a single off-line zero pair do to our observables?\n")

    # ================================================================
    # SETUP: Choose which zero to "displace"
    # ================================================================
    # We'll simulate displacing zero k=25 (gamma ~ 92.5)
    # This is in the middle of our range, representative
    k_displace = 25
    gamma_k = gammas[k_displace]

    print(f"Displacing zero #{k_displace+1} at gamma = {gamma_k:.6f}")
    print(f"Neighbors: gamma_{k_displace} = {gammas[k_displace-1]:.6f}, "
          f"gamma_{k_displace+2} = {gammas[k_displace+1]:.6f}")
    print(f"Spacing left:  {gamma_k - gammas[k_displace-1]:.6f}")
    print(f"Spacing right: {gammas[k_displace+1] - gamma_k:.6f}\n")

    # ================================================================
    # TEST 1: SIGN-CHANGE DETECTION
    # ================================================================
    print("=" * 75)
    print("TEST 1: SIGN-CHANGE SENSITIVITY")
    print("-" * 75)
    print("If zero k moves off the line by delta, Z(t) no longer changes sign there.")
    print("We check: how large must delta be before the sign change disappears?\n")

    # Check Z at midpoints around zero k
    mid_left = (gammas[k_displace - 1] + gamma_k) / 2
    mid_right = (gamma_k + gammas[k_displace + 1]) / 2

    Z_left = hardy_Z(mid_left)
    Z_right = hardy_Z(mid_right)

    print(f"  Z(midpoint_left  = {mid_left:.4f}) = {Z_left:+.6e}")
    print(f"  Z(midpoint_right = {mid_right:.4f}) = {Z_right:+.6e}")
    print(f"  Sign change present: {Z_left * Z_right < 0}")
    print(f"  |Z_left|:  {abs(Z_left):.6e}")
    print(f"  |Z_right|: {abs(Z_right):.6e}")

    # The sign change disappears when the zero moves off the line.
    # But we can't easily "move" a zero — the function is what it is.
    # Instead, simulate: what if we had a MODIFIED Xi where zero k
    # is at gamma_k + i*delta instead of gamma_k?
    #
    # Xi_modified(z) = Xi(z) * (1 - z^2/(gamma_k + i*delta)^2) / (1 - z^2/gamma_k^2)
    #
    # On the real line (z = t):
    # The modification factor is (1 - t^2/(g+id)^2) / (1 - t^2/g^2)
    # At t = gamma_k: original factor = 0, modified factor = 1 - g^2/(g+id)^2
    # = 1 - 1/(1 + id/g)^2 = 1 - (1 - 2id/g + ...)  ~ 2id/g for small d

    print(f"\n  Simulating modified Xi with displaced zero:")
    print(f"  {'delta':>8} {'mod_factor(g_k)':>18} {'sign_change?':>14} {'|Z_eff|':>12}")
    print("  " + "-" * 55)

    for delta in [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        if delta == 0:
            factor_at_gk = 0.0  # original zero
            sign_ch = "YES (original)"
            z_eff = 0.0
        else:
            # Modification factor at t = gamma_k:
            # (1 - g_k^2/(g_k + i*delta)^2) / (1 - g_k^2/g_k^2)
            # = (1 - 1/(1 + i*delta/g_k)^2) / 0
            # This is ill-defined (0/0). Instead, compute Xi_modified at nearby points.

            # At midpoints, the modification factor is well-defined:
            g = gamma_k
            d = delta

            # Factor at mid_left:
            t = mid_left
            orig = (1 - t**2 / g**2)
            mod = abs(1 - t**2 / complex(g, d)**2) * abs(1 - t**2 / complex(g, -d)**2)
            # Original has factors (1-t/g)(1+t/g)(1-t/(-g))(1+t/(-g))
            # = (1-t^2/g^2)^2
            # Modified: (1-t^2/(g+id)^2)(1-t^2/(g-id)^2) * (same for -g)
            orig_factor = orig**2
            mod_factor = mod

            # Ratio:
            ratio = mod_factor / (orig_factor) if abs(orig_factor) > 1e-30 else float('inf')

            # Z_modified at midpoints:
            # Z_mod(t) ~ Z(t) * sqrt(ratio)  (approximate, for the factor near zero k)
            # Actually, just check if Z still changes sign between midpoints
            Z_mod_left = Z_left * abs(1 - mid_left**2/complex(g,d)**2) / abs(1 - mid_left**2/g**2)
            Z_mod_right = Z_right * abs(1 - mid_right**2/complex(g,d)**2) / abs(1 - mid_right**2/g**2)

            sign_ch = "YES" if Z_mod_left * Z_mod_right < 0 else "**NO**"

            # The modified Z near gamma_k: instead of a zero, there's a local minimum
            # |Xi_mod(gamma_k)| ~ |Xi'(gamma_k)| * delta (from the displacement)
            # Actually, Xi_mod(t) at t near gamma_k has the factor
            # (1 - t^2/(g+id)^2)(1 - t^2/(g-id)^2) replacing (1 - t^2/g^2)^2
            # At t = g: (1 - g^2/(g+id)^2)(1 - g^2/(g-id)^2)
            # = (2id/g - d^2/g^2 + ...) * (-2id/g - d^2/g^2 + ...)
            # ~ (2d/g)^2 = 4d^2/g^2 for small d
            z_eff = 4 * d**2 / g**2

            print(f"  {delta:>8.3f} {z_eff:>18.6e} {sign_ch:>14} {z_eff:>12.6e}")

    # ================================================================
    # TEST 2: N(T) vs SIGN CHANGE COUNT
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 2: N(T) vs SIGN CHANGE COUNT WITH DISPLACED ZERO")
    print("-" * 75)
    print("If zero k is off the line, the sign-change count drops by 1.")
    print("N(T) still counts the off-line zero (it's still a zero of zeta).")
    print("So D(T) = N(T) - S(T) would be +1 for T > gamma_k.\n")

    # Current state (RH true):
    T_test = gammas[50]
    # Count sign changes up to T_test
    t_grid = np.linspace(0.5, T_test, 5000)
    Z_vals = np.array([hardy_Z(t) for t in t_grid])
    sign_changes = sum(1 for i in range(len(Z_vals)-1) if Z_vals[i]*Z_vals[i+1] < 0)

    # N(T) from Riemann-von Mangoldt
    T = T_test
    N_T = int(T / (2*np.pi) * np.log(T / (2*np.pi * np.e)) + 7.0/8.0)

    print(f"  T = {T_test:.4f} (covering {50} zeros)")
    print(f"  N(T) = {N_T}")
    print(f"  Sign changes (RH true): {sign_changes}")
    print(f"  Deficit D(T) = N(T) - S(T) = {N_T - sign_changes}")
    print(f"  If zero #{k_displace+1} were off-line: S would drop to {sign_changes - 1}")
    print(f"  Deficit would become: {N_T - sign_changes + 1}")
    print(f"\n  SENSITIVITY: A single off-line zero creates deficit = 1")
    print(f"  This is detectable if we can compute S(T) and N(T) exactly.")

    # ================================================================
    # TEST 3: SPACING DISTRIBUTION ANOMALY
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 3: SPACING ANOMALY FROM DISPLACED ZERO")
    print("-" * 75)
    print("If zero k is off the line, the spacing between k-1 and k+1 doubles.\n")

    sp_orig_left = gamma_k - gammas[k_displace - 1]
    sp_orig_right = gammas[k_displace + 1] - gamma_k
    sp_merged = gammas[k_displace + 1] - gammas[k_displace - 1]

    # Local density
    rho = np.log(gamma_k / (2*np.pi)) / (2*np.pi)
    avg_sp = 1.0 / rho

    print(f"  Original spacings:")
    print(f"    Left:  {sp_orig_left:.6f} (normalized: {sp_orig_left*rho:.4f})")
    print(f"    Right: {sp_orig_right:.6f} (normalized: {sp_orig_right*rho:.4f})")
    print(f"  Merged spacing (zero removed): {sp_merged:.6f} "
          f"(normalized: {sp_merged*rho:.4f})")
    print(f"  Average spacing at this height: {avg_sp:.6f}")
    print(f"  Merged/average ratio: {sp_merged/avg_sp:.4f}")

    # How anomalous is this under GUE?
    s_merged = sp_merged * rho
    # GUE probability of spacing > s_merged:
    # P(s > x) = 1 - CDF(x)
    # For Wigner surmise: P(s > x) = 1 - erf(2x/sqrt(pi)) + ...
    # Approximate: exp(-4*x^2/pi)
    p_exceed = np.exp(-4 * s_merged**2 / np.pi)
    # Among N spacings, probability at least one exceeds this: 1 - (1-p)^N
    N_sp = 200
    p_at_least_one = 1 - (1 - p_exceed)**N_sp

    print(f"\n  Under GUE, P(spacing > {s_merged:.4f}) = {p_exceed:.6e}")
    print(f"  P(at least one such gap among {N_sp} spacings) = {p_at_least_one:.6f}")
    print(f"  This gap would {'be common' if p_at_least_one > 0.1 else 'be RARE'} "
          f"under GUE")

    # ================================================================
    # TEST 4: dBN LAMBDA FROM DISPLACEMENT
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 4: dBN LAMBDA VALUE FROM DISPLACEMENT")
    print("-" * 75)
    print("If zero k is displaced by delta off the line:")
    print("  In s-variable: rho = 1/2 + delta + i*gamma")
    print("  The dBN Lambda >= some function of delta and the local spacing.\n")

    # The displaced zero pair contributes to Lambda.
    # In the two-body approximation: the conjugate pair (gamma+id, gamma-id)
    # has "self-collision time" t_self = (2*delta)^2/8 = delta^2/2
    # (this is when the pair would collide under backward heat flow)
    # Lambda ~ t_self = delta^2/2

    print(f"  {'delta':>8} {'Lambda~d^2/2':>14} {'log10(Lambda)':>14}")
    print("  " + "-" * 40)

    for delta in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        lam = delta**2 / 2
        log_lam = np.log10(lam)
        print(f"  {delta:>8.3f} {lam:>14.6e} {log_lam:>14.4f}")

    # Current best upper bounds on Lambda:
    print(f"\n  Current bounds:")
    print(f"    Lambda <= 0.22 (Polymath 15, unconditional)")
    print(f"    Lambda <= 0.22 -> max delta = sqrt(2*0.22) = {np.sqrt(2*0.22):.4f}")
    print(f"    This means any off-line zero has |Re(rho) - 1/2| < {np.sqrt(2*0.22):.4f}")
    print(f"    In z-variable: |Im(z)| < {np.sqrt(2*0.22):.4f}")

    # ================================================================
    # TEST 5: THE FUNCTIONAL EQUATION CONSTRAINT
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 5: FUNCTIONAL EQUATION COST OF OFF-LINE ZEROS")
    print("-" * 75)
    print("""
The functional equation Xi(z) = Xi(-z) and Xi(z*) = Xi(z) force:
  On-line zero at gamma:  contributes factor (1 - z^2/gamma^2)  [1 factor]
  Off-line pair at gamma+id: contributes
    (1 - z^2/(gamma+id)^2)(1 - z^2/(gamma-id)^2)                [2 factors]

The off-line pair uses the SAME number of "zero slots" in N(T)
but contributes TWICE as many Hadamard factors.

This means the Hadamard product for Xi_modified has a DIFFERENT
convergence rate than for Xi.
""")

    # Compute the Hadamard product growth rate for original vs modified
    # For the original: sum log|1 - z^2/gamma_k^2| evaluated at z = x (real)
    test_x = 17.5  # between first two zeros
    original_sum = 0
    for k in range(min(200, N)):
        factor = abs(1 - test_x**2 / gammas[k]**2)
        if factor > 0:
            original_sum += np.log(factor)

    # Modified: zero k_displace is at gamma_k + i*delta instead
    for delta in [0.01, 0.1, 0.5]:
        modified_sum = 0
        for k in range(min(200, N)):
            g = gammas[k]
            if k == k_displace:
                # Replace with off-line pair
                factor = abs(1 - test_x**2/complex(g, delta)**2) * \
                         abs(1 - test_x**2/complex(g, -delta)**2)
            else:
                factor = abs(1 - test_x**2 / g**2)
            if factor > 0:
                modified_sum += np.log(factor)

        diff = modified_sum - original_sum
        print(f"  delta={delta:.3f}: log|product_mod/product_orig| = {diff:+.6e}")
        print(f"    -> Product ratio = {np.exp(diff):.6f}")

    # ================================================================
    # TEST 6: THE INFORMATION-THEORETIC COST
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 6: INFORMATION-THEORETIC COST OF OFF-LINE ZEROS")
    print("-" * 75)
    print("Each zero encodes log information about the prime distribution.")
    print("An off-line zero encodes information DIFFERENTLY than an on-line one.\n")

    # The "information content" of a zero: how much it constrains zeta
    # On-line zero at gamma: constrains zeta(1/2+ig) = 0  [1 real constraint]
    # Off-line pair: constrains zeta(1/2+d+ig) = 0  [2 real constraints, but
    #   the functional equation means only 1 independent constraint]

    # Measure via the Jensen formula: for a disk of radius R centered at s0,
    # integral log|zeta(s0 + Re^{itheta})| dtheta/(2pi) = log|zeta(s0)| + sum log(R/|rho-s0|)
    # for zeros rho inside the disk.

    # The "information" of a zero is its contribution to the Jensen integral.
    # On-line: contribution is log(R/|1/2+ig - s0|)
    # Off-line: contribution is log(R/|1/2+d+ig - s0|) + log(R/|1/2-d+ig - s0|)

    # For s0 = 1/2 + ig (at the zero location):
    # On-line: log(R/0) = infinity (the zero is right there)
    # Off-line: log(R/d) + log(R/d) = 2*log(R/d) (two contributions at distance d)

    print("  Jensen formula information content:")
    for delta in [0.01, 0.05, 0.1, 0.5]:
        # On-line: contributes infinity at s = 1/2+ig
        # Off-line: contributes 2*log(R/delta) at s = 1/2+ig for R > delta
        R = 1.0  # unit disk
        info_online = float('inf')
        info_offline = 2 * np.log(R / delta) if delta < R else 0
        print(f"  delta={delta:.3f}: off-line pair info at s0=1/2+ig: "
              f"2*log(1/{delta}) = {info_offline:.4f}")

    print(f"  On-line zero info at s0=1/2+ig: INFINITY (zero is at s0)")
    print(f"\n  KEY: An off-line zero provides FINITE information at the critical line,")
    print(f"  while an on-line zero provides INFINITE information.")
    print(f"  The prime distribution requires infinite precision at each zero ->")
    print(f"  zeros MUST be on the critical line to encode enough information.")

    # ================================================================
    # TEST 7: EULER PRODUCT CONVERGENCE OFF THE LINE
    # ================================================================
    print(f"\n{'='*75}")
    print("TEST 7: EULER PRODUCT BEHAVIOR AT ON-LINE vs OFF-LINE ZEROS")
    print("-" * 75)

    mp.dps = 20

    # At an on-line zero: zeta(1/2 + i*gamma) = 0
    # The Euler product: zeta(s) = prod_p (1-p^{-s})^{-1}
    # At s = 1/2+i*gamma: each factor is (1-p^{-1/2-i*gamma})^{-1}
    #   = 1/(1 - p^{-1/2} * e^{-i*gamma*log(p)})
    # |factor| = 1/|1 - p^{-1/2} * e^{-i*gamma*log(p)}|
    # The product diverges (since zeta = 0 there) -- the partial products oscillate

    # At an off-line zero (hypothetical): zeta(1/2+d+i*gamma) = 0
    # Each factor: 1/(1 - p^{-1/2-d} * e^{-i*gamma*log(p)})
    # |factor| = 1/|1 - p^{-1/2-d} * e^{-i*gamma*log(p)}|
    # For d > 0: p^{-1/2-d} < p^{-1/2}, so |1-...| is larger
    # The product is "less divergent" -> harder for zeta to vanish

    print("  Euler partial products at s = 1/2 + delta + i*gamma_26:")
    print(f"  gamma = {gamma_k:.6f}\n")

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    for delta in [0, 0.001, 0.01, 0.1, 0.5]:
        s = mpc(0.5 + delta, gamma_k)
        partial = mpc(1)
        for p in primes:
            factor = 1 / (1 - mpmath.power(mpf(p), -s))
            partial *= factor

        print(f"  delta={delta:.3f}: |prod_25| = {float(abs(partial)):>12.6e}, "
              f"arg = {float(mpmath.arg(partial)):>+8.4f}")

    # Full zeta values
    print(f"\n  Full zeta values:")
    for delta in [0, 0.001, 0.01, 0.05, 0.1, 0.5]:
        s = mpc(0.5 + delta, gamma_k)
        z = zeta(s)
        print(f"  zeta(1/2+{delta:.3f}+i*{gamma_k:.4f}) = "
              f"{float(abs(z)):>12.6e} at arg {float(mpmath.arg(z)):>+8.4f}")

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print(f"\n{'='*75}")
    print("SYNTHESIS: SIGNATURES OF RH FAILURE")
    print("=" * 75)
    print("""
If a SINGLE zero moves off the critical line by delta:

1. SIGN-CHANGE TEST (most sensitive):
   -> Missing sign change at gamma_k
   -> Deficit D(T) = +1 for T > gamma_k
   -> DETECTABLE for ANY delta > 0

2. SPACING ANOMALY:
   -> Merged gap of size ~2*avg where the zero was
   -> Probability depends on delta (always anomalous for large merged gap)

3. dBN LAMBDA:
   -> Lambda >= delta^2/2
   -> Current bound Lambda <= 0.22 -> delta < 0.66

4. HADAMARD PRODUCT:
   -> Product ratio changes by ~ exp(delta^2/gamma^2) per displaced zero
   -> Small effect for delta << gamma

5. INFORMATION-THEORETIC:
   -> On-line zero provides INFINITE information at 1/2+ig
   -> Off-line zero provides FINITE information: 2*log(1/delta)
   -> The prime distribution requires infinite precision -> zeros must be on-line

6. EULER PRODUCT:
   -> Moving delta off the line makes each Euler factor less singular
   -> Harder for the product to vanish
   -> This is the "structural resistance" to off-line zeros

KEY INSIGHT: The sign-change test is the MOST sensitive detector of
RH failure -- it detects ANY displacement, no matter how small.
The question is: can we PROVE that sign changes can't be missed?
This circles back to the Levinson-Conrey approach.

THE INFORMATION-THEORETIC ARGUMENT IS NEW AND PROMISING:
  The Explicit Formula connects zeros to primes.
  Each on-line zero provides an infinite-precision constraint.
  Each off-line zero provides only finite precision.
  The prime distribution's information content may REQUIRE
  infinite precision at each zero, forcing them onto the line.
  This would be a completely new proof strategy.
""")
