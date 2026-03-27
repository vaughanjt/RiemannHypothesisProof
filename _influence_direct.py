"""Direct influence function test: does z_G * z_A < 2r hold?

The regression deficit R < 1 is SUFFICIENT but not NECESSARY for delta_r < 0.
The actual condition is weaker: z_G * z_A < 2r.

For beta near 1/2: A -> 0, z_A < 0, z_G > 0, product < 0 << 2r. Trivially satisfied.
For beta = 1 (worst case): A = |zeta'|/4, z_A might be positive, but the condition
z_G * z_A < 2r is still much easier than R < 1.

ALSO: test a stronger formulation — what is the actual delta_r for EACH zero?
This bypasses both the deficit and the influence function approximation.
"""
import numpy as np
from scipy.stats import pearsonr, linregress
import mpmath
mpmath.mp.dps = 25
import time

t0 = time.time()

def hardy_Z(t):
    return float(mpmath.siegelz(t))

def zeta_deriv_at_zero(gamma):
    s = mpmath.mpc(0.5, gamma)
    return float(abs(mpmath.diff(mpmath.zeta, s)))

def approx_n_for_T(T):
    return int(T / (2*np.pi) * np.log(T / (2*np.pi*np.e))) + 1

# Use windows at different heights
windows = [
    ("T~400", 50, 250),
    ("T~1500", 500, 700),
    ("T~5000", 2000, 2200),
    ("T~10000", 4500, 4700),
    ("T~20000", 10000, 10200),
    ("T~50000", 28000, 28200),
]

print("="*80)
print("DIRECT INFLUENCE FUNCTION TEST: z_G * z_A < 2r")
print("="*80)

for label, n_start, n_end in windows:
    t_start = time.time()

    # Compute zeros
    zeros = np.array([float(mpmath.im(mpmath.zetazero(n)))
                       for n in range(n_start, n_end + 1)])
    T_mid = np.mean(zeros)

    # Gaps, midpoints, peaks
    gaps = np.diff(zeros)
    mids = (zeros[:-1] + zeros[1:]) / 2
    peaks = np.array([abs(hardy_Z(m)) for m in mids])

    # Trim edges
    trim = int(0.1 * len(gaps))
    g_core = gaps[trim:-trim]
    p_core = peaks[trim:-trim]
    z_core = zeros[trim:-trim-1]

    # Statistics
    r_val = pearsonr(g_core, p_core)[0]
    g_bar = np.mean(g_core)
    s_g = np.std(g_core, ddof=1)
    P_bar = np.mean(p_core)
    s_P = np.std(p_core, ddof=1)
    M = len(g_core)

    print(f"\n{'='*60}")
    print(f"  {label} (T ~ {T_mid:.0f}): r = {r_val:+.4f}, M = {M}")
    print(f"  g_bar = {g_bar:.4f}, s_g = {s_g:.4f}, P_bar = {P_bar:.4f}, s_P = {s_P:.4f}")
    print(f"  Threshold: z_G * z_A < 2r = {2*r_val:.4f}")
    print(f"{'='*60}")

    # For each interior zero, compute z_G * z_A at beta = 1
    products = []
    z_Gs = []
    z_As = []
    actual_deltas = []

    print(f"\n  Computing influence for {len(z_core)-2} zeros...", flush=True)

    for i in range(1, len(z_core) - 1):
        gamma = z_core[i]

        # Merged gap
        G = g_core[i-1] + g_core[i]
        z_G = (G - g_bar) / s_g

        # Phantom amplitude at beta = 1
        zp = zeta_deriv_at_zero(gamma)
        A = zp / 4  # worst case beta = 1
        z_A = (A - P_bar) / s_P

        product = z_G * z_A
        products.append(product)
        z_Gs.append(z_G)
        z_As.append(z_A)

        # ACTUAL delta_r: remove this zero, recompute r
        # Remove zero at index (trim + i) from the core sequence
        # The merged gap replaces g[i-1] and g[i] with G
        # The merged peak is at the midpoint of the new gap
        # Use original Z value (no modification factor, which underestimates the effect)
        g_mod = np.concatenate([g_core[:i-1], [G], g_core[i+1:]])
        mid_merged = (z_core[i-1] + z_core[i+1]) / 2
        P_merged = abs(hardy_Z(mid_merged))
        p_mod = np.concatenate([p_core[:i-1], [P_merged], p_core[i+1:]])

        r_mod = pearsonr(g_mod, p_mod)[0]
        actual_deltas.append(r_mod - r_val)

    products = np.array(products)
    z_Gs = np.array(z_Gs)
    z_As = np.array(z_As)
    actual_deltas = np.array(actual_deltas)

    # Influence function prediction
    predicted_deltas = (products - 2*r_val) / (M - 1)

    # Test 1: z_G * z_A < 2r (influence function condition)
    influence_violations = np.sum(products >= 2*r_val)
    print(f"\n  TEST 1: z_G * z_A < 2r")
    print(f"    max(z_G * z_A) = {np.max(products):.4f} vs 2r = {2*r_val:.4f}")
    print(f"    Violations: {influence_violations}/{len(products)}")
    print(f"    min product = {np.min(products):.4f}, mean = {np.mean(products):.4f}")

    # The z_A values tell us: is the phantom above or below the mean peak?
    print(f"\n  z_A stats (phantom vs mean peak at beta=1):")
    print(f"    z_A > 0 (phantom > P_bar): {np.sum(z_As > 0)}/{len(z_As)}")
    print(f"    max z_A = {np.max(z_As):.4f}, mean z_A = {np.mean(z_As):.4f}")
    print(f"    max z_G = {np.max(z_Gs):.4f}, mean z_G = {np.mean(z_Gs):.4f}")

    # Test 2: actual delta_r (the ground truth)
    actual_violations = np.sum(actual_deltas >= 0)
    print(f"\n  TEST 2: ACTUAL delta_r (remove zero, recompute r)")
    print(f"    max(delta_r) = {np.max(actual_deltas):+.6f}")
    print(f"    mean(delta_r) = {np.mean(actual_deltas):+.6f}")
    print(f"    Actual violations (delta_r >= 0): {actual_violations}/{len(actual_deltas)}")

    # Compare predicted vs actual
    corr_pred = pearsonr(predicted_deltas, actual_deltas)[0]
    ratio = np.mean(actual_deltas) / np.mean(predicted_deltas) if np.mean(predicted_deltas) != 0 else 0
    print(f"\n  Influence function accuracy:")
    print(f"    Corr(predicted, actual) = {corr_pred:.4f}")
    print(f"    mean(actual) / mean(predicted) = {ratio:.4f}")
    print(f"    (ratio > 1 means influence UNDER-predicts the decrease)")

    # Worst-case zero: highest z_G * z_A
    worst_idx = np.argmax(products)
    print(f"\n  Worst-case zero (highest z_G * z_A):")
    print(f"    gamma = {z_core[worst_idx+1]:.2f}")
    print(f"    z_G = {z_Gs[worst_idx]:.3f}, z_A = {z_As[worst_idx]:.3f}")
    print(f"    product = {products[worst_idx]:.4f} vs 2r = {2*r_val:.4f}")
    print(f"    predicted delta_r = {predicted_deltas[worst_idx]:+.6f}")
    print(f"    actual delta_r = {actual_deltas[worst_idx]:+.6f}")

    elapsed = time.time() - t_start
    print(f"\n  ({elapsed:.1f}s)", flush=True)


# ============================================================
# FINAL ANALYSIS: Does the influence condition have the same
# log(T) growth problem as the deficit?
# ============================================================
print(f"\n{'='*80}")
print("CONCLUSION: INFLUENCE CONDITION SCALING")
print("="*80)
print("""
The deficit condition R < 1 grows as ~0.06*log(T) and extrapolates to failure at T ~ 3e8.

The direct influence condition z_G * z_A < 2r is WEAKER.
If the max product grows slower than 2r (or stays bounded), the approach works for all T.

Key: r decreases with T (from ~0.92 to ~0.81 in our data), but 2r is still large.
The question is whether max(z_G * z_A) stays below 2r as both evolve.
""")

print(f"\nTotal time: {time.time()-t0:.1f}s")
