"""THE PIVOT: Can off-line zeros survive the eigenvector rigidity constraint?

SETUP:
  RH says all nontrivial zeros have Re(s) = 1/2.
  Suppose one zero moves off-line: rho = beta + i*gamma with beta != 1/2.
  By the functional equation, 1-beta + i*gamma is also a zero.

  Effects on the Z-function zero sequence:
  1. Z(t) has zeros where zeta(1/2+it) = 0.
  2. An off-line zero at beta+i*gamma means zeta(1/2+i*gamma) != 0.
  3. So Z(gamma) != 0 — the zero is "missing" from the Z sequence.
  4. The counting function N(T) gains +1 (one zero splits into two).
  5. The Z-zero sequence has a gap where the missing zero should be.

THE TEST:
  Does removing a zero from the sequence (simulating off-line movement)
  INCREASE or DECREASE the peak-gap correlation r?

  If decrease: off-line zeros hurt r, eigenvector rigidity constrains them.
  If increase: need a more subtle argument (or this approach doesn't work).

ALSO:
  Compute the EXACT effect on Z(t) via the Hadamard product ratio
  when a zero moves from 1/2+ig to beta+ig.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy.stats import pearsonr
import mpmath
mpmath.mp.dps = 30

t0 = time.time()

N = 200
zeta_zeros = np.load("_zeros_200.npy")
trim = int(0.1 * N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

def hardy_Z(t):
    return float(mpmath.siegelz(t))


# ============================================================
# Precompute: exact r for complete zero sequence
# ============================================================
print("Computing baseline r for all 200 zeros...", flush=True)
gaps_exact = np.diff(zeta_zeros)
mids_exact = (zeta_zeros[:-1] + zeta_zeros[1:]) / 2
peaks_exact = np.array([abs(hardy_Z(m)) for m in mids_exact])

nt = int(0.1 * len(gaps_exact))
r_exact, p_exact = pearsonr(gaps_exact[nt:-nt], peaks_exact[nt:-nt])
print(f"  Baseline: r = {r_exact:+.4f} (p = {p_exact:.2e})")


# ============================================================
# TEST 1: REMOVE ONE ZERO — does r go up or down?
# ============================================================
print("\n" + "="*70)
print("TEST 1: REMOVE ONE ZERO (simulate going off-line)")
print("="*70)

# For each zero k, remove it and recompute r
# When zero_k is removed:
#   - gap_{k-1} and gap_k merge into one big gap
#   - midpoint shifts
#   - Z at new midpoint needs recomputing

print(f"\n  Removing each zero in turn (trimmed region only)...", flush=True)

r_without = np.zeros(N)
for k_remove in range(nt, N - nt):
    # Build modified zero sequence without zero_k
    zeros_mod = np.concatenate([zeta_zeros[:k_remove], zeta_zeros[k_remove+1:]])
    gaps_mod = np.diff(zeros_mod)
    mids_mod = (zeros_mod[:-1] + zeros_mod[1:]) / 2

    # Z at new midpoints — only need to recompute near the removed zero
    peaks_mod = np.zeros(len(mids_mod))
    for j in range(len(mids_mod)):
        if abs(j - k_remove) <= 1:
            # Near the removed zero: recompute Z
            peaks_mod[j] = abs(hardy_Z(mids_mod[j]))
        elif j < k_remove:
            peaks_mod[j] = peaks_exact[j]  # unchanged
        else:
            peaks_mod[j] = peaks_exact[j + 1]  # shifted by 1

    nt_mod = int(0.1 * len(gaps_mod))
    r_mod, _ = pearsonr(gaps_mod[nt_mod:-nt_mod], peaks_mod[nt_mod:-nt_mod])
    r_without[k_remove] = r_mod

# Statistics
core = r_without[nt:N-nt]
core = core[core != 0]  # exclude zeros from unfilled entries

print(f"\n  Removing one zero from the sequence:")
print(f"    Baseline r:         {r_exact:+.4f}")
print(f"    Mean r(removed):    {np.mean(core):+.4f}")
print(f"    Min r(removed):     {np.min(core):+.4f}")
print(f"    Max r(removed):     {np.max(core):+.4f}")
print(f"    Std r(removed):     {np.std(core):.4f}")

delta_r = np.mean(core) - r_exact
print(f"\n    EFFECT: removing one zero {'DECREASES' if delta_r < 0 else 'INCREASES'} "
      f"r by {abs(delta_r):.4f}")
print(f"    Direction: {'GOOD for elimination approach' if delta_r < 0 else 'BAD — need different argument'}")


# ============================================================
# TEST 2: REMOVE MULTIPLE ZEROS — cumulative effect
# ============================================================
print("\n" + "="*70)
print("TEST 2: REMOVE MULTIPLE ZEROS — cumulative effect on r")
print("="*70)

rng = np.random.default_rng(42)

print(f"\n  {'n_removed':>10} {'r':>10} {'delta_r':>10} {'per_zero':>10}")
print(f"  {'-'*44}")

for n_remove in [0, 1, 2, 5, 10, 20, 50]:
    if n_remove == 0:
        r_val = r_exact
        print(f"  {n_remove:>10} {r_val:>+10.4f} {'---':>10} {'---':>10}")
        continue

    # Remove n_remove zeros uniformly from the core region
    r_trials = []
    for trial in range(10):
        removable = np.arange(nt, N - nt)
        chosen = rng.choice(removable, size=min(n_remove, len(removable)), replace=False)
        chosen = np.sort(chosen)

        zeros_mod = np.delete(zeta_zeros, chosen)
        gaps_mod = np.diff(zeros_mod)
        mids_mod = (zeros_mod[:-1] + zeros_mod[1:]) / 2
        peaks_mod = np.array([abs(hardy_Z(m)) for m in mids_mod])

        nt_mod = int(0.1 * len(gaps_mod))
        if nt_mod > 0 and len(gaps_mod) > 2 * nt_mod:
            r_mod, _ = pearsonr(gaps_mod[nt_mod:-nt_mod], peaks_mod[nt_mod:-nt_mod])
            r_trials.append(r_mod)

    r_mean = np.mean(r_trials)
    delta = r_mean - r_exact
    per_zero = delta / n_remove if n_remove > 0 else 0
    print(f"  {n_remove:>10} {r_mean:>+10.4f} {delta:>+10.4f} {per_zero:>+10.4f}")


# ============================================================
# TEST 3: The PHYSICAL effect — what happens to Z(t) when a
# zero moves off-line?
# ============================================================
print("\n" + "="*70)
print("TEST 3: HADAMARD PRODUCT — Z(t) when a zero moves off-line")
print("="*70)

# If rho = 1/2 + i*gamma is a zero of zeta, and we "move" it to
# beta + i*gamma (with beta != 1/2), the zeta function changes by
# a multiplicative factor:
#
#   zeta_mod(s) / zeta(s) = [(s - beta - i*gamma)(s - (1-beta) - i*gamma)] /
#                            [(s - 1/2 - i*gamma)^2]
#   (approximately — ignoring the e^{s/rho} factors and the conjugate pair)
#
# More precisely for Z(t):
#   Z_mod(t) = Z(t) * |factor at s = 1/2 + it|
#
# Near t = gamma: Z(gamma) = 0 for the original.
#   Z_mod(gamma) = Z'(gamma) * 0 * factor...
#   Actually Z has a simple zero at gamma, so Z(t) ~ Z'(gamma)(t-gamma) near gamma.
#   The factor [(1/2+it-beta-ig)(1/2+it-(1-beta)-ig)] / [(1/2+it-1/2-ig)^2]
#   = [(1/2-beta+i(t-g))((beta-1/2)+i(t-g))] / [i(t-g)]^2
#   = [-(beta-1/2)^2 - (t-g)^2 + ...] / [-(t-g)^2]
#   At t=gamma: = [-(beta-1/2)^2] / [0] -> diverges!
#
# So Z_mod(t) near t=gamma has: Z(t)~(t-gamma) * factor~1/(t-gamma)^2 * ...
# The zero of Z cancels one power of (t-gamma), leaving a pole-like feature.
# This means Z_mod(gamma) is NOT zero, and the singularity structure changes.

# Let's compute this numerically. For a zero at gamma_k, move it to beta+i*gamma_k.
# Compute the modified Z function values at nearby points.

print("\n  Moving zero_50 off-line to beta + i*gamma_50...")
k_test = 50  # middle of the sequence
gamma_k = zeta_zeros[k_test - 1]
print(f"  gamma_{k_test} = {gamma_k:.6f}")

# The modification factor for zeta(s):
# R(s) = zeta_mod(s)/zeta(s)
#       = (s - rho_new)(s - rho_new_bar)(s - (1-rho_new))(s - (1-rho_new_bar))
#         / [(s - rho)(s - rho_bar)(s - (1-rho))(s - (1-rho_bar))]
# where rho = 1/2 + i*gamma, rho_new = beta + i*gamma

# For s = 1/2 + it (on the critical line):
# The original zero contributes (s-rho) = i(t-gamma)
# and (s-(1-rho)) = (s - 1/2 + i*gamma) = i(t+gamma) (far away for t near gamma)
# Plus conjugates: (s-rho_bar) = (1/2+it-1/2+ig) = i(t+gamma) also far

# Simplified: near t ~ gamma, the dominant term is
# R(1/2+it) ~ [(1/2+it - beta - ig)(1/2+it - (1-beta) - ig)] / [(1/2+it - 1/2 - ig)^2]
#            * [conjugate pair corrections]
# The conjugate pair (at -gamma) is far away and ~1 near t=gamma.

def modification_factor(t, gamma, beta):
    """Ratio zeta_mod(1/2+it) / zeta(1/2+it) when moving a zero from
    1/2+i*gamma to beta+i*gamma (and its functional equation partner)."""
    s = mpmath.mpc(0.5, t)

    # Original zero pair: rho = 1/2+ig, 1-rho = 1/2-ig (= rho_bar for real gamma)
    rho_orig = mpmath.mpc(0.5, gamma)
    rho_orig_conj = mpmath.mpc(0.5, -gamma)  # = 1 - rho_orig

    # New zero pair: rho_new = beta+ig, and 1-rho_new = (1-beta)+ig
    rho_new = mpmath.mpc(beta, gamma)
    rho_new_partner = mpmath.mpc(1 - beta, gamma)
    rho_new_conj = mpmath.mpc(beta, -gamma)
    rho_new_partner_conj = mpmath.mpc(1 - beta, -gamma)

    # Numerator: (s - rho_new)(s - rho_new_partner)(s - rho_new_conj)(s - rho_new_partner_conj)
    num = (s - rho_new) * (s - rho_new_partner) * (s - rho_new_conj) * (s - rho_new_partner_conj)

    # Denominator: (s - rho_orig)^2 * (s - rho_orig_conj)^2
    # Wait — the original has rho = 1/2+ig and 1-rho = 1/2-ig. These are 2 zeros.
    # But for RH zeros, rho and 1-rho_bar are the same: 1-(1/2-ig) = 1/2+ig.
    # So the original has: rho, rho_bar as the two zeros (at +-gamma).
    # The new has: rho_new, rho_new_partner, rho_new_conj, rho_new_partner_conj (4 zeros for 2).
    # Net: gained 2 extra zeros. But that's the real count issue.

    # Actually: for each pair of zeros {rho, 1-rho_bar} that are identical on the
    # critical line (since 1-(1/2-ig) = 1/2+ig = rho), moving off-line splits them:
    # rho_new != 1-rho_new_bar in general.

    # For a single on-line zero rho = 1/2+ig, the partner under s->1-s is 1/2-ig = rho_bar.
    # Moving off: rho -> beta+ig, partner -> (1-beta)+ig.
    # Conjugates: beta-ig and (1-beta)-ig.
    # Original: 1/2+ig and 1/2-ig (= 2 zeros for one height gamma)
    # New: beta+ig, (1-beta)+ig, beta-ig, (1-beta)-ig (= 4 zeros for one height gamma)
    # But the original had 2 and new has 4 — that's 2 extra. Not right for "moving".

    # The correct accounting: the original zero at height gamma contributes
    # one zero rho = 1/2+ig in the upper half-plane (ignoring lower half).
    # Moving it off-line gives TWO zeros in the upper half-plane:
    # beta+ig and (1-beta)+ig.
    # This means N(T) increases by 1 for T > gamma. Fine.

    # For the modification factor, we only need upper half plane:
    # Original: (s - (1/2+ig))
    # New: (s - (beta+ig)) * (s - ((1-beta)+ig))
    # Lower half (conjugates) are far from t~gamma and contribute ~1.

    num_upper = (s - rho_new) * (s - rho_new_partner)
    den_upper = (s - rho_orig)

    # Also include the lower half plane terms (small correction near t~gamma)
    num_lower = (s - rho_new_conj) * (s - rho_new_partner_conj)
    den_lower = (s - rho_orig_conj)

    R = (num_upper * num_lower) / (den_upper * den_lower)
    return R


# Compute Z_mod(t) near gamma_k for various beta
t_range = np.linspace(gamma_k - 3, gamma_k + 3, 200)

print(f"\n  Z(t) and Z_mod(t) near gamma_{k_test} = {gamma_k:.4f}:")
print(f"  {'beta':>8} {'Z_mod(gamma)':>14} {'|Z_mod(gamma)|':>16} "
      f"{'max|Z_mod|':>12} {'min|Z_mod|':>12}")
print(f"  {'-'*66}")

for beta in [0.5, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99]:
    Z_mod_vals = []
    for t in t_range:
        Z_orig = hardy_Z(t)
        R = modification_factor(t, gamma_k, beta)
        # Z_mod(t) = Z(t) * |R(1/2+it)| * phase_correction
        # Actually Z = e^{i*theta} * zeta(1/2+it), so
        # Z_mod = e^{i*theta} * zeta_mod(1/2+it) = e^{i*theta} * R * zeta(1/2+it)
        # = R * Z(t) ... but R is complex. We need |Z_mod| or Re(Z_mod).
        # Since Z is real, Z_mod = Re(R * Z_orig) ... no, that's not right either.
        # The modification changes zeta, not Z directly.
        # Z(t) = e^{i*theta(t)} * zeta(1/2+it), which is real.
        # Z_mod(t) = e^{i*theta(t)} * zeta_mod(1/2+it) = e^{i*theta(t)} * R(1/2+it) * zeta(1/2+it)
        # = R(1/2+it) * Z(t)  ... but R is complex, so Z_mod is complex!
        #
        # Wait: if beta != 1/2, zeta_mod does NOT satisfy the same functional equation.
        # So Z_mod(t) = e^{i*theta(t)} * zeta_mod(1/2+it) is NOT necessarily real.
        # The "Z function" framework breaks down for modified zeta.
        #
        # Better: just compute |zeta_mod(1/2+it)| and use that as the "amplitude".
        z_val = float(abs(complex(mpmath.zeta(mpmath.mpc(0.5, t)))))
        R_val = complex(modification_factor(t, gamma_k, beta))
        z_mod = z_val * abs(R_val)
        Z_mod_vals.append(z_mod)

    Z_mod_arr = np.array(Z_mod_vals)
    # Value at gamma
    idx_gamma = np.argmin(np.abs(t_range - gamma_k))
    z_at_gamma = Z_mod_arr[idx_gamma]

    print(f"  {beta:>8.2f} {z_at_gamma:>14.6f} {abs(z_at_gamma):>16.6f} "
          f"{np.max(Z_mod_arr):>12.6f} {np.min(Z_mod_arr):>12.6f}")


# ============================================================
# TEST 4: The CRITICAL test — move a zero off-line and measure r
# ============================================================
print("\n" + "="*70)
print("TEST 4: MOVE ZERO OFF-LINE — MEASURE r DIRECTLY")
print("="*70)

# The proper test: if zero_k moves to beta+ig, then:
# 1. Remove gamma_k from the Z-zero sequence
# 2. Z(gamma_k) is no longer zero — it has value related to (beta-1/2)
# 3. The gap around gamma_k enlarges
# 4. Compute the new gaps and Z-peaks, measure r

# For Z-zeros: they are the zeros of Z(t) = real-valued function.
# If we remove one zero, the adjacent zeros are unchanged (they're still zeros of zeta on critical line).
# Only the gap structure and the Z-values at midpoints change.

# Method: remove zero_k, recompute gaps and peaks, measure r.
# We already did this in Test 1. But now let's also ask:
# What is |Z| at the midpoint of the enlarged gap?

print(f"\n  Detailed analysis: removing zero_k and measuring gap + peak")
print(f"  {'k':>6} {'gap_before':>12} {'gap_after':>12} {'|Z|_before':>12} "
      f"{'|Z|_after':>12} {'effect':>10}")
print(f"  {'-'*68}")

for k_remove in [20, 50, 80, 100, 120, 150, 180]:
    if k_remove >= N:
        continue
    # Before: two gaps around zero_k
    if k_remove > 0 and k_remove < N - 1:
        gap_L = zeta_zeros[k_remove] - zeta_zeros[k_remove - 1]
        gap_R = zeta_zeros[k_remove + 1] - zeta_zeros[k_remove]
        peak_L = abs(hardy_Z((zeta_zeros[k_remove-1] + zeta_zeros[k_remove]) / 2))
        peak_R = abs(hardy_Z((zeta_zeros[k_remove] + zeta_zeros[k_remove+1]) / 2))

        # After: one big gap
        gap_merged = zeta_zeros[k_remove + 1] - zeta_zeros[k_remove - 1]
        mid_merged = (zeta_zeros[k_remove - 1] + zeta_zeros[k_remove + 1]) / 2
        peak_merged = abs(hardy_Z(mid_merged))

        # The merged gap is larger. Is the merged peak larger too?
        avg_peak_before = (peak_L + peak_R) / 2
        effect = "r+" if (gap_merged > gap_L + gap_R - 0.01 and
                          peak_merged > avg_peak_before) else "r-"

        print(f"  {k_remove:>6} {gap_L:>6.3f}+{gap_R:<5.3f} {gap_merged:>12.3f} "
              f"{avg_peak_before:>12.4f} {peak_merged:>12.4f} {effect:>10}")


# ============================================================
# TEST 5: SYSTEMATIC — r as function of number of removed zeros
# with POSITION-AWARE removal
# ============================================================
print("\n" + "="*70)
print("TEST 5: r vs n_removed (position-aware)")
print("="*70)

# Key insight: if removing zeros that are in SMALL gaps hurts r more
# than removing zeros in LARGE gaps, that tells us about the mechanism.

# Sort zeros by their gap size
gap_sizes = np.diff(zeta_zeros)
# For zero_k, its "gap context" is min(gap_L, gap_R)
zero_gap_context = np.zeros(N)
for k in range(1, N-1):
    zero_gap_context[k] = min(gap_sizes[k-1], gap_sizes[k])

# Remove zeros from smallest gaps first vs largest gaps first
print(f"\n  {'strategy':>20} {'n_rem':>6} {'r':>10} {'delta_r':>10}")
print(f"  {'-'*50}")

for strategy, label in [("small_gap", "Small gaps first"),
                         ("large_gap", "Large gaps first"),
                         ("random", "Random")]:
    for n_rem in [1, 5, 10, 20]:
        r_trials = []
        for trial in range(5 if strategy == "random" else 1):
            removable = np.arange(nt + 5, N - nt - 5)  # stay away from edges

            if strategy == "small_gap":
                order = np.argsort(zero_gap_context[removable])
                chosen = removable[order[:n_rem]]
            elif strategy == "large_gap":
                order = np.argsort(zero_gap_context[removable])[::-1]
                chosen = removable[order[:n_rem]]
            else:
                chosen = rng.choice(removable, size=n_rem, replace=False)

            zeros_mod = np.delete(zeta_zeros, chosen)
            gaps_mod = np.diff(zeros_mod)
            mids_mod = (zeros_mod[:-1] + zeros_mod[1:]) / 2
            peaks_mod = np.array([abs(hardy_Z(m)) for m in mids_mod])

            nt_mod = int(0.1 * len(gaps_mod))
            if len(gaps_mod) > 2 * nt_mod:
                r_mod, _ = pearsonr(gaps_mod[nt_mod:-nt_mod], peaks_mod[nt_mod:-nt_mod])
                r_trials.append(r_mod)

        r_mean = np.mean(r_trials) if r_trials else 0
        delta = r_mean - r_exact
        print(f"  {label:>20} {n_rem:>6} {r_mean:>+10.4f} {delta:>+10.4f}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70)
print("VERDICT: ELIMINATION APPROACH VIABILITY")
print("="*70)

print(f"""
  Baseline r (200 zeros): {r_exact:+.4f}

  KEY QUESTION: Does removing zeros (simulating off-line movement)
  increase or decrease r?

  If DECREASE: The eigenvector rigidity r = +0.88 constrains the
  number of off-line zeros. The higher r is, the fewer can be off-line.
  If we can push this to "zero off-line zeros allowed", that's RH.

  If INCREASE: The correlation structure is RESILIENT to missing zeros,
  and we need a different argument for the elimination approach.

  Mean effect of removing one zero: {delta_r:+.4f}
""")

print(f"Total time: {time.time()-t0:.1f}s", flush=True)
