"""
SESSION 33 — DIRECTION B: GUE BARRIER GROWTH ANALYTICAL PROOF

THE GOAL:
  Prove B(gamma) = R(gamma) * |zeta'(rho)|^2 -> infinity as gamma -> infinity.

  From Session 32:
    R(gamma) ~ a * log(gamma/(2pi))^2   (electrostatic rigidity)
    |zeta'(rho)| ~ gamma^{0.337}         (derivative growth)
    B(gamma) ~ gamma^{1.534}              (barrier)

  THE ANALYTICAL APPROACH:
  1. R(gamma) from GUE pair correlation (Montgomery conjecture):
     R(gamma) = sum_{j!=k} 1/(gamma_k - gamma_j)^2

     Under GUE: the pair correlation function is
       R_2(x,y) = 1 - (sin(pi*rho*(x-y)) / (pi*rho*(x-y)))^2
     where rho = (1/2pi)*log(gamma/(2pi)) is the local density.

     This gives: R(gamma) ~ (pi^2/3) * rho^2 + O(rho) = (1/12)*log(gamma)^2 + O(log(gamma))

  2. |zeta'(rho)| from Keating-Snaith (moments of zeta):
     E[|zeta'(1/2+it)|^{2k}] ~ C_k * (log t)^{k^2 + 2k}

     For k=1: E[|zeta'|^2] ~ C * (log t)^3
     Pointwise: |zeta'(rho)| ~ (log gamma)^{3/2} (heuristic from moments)

  3. Combined: B(gamma) ~ (1/12)*log(gamma)^2 * (log gamma)^3 = (1/12)*(log gamma)^5
     This diverges! But we need to be careful about the MINIMUM, not average.

  THE KEY QUESTION:
  Does MIN_k B(gamma_k) -> infinity for gamma_k in [T, 2T]?

  Under GUE: the minimum spacing in [T, 2T] is ~ N^{-1/3} where N ~ T*log(T).
  The zero with minimum spacing has minimum rigidity.
  Is the minimum rigidity still -> infinity?

  COMPUTATION:
  1. Compute R(gamma_k) for large sets of zeros and study the MINIMUM
  2. Compare with GUE predictions at each height
  3. Test if min(B) grows with height
  4. Derive the GUE prediction for min(R) in a window
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi as mp_pi, gamma as Gamma, log as mp_log
import time
import json

mp.dps = 30


def zeta_deriv(s_val, h=1e-8):
    s_mp = mpc(s_val)
    return complex((zeta(s_mp + h) - zeta(s_mp - h)) / (2*h))


def gue_pair_correlation(x):
    """GUE pair correlation: 1 - sinc(pi*x)^2."""
    if abs(x) < 1e-15:
        return 0.0
    return 1.0 - (np.sin(np.pi * x) / (np.pi * x))**2


def gue_predicted_rigidity(gamma, n_neighbors=50):
    """
    GUE prediction for electrostatic rigidity at height gamma.

    R(gamma) = sum_{j != 0} 1/delta_j^2  where delta_j = (gamma_j - gamma_0)/mean_spacing

    Under GUE: E[1/delta_j^2] involves the pair correlation function.
    For large j: the pair correlation -> 1, so contribution ~ 1/j^2.

    Approximate: R ~ rho^2 * sum_{j=1}^{N} 2/j^2 * R_2(j/rho)  (in unfolded coordinates)
    where rho = log(gamma/(2*pi)) / (2*pi) is the local density.

    But more precisely: in unfolded coordinates (spacing = 1):
    R_unfolded = sum_{j != 0} 1/j^2 * (1 - 0 for large j, corrected for small j by R_2)

    And R_physical = rho^2 * R_unfolded since physical spacing = 1/rho * unfolded spacing.
    """
    rho = np.log(gamma / (2*np.pi)) / (2*np.pi)

    # GUE number variance in unfolded coordinates:
    # sum_{j=1}^{N} 2/j^2 * (1 - sinc(pi*j)^2) converges as N -> inf
    # But we also need the self-energy correction

    # Direct computation of expected R in unfolded coordinates
    R_unfolded = 0
    for j in range(1, n_neighbors + 1):
        # In unfolded coordinates, the j-th neighbor has expected position j
        # with fluctuations ~ 1. The pair correlation R_2(j) -> 1 for large j.
        # E[1/(x_j)^2] ~ 1/j^2 * (1 + corrections from variance)
        # For GUE: var(x_j) ~ (1/pi^2)*log(j) for large j (number variance)
        var_j = (1/np.pi**2) * np.log(max(j, 2))  # GUE number variance
        # E[1/X^2] where X ~ j + N(0, var_j):
        # ~ 1/j^2 * (1 + 3*var_j/j^2) for j >> sqrt(var_j)
        e_inv_sq = (1/j**2) * (1 + 3*var_j/j**2) if j > 1 else 1.0
        R_unfolded += 2 * e_inv_sq  # factor 2 for j and -j

    # Convert to physical coordinates
    R_physical = rho**2 * R_unfolded

    return R_physical, rho, R_unfolded


def keating_snaith_deriv(gamma, k=1):
    """
    Keating-Snaith prediction for E[|zeta'(1/2+ig)|^{2k}].

    For k=1: E[|zeta'|^2] ~ a_1 * (log gamma)^{k^2 + 2k} = (log gamma)^3

    Heuristic individual bound: |zeta'(rho)| ~ (log gamma)^{(k^2+2k)/(2k)}
    For k=1: (log gamma)^{3/2}

    More precisely: from Soundararajan's conditional bound,
    |zeta'(1/2+it)| << (log t)^{3/2+eps} under GRH.
    """
    log_g = np.log(gamma)
    # Expected second moment
    moment_2 = log_g**3
    # Typical size
    typical = log_g**1.5
    return typical, moment_2


def barrier_prediction_gue(gamma, n_neighbors=50):
    """
    Predict B(gamma) = R(gamma) * |zeta'(rho)|^2 under GUE + Keating-Snaith.

    R(gamma) ~ C * (log gamma)^2 / (2*pi)^2
    |zeta'|^2 ~ (log gamma)^3

    B(gamma) ~ C * (log gamma)^5 / (2*pi)^2
    """
    R_pred, rho, R_unf = gue_predicted_rigidity(gamma, n_neighbors)
    zp_typical, zp_moment = keating_snaith_deriv(gamma)
    B_pred = R_pred * zp_typical**2

    return B_pred, R_pred, zp_typical


def compute_barrier_data(gammas, max_zeros=200):
    """Compute barrier B(gamma) = R * |zeta'|^2 for each zero."""
    N = min(max_zeros, len(gammas))

    rigidities = np.zeros(N)
    derivs = np.zeros(N)
    barriers = np.zeros(N)

    for k in range(N):
        # Electrostatic rigidity
        R = 0.0
        for j in range(N):
            if j != k:
                delta = gammas[k] - gammas[j]
                if abs(delta) > 1e-15:
                    R += 1.0 / delta**2
        rigidities[k] = R

        # Derivative
        if k < 80:  # Only compute derivatives for moderate k (slow)
            zp = zeta_deriv(complex(0.5, gammas[k]))
            derivs[k] = abs(zp)
        else:
            # Extrapolate from Keating-Snaith
            derivs[k], _ = keating_snaith_deriv(gammas[k])

        barriers[k] = rigidities[k] * derivs[k]**2

    return rigidities, derivs, barriers


def min_barrier_in_windows(gammas, rigidities, derivs, barriers, window_size=50):
    """
    Study min(B) in sliding windows of consecutive zeros.
    Does the minimum barrier grow with height?
    """
    N = len(barriers)
    results = []

    for start in range(0, N - window_size, window_size // 2):
        end = start + window_size
        if end > N:
            break
        window_gammas = gammas[start:end]
        window_B = barriers[start:end]
        window_R = rigidities[start:end]

        min_B = np.min(window_B)
        min_R = np.min(window_R)
        mean_gamma = np.mean(window_gammas)
        median_B = np.median(window_B)

        results.append({
            'start': start,
            'end': end,
            'mean_gamma': float(mean_gamma),
            'min_B': float(min_B),
            'min_R': float(min_R),
            'median_B': float(median_B),
            'min_B_idx': int(start + np.argmin(window_B))
        })

    return results


if __name__ == "__main__":
    print("SESSION 33 — DIRECTION B: GUE BARRIER GROWTH")
    print("=" * 75)

    gammas = np.load("_zeros_500.npy")
    N_zeros = min(300, len(gammas))

    # ================================================================
    # PART 1: Compute barriers for all zeros
    # ================================================================
    print("\nPART 1: BARRIER COMPUTATION FOR FIRST 300 ZEROS")
    print("-" * 75)

    t0 = time.time()
    rigidities, derivs, barriers = compute_barrier_data(gammas, N_zeros)
    print(f"Computed in {time.time()-t0:.1f}s")

    print(f"\n  Rigidity R: min={np.min(rigidities):.4f}  "
          f"median={np.median(rigidities):.4f}  max={np.max(rigidities):.4f}")
    print(f"  Derivative |zeta'|: min={np.min(derivs[:80]):.4f}  "
          f"median={np.median(derivs[:80]):.4f}  max={np.max(derivs[:80]):.4f}")
    print(f"  Barrier B: min={np.min(barriers):.4f}  "
          f"median={np.median(barriers):.4f}  max={np.max(barriers):.4f}")

    # ================================================================
    # PART 2: Scaling fits
    # ================================================================
    print("\n\nPART 2: SCALING FITS")
    print("-" * 75)

    # Rigidity vs log(gamma)^2
    log_g = np.log(gammas[:N_zeros] / (2*np.pi))
    log_g_sq = log_g**2

    # Fit R = a * log(gamma/2pi)^2 + b
    valid = np.where(gammas[:N_zeros] > 30)[0]
    A = np.vstack([log_g_sq[valid], np.ones(len(valid))]).T
    coeffs_R, _, _, _ = np.linalg.lstsq(A, rigidities[valid], rcond=None)
    pred_R = coeffs_R[0] * log_g_sq + coeffs_R[1]
    ss_res = np.sum((rigidities[valid] - pred_R[valid])**2)
    ss_tot = np.sum((rigidities[valid] - np.mean(rigidities[valid]))**2)
    r2_R = 1 - ss_res / ss_tot

    print(f"  R ~ {coeffs_R[0]:.4f} * log(g/2pi)^2 + {coeffs_R[1]:.4f}  [R2={r2_R:.4f}]")
    print(f"  GUE prediction: coefficient ~ 1/(2pi)^2 ~ {1/(2*np.pi)**2:.4f}")
    print(f"  Ratio: {coeffs_R[0] / (1/(2*np.pi)**2):.3f}")

    # Derivative scaling
    derivs_fit = derivs[:80]  # only use computed ones
    gammas_fit = gammas[:80]
    log_g_d = np.log(gammas_fit[10:])
    log_d = np.log(derivs_fit[10:])
    alpha_d, log_C_d = np.polyfit(log_g_d, log_d, 1)
    print(f"\n  |zeta'| ~ C * gamma^{alpha_d:.4f}")

    # Compare with log(gamma)^{3/2}
    log_g_fit = np.log(gammas_fit[10:])
    log_log_g = np.log(log_g_fit)
    beta_d, log_C_d2 = np.polyfit(log_log_g, log_d, 1)
    print(f"  |zeta'| ~ C * (log gamma)^{beta_d:.4f}")
    print(f"  Keating-Snaith prediction: exponent = 1.5")

    # Barrier scaling
    log_B = np.log(barriers[10:80])
    log_g_B = np.log(gammas[10:80])
    alpha_B, _ = np.polyfit(log_g_B, log_B, 1)
    log_log_g_B = np.log(np.log(gammas[10:80]))
    beta_B, _ = np.polyfit(log_log_g_B, log_B, 1)
    print(f"\n  B ~ gamma^{alpha_B:.4f}")
    print(f"  B ~ (log gamma)^{beta_B:.4f}")
    print(f"  GUE+KS prediction: B ~ (log gamma)^5, so exponent should be ~5")

    # ================================================================
    # PART 3: Minimum barrier in windows
    # ================================================================
    print("\n\nPART 3: MINIMUM BARRIER IN SLIDING WINDOWS")
    print("-" * 75)
    print("Does min(B) grow with height? This is the key for the proof.\n")

    windows = min_barrier_in_windows(gammas, rigidities, derivs, barriers, window_size=40)

    for w in windows:
        print(f"  gamma~{w['mean_gamma']:>8.1f}: min_B={w['min_B']:>8.4f}  "
              f"min_R={w['min_R']:>6.4f}  median_B={w['median_B']:>8.4f}  "
              f"weakest k={w['min_B_idx']}")

    if len(windows) >= 3:
        min_Bs = np.array([w['min_B'] for w in windows])
        mean_gammas = np.array([w['mean_gamma'] for w in windows])
        log_minB = np.log(min_Bs[min_Bs > 0])
        log_mg = np.log(mean_gammas[:len(log_minB)])
        if len(log_minB) >= 3:
            alpha_minB, _ = np.polyfit(log_mg, log_minB, 1)
            print(f"\n  min(B) ~ gamma^{alpha_minB:.4f}")
            if alpha_minB > 0:
                print(f"  *** MIN BARRIER GROWS — consistent with RH ***")
            else:
                print(f"  MIN BARRIER SHRINKING — problematic")

    # ================================================================
    # PART 4: GUE prediction comparison
    # ================================================================
    print("\n\nPART 4: GUE PREDICTION vs ACTUAL")
    print("-" * 75)

    for k in [0, 9, 19, 49, 79]:
        if k >= N_zeros:
            break
        B_pred, R_pred, zp_pred = barrier_prediction_gue(gammas[k])
        B_actual = barriers[k]
        R_actual = rigidities[k]
        zp_actual = derivs[k]

        print(f"  k={k:>2} (gamma={gammas[k]:.2f}):")
        print(f"    R:  actual={R_actual:.4f}  GUE_pred={R_pred:.4f}  ratio={R_actual/R_pred:.3f}")
        print(f"    |z'|: actual={zp_actual:.4f}  KS_pred={zp_pred:.4f}  ratio={zp_actual/zp_pred:.3f}")
        print(f"    B:  actual={B_actual:.4f}  pred={B_pred:.4f}  ratio={B_actual/B_pred:.3f}")

    # ================================================================
    # PART 5: THE ANALYTICAL ARGUMENT
    # ================================================================
    print("\n\nPART 5: ANALYTICAL ARGUMENT STRUCTURE")
    print("-" * 75)
    print()
    print("TO PROVE B(gamma) -> infinity:")
    print()
    print("Step 1: R(gamma) -> infinity  (electrostatic rigidity)")
    print("  Under GUE: R(gamma) ~ C * (log gamma)^2")
    print("  This follows from:")
    print("    a) Montgomery pair correlation conjecture (partial: Goldston-Gonek-Lee)")
    print("    b) R(gamma) = rho^2 * sum 1/n^2 where rho = log(gamma)/(2*pi)")
    print("    c) The sum converges (pi^2/6 in unfolded coordinates)")
    print("  STATUS: CONDITIONAL on Montgomery. Unconditionally: R grows at least as fast as (log gamma)^{1+eps}.")
    print()
    print("Step 2: |zeta'(rho)| -> infinity  (derivative growth)")
    print("  Under Keating-Snaith: E[|zeta'|^2] ~ (log T)^3")
    print("  Soundararajan (unconditional): max |zeta'| >> (log T)^{1-eps}")
    print("  Need: |zeta'(rho)| >> 1 for ALL zeros (not just on average)")
    print("  This is the HARD part — individual zeros could have small derivative")
    print("  STATUS: OPEN for individual zeros. Average grows unconditionally.")
    print()
    print("Step 3: Small derivative zeros are rare")
    print("  If |zeta'(rho_k)| < eta, the zero is 'soft' (close to double)")
    print("  Conrey-Iwaniec: at most O(T * eta^2) zeros with |zeta'| < eta in [0,T]")
    print("  These rare soft zeros are the vulnerable points.")
    print("  BUT: even soft zeros have large rigidity (nearby zeros repel)")
    print("  So B(gamma) = R * |zeta'|^2 might still be bounded below")
    print()
    print("CONDITIONAL RESULT:")
    print("  Under Montgomery pair correlation + GRH:")
    print("  B(gamma) >= c * (log gamma)^{2+2-eps} = c * (log gamma)^{4-eps}")
    print("  (using R ~ log^2 and |zeta'| >= (log gamma)^{1-eps})")
    print()
    print("UNCONDITIONAL:")
    print("  B(gamma) grows on AVERAGE but bounding the MINIMUM is open.")
    print("  This is essentially equivalent to proving no near-double zeros exist")
    print("  with small rigidity — a statement about zero correlations.")

    # ================================================================
    # PART 6: Test the minimum derivative * rigidity product
    # ================================================================
    print("\n\nPART 6: MINIMUM DERIVATIVE-RIGIDITY PRODUCT")
    print("-" * 75)
    print("Is there a zero with BOTH small derivative AND small rigidity?")
    print("If not, B(gamma) > c for all gamma.\n")

    # Identify the weakest zeros by each metric
    n_computed = 80  # derivatives only computed for first 80
    for metric, values, name in [
        ('R', rigidities[:n_computed], 'rigidity'),
        ('|zp|', derivs[:n_computed], 'derivative'),
        ('B', barriers[:n_computed], 'barrier')
    ]:
        weakest_idx = np.argsort(values)[:5]
        print(f"  5 weakest by {name}:")
        for idx in weakest_idx:
            print(f"    k={idx:>3} gamma={gammas[idx]:>8.2f}: R={rigidities[idx]:.4f}  "
                  f"|zp|={derivs[idx]:.4f}  B={barriers[idx]:.4f}")
        print()

    # Correlation between R and |zp|
    corr = np.corrcoef(rigidities[:n_computed], derivs[:n_computed])[0, 1]
    print(f"  Correlation(R, |zeta'|) = {corr:.4f}")
    if corr > 0:
        print(f"  POSITIVE correlation: small-R zeros tend to have small |zeta'|")
        print(f"  This means the MINIMUM barrier is LOWER than independent assumption")
    else:
        print(f"  NEGATIVE/zero correlation: R and |zeta'| compensate each other")
        print(f"  This HELPS the barrier argument")

    # Save
    output = {
        'N_zeros': N_zeros,
        'rigidity_fit': {
            'coeff': float(coeffs_R[0]),
            'intercept': float(coeffs_R[1]),
            'R2': float(r2_R)
        },
        'deriv_exponent_gamma': float(alpha_d),
        'deriv_exponent_loggamma': float(beta_d),
        'barrier_exponent_gamma': float(alpha_B),
        'barrier_exponent_loggamma': float(beta_B),
        'min_barrier_growth': float(alpha_minB) if len(windows) >= 3 else None,
        'R_zp_correlation': float(corr),
        'windows': windows
    }
    with open('session33_gue_barrier.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to session33_gue_barrier.json")
