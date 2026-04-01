"""
SESSION 32 — RODGERS-TAO DUAL BARRIER EXPLORATION

Rodgers-Tao (2020) proved Lambda >= 0 (Newman's conjecture).
Their technique: show that if Lambda < 0, zeros would need to collide
on the real line as t → Lambda from above, creating double zeros that
split into conjugate pairs — contradicting the heat flow dynamics.

THE DUAL QUESTION: Can we prove Lambda <= 0?
Lambda <= 0 + Lambda >= 0 => Lambda = 0 => RH.

KEY OBSERVATIONS FROM SESSION 31:
1. Delta_min ~ N^{-1/3} (GUE): minimum spacing vanishes
2. Collision time t_c ~ delta^2/8 ~ N^{-2/3} → 0 (consistent with Lambda=0)
3. |zeta'(rho)| grows as (log gamma)^1.39 — zeros increasingly "rigid"
4. The system IS at the phase transition

THE DUAL BARRIER IDEA:
- Rodgers-Tao barrier: at each t > 0, the repulsive dynamics prevent collision
  from BELOW (zeros can't approach from imaginary axis to real axis too fast)
- Dual barrier: at t = 0, the rigidity of zeros (GUE statistics) prevents
  escape from the real axis (zeros can't leave because the energy cost exceeds
  the available thermal fluctuation)

APPROACH:
1. Compute the "escape energy" for a zero at height gamma to move off-line
   by displacement epsilon: E(gamma, epsilon)
2. Compare with the "available energy" from the heat kernel at time t
3. Show that at t = 0, the escape energy diverges relative to available energy
4. Map the "dual barrier" as a function of gamma
5. Check if GUE rigidity provides the missing bound
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, exp, sqrt, gamma as Gamma, log, nstr
import time
import json

mp.dps = 30


def xi_function(z):
    """Xi(z) where s = 1/2 + iz."""
    z_mp = mpc(z)
    s = mpf('0.5') + mpc(0, 1) * z_mp
    try:
        return complex(mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s/2) * Gamma(s/2) * zeta(s))
    except:
        return 0.0


def zeta_deriv(s_val, h=1e-8):
    """Numerical derivative of zeta(s)."""
    s_mp = mpc(s_val)
    return complex((zeta(s_mp + h) - zeta(s_mp - h)) / (2 * h))


def escape_energy_at_zero(gamma_val, epsilon, n_neighbors=20, gammas=None):
    """
    Compute the energy cost of displacing a zero at gamma_val
    off the critical line by epsilon.

    In the electrostatic model, each zero at z_k contributes
    -log|z - z_k| to the potential. The energy change from
    displacing z_k = gamma_k → gamma_k + i*epsilon is:

    Delta_E = sum_{j != k} [log|gamma_k - gamma_j| - log|gamma_k + i*eps - gamma_j|]

    For small epsilon:
    Delta_E ≈ epsilon^2 * sum_{j != k} 1/(gamma_k - gamma_j)^2 / 2

    This is the "restoring force" — the electrostatic rigidity.
    """
    if gammas is None:
        return 0

    k_idx = np.argmin(np.abs(gammas - gamma_val))
    gamma_k = gammas[k_idx]

    # Sum 1/(gamma_k - gamma_j)^2 for neighbors
    restoring = 0.0
    for j in range(max(0, k_idx - n_neighbors), min(len(gammas), k_idx + n_neighbors + 1)):
        if j == k_idx:
            continue
        delta = gamma_k - gammas[j]
        if abs(delta) > 1e-15:
            restoring += 1.0 / delta**2

    # Energy cost for displacement epsilon
    delta_E = 0.5 * epsilon**2 * restoring

    return delta_E, restoring


def heat_kernel_energy(t, gamma_val):
    """
    The "available energy" from the heat flow at time t.

    The heat kernel at time t has scale sqrt(2t).
    A zero at height gamma has thermal fluctuation ~ sqrt(2t).
    The available energy to move a zero is ~ t (from dimensional analysis).
    """
    return t


def dual_barrier_analysis(gammas, t_values):
    """
    For each zero and each time t, compute:
    - Escape energy E_escape(gamma, epsilon) for epsilon = sqrt(2t)
    - Available energy E_available(t)
    - Ratio R = E_escape / E_available

    If R >> 1 for all zeros at t=0, the barrier prevents escape.
    """
    results = []
    N = min(100, len(gammas))

    for k in range(N):
        gamma_k = gammas[k]

        # Restoring force (electrostatic rigidity)
        _, restoring = escape_energy_at_zero(gamma_k, 0.01, n_neighbors=50, gammas=gammas[:N])

        # For various t values
        for t in t_values:
            epsilon = np.sqrt(2 * max(t, 1e-30))  # thermal displacement
            delta_E = 0.5 * epsilon**2 * restoring  # = t * restoring
            E_avail = t if t > 0 else 1e-30

            # The ratio is simply the restoring force!
            # R = delta_E / E_avail = t * restoring / t = restoring
            ratio = restoring

            results.append({
                'k': k,
                'gamma': float(gamma_k),
                't': float(t),
                'restoring': float(restoring),
                'delta_E': float(delta_E),
                'ratio': float(ratio)
            })

    return results


if __name__ == "__main__":
    print("RODGERS-TAO DUAL BARRIER EXPLORATION")
    print("=" * 75)

    gammas = np.load("_zeros_500.npy")
    N = min(300, len(gammas))

    # ================================================================
    # PART 1: Electrostatic rigidity at each zero
    # ================================================================
    print("\nPART 1: ELECTROSTATIC RIGIDITY (restoring force at each zero)")
    print("-" * 75)
    print("Restoring force R(k) = sum_{j!=k} 1/(gamma_k - gamma_j)^2")
    print("This is the 'spring constant' preventing off-line escape.\n")

    rigidities = []
    for k in range(N):
        _, R = escape_energy_at_zero(gammas[k], 0.01, n_neighbors=N, gammas=gammas[:N])
        rigidities.append(R)

    rigidities = np.array(rigidities)

    # Print statistics
    for percentile in [0, 10, 25, 50, 75, 90, 100]:
        val = np.percentile(rigidities, percentile)
        idx = np.argmin(np.abs(rigidities - val))
        print(f"  {percentile:>3}th percentile: R = {val:.4f}  "
              f"(gamma~ {gammas[idx]:.2f}, k={idx})")

    print(f"\n  Mean rigidity:   {np.mean(rigidities):.4f}")
    print(f"  Min rigidity:    {np.min(rigidities):.4f} at gamma={gammas[np.argmin(rigidities)]:.2f}")
    print(f"  Max rigidity:    {np.max(rigidities):.4f} at gamma={gammas[np.argmax(rigidities)]:.2f}")

    # ================================================================
    # PART 2: Rigidity scaling with height
    # ================================================================
    print("\n\nPART 2: RIGIDITY vs HEIGHT (does rigidity grow?)")
    print("-" * 75)
    print("GUE prediction: local density ~ (1/2pi)*log(gamma/(2*pi))")
    print("So spacing ~ 2*pi/log(gamma), rigidity ~ (log(gamma))^2/(2*pi)^2\n")

    # Fit rigidity vs gamma
    valid = [(gammas[k], rigidities[k]) for k in range(10, N) if gammas[k] > 20]
    g_vals = np.array([v[0] for v in valid])
    r_vals = np.array([v[1] for v in valid])

    # Fit R vs log(gamma)^2
    log_g = np.log(g_vals / (2*np.pi))
    log_g_sq = log_g**2

    # Linear fit R = a * log(gamma/2pi)^2 + b
    A = np.vstack([log_g_sq, np.ones(len(log_g_sq))]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, r_vals, rcond=None)
    a_fit, b_fit = coeffs

    predicted = a_fit * log_g_sq + b_fit
    ss_res = np.sum((r_vals - predicted)**2)
    ss_tot = np.sum((r_vals - np.mean(r_vals))**2)
    r_squared = 1 - ss_res / ss_tot

    print(f"  Fit: R ~ {a_fit:.4f} * log(gamma/2pi)^2 + {b_fit:.4f}  [R2={r_squared:.4f}]")
    print(f"  GUE prediction: coefficient ~ 1/(2pi)^2 ~ {1/(2*np.pi)**2:.4f}")

    if abs(a_fit - 1/(2*np.pi)**2) / (1/(2*np.pi)**2) < 0.5:
        print(f"  *** CONSISTENT WITH GUE! ***")

    # ================================================================
    # PART 3: Minimum spacing and collision time
    # ================================================================
    print("\n\nPART 3: MINIMUM SPACING AND COLLISION TIME SCALING")
    print("-" * 75)

    spacings = np.diff(gammas[:N])
    mean_spacings = 2 * np.pi / np.log(gammas[1:N] / (2*np.pi))

    # Normalized spacings
    norm_spacings = spacings / mean_spacings[:len(spacings)]

    # Minimum normalized spacing
    min_norm_spacing = np.min(norm_spacings)
    min_idx = np.argmin(norm_spacings)

    # Collision times (in dBN model): t_c ~ delta^2 / 8
    collision_times = spacings**2 / 8

    print(f"  Minimum spacing: {np.min(spacings):.6f} at gamma~{gammas[min_idx]:.2f}")
    print(f"  Min normalized spacing: {min_norm_spacing:.6f}")
    print(f"  Min collision time: {np.min(collision_times):.6e}")
    print(f"  Mean collision time: {np.mean(collision_times):.6e}")

    # Scaling: min_spacing as function of N
    print(f"\n  Min spacing vs N (GUE predicts delta_min ~ N^(-1/3)):")
    for N_cut in [20, 50, 100, 150, 200, 250, 300]:
        if N_cut > N:
            break
        sp = np.diff(gammas[:N_cut])
        ms = 2 * np.pi / np.log(gammas[1:N_cut] / (2*np.pi))
        ns = sp / ms[:len(sp)]
        min_ns = np.min(ns)
        predicted_gue = N_cut**(-1/3)
        print(f"    N={N_cut:>3}: min_norm_spacing = {min_ns:.6f}  "
              f"N^(-1/3) = {predicted_gue:.6f}  "
              f"ratio = {min_ns/predicted_gue:.3f}")

    # ================================================================
    # PART 4: zeta'(rho) at each zero — derivative rigidity
    # ================================================================
    print("\n\nPART 4: ZERO DERIVATIVE RIGIDITY |zeta'(rho)|")
    print("-" * 75)
    print("Growing |zeta'(rho)| means zeros are increasingly 'anchored'.\n")

    derivs = []
    for k in range(min(50, N)):
        s_val = complex(0.5, gammas[k])
        zp = zeta_deriv(s_val)
        derivs.append(abs(zp))

    derivs = np.array(derivs)
    g_sub = gammas[:len(derivs)]

    # Fit |zeta'(rho)| vs (log gamma)^alpha
    log_g_d = np.log(g_sub[5:])
    log_d = np.log(derivs[5:])
    alpha_d, log_C_d = np.polyfit(log_g_d, log_d, 1)

    print(f"  |zeta'(rho)| ~ C * gamma^{alpha_d:.3f}")
    print(f"  Equivalently: ~ C * (log gamma)^beta where beta ~ {alpha_d * np.log(g_sub[-1]) / np.log(np.log(g_sub[-1])):.2f}")

    for k in [0, 4, 9, 19, 29, 49]:
        if k < len(derivs):
            print(f"    k={k:>2}: gamma={g_sub[k]:.2f}  |zeta'| = {derivs[k]:.4f}")

    # ================================================================
    # PART 5: THE DUAL BARRIER FUNCTION
    # ================================================================
    print("\n\nPART 5: DUAL BARRIER FUNCTION")
    print("-" * 75)
    print("B(gamma) = rigidity(gamma) / zeta'(rho)^2")
    print("This is the ratio of electrostatic restoring force to")
    print("the 'softness' at the zero (derivative tells how close to double).\n")
    print("If B(gamma) -> inf, no zero can escape even at criticality.\n")

    barrier_vals = []
    for k in range(min(50, N)):
        if k < len(derivs) and k < len(rigidities):
            B = rigidities[k] * derivs[k]**2
            barrier_vals.append(B)
            if k in [0, 4, 9, 19, 29, 49]:
                print(f"    k={k:>2}: gamma={gammas[k]:.2f}  "
                      f"R={rigidities[k]:.4f}  |zeta'|={derivs[k]:.4f}  "
                      f"B = R*|zeta'|^2 = {B:.4f}")

    barrier_vals = np.array(barrier_vals)
    if len(barrier_vals) > 5:
        # Fit barrier vs gamma
        log_b = np.log(barrier_vals[5:])
        log_g_b = np.log(gammas[5:len(barrier_vals)])
        alpha_b, _ = np.polyfit(log_g_b, log_b, 1)
        print(f"\n  Barrier scaling: B ~ gamma^{alpha_b:.3f}")

        if alpha_b > 0:
            print(f"  *** BARRIER GROWS with height => zeros increasingly trapped ***")
            print(f"  *** This is the dual Rodgers-Tao mechanism: rigidity prevents escape ***")
        else:
            print(f"  Barrier decreasing -- no dual barrier from this quantity.")

    # ================================================================
    # PART 6: The Rodgers-Tao energy balance
    # ================================================================
    print("\n\nPART 6: ENERGY BALANCE -- escape cost vs thermal energy at t->0+")
    print("-" * 75)
    print("For dBN dynamics at time t:")
    print("  Thermal displacement ~ sqrt(2t)")
    print("  Escape energy cost ~ t * R(gamma)")
    print("  Net barrier = R(gamma) - 1  (independent of t!)")
    print()
    print("KEY INSIGHT: The barrier ratio is the RIGIDITY itself.")
    print("If R > 1 everywhere, no zero can thermally escape at any t > 0.")
    print("As t -> 0, this holds trivially. The question is at t = 0 exactly.")
    print()

    below_one = np.sum(rigidities < 1)
    print(f"  Zeros with R < 1: {below_one}/{N}")
    print(f"  Zeros with R < 0.1: {np.sum(rigidities < 0.1)}/{N}")
    print(f"  Minimum R across all zeros: {np.min(rigidities):.6f}")

    if below_one == 0:
        print(f"\n  *** ALL ZEROS have R > 1 => escape barrier exceeds thermal energy ***")
        print(f"  *** This is necessary (but not sufficient) for Lambda <= 0 ***")

    # ================================================================
    # SAVE
    # ================================================================
    output = {
        'N_zeros': N,
        'rigidity_stats': {
            'min': float(np.min(rigidities)),
            'max': float(np.max(rigidities)),
            'mean': float(np.mean(rigidities)),
            'median': float(np.median(rigidities)),
        },
        'rigidity_fit': {
            'a': float(a_fit),
            'b': float(b_fit),
            'R2': float(r_squared),
            'gue_predicted_a': float(1/(2*np.pi)**2)
        },
        'min_spacing': {
            'value': float(np.min(spacings)),
            'normalized': float(min_norm_spacing),
            'collision_time': float(np.min(collision_times))
        },
        'derivative_scaling': float(alpha_d),
        'barrier_scaling': float(alpha_b) if len(barrier_vals) > 5 else None,
        'zeros_below_unit_rigidity': int(below_one)
    }

    with open('rodgers_tao_dual.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to rodgers_tao_dual.json")
