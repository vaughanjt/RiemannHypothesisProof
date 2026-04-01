"""
SESSION 32 SYNTHESIS -- Connecting Connes Q_W and Rodgers-Tao dual barrier.

KEY DISCOVERY: Both analyses reveal the SAME critical structure.

FROM CONNES:
  - eps_0 ~ lambda^{-0.6} (vanishing, confirming criticality)
  - Gap ratio eps_1/eps_0 diverges up to 203x (critical slowing down)
  - Cancellation factor ~ lambda^{0.9} (growing, but eps_0 still positive)
  - Tracy-Widom scaling FAILS -- this is NOT random matrix edge behavior

FROM RODGERS-TAO:
  - Barrier B = R * |zeta'|^2 grows as gamma^{1.5}
  - Higher zeros are increasingly locked in place
  - First zeros are weakest (gamma_1 has B = 0.033)
  - But first zeros are computationally verified

THE CONNECTION:
The Connes Q_W eps_0 IS the "barrier" for the Weil distribution.
The Rodgers-Tao rigidity IS the barrier for the heat flow.
Both are encoding the SAME physics: the cost of moving zeros off-line.

NEW ANALYSIS:
1. Does the Connes eps_0 decay rate match the Rodgers-Tao barrier?
2. Can we relate lambda in Connes to gamma in Rodgers-Tao?
3. Is there a UNIVERSAL critical exponent?
4. What happens at the "vulnerable" first zeros?
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, exp, gamma as Gamma, log, sinh
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 30


def xi_function(z):
    z_mp = mpc(z)
    s = mpf('0.5') + mpc(0, 1) * z_mp
    try:
        return complex(mpf('0.5') * s * (s-1) * mpmath.power(pi, -s/2) * Gamma(s/2) * zeta(s))
    except:
        return 0.0


def zeta_deriv(s_val, h=1e-8):
    s_mp = mpc(s_val)
    return complex((zeta(s_mp + h) - zeta(s_mp - h)) / (2*h))


if __name__ == "__main__":
    print("SESSION 32 SYNTHESIS")
    print("=" * 75)

    gammas = np.load("_zeros_500.npy")
    N_zeros = min(200, len(gammas))

    # ================================================================
    # PART 1: The connection between lambda and gamma
    # ================================================================
    print("\nPART 1: CONNES lambda vs ZERO HEIGHT gamma")
    print("-" * 75)
    print("In Connes framework, lambda^2 is the cutoff W = {n : n <= lambda^2}.")
    print("The number of zeros up to height T is ~ (T/2pi)*log(T/2pi).")
    print("The primes up to lambda^2 contribute to Q_W.")
    print("Connection: lambda^2 ~ e^{2*pi*gamma/log(gamma)} (by PNT)")
    print()

    # For each lambda^2, which zeros does it "see"?
    # The Weil explicit formula connects sum over primes <= lambda to sum over zeros
    # The effective range: primes up to lambda^2 capture information about
    # zeros up to height ~ lambda^2 / (2*pi)
    for lam_sq in [50, 200, 1000, 5000]:
        # How many primes up to lam_sq?
        n_primes = sum(1 for p in range(2, lam_sq+1)
                       if all(p % d != 0 for d in range(2, min(int(p**0.5)+1, p))))
        # Effective zero height from Weil duality
        effective_height = lam_sq / (2*np.pi)
        # How many zeros up to that height?
        n_zeros_approx = effective_height / (2*np.pi) * np.log(effective_height / (2*np.pi))
        print(f"  lam^2={lam_sq:>5}: {n_primes:>4} primes, "
              f"effective height~{effective_height:.1f}, "
              f"~{n_zeros_approx:.0f} zeros in range")

    # ================================================================
    # PART 2: Rigidity at heights corresponding to Connes cutoffs
    # ================================================================
    print("\n\nPART 2: RIGIDITY AT CONNES-RELEVANT HEIGHTS")
    print("-" * 75)

    # For each lambda, compute rigidity at the zeros in its range
    for lam_sq in [50, 200, 1000, 5000]:
        effective_height = lam_sq / (2*np.pi)
        # Find zeros up to this height
        relevant_zeros = gammas[gammas < effective_height]
        n_rel = len(relevant_zeros)
        if n_rel < 2:
            print(f"  lam^2={lam_sq}: only {n_rel} zeros in range, skip")
            continue

        # Compute rigidity for each relevant zero
        rigidities = []
        for k in range(n_rel):
            R = 0.0
            for j in range(n_rel):
                if j != k:
                    delta = relevant_zeros[k] - relevant_zeros[j]
                    if abs(delta) > 1e-15:
                        R += 1.0 / delta**2
            rigidities.append(R)

        rigidities = np.array(rigidities)
        min_R = np.min(rigidities)
        mean_R = np.mean(rigidities)
        below_1 = np.sum(rigidities < 1)

        print(f"  lam^2={lam_sq}: {n_rel} zeros, "
              f"min_R={min_R:.4f}, mean_R={mean_R:.4f}, "
              f"below_1={below_1}/{n_rel}")

    # ================================================================
    # PART 3: The critical exponent from both sides
    # ================================================================
    print("\n\nPART 3: CRITICAL EXPONENT COMPARISON")
    print("-" * 75)
    print("Connes: eps_0 ~ lambda^{-alpha_C}")
    print("Rodgers-Tao: B(gamma) ~ gamma^{alpha_R}")
    print("Connection: if lambda^2 ~ gamma, then alpha_C should relate to alpha_R")
    print()

    # From our data:
    alpha_C = 0.597  # eps_0 vs lambda exponent
    alpha_R = 1.534  # barrier vs gamma exponent

    print(f"  Connes alpha (eps_0 decay): {alpha_C:.3f}")
    print(f"  Rodgers-Tao alpha (barrier growth): {alpha_R:.3f}")
    print(f"  If lambda^2 ~ gamma: eps_0 ~ gamma^{{-{alpha_C/2:.3f}}}")
    print(f"  Barrier-adjusted: B * eps_0 ~ gamma^{{{alpha_R - alpha_C/2:.3f}}}")
    print()

    combined = alpha_R - alpha_C / 2
    if combined > 0:
        print(f"  *** COMBINED BARRIER GROWS: B * eps_0 ~ gamma^{{{combined:.3f}}} ***")
        print(f"  This means the product of Connes positivity and RT rigidity")
        print(f"  grows with height -- the proof gets EASIER for higher zeros.")
    else:
        print(f"  Combined barrier shrinks -- problematic.")

    # ================================================================
    # PART 4: First zero vulnerability analysis
    # ================================================================
    print("\n\nPART 4: FIRST ZERO VULNERABILITY")
    print("-" * 75)
    print("gamma_1 = 14.134... is the weakest zero. Quantify the barrier.")
    print()

    # Connes at lam^2 corresponding to gamma_1
    # gamma_1 ~ 14.13, so lam^2 ~ 2*pi*gamma_1 ~ 89
    lam_sq_gamma1 = int(2 * np.pi * gammas[0])
    print(f"  gamma_1 = {gammas[0]:.4f}")
    print(f"  Corresponding Connes cutoff: lam^2 ~ {lam_sq_gamma1}")

    # Build Q_W at this cutoff
    L_f = np.log(lam_sq_gamma1)
    N = max(21, round(8 * L_f))
    W02, M, QW = build_all(lam_sq_gamma1, N)
    evals_qw = np.linalg.eigvalsh(QW)
    eps_0 = evals_qw[0]
    print(f"  eps_0 at lam^2={lam_sq_gamma1}: {eps_0:.6e}")

    # Rigidity at gamma_1
    R_gamma1 = 0
    for j in range(1, min(50, N_zeros)):
        delta = gammas[0] - gammas[j]
        R_gamma1 += 1.0 / delta**2
    print(f"  Rigidity R(gamma_1) = {R_gamma1:.6f}")

    # Derivative at gamma_1
    zp = zeta_deriv(complex(0.5, gammas[0]))
    print(f"  |zeta'(rho_1)| = {abs(zp):.6f}")

    # Combined barrier
    B1 = R_gamma1 * abs(zp)**2
    print(f"  Barrier B(gamma_1) = R * |zeta'|^2 = {B1:.6f}")

    # How much displacement can gamma_1 tolerate?
    # From the energy argument: escape requires E > barrier
    # Thermal energy at time t: E ~ t
    # So gamma_1 safe for t < B1
    print(f"  gamma_1 protected for dBN t < {B1:.6f}")
    print(f"  (Rodgers-Tao proved Lambda >= 0, so we need t=0 protection)")

    # ================================================================
    # PART 5: The spectral gap as criticality marker
    # ================================================================
    print("\n\nPART 5: SPECTRAL GAP DIVERGENCE (criticality signature)")
    print("-" * 75)

    gaps = []
    for lam_sq in [20, 50, 100, 200, 500, 1000, 2000, 5000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        W02, M, QW = build_all(lam_sq, N)
        evals = np.linalg.eigvalsh(QW)
        eps_0, eps_1 = evals[0], evals[1]
        ratio = eps_1 / eps_0 if eps_0 > 0 else float('inf')
        gaps.append((lam_sq, eps_0, eps_1, ratio))
        print(f"  lam^2={lam_sq:>5}: eps_0={eps_0:.4e} eps_1={eps_1:.4e} ratio={ratio:.1f}")
        sys.stdout.flush()

    # Fit gap ratio vs lambda
    lam_arr = np.array([g[0] for g in gaps])
    ratio_arr = np.array([g[3] for g in gaps])
    log_ratio = np.log(ratio_arr[ratio_arr > 1])
    log_lam = np.log(lam_arr[:len(log_ratio)])
    if len(log_ratio) >= 3:
        alpha_gap, _ = np.polyfit(log_lam, log_ratio, 1)
        print(f"\n  Gap ratio ~ lambda^{{{alpha_gap:.3f}}}")
        print(f"  In statistical physics: gap ~ L^{{-z}} where z is dynamic exponent")
        print(f"  Here: gap_ratio ~ lambda^{{{alpha_gap:.3f}}} => z = {alpha_gap:.3f}")

    # ================================================================
    # PART 6: Summary and next steps
    # ================================================================
    print("\n\n" + "=" * 75)
    print("SYNTHESIS SUMMARY")
    print("=" * 75)
    print()
    print("1. RH IS CRITICAL (Lambda=0):")
    print(f"   - eps_0 ~ lambda^{{-0.60}} (vanishing)")
    print(f"   - Spectral gap ratio diverges (up to 203x)")
    print(f"   - Cancellation factor ~ lambda^{{0.89}} (growing)")
    print()
    print("2. BARRIER GROWS WITH HEIGHT:")
    print(f"   - B(gamma) ~ gamma^{{1.53}} (from rigidity * derivative^2)")
    print(f"   - Combined Connes+RT barrier grows as gamma^{{1.23}}")
    print(f"   - Higher zeros are SAFER (consistent with all approaches)")
    print()
    print("3. FIRST ZERO IS WEAKEST:")
    print(f"   - gamma_1 = 14.13 has min barrier B = {B1:.4f}")
    print(f"   - Computationally verified to ~10^13 zeros")
    print(f"   - The proof must handle the transition from finite to infinite")
    print()
    print("4. TRACY-WIDOM DOES NOT APPLY:")
    print(f"   - eps_0 * D_null^(2/3) does NOT converge")
    print(f"   - The Q_W structure is different from GUE edge statistics")
    print(f"   - The cancellation mechanism is specific to the Weil distribution")
    print()
    print("5. PROMISING PROOF PATH:")
    print("   a. Prove B(gamma) -> infinity analytically (GUE + Keating-Snaith)")
    print("   b. Find explicit C s.t. B(gamma) > 1 for gamma > C")
    print("   c. Verify B > 0 for gamma < C computationally")
    print("   d. Connect to Connes Q_W via Weil explicit formula")
    print("   e. This gives Lambda <= 0, hence Lambda = 0, hence RH")
    print()
    print("OBSTRUCTION: Step (a) requires proving that the PRODUCT of")
    print("electrostatic rigidity and zero derivative grows. This is")
    print("essentially proving GUE universality for the Riemann zeros,")
    print("which is itself a major open problem (Montgomery conjecture).")
    print()
    print("POSSIBLE BYPASS: If we can prove eps_0 > 0 for ALL lambda")
    print("directly in the Connes framework (not relating to RT barrier),")
    print("that gives RH by Connes' theorem. The eps_0 > 0 statement")
    print("is equivalent to: the Weil distribution is positive on test")
    print("functions of bandwidth <= lambda. This is a statement about")
    print("the PRIMES, not the zeros -- and may be attackable via")
    print("sieve theory or the Selberg formula.")

    # Save synthesis
    output = {
        'connes_alpha': alpha_C,
        'rt_alpha': alpha_R,
        'combined_alpha': float(combined),
        'gamma1_barrier': float(B1),
        'gamma1_rigidity': float(R_gamma1),
        'gamma1_deriv': float(abs(zp)),
        'gap_exponent': float(alpha_gap) if len(log_ratio) >= 3 else None,
    }
    with open('session32_synthesis.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to session32_synthesis.json")
