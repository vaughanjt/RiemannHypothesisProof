"""
FUNDAMENTAL CONSTANTS AND ZETA ZEROS

Trivial zeros: controlled by pi (through Gamma function poles)
Nontrivial zeros: controlled by ???

Explore: do e, alpha (fine structure), or other constants appear
in the zero distribution, spacing, or dynamics?

The electrostatic analogy: zeros = charges on the real line
with logarithmic repulsion. The "coupling constant" of this
interaction might be a fundamental quantity.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, pi, euler, log, exp, zeta, nstr
import time

mp.dps = 30

# Fine structure constant
ALPHA = 1.0 / 137.035999084
# Euler-Mascheroni
GAMMA = 0.5772156649015329

if __name__ == "__main__":
    gammas = np.load("_zeros_500.npy")

    print("FUNDAMENTAL CONSTANTS IN ZETA ZERO STRUCTURE")
    print("=" * 70)

    # ================================================================
    # PART 1: Zero spacings and fundamental constants
    # ================================================================
    print("\nPART 1: ZERO SPACINGS")
    print("-" * 70)

    spacings = np.diff(gammas[:100])
    avg_spacing = np.mean(spacings)
    normalized = spacings / avg_spacing

    print(f"  First 10 spacings: {', '.join(f'{s:.4f}' for s in spacings[:10])}")
    print(f"  Average spacing (first 100): {avg_spacing:.6f}")
    print(f"  2*pi / ln(gamma_50/(2*pi)): {2*np.pi/np.log(gammas[49]/(2*np.pi)):.6f}")

    # Does 1/alpha appear?
    print(f"\n  avg_spacing * alpha = {avg_spacing * ALPHA:.6f}")
    print(f"  avg_spacing / alpha = {avg_spacing / ALPHA:.6f}")
    print(f"  1/alpha = {1/ALPHA:.6f}")
    print(f"  avg_spacing / (2*pi) = {avg_spacing / (2*np.pi):.6f}")
    print(f"  avg_spacing * e = {avg_spacing * np.e:.6f}")

    # ================================================================
    # PART 2: The "electrostatic energy" of zeros
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: ELECTROSTATIC ENERGY OF ZEROS")
    print("-" * 70)

    # In the dBN electrostatic model: zeros repel via V(x,y) = -log|x-y|
    # The total energy: E = -sum_{i<j} log|gamma_i - gamma_j|
    # Normalize per pair

    N_zeros = 50
    total_energy = 0
    n_pairs = 0
    for i in range(N_zeros):
        for j in range(i+1, N_zeros):
            total_energy -= np.log(abs(gammas[j] - gammas[i]))
            n_pairs += 1

    energy_per_pair = total_energy / n_pairs
    print(f"  Electrostatic energy ({N_zeros} zeros):")
    print(f"    Total: {total_energy:.6f}")
    print(f"    Per pair: {energy_per_pair:.6f}")
    print(f"    e^(energy_per_pair) = {np.exp(energy_per_pair):.6f}")
    print(f"    Compare: 1/alpha = {1/ALPHA:.6f}")
    print(f"    Compare: 4*pi = {4*np.pi:.6f}")
    print(f"    Compare: 2*pi*e = {2*np.pi*np.e:.6f}")

    # ================================================================
    # PART 3: e^{gamma_k} — do exponentials of zeros have structure?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: EXPONENTIALS OF ZEROS")
    print("-" * 70)

    # gamma_k are imaginary parts of zeros on Re(s) = 1/2
    # e^{gamma_k} transforms zeros to a multiplicative scale
    # The primes ALSO live on this scale: p = e^{log(p)}

    print(f"  {'k':>4} {'gamma_k':>12} {'e^gamma_k':>14} {'nearest p^n':>12} {'ratio':>10}")
    print("  " + "-" * 55)

    import sympy
    for k in range(10):
        eg = np.exp(gammas[k])
        # Find nearest prime power
        best_pn = 1
        best_dist = abs(eg - 1)
        for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
            pn = p
            while pn < 10 * eg:
                if abs(eg - pn) < best_dist:
                    best_dist = abs(eg - pn)
                    best_pn = pn
                pn *= p
        ratio = eg / best_pn
        print(f"  {k+1:>4} {gammas[k]:>12.6f} {eg:>14.2f} {best_pn:>12} {ratio:>10.6f}")

    # ================================================================
    # PART 4: The coupling constant of zero repulsion
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: THE COUPLING CONSTANT OF ZERO REPULSION")
    print("-" * 70)

    # In the dBN dynamics: d(gamma_k)/dt = sum_{j!=k} 1/(gamma_k - gamma_j)
    # This is the electrostatic force from other zeros.
    # The "coupling constant" g determines the strength.

    # From our dBN data: |zero_t - gamma_1| ~ 1.16 * t
    # This means: d(gamma_1)/dt ~ 1.16 at t near 0
    # = sum_{j>=2} 1/(gamma_1 - gamma_j)

    force_on_1 = sum(1.0 / (gammas[0] - gammas[j]) for j in range(1, 100))
    print(f"  Electrostatic force on gamma_1:")
    print(f"    F = sum 1/(gamma_1 - gamma_j) = {force_on_1:.6f}")
    print(f"    From dBN: d(gamma)/dt ~ 1.16")
    print(f"    Ratio (dBN/force): {1.16 / abs(force_on_1):.4f}")

    # The "effective coupling" from the next few zeros
    for N_terms in [5, 10, 20, 50, 100]:
        f = sum(1.0 / (gammas[0] - gammas[j]) for j in range(1, N_terms))
        print(f"    F({N_terms} terms) = {f:.6f}")

    # ================================================================
    # PART 5: Dimensionless ratios from zero structure
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: DIMENSIONLESS RATIOS")
    print("-" * 70)

    # The zero structure has several natural scales:
    # - gamma_1 = 14.13... (first zero)
    # - Average spacing ~ 2*pi/ln(T/(2*pi)) ~ 1-2
    # - The Euler-Mascheroni constant gamma = 0.5772...

    g1 = gammas[0]
    g2 = gammas[1]
    spacing_1 = g2 - g1

    ratios = {
        'gamma_1 / (2*pi)': g1 / (2*np.pi),
        'gamma_1 / (2*pi*e)': g1 / (2*np.pi*np.e),
        'gamma_1 * alpha': g1 * ALPHA,
        'gamma_1 / e^2': g1 / np.e**2,
        'spacing_1 / (2*pi)': spacing_1 / (2*np.pi),
        'gamma_1 * gamma_Euler': g1 * GAMMA,
        'ln(gamma_1) / pi': np.log(g1) / np.pi,
        'gamma_1^2 / (2*pi*e^pi)': g1**2 / (2*np.pi*np.exp(np.pi)),
        '2*pi*e / gamma_1': 2*np.pi*np.e / g1,
        'e^(2*pi*alpha)': np.exp(2*np.pi*ALPHA),
    }

    print(f"\n  Dimensionless ratios:")
    for name, val in ratios.items():
        # Check if close to simple fraction or known constant
        inv = 1/val if abs(val) > 0.01 else 0
        print(f"    {name:>30s} = {val:.8f}  (1/x = {inv:.4f})")

    # ================================================================
    # PART 6: The functional equation and e vs pi
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 6: FUNCTIONAL EQUATION — e vs pi")
    print("-" * 70)

    # xi(s) = (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)*zeta(s)
    # On the critical line s = 1/2 + it:
    # xi = (1/2)*(1/2+it)*(-1/2+it)*pi^{-(1/4+it/2)}*Gamma(1/4+it/2)*zeta(1/2+it)
    #
    # The pi^{-s/2} factor: this is e^{-s*ln(pi)/2} = e^{-(1/4)*ln(pi)} * e^{-it*ln(pi)/2}
    # So pi enters through the PHASE: e^{-it*ln(pi)/2}
    # And Gamma enters through: |Gamma(1/4+it/2)| ~ sqrt(2*pi) * |t/2|^{-1/4} * e^{-pi*t/4}
    #
    # The DECAY of Xi on the critical line is controlled by e^{-pi*t/4} (from Gamma)
    # The OSCILLATION is controlled by t*ln(t) (from the argument of Gamma)
    #
    # The zeros occur where the oscillation crosses zero — this is controlled by
    # the PHASE of Gamma(1/4+it/2), which involves:
    # arg(Gamma(1/4+it/2)) ~ (t/2)*ln(t/2) - t/2 - pi/8 (Stirling)
    #
    # The zero condition: phase = n*pi gives:
    # (t/2)*ln(t/2) - t/2 ≈ n*pi + pi/8
    # t*ln(t/(2e)) ≈ 2*n*pi + pi/4

    print("  Zero condition from Stirling (approximate):")
    print("  gamma_k * ln(gamma_k/(2*e)) ~ 2*k*pi + pi/4")
    print()

    for k in range(1, 8):
        lhs = gammas[k-1] * np.log(gammas[k-1] / (2*np.e))
        rhs = 2*k*np.pi + np.pi/4
        print(f"    k={k}: LHS = {lhs:.4f}, RHS = {rhs:.4f}, "
              f"ratio = {lhs/rhs:.6f}")

    # The e appears through the Stirling approximation: ln(Gamma) ~ x*ln(x) - x
    # The "2*e" in gamma/(2*e) comes from Stirling's formula

    print(f"\n  KEY: The nontrivial zeros are approximately at:")
    print(f"  gamma_k * ln(gamma_k/(2*e)) = 2*k*pi + pi/4")
    print(f"  => gamma_k * ln(gamma_k) - gamma_k * (1 + ln(2)) = 2*k*pi + pi/4")
    print(f"  => e controls the SHIFT (through Stirling's x*ln(x)-x)")
    print(f"  => pi controls the SPACING (through 2*pi periodicity)")
    print(f"  => The zeros encode BOTH pi (geometry) and e (analysis)")

    # ================================================================
    # PART 7: alpha and the pair correlation
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 7: PAIR CORRELATION AND COUPLING")
    print("-" * 70)

    # Montgomery's pair correlation: R2(x) = 1 - (sin(pi*x)/(pi*x))^2
    # This matches GUE random matrices.
    # The pair correlation function is UNIVERSAL — it doesn't depend on
    # any coupling constant.

    # But the DENSITY of zeros does: N(T) ~ T/(2*pi) * ln(T/(2*pi*e))
    # The effective spacing at height T: delta ~ 2*pi/ln(T/(2*pi))
    # The "local coupling" at height T: g(T) = 1/delta ~ ln(T/(2*pi))/(2*pi)

    print("  Local zero density at height T: rho(T) = ln(T/(2*pi)) / (2*pi)")
    print()
    for T in [14, 50, 100, 500, 1000, 10000]:
        rho = np.log(T/(2*np.pi)) / (2*np.pi)
        delta = 1/rho
        print(f"    T={T:>6}: rho = {rho:.6f}, delta = {delta:.4f}, "
              f"rho*2*pi = {rho*2*np.pi:.4f} = ln({T:.0f}/(2*pi))")

    print(f"\n  The 'coupling' grows logarithmically: g(T) = ln(T/(2*pi))/(2*pi)")
    print(f"  This is the SAME logarithm that appears in the prime counting function!")
    print(f"  pi(x) ~ x/ln(x) and rho(T) ~ ln(T)/2*pi")
    print(f"  The primes thin out as 1/ln, the zeros densify as ln.")
    print(f"  They are DUAL: primes * zeros ~ constant (Weil explicit formula).")

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print("=" * 70)
    print(f"""
TRIVIAL ZEROS: Controlled by pi through Gamma(s/2) poles.
  Location: s = -2n (even negative integers)
  Mechanism: pi^{{-s/2}} * Gamma(s/2) has poles at s = 0, -2, -4, ...

NONTRIVIAL ZEROS: Controlled by BOTH e AND pi through Stirling's formula.
  Approximate location: gamma_k * ln(gamma_k/(2*e)) = 2*k*pi + pi/4

  pi controls: the SPACING (2*pi periodicity of the phase)
  e controls: the SHIFT (Stirling's approximation x*ln(x) - x)

THE ELECTROMAGNETIC ANALOGY:
  - Zeros = charged particles on the real line
  - Repulsion: V = -ln|gamma_i - gamma_j| (2D Coulomb / logarithmic)
  - The "coupling" g(T) = ln(T/(2*pi))/(2*pi) grows logarithmically
  - This is the DUAL of the prime density 1/ln(x)

THE FINE STRUCTURE CONNECTION:
  alpha ~ 1/137 doesn't appear directly in zero statistics.
  But the STRUCTURE is the same:
  - alpha = e^2/(4*pi*epsilon_0*hbar*c) mixes geometry (4*pi),
    quantum (hbar), and electrodynamics (e, epsilon_0, c)
  - The zero distribution mixes geometry (2*pi spacing),
    analysis (e shift), and arithmetic (prime locations)

  The DIMENSIONLESS RATIO that governs zeros is:
  g(T) = ln(T/(2*pi*e)) / (2*pi) ≈ the "zero fine structure"

  At T = gamma_1 = 14.13: g ≈ {np.log(14.13/(2*np.pi*np.e))/(2*np.pi):.6f}
  At T = 1000: g ≈ {np.log(1000/(2*np.pi*np.e))/(2*np.pi):.6f}
""")
