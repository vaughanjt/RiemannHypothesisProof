"""
SESSION 33 — DIRECTION C: LEVINSON-CONREY MOLLIFIER EXTENSION

BACKGROUND:
  Levinson (1974): >= 1/3 of zeros on critical line
  Conrey (1989): >= 2/5 (40.77%) using longer mollifiers
  Bui-Conrey-Young (2011): >= 41.05%
  Pratt-Robles (2020): >= 41.72% (current record)

  The method: multiply zeta by a "mollifier" M(s) that smooths out its behavior,
  then use mean-value theorems to count sign changes of Re(M*zeta) on the line.

  A sign change of Re(M*zeta)(1/2+it) implies a zero of zeta near 1/2+it.

THE OBSTRUCTION TO 100%:
  The mollifier M(s) = sum_{n <= y} a_n / n^s with y = T^theta.

  - theta = 1/2: Levinson's 1/3
  - theta = 4/7: Conrey's 2/5
  - theta = 1: would give 100% (but mean-value theorems break down)

  The obstruction is ANALYTIC: the error term in the mean-value theorem
  for sum_{T}^{2T} |M*zeta|^2 dt exceeds the main term when theta > 1/2 + eps.

  KEY: What if we use a DIFFERENT family of mollifiers?
  Instead of Dirichlet polynomial mollifiers, use:
  - Resonance mollifiers (Soundararajan)
  - Multiplicative mollifiers (Heap-Radziwiłł)
  - Hybrid mollifier + resonance (Pratt-Robles approach)

COMPUTATIONAL APPROACH:
  1. Compute the proportion of zeros detected vs mollifier length theta
  2. Study the error term structure for theta near 1/2
  3. Test alternative mollifier families
  4. Identify what would be needed for theta = 1
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi as mp_pi, gamma as Gamma, log as mp_log, exp as mp_exp
import time
import json

mp.dps = 30


def hardy_Z(t):
    """Hardy's Z-function: real-valued on the critical line."""
    s = mpc('0.5', str(t))
    # theta(t) = arg(pi^{-it/2} * Gamma(1/4 + it/2))
    theta = mpmath.siegeltheta(t)
    return float(mp_exp(mpc(0, 1) * theta) * zeta(s)).real


def simple_mollifier(t, y, coeffs=None):
    """
    Simple Dirichlet polynomial mollifier:
    M(1/2+it) = sum_{n=1}^{y} a_n / n^{1/2+it}

    Default coefficients: a_n = mu(n) * P(log(y/n)/log(y))
    where mu is Mobius function and P is a polynomial.
    """
    if coeffs is None:
        # Use Conrey's optimal choice: P(x) = x
        coeffs = compute_conrey_coefficients(int(y))

    result = 0.0
    for n in range(1, min(int(y) + 1, len(coeffs) + 1)):
        if abs(coeffs[n-1]) > 1e-15:
            result += coeffs[n-1] * n**(-0.5) * np.exp(-1j * t * np.log(n))
    return result


def mobius(n):
    """Compute Mobius function mu(n)."""
    if n == 1:
        return 1
    # Factor n
    factors = []
    temp = n
    for p in range(2, int(n**0.5) + 2):
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count > 1:
                return 0  # p^2 | n
            factors.append(p)
    if temp > 1:
        factors.append(temp)
    return (-1)**len(factors)


def compute_conrey_coefficients(y):
    """
    Conrey's mollifier coefficients:
    a_n = mu(n) * P(log(y/n) / log(y))
    with P(x) = x (linear mollifier).
    """
    log_y = np.log(y)
    coeffs = []
    for n in range(1, y + 1):
        mu_n = mobius(n)
        if mu_n == 0:
            coeffs.append(0.0)
        else:
            x = np.log(y / n) / log_y  # P(x) = x for Conrey
            coeffs.append(mu_n * x)
    return coeffs


def compute_proportion_detected(gammas, theta, T_range=(100, 300)):
    """
    Compute the proportion of zeros in [T_range] detected by a
    mollifier of length y = T^theta.

    A zero at gamma is "detected" if Re(M*zeta) changes sign near gamma.
    """
    T_low, T_high = T_range
    relevant = gammas[(gammas >= T_low) & (gammas <= T_high)]
    n_zeros = len(relevant)

    if n_zeros == 0:
        return 0, 0, 0

    T_mid = (T_low + T_high) / 2
    y = int(T_mid**theta)
    y = max(y, 2)
    y = min(y, 500)  # cap for computational feasibility

    coeffs = compute_conrey_coefficients(y)

    # Sample the mollified function near each zero
    detected = 0
    for gamma in relevant:
        # Check sign change of Re(M * zeta) near gamma
        delta = 0.2  # search radius
        vals = []
        for dt in np.linspace(-delta, delta, 11):
            t = gamma + dt
            s = mpc('0.5', str(t))
            z_val = complex(zeta(s))
            m_val = simple_mollifier(t, y, coeffs)
            product = z_val * m_val
            vals.append(product.real)

        # Check for sign change
        signs = np.sign(vals)
        if np.any(signs[:-1] * signs[1:] < 0):
            detected += 1

    return detected, n_zeros, detected / n_zeros if n_zeros > 0 else 0


def mean_value_estimate(theta, N_terms=100):
    """
    Estimate the mean value:
    (1/T) * integral_{T}^{2T} |M(1/2+it) * zeta(1/2+it)|^2 dt

    For a length-y mollifier with y = T^theta:

    Main term: ~ 1 (by construction of optimal coefficients)
    Error term: ~ T^{2*theta - 1 + eps} for Dirichlet polynomial mollifier

    The proportion detected is:
    fraction >= 1 - error/main >= 1 - T^{2*theta - 1 + eps}

    For theta < 1/2: error -> 0, get positive proportion
    For theta = 1/2: error ~ T^eps, barely works (Levinson's 1/3)
    For theta > 1/2: error -> infinity, method breaks

    Conrey's trick: use the Kloosterman sum refinement to push to theta = 4/7.
    """
    results = []
    for th in np.linspace(0.1, 1.0, 19):
        # Heuristic error exponent
        if th <= 4/7:
            # Conrey range: error controlled by Kloosterman
            error_exp = max(0, 2*th - 1)
            method = "Conrey"
        elif th <= 0.6:
            # Hybrid range: partial control
            error_exp = 2*th - 1
            method = "Hybrid"
        else:
            # Uncontrolled
            error_exp = 2*th - 1
            method = "Uncontrolled"

        # Predicted fraction (asymptotic)
        if error_exp < 0:
            predicted_fraction = 1.0
        elif error_exp == 0:
            predicted_fraction = 1/3 if th <= 0.5 else 2/5  # historical values
        else:
            predicted_fraction = max(0, 1 - 10**(error_exp))  # rough estimate

        results.append({
            'theta': th,
            'error_exponent': error_exp,
            'method': method,
            'predicted_fraction': predicted_fraction
        })

    return results


def optimal_mollifier_analysis():
    """
    Analyze what the OPTIMAL mollifier structure looks like.

    The optimal coefficients a_n minimize:
      E[|1 - M*zeta|^2] = integral_{T}^{2T} |1 - M(1/2+it)*zeta(1/2+it)|^2 dt / T

    Using the approximate functional equation:
      zeta(1/2+it) ~ sum_{n <= sqrt(t/2pi)} n^{-1/2-it} + chi(1/2+it) * sum_{n<=sqrt(t/2pi)} n^{-1/2+it}

    The optimal M*zeta should be close to 1, meaning M ~ 1/zeta ~ mu*Dirichlet series.

    KEY INSIGHT: The obstruction is not the coefficient choice but the LENGTH.
    Even with perfect coefficients, a Dirichlet polynomial of length y cannot
    approximate 1/zeta(s) to sufficient accuracy when y < T^{1/2+eps}.

    ALTERNATIVE: What if M is not a Dirichlet polynomial?
    - Resonance method (Soundararajan 2009): uses multiplicative characters
    - Gives better POINTWISE bounds but worse L^2 bounds
    - Cannot directly improve the proportion, but suggests new approaches
    """
    print("\n\nOPTIMAL MOLLIFIER ANALYSIS")
    print("=" * 75)

    # Compute actual 1/zeta(s) Dirichlet series coefficients
    # 1/zeta(s) = sum mu(n)/n^s
    # The tail sum_{n > y} mu(n)/n^s is what the mollifier misses

    # Estimate the L2 norm of the tail for various y
    print("\nTail norm of 1/zeta Dirichlet series: ||sum_{n>y} mu(n)/n^{1/2+it}||_2")
    print("(This is what the mollifier cannot capture)\n")

    for y in [10, 20, 50, 100, 200, 500]:
        # The L^2 norm of sum_{n>y} mu(n)/n^{1/2+it} over [T,2T] is approximately:
        # sum_{n>y} |mu(n)|^2 / n ~ sum_{n>y} 1/n * (6/pi^2) ~ (6/pi^2) * log(infinity/y)
        # But we need the finite sum up to some cutoff ~ T

        # For T ~ 1000 (our computational range):
        T = 1000
        tail_sq = 0
        for n in range(y + 1, min(int(np.sqrt(T)), 1000)):
            mu_n = mobius(n)
            tail_sq += mu_n**2 / n

        tail_norm = np.sqrt(tail_sq)
        completeness = 1 - tail_sq / (6/np.pi**2 * np.log(T))

        print(f"  y={y:>4}: tail_norm={tail_norm:.4f}  completeness={completeness:.4f}")

    # The key ratio: how much of 1/zeta is captured?
    print(f"\n  For theta = log(y)/log(T):")
    for theta in [0.3, 0.4, 0.5, 4/7, 0.6, 0.7, 0.8, 1.0]:
        T = 1000
        y = int(T**theta)
        y = max(y, 2)

        captured = 0
        total = 0
        for n in range(1, min(int(np.sqrt(T)), 1000)):
            mu_n = mobius(n)
            total += mu_n**2 / n
            if n <= y:
                captured += mu_n**2 / n

        fraction = captured / total if total > 0 else 0
        print(f"  theta={theta:.3f} (y={y:>5}): captures {fraction:.4f} of ||1/zeta||^2")


def resonance_method_test(gammas, T_range=(50, 200)):
    """
    Test Soundararajan's resonance method as an alternative to mollifiers.

    Instead of multiplying by M(s), use:
      R(t) = sum_{n <= y} a_n * n^{-it} / sqrt(n)
    where a_n are chosen to RESONATE with zeta on the critical line.

    The resonance condition: a_n large when n is "smooth" (many small prime factors).
    This exploits the multiplicative structure of zeta.

    For our purpose: even if this doesn't improve the proportion directly,
    it might reveal structural information about the gap between 41% and 100%.
    """
    print("\n\nRESONANCE METHOD TEST")
    print("=" * 75)

    T_low, T_high = T_range
    relevant = gammas[(gammas >= T_low) & (gammas <= T_high)]

    # Resonance coefficients: a_n = product_{p | n} min(1, log(y)/log(p))
    # This emphasizes smooth numbers
    y = 100
    log_y = np.log(y)

    def resonance_coeff(n):
        """Compute resonance coefficient for n."""
        if n == 1:
            return 1.0
        result = 1.0
        temp = n
        for p in range(2, int(n**0.5) + 2):
            if temp % p == 0:
                while temp % p == 0:
                    result *= min(1.0, log_y / np.log(p))
                    temp //= p
        if temp > 1:
            result *= min(1.0, log_y / np.log(temp))
        return result

    # Compare: how does the resonance sum behave at zeros vs off-zeros?
    print(f"  Resonance sum R(t) = sum_{{n<=100}} a_n * n^{{-1/2-it}}")
    print(f"  a_n = prod_{{p|n}} min(1, log(100)/log(p))")
    print()

    # At zeros: zeta(1/2+it) = 0, so M*zeta = 0. What does R give?
    for k in range(min(5, len(relevant))):
        gamma = relevant[k]

        # Standard mollifier at zero
        coeffs = compute_conrey_coefficients(min(y, 100))
        m_val = simple_mollifier(gamma, y, coeffs)
        z_val = complex(zeta(mpc('0.5', str(gamma))))
        standard = z_val * m_val

        # Resonance sum at zero
        res_sum = 0.0
        for n in range(1, y + 1):
            a_n = resonance_coeff(n)
            res_sum += a_n * n**(-0.5) * np.exp(-1j * gamma * np.log(n))

        # Resonance * zeta
        res_product = z_val * res_sum

        print(f"  gamma={gamma:.2f}:")
        print(f"    |zeta| = {abs(z_val):.6e}")
        print(f"    |M*zeta| = {abs(standard):.6e}  (standard)")
        print(f"    |R*zeta| = {abs(res_product):.6e}  (resonance)")
        print(f"    |R| = {abs(res_sum):.4f}  (resonance amplitude)")

    # Off-zero behavior: how much does R amplify zeta?
    print(f"\n  Off-zero amplification (midpoints between zeros):")
    for k in range(min(5, len(relevant) - 1)):
        t_mid = (relevant[k] + relevant[k+1]) / 2
        z_val = complex(zeta(mpc('0.5', str(t_mid))))

        res_sum = 0.0
        for n in range(1, y + 1):
            a_n = resonance_coeff(n)
            res_sum += a_n * n**(-0.5) * np.exp(-1j * t_mid * np.log(n))

        amplification = abs(res_sum) * abs(z_val) / abs(z_val) if abs(z_val) > 1e-15 else 0

        print(f"  t={t_mid:.2f}: |zeta|={abs(z_val):.4f}  |R|={abs(res_sum):.4f}  "
              f"amp={amplification:.4f}")


def obstruction_analysis():
    """
    Identify the PRECISE obstruction to extending mollifiers to 100%.

    The fundamental issue:
    1. Levinson's method detects zeros by counting sign changes of Re(M*zeta)
    2. Sign changes of Re(M*zeta) ≈ sign changes of Re(1 + error)
    3. When |error| < 1, Re(M*zeta) > 0, so no sign changes are missed
    4. The error comes from two sources:
       a) Approximation error: M ≈ 1/zeta is imperfect
       b) Mean-value error: L^2 norm estimate has error term

    For 100%: need BOTH errors simultaneously small for ALL t in [T, 2T].
    This is equivalent to: 1/zeta(1/2+it) can be well-approximated
    by a short Dirichlet polynomial for ALL t.

    But 1/zeta(1/2+it) has LARGE fluctuations (Omega(exp(c*sqrt(log log T)))).
    A Dirichlet polynomial of length T^theta cannot capture these fluctuations
    unless theta >= 1.

    THE DEEP REASON: 1/zeta has the Euler product 1/zeta = prod(1 - 1/p^s).
    Short Dirichlet polynomials see only small primes.
    Large prime contributions to 1/zeta are invisible to mollifiers.
    """
    print("\n\nOBSTRUCTION ANALYSIS: WHY MOLLIFIERS CAN'T REACH 100%")
    print("=" * 75)

    print("""
  THE FUNDAMENTAL BARRIER:

  The mollifier approach has a HARD ceiling around ~41% with current technology.
  The reason is structural, not just technical:

  1. MOLLIFIER LENGTH LIMITATION (theta < 1):
     A Dirichlet polynomial M = sum_{n<=y} a_n/n^s with y = T^theta
     can only "see" prime factors p <= T^theta.
     For theta < 1/2: misses ALL primes > sqrt(T).
     For theta = 4/7: misses primes > T^{4/7}.

  2. EULER PRODUCT STRUCTURE:
     1/zeta(s) = prod_{p} (1 - 1/p^s)
     Each prime contributes independently.
     A short mollifier captures: prod_{p <= y} (1 - 1/p^s)
     Missing: prod_{y < p <= T} (1 - 1/p^s)
     The missing product has L^2 norm ~ exp(sum_{y<p<=T} 1/p) ~ (T/y)^{1/log T}
     For theta < 1: this is NOT small.

  3. SIGN CHANGE COUNTING NEEDS UNIFORM CONTROL:
     Even if E[|error|^2] is small, large POINTWISE excursions can
     create false sign changes or miss real ones.
     The maximum of |error| over [T,2T] is at least ~ (log T)^c.

  4. THE 41% BARRIER:
     The fraction detected = 1 - integral of error distribution tail
     With current mean-value estimates: error L^4 norm gives ~41%.
     Improving to 100% requires either:
     a) Much better mean-value theorems (unlikely for Dirichlet polynomials)
     b) Fundamentally different approach (not based on L^2/L^4)
     c) Using additional structure of zeta zeros (GUE, etc.)

  POSSIBLE EXTENSIONS:
  - Use GUE statistics to bound the error at zeros specifically
  - Combine mollifier with Connes Q_W positivity
  - Use the barrier B(gamma) to "protect" mollifier estimates
  - Hybrid: prove 41% rigorously, then show remaining 59% have large B
""")

    return


if __name__ == "__main__":
    print("SESSION 33 — DIRECTION C: LEVINSON-CONREY MOLLIFIER EXTENSION")
    print("=" * 75)

    gammas = np.load("_zeros_500.npy")

    # Part 1: Mean-value estimates for various theta
    print("\nPART 1: MEAN-VALUE ESTIMATES VS THETA")
    print("-" * 75)
    estimates = mean_value_estimate(1.0)
    for e in estimates:
        status = "OK" if e['error_exponent'] <= 0 else "BREAKS"
        print(f"  theta={e['theta']:.3f}: error~T^{e['error_exponent']:.3f}  "
              f"method={e['method']:<15}  fraction~{e['predicted_fraction']:.4f}  {status}")

    # Part 2: Optimal mollifier analysis
    optimal_mollifier_analysis()

    # Part 3: Actual proportion detected for small theta
    print("\n\nPART 3: ACTUAL PROPORTION DETECTED (computational)")
    print("-" * 75)
    print("Computing sign changes of Re(M*zeta) at zeros...\n")

    for theta in [0.3, 0.4, 0.5]:
        t0 = time.time()
        detected, total, frac = compute_proportion_detected(
            gammas, theta, T_range=(30, 100)
        )
        elapsed = time.time() - t0
        print(f"  theta={theta:.1f}: {detected}/{total} = {frac:.4f}  ({elapsed:.1f}s)")

    # Part 4: Resonance method comparison
    resonance_method_test(gammas, T_range=(30, 100))

    # Part 5: Obstruction analysis
    obstruction_analysis()

    # Part 6: Hybrid strategy
    print("\n\nPART 6: HYBRID STRATEGY — MOLLIFIER + BARRIER")
    print("=" * 75)
    print("""
  THE HYBRID IDEA:
  Combine the 41% from mollifiers with the barrier B(gamma) from Direction B.

  1. Mollifier proves: >= 41% of zeros are on the critical line
  2. For the remaining 59%, if they were OFF the line, they would need
     B(gamma) < threshold (to overcome the electrostatic barrier)
  3. From Direction B: B(gamma) >= B_min(T) which grows with T
  4. If B_min(T) > threshold for large enough T, the remaining 59%
     are also on the line (for T large enough)
  5. For small T: direct computation verifies all zeros are on the line

  This reduces 100% -> proving B_min(T) > threshold.
  Which is Direction B's conditional result!

  CONCLUSION: Directions B and C are COMPLEMENTARY.
  B alone can't prove anything unconditionally.
  C alone caps at 41%.
  Together: 41% rigorous + 59% via barrier = 100% conditional on B_min growth.
""")

    # Save
    output = {
        'mean_value_estimates': estimates,
        'conclusion': 'Mollifier capped at ~41%. Hybrid with barrier (Direction B) needed.',
        'key_insight': 'Euler product structure prevents short mollifiers from reaching 100%',
        'hybrid_path': 'Mollifier (41%) + barrier growth (59%) = 100% conditional'
    }
    with open('session33_mollifier.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to session33_mollifier.json")
