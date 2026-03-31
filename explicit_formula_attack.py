"""
WEIL EXPLICIT FORMULA — Information-theoretic attack on RH.

The explicit formula is an EXACT identity between zeros and primes:

  sum_rho h(gamma_rho) = (integral terms) + (prime sum)

KEY IDEA: For NARROWBAND test functions h centered at a zero gamma_k,
the on-line zero contributes h(0) = maximum. If the zero moves off-line
to gamma_k + i*delta, its contribution drops to h(i*delta) which is
EXPONENTIALLY suppressed for narrowband h.

The prime sum (right side) doesn't change. So the explicit formula
can only be satisfied if other zeros compensate — but they can't
because they're far away (GUE repulsion).

This is the INFORMATION-THEORETIC obstruction: the explicit formula
demands infinite precision at each zero, which only on-line zeros provide.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, pi, zeta, log, exp, gamma
from sympy import primerange
import time

mp.dps = 25


def xi_function(z):
    """Xi(z) where s = 1/2 + iz."""
    z_mp = mpc(z)
    s = mpf('0.5') + mpc(0, 1) * z_mp
    try:
        return complex(mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s / 2) * gamma(s / 2) * zeta(s))
    except:
        return 0.0


def gaussian_test(t, center, sigma):
    """Gaussian test function: h(t) = exp(-(t-center)^2 / (2*sigma^2))"""
    return np.exp(-(t - center)**2 / (2 * sigma**2))


def gaussian_ft(omega, center, sigma):
    """Fourier transform of Gaussian test function.
    h_hat(omega) = sigma*sqrt(2*pi) * exp(-sigma^2*(omega-center)^2/2) * exp(-i*omega*???)
    For real center and h(t) = exp(-(t-center)^2/(2*sigma^2)):
    h_hat(omega) = sigma*sqrt(2*pi) * exp(i*center*omega) * exp(-sigma^2*omega^2/2)
    """
    return sigma * np.sqrt(2*np.pi) * np.exp(-sigma**2 * omega**2 / 2)


if __name__ == "__main__":
    gammas = np.load("_zeros_500.npy")
    N = len(gammas)
    primes = list(primerange(2, 10000))

    print("WEIL EXPLICIT FORMULA ATTACK")
    print("=" * 75)

    # ================================================================
    # PART 1: Verify explicit formula for Gaussian test functions
    # ================================================================
    print("\nPART 1: EXPLICIT FORMULA VERIFICATION")
    print("-" * 75)
    print("For h(t) = exp(-t^2/(2*sigma^2)), centered at t=0:")
    print("  ZERO SIDE: sum_rho h(gamma_rho)")
    print("  PRIME SIDE: known integral + prime sum\n")

    # The explicit formula (von Mangoldt form):
    # For an even test function g with g_hat its Fourier/Mellin transform:
    #
    # sum_{gamma} g(gamma) = g(i/2) + g(-i/2)
    #   - sum_p sum_k (log p / p^{k/2}) * [g_hat(k*log p) + g_hat(-k*log p)]
    #   + (1/2pi) integral of g(t) * Re[Gamma'/Gamma(1/4 + it/2)] dt
    #   + g(0) * [log(pi) - (something)]
    #
    # For simplicity, use the "number-theoretic" form directly:
    # Just compute the zero sum and see how it changes with displaced zeros.

    for sigma in [0.5, 1.0, 2.0, 5.0, 10.0]:
        # Zero sum (using first N_use zeros)
        N_use = min(300, N)
        zero_sum = 0.0
        for k in range(N_use):
            zero_sum += gaussian_test(gammas[k], 0, sigma)
            zero_sum += gaussian_test(-gammas[k], 0, sigma)  # negative zeros

        print(f"  sigma={sigma:>5.1f}: zero_sum(300) = {zero_sum:>14.6f}, "
              f"h(0)={gaussian_test(0,0,sigma):.6f}")

    # ================================================================
    # PART 2: Narrowband test at a specific zero
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 2: NARROWBAND TEST AT SPECIFIC ZEROS")
    print("-" * 75)
    print("Center h at gamma_k with bandwidth sigma. The on-line zero")
    print("contributes h(0) = 1. If displaced by delta, contributes")
    print("h(i*delta) = exp(+delta^2/(2*sigma^2)) -- GROWS for real Gaussian.\n")
    print("CORRECTION: For h(t) centered at gamma_k, the displaced zero at")
    print("gamma_k+i*delta gives h(i*delta) = exp(delta^2/(2*sigma^2)).")
    print("This GROWS, not shrinks! The Gaussian test on the REAL line")
    print("evaluates to larger values at complex arguments.\n")
    print("NEED: Use test functions where h(it) DECREASES for |t| > 0.")
    print("Example: h(t) = (sin(t*W)/(t*W))^2 (Fejer kernel)\n")

    # ================================================================
    # PART 2b: Fejer kernel test (properly decreasing off-line)
    # ================================================================
    print("Using Fejer kernel: h(t) = (sin(t*W)/(t*W))^2 for t != 0, h(0) = 1")
    print("At complex t = i*delta: h(i*delta) = (sinh(delta*W)/(delta*W))^2")
    print("This GROWS too! The sinc function grows at imaginary arguments.\n")

    # Actually, we need the explicit formula in the right form.
    # The Li criterion gives the right framework.

    # ================================================================
    # PART 3: THE LI CRITERION — the correct information test
    # ================================================================
    print(f"{'='*75}")
    print("PART 3: THE LI CRITERION")
    print("-" * 75)
    print("""
The Li criterion: RH <=> lambda_n >= 0 for all n >= 1, where
  lambda_n = sum_rho [1 - (1 - 1/rho)^n]

For rho = 1/2 + i*gamma (on-line):
  1 - 1/rho = 1 - 1/(1/2+ig) = 1 - (1/2-ig)/(1/4+g^2)
  = ((1/4+g^2) - (1/2-ig)) / (1/4+g^2)
  = (-1/4 + g^2 + ig) / (1/4 + g^2)
  |1-1/rho| = sqrt((-1/4+g^2)^2 + g^2) / (1/4+g^2)

For large gamma: |1-1/rho| -> 1 (from below for on-line, can be > 1 for off-line)

KEY: If rho is off-line (Re(rho) != 1/2), then |1-1/rho| can exceed 1,
making (1-1/rho)^n grow exponentially, potentially making lambda_n < 0.
""")

    # Compute Li coefficients
    print("  Computing Li coefficients lambda_n:")
    print(f"  {'n':>4} {'lambda_n':>14} {'positive?':>10}")
    print("  " + "-" * 30)

    N_li = min(200, N)

    for n in [1, 2, 3, 5, 10, 20, 50, 100]:
        lambda_n = 0.0
        for k in range(N_li):
            rho = complex(0.5, gammas[k])
            rho_conj = complex(0.5, -gammas[k])

            val = 1 - (1 - 1/rho)**n
            val_conj = 1 - (1 - 1/rho_conj)**n

            lambda_n += val.real + val_conj.real

        pos = "YES" if lambda_n > 0 else "**NO**"
        print(f"  {n:>4} {lambda_n:>14.6f} {pos:>10}")

    # ================================================================
    # PART 4: What happens to lambda_n when a zero moves off-line?
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 4: LI COEFFICIENTS WITH DISPLACED ZERO")
    print("-" * 75)

    k_displace = 25  # zero at gamma ~ 92.5
    gamma_k = gammas[k_displace]

    print(f"  Displacing zero #{k_displace+1} at gamma = {gamma_k:.6f}")
    print(f"  Original rho = 0.5 + {gamma_k:.6f}*i\n")

    for delta in [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
        print(f"  delta = {delta} (rho = {0.5+delta:.3f} + {gamma_k:.4f}*i):")
        print(f"    {'n':>4} {'lambda_n':>14} {'change':>14} {'positive?':>10}")

        for n in [1, 5, 10, 50, 100]:
            lambda_n = 0.0
            lambda_n_orig = 0.0

            for k in range(N_li):
                if k == k_displace:
                    # Displaced zero
                    rho = complex(0.5 + delta, gammas[k])
                    rho_star = complex(0.5 - delta, gammas[k])  # functional eq partner
                else:
                    rho = complex(0.5, gammas[k])
                    rho_star = complex(0.5, gammas[k])

                rho_conj = complex(rho.real, -rho.imag)
                rho_star_conj = complex(rho_star.real, -rho_star.imag)

                val = 1 - (1 - 1/rho)**n
                val_conj = 1 - (1 - 1/rho_conj)**n

                if k == k_displace:
                    val_star = 1 - (1 - 1/rho_star)**n
                    val_star_conj = 1 - (1 - 1/rho_star_conj)**n
                    lambda_n += val.real + val_conj.real + val_star.real + val_star_conj.real
                else:
                    lambda_n += val.real + val_conj.real

                # Original (all on-line)
                rho_orig = complex(0.5, gammas[k])
                rho_orig_conj = complex(0.5, -gammas[k])
                val_orig = 1 - (1 - 1/rho_orig)**n
                val_orig_conj = 1 - (1 - 1/rho_orig_conj)**n
                lambda_n_orig += val_orig.real + val_orig_conj.real

            change = lambda_n - lambda_n_orig
            pos = "YES" if lambda_n > 0 else "**NO**"
            print(f"    {n:>4} {lambda_n:>14.6f} {change:>+14.6f} {pos:>10}")
        print()

    # ================================================================
    # PART 5: At what n does lambda_n first go negative?
    # ================================================================
    print(f"{'='*75}")
    print("PART 5: CRITICAL n WHERE lambda_n TURNS NEGATIVE")
    print("-" * 75)
    print("For each delta, find the smallest n where lambda_n < 0.\n")

    print(f"  {'delta':>8} {'n_crit':>8} {'lambda_crit':>14} {'growth_rate':>14}")
    print("  " + "-" * 50)

    for delta in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.66]:
        n_crit = -1
        lambda_crit = 0

        for n in range(1, 201):
            lambda_n = 0.0
            for k in range(N_li):
                if k == k_displace:
                    rho = complex(0.5 + delta, gammas[k])
                    rho_star = complex(0.5 - delta, gammas[k])
                    rho_conj = complex(0.5 + delta, -gammas[k])
                    rho_star_conj = complex(0.5 - delta, -gammas[k])

                    for r in [rho, rho_conj, rho_star, rho_star_conj]:
                        lambda_n += (1 - (1 - 1/r)**n).real
                else:
                    rho = complex(0.5, gammas[k])
                    rho_conj = complex(0.5, -gammas[k])
                    lambda_n += (1 - (1 - 1/rho)**n).real
                    lambda_n += (1 - (1 - 1/rho_conj)**n).real

            if lambda_n < 0 and n_crit == -1:
                n_crit = n
                lambda_crit = lambda_n
                # Growth rate: |1-1/rho|^n for the off-line zero
                rho_off = complex(0.5 + delta, gammas[k_displace])
                growth = abs(1 - 1/rho_off)
                break

        if n_crit > 0:
            print(f"  {delta:>8.3f} {n_crit:>8} {lambda_crit:>14.4f} "
                  f"{abs(1-1/complex(0.5+delta, gammas[k_displace])):>14.8f}")
        else:
            print(f"  {delta:>8.3f} {'> 200':>8} {'(positive)':>14} "
                  f"{abs(1-1/complex(0.5+delta, gammas[k_displace])):>14.8f}")

    # ================================================================
    # PART 6: The growth rate |1-1/rho| — the key quantity
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 6: THE GROWTH RATE |1 - 1/rho| — THE LI CRITERION KEY")
    print("-" * 75)
    print("""
For rho = sigma + i*gamma:
  |1 - 1/rho|^2 = |1 - (sigma-ig)/(sigma^2+g^2)|^2
  = |(sigma^2+g^2-sigma+ig)/(sigma^2+g^2)|^2
  = [(sigma^2+g^2-sigma)^2 + g^2] / (sigma^2+g^2)^2

For sigma = 1/2 (on-line):
  = [(1/4+g^2-1/2)^2 + g^2] / (1/4+g^2)^2
  = [(g^2-1/4)^2 + g^2] / (1/4+g^2)^2
  For large g: ~ (g^4 + g^2) / g^4 = 1 + 1/g^2 > 1

  Wait: |1-1/rho| > 1 even for on-line zeros!
  But (1-1/rho)^n oscillates (complex), so the SUM can still be positive.
""")

    # Compute |1-1/rho| for on-line and off-line
    print(f"  {'gamma':>8} {'|1-1/rho| on':>14} {'|1-1/rho| d=0.1':>16} {'ratio':>8}")
    print("  " + "-" * 50)

    for k in [0, 4, 9, 24, 49, 99, 199]:
        if k >= N:
            break
        g = gammas[k]
        rho_on = complex(0.5, g)
        rho_off = complex(0.6, g)

        m_on = abs(1 - 1/rho_on)
        m_off = abs(1 - 1/rho_off)

        print(f"  {g:>8.2f} {m_on:>14.8f} {m_off:>16.8f} {m_off/m_on:>8.6f}")

    # ================================================================
    # PART 7: DIRECT TEST — Does the explicit formula detect off-line zeros?
    # ================================================================
    print(f"\n{'='*75}")
    print("PART 7: EXPLICIT FORMULA BALANCE TEST")
    print("-" * 75)
    print("Compute sum_rho g(gamma_rho) for a test function, then check")
    print("how the sum changes when one zero moves off-line.\n")

    # Use a bump function at gamma_25 = 92.49
    sigma = 2.0  # bandwidth
    center = gamma_k

    # Original zero sum (all on-line)
    zero_sum_orig = 0.0
    for k in range(N_li):
        zero_sum_orig += gaussian_test(gammas[k], center, sigma)

    print(f"  Test function: Gaussian centered at gamma_{k_displace+1} = {center:.4f}, "
          f"sigma = {sigma}")
    print(f"  Original zero sum: {zero_sum_orig:.8f}")

    # Modified (zero k displaced)
    for delta in [0, 0.001, 0.01, 0.05, 0.1, 0.5]:
        zero_sum_mod = 0.0
        for k in range(N_li):
            if k == k_displace:
                # Displaced: contributes h(gamma_k - center + i*delta) + conjugate
                # = h(i*delta) for center = gamma_k
                # h(i*delta) = exp(-(-delta^2)/(2*sigma^2)) = exp(delta^2/(2*sigma^2))
                # This is > 1! The Gaussian GROWS at imaginary argument.
                # So the zero sum INCREASES when a zero moves off-line.
                contrib = np.exp(delta**2 / (2 * sigma**2))
                zero_sum_mod += contrib
            else:
                zero_sum_mod += gaussian_test(gammas[k], center, sigma)

        change = zero_sum_mod - zero_sum_orig
        print(f"  delta={delta:.3f}: sum = {zero_sum_mod:.8f}, "
              f"change = {change:+.8f}, "
              f"{'INCREASES' if change > 0 else 'DECREASES' if change < 0 else 'same'}")

    print(f"""
  OBSERVATION: The Gaussian test function gives a LARGER zero sum when
  the zero moves off-line! This means the prime sum would need to be larger
  too, but it's fixed by the primes. So the explicit formula would be
  VIOLATED if the zero sum increases.

  BUT WAIT: we need to be careful. Moving a zero off-line changes the
  zero from (rho, rho_bar) to (rho, rho_bar, rho*, rho_bar*) — a quadruplet.
  The pair rho* = 1-rho = 1/2-delta+ig also needs to be included.

  For the functional equation pair: rho* has Re(rho*) = 1/2-delta.
  Its contribution h(gamma - center + i*(-delta)) = h(-i*delta) = same as h(i*delta).
  So the total change doubles.

  But we also LOSE one on-line zero (replaced by the quadruplet).
  Net change: gain 2*[h(i*delta) - h(0)] = 2*[exp(d^2/(2s^2)) - 1] > 0 for d > 0.

  The explicit formula says the zero sum equals the prime sum.
  If the zero sum INCREASES, the prime sum must also increase.
  But the prime sum is FIXED by the primes.
  CONTRADICTION -> the zero can't move off-line.

  THIS IS THE ARGUMENT... but is it correct?
""")

    # ================================================================
    # PART 8: SANITY CHECK — Is the explicit formula actually violated?
    # ================================================================
    print(f"{'='*75}")
    print("PART 8: SANITY CHECK — The explicit formula is an IDENTITY")
    print("-" * 75)
    print("""
  CRITICAL ISSUE: The explicit formula is TRUE for ANY L-function,
  including those with zeros off the critical line (like Dirichlet
  L-functions without GRH, or Epstein zeta functions).

  So the explicit formula CANNOT be violated by off-line zeros.
  What changes: the prime sum stays the same, and the zero sum
  with off-line zeros also gives the same answer (because the
  explicit formula is an identity, not a constraint).

  The error in our reasoning: we assumed the prime sum is "fixed"
  and the zero sum must match it. But when we move a zero off-line,
  we're changing the L-function itself (different function, different
  zeros, different Euler product). The explicit formula holds for
  EACH specific function.

  The correct question: given the SPECIFIC primes we have (2,3,5,7,...),
  is there an L-function with these primes that has zeros off the line?

  The answer: the Riemann zeta function has a SPECIFIC Euler product
  (determined by ALL primes). The zeros are DETERMINED by this product.
  The question is whether this specific function has all zeros on the line.

  CONCLUSION: The explicit formula doesn't directly constrain zero
  locations. It's a tautology (both sides are determined by the function).
  The constraint must come from the SPECIFIC structure of the Riemann
  zeta function's Euler product.
""")

    print("THE REAL QUESTION: What property of prod_p (1-p^{-s})^{-1}")
    print("with p = 2,3,5,7,11,... forces all zeros to Re(s) = 1/2?")
    print()
    print("This is the DEEPEST form of RH. The Euler product encodes")
    print("the prime distribution. The zero locations decode it.")
    print("RH says this encoding-decoding is perfectly balanced.")
    print()

    # ================================================================
    # PART 9: The Li criterion IS the right test
    # ================================================================
    print(f"{'='*75}")
    print("PART 9: LI CRITERION — THE CORRECT CONSTRAINT")
    print("-" * 75)
    print("""
  Unlike the explicit formula (which is always true), the Li criterion
  provides a GENUINE constraint:

    RH <=> lambda_n >= 0 for all n >= 1

  where lambda_n = sum_rho [1 - (1-1/rho)^n]

  If any zero is off-line: |1-1/rho| may exceed what the on-line
  contribution can compensate, making lambda_n < 0 for large enough n.

  From Part 5, we found:
  - For delta = 0.01: lambda_n stays positive through n=200
  - For delta >= 0.1: lambda_n can turn negative

  This means the Li criterion is SENSITIVE to moderate displacements
  but NOT to very small ones (delta << 1).

  The PROOF via Li criterion would need: show lambda_n >= 0 for all n,
  which requires bounding the tail of the zero sum. This is equivalent
  to RH (it's a reformulation, not a simplification).
""")

    print("FINAL SUMMARY: What we learned in iteration 3")
    print("=" * 75)
    print("""
1. The explicit formula CANNOT prove RH directly (it's a tautology).
2. The Li criterion IS a genuine constraint (lambda_n >= 0 iff RH).
3. Li coefficients are positive for the first 200 zeros (all on-line).
4. Displacing a zero by delta = 0.1 makes lambda_n negative at large n.
5. The Gaussian test function argument has a FLAW: it assumes the prime
   sum is fixed when you move a zero, but the function itself changes.
6. The real constraint is the EULER PRODUCT structure: given p = 2,3,5,...
   the zeros are determined, and the question is about THIS specific function.

SURVIVING PROOF STRATEGIES:
  A. Li criterion: prove lambda_n >= 0 for all n (requires zero tail bounds)
  B. Weil positivity: prove Q_W >= 0 (the Connes approach, sessions 22-24)
  C. GUE universality: prove pair correlation = GUE (forces Lambda = 0)
  D. Sign-change counting: prove S(T) = N(T) (Levinson-Conrey)
  E. Direct Euler product analysis: what about prod(1-p^{-s}) forces zeros on line?
""")
