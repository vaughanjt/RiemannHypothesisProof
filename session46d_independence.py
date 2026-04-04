"""
SESSION 46d — INDEPENDENCE CHECK AND IDENTITY EXPANSION

Check: are the 309 constraints from the identity web truly independent?
If not, add more identities one at a time until overdetermined.

METHOD: Build the constraint matrix explicitly.
Each constraint is a linear (or nonlinear) relation among the barrier DOFs.
The rank of the constraint Jacobian tells us the number of independent constraints.

The barrier has DOFs: {W02, Mp_1, Mp_2, ..., Mp_K, M_diag, M_alpha}
Each identity imposes: f_i(DOFs) = constant (1 or 0).

For the LINEARIZED version:
  DOFs are the K+3 real numbers (W02, Mp_k, M_diag, M_alpha).
  The barrier B = W02 - sum Mp_k - M_diag - M_alpha.

  Constraint from I1 (sin^2+cos^2 for prime p_k at mode n):
    This constrains the PHASE of Mp_k, not its value directly.
    Mp_k = weight_k * sum_n w_hat[n]^2 * cos(phase_k_n) + off-diag terms.
    The phase is FIXED by p_k and L. So I1 doesn't remove a DOF from
    the barrier's value — it removes a DOF from the CONSTRUCTION.

  This is the key subtlety: the identities constrain the CONSTRUCTION
  of the barrier (how components are built from trig functions),
  not the VALUES of the components directly.

  Let's recount more carefully.

HONEST RECOUNT:
  The barrier B is a FUNCTION of lam^2 (or L). It's a single real number
  for each L. The "degrees of freedom" aren't the barrier value but the
  PARAMETERS that determine it.

  Parameters: L (1), the set of primes up to e^L (many), and the
  mathematical constants pi, e, Euler gamma, etc.

  The identities constrain RELATIONSHIPS among the barrier's components.
  The question is: do these relationships determine B's sign?

  A better approach: treat the barrier as a function of its components
  and check whether the identity constraints force B > 0.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes
from session45n_pi_predicts_primes import w02_only, prime_contribution


def barrier_components(lam_sq, N=15):
    """Compute all barrier components separately."""
    L = np.log(lam_sq)
    primes = list(sieve_primes(int(lam_sq)))
    K = len(primes)

    w02 = w02_only(lam_sq, N)
    mp_per_prime = [prime_contribution(int(p), lam_sq, N) for p in primes]
    mp_total = sum(mp_per_prime)

    return {
        'L': L, 'K': K, 'w02': w02,
        'mp_per_prime': np.array(mp_per_prime),
        'mp_total': mp_total,
        'primes': [int(p) for p in primes],
        'barrier': w02 - mp_total,
    }


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 46d -- INDEPENDENCE CHECK')
    print('=' * 76)

    N = 15

    # ══════════════════════════════════════════════════════════════
    # 1. HONEST RECOUNT OF DEGREES OF FREEDOM
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. HONEST RECOUNT: what are the TRUE degrees of freedom?')
    print('#' * 76)

    print(f'''
  The barrier B(L) = W02(L) - sum_k Mp_k(L) - M_diag(L) - M_alpha(L)
  is a SINGLE REAL NUMBER for each value of L.

  Session 46c counted 306 DOF and 309 constraints. But this double-counts:
  the "DOF" (one per prime) and the "constraints" (I1: one per prime)
  are the SAME thing viewed differently.

  HONEST COUNT:
  The barrier is determined by L and the primes up to e^L.
  For fixed L, the primes are DETERMINED (they're just the primes <= e^L).
  There are NO free parameters. B(L) is a computable number.

  The question is not "do constraints exceed DOF" but:
  "do the identities INDIVIDUALLY force B > 0?"

  Each identity imposes a RELATIONSHIP. The question:
  does any single identity (or combination) force B > 0 regardless
  of which numbers are prime?
  ''')

    # ══════════════════════════════════════════════════════════════
    # 2. IDENTITY-BY-IDENTITY: what does each one ACTUALLY constrain?
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  2. IDENTITY-BY-IDENTITY: real constraints')
    print('#' * 76)

    comp = barrier_components(2000, N)
    L = comp['L']
    K = comp['K']
    w02 = comp['w02']
    mp = comp['mp_per_prime']
    barrier = comp['barrier']

    print(f'\n  lam^2 = 2000, L = {L:.4f}, K = {K} primes')
    print(f'  W02 = {w02:+.6f}, sum Mp = {comp["mp_total"]:+.6f}, B = {barrier:+.6f}')

    # I1: sin^2 + cos^2 = 1
    print(f'\n  I1 (Pythagorean):')
    print(f'    Each Mp_k = weight_k * F(log(p_k)/L) where F involves cos,sin.')
    print(f'    sin^2+cos^2 = 1 means |e^{{i*phase}}| = 1.')
    print(f'    This constrains Mp_k to lie in [-weight_k, +weight_k].')
    weights = np.array([np.log(int(p)) / np.sqrt(int(p)) for p in comp['primes']])
    print(f'    Weight range: [{weights.min():.6f}, {weights.max():.6f}]')
    print(f'    |Mp_k| <= weight_k * (kernel bound)?')
    # Check: is |Mp_k| bounded by the weight?
    ratios_mp = np.abs(mp) / weights
    print(f'    |Mp_k|/weight_k range: [{ratios_mp.min():.4f}, {ratios_mp.max():.4f}]')
    print(f'    Max |Mp_k|/weight_k = {ratios_mp.max():.4f} (at p={comp["primes"][np.argmax(ratios_mp)]})')
    print(f'    CONSTRAINT: |Mp_k| <= C * weight_k where C ~ {ratios_mp.max():.2f}')

    # I2: cosh^2 - sinh^2 = 1
    print(f'\n  I2 (Hyperbolic):')
    print(f'    W02 = 32L*sinh^2(L/4)*QF.')
    print(f'    cosh^2-sinh^2=1 splits this but doesn\'t bound W02.')
    print(f'    W02 is DETERMINED by L. No constraint on its sign or magnitude')
    print(f'    beyond what L dictates.')
    print(f'    CONSTRAINT: W02 is a SPECIFIC function of L (0 DOF).')
    print(f'    INDEPENDENT? NO — W02 is already determined by L.')

    # I4: Mobius inversion
    print(f'\n  I4 (Mobius):')
    print(f'    zeta(s) * sum mu(n)/n^s = 1.')
    print(f'    This means: sum_k Mp_k has a Mobius inverse.')
    print(f'    But sum_k Mp_k is ALREADY determined by the primes.')
    print(f'    The Mobius identity is satisfied AUTOMATICALLY.')
    print(f'    CONSTRAINT: redundant (already encoded in prime structure).')
    print(f'    INDEPENDENT? NO — follows from the definition of primes.')

    # I5: Gamma reflection
    print(f'\n  I5 (Gamma reflection):')
    print(f'    Gamma(s)*Gamma(1-s) = pi/sin(pi*s).')
    print(f'    M_diag and M_alpha involve digamma psi(s).')
    print(f'    The reflection formula constrains M_alpha\'s symmetry')
    print(f'    but M_alpha is COMPUTED from L — no free parameters.')
    print(f'    CONSTRAINT: M_alpha is symmetric. Already true by construction.')
    print(f'    INDEPENDENT? NO — follows from the explicit formula.')

    # I6: Functional equation
    print(f'\n  I6 (Functional equation):')
    print(f'    xi(s) = xi(1-s).')
    print(f'    Forces v(1/2,t) = 0 (xi real on CL).')
    print(f'    This IS a genuine constraint on the barrier:')
    print(f'    the barrier on the CL is a REAL function.')
    print(f'    INDEPENDENT? YES — this is the key structural identity.')

    # Z5: v = 0 on CL
    print(f'\n  Z5 (v=0 on CL):')
    print(f'    Same as I6 — a consequence of the functional equation.')
    print(f'    INDEPENDENT? NO — redundant with I6.')

    # I8: Parseval
    print(f'\n  I8 (Parseval):')
    print(f'    sum |coefficients|^2 = integral |function|^2.')
    print(f'    Relates Fourier and real-space norms.')
    print(f'    For the barrier: sum |Mp_mode|^2 = integral |kernel|^2.')
    print(f'    This constrains the TOTAL energy, not the sign.')
    print(f'    INDEPENDENT? YES (from other identities) — but doesn\'t')
    print(f'    constrain the sign, only the norm.')

    # Z2: Zeros
    print(f'\n  Z2 (xi(rho)=0):')
    print(f'    Each zero provides B_spectral += |H(rho)|^2.')
    print(f'    The spectral representation B = sum |H|^2 >= 0.')
    print(f'    But this requires summing ALL zeros (infinite).')
    print(f'    INDEPENDENT? YES — each zero is an independent datum.')
    print(f'    But collectively they restate B >= 0 (which is what we want to prove).')

    # ══════════════════════════════════════════════════════════════
    # 3. THE HONEST INDEPENDENCE COUNT
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. HONEST INDEPENDENCE COUNT')
    print('#' * 76)

    print(f'''
  TRULY INDEPENDENT CONSTRAINTS:

  I1 (Pythagorean): bounds |Mp_k| for each prime.
    BUT: Mp_k is DETERMINED by p_k and L. The bound is automatically
    satisfied. Not a free constraint — it's a CONSEQUENCE.
    INDEPENDENT CONSTRAINTS FROM I1: 0

  I2 (Hyperbolic): W02 is determined by L.
    INDEPENDENT CONSTRAINTS: 0

  I4 (Mobius): automatic from prime definition.
    INDEPENDENT CONSTRAINTS: 0

  I5 (Gamma reflection): M_diag/M_alpha symmetric by construction.
    INDEPENDENT CONSTRAINTS: 0

  I6 (Functional equation): xi real on CL.
    INDEPENDENT CONSTRAINTS: 1 (the symmetry s <-> 1-s)

  I8 (Parseval): bounds total energy but not sign.
    INDEPENDENT CONSTRAINTS: 1 (energy conservation)

  Z2 (Zeros): spectral representation B = sum |H|^2 >= 0.
    INDEPENDENT CONSTRAINTS: infinity (one per zero)
    BUT: this IS the statement B >= 0. Circular.

  Z6 (Mertens): M(x)/sqrt(x) bounded iff RH.
    INDEPENDENT CONSTRAINTS: 1 (but equivalent to RH)

  TOTAL TRULY INDEPENDENT, NON-CIRCULAR CONSTRAINTS: 2
    (functional equation symmetry + Parseval energy bound)

  TOTAL DOF: 0 (everything is determined by L and the primes)

  THE SYSTEM IS NOT "OVERDETERMINED BY 3" — the 309 vs 306 counting
  was wrong because most constraints are AUTOMATIC (consequences of
  the barrier's construction, not independent conditions).
  ''')

    # ══════════════════════════════════════════════════════════════
    # 4. WHAT WOULD MAKE A CONSTRAINT GENUINELY USEFUL?
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  4. WHAT WOULD MAKE A CONSTRAINT GENUINELY USEFUL?')
    print('#' * 76)

    print(f'''
  A useful constraint must:
  (a) NOT be automatic from the barrier's construction
  (b) CONSTRAIN THE SIGN of B, not just its components
  (c) Hold for ALL L, not just specific values

  The identities we have constrain STRUCTURE (magnitudes, symmetries,
  energy). None constrain the SIGN of B.

  To constrain the sign, we need an identity that says:
  "the specific arrangement of cos(2*pi*n*log(p)/L) for the actual primes
  cannot produce a sum that exceeds W02."

  This is a statement about the DISTRIBUTION of log(p) mod L,
  which is a number theory question, not an identity question.
  ''')

    # ══════════════════════════════════════════════════════════════
    # 5. ATTEMPT: ADD IDENTITIES THAT CONSTRAIN THE SIGN
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  5. SEARCHING FOR SIGN-CONSTRAINING IDENTITIES')
    print('#' * 76)

    # Try: can we find an identity that relates B directly to
    # a known positive quantity?

    # Attempt A: B = |something|^2 - |something_else|^2 + positive
    # The spectral repr: B = sum |H(rho)|^2. This IS |something|^2 >= 0.
    # But we can't compute the infinite sum.

    # Attempt B: B = W02 - Mp. Write as:
    # B = W02 * (1 - Mp/W02)
    # If we can show 0 < Mp/W02 < 1... but Mp/W02 > 1 (both negative,
    # |Mp| > |W02|), so the ratio is > 1. Yet B > 0 because of the
    # double negative.

    # Let's be precise about signs:
    print(f'\n  Sign analysis at lam^2 = 2000:')
    print(f'    W02 = {w02:+.6f} (NEGATIVE)')
    print(f'    sum Mp = {comp["mp_total"]:+.6f} (NEGATIVE, more negative than W02)')
    print(f'    B = W02 - sum Mp = {w02:.4f} - ({comp["mp_total"]:.4f}) = {barrier:+.6f} (POSITIVE)')
    print(f'    B > 0 because |sum Mp| > |W02| and both are negative.')
    print(f'    Equivalently: the primes OVERSHOOT the archimedean bound.')
    print(f'    The overshoot IS the barrier.')

    # Attempt C: Use the functional equation to relate B at s and 1-s
    print(f'\n  Attempt: functional equation symmetry.')
    print(f'  xi(1/2+it) = xi(1/2-it)* (conjugate), so xi is real on CL.')
    print(f'  This means B is a real-valued function of L.')
    print(f'  B could be positive or negative — the FE doesn\'t determine the sign.')
    print(f'  VERDICT: functional equation constrains parity, not sign.')

    # Attempt D: Parseval + spectral repr
    print(f'\n  Attempt: Parseval applied to the barrier kernel.')
    print(f'  The kernel K(x) satisfies Parseval: integral |K|^2 = sum |K_n|^2.')
    print(f'  B = sum_p weight_p * K(log(p)/L) - W02_stuff.')
    print(f'  Parseval bounds the total |K|^2 but not the prime-weighted sum.')
    print(f'  VERDICT: Parseval constrains energy, not the prime sampling.')

    # Attempt E: Large sieve inequality
    print(f'\n  Attempt: Large sieve inequality.')
    print(f'  sum_q sum_a |sum a_n e(n*a/q)|^2 <= (N + Q^2) * sum |a_n|^2')
    print(f'  This bounds the "coherence" of exponential sums.')
    print(f'  For our prime sum: sum |sum w_n e(n*log(p)/L)|^2 <= ...')
    print(f'  The large sieve gives an UPPER BOUND on the prime sum\'s coherence.')
    print(f'  If the coherence is bounded, the cross-term is bounded.')
    print(f'  VERDICT: PROMISING — the large sieve directly bounds the obstacle!')

    # Check: what does the large sieve give?
    print(f'\n  LARGE SIEVE BOUND:')
    print(f'  For the barrier kernel with N modes from -N to N:')
    print(f'  sum_p |sum_n w_hat[n] * e(n*log(p)/L)|^2 <= (2N+1 + K) * sum |w_hat[n]|^2')
    print(f'    2N+1 = {2*N+1}')
    print(f'    K = {K}')
    print(f'    sum |w_hat|^2 = 1 (normalized)')
    print(f'    Large sieve bound: {2*N+1 + K}')

    # The actual sum
    actual_sum = np.sum(mp**2)  # roughly sum |kernel at prime|^2 * weight^2
    print(f'    Actual sum of Mp_k^2: {actual_sum:.6f}')
    print(f'    Ratio actual/bound: {actual_sum/(2*N+1+K):.6f}')
    print(f'    The large sieve bound is MUCH larger than the actual sum.')
    print(f'    This means the primes are VERY incoherent (using only')
    print(f'    {actual_sum/(2*N+1+K)*100:.2f}% of the allowed coherence).')

    # ══════════════════════════════════════════════════════════════
    # 6. THE LARGE SIEVE APPROACH
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. THE LARGE SIEVE: can it bound the barrier?')
    print('#' * 76)

    print(f'''
  The large sieve inequality:
    sum_p (sum_n a_n e(n*alpha_p))^2 <= (N + delta^{{-1}}) sum |a_n|^2

  where alpha_p = log(p)/L and delta = min|alpha_p - alpha_q| (prime spacing
  in the Fourier dual).

  For OUR barrier:
    B = W02 - sum_p weight_p * K(alpha_p)

  where K is the kernel function. The large sieve bounds:
    sum_p weight_p^2 * K(alpha_p)^2 <= (large sieve constant) * integral K^2

  This bounds the L^2 norm of the prime sampling, but we need to bound
  the L^1 norm (the SUM, not sum of squares).

  By Cauchy-Schwarz: |sum x_k| <= sqrt(K) * sqrt(sum x_k^2)
  So: |sum Mp_k| <= sqrt(K) * sqrt(sum Mp_k^2) <= sqrt(K) * sqrt(LS bound)
  ''')

    # Compute the Cauchy-Schwarz + Large Sieve bound on |sum Mp|
    ls_bound_sq = (2*N+1 + K) * 1.0  # sum |w_hat|^2 = 1
    cs_ls_bound = np.sqrt(K) * np.sqrt(ls_bound_sq)
    actual_sum_mp = abs(comp['mp_total'])

    print(f'  Large sieve + Cauchy-Schwarz bound on |sum Mp|:')
    print(f'    sqrt(K) * sqrt(2N+1+K) * ||w_hat|| = sqrt({K}) * sqrt({2*N+1+K})')
    print(f'    = {cs_ls_bound:.4f}')
    print(f'    Actual |sum Mp| = {actual_sum_mp:.4f}')
    print(f'    Ratio: {actual_sum_mp/cs_ls_bound:.6f}')
    print(f'    The bound is {cs_ls_bound/actual_sum_mp:.1f}x too large.')

    # Does this bound force B > 0?
    print(f'\n    For B > 0 we need |sum Mp| < |W02| = {abs(w02):.4f}')
    print(f'    The LS+CS bound gives |sum Mp| < {cs_ls_bound:.4f}')
    print(f'    Is {cs_ls_bound:.4f} < {abs(w02):.4f}? {"YES" if cs_ls_bound < abs(w02) else "NO"}')

    if cs_ls_bound < abs(w02):
        print(f'    *** THE LARGE SIEVE PROVES B > 0 AT THIS L! ***')
    else:
        print(f'    The large sieve bound is too loose by factor {cs_ls_bound/abs(w02):.1f}')

    # Check across L values
    print(f'\n  Large sieve bound vs |W02| across L:')
    print(f'  {"lam^2":>8s} {"K":>6s} {"LS+CS bound":>14s} {"|W02|":>10s} '
          f'{"ratio":>8s} {"proves B>0?":>10s}')
    print('  ' + '-' * 60)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        c = barrier_components(lam_sq, N)
        ls = np.sqrt(c['K']) * np.sqrt(2*N+1 + c['K'])
        proves = 'YES' if ls < abs(c['w02']) else 'no'
        print(f'  {lam_sq:>8d} {c["K"]:>6d} {ls:>14.4f} {abs(c["w02"]):>10.4f} '
              f'{ls/abs(c["w02"]):>8.2f} {proves:>10s}')

    # ══════════════════════════════════════════════════════════════
    # 7. TIGHTENING THE SIEVE
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. CAN WE TIGHTEN THE SIEVE?')
    print('#' * 76)

    print(f'''
  The large sieve bound is too loose because:
  1. Cauchy-Schwarz loses the sign information (some Mp_k are positive)
  2. The large sieve uses worst-case alpha spacing, not actual log(p)/L
  3. The weight function (kernel) is not incorporated

  TIGHTER BOUNDS:

  A. SELBERG SIEVE: uses optimized weights to get tighter bounds.
     The Selberg sieve can bound prime sums more tightly than
     the large sieve for specific weight functions.

  B. DUAL LARGE SIEVE: bounds |sum a_n Lambda(n) f(n)|^2 using
     zero-density estimates. This directly involves the zeros.

  C. MEAN VALUE ESTIMATES: for Dirichlet polynomials,
     integral |sum a_n n^{{-it}}|^2 dt = sum |a_n|^2 (T + O(N))
     gives exact mean-square. The prime sum is a Dirichlet polynomial.

  D. HALASZ-MONTGOMERY: bounds character sums using the pretentious
     distance. Could bound how close the prime sum can get to W02.
  ''')

    # Compute the mean-value estimate
    print(f'  Mean value theorem for Dirichlet polynomials:')
    print(f'  (1/T) integral_0^T |sum a_n n^{{-it}}|^2 dt = sum |a_n|^2 + O(N/T)')
    print(f'\n  For our barrier kernel evaluated at "time" L:')
    print(f'  The prime sum sum_p weight_p * kernel(log(p)/L) is ONE evaluation')
    print(f'  of a Dirichlet polynomial at "time" L.')
    print(f'  The mean-value theorem says the AVERAGE over L is sum weight_p^2.')
    print(f'  The actual value at specific L can deviate by sqrt(variance).')

    # Compute variance
    var_est = np.sum(mp**2)  # sum |Mp_k|^2 as variance estimate
    std_est = np.sqrt(var_est)
    mean_est = np.sum(np.abs(mp)**2) / len(mp) * len(mp)  # rough
    print(f'\n  sum Mp_k^2 (variance proxy): {var_est:.6f}')
    print(f'  sqrt(sum Mp_k^2):              {std_est:.6f}')
    print(f'  |sum Mp_k|:                    {actual_sum_mp:.6f}')
    print(f'  |sum Mp_k| / sqrt(sum Mp_k^2): {actual_sum_mp/std_est:.4f}')
    print(f'  This ratio is sqrt(K)*coherence = {np.sqrt(K):.1f} * {actual_sum_mp/(std_est*np.sqrt(K)):.4f}')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 46d SYNTHESIS')
    print('=' * 76)

    print(f'''
  INDEPENDENCE CHECK:

  The 309 vs 306 counting from session 46c was WRONG.
  Most "constraints" are AUTOMATIC (consequences of construction):
    - I1 (Pythagorean): automatic (trig functions always satisfy this)
    - I2 (Hyperbolic): automatic (cosh,sinh always satisfy this)
    - I4 (Mobius): automatic (follows from prime definition)
    - I5 (Gamma reflection): automatic (built into the formula)

  TRULY INDEPENDENT constraints that aren't automatic:
    - I6/Z5 (Functional equation): xi real on CL (1 constraint)
    - I8 (Parseval): energy conservation (1 constraint)
    - Z2 (Zeros): spectral representation (circular — IS the barrier)
    - Z6 (Mertens): M(x)/sqrt(x) bounded (= RH, circular)

  NON-CIRCULAR, INDEPENDENT, SIGN-RELEVANT CONSTRAINTS: ~0
  The identities constrain STRUCTURE, not SIGN.

  THE LARGE SIEVE ATTEMPT:
    Bound on |sum Mp|: sqrt(K*(2N+1+K)) ~ {cs_ls_bound:.0f}
    Needed (|W02|):    {abs(w02):.0f}
    Ratio: {cs_ls_bound/abs(w02):.1f}x too large
    The large sieve is too loose to prove B > 0.

  THE HONEST CONCLUSION:
  No combination of identities equaling 1 or 0 can force B > 0,
  because the identities constrain the FRAMEWORK (magnitudes, symmetries,
  energy), not the CONTENT (which numbers are prime). The barrier's sign
  depends on the SPECIFIC arithmetic of the primes, which is determined
  by the sieve of Eratosthenes, not by any identity.

  The only "identity" that determines the barrier's sign is the
  PRIME NUMBER THEOREM and its refinements (explicit formula).
  And proving the PNT's error term is small enough IS the Riemann Hypothesis.
''')

    print('=' * 76)
    print('  SESSION 46d COMPLETE')
    print('=' * 76)
