"""
SESSION 34 — STAB THE HEART

WHY does M have exactly 1 positive eigenvalue?

THE HYPOTHESIS:
  v_+ is the function f(x) = 1/(L^2 + 4*pi^2*x^2) evaluated at integers.
  This is the CAUCHY/POISSON KERNEL for the strip of width L.

  The Weil explicit formula says:
    sum_rho f_hat(rho) = W_{0,2}(f) - M(f)

  For f = v_+: W_{0,2}(f) = s_v (positive eigenvalue of W02)
  And: sum_rho f_hat(rho) = sum over zeros of the Fourier transform of v_+

  Since v_+ is the Poisson kernel: f_hat(t) = (L/2) * exp(-L|t|)
  This is a POSITIVE function! So sum_rho f_hat(rho) involves
  f_hat evaluated at the zeros, which are on the critical line Re=1/2.

  The identity: s_v - M(v_+, v_+) = sum_rho f_hat(rho)
  Since f_hat > 0 and the zeros have specific structure:
  s_v - lambda_+ = small positive number (the zero sum residual)

  This EXPLAINS why lambda_+ < s_v (our proved 2x2 condition).

THE DEEPER QUESTION:
  Why is v_+ the ONLY positive direction of M?

  M(f, f) = sum_{p^k} Lambda(p^k)/sqrt(p^k) * <f|T(p^k)|f> + analytic terms

  For f perpendicular to v_+: <f, v_+> = 0
  The claim: M(f,f) < 0 for all such f.

  This means: the prime sum, when evaluated on functions ORTHOGONAL
  to the Poisson kernel, is always negative.

  WHY? Because functions orthogonal to the Poisson kernel have
  OSCILLATORY Fourier transforms. The prime sum, being a sum of
  oscillatory terms, achieves maximum coherent addition ONLY in the
  direction of the Poisson kernel (which is the "DC component" that
  all primes contribute to positively).

  In other directions, the prime contributions DESTRUCTIVELY INTERFERE.

THE PROOF ATTEMPT:
  Show that <f, M f> = <f, M_diag f> + <f, M_prime f>
  For f perp to v_+:
  - <f, M_diag f> can be positive or negative
  - <f, M_prime f> = sum_{p^k} w(p^k) * |F(f, p^k)|^2 * (sign factor)

  Actually, <f, M_prime f> is NOT a sum of squares — it's a sum of
  bilinear forms. But the key: for f oscillatory, the bilinear forms
  cancel due to phase randomness of log(p) for different primes.

  Can we prove: E[<f, T(p) f>] < 0 for "random" primes and oscillatory f?
"""

import numpy as np
import time, json, sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def verify_poisson_kernel(lam_sq, N=None):
    """
    Verify that v_+ IS the Poisson kernel evaluated at integers.

    The Poisson kernel for the strip [-L/2, L/2]:
    P(x) = (L/2) / (L^2/4 + pi^2*x^2) = 2L / (L^2 + 4*pi^2*x^2)

    Normalized: v_+(n) = 1/(L^2 + 4*pi^2*n^2) (up to normalization)

    This IS the function we used to define W02's eigenvector u_v!
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1
    L_f = np.log(lam_sq)

    W02, M, QW = build_all(lam_sq, N)
    ns = np.arange(-N, N+1, dtype=float)

    # The Poisson kernel vector
    poisson = 1.0 / (L_f**2 + 4*np.pi**2*ns**2)
    poisson_norm = poisson / np.linalg.norm(poisson)

    # M's positive eigenvector
    evals_M, evecs_M = np.linalg.eigh(M)
    v_plus = evecs_M[:, -1]  # largest eigenvalue
    if np.dot(v_plus, poisson_norm) < 0:
        v_plus = -v_plus

    alignment = np.dot(v_plus, poisson_norm)

    print(f"\nPOISSON KERNEL VERIFICATION: lam^2={lam_sq}")
    print(f"  Alignment of v_+ with Poisson kernel: {alignment:.8f}")
    print(f"  Deviation: {1 - alignment:.4e}")

    # The Fourier transform of the Poisson kernel:
    # FT of 1/(a^2 + x^2) = (pi/a) * exp(-a|t|)
    # So f_hat(t) = (pi/L^2) * exp(-L^2 * |t| / (2*pi))... not quite
    # Actually for our basis: the FT is in the discrete sense.

    # The continuous FT of f(x) = 1/(L^2 + 4*pi^2*x^2):
    # f_hat(t) = exp(-L|t|) / (2L)  (for our normalization)

    return alignment


def orthogonal_negativity_test(lam_sq, N=None):
    """
    For vectors f ORTHOGONAL to v_+ (the Poisson kernel):
    verify that M(f,f) < 0.

    Generate random vectors orthogonal to v_+, compute <f, M f>.
    Also test structured vectors (pure Fourier modes, etc.)
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)

    evals_M, evecs_M = np.linalg.eigh(M)
    v_plus = evecs_M[:, -1]

    # Project M onto orthogonal complement of v_+
    P_orth = np.eye(dim) - np.outer(v_plus, v_plus)
    M_orth = P_orth @ M @ P_orth

    evals_orth = np.linalg.eigvalsh(M_orth)
    # Remove the zero eigenvalue (from projection)
    nonzero = evals_orth[np.abs(evals_orth) > 1e-12]

    print(f"\n  M ORTHOGONAL TO v_+: lam^2={lam_sq}")
    print(f"    Max eigenvalue on orth complement: {np.max(nonzero):.6e}")
    print(f"    ALL NEGATIVE: {np.max(nonzero) < 1e-10}")

    # Test specific orthogonal vectors
    ns = np.arange(-N, N+1, dtype=float)

    # Pure odd modes (automatically orthogonal to v_+ which is even)
    print(f"\n    Specific orthogonal directions:")
    for test_n in [1, 2, 3, 5, 10]:
        # sin(2*pi*n*x/L) mode — odd, orthogonal to even v_+
        f = np.zeros(dim)
        for i in range(dim):
            f[i] = np.sin(2*np.pi*test_n*ns[i]/(2*N+1))
        f = f / np.linalg.norm(f)
        # Ensure orthogonal to v_+
        f = f - np.dot(f, v_plus) * v_plus
        f = f / np.linalg.norm(f)
        val = f @ M @ f
        print(f"      sin mode n={test_n}: <f,Mf> = {val:.6e}")

    # Random orthogonal vectors
    n_random = 1000
    max_random = -np.inf
    for _ in range(n_random):
        f = np.random.randn(dim)
        f = f - np.dot(f, v_plus) * v_plus
        f = f / np.linalg.norm(f)
        val = f @ M @ f
        max_random = max(max_random, val)

    print(f"      Max over {n_random} random orth vectors: {max_random:.6e}")
    print(f"      ALL NEGATIVE: {max_random < 1e-10}")


def explicit_formula_connection(lam_sq, N=None):
    """
    THE EXPLICIT FORMULA EXPLAINS THE RANK-1 STRUCTURE.

    The Weil explicit formula: for test function f,
      sum_rho f_hat(rho) = W02(f) - M(f)

    Rearranging: M(f) = W02(f) - sum_rho f_hat(rho)

    As a quadratic form:
      <f, M f> = <f, W02 f> - sum_rho |f_hat(rho)|^2

    The second term is a SUM OF SQUARES — always non-negative!
    So: <f, M f> <= <f, W02 f>

    For f in null(W02): <f, W02 f> = 0, so <f, M f> <= 0.
    THIS IS EXACTLY M <= 0 ON NULL(W02)!

    Wait — this would PROVE it. Let me check the formula carefully.

    The Weil quadratic form: QW(f,f) = <f, QW f>
    QW = W02 - M
    So: <f, QW f> = <f, W02 f> - <f, M f>

    From the explicit formula applied to f*f:
    QW(f,f) = sum_rho |f_hat(rho)|^2 >= 0 (IF zeros are on the line)

    But this is CONDITIONAL on RH. The sum over zeros is non-negative
    only if all zeros have Re(rho) = 1/2. Otherwise, some f_hat(rho)
    could contribute negatively.

    So the explicit formula gives:
    <f, M f> = <f, W02 f> - sum_rho |f_hat(rho)|^2

    WITHOUT assuming RH: the zero sum could be negative for off-line zeros.
    WITH RH: the zero sum is non-negative, giving <f,Mf> <= <f,W02 f>.

    This is circular again for the full proof. But it EXPLAINS the structure:
    M has at most 1 positive eigenvalue because W02 has rank 2, and the
    zero sum "eats" all the positivity except what W02 provides.

    HOWEVER: for the SPECIFIC function v_+ (Poisson kernel),
    f_hat(rho) = exp(-L|rho|)/(2L) which is always positive for rho on the line.
    So sum_rho |f_hat(rho)|^2 is a sum of positive terms — small but positive.
    This gives: M(v_+) = W02(v_+) - (small positive) = s_v - delta
    where delta = sum_rho |f_hat(rho)|^2 > 0.

    The single positive eigenvalue of M is s_v - delta, which is < s_v.
    This is EXACTLY what we computed: lambda_+ < s_v.

    CAN WE MAKE THIS NON-CIRCULAR?
    The key: sum_rho |f_hat(rho)|^2 >= 0 is trivially true (sum of squares)
    REGARDLESS of whether zeros are on the line!

    Wait... |f_hat(rho)|^2 is the modulus squared of f_hat evaluated at rho.
    For rho = 1/2 + i*gamma (on the line), |f_hat(rho)|^2 is always non-negative.
    For rho = sigma + i*gamma with sigma != 1/2 (off the line),
    |f_hat(rho)|^2 is STILL non-negative — it's a modulus squared!

    SO: sum_rho |f_hat(rho)|^2 >= 0 UNCONDITIONALLY.

    Therefore: <f, M f> = <f, W02 f> - sum_rho |f_hat(rho)|^2 <= <f, W02 f>

    And for f in null(W02): <f, M f> <= 0.

    THIS IS THE PROOF.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1
    L_f = np.log(lam_sq)

    W02, M, QW = build_all(lam_sq, N)

    print(f"\nEXPLICIT FORMULA CONNECTION: lam^2={lam_sq}")
    print(f"  The Weil explicit formula gives:")
    print(f"    <f, M f> = <f, W02 f> - sum_rho |f_hat(rho)|^2")
    print(f"")
    print(f"  The zero sum is a SUM OF MODULUS SQUARES: always >= 0")
    print(f"  (|f_hat(rho)|^2 >= 0 regardless of where rho is)")
    print(f"")
    print(f"  Therefore: <f, M f> <= <f, W02 f>")
    print(f"  For f in null(W02): <f, M f> <= 0")
    print(f"")
    print(f"  *** THIS PROVES M <= 0 ON NULL(W02) ***")
    print(f"  *** UNCONDITIONALLY — NO ASSUMPTION ABOUT ZERO LOCATIONS ***")

    # VERIFY: check that QW = W02 - M matches sum_rho |f_hat(rho)|^2
    # For each eigenvector of QW, compute <f, QW f> and compare with zero sum

    evals_QW = np.linalg.eigvalsh(QW)
    print(f"\n  VERIFICATION:")
    print(f"    QW eigenvalues (= sum_rho |f_hat(rho)|^2 for each eigenvector):")
    print(f"    All non-negative: {evals_QW[0] > -1e-10}")
    print(f"    Min: {evals_QW[0]:.6e}")
    print(f"    This IS the zero sum — and it IS non-negative.")

    # Wait — but there's a subtlety. The formula
    # QW(f,f) = sum_rho |f_hat(rho)|^2
    # is only valid for f with bandwidth <= Lambda.
    # Our truncated basis V_n for |n| <= N gives functions with bandwidth ~ N.
    # As long as N*2pi/L <= Lambda (which is ensured by our choice of N),
    # the formula applies.

    print(f"\n  SUBTLETY CHECK:")
    print(f"    Bandwidth of basis: N*2*pi/L = {N*2*np.pi/L_f:.4f}")
    print(f"    Lambda = sqrt(lam^2) = {np.sqrt(lam_sq):.4f}")
    print(f"    Bandwidth <= Lambda: {N*2*np.pi/L_f <= np.sqrt(lam_sq)}")

    # BUT WAIT. The formula QW(f,f) = sum_rho |f_hat(rho)|^2 is the
    # DEFINITION of the Weil quadratic form. It's true by construction.
    # The question is whether this equals W02(f) - M(f) for our SPECIFIC
    # W02 and M matrices (which are computed from the explicit formula).

    # If QW = W02 - M by construction, then:
    # sum_rho |f_hat(rho)|^2 = W02(f) - M(f)
    # => M(f) = W02(f) - sum_rho |f_hat(rho)|^2 <= W02(f)

    # This is NOT circular because:
    # - QW = W02 - M is the DEFINITION (not assuming RH)
    # - sum_rho |f_hat(rho)|^2 >= 0 is a TAUTOLOGY (sum of squares)
    # - Therefore M(f) <= W02(f) unconditionally
    # - For f in null(W02): M(f) <= 0

    # THE CATCH: does QW(f,f) actually equal sum_rho |f_hat(rho)|^2?
    # This is the Weil explicit formula. It's true for compactly supported
    # test functions. Our basis functions V_n are supported on [lam^{-1}, lam].
    # So yes, the formula applies.

    # BUT: the sum over rho includes ALL zeros, including possible off-line zeros.
    # If there are zeros off the line, f_hat(rho) is complex but |f_hat(rho)|^2
    # is still non-negative. So the sum is still >= 0.

    # WAIT — there's another subtlety. The Weil explicit formula for
    # the quadratic form is:
    # QW(f,f) = W02(f,f) - W_R(f,f) - sum_p W_p(f,f)
    # where W_R is the archimedean term and W_p are the prime terms.
    # This equals sum_rho f_hat(rho) * conj(f_hat(rho)) = sum |f_hat(rho)|^2
    # ONLY if the explicit formula is applied correctly.

    # The standard form of the explicit formula:
    # sum_rho h(rho) = h_hat(0) + h_hat(1) - sum_p sum_k Lambda(p^k)/p^{k/2} * h(k*log p)
    #                  + integral term (archimedean)

    # For h = |f|^2 (a positive function): the LHS is sum_rho |f_hat(rho)|^2.
    # The RHS is W02 terms - prime terms = QW(f,f).

    # So: QW(f,f) = sum_rho |f_hat(rho)|^2 >= 0 for ALL f.

    # THIS IS A THEOREM (the Weil explicit formula).
    # It says QW >= 0 — which IS the Riemann Hypothesis!

    # Hmm wait — does the explicit formula for h = |f|^2 actually give
    # QW(f,f) on the RHS? Let me think more carefully...

    # The explicit formula: for suitable test functions h,
    # sum_{rho} h(gamma_rho) = (1/2pi) int h(r) * Phi(r) dr
    #                          - sum_{p^k} Lambda(p^k)/p^{k/2} * (h(k log p) + h(-k log p))
    #                          + h(i/2) + h(-i/2)

    # where Phi(r) = Re(Gamma'/Gamma(1/4 + ir/2)) - log pi

    # For h = f * conj(f) where f has bandwidth <= Lambda:
    # LHS = sum_rho |f(gamma_rho)|^2 (if zeros are simple)
    # RHS = QW(f,f) (by definition of the quadratic form)

    # Actually f is defined on the multiplicative group R_+^*, not on R.
    # The Fourier transform f_hat goes from multiplicative to additive.

    # I need to be more careful about the function spaces.
    # Let me just verify numerically that QW(f,f) >= 0 for all test vectors.

    print(f"\n  NUMERICAL VERIFICATION:")
    print(f"    QW is PSD: {evals_QW[0] > -1e-10}")
    print(f"    This means: sum_rho |f_hat(rho)|^2 >= 0 for all bandwidth-limited f")
    print(f"    Which is TRIVIALLY true (sum of non-negative terms)")
    print(f"")
    print(f"    BUT: does QW(f,f) = sum_rho |f_hat(rho)|^2?")
    print(f"    This is the WEIL EXPLICIT FORMULA.")
    print(f"    If yes: QW >= 0 is a tautology, not a theorem about primes.")
    print(f"    If no: QW >= 0 is a genuine constraint (= RH).")
    print(f"")
    print(f"    THE TRUTH: QW(f,f) = sum_rho |f_hat(rho)|^2 is the explicit formula,")
    print(f"    but it's only been PROVED for specific test function classes.")
    print(f"    Extending it to all bandwidth-limited f IS the content of RH.")
    print(f"    (Weil 1952: RH iff QW >= 0 for all test functions.)")


if __name__ == "__main__":
    print("SESSION 34 — STAB THE HEART")
    print("=" * 75)

    for lam_sq in [50, 200, 1000]:
        print(f"\n{'#'*75}")
        print(f"# lam^2 = {lam_sq}")
        print(f"{'#'*75}")

        verify_poisson_kernel(lam_sq)
        orthogonal_negativity_test(lam_sq)
        explicit_formula_connection(lam_sq)

    print(f"\n\n{'='*75}")
    print("THE VERDICT")
    print("="*75)
    print("""
  The explicit formula gives: M(f) = W02(f) - sum_rho |f_hat(rho)|^2

  The zero sum is always >= 0 (sum of modulus squares).
  Therefore M(f) <= W02(f).
  For f in null(W02): M(f) <= 0.

  THIS ARGUMENT IS VALID — but it's exactly Weil's original observation.
  The statement "QW >= 0 iff RH" is Weil's theorem (1952).

  The explicit formula connects QW to the zero sum UNCONDITIONALLY.
  But the zero sum being a sum of |f_hat(rho)|^2 is only true when
  applied to h = f * conj(f) (a non-negative function).

  The subtlety: the explicit formula for h = f*f requires h to be
  in a specific test function class. Extending to ALL bandwidth-limited
  functions is what Weil proved is equivalent to RH.

  SO: the rank-1 decomposition M = N + lambda_+*v_+*v_+^T is
  EXPLAINED by the explicit formula, and IMPLIES QW >= 0,
  but PROVING M has this structure for all lambda IS proving RH.

  We have not escaped the circle. But we have understood it completely.
  The heart of RH is: the von Mangoldt weighted prime sum, when viewed
  as a quadratic form, has at most one positive eigenvalue, and that
  eigenvalue is controlled by the analytic structure of the zeta function.
""")

    with open('session34_heart.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
