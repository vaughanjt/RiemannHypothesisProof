"""
SESSION 71 -- BRÄNDÉN-HUH LORENTZIAN POLYNOMIAL CONNECTION

Brändén-Huh (Annals 2020) defined Lorentzian polynomials: homogeneous
polynomials p(x_1,...,x_n) of degree d such that:
  (a) All coefficients are non-negative
  (b) The Hessian H_p = (d^2 p / dx_i dx_j) has Lorentzian signature
      (at most 1 positive eigenvalue) at every point in the positive
      orthant R^n_{>0}.

Key theorem: If p is Lorentzian, then its coefficient sequence is
log-concave (ultra-log-concave, in fact).

Our situation:
  - M(lambda) has Lorentzian signature (1, d-1) at ALL tested lambda
  - The xi Taylor coefficients c_k are log-concave (Turan inequalities)
  - Question: is M the Hessian of some Lorentzian polynomial?

This session tests:
  1. Is M(lambda) a valid Hessian? (Symmetric: yes. Can we integrate it?)
  2. If M = Hess(p), what is p? (Integrate M -> gradient -> p)
  3. Does p have non-negative coefficients? (Required for Lorentzian)
  4. Does p's Hessian maintain Lorentzian signature everywhere?
  5. Connection to Turan: are the c_k related to coefficients of p?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import build_all_fast


def get_M_components(lam_sq, N=None):
    """Get M and its components at given lambda^2."""
    L = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L))

    W02, M, QW = build_all_fast(lam_sq, N)

    return W02, M, QW, N, L


def analyze_M_as_bilinear(lam_sq):
    """Analyze M(lambda) as a bilinear form.

    M[n,m] has Cauchy off-diagonal: (B_m - B_n)/(n-m) for n != m
    M[n,n] = a_n (diagonal from wr_diag + alpha)

    If M = Hess(p), then p(x) = (1/2) x^T M x is a quadratic form.
    For a quadratic, the Hessian IS the matrix (constant, independent of x).

    A quadratic p(x) = (1/2) x^T M x with Lorentzian M has:
      - p >= 0 on the positive eigenvector direction
      - p <= 0 on all other directions
      - This is NOT a Lorentzian polynomial (those need non-negative coefficients)

    So the naive "M = Hess of quadratic" doesn't work because M has
    negative eigenvalues, and the quadratic takes negative values.

    But Brändén-Huh's theory is about HIGHER-DEGREE polynomials.
    The Hessian of a degree-d polynomial is degree-(d-2).
    We need: is there a degree-d polynomial p such that
    Hess(p)(v) = M for some specific v in the positive orthant?
    """
    W02, M, QW, N, L = get_M_components(lam_sq)
    dim = 2 * N + 1

    # Eigendecomposition
    evals, evecs = np.linalg.eigh(M)

    n_pos = np.sum(evals > 1e-10)
    n_neg = np.sum(evals < -1e-10)

    print(f'  lambda^2 = {lam_sq}, dim = {dim}')
    print(f'  M signature: ({n_pos}+, {n_neg}-, {dim - n_pos - n_neg} zero)')
    print(f'  Top 5 eigenvalues: {evals[-5:][::-1]}')
    print(f'  Bottom 5 eigenvalues: {evals[:5]}')
    print(f'  Trace(M) = {np.trace(M):.6f}')

    return M, evals, evecs, N, L, dim


def test_generating_function_hessian(c_coeffs, lam_sq):
    """Test: is M related to the Hessian of the generating function
    G(x) = Sum_k c_k * e_k(x) where e_k is the k-th elementary
    symmetric polynomial, evaluated at x = (p^{-s} for primes p)?

    The Euler product: zeta(s) = prod_p 1/(1-p^{-s})
    log zeta(s) = Sum_p Sum_k p^{-ks}/k
    xi(s) = F(s) * zeta(s)

    The connection might go through the SYMMETRIC FUNCTION interpretation
    of the Euler product.
    """
    print(f'\n  Generating function Hessian test (lam^2 = {lam_sq})...')

    # The Euler product viewed as a symmetric function:
    # zeta(s) = Sum_n n^{-s} = Sum over partitions lambda: |Aut(lambda)|^{-1} * prod p^{-k*s}
    # This is the plethystic exponential: PE[Sum_p p^{-s}]

    # For our matrix: M comes from the Weil explicit formula.
    # The prime sum W_p[n,m] = Sum_{p^k <= lam^2} log(p) * p^{-k/2} * q(n,m,log p^k)
    # where q involves the test function.

    # Key observation: M_prime = Sum_p log(p) * p^{-1/2} * (rank-1 outer product)
    # Each prime contributes a rank-1 positive matrix.
    # The sum over primes gives a matrix whose rank grows with the number of primes.

    # A SUM of rank-1 PSD matrices IS the Gram matrix of vectors v_p.
    # M_prime = V V^T where V = [v_p1 | v_p2 | ...]
    # This is always PSD.

    # But our full M = M_prime + M_diag + M_alpha.
    # M_diag is diagonal with mostly negative entries.
    # M_alpha is off-diagonal with small entries.

    # The Lorentzian property: M_diag + M_alpha kills all but 1 direction of M_prime.

    print('  (This is a structural analysis, not a direct computation.)')
    print('  M_prime is PSD (sum of rank-1 outer products from primes)')
    print('  M_diag is mostly negative (archimedean)')
    print('  M = M_prime + M_diag + M_alpha has signature (1, d-1)')


def test_quadratic_form_on_coefficients(M, c_xi, N):
    """Test: is there a relationship between x^T M x and the log-concavity
    of the c_k when x = some specific vector derived from the c_k?

    Hypothesis: set x_n = c_{|n|} (or some function of the Taylor coefficients).
    Then x^T M x might relate to some Turan-like expression.
    """
    print(f'\n  Quadratic form M evaluated on coefficient vectors...')

    dim = 2 * N + 1

    # Test 1: x_n = c_{|n|} for |n| <= K
    K = min(len(c_xi) - 1, N)
    x = np.zeros(dim)
    for n in range(-N, N + 1):
        idx = n + N
        k = abs(n)
        if k < len(c_xi):
            x[idx] = c_xi[k]

    val = x @ M @ x
    print(f'  x = c_|n|: x^T M x = {val:+.10e}')
    print(f'  |x| = {np.linalg.norm(x):.6e}')

    # Test 2: x_n = (-1)^n * c_{|n|}
    x2 = np.zeros(dim)
    for n in range(-N, N + 1):
        idx = n + N
        k = abs(n)
        if k < len(c_xi):
            x2[idx] = (-1)**abs(n) * c_xi[k]

    val2 = x2 @ M @ x2
    print(f'  x = (-1)^|n| c_|n|: x^T M x = {val2:+.10e}')

    # Test 3: x = unit vector along positive eigenvector
    evals, evecs = np.linalg.eigh(M)
    v_plus = evecs[:, -1]  # positive eigenvector
    val_plus = v_plus @ M @ v_plus
    print(f'  x = v_+: x^T M x = {val_plus:+.10e} (= max eigenvalue)')

    return val, val2


def test_lorentzian_polynomial_d3(lam_sq):
    """Test if there's a degree-3 polynomial whose Hessian at a specific
    point equals M.

    For degree 3: p(x) = Sum_{i,j,k} a_{ijk} x_i x_j x_k
    Hess(p)(x) = d^2p/dx_i dx_j = Sum_k 6*a_{ijk} x_k

    So M_{ij} = Sum_k 6*a_{ijk} * x_k = 6 * T_{ij.} . x
    where T is the 3-tensor of coefficients.

    This means: T_{ij.} . x = M_{ij}/6
    i.e., T contracted with x in the third index gives M/6.

    For a SYMMETRIC degree-3 polynomial, T is symmetric in all indices.
    Given M and x, we can solve for T... but T must be totally symmetric
    and have non-negative entries (Lorentzian condition).

    Simplification: restrict to even-parity variables only (since M
    splits into even and odd blocks by Session 57).
    """
    print(f'\n  === DEGREE-3 LORENTZIAN TEST (lam^2={lam_sq}) ===\n')

    W02, M, QW, N, L = get_M_components(lam_sq)
    dim = 2 * N + 1

    # Even subspace
    # Basis: e_0, (e_n + e_{-n})/sqrt(2) for n = 1, ..., N
    # Parity matrix P that extracts even block
    dim_even = N + 1
    P = np.zeros((dim, dim_even))
    P[N, 0] = 1.0  # n=0 -> first even basis
    for n in range(1, N + 1):
        P[N + n, n] = 1 / np.sqrt(2)
        P[N - n, n] = 1 / np.sqrt(2)

    M_even = P.T @ M @ P

    evals_even = np.linalg.eigvalsh(M_even)
    n_pos_even = np.sum(evals_even > 1e-10)
    print(f'  M_even: {dim_even}x{dim_even}, signature ({n_pos_even}+, ...)')
    print(f'  Top eigenvalue: {evals_even[-1]:.6f}')
    print(f'  2nd eigenvalue: {evals_even[-2]:.6e}')
    print(f'  Bottom eigenvalue: {evals_even[0]:.6e}')

    # For a degree-3 poly in dim_even variables:
    # We need x such that T contracted with x gives M_even.
    # Choose x = (1, 1, ..., 1) (all-ones vector in even space).
    # Then T_{ij} := M_even[i,j] / 6 contracted with (1,...,1) works
    # iff we can find T_{ijk} with T_{ij,sum_k} = M_even[i,j]/6.

    # For the ALL-ONES x, the simplest T is:
    # T_{ijk} = M_even[i,j] / (6 * dim_even) for all k
    # But this is NOT totally symmetric unless M_even[i,j] = const.

    # Instead: does M_even have the structure of a "slice" of a
    # symmetric tensor?
    #
    # Necessary condition: M_even must be in the image of the
    # "flattening" map from Sym^3(R^d) -> R^{d x d} via contraction
    # with some x.

    # A simpler test: does there exist a vector x > 0 such that
    # M_even = A . x for some symmetric 3-tensor A with A >= 0?
    # Equivalently: can M_even be written as Sum_k x_k * S_k
    # where each S_k is a PSD matrix (slice of the symmetric tensor)?

    # If M_even = Sum_k x_k S_k with x_k > 0 and S_k PSD,
    # then M_even restricted to any subspace has at most
    # sum_k x_k * rank(S_k) positive eigenvalues.
    # For Lorentzian: we need at most 1 positive eigenvalue,
    # which means the S_k must be simultaneously rank-1 or less.

    # Test: express M_even as a sum of components from each prime.
    # M_prime_even = Sum_p M_p_even where M_p_even is rank-1 PSD.
    # M_diag_even is diagonal (negative).
    # So M_even = (Sum_p M_p_even) + M_diag_even.

    # This IS a decomposition M = A.x + D where A.x is PSD and D is negative.
    # For Lorentzian: we need the PSD part to have rank 1 effectively.
    # But Sum_p M_p_even has rank = number of primes >> 1.

    # HOWEVER: the effective rank of M_prime is much lower (~25) due to
    # prolate structure (Session 56). And the DOMINANT eigenvalue is 20x
    # the second. This "almost rank-1" structure is what makes M Lorentzian.

    print()
    print('  STRUCTURAL OBSERVATION:')
    print('  M_prime = Sum_p (rank-1 PSD matrices from primes)')
    print('  M_diag  = diagonal, mostly negative (archimedean)')
    print('  M       = M_prime + M_diag: signature (1, d-1)')
    print()
    print('  For Brändén-Huh: need M = Hess(p) at some x > 0.')
    print('  For degree 3: M = Sum_k x_k * S_k (symmetric slices)')
    print('  M_prime IS already a sum of PSD rank-1 matrices.')
    print('  Question: can M_diag be absorbed into the Lorentzian structure?')

    return M_even, evals_even


def test_support_function(lam_sq):
    """Test: is there a convex body K such that M(lambda) is related
    to the Hessian of its support function?

    For a convex body K in R^n, the support function
    h_K(u) = max_{x in K} <x, u>
    has Hessian at u with eigenvalues 0 (in the u direction)
    and the principal curvatures of the boundary of K in the
    direction u (all non-negative for convex body).

    So Hess(h_K)(u) is PSD with one zero eigenvalue.
    NOT Lorentzian (which needs exactly 1 positive eigenvalue).

    Alternative: the SECOND MIXED VOLUME is a Lorentzian polynomial.
    Brändén-Huh proved that mixed discriminants of PSD matrices
    are Lorentzian. Our M_prime is a sum of PSD rank-1 matrices...

    Test: compute the "mixed discriminant polynomial" from M_prime's
    rank-1 components and check if its Hessian relates to M.
    """
    print(f'\n  === SUPPORT FUNCTION / MIXED VOLUME TEST ===\n')

    W02, M, QW, N, L = get_M_components(lam_sq)

    # Build M_prime from individual prime contributions
    primes = list(sieve_primes(int(lam_sq) + 1))
    dim = 2 * N + 1

    # Each prime p contributes: log(p) * p^{-1/2} * v_p v_p^T
    # where v_p[n] = cos(n * log(p)) (approximately, for the Lorentzian test function)
    #
    # Actually: M_prime[n,m] = Sum_{p^k <= lam^2} log(p)/p^{k/2} * q(n,m,log(p^k))
    # where q is the test function convolution.
    #
    # For the Lorentzian test function h(r) = 1/(L^2+r^2):
    # q(n,m,g) = e^{-L/2} * e^{-|n-m|g} * [some function]
    #
    # The key point: each prime power contributes a matrix that depends on
    # cos(n*log(p^k)) and sin(n*log(p^k)), creating the phase structure.

    # Compute the generating polynomial from prime phases
    # For a FIXED lambda, define:
    # phi_p(t) = Sum_{n=-N}^{N} t^{n+N} * exp(i*n*log(p))  [Fourier phase]
    # Then M_prime ~ Sum_p w_p * Re(phi_p phi_p^*)  [outer product]

    # The "volume polynomial" would be:
    # Vol(t_1 K_1 + ... + t_P K_P) for K_p = rank-1 body from prime p
    # This is a degree-d polynomial in t_p's.

    # For rank-1 K_p: K_p = v_p v_p^T, so
    # Vol(Sum t_p v_p v_p^T) = det(Sum t_p v_p v_p^T)
    # But det of a rank-1 matrix is 0 unless dim=1.

    # The right object: the PERMANENT or mixed discriminant.
    # Mixed discriminant of A_1, ..., A_d (d x d matrices):
    # D(A_1,...,A_d) = (1/d!) Sum_sigma prod_i (A_{sigma(i)})_{ii}

    # For our case: this is getting complicated. Let's test a simpler idea.

    # SIMPLE TEST: The characteristic polynomial det(xI - M) is related
    # to the eigenvalues. For M Lorentzian, det(xI - M) = x^{d-1}(x - lambda_+)
    # times corrections from negative eigenvalues. Is this log-concave in x?

    evals = np.linalg.eigvalsh(M)

    # Characteristic polynomial coefficients via Vieta's formulas
    # det(xI - M) = prod_i (x - lambda_i) = Sum_k e_{d-k}(-lambda) x^k
    # where e_k is the k-th elementary symmetric polynomial of eigenvalues

    d = len(evals)
    # Use numpy polynomial
    char_poly = np.polynomial.polynomial.polyfromroots(evals)
    # char_poly[k] is coefficient of x^k

    print(f'  Characteristic polynomial of M (degree {d}):')
    print(f'  First 10 coefficients (low degree):')
    for k in range(min(10, len(char_poly))):
        print(f'    x^{k}: {char_poly[k]:+.6e}')

    # Check log-concavity of |char_poly[k]|
    print(f'\n  Log-concavity of |char poly coefficients|:')
    abs_coeffs = np.abs(char_poly)
    for k in range(1, min(len(char_poly) - 1, 15)):
        if abs_coeffs[k - 1] > 0 and abs_coeffs[k + 1] > 0:
            ratio = abs_coeffs[k]**2 / (abs_coeffs[k - 1] * abs_coeffs[k + 1])
            print(f'    k={k:>2d}: R_k = {ratio:>12.6f}  {">=1 OK" if ratio >= 1 else "<1 FAIL"}')
    sys.stdout.flush()

    return char_poly


def run():
    print()
    print('#' * 76)
    print('  SESSION 71 -- BRÄNDÉN-HUH LORENTZIAN POLYNOMIAL CONNECTION')
    print('#' * 76)

    # ==================================================================
    # STEP 1: M's Lorentzian structure at several lambda
    # ==================================================================
    print(f'\n  === STEP 1: M LORENTZIAN STRUCTURE ===\n')

    for lam_sq in [50, 200, 1000, 5000]:
        M, evals, evecs, N, L, dim = analyze_M_as_bilinear(lam_sq)
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Quadratic form on coefficient vectors
    # ==================================================================
    print(f'  === STEP 2: M APPLIED TO xi COEFFICIENTS ===\n')

    # Load Taylor coefficients from Session 70
    import mpmath
    from mpmath import mp, mpf
    mp.dps = 50

    def xi_func(s):
        return mpf('0.5') * s * (s - 1) * mpmath.power(mpmath.pi, -s / 2) * \
               mpmath.gamma(s / 2) * mpmath.zeta(s)

    K = 12
    print(f'  Computing xi Taylor coefficients c_0..c_{K}...')
    s = mpf('0.5')
    xi_val = xi_func(s)
    c_xi = [1.0]
    for k in range(1, K + 1):
        deriv = mpmath.diff(xi_func, s, n=2 * k)
        c_xi.append(float(deriv / xi_val * mpf(-1)**k / mpmath.factorial(2 * k)))
    print(f'  Done. c_1 = {c_xi[1]:.6e}, c_{K} = {c_xi[K]:.6e}')
    sys.stdout.flush()

    for lam_sq in [200, 1000]:
        W02, M, QW, N, L = get_M_components(lam_sq)
        test_quadratic_form_on_coefficients(M, c_xi, N)
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: Degree-3 Lorentzian test on even block
    # ==================================================================
    print(f'  === STEP 3: DEGREE-3 LORENTZIAN TEST ===\n')

    for lam_sq in [200, 1000]:
        M_even, evals_even = test_lorentzian_polynomial_d3(lam_sq)
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: Characteristic polynomial log-concavity
    # ==================================================================
    print(f'  === STEP 4: CHARACTERISTIC POLYNOMIAL ===\n')

    for lam_sq in [50, 200, 1000]:
        char_poly = test_support_function(lam_sq)
        print()
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: The generating polynomial approach
    # ==================================================================
    print(f'\n  === STEP 5: GENERATING POLYNOMIAL ===\n')

    test_generating_function_hessian(c_xi, 1000)

    # ==================================================================
    # STEP 6: Connection between M eigenvalues and Turan ratios
    # ==================================================================
    print(f'\n  === STEP 6: EIGENVALUE-TURAN CONNECTION ===\n')

    # At each lambda, M has eigenvalues evals_1 > 0 > evals_2 >= ... >= evals_d.
    # The Turan ratios are R_k = c_k^2/(c_{k-1}*c_{k+1}).
    # Is there a functional relationship?

    # Collect eigenvalue data across lambda
    print(f'  {"lam^2":>8} {"dim":>5} {"eig_1":>12} {"eig_2":>12} '
          f'{"ratio":>10} {"trace":>12} {"|eig_2/eig_1|":>14}')
    print('  ' + '-' * 80)

    for lam_sq in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        try:
            W02, M, QW, N, L = get_M_components(lam_sq)
            evals = np.linalg.eigvalsh(M)
            e1 = evals[-1]
            e2 = evals[-2]
            ratio = e1 / abs(e2) if abs(e2) > 1e-15 else float('inf')
            tr = np.trace(M)
            rel = abs(e2 / e1)
            print(f'  {lam_sq:>8d} {2*N+1:>5d} {e1:>+12.4f} {e2:>+12.6e} '
                  f'{ratio:>10.2f} {tr:>+12.4f} {rel:>14.6e}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 71 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
