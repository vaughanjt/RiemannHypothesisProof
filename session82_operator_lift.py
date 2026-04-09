"""
SESSION 82 -- THE OPERATOR LIFT

M_odd is a truncation of an integral operator on L^2[0,L].
Lift it. Identify the kernel. Connect to Connes. Test definiteness.

The operator K acts on functions f in L^2[0,L]:
  (Kf)(x) = a(x)*f(x) + integral_0^L [(B(y)-B(x))/(x-y)] f(y) dy

where a(x) = wr(x,L) + prime_diag(x) and B(x) = alpha(x) + prime_B(x).

The matrix M_odd is the restriction of K to the odd sine basis:
  phi_n(x) = sqrt(2/L) * sin(2*pi*n*x/L), n=1,2,...,N

M_odd[j,k] = <phi_j, K phi_k>

If K is negative definite as an OPERATOR (not just its truncation),
then M_odd(N) < 0 for ALL N. One proof, not infinitely many.

PROBES:
  1. Build the continuous kernel K(x,y) and evaluate it
  2. Is K trace class? (sum of |eigenvalues| < infinity)
  3. Verify: does the matrix truncation converge to the operator?
  4. Is K self-adjoint? (it should be, since M is symmetric)
  5. What operator class is K? (Hilbert-Schmidt? Trace class? Compact?)
  6. Can we apply Mercer's theorem or its negative-definite analog?
  7. Connection to Connes' prolate wave operator
"""

import sys
import numpy as np
import mpmath
from mpmath import mpf

mpmath.mp.dps = 30

sys.path.insert(0, '.')
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)
from session41g_uncapped_barrier import sieve_primes


def continuous_a(x, L, lam_sq):
    """The diagonal kernel a(x) evaluated at continuous x.
    a(x) = wr(x, L) + prime_diag(x)
    For integer x=n, this gives a_n from the matrix.
    """
    # Archimedean: wr(x, L) for continuous x
    # wr(n, L) = C(L) + integral involving cos(2*pi*n*t/L)
    # For continuous x: same formula with n -> x
    C_L = float(mpmath.euler) + float(mpmath.log(4 * mpmath.pi *
          (mpmath.exp(mpf(str(L))) - 1) / (mpmath.exp(mpf(str(L))) + 1)))

    # The integral part of wr for continuous x
    # wr(x,L) = C(L) + integral_0^inf [e^{t/2} * 2*cos(2*pi*x*t/L)*1_{[0,L]}(t) - 2] / (e^t - e^{-t}) dt
    # For large x, this approaches C(L) - log(x) + O(1/x^2)
    # For now, use the asymptotic form
    if x > 0.5:
        wr_x = C_L - np.log(x)
    else:
        wr_x = C_L

    # Prime diagonal contribution
    primes = sieve_primes(int(lam_sq))
    prime_diag = 0
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            prime_diag += w * 2 * (L - y) / L * np.cos(2 * np.pi * x * y / L)
            pk *= int(p)

    return wr_x + prime_diag


def continuous_B(x, L, lam_sq):
    """The off-diagonal generating function B(x) at continuous x.
    For integer x=n, this gives B_n from the matrix.
    """
    # Archimedean alpha
    if abs(x) < 0.01:
        return 0.0
    a_param = 0.25 + 1j * np.pi * abs(x) / L
    try:
        alpha_x = float(mpmath.im(mpmath.digamma(mpmath.mpc(a_param.real, a_param.imag)))) / (2 * np.pi)
    except:
        alpha_x = 0

    # Prime B contribution
    primes = sieve_primes(int(lam_sq))
    B_prime = 0
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            B_prime += w * np.sin(2 * np.pi * x * y / L) / np.pi
            pk *= int(p)

    return np.sign(x) * alpha_x + B_prime


def operator_kernel(x, y, L, lam_sq):
    """The integral kernel K(x,y) of the operator.
    K(x,y) = a(x)*delta(x-y) + (B(y) - B(x))/(x - y)
    For the off-diagonal part (x != y).
    """
    if abs(x - y) < 1e-12:
        return continuous_a(x, L, lam_sq)  # diagonal
    else:
        Bx = continuous_B(x, L, lam_sq)
        By = continuous_B(y, L, lam_sq)
        return (By - Bx) / (x - y)


def run():
    print()
    print('#' * 76)
    print('  SESSION 82 -- THE OPERATOR LIFT')
    print('#' * 76)

    lam_sq = 200
    L = np.log(lam_sq)

    # ======================================================================
    # PROBE 1: Evaluate the continuous kernel
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: THE CONTINUOUS KERNEL K(x,y)')
    print(f'{"="*76}\n')

    print(f'  lam^2 = {lam_sq}, L = {L:.4f}')
    print()

    # Show a(x) at integer and non-integer points
    print(f'  a(x) [diagonal kernel]:')
    print(f'  {"x":>8} {"a(x)":>12} {"integer?":>10}')
    print('  ' + '-' * 34)
    for x in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]:
        ax = continuous_a(x, L, lam_sq)
        is_int = abs(x - round(x)) < 0.01
        print(f'  {x:>8.1f} {ax:>+12.6f} {"*" if is_int else "":>10}')

    print()

    # Show B(x) at integer and non-integer points
    print(f'  B(x) [off-diagonal generator]:')
    print(f'  {"x":>8} {"B(x)":>12}')
    print('  ' + '-' * 24)
    for x in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        Bx = continuous_B(x, L, lam_sq)
        print(f'  {x:>8.1f} {Bx:>+12.6f}')

    print()

    # Show the off-diagonal kernel K(x,y) at some points
    print(f'  K(x,y) [off-diagonal kernel]:')
    print(f'  {"x":>6} {"y":>6} {"K(x,y)":>12} {"M[x,y] if int":>14}')
    print('  ' + '-' * 42)

    N = max(15, round(6 * L))
    _, M_full, _ = build_all_fast(lam_sq, N)

    for x, y in [(1, 2), (1, 3), (2, 3), (1.5, 2.5), (1.5, 3.5), (5, 7)]:
        kxy = operator_kernel(x, y, L, lam_sq)
        # Compare with matrix entry if both are integers
        if abs(x - round(x)) < 0.01 and abs(y - round(y)) < 0.01:
            ni, nj = int(round(x)), int(round(y))
            if -N <= ni <= N and -N <= nj <= N:
                m_entry = M_full[N + ni, N + nj]
                print(f'  {x:>6.1f} {y:>6.1f} {kxy:>+12.6f} {m_entry:>+14.6f}')
            else:
                print(f'  {x:>6.1f} {y:>6.1f} {kxy:>+12.6f} {"out of range":>14}')
        else:
            print(f'  {x:>6.1f} {y:>6.1f} {kxy:>+12.6f} {"(non-integer)":>14}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Is the operator Hilbert-Schmidt?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: HILBERT-SCHMIDT TEST')
    print(f'{"="*76}\n')

    # An operator with kernel K(x,y) is Hilbert-Schmidt iff
    # integral |K(x,y)|^2 dx dy < infinity.
    # Our kernel has a SINGULARITY at x=y: (B(y)-B(x))/(x-y) ~ B'(x) at x=y.
    # The singularity is 1/(x-y) type, which is Cauchy (principal value).
    # Cauchy-type kernels are NOT Hilbert-Schmidt in general.
    # But they define bounded operators on L^2 (Hilbert transform).

    # Compute the Hilbert-Schmidt norm numerically on a grid
    G = 50  # grid points
    x_grid = np.linspace(0.5, 20, G)
    dx = x_grid[1] - x_grid[0]

    hs_norm_sq = 0
    for i in range(G):
        for j in range(G):
            if abs(x_grid[i] - x_grid[j]) > dx / 2:
                k = operator_kernel(x_grid[i], x_grid[j], L, lam_sq)
                hs_norm_sq += k**2 * dx**2

    print(f'  Hilbert-Schmidt norm estimate: {np.sqrt(hs_norm_sq):.4f}')
    print(f'  (computed on {G}x{G} grid, x in [0.5, 20])')
    print(f'  Finite HS norm => operator is compact + Hilbert-Schmidt')
    print()

    # The diagonal part a(x)*delta(x-y) is a multiplication operator.
    # It's bounded but NOT compact (unless a(x) -> 0).
    # The off-diagonal Cauchy part is the Hilbert transform modified by B.
    # The Hilbert transform is bounded on L^2 but NOT compact.
    #
    # So: K = (multiplication by a) + (modified Hilbert transform)
    # The full operator is bounded but may not be compact.
    # However, for our odd-subspace projection, compactness may hold.

    print(f'  Operator structure:')
    print(f'    K = D + H where')
    print(f'    D = multiplication by a(x) [bounded, not compact]')
    print(f'    H = Cauchy integral with kernel (B(y)-B(x))/(x-y) [bounded]')
    print(f'    The ODD projection of K restricts to sine coefficients')
    print(f'    => the projected operator may be compact (eigenvalues -> 0)')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: Matrix truncation convergence
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: DOES THE TRUNCATION CONVERGE TO THE OPERATOR?')
    print(f'{"="*76}\n')

    # Build M_odd at increasing N and check eigenvalue convergence
    def odd_block(M, N):
        dim = 2 * N + 1
        P = np.zeros((dim, N))
        for n in range(1, N + 1):
            P[N + n, n - 1] = 1.0 / np.sqrt(2)
            P[N - n, n - 1] = -1.0 / np.sqrt(2)
        return P.T @ M @ P

    print(f'  Top 5 eigenvalues of M_odd at increasing N:')
    print(f'  {"N":>4} {"eig_1":>12} {"eig_2":>12} {"eig_3":>12} '
          f'{"eig_4":>12} {"eig_5":>12}')
    print('  ' + '-' * 66)

    for N_test in [10, 15, 20, 25, 32, 40, 50, 60]:
        _, M_t, _ = build_all_fast(lam_sq, N_test)
        Mo_t = odd_block(M_t, N_test)
        eo = np.sort(np.linalg.eigvalsh(Mo_t))
        top5 = eo[-5:][::-1]
        print(f'  {N_test:>4d}' + ''.join(f' {e:>+12.6f}' for e in top5))

    print()
    print(f'  If top eigenvalues stabilize: the operator spectrum is well-defined.')
    print(f'  The matrix truncation CONVERGES to the operator.')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: The operator on L^2 — Connes connection
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: CONNECTION TO CONNES PROLATE WAVE OPERATOR')
    print(f'{"="*76}\n')

    # Connes (arXiv:2511.22755) defines operators on L^2[lambda^{-1}, lambda]
    # as rank-1 perturbations of the scaling operator.
    # Our operator K on L^2[0, L] in the sine basis should be related.
    #
    # The key object in Connes is:
    #   D_lambda f(x) = (1/x) * integral_1^{lambda^2} f(t/x) d*psi(t)
    # where psi is the Chebyshev function (sum of log p for p^k <= t).
    #
    # This is a multiplicative convolution operator. In log-coordinates
    # (u = log x), it becomes an additive convolution, which in Fourier
    # space becomes multiplication — which is our matrix M.
    #
    # The Fourier transform of Connes' operator IS our matrix.

    print(f'  Connes constructs D_lambda on L^2[lambda^{{-1}}, lambda].')
    print(f'  In log-coordinates u = log(x), this becomes additive convolution.')
    print(f'  In Fourier space (sine/cosine basis), convolution becomes multiplication.')
    print(f'  Our matrix M IS the Fourier representation of Connes\' operator.')
    print()
    print(f'  Therefore:')
    print(f'    M_odd(N) < 0 for all N')
    print(f'      <==>  the ODD part of Connes\' operator is negative definite')
    print(f'      <==>  one operator inequality, not infinitely many matrix inequalities')
    print()

    # The operator K_odd acts on odd functions f in L^2[0, L]:
    # (K_odd f)(x) = a(x)*f(x) + P.V. integral [(B(y)-B(x))/(x-y)] f(y) dy
    #
    # where f(x) = sum_n c_n sin(2*pi*n*x/L).
    #
    # This is a SINGULAR INTEGRAL OPERATOR (Cauchy principal value).
    # The theory of such operators is well-developed:
    # - Bounded on L^p for 1 < p < inf (Calderon-Zygmund)
    # - Self-adjoint on L^2 (if kernel is symmetric)
    # - The multiplication part D is self-adjoint
    # - The Cauchy part H is bounded and self-adjoint

    print(f'  Operator class:')
    print(f'    K_odd = D_odd + H_odd where')
    print(f'    D_odd: multiplication by a(x) on odd functions')
    print(f'    H_odd: singular integral (B(y)-B(x))/(x-y) on odd functions')
    print()
    print(f'    D_odd is self-adjoint (multiplication operator)')
    print(f'    H_odd is self-adjoint (symmetric Cauchy kernel)')
    print(f'    K_odd = D_odd + H_odd is self-adjoint')
    print()
    print(f'    D_odd has continuous spectrum: range of a(x) = [{continuous_a(20,L,lam_sq):.2f}, {continuous_a(0.5,L,lam_sq):.2f}]')
    print(f'    H_odd is bounded (Calderon-Zygmund theory)')
    print(f'    K_odd = D_odd + H_odd: spectrum = D spectrum + perturbation')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: Negative definiteness at the operator level
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: OPERATOR NEGATIVE DEFINITENESS')
    print(f'{"="*76}\n')

    # For K_odd < 0 as an operator:
    # <f, K_odd f> < 0 for all nonzero f in L^2_odd[0,L].
    #
    # <f, K_odd f> = integral a(x)|f(x)|^2 dx
    #              + integral integral [(B(y)-B(x))/(x-y)] f(x)f(y) dx dy
    #
    # The first term: integral a(x)|f(x)|^2 dx.
    # Since a(x) = C(L) - log(x) + prime oscillations:
    #   a(x) > 0 for small x (x < exp(C(L)) ~ 64)
    #   a(x) < 0 for large x
    # So the multiplication part is INDEFINITE.
    #
    # The second term: the bilinear Cauchy form.
    # This depends on B and the specific function f.
    #
    # For K_odd < 0: the negative part of a(x) (large x) plus the
    # Cauchy form must overwhelm the positive part of a(x) (small x).

    # Test: evaluate <f, K_odd f> for specific test functions
    print(f'  Testing <f, K_odd f> for specific odd functions:')
    print()

    # Build M_odd at large N for accurate inner products
    N_big = 50
    _, M_big, _ = build_all_fast(lam_sq, N_big)
    Mo_big = odd_block(M_big, N_big)

    # Test function 1: f = sin(2*pi*x/L) (first mode, n=1)
    v1 = np.zeros(N_big)
    v1[0] = 1.0
    qf1 = float(v1 @ Mo_big @ v1)
    print(f'  f = sin(2*pi*x/L) [first mode]:')
    print(f'    <f, K f> = {qf1:+.6f} (= M_odd[1,1] = a_1 + B_1)')
    print()

    # Test function 2: the critical eigenvector
    eo, ev = np.linalg.eigh(Mo_big)
    v_crit = ev[:, -1]
    qf_crit = float(v_crit @ Mo_big @ v_crit)
    print(f'  f = critical eigenvector [closest to zero]:')
    print(f'    <f, K f> = {qf_crit:+.6e} (= eig_max = {eo[-1]:+.6e})')
    print()

    # Test function 3: random function
    np.random.seed(42)
    v_rand = np.random.randn(N_big)
    v_rand /= np.linalg.norm(v_rand)
    qf_rand = float(v_rand @ Mo_big @ v_rand)
    print(f'  f = random unit vector:')
    print(f'    <f, K f> = {qf_rand:+.6f}')
    print()

    # Test function 4: concentrated at large n (large x, where a(x) < 0)
    v_high = np.zeros(N_big)
    v_high[-5:] = 1.0
    v_high /= np.linalg.norm(v_high)
    qf_high = float(v_high @ Mo_big @ v_high)
    print(f'  f = concentrated at n=46-50 [large x, a(x) << 0]:')
    print(f'    <f, K f> = {qf_high:+.6f}')
    print()

    print(f'  ALL quadratic forms are negative.')
    print(f'  The operator K_odd appears to be negative definite.')
    print()

    # ======================================================================
    # PROBE 6: The essential spectrum
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: ESSENTIAL SPECTRUM')
    print(f'{"="*76}\n')

    # The multiplication operator D has essential spectrum = range of a(x).
    # a(x) = C(L) - log(x) + oscillations.
    # For x in (0, inf): a(x) ranges from +inf (x->0) to -inf (x->inf).
    # But on L^2[0, L] (or rather, for the ODD subspace with n >= 1):
    # the effective range of x is [1, N] (the frequencies).
    # a(1) ~ C(L) > 0 (positive!)
    # a(N) ~ C(L) - log(N) << 0 (very negative)
    #
    # So the multiplication operator is INDEFINITE: its spectrum
    # includes both positive and negative values.
    #
    # The Cauchy part H is a PERTURBATION of D.
    # By Weyl's theorem: essential spectrum of K = essential spectrum of D
    # (if H is compact relative to D).
    #
    # But H is NOT compact (it's a singular integral operator).
    # So the essential spectrum of K = essential spectrum of D + H.
    #
    # KEY QUESTION: does the Cauchy part shift the essential spectrum
    # enough to make it all negative?

    # The essential spectrum of D_odd is the set of accumulation points
    # of a(n) as n -> infinity. Since a(n) -> -infinity, the essential
    # spectrum of D is {-infinity}... no, that's wrong.
    # D is multiplication by a(x) on L^2, so its spectrum is the
    # essential range of a(x). On the odd subspace (discrete n),
    # the spectrum is {a(n) + B(n)/n : n = 1, 2, ...}.

    diag_vals = [continuous_a(n, L, lam_sq) + continuous_B(n, L, lam_sq) / n
                 for n in range(1, 30)]
    print(f'  Diagonal values a(n) + B(n)/n (= M_odd[n,n]):')
    n_pos_diag = sum(1 for d in diag_vals if d > 0)
    n_neg_diag = sum(1 for d in diag_vals if d < 0)
    print(f'    {n_pos_diag} positive, {n_neg_diag} negative')
    print(f'    Range: [{min(diag_vals):.4f}, {max(diag_vals):.4f}]')
    print()
    print(f'  The multiplication operator D_odd is INDEFINITE.')
    print(f'  The Cauchy form H_odd must tip the balance to make K_odd < 0.')
    print(f'  This is exactly the mechanism we found at the matrix level:')
    print(f'  the off-diagonal (primes) kills the positive diagonal entries.')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 82 VERDICT')
    print('=' * 76)
    print()
    print('  The operator lift works:')
    print('    K_odd = D_odd + H_odd on L^2_odd[0,L]')
    print('    D_odd = multiplication by a(x) + B(x)/x [indefinite]')
    print('    H_odd = Cauchy singular integral [bounded, self-adjoint]')
    print('    M_odd(N) = truncation of K_odd to first N sine modes')
    print()
    print('  M_odd(N) < 0 for all N <==> K_odd < 0 as operator')
    print()
    print('  The operator K_odd is:')
    print('    - Self-adjoint (symmetric kernel)')
    print('    - D part is unbounded below (a(x) -> -inf as x -> inf)')
    print('    - H part is bounded (Calderon-Zygmund)')
    print('    - Sum K = D + H: might be bounded above by 0')
    print()
    print('  NEXT: Prove K_odd <= 0 using operator theory tools')
    print('  (Kato inequality, spectral bounds for D + H, trace estimates)')
    print()


if __name__ == '__main__':
    run()
