"""
SESSION 38f — HARD LEFSCHETZ FOR THE DISCRETE SCALING SITE

In Kahler geometry for a surface (complex dim 1):
  H^0 --L--> H^1 --L--> H^2
  Hard Lefschetz: L: H^0 -> H^2 is an isomorphism
  Hodge-Riemann: on P^1 = ker(L: H^1 -> H^3) = H^1, the pairing
    Q(a,b) = -<a, L b> is positive definite

In our setting:
  W02 plays the role of L (the ample class)
  The FULL space C^dim plays the role of H^0 + H^1 + H^2

But the Lefschetz structure is richer than just range/null of W02.
The key is the GRADED structure and the Lefschetz DECOMPOSITION.

For a graded vector space V = V_0 + V_1 + V_2 with L: V_k -> V_{k+1}:
  Hard Lefschetz: L: V_0 -> V_2 is an isomorphism
  This gives the primitive decomposition: V_1 = P^1 + L(V_0)
  And Hodge-Riemann: the form Q(a) = -<a, La> is > 0 on P^1

QUESTION: Can we find a GRADING on our vector space C^dim such that
W02 acts as the Lefschetz operator between grades, and the Hard
Lefschetz property implies Q_W >= 0 on null(W02)?

THE APPROACH:
The Fourier index n gives a natural grading by |n|.
The Weil explicit formula gives a natural action of the "scaling" operator.
Can we organize this into a Lefschetz package?

ALSO: The Connes-Consani paper arXiv:2006.13771 explicitly constructs
the prolate / Sonin / Lefschetz structure. Let me implement the
DISCRETE version of their construction and test it.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def lefschetz_structure(lam_sq, N=None):
    """
    Identify the Lefschetz structure in our matrices.

    The key observation: W02 is rank 2 with eigenvalues lambda_+, lambda_-.
    In Kahler geometry, the ample class has a SIGN: lambda_+ > 0 (ample),
    lambda_- < 0 (the "anti-ample" direction).

    For an arithmetic surface (dim 1):
    - H^0 = span of the "even Poisson" eigenvector of W02 (eigenvalue > 0)
    - H^2 = (implicitly encoded)
    - H^1 = null(W02) (the primitive cohomology)
    - The "odd Poisson" eigenvector (eigenvalue < 0) relates to the
      functional equation / Poincare duality

    The Hodge-Riemann bilinear relation says:
    For phi in H^1 (= null(W02)):
      <phi, Q_W phi> = <phi, W02 phi> - <phi, M phi> = 0 - <phi, M phi> >= 0
    i.e., M <= 0 on null(W02).

    Hard Lefschetz for dim 1 is AUTOMATIC (L: H^0 -> H^2 is an isomorphism
    iff the ample class is non-degenerate, which it is since W02 has rank 2).

    So where is the non-trivial content?

    THE ANSWER: The non-trivial content is in the KAHLER IDENTITIES.
    In classical geometry: [L, Lambda] = (k-n) on H^k (the Lefschetz commutation)
    where Lambda is the "dual Lefschetz" (contraction with the Kahler form).

    In our setting: Lambda would be the ADJOINT of "multiplication by W02."
    The Kahler identity would be: [W02, Lambda] has a specific spectral structure
    that forces M <= 0 on null(W02).

    Can we identify Lambda?
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10

    # The two nonzero eigenvalues of W02
    range_idx = np.where(np.abs(ew) > thresh)[0]
    lambda_pos = ew[range_idx[1]]  # The larger (positive) eigenvalue
    lambda_neg = ew[range_idx[0]]  # The smaller (negative) eigenvalue
    u_pos = ev[:, range_idx[1]]    # "Even Poisson" / ample direction
    u_neg = ev[:, range_idx[0]]    # "Odd Poisson" / anti-ample direction

    print(f"LEFSCHETZ STRUCTURE: lam^2={lam_sq}, dim={dim}", flush=True)
    print(f"  W02 eigenvalues: {lambda_pos:+.4f} (ample), {lambda_neg:+.4f} (anti-ample)", flush=True)

    # The Lefschetz operator L = W02
    # L maps the full space to range(W02) = span(u_pos, u_neg)
    # L restricted to range: L|_range has eigenvalues lambda_+, lambda_-

    # In Kahler geometry, the sl(2) triple is (L, Lambda, H) where:
    # H = degree operator (eigenvalue k-n on H^k)
    # [H, L] = 2L, [H, Lambda] = -2Lambda, [L, Lambda] = H

    # For our rank-2 W02: L = lambda_+ * u+ u+^T + lambda_- * u- u-^T
    # The natural "Lambda" (dual Lefschetz) would satisfy [L, Lambda] = H
    # where H is a "grading" operator.

    # NATURAL GRADING: H[n,n] = f(|n|) for some function f.
    # The Poisson kernel u+ has components ~ 1/(L^2 + 4*pi^2*n^2) (concentrated at n=0)
    # The anti-Poisson u- has components ~ n/(L^2 + 4*pi^2*n^2) (concentrated near n=0, odd)
    # Both are concentrated at LOW |n|.

    # In Hodge theory, L maps H^k to H^{k+2} (increases degree by 2).
    # If we grade by |n|, then "L increases |n|" -- but W02 DECREASES |n|
    # (it projects onto low-|n| directions).

    # So the grading should be OPPOSITE: the "degree" is related to
    # CLOSENESS to the Poisson kernel, not distance from it.

    # Alternative: use the EIGENVALUES of M as the grading.
    # The deeply negative eigenvalues (seeing modes) = "high degree"
    # The zero eigenvalues (silent modes) = "low degree"

    # But this is circular (uses M's eigenstructure).

    # THE KEY INSIGHT: The Lefschetz package in Connes' framework uses
    # the SCALING OPERATOR theta, not a degree grading.
    # theta: f(x) -> f(lambda*x) (scaling by lambda)
    # In our Fourier basis: theta shifts the frequency index.

    # The scaling operator in our discrete basis:
    # theta_lambda acts by: omega_n(x) -> omega_n(x/lambda) (rescale argument)
    # In the Fourier basis with index n and bandwidth L:
    # omega_n(x) = 2(1-x/L)cos(2*pi*n*x/L)
    # After scaling x -> lambda*x:
    # omega_n(lambda*x) = 2(1-lambda*x/L)cos(2*pi*n*lambda*x/L)
    # This mixes different Fourier modes (it's not diagonal in n).

    # For the DISCRETE scaling: theta[n,m] = <omega_n, theta omega_m>
    # = integral_0^L omega_n(x) * omega_m(x/lambda) dx (if we define theta as scaling)

    # Actually, in Connes' framework, the scaling operator theta_lambda
    # is defined on the adele class space and acts on the Weil distribution.
    # Let me try a simpler construction.

    # THE SL(2) APPROACH:
    # If W02 is L, we need Lambda such that [L, Lambda] = H.
    # The simplest: Lambda = pseudoinverse of L restricted to range.
    # Lambda = (1/lambda_+) * u+ u+^T + (1/lambda_-) * u- u-^T

    Lambda_op = (1.0/lambda_pos) * np.outer(u_pos, u_pos) + (1.0/lambda_neg) * np.outer(u_neg, u_neg)

    # Commutator [L, Lambda] = L*Lambda - Lambda*L
    H = W02 @ Lambda_op - Lambda_op @ W02
    # Since L and Lambda are both in span(u+, u-), they commute on that subspace!
    # [L, Lambda] = 0 on range(W02).
    # [L, Lambda] = 0 on null(W02) (both are zero there).
    # So H = 0. Trivial.

    print(f"\n  Naive Lambda = pseudoinverse of W02:", flush=True)
    print(f"  [W02, Lambda] = 0 (trivial, both operate on same 2D subspace)", flush=True)

    # The non-trivial Lambda must involve M or the prime structure!
    # In Kahler geometry, Lambda is determined by the METRIC (not just the Kahler class).
    # The metric in our setting involves the Weil distribution, which includes M.

    # IDEA: Lambda should be related to Q_W^{-1} or M^{-1} on some subspace.
    # The Kahler metric in our setting is not W02 alone — it's the FULL
    # Weil distribution including the prime contributions.

    # THE CONNES CONSTRUCTION:
    # The Weil distribution W_infty is related to the archimedean place.
    # The scaling operator theta acts on Sonin space.
    # The Lefschetz structure comes from the INTERPLAY between
    # theta (geometric/archimedean) and the prime corrections.

    # Let me try: define Lambda using M itself.
    # If M plays the role of the "metric" and W02 is the "Kahler class,"
    # then Lambda could be M^{-1} * W02 * M^{-1} (schematically).

    # But M is singular on null(W02) (zero eigenvalues = silent modes).
    # Use the pseudoinverse of M.

    # M on null(W02) has eigenvalues from -6 to 0.
    # M_pinv = pseudoinverse (invert nonzero eigenvalues, leave zero unchanged)
    P_null = ev[:, np.abs(ew) <= thresh]
    M_null = P_null.T @ M @ P_null
    evals_M, evecs_M = np.linalg.eigh(M_null)

    # Pseudoinverse of M_null
    M_null_pinv = np.zeros_like(M_null)
    for i in range(len(evals_M)):
        if abs(evals_M[i]) > 1e-6:
            M_null_pinv += (1.0/evals_M[i]) * np.outer(evecs_M[:, i], evecs_M[:, i])

    print(f"\n  M_null pseudoinverse:", flush=True)
    print(f"  Rank of M_null: {np.sum(np.abs(evals_M) > 1e-6)}", flush=True)

    # THE CRITICAL COMPUTATION:
    # Define the "Hodge star" operator * on null(W02).
    # In Kahler geometry: * relates H^{p,q} to H^{n-p,n-q}.
    # For a surface (n=1): * maps H^{1,0} to H^{0,1}.
    #
    # In our setting: the "Hodge star" might be related to the
    # SYMMETRY of the Fourier index: n -> -n.
    # This is because the Weil explicit formula has a functional equation
    # symmetry s -> 1-s, which in Fourier space is n -> -n.

    # Define the parity operator P: (P*phi)_n = phi_{-n}
    # This is the "Fourier reflection" or "time reversal"
    P_parity = np.zeros((dim, dim))
    for i in range(dim):
        P_parity[i, dim - 1 - i] = 1.0

    # P_parity maps n -> -n in our indexing (since ns = [-N,...,N])
    # Check: P^2 = I
    assert np.allclose(P_parity @ P_parity, np.eye(dim))

    # How does W02 relate to parity?
    # W02[n,m] = pf * (L^2 - 4*pi^2*mn) / ((L^2 + 4*pi^2*m^2)(L^2 + 4*pi^2*n^2))
    # Under n -> -n, m -> -m: W02[-n,-m] = pf * (L^2 - 4*pi^2*mn) / (same) = W02[n,m]
    # So W02 commutes with parity!

    comm_W02_P = W02 @ P_parity - P_parity @ W02
    print(f"\n  [W02, Parity] = {np.linalg.norm(comm_W02_P):.2e} (should be 0)", flush=True)

    # How does M relate to parity?
    comm_M_P = M @ P_parity - P_parity @ M
    print(f"  [M, Parity]   = {np.linalg.norm(comm_M_P):.2e}", flush=True)

    # M ALSO commutes with parity (all components are even in n -> -n).
    # This means M and W02 share the parity symmetry.

    # The parity decomposes C^dim = V_even + V_odd
    # V_even: phi_n = phi_{-n} (even functions)
    # V_odd:  phi_n = -phi_{-n} (odd functions)

    # The even Poisson kernel u+ is in V_even.
    # The odd Poisson kernel u- is in V_odd.

    # On V_even restricted to null(W02): this is the "H^{1,1}_prim, even" part.
    # On V_odd restricted to null(W02): this is the "H^{1,1}_prim, odd" part.

    # Do even and odd modes behave differently?
    # Restrict M to V_even ^ null(W02) and V_odd ^ null(W02) separately.

    # Build even/odd projectors
    P_even = (np.eye(dim) + P_parity) / 2
    P_odd = (np.eye(dim) - P_parity) / 2

    # null(W02) ^ V_even
    P_null_even = P_null.T @ P_even @ P_null
    evals_ne = np.linalg.eigvalsh(P_null_even)
    null_even_idx = np.where(evals_ne > 0.5)[0]  # eigenvalues of projector are 0 or 1
    null_even_basis = (P_null @ np.linalg.eigh(P_null_even)[1][:, null_even_idx])

    P_null_odd = P_null.T @ P_odd @ P_null
    evals_no = np.linalg.eigvalsh(P_null_odd)
    null_odd_idx = np.where(evals_no > 0.5)[0]
    null_odd_basis = (P_null @ np.linalg.eigh(P_null_odd)[1][:, null_odd_idx])

    d_even = len(null_even_idx)
    d_odd = len(null_odd_idx)

    print(f"\n  PARITY DECOMPOSITION of null(W02):", flush=True)
    print(f"  null ^ V_even: dim {d_even}", flush=True)
    print(f"  null ^ V_odd:  dim {d_odd}", flush=True)
    print(f"  Total: {d_even + d_odd} (null dim = {P_null.shape[1]})", flush=True)

    # M restricted to even and odd parts
    if d_even > 0:
        M_even = null_even_basis.T @ M @ null_even_basis
        evals_Me = np.linalg.eigvalsh(M_even)
        print(f"\n  M on null^V_even: [{np.min(evals_Me):+.4f}, {np.max(evals_Me):+.6e}]", flush=True)
        print(f"  Seeing: {np.sum(np.abs(evals_Me) > 0.001)}, Silent: {np.sum(np.abs(evals_Me) <= 0.001)}", flush=True)

    if d_odd > 0:
        M_odd = null_odd_basis.T @ M @ null_odd_basis
        evals_Mo = np.linalg.eigvalsh(M_odd)
        print(f"  M on null^V_odd:  [{np.min(evals_Mo):+.4f}, {np.max(evals_Mo):+.6e}]", flush=True)
        print(f"  Seeing: {np.sum(np.abs(evals_Mo) > 0.001)}, Silent: {np.sum(np.abs(evals_Mo) <= 0.001)}", flush=True)

    # THE SCALING OPERATOR
    # In Connes' framework, the key operator is theta_lambda: f(x) -> lambda^{1/2} f(lambda*x)
    # In our Fourier basis, this would mix modes (since scaling changes the argument).
    # But there's a simpler version: the "number operator" N: (N*phi)_n = |n| * phi_n
    # This counts the "degree" of each Fourier mode.

    # The number operator in our basis:
    N_op = np.diag(np.abs(ns))

    # Does [W02, N_op] have a specific structure?
    comm_W_N = W02 @ N_op - N_op @ W02
    print(f"\n  THE NUMBER OPERATOR N = diag(|n|):", flush=True)
    print(f"  [W02, N]_op   = {np.linalg.norm(comm_W_N, 2):.4f}", flush=True)
    print(f"  [M, N]_op     = {np.linalg.norm(M @ N_op - N_op @ M, 2):.4f}", flush=True)
    print(f"  [Q_W, N]_op   = {np.linalg.norm(QW @ N_op - N_op @ QW, 2):.4f}", flush=True)

    # The sl(2) triple test: is there an operator Lambda such that
    # L = W02, H = [L, Lambda], [H, L] = 2L, [H, Lambda] = -2Lambda?
    # For rank-2 L, this is very constrained.

    # THE FROBENIUS / HECKE OPERATOR
    # In the function field case, the key operator is Frobenius.
    # In our setting, the analogue might be the "Hecke operator" T_p
    # for each prime p. The Hecke operators commute with each other
    # and with the Lefschetz operator (in the function field case).

    # In our matrices: each T(p^k) is a "Hecke-like" operator.
    # The commutativity [T(p), T(q)] for different primes is a
    # key structural property.

    # Test: do our T(p^k) matrices commute?
    M_diag, M_alpha, M_prime, _, primes = compute_M_decomposition(lam_sq, N)

    # Build T(2) and T(3)
    T_matrices = {}
    for pk, logp, logpk in primes[:5]:
        Q = np.zeros((dim, dim))
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n = ns[j]
                if m != n:
                    Q[i, j] = (np.sin(2*np.pi*n*logpk/L_f) -
                               np.sin(2*np.pi*m*logpk/L_f)) / (np.pi*(m-n))
                else:
                    Q[i, j] = 2*(L_f - logpk)/L_f * np.cos(2*np.pi*m*logpk/L_f)
        Q = (Q + Q.T) / 2
        w = logp * pk**(-0.5)
        T_matrices[pk] = w * Q

    print(f"\n  HECKE COMMUTATIVITY:", flush=True)
    pks = sorted(T_matrices.keys())
    for i in range(len(pks)):
        for j in range(i+1, len(pks)):
            p1, p2 = pks[i], pks[j]
            comm = T_matrices[p1] @ T_matrices[p2] - T_matrices[p2] @ T_matrices[p1]
            print(f"  [T({p1}), T({p2})] = {np.linalg.norm(comm, 'fro'):.4e}", flush=True)

    # Do T(pk) commute with W02?
    print(f"\n  HECKE-LEFSCHETZ COMMUTATIVITY:", flush=True)
    for pk in pks:
        comm = T_matrices[pk] @ W02 - W02 @ T_matrices[pk]
        print(f"  [T({pk}), W02] = {np.linalg.norm(comm, 'fro'):.4e}", flush=True)

    return W02, M, QW, T_matrices


def kahler_identities_test(lam_sq, N=None):
    """
    Test whether any version of the Kahler identities holds.

    In classical geometry:
    [Lambda, d] = -d^c  (the Kahler identity)
    [Lambda, d^c] = d
    These imply: Delta_d = Delta_{d^c} = Delta_{d+d^c}/2

    In our setting: the "d" operator might be related to the DIFFERENCE
    between consecutive Fourier modes (a finite difference operator).
    And d^c might be related to the HILBERT TRANSFORM (the odd part of
    the Fourier decomposition).

    The finite difference operator D: (D*phi)_n = phi_{n+1} - phi_n
    The Hilbert transform H: (H*phi)_n = i*sign(n)*phi_n

    Do these satisfy any identity with W02 and M?
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    print(f"\nKAHLER IDENTITIES TEST at lam^2={lam_sq}", flush=True)

    # Finite difference operator D
    D = np.zeros((dim, dim))
    for i in range(dim - 1):
        D[i, i + 1] = 1
        D[i, i] = -1
    D = (D - D.T) / 2  # Anti-symmetrize for a "d" operator

    # Hilbert-like transform
    H_op = np.diag(np.sign(ns))  # sign(n) operator

    # Multiplication by n
    n_mult = np.diag(ns)

    # Test: [W02, D], [W02, H], [W02, n_mult]
    for name, op in [("D (diff)", D), ("H (Hilbert)", H_op), ("n (mult)", n_mult)]:
        comm = W02 @ op - op @ W02
        anti = W02 @ op + op @ W02
        print(f"  [W02, {name}]_op  = {np.linalg.norm(comm, 2):.4f}", flush=True)

    # THE REAL QUESTION: Is there an operator X such that
    # M = [W02, X] + (something obviously NSD)?
    #
    # If M = [W02, X] on null(W02), then for phi in null(W02):
    # <phi, M phi> = <phi, W02*X phi> - <phi, X*W02 phi>
    #             = 0 - 0 = 0 (since W02*phi = 0)
    #
    # Wait -- that gives M = 0 on null(W02), not M <= 0!
    # The commutator [W02, X] is ZERO on null(W02) for ANY X!
    # Because W02*phi = 0 for phi in null(W02).
    #
    # So writing M = [W02, X] + R gives M|null = R|null.
    # The R must be NSD on null(W02).

    # Compute: R = M - [W02, X] for natural choices of X
    # (minimize ||R|| to find the best X)

    # Try X = M_prime / lambda_+ (schematic)
    ew, ev = np.linalg.eigh(W02)
    lambda_max = np.max(np.abs(ew))

    X = M / lambda_max
    comm_WX = W02 @ X - X @ W02
    R = M - comm_WX

    P_null = ev[:, np.abs(ew) <= np.max(np.abs(ew)) * 1e-10]
    R_null = P_null.T @ R @ P_null
    evals_R = np.linalg.eigvalsh(R_null)

    print(f"\n  Decomposition M = [W02, M/||W02||] + R:", flush=True)
    print(f"  R|null eigenvalues: [{np.min(evals_R):+.4f}, {np.max(evals_R):+.6e}]", flush=True)
    print(f"  R|null NSD: {np.max(evals_R) < 1e-6}", flush=True)
    print(f"  (R|null = M|null since [W02, anything] = 0 on null(W02))", flush=True)

    # Confirmed: [W02, X]|null = 0 for ANY X.
    # So the Kahler identity approach via commutators doesn't directly help.
    # The content must be elsewhere.

    # THE DEEPER STRUCTURE:
    # In Kahler geometry, the Hodge-Riemann bilinear relations follow from
    # the REPRESENTATION THEORY of sl(2, C) on the cohomology.
    # The sl(2) representation decomposes H^* into irreducible components,
    # and the signature on each component is determined by the representation.
    #
    # For a surface: sl(2) acts on H^0 + H^1 + H^2.
    # H^1 = primitive (the part that doesn't come from H^0 via Lefschetz).
    # The Hodge-Riemann sign on H^1 is (-1)^1 = -1 times the "positivity"
    # of the representation.
    #
    # In our setting: the "sl(2) action" comes from (W02, Lambda, H).
    # The representation theory would give the sign of Q_W on null(W02).
    #
    # But we showed [W02, Lambda] = 0 for the natural Lambda.
    # So the sl(2) structure is DEGENERATE — it doesn't give a non-trivial
    # representation.
    #
    # THE RESOLUTION: the sl(2) must involve MORE than just W02.
    # In Connes' framework, the sl(2) comes from the SCALING ACTION theta,
    # not from W02 alone. The full structure involves:
    # - L = W02 (Lefschetz, from the archimedean place)
    # - theta = scaling operator (from the multiplicative structure)
    # - Their interaction (via the Weil explicit formula)

    # Let me test: what is the SCALING OPERATOR in our basis?
    # theta_s : f(x) -> |s|^{1/2} f(sx) for s in R^+
    # In the infinitesimal: d/ds theta_s |_{s=1} = (1/2 + x*d/dx) * f(x)
    #
    # In our Fourier basis with omega_n(x) = 2(1-x/L)cos(2*pi*n*x/L):
    # (x*d/dx) omega_n = -2(x/L)*cos(2*pi*n*x/L) + 2(1-x/L)*(-2*pi*n/L)*x*sin(2*pi*n*x/L)
    # This mixes modes in a specific way.

    # The matrix elements: theta_nm = <omega_n, x*d/dx omega_m>
    # = integral_0^L omega_n(x) * [x * d/dx omega_m(x)] dx

    # This is computable! Let me build it.
    theta_inf = np.zeros((dim, dim))  # infinitesimal scaling
    dx = 0.001
    x_pts = np.arange(dx/2, float(log(mpf(lam_sq))), dx)
    L_f = float(log(mpf(lam_sq)))

    for i in range(dim):
        ni = ns[i]
        omega_n = 2 * (1 - x_pts/L_f) * np.cos(2*np.pi*ni*x_pts/L_f)
        for j in range(dim):
            nj = ns[j]
            # x * d/dx omega_m(x)
            # d/dx [2(1-x/L)cos(2*pi*m*x/L)] = -2/L*cos(2*pi*m*x/L) - 2(1-x/L)*(2*pi*m/L)*sin(2*pi*m*x/L)
            domega_m = (-2.0/L_f * np.cos(2*np.pi*nj*x_pts/L_f)
                        - 2*(1 - x_pts/L_f)*(2*np.pi*nj/L_f)*np.sin(2*np.pi*nj*x_pts/L_f))
            x_domega_m = x_pts * domega_m

            theta_inf[i, j] = np.sum(omega_n * x_domega_m) * dx

    theta_inf = (theta_inf + theta_inf.T) / 2  # Symmetrize

    print(f"\n  SCALING OPERATOR (infinitesimal theta):", flush=True)
    print(f"  ||theta||_op = {np.linalg.norm(theta_inf, 2):.4f}", flush=True)
    print(f"  rank(theta) = {np.sum(np.linalg.svd(theta_inf, compute_uv=False) > 1e-6)}", flush=True)

    # Test the KEY commutator: [W02, theta]
    comm_W_theta = W02 @ theta_inf - theta_inf @ W02
    print(f"  [W02, theta]_op = {np.linalg.norm(comm_W_theta, 2):.4f}", flush=True)

    # Test: [M, theta]
    comm_M_theta = M @ theta_inf - theta_inf @ M
    print(f"  [M, theta]_op   = {np.linalg.norm(comm_M_theta, 2):.4f}", flush=True)

    # Test: [Q_W, theta]
    comm_Q_theta = QW @ theta_inf - theta_inf @ QW
    print(f"  [Q_W, theta]_op = {np.linalg.norm(comm_Q_theta, 2):.4f}", flush=True)

    # THE CRITICAL QUESTION: does [W02, theta] on null(W02) have a definite sign?
    # If [W02, theta]|null is PSD, this is a Kahler-type identity.
    comm_null = P_null.T @ comm_W_theta @ P_null
    evals_comm = np.linalg.eigvalsh(comm_null)
    print(f"\n  [W02, theta] on null(W02):", flush=True)
    print(f"  eigenvalues: [{np.min(evals_comm):+.4f}, {np.max(evals_comm):+.4f}]", flush=True)
    print(f"  PSD: {np.min(evals_comm) > -1e-6}", flush=True)
    print(f"  NSD: {np.max(evals_comm) < 1e-6}", flush=True)

    # And: how does [W02, theta]|null relate to M|null?
    M_null = P_null.T @ M @ P_null
    # Is M proportional to [W02, theta] on null?
    if np.linalg.norm(comm_null) > 1e-6:
        # Find best scalar alpha such that M_null ≈ alpha * comm_null
        alpha = np.sum(M_null * comm_null) / np.sum(comm_null * comm_null)
        residual = M_null - alpha * comm_null
        rel_err = np.linalg.norm(residual, 'fro') / np.linalg.norm(M_null, 'fro')
        print(f"\n  Best fit: M|null ≈ {alpha:.4f} * [W02, theta]|null", flush=True)
        print(f"  Relative error: {rel_err:.4f} ({rel_err*100:.1f}%)", flush=True)

    return theta_inf, comm_W_theta


if __name__ == "__main__":
    print("SESSION 38f — HARD LEFSCHETZ FOR DISCRETE SCALING SITE", flush=True)
    print("=" * 80, flush=True)

    W02, M, QW, T_mats = lefschetz_structure(50)
    theta, comm = kahler_identities_test(50)

    print(f"\nDone.", flush=True)
