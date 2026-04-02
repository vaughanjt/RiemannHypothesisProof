"""
SESSION 39 — CONSTRUCTING THE DISCRETE HODGE STAR

The Hodge star * on null(W02) must satisfy:
  1. *^2 = -Id (complex structure / almost complex structure)
  2. Decomposes null(W02) into two eigenspaces (eigenvalues +i, -i)
  3. M has a definite relationship with * that gives eigenvalue signs

CANDIDATE 1: FUNCTIONAL EQUATION
  The functional equation zeta(s) = chi(s)*zeta(1-s) induces s -> 1-s.
  On the Mellin transform: g-hat(s) -> g-hat(1-s).
  On the critical line (s=1/2+it): this is trivial (t -> t).
  But OFF the critical line: it pairs rho with 1-bar(rho).
  The induced operator on our basis might be the "star."

CANDIDATE 2: HILBERT TRANSFORM
  H: phi_n -> i*sign(n)*phi_n (or sign(n)*phi_n for real version)
  This rotates the Fourier phase by 90 degrees.
  H^2 = -Id on the odd-indexed subspace.
  In classical complex analysis, the Hilbert transform IS the operator
  that relates real and imaginary parts of analytic functions.

CANDIDATE 3: SCALING OPERATOR
  The infinitesimal scaling theta has spectral decomposition.
  Its eigenspaces might define a complex structure.

CANDIDATE 4: THE J OPERATOR (from symplectic geometry)
  If null(W02) has even dimension, there might be a natural
  symplectic form and compatible J with J^2 = -Id.
  The Weil explicit formula might define this symplectic form.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh, gamma
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def setup(lam_sq):
    """Common setup: build matrices and null space."""
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    M_null = P_null.T @ M @ P_null

    return W02, M, QW, P_null, M_null, N, dim, ns, L_f


def test_star_candidate(name, star_null, M_null, d_null):
    """
    Test a candidate Hodge star operator on null(W02).

    Check:
    1. *^2 = -Id (or +Id, or something else)
    2. Eigenvalues of * (should be +i, -i for complex structure)
    3. Decomposition into eigenspaces
    4. M restricted to each eigenspace: definite sign?
    5. Anti-commutation: {M, *} = M* + *M structure
    """
    print(f"\n  CANDIDATE: {name}", flush=True)
    print(f"  Dimension: {d_null}x{d_null}", flush=True)

    # 1. What is *^2?
    star_sq = star_null @ star_null
    # Check if *^2 = -Id
    neg_id_err = np.linalg.norm(star_sq + np.eye(d_null), 'fro') / d_null
    pos_id_err = np.linalg.norm(star_sq - np.eye(d_null), 'fro') / d_null
    zero_err = np.linalg.norm(star_sq, 'fro') / d_null

    if neg_id_err < 0.01:
        print(f"  *^2 = -Id (error {neg_id_err:.2e}) -- COMPLEX STRUCTURE!", flush=True)
        is_complex = True
    elif pos_id_err < 0.01:
        print(f"  *^2 = +Id (error {pos_id_err:.2e}) -- involution, not complex", flush=True)
        is_complex = False
    else:
        print(f"  *^2 = neither +/-Id (errors: -Id={neg_id_err:.2e}, +Id={pos_id_err:.2e})", flush=True)
        is_complex = False

    # 2. Eigenvalues of *
    evals_star = np.linalg.eigvals(star_null)
    # For complex structure: should be +i and -i
    # For involution: should be +1 and -1
    n_plus_i = np.sum(np.abs(evals_star - 1j) < 0.1)
    n_minus_i = np.sum(np.abs(evals_star + 1j) < 0.1)
    n_plus_1 = np.sum(np.abs(evals_star - 1.0) < 0.1)
    n_minus_1 = np.sum(np.abs(evals_star + 1.0) < 0.1)

    print(f"  Eigenvalues near +i: {n_plus_i}, near -i: {n_minus_i}", flush=True)
    print(f"  Eigenvalues near +1: {n_plus_1}, near -1: {n_minus_1}", flush=True)

    # 3. Decompose into eigenspaces
    if is_complex and n_plus_i > 0 and n_minus_i > 0:
        # Eigenspaces of *: V_+ (eigenvalue +i) and V_- (eigenvalue -i)
        # Use: P_+ = (I - i*)/2, P_- = (I + i*)/2
        P_plus = (np.eye(d_null) - 1j * star_null) / 2
        P_minus = (np.eye(d_null) + 1j * star_null) / 2

        # These should be projectors: P^2 = P
        proj_err_p = np.linalg.norm(P_plus @ P_plus - P_plus, 'fro')
        proj_err_m = np.linalg.norm(P_minus @ P_minus - P_minus, 'fro')
        print(f"  Projector errors: P_+ {proj_err_p:.2e}, P_- {proj_err_m:.2e}", flush=True)

        rank_plus = round(np.real(np.trace(P_plus)))
        rank_minus = round(np.real(np.trace(P_minus)))
        print(f"  dim(V_+) = {rank_plus}, dim(V_-) = {rank_minus}", flush=True)

        # 4. M restricted to V_+ and V_-
        # M is real, so we need to work with the real parts
        # The WEIL OPERATOR C = i on V_+, -i on V_-
        # Hodge-Riemann: <a, C*M*a> is positive definite on primitives
        # C = i*P_+ + (-i)*P_- = i*(P_+ - P_-)= i*(-i*) = * (the star itself!)
        # So C = * (the Hodge star IS the Weil operator for a curve)

        # Test: <phi, * M phi> for real phi in null(W02)
        # Since * is real-to-complex, need to be careful.
        # Actually: the Hodge-Riemann form is h(a) = (-1)^{k(k-1)/2} * <a, C bar(a)>
        # For k=1 (our case): h(a) = <a, C bar(a)> = <a, * bar(a)>

        # For REAL vectors phi: <phi, * phi> is the Hodge-Riemann form
        # This should be > 0 for phi in null(W02) with <phi, M phi> < 0 (seeing)
        # and = 0 for silent modes

        # Actually, the correct statement is:
        # h(a) = i * <a, bar(a)>_{Hodge} where <,>_{Hodge} includes the cup product AND *
        # Let me just compute <phi, *M phi> and check its sign
        star_M = star_null @ M_null
        evals_star_M = np.linalg.eigvals(star_M)

        # For the Hodge-Riemann test, consider the SESQUILINEAR form
        # h(phi, psi) = <phi, * M psi>
        # This should be Hermitian and positive (or negative) definite
        H_form = star_M
        sym_err = np.linalg.norm(H_form - H_form.T, 'fro') / np.linalg.norm(H_form, 'fro')
        print(f"\n  HODGE-RIEMANN TEST:", flush=True)
        print(f"  Form h = *M: symmetric error = {sym_err:.4f}", flush=True)
        evals_h = np.linalg.eigvalsh((H_form + H_form.T)/2)
        print(f"  h eigenvalues: [{np.min(evals_h):+.4f}, {np.max(evals_h):+.4f}]", flush=True)
        n_pos_h = np.sum(evals_h > 1e-6)
        n_neg_h = np.sum(evals_h < -1e-6)
        n_zero_h = d_null - n_pos_h - n_neg_h
        print(f"  Positive: {n_pos_h}, Negative: {n_neg_h}, Zero: {n_zero_h}", flush=True)
        if n_neg_h == 0:
            print(f"  *** h = *M IS POSITIVE SEMIDEFINITE! ***", flush=True)
            print(f"  *** THIS IS THE HODGE-RIEMANN BILINEAR RELATION ***", flush=True)

    elif n_plus_1 > 0 and n_minus_1 > 0:
        # Involution: eigenspaces V_+ (eigenvalue +1) and V_- (eigenvalue -1)
        P_plus = (np.eye(d_null) + star_null) / 2
        P_minus = (np.eye(d_null) - star_null) / 2

        rank_plus = round(np.trace(P_plus))
        rank_minus = round(np.trace(P_minus))
        print(f"  dim(V_+) = {rank_plus}, dim(V_-) = {rank_minus}", flush=True)

        # M on each eigenspace
        if rank_plus > 0:
            M_plus = P_plus @ M_null @ P_plus
            evals_mp = np.linalg.eigvalsh(M_plus)
            evals_mp_nz = evals_mp[np.abs(evals_mp) > 1e-10]
            if len(evals_mp_nz) > 0:
                print(f"  M|V_+: [{np.min(evals_mp_nz):+.4f}, {np.max(evals_mp_nz):+.6e}]", flush=True)
        if rank_minus > 0:
            M_minus = P_minus @ M_null @ P_minus
            evals_mm = np.linalg.eigvalsh(M_minus)
            evals_mm_nz = evals_mm[np.abs(evals_mm) > 1e-10]
            if len(evals_mm_nz) > 0:
                print(f"  M|V_-: [{np.min(evals_mm_nz):+.4f}, {np.max(evals_mm_nz):+.6e}]", flush=True)

        # Anti-commutator {*, M} = *M + M*
        anti_comm = star_null @ M_null + M_null @ star_null
        print(f"  ||{{*, M}}||_F = {np.linalg.norm(anti_comm, 'fro'):.4f}", flush=True)
        # Commutator [*, M]
        comm = star_null @ M_null - M_null @ star_null
        print(f"  ||[*, M]||_F = {np.linalg.norm(comm, 'fro'):.4f}", flush=True)

    return star_null


def candidate1_functional_equation(lam_sq):
    """
    CANDIDATE 1: The functional equation s -> 1-s.

    The xi function satisfies xi(s) = xi(1-s).
    On the Mellin side: g-hat(s) -> g-hat(1-s).

    For our basis omega_n(x) = 2(1-x/L)cos(2*pi*n*x/L):
    The Mellin transform at s = sigma+it is:
      omega_n_hat(s) = integral_0^L omega_n(x) x^{s-1} dx

    The functional equation map F: phi -> F(phi) where
    F(phi)_hat(s) = phi_hat(1-s).

    In the x-domain: if phi_hat(s) = integral phi(x) x^{s-1} dx,
    then phi_hat(1-s) = integral phi(x) x^{-s} dx = integral phi(1/x) x^{s-2} dx / x
    So F(phi)(x) = phi(1/x) / x  (Mellin inversion of s -> 1-s).

    But our functions are supported on [0,L], and 1/x maps [0,L] to [1/L, infinity].
    So F doesn't preserve our function space!

    RESOLUTION: Use the COMPLETED L-function's functional equation,
    which includes the Gamma factor. The complete xi function:
    xi(s) = s(s-1)/2 * pi^{-s/2} * Gamma(s/2) * zeta(s)

    The functional equation xi(s) = xi(1-s) is a GLOBAL symmetry.
    In the Connes framework, this is encoded in the W02 matrix.

    Actually, W02 itself comes from the archimedean functional equation.
    The SIGN of W02's negative eigenvalue (-2.93) vs positive (+10.84)
    encodes the functional equation's asymmetry.

    For the DISCRETE operator: the functional equation induces a map
    on the Fourier coefficients. Since xi(s) = xi(1-s) and our test
    functions are even, the map is related to the COSINE TRANSFORM
    evaluated at s and 1-s.
    """
    W02, M, QW, P_null, M_null, N, dim, ns, L_f = setup(lam_sq)
    d_null = P_null.shape[1]

    print(f"\nCANDIDATE 1: FUNCTIONAL EQUATION at lam^2={lam_sq}", flush=True)

    # The functional equation map on our basis:
    # F_nm = <omega_n, F(omega_m)> where F(omega_m)(x) involves the
    # Mellin-inverted functional equation.
    #
    # For a practical approximation: use the xi function's symmetry.
    # xi_hat(n, s) = integral omega_n(x) x^{s-1} dx
    # F maps xi_hat(n, s) -> xi_hat(n, 1-s)
    # The induced kernel: F_nm = integral xi_hat(n, 1-s) * conj(xi_hat(m, s)) ds

    # This is complicated. Let me try a simpler version.
    # The REFLECTION FORMULA for the Gamma function:
    # Gamma(s/2) * pi^{-s/2} = (functional equation factor) * Gamma((1-s)/2) * pi^{-(1-s)/2}
    # The ratio: chi(s) = pi^{s-1/2} * Gamma((1-s)/2) / Gamma(s/2)

    # On the critical line s = 1/2 + it:
    # chi(1/2+it) = pi^{it} * Gamma(1/4 - it/2) / Gamma(1/4 + it/2)
    # |chi| = 1 (it's a phase: chi = e^{i*theta(t)} where theta is the Riemann-Siegel theta)

    # The functional equation map on the critical line is MULTIPLICATION BY chi.
    # In our basis: F_nm = integral omega_n_hat(1/2+it) * chi(1/2+it) * conj(omega_m_hat(1/2+it)) dt / (2*pi)

    # This is the PHASE ROTATION by the Riemann-Siegel theta function!
    # theta(t) = arg(Gamma(1/4 + it/2)) - t/2 * log(pi)

    # Let me compute this operator numerically.
    mp.dps = 30

    def omega_hat(n_val, t):
        """Mellin transform of omega_n at s = 1/2 + it."""
        n_pts = 500
        dx = L_f / n_pts
        result = 0.0 + 0.0j
        for k in range(n_pts):
            x = dx * (k + 0.5)
            val = 2 * (1 - x/L_f) * np.cos(2*np.pi*n_val*x/L_f)
            result += val * np.exp(1j * t * x) * dx
        return result

    def chi_phase(t):
        """The functional equation phase chi(1/2+it) = e^{2i*theta(t)}."""
        if abs(t) < 0.1:
            return 1.0 + 0.0j
        # theta(t) = Im(log(Gamma(1/4 + it/2))) - t/2 * log(pi)
        s = mpc(0.25, float(t)/2)
        log_gamma = mpmath.loggamma(s)
        theta = float(log_gamma.imag) - float(t)/2 * np.log(np.pi)
        return np.exp(2j * theta)

    # Build the functional equation operator F on the full space
    # F_nm = integral omega_n_hat(t) * chi(t) * conj(omega_m_hat(t)) dt / (2*pi)
    # Approximate with a discrete sum over t values
    t_max = N * 2 * np.pi / L_f  # Nyquist-like cutoff
    n_t = 200
    t_vals = np.linspace(-t_max, t_max, n_t)
    dt = t_vals[1] - t_vals[0]

    print(f"  Building F operator (functional equation phase)...", flush=True)
    print(f"  t range: [{-t_max:.1f}, {t_max:.1f}], {n_t} points", flush=True)

    # Precompute omega_hat values
    omega_hats = np.zeros((dim, n_t), dtype=complex)
    for i in range(dim):
        for j, t in enumerate(t_vals):
            omega_hats[i, j] = omega_hat(ns[i], t)

    # Precompute chi values
    chi_vals = np.array([chi_phase(t) for t in t_vals])

    # F_nm = sum_t omega_hat_n(t) * chi(t) * conj(omega_hat_m(t)) * dt / (2*pi)
    F_op = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            F_op[i, j] = np.sum(omega_hats[i, :] * chi_vals * np.conj(omega_hats[j, :])) * dt / (2*np.pi)

    # Project to null(W02)
    F_null = P_null.T @ np.real(F_op) @ P_null
    F_null_imag = P_null.T @ np.imag(F_op) @ P_null

    print(f"  ||F_real||_F = {np.linalg.norm(F_null, 'fro'):.4f}", flush=True)
    print(f"  ||F_imag||_F = {np.linalg.norm(F_null_imag, 'fro'):.4f}", flush=True)

    # Use the IMAGINARY part as the star candidate (it's the phase rotation)
    # Or use the full complex operator
    test_star_candidate("FuncEq (real part)", F_null, M_null, d_null)
    if np.linalg.norm(F_null_imag, 'fro') > 0.01:
        test_star_candidate("FuncEq (imag part)", F_null_imag, M_null, d_null)

    return F_op, F_null


def candidate2_hilbert_transform(lam_sq):
    """
    CANDIDATE 2: The Hilbert transform.

    H: phi_n -> -i * sign(n) * phi_n (multiplies positive frequencies by -i,
    negative by +i).

    In real form: H maps cos(n*x) -> sin(n*x) and sin(n*x) -> -cos(n*x).
    This is a 90-degree phase shift.

    H^2 = -Id (on the space of nonzero-frequency functions).
    At n=0: H = 0 (the DC component has no phase to shift).

    The Hilbert transform is the CANONICAL complex structure on L^2(R).
    It's what makes the Hardy space H^2 work — analytic functions have
    only positive-frequency components.

    In our setting: H restricted to null(W02) should give a complex structure
    on the primitive cohomology. The H^{1,0} and H^{0,1} decomposition
    would correspond to the "analytic" and "anti-analytic" parts.
    """
    W02, M, QW, P_null, M_null, N, dim, ns, L_f = setup(lam_sq)
    d_null = P_null.shape[1]

    print(f"\nCANDIDATE 2: HILBERT TRANSFORM at lam^2={lam_sq}", flush=True)

    # Hilbert transform in Fourier basis: H[n,n] = -i*sign(n)
    # For real matrices: H_real[n,m] = 0 for n != m
    # But wait -- we need a REAL operator on real vectors.
    # The real version: (Hf)(x) = PV integral f(t)/(x-t) dt / pi
    # In Fourier: H maps cos(nx) -> sin(nx), sin(nx) -> -cos(nx)
    #
    # In our basis omega_n(x) = 2(1-x/L)cos(2*pi*n*x/L):
    # The Hilbert transform would give something like 2(1-x/L)sin(2*pi*n*x/L)
    # But these sin functions are NOT in our basis!
    # Our basis is purely cosines (even in n <-> -n).
    #
    # So the Hilbert transform maps our EVEN basis to an ODD basis.
    # It's not an endomorphism of our function space!
    #
    # RESOLUTION: Use the SIGN operator instead.
    # sign(n) maps phi_n -> sign(n)*phi_n.
    # sign^2 = Id on n != 0 (involution, not complex structure).
    # But: i*sign on the complexified space would give complex structure.

    # For a REAL complex structure: we need J with J^2 = -Id.
    # The standard construction: pair n with -n.
    # J(phi)_n = phi_{-n} * sign(n) (or similar)

    # Actually, the correct real complex structure from the Hilbert transform:
    # On the 2D subspace {cos(n*x), sin(n*x)}: J maps cos -> sin, sin -> -cos.
    # This gives J^2 = -Id on each 2D block.
    #
    # In our basis (only cos): we need to EXTEND to include sin.
    # But our null(W02) is even (M commutes with parity), so we only have cos.
    #
    # The right thing: pair phi_n with phi_{-n} using the ANTISYMMETRIC combination.
    # Even functions: phi_n = phi_{-n} -> "real part"
    # Odd functions: phi_n = -phi_{-n} -> "imaginary part"
    # J maps even -> odd, odd -> -even.

    # Build J: on the FULL space, J maps e_n -> -sign(n)*e_{-n}
    # (This is: J cos(nx) = sin(nx), J sin(nx) = -cos(nx) in a suitable sense)
    J = np.zeros((dim, dim))
    for i in range(dim):
        n = int(ns[i])
        # Map n -> -n with a sign
        j = dim - 1 - i  # index of -n
        if n > 0:
            J[i, j] = -1  # positive n -> negative n with sign
        elif n < 0:
            J[i, j] = 1   # negative n -> positive n with sign
        # n = 0: J = 0 (no partner)

    J = (J - J.T) / 2  # Antisymmetrize to ensure J^T = -J

    # Check J^2
    J_sq = J @ J
    neg_id_err = np.linalg.norm(J_sq + np.eye(dim), 'fro') / dim
    print(f"  J^2 + Id error (full space): {neg_id_err:.4e}", flush=True)

    # Project J to null(W02)
    J_null = P_null.T @ J @ P_null

    test_star_candidate("Hilbert J (antisymmetric n<->-n)", J_null, M_null, d_null)

    # Also try: the WEIGHTED version J_w where the sign depends on
    # which side of the Poisson kernel peak the mode is on
    # J_w[n,m] = weight(n) * J[n,m]
    # The weight could be 1/sqrt(|wr_diag[n]|) or similar

    return J_null


def candidate3_scaling_spectral(lam_sq):
    """
    CANDIDATE 3: Scaling operator spectral decomposition.

    The infinitesimal scaling operator theta = x*d/dx + 1/2
    has eigenvalues and eigenspaces. If the eigenvalues come in
    +/- pairs, the pairing gives a complex structure.
    """
    W02, M, QW, P_null, M_null, N, dim, ns, L_f = setup(lam_sq)
    d_null = P_null.shape[1]

    print(f"\nCANDIDATE 3: SCALING OPERATOR at lam^2={lam_sq}", flush=True)

    # Build theta = x*d/dx (infinitesimal scaling)
    n_pts = 2000
    x_pts = np.linspace(0.001, L_f - 0.001, n_pts)
    dx = x_pts[1] - x_pts[0]

    theta = np.zeros((dim, dim))
    for i in range(dim):
        ni = ns[i]
        omega_n = 2 * (1 - x_pts/L_f) * np.cos(2*np.pi*ni*x_pts/L_f)
        for j in range(dim):
            nj = ns[j]
            domega_m = (-2.0/L_f * np.cos(2*np.pi*nj*x_pts/L_f)
                        - 2*(1 - x_pts/L_f)*(2*np.pi*nj/L_f)*np.sin(2*np.pi*nj*x_pts/L_f))
            x_domega_m = x_pts * domega_m
            theta[i, j] = np.sum(omega_n * x_domega_m) * dx

    theta = (theta + theta.T) / 2  # Symmetrize

    # Project to null(W02)
    theta_null = P_null.T @ theta @ P_null

    # Eigendecomposition of theta_null
    evals_theta, evecs_theta = np.linalg.eigh(theta_null)

    print(f"  theta|null eigenvalues: [{np.min(evals_theta):.4f}, {np.max(evals_theta):.4f}]", flush=True)
    print(f"  # positive: {np.sum(evals_theta > 0.01)}", flush=True)
    print(f"  # negative: {np.sum(evals_theta < -0.01)}", flush=True)
    print(f"  # near zero: {np.sum(np.abs(evals_theta) <= 0.01)}", flush=True)

    # If eigenvalues come in +/- pairs, define J using the pairing
    # Sort by absolute value and pair up
    abs_evals = np.abs(evals_theta)
    sorted_idx = np.argsort(abs_evals)[::-1]

    # Build J from +/- pairing
    J_theta = np.zeros((d_null, d_null))
    used = set()
    pairs = []
    for i in sorted_idx:
        if i in used:
            continue
        if abs(evals_theta[i]) < 0.01:
            continue
        # Find the partner with opposite sign and similar magnitude
        best_j = None
        best_diff = float('inf')
        for j in sorted_idx:
            if j in used or j == i:
                continue
            if evals_theta[i] * evals_theta[j] < 0:  # opposite signs
                diff = abs(abs(evals_theta[i]) - abs(evals_theta[j]))
                if diff < best_diff:
                    best_diff = diff
                    best_j = j
        if best_j is not None and best_diff < 0.1 * abs(evals_theta[i]):
            pairs.append((i, best_j))
            used.add(i)
            used.add(best_j)
            # J maps v_i -> v_j and v_j -> -v_i
            v_i = evecs_theta[:, i]
            v_j = evecs_theta[:, best_j]
            J_theta += np.outer(v_j, v_i) - np.outer(v_i, v_j)

    print(f"  Found {len(pairs)} eigenvalue pairs", flush=True)

    if len(pairs) > 0:
        test_star_candidate("Scaling J (eigenvalue pairing)", J_theta, M_null, d_null)

    return theta_null, J_theta


def candidate4_symplectic(lam_sq):
    """
    CANDIDATE 4: Symplectic structure from the explicit formula.

    The Weil explicit formula defines a PAIRING between the "analytic side"
    and the "spectral side." This pairing might be symplectic.

    A symplectic form omega on null(W02) with compatible J (J^2 = -Id,
    omega(Jx, Jy) = omega(x,y)) gives a Kahler structure.

    The explicit formula: <phi, Q_W psi> = sum_rho g_phi(rho) * conj(g_psi(rho))
    Under RH, this IS a Hermitian inner product (a Kahler metric!).
    The imaginary part of this Hermitian form is the symplectic form.

    But we can also define omega from the COMMUTATOR structure:
    omega(phi, psi) = <phi, M psi> - <psi, M phi> = 0 (since M is symmetric)

    Hmm, the natural pairing from M is symmetric, not antisymmetric.
    We need an ANTISYMMETRIC pairing.

    The natural antisymmetric pairing on H^1: the CUP PRODUCT.
    For an arithmetic surface: the cup product on H^1 is the INTERSECTION FORM.
    In our setting: what is the intersection form?

    The Weil explicit formula gives:
    <phi, Q_W psi> = <phi, W02 psi> - <phi, M psi>

    This is symmetric. The antisymmetric part comes from the COMPLEX structure.
    If J is the complex structure, then:
    omega(phi, psi) = <phi, J psi>  (the Kahler form)
    g(phi, psi) = <phi, psi>  (the metric)
    h = g + i*omega  (the Hermitian form)

    So: the symplectic form IS omega(phi,psi) = <phi, J psi>.
    We need J first, then omega follows.

    REVERSED APPROACH: Can we define omega independently and extract J?
    The exterior product structure on the explicit formula might give omega.
    """
    W02, M, QW, P_null, M_null, N, dim, ns, L_f = setup(lam_sq)
    d_null = P_null.shape[1]

    print(f"\nCANDIDATE 4: SYMPLECTIC / CUP PRODUCT at lam^2={lam_sq}", flush=True)

    # The cup product on H^1 of a curve is the intersection pairing.
    # For our discrete setting, a natural antisymmetric form:
    # omega(phi, psi) = sum_n n * (phi_n * psi_{-n} - phi_{-n} * psi_n)
    # This pairs positive and negative Fourier modes with the index as weight.

    # Build omega matrix: omega[i,j] = n_i * delta(n_i, -n_j) - (i <-> j)
    omega = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            ni, nj = int(ns[i]), int(ns[j])
            if ni == -nj and ni != 0:
                omega[i, j] = ni

    # Antisymmetrize
    omega = (omega - omega.T) / 2

    # Project to null(W02)
    omega_null = P_null.T @ omega @ P_null

    print(f"  Cup product omega: rank = {np.linalg.matrix_rank(omega_null, tol=1e-6)}", flush=True)
    print(f"  Antisymmetric: {np.linalg.norm(omega_null + omega_null.T):.2e}", flush=True)

    # If omega is non-degenerate (full rank on null), we can define J = omega^{-1} * g
    # where g is the metric (= identity or = -M).
    # J = -M^{-1} * omega or omega * (-M)^{-1}

    if np.linalg.matrix_rank(omega_null, tol=1e-6) == d_null:
        print(f"  omega is NON-DEGENERATE on null(W02)!", flush=True)

        # J = omega^{-1} * (-M_null)
        # This gives J^2 = omega^{-1} * (-M) * omega^{-1} * (-M) = omega^{-2} * M^2
        # For J^2 = -Id: need omega^{-2} * M^2 = -Id, i.e., M^2 = -omega^2
        omega_inv = np.linalg.inv(omega_null)
        J_symp = omega_inv @ (-M_null)

        test_star_candidate("Symplectic J = omega^{-1}*(-M)", J_symp, M_null, d_null)

        # Also try: J = (-M_null)^{-1/2} * omega * (-M_null)^{-1/2}
        # (This is the "compatible complex structure" from symplectic geometry)
        # Need M_null to be NSD (it is!) to take sqrt of -M_null
        neg_M = -M_null
        evals_nM, evecs_nM = np.linalg.eigh(neg_M)
        evals_nM = np.maximum(evals_nM, 0)  # clip
        sqrt_neg_M = evecs_nM @ np.diag(np.sqrt(evals_nM)) @ evecs_nM.T
        inv_sqrt_neg_M = np.zeros_like(neg_M)
        for i in range(d_null):
            if evals_nM[i] > 1e-10:
                inv_sqrt_neg_M += (1.0/np.sqrt(evals_nM[i])) * np.outer(evecs_nM[:, i], evecs_nM[:, i])

        J_compat = inv_sqrt_neg_M @ omega_null @ inv_sqrt_neg_M
        test_star_candidate("Compatible J = (-M)^{-1/2} omega (-M)^{-1/2}", J_compat, M_null, d_null)
    else:
        print(f"  omega is DEGENERATE (rank {np.linalg.matrix_rank(omega_null, tol=1e-6)} < {d_null})", flush=True)
        print(f"  Cannot define J from omega.", flush=True)

    return omega_null


if __name__ == "__main__":
    print("SESSION 39 -- CONSTRUCTING THE DISCRETE HODGE STAR", flush=True)
    print("=" * 80, flush=True)

    lam_sq = 50

    # Candidate 1: Functional equation
    F_op, F_null = candidate1_functional_equation(lam_sq)

    # Candidate 2: Hilbert transform
    J_hilbert = candidate2_hilbert_transform(lam_sq)

    # Candidate 3: Scaling operator
    theta_null, J_theta = candidate3_scaling_spectral(lam_sq)

    # Candidate 4: Symplectic
    omega_null = candidate4_symplectic(lam_sq)

    print(f"\nDone.", flush=True)
