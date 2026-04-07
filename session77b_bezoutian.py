"""
SESSION 77b -- BEZOUTIAN / DISPLACEMENT INERTIA

The displacement equation (Session 74):
  D*M - M*D = 1*B^T - B*1^T     (rank 2)

where D = diag(n), 1 = ones vector, B = B_n vector.

This makes M a "Cauchy-like" matrix with displacement rank 2.

BEZOUTIAN CONNECTION:
The Bezoutian of polynomials p(x), q(x) is:
  Bez(p,q)[i,j] = coefficient of x^i y^j in (p(x)q(y)-p(y)q(x))/(x-y)

Its inertia counts roots: #pos eigs = #roots of p/q in upper half-plane.
(Hermite's theorem / Krein-Naimark)

Our displacement equation has the EXACT same structure:
  (D*M - M*D)[i,j] = (n_i - n_j) * M[i,j] for off-diagonal
                     = B_j - B_i           (from rank-2 RHS)

If M is a Bezoutian, its inertia is determined by a root-counting formula.

PROBES:
  1. Express M in Bezoutian form: find polynomials p, q such that M = Bez(p,q)
  2. Test Heinig-Rost inertia formula for displacement-structured matrices
  3. Does the displacement equation constrain the spectrum directly?
  4. Interlacing as primes are added one-by-one (rank-1 perturbation tracking)
  5. The generating function of B_n — is it related to xi(s)?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def build_M_by_primes(lam_sq, N=None):
    """Build M incrementally: M_arch, then add one prime at a time."""
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    # Archimedean part
    a_arch = np.array([wr[abs(int(n))] for n in ns])
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_offdiag = (alpha[None, :] - alpha[:, None]) / nm
    np.fill_diagonal(alpha_offdiag, 0)
    M_arch = np.diag(a_arch) + alpha_offdiag
    M_arch = (M_arch + M_arch.T) / 2

    # Prime contributions: one rank-1(-ish) update per prime power
    primes = sieve_primes(int(lam_sq))
    prime_updates = []  # list of (p, k, delta_M) for each prime power p^k

    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            phase = 2 * np.pi * ns * y / L
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)

            # Diagonal contribution: w * 2 * cos(2*pi*n*y/L)
            delta_diag = w * 2 * cos_p

            # Off-diagonal: w * (sin(phase_m) - sin(phase_n)) / (pi * (n-m))
            sin_diff = sin_p[None, :] - sin_p[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                delta_offdiag = w * sin_diff / (np.pi * nm)
            np.fill_diagonal(delta_offdiag, 0)

            delta_M = np.diag(delta_diag) + delta_offdiag
            delta_M = (delta_M + delta_M.T) / 2

            prime_updates.append((int(p), pk, delta_M))
            pk *= int(p)

    return M_arch, prime_updates, N, L, dim, ns


def run():
    print()
    print('#' * 76)
    print('  SESSION 77b -- BEZOUTIAN / DISPLACEMENT INERTIA')
    print('#' * 76)

    # ======================================================================
    # PROBE 1: Verify displacement equation on M_odd
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: DISPLACEMENT EQUATION ON M_ODD')
    print(f'{"="*76}\n')

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    _, M, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M, N)

    # D_odd = diag(1, 2, ..., N)
    D_odd = np.diag(np.arange(1, N + 1, dtype=float))

    # Displacement: D*Mo - Mo*D
    disp = D_odd @ Mo - Mo @ D_odd
    disp_rank = np.linalg.matrix_rank(disp, tol=1e-8)

    # Factor: disp = G * J * H^T where J has signature
    U, S, Vt = np.linalg.svd(disp)
    print(f'  lam^2={lam_sq}, N={N}')
    print(f'  Displacement rank: {disp_rank}')
    print(f'  Top 5 singular values: {S[:5]}')
    print(f'  ||D*Mo - Mo*D|| = {np.linalg.norm(disp):.6f}')

    # What's the structure? Is it u*v^T - v*u^T (rank 2, antisymmetric)?
    print(f'  Antisymmetric? ||disp + disp^T|| / ||disp|| = '
          f'{np.linalg.norm(disp + disp.T) / np.linalg.norm(disp):.6e}')

    # Extract the rank-2 factors
    if disp_rank <= 4:
        u1 = U[:, 0] * np.sqrt(S[0])
        v1 = Vt[0, :] * np.sqrt(S[0])
        u2 = U[:, 1] * np.sqrt(S[1])
        v2 = Vt[1, :] * np.sqrt(S[1])
        print(f'  Factor 1: ||u1||={np.linalg.norm(u1):.4f}, ||v1||={np.linalg.norm(v1):.4f}')
        print(f'  Factor 2: ||u2||={np.linalg.norm(u2):.4f}, ||v2||={np.linalg.norm(v2):.4f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Generating function of B_n — connection to xi(s)?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: GENERATING FUNCTION OF B_n')
    print(f'{"="*76}\n')

    M_arch, prime_updates, N, L, dim, ns = build_M_by_primes(lam_sq)

    # Compute B_n
    alpha = _compute_alpha(L, N)
    B_prime = np.zeros(dim)
    for p, pk, _ in prime_updates:
        w = np.log(p) * pk ** (-0.5)
        y = np.log(pk)
        B_prime += w * np.sin(2 * np.pi * ns * y / L) / np.pi

    B_n = alpha + B_prime

    # B(n) is the Fourier coefficient of some function on [0, L].
    # B(n) = integral_0^L f(y) * sin(2*pi*n*y/L) dy  (approx)
    # What function? The explicit formula gives:
    # B_prime(n) = sum_p sum_k log(p) * p^{-k/2} * sin(2*pi*n*log(p^k)/L) / pi
    # This is the Fourier sine coefficient of the PRIME DISTRIBUTION function
    # f_prime(y) = sum_{p^k < e^L} log(p) * p^{-k/2} * delta(y - log(p^k))

    # The generating function F(z) = sum_n B(n) * z^n
    # For |z| < 1, this should converge and be related to a zeta-like function.

    print(f'  B_n values (lam^2={lam_sq}):')
    print(f'  B is odd: B(-n) = -B(n) for all n')
    print(f'  First 15 B values:')
    for n in range(1, 16):
        print(f'    B({n:>2d}) = {B_n[N+n]:>+12.8f}')

    # Generate and evaluate F(z) = sum B(n) z^n at z = e^{i*theta}
    # This is the Fourier transform on the unit circle
    theta_vals = np.linspace(0, 2*np.pi, 1000)
    F_vals = np.zeros(len(theta_vals), dtype=complex)
    for n in range(-N, N+1):
        F_vals += B_n[N + n] * np.exp(1j * n * theta_vals)

    print(f'\n  |F(e^{{i*theta}})| statistics:')
    print(f'    max = {np.max(np.abs(F_vals)):.4f}')
    print(f'    min = {np.min(np.abs(F_vals)):.4f}')
    print(f'    F is real (imaginary part): max = {np.max(np.abs(F_vals.imag)):.6e}')
    print(f'    F changes sign: {np.any(np.diff(np.sign(F_vals.real)) != 0)}')
    n_sign_changes = np.sum(np.abs(np.diff(np.sign(F_vals.real))) > 0)
    print(f'    Number of sign changes: {n_sign_changes}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: Prime-by-prime interlacing on M_odd
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: PRIME-BY-PRIME EIGENVALUE INTERLACING (M_ODD)')
    print(f'{"="*76}\n')

    lam_sq = 200  # smaller for speed
    M_arch, prime_updates, N, L, dim, ns = build_M_by_primes(lam_sq)

    M_current = M_arch.copy()
    Mo_current = odd_block(M_current, N)
    eigs_current = np.linalg.eigvalsh(Mo_current)
    n_pos = np.sum(eigs_current > 1e-10)
    n_neg = np.sum(eigs_current < -1e-10)

    print(f'  Start: M_arch only (no primes)')
    print(f'    M_odd signature: ({n_pos}, {n_neg})')
    print(f'    eig_max = {eigs_current.max():+.6f}')
    print(f'    eig_min = {eigs_current.min():+.6f}')
    print()

    print(f'  Adding primes one by one:')
    print(f'  {"prime":>6} {"pk":>6} {"#pos":>5} {"#neg":>5} '
          f'{"eig_max":>14} {"eig_2":>14} {"neg_def?":>8}')
    print('  ' + '-' * 62)

    last_prime = 0
    for i, (p, pk, delta_M) in enumerate(prime_updates):
        M_current += delta_M
        Mo_current = odd_block(M_current, N)
        eigs = np.linalg.eigvalsh(Mo_current)
        n_pos = np.sum(eigs > 1e-10)
        n_neg = np.sum(eigs < -1e-10)
        eig_max = eigs.max()
        eig_2 = sorted(eigs)[-2] if len(eigs) >= 2 else 0
        is_nd = np.all(eigs < 0)

        # Only print at new primes and key transitions
        if p != last_prime:
            print(f'  {p:>6d} {pk:>6d} {n_pos:>5d} {n_neg:>5d} '
                  f'{eig_max:>+14.6e} {eig_2:>+14.6e} '
                  f'{"YES" if is_nd else "no":>8}')
            last_prime = p
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: When does M_odd BECOME negative definite?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: WHEN DOES M_ODD BECOME NEGATIVE DEFINITE?')
    print(f'{"="*76}\n')

    # Reset and track prime-by-prime
    lam_sq = 1000
    M_arch, prime_updates, N, L, dim, ns = build_M_by_primes(lam_sq)

    M_current = M_arch.copy()

    # Track after each batch of primes
    added_primes = set()
    n_updates = len(prime_updates)

    print(f'  {"#updates":>8} {"last_p":>7} {"eig_max(Mo)":>14} {"#pos":>5} {"neg_def":>8}')
    print('  ' + '-' * 48)

    for i, (p, pk, delta_M) in enumerate(prime_updates):
        M_current += delta_M
        added_primes.add(p)

        # Print at intervals
        if i < 10 or (i < 50 and i % 5 == 0) or i % 20 == 0 or i == n_updates - 1:
            Mo_current = odd_block(M_current, N)
            eigs = np.linalg.eigvalsh(Mo_current)
            eig_max = eigs.max()
            n_pos = np.sum(eigs > 1e-10)
            is_nd = np.all(eigs < 0)

            print(f'  {i+1:>8d} {p:>7d} {eig_max:>+14.6e} {n_pos:>5d} '
                  f'{"YES" if is_nd else "no":>8}')

    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: Displacement equation -> Bezoutian test
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: IS M A BEZOUTIAN?')
    print(f'{"="*76}\n')

    # A Bezoutian Bez(p,q) at nodes x_1,...,x_n satisfies:
    #   D * Bez - Bez * D = displacement of rank <= 2
    # where D = diag(x_1,...,x_n).
    #
    # More precisely: for Bezoutian of polynomials p, q of degree <= n:
    #   Bez[i,j] = sum_k c_k * x_i^{a_k} * x_j^{b_k}  (structured)
    #
    # The INERTIA of the Bezoutian counts roots:
    #   #pos_eigs = # roots of p/q in upper half-plane
    #   #neg_eigs = # roots of p/q in lower half-plane
    #   (Hermite-Biehler type theorem)
    #
    # For OUR M_odd: if it's a Bezoutian of some (p, q) at nodes 1,2,...,N:
    #   #neg_eigs(M_odd) = N means ALL roots of p/q are in lower half-plane
    #   This would give a ROOT-COUNTING interpretation of M_odd < 0!

    lam_sq = 1000
    N_test = max(15, round(6 * np.log(lam_sq)))
    _, M, _ = build_all_fast(lam_sq, N_test)
    Mo = odd_block(M, N_test)

    D_odd = np.diag(np.arange(1, N_test + 1, dtype=float))
    disp_odd = D_odd @ Mo - Mo @ D_odd

    # For a Bezoutian: disp = u * e_n^T - e_1 * v^T (specific structure)
    # Check if displacement has this form
    print(f'  M_odd displacement at lam^2={lam_sq}:')
    U, S, Vt = np.linalg.svd(disp_odd)
    print(f'    Rank: {np.sum(S > 1e-8 * S[0])}')
    print(f'    Singular values: {S[:6]}')

    # Bezoutian has displacement = col_vec * row_vec^T (rank 1 per polynomial)
    # Check: can displacement be written as g*h^T - h*g^T? (antisymmetric, rank 2)
    # Antisymmetric part
    asym = (disp_odd - disp_odd.T) / 2
    sym = (disp_odd + disp_odd.T) / 2
    print(f'    ||antisymmetric|| = {np.linalg.norm(asym):.6f}')
    print(f'    ||symmetric||     = {np.linalg.norm(sym):.6e}')
    print(f'    Displacement is {"antisymmetric" if np.linalg.norm(sym)/np.linalg.norm(asym) < 1e-8 else "NOT antisymmetric"}')

    # For an antisymmetric rank-2 matrix: disp = g*h^T - h*g^T
    # Extract g, h from the SVD
    if np.sum(S > 1e-8 * S[0]) == 2:
        g = U[:, 0] * np.sqrt(S[0])
        h = Vt[0, :] * np.sqrt(S[0])
        # If antisymmetric: g*h^T = -h*g^T implies g proportional to h?
        # No: g*h^T - h*g^T is antisymmetric if g != h.
        cos_angle = abs(g @ h) / (np.linalg.norm(g) * np.linalg.norm(h))
        print(f'    |cos(g,h)| = {cos_angle:.6f}')

    # Can we RECOVER the Bezoutian polynomial from the displacement?
    # For Cauchy-like: M[i,j] = sum_k g_k(i) * h_k(j) / (x_i - x_j)
    # where x_i = i (our nodes).
    # The displacement gives: g_k, h_k are columns of the displacement factors.
    print(f'\n    If M_odd is Cauchy-like with generators from displacement,')
    print(f'    the inertia counts roots of a related function.')
    print(f'    M_odd has {np.sum(np.linalg.eigvalsh(Mo) < 0)} negative eigenvalues')
    print(f'    out of {N_test} total -> root-counting: ALL roots in one half-plane')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: Rank-1 perturbation interlacing — theory test
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: RANK-1 PERTURBATION THEORY FOR PRIME CONTRIBUTIONS')
    print(f'{"="*76}\n')

    # Each prime power contributes delta_M = diag(delta_a) + Cauchy_offdiag
    # The diagonal part is a rank-1 correction (cos pattern)
    # The off-diagonal part is a Cauchy matrix of rank ~1-2

    lam_sq = 200
    M_arch, prime_updates, N, L, dim, ns = build_M_by_primes(lam_sq)

    # Check: is each prime's delta_M approximately rank 1?
    print(f'  Rank of each prime contribution (lam^2={lam_sq}):')
    print(f'  {"p":>4} {"pk":>6} {"rank(1%)":>8} {"rank(0.1%)":>10} '
          f'{"sv1":>10} {"sv2":>10} {"sv1/sv2":>10}')
    print('  ' + '-' * 62)

    for p, pk, delta_M in prime_updates[:15]:
        sv = np.linalg.svd(delta_M, compute_uv=False)
        r1 = np.sum(sv > 0.01 * sv[0])
        r2 = np.sum(sv > 0.001 * sv[0])
        ratio = sv[0] / sv[1] if sv[1] > 1e-15 else float('inf')
        print(f'  {p:>4d} {pk:>6d} {r1:>8d} {r2:>10d} '
              f'{sv[0]:>10.4f} {sv[1]:>10.4f} {ratio:>10.2f}')

    # On the odd subspace: rank of each delta_M_odd?
    print(f'\n  On odd subspace:')
    print(f'  {"p":>4} {"pk":>6} {"rank":>5} {"sv1":>10} {"sv2":>10} {"sv1/sv2":>10}')
    print('  ' + '-' * 50)

    for p, pk, delta_M in prime_updates[:15]:
        delta_Mo = odd_block(delta_M, N)
        sv = np.linalg.svd(delta_Mo, compute_uv=False)
        r1 = np.sum(sv > 0.01 * sv[0])
        ratio = sv[0] / sv[1] if sv[1] > 1e-15 else float('inf')
        print(f'  {p:>4d} {pk:>6d} {r1:>5d} {sv[0]:>10.4f} {sv[1]:>10.4f} '
              f'{ratio:>10.2f}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 77b VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
