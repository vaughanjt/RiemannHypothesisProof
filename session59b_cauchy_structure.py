"""
SESSION 59b -- M HAS CAUCHY STRUCTURE (= CONNES' MATRIX)

Key realization from arXiv:2511.22755: Connes' truncation matrix has
the form tau_{i,j} = a_i * delta_{ij} + (b_i - b_j)/(i - j).

OUR M matrix has EXACTLY the same structure:

  M[n,m] = a_n * delta_{nm} + (B_m - B_n) / (n - m)

where:
  a_n = wr_diag[|n|] + sum_{pk} w_pk * 2*cos(2*pi*n*y_pk/L)
  B_n = alpha[n] + sum_{pk} w_pk * sin(2*pi*n*y_pk/L) / pi

This is the "Cauchy + diagonal" structure that CF theory controls.

Plan:
  1. Extract (a_n, B_n) from our matrix M.
  2. Verify M = diag(a) + Cauchy(B) to machine precision.
  3. Do the same on the odd block.
  4. Compute the associated CF symbol.
  5. Check if the symbol determines negativity of M_odd.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _build_M_prime, _compute_alpha, _compute_wr_diag
)


def extract_cauchy_coefficients(lam_sq):
    """
    Decompose M into diag(a) + Cauchy(B) where Cauchy(B)[n,m] = (B_m - B_n)/(n-m).

    Returns a_n, B_n arrays and the reconstructed matrix for verification.
    """
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # wr_diag
    wr = _compute_wr_diag(L, N)

    # alpha
    alpha = _compute_alpha(L, N)

    # Prime contributions
    primes = sieve_primes(int(lam_sq))
    a_prime = np.zeros(dim)  # diagonal part from primes
    B_prime = np.zeros(dim)  # Cauchy b-function from primes

    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            # Diagonal: w * 2*(L-y)/L * cos(2*pi*n*y/L)
            # But the Cauchy limit gives: w * (-2*y/L) * cos(2*pi*n*y/L)
            # And the extra diagonal is: w * 2 * cos(2*pi*n*y/L)
            # So: diagonal_total = w * 2*(1 - y/L) * cos(2*pi*n*y/L)
            #                    = w * 2 * cos(2*pi*n*y/L) + Cauchy_diag_limit
            # where Cauchy_diag_limit = -w * 2*y/L * cos(2*pi*n*y/L)
            # And Cauchy off-diagonal = w * (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))

            # The Cauchy part has b_n = w * sin(2*pi*n*y/L) / pi
            # Its diagonal limit is: d/dn [w * sin(2*pi*n*y/L)/pi] = w * 2*y/L * cos(2*pi*n*y/L)/pi ... hmm

            # Let me think more carefully.
            # Off-diagonal: (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))
            # This is the Cauchy matrix with b_n = sin(2*pi*n*y/L) / pi
            # and Cauchy[n,m] = (b_m - b_n) / (n-m)
            # Diagonal limit: lim_{m->n} = b'(n) = (2*y/L)*cos(2*pi*n*y/L)/pi * (1/pi)
            # Wait: b_n = sin(2*pi*n*y/L)/pi, db/dn = (2*pi*y/L)*cos(2*pi*n*y/L)/pi = 2*y/L * cos(2*pi*n*y/L)

            # Hmm, that gives Cauchy diagonal = 2*y/L * cos(2*pi*n*y/L)
            # But the actual M_prime diagonal is 2*(L-y)/L * cos(2*pi*n*y/L)
            # = 2*cos(2*pi*n*y/L) - 2*y/L * cos(2*pi*n*y/L)

            # So the split is:
            # M_prime[n,n] = 2*cos(2*pi*n*y/L) - 2*y/L * cos(2*pi*n*y/L)
            #              = 2*cos(2*pi*n*y/L) + [Cauchy diagonal limit with NEGATIVE sign]
            #
            # Because Cauchy[n,m] = (b_m - b_n)/(n-m), the diagonal limit is:
            # lim_{m->n} (b_m - b_n)/(n-m) = -b'(n) = -2*y/L * cos(2*pi*n*y/L)
            #
            # So M_prime = diag(2*cos) + Cauchy(b=sin/pi) where Cauchy diagonal = -2*y/L*cos
            # Total M_prime diagonal = 2*cos - 2*y/L*cos = 2*(1-y/L)*cos ✓

            a_prime += w * 2 * np.cos(2 * np.pi * ns * y / L)
            B_prime += w * np.sin(2 * np.pi * ns * y / L) / np.pi

            pk *= int(p)

    # Total a_n and B_n
    a_n = np.array([wr[abs(int(n))] for n in ns]) + a_prime
    B_n = alpha + B_prime

    # Reconstruct M from (a, B) and compare
    # M_reconstructed[n,m] = a_n * delta + (B_m - B_n) / (n - m)
    M_recon = np.diag(a_n)
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        cauchy = (B_n[None, :] - B_n[:, None]) / nm
    # Diagonal of Cauchy: derivative limit
    # d B_n / dn = d/dn [alpha(n) + sum w*sin(2*pi*n*y/L)/pi]
    # For alpha: alpha[N+n] = val, alpha[N-n] = -val, so d alpha/dn ≈ alpha[N+n+1]-alpha[N+n]
    # This is approximate. Instead, fill diagonal from the ACTUAL M diagonal.
    np.fill_diagonal(cauchy, 0.0)
    M_recon += cauchy
    M_recon = (M_recon + M_recon.T) / 2

    return a_n, B_n, M_recon, L, N, dim


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P, P


def run():
    print()
    print('#' * 76)
    print('  SESSION 59b -- M HAS CAUCHY STRUCTURE')
    print('#' * 76)

    # == Part 1: Extract and verify Cauchy decomposition ==
    print('\n  === PART 1: CAUCHY DECOMPOSITION VERIFICATION ===')
    print(f'  M = diag(a) + Cauchy(B) where Cauchy[n,m] = (B_m - B_n)/(n-m)')
    print()

    for lam_sq in [200, 1000, 5000]:
        a_n, B_n, M_recon, L, N, dim = extract_cauchy_coefficients(lam_sq)

        # Build actual M
        _, M_actual, _ = build_all_fast(lam_sq, N)

        # Compare (off-diagonal only, since diagonal of Cauchy needs derivative limit)
        mask = ~np.eye(dim, dtype=bool)
        diff_offdiag = np.abs(M_recon[mask] - M_actual[mask])
        max_diff = diff_offdiag.max()
        rel_diff = max_diff / np.abs(M_actual[mask]).max()

        print(f'  lam^2={lam_sq:>6d}: dim={dim}, '
              f'max|off-diag diff|={max_diff:.2e}, '
              f'relative={rel_diff:.2e}')

        # Check diagonal
        diag_diff = np.abs(np.diag(M_recon) - np.diag(M_actual))
        print(f'    diagonal diff: max={diag_diff.max():.2e}, '
              f'mean={diag_diff.mean():.2e}')
    sys.stdout.flush()

    # == Part 2: Cauchy coefficients on odd block ==
    print('\n  === PART 2: CAUCHY STRUCTURE ON ODD BLOCK ===')
    print(f'  On odd subspace, the Cauchy structure becomes:')
    print(f'  M_odd[i,j] = a_odd[i]*delta + (B_odd[j] - B_odd[i])/(i-j)')
    print()

    lam_sq = 1000
    a_n, B_n, M_recon, L, N, dim = extract_cauchy_coefficients(lam_sq)

    # a_n and B_n for odd indices (n=1..N)
    # In the odd subspace basis (|n> - |-n>)/sqrt(2):
    # a_odd[k] involves a_{k+1} and a_{-(k+1)}
    # For even functions: a_n = a_{-n}, so on odd subspace the diagonal is a_n.
    # For B: B_{-n} = -B_n (alpha is odd, sin is odd), so B_odd = B_n for n>0.

    a_odd = np.array([a_n[N + k] for k in range(1, N + 1)])
    B_odd = np.array([B_n[N + k] for k in range(1, N + 1)])

    print(f'  a_odd (first 10): {a_odd[:10]}')
    print(f'  B_odd (first 10): {B_odd[:10]}')
    print()

    # Reconstruct M_odd from Cauchy
    idx = np.arange(1, N + 1, dtype=float)
    idx_diff = idx[:, None] - idx[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        M_odd_recon = np.diag(a_odd) + (B_odd[None, :] - B_odd[:, None]) / idx_diff
    np.fill_diagonal(M_odd_recon, a_odd)  # Cauchy diagonal = 0, total diag = a_odd
    # Actually need the Cauchy diagonal limit too
    # Cauchy[k,k] = lim_{j->k} (B_j - B_k)/(k-j) = -B'(k)
    # Approximate B'(k) by finite difference
    B_deriv = np.zeros(N)
    for k in range(N):
        if k == 0:
            B_deriv[k] = B_odd[1] - B_odd[0]
        elif k == N - 1:
            B_deriv[k] = B_odd[k] - B_odd[k - 1]
        else:
            B_deriv[k] = (B_odd[k + 1] - B_odd[k - 1]) / 2
    np.fill_diagonal(M_odd_recon, a_odd - B_deriv)

    M_odd_recon = (M_odd_recon + M_odd_recon.T) / 2

    # Compare to actual M_odd
    _, M_actual, _ = build_all_fast(lam_sq, N)
    M_odd_actual, P = odd_block(M_actual, N)

    diff = np.abs(M_odd_recon - M_odd_actual)
    print(f'  Cauchy reconstruction of M_odd:')
    print(f'    max diff: {diff.max():.4e}')
    print(f'    mean diff: {diff.mean():.4e}')
    print(f'    Frobenius: {np.linalg.norm(diff):.4e} / {np.linalg.norm(M_odd_actual):.4e}')
    sys.stdout.flush()

    # == Part 3: The Cauchy symbol ==
    print('\n  === PART 3: CAUCHY/CF SYMBOL ===')
    print(f'  For a matrix tau_ij = a_i*delta + (b_i-b_j)/(i-j),')
    print(f'  define the CF generating function:')
    print(f'    phi(theta) = sum_n (a_n + i*B_n) * exp(i*n*theta)')
    print(f'  The REAL part Re(phi) controls the eigenvalues.')
    print()

    # On odd block: indices n = 1, 2, ..., N
    thetas = np.linspace(0, 2 * np.pi, 1000)
    phi = np.zeros(len(thetas), dtype=complex)
    for k in range(N):
        n = k + 1
        phi += (a_odd[k] + 1j * B_odd[k]) * np.exp(1j * n * thetas)

    re_phi = phi.real
    im_phi = phi.imag

    print(f'  Re(phi) range: [{re_phi.min():+.4f}, {re_phi.max():+.4f}]')
    print(f'  Im(phi) range: [{im_phi.min():+.4f}, {im_phi.max():+.4f}]')
    print(f'  Re(phi) < 0 everywhere: {re_phi.max() < 0}')

    if re_phi.max() < 0:
        print(f'  ** CF SYMBOL IS NEGATIVE ** -> M_odd is negative definite!')
    else:
        print(f'  CF symbol touches positive at {re_phi.max():+.6f}')
        # Where does it go positive?
        pos_idx = np.where(re_phi > 0)[0]
        if len(pos_idx) > 0:
            theta_pos = thetas[pos_idx[0]]
            print(f'  First positive at theta = {theta_pos:.4f} '
                  f'({theta_pos/np.pi:.4f}*pi)')
    sys.stdout.flush()

    # == Part 4: Compare a_odd and B_odd profiles ==
    print('\n  === PART 4: a_n AND B_n PROFILES ===')
    print(f'  At lam^2=1000:')
    print(f'  {"n":>4} {"a_odd[n]":>14} {"B_odd[n]":>14} {"a+iB":>20}')
    print('  ' + '-' * 56)
    for k in range(min(N, 15)):
        n = k + 1
        print(f'  {n:>4d} {a_odd[k]:>+14.6f} {B_odd[k]:>+14.6f} '
              f'{a_odd[k]:>+.4f}{B_odd[k]:>+.4f}i')
    sys.stdout.flush()

    # == Part 5: Lambda sweep of symbol ==
    print('\n  === PART 5: CF SYMBOL vs LAMBDA ===')
    print(f'  {"lam^2":>8} {"max Re(phi)":>14} {"min Re(phi)":>14} '
          f'{"negative?":>10}')
    print('  ' + '-' * 52)

    for lam_sq in [50, 200, 1000, 5000, 20000]:
        a_n_t, B_n_t, _, L_t, N_t, _ = extract_cauchy_coefficients(lam_sq)
        a_o = np.array([a_n_t[N_t + k] for k in range(1, N_t + 1)])
        B_o = np.array([B_n_t[N_t + k] for k in range(1, N_t + 1)])

        phi_t = np.zeros(len(thetas), dtype=complex)
        for k in range(N_t):
            phi_t += (a_o[k] + 1j * B_o[k]) * np.exp(1j * (k + 1) * thetas)
        re_t = phi_t.real
        neg = re_t.max() < 0
        print(f'  {lam_sq:>8d} {re_t.max():>+14.6f} {re_t.min():>+14.6f} '
              f'{"YES" if neg else "NO":>10}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
