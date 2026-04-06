"""
SESSION 58c -- THE EXPLICIT INEQUALITY

The critical direction is v ~ c1|1> + c2|2> on the odd subspace.
At lam^2=1000: c1=-0.54, c2=+0.84.

On this 2-mode vector, the Rayleigh quotients reduce to:
  M_diag(v) = c1^2 * wr_diag[1] + c2^2 * wr_diag[2]
  M_alpha(v) = explicit function of alpha[1], alpha[2], c1, c2
  M_prime(v) = sum over prime powers of simple trig weights

Plan:
  1. Track (c1, c2) across lambda — stable or varying?
  2. Derive M_prime(v) as an explicit prime sum
  3. Derive M_diag(v) as explicit special functions
  4. Express the inequality M_diag + M_alpha > -M_prime
  5. Test: does a FIXED 2-mode vector (c1, c2 independent of lambda)
     also satisfy M < 0? If so, the inequality is about a fixed
     direction, not an eigenvalue problem.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _build_M_prime, _compute_alpha, _compute_wr_diag
)


def odd_block_eigh(M, N):
    """Eigendecompose the odd block."""
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    M_odd = P.T @ M @ P
    e, v = np.linalg.eigh(M_odd)
    return e, v, P


def rayleigh_on_2mode(c1, c2, lam_sq, N=None):
    """
    Compute Rayleigh quotients of M components on the odd 2-mode vector
    v = c1*(|1>-|-1>)/sqrt(2) + c2*(|2>-|-2>)/sqrt(2).

    Returns dict with explicit formulas evaluated.
    """
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))

    # Normalize
    norm = np.sqrt(c1**2 + c2**2)
    c1n, c2n = c1/norm, c2/norm

    # wr_diag
    wr = _compute_wr_diag(L, N)
    rq_diag = c1n**2 * wr[1] + c2n**2 * wr[2]

    # alpha
    alpha = _compute_alpha(L, N)
    # In odd basis, index 0 = n=1, index 1 = n=2
    # alpha[N+n] = alpha_n, alpha[N-n] = -alpha_n
    a1 = alpha[N + 1]  # alpha[1]
    a2 = alpha[N + 2]  # alpha[2]

    # M_alpha on odd block: (alpha_m - alpha_n) / (n - m) for the ODD projections
    # For n=1,2 in odd basis: the odd projection of alpha gives
    # M_alpha_odd[i,j] for i,j in {0,1} (mapping to n=1,2)
    # Actually need to compute this from the full matrix restricted to odd
    # Let me just compute the 2x2 directly
    # M_alpha[n,m] = (alpha[m] - alpha[n]) / (n-m) for n != m in {-N..N}
    # On odd basis |1> and |2>:
    # The odd projection of row n=1: involves entries at n=1 and n=-1
    # This is getting messy. Let me just use the full matrix approach for alpha.

    ns = np.arange(-N, N + 1, dtype=float)
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        Ma = (alpha[None, :] - alpha[:, None]) / nm
    np.fill_diagonal(Ma, 0.0)
    Ma = (Ma + Ma.T) / 2

    # Build 2-mode vector in full space
    dim = 2 * N + 1
    v = np.zeros(dim)
    v[N + 1] = c1n / np.sqrt(2)
    v[N - 1] = -c1n / np.sqrt(2)
    v[N + 2] = c2n / np.sqrt(2)
    v[N - 2] = -c2n / np.sqrt(2)

    rq_alpha = float(v @ Ma @ v)

    # M_prime: explicit sum over prime powers
    # v^T Q(y) v for the kernel Q at log-height y
    # The kernel Q(y) has:
    #   diagonal: 2*(L-y)/L * cos(2*pi*n*y/L)
    #   off-diag: (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))
    # On the 2-mode vector, this becomes:
    #   sum over n,m in {-2,-1,1,2} of v[n]*v[m]*Q[n,m](y)

    primes = sieve_primes(int(lam_sq))
    rq_prime = 0.0
    prime_terms = []  # (p, k, weight, contribution)

    for p in primes:
        pk = int(p)
        logp = np.log(p)
        k = 1
        while pk <= lam_sq:
            weight = logp * pk**(-0.5)
            y = k * logp

            # Compute v^T Q(y) v for our 4-component vector
            # Nonzero components: v[N+1], v[N-1], v[N+2], v[N-2]
            indices = [(N+1, c1n/np.sqrt(2)),
                       (N-1, -c1n/np.sqrt(2)),
                       (N+2, c2n/np.sqrt(2)),
                       (N-2, -c2n/np.sqrt(2))]

            contrib = 0.0
            for i_idx, (i, vi) in enumerate(indices):
                ni = ns[i]
                # Diagonal term
                contrib += vi**2 * 2*(L-y)/L * np.cos(2*np.pi*ni*y/L)
                # Off-diagonal terms
                for j_idx, (j, vj) in enumerate(indices):
                    if i_idx != j_idx:
                        nj = ns[j]
                        si = np.sin(2*np.pi*ni*y/L)  # sin for column
                        sj = np.sin(2*np.pi*nj*y/L)  # sin for row
                        # Q[i,j] = (sin(2*pi*j*y/L) - sin(2*pi*i*y/L)) / (pi*(i-j))
                        # Wait, the convention in the code is:
                        # off_diag[n,m] = (sin_arr[m] - sin_arr[n]) / (pi * (n-m))
                        # where sin_arr[idx] = sin(2*pi*ns[idx]*y/L)
                        sin_i = np.sin(2*np.pi*ns[i]*y/L)
                        sin_j = np.sin(2*np.pi*ns[j]*y/L)
                        if abs(ns[i] - ns[j]) > 1e-10:
                            off = (sin_j - sin_i) / (np.pi * (ns[i] - ns[j]))
                        else:
                            off = 0.0
                        contrib += vi * vj * off

            rq_prime += weight * contrib
            if k == 1 and p <= 20:
                prime_terms.append((int(p), k, weight, weight * contrib))

            pk *= int(p)
            k += 1

    rq_total = rq_prime + rq_diag + rq_alpha

    return dict(
        lam_sq=lam_sq, L=L, N=N, c1=c1n, c2=c2n,
        wr1=float(wr[1]), wr2=float(wr[2]),
        a1=float(a1), a2=float(a2),
        rq_diag=float(rq_diag), rq_alpha=rq_alpha,
        rq_prime=rq_prime, rq_total=rq_total,
        prime_terms=prime_terms,
    )


def run():
    print()
    print('#' * 76)
    print('  SESSION 58c -- THE EXPLICIT INEQUALITY')
    print('#' * 76)

    # == Part 1: Critical eigenvector coefficients across lambda ==
    print('\n  === PART 1: EIGENVECTOR STABILITY ===')
    print(f'  Do (c1, c2) change with lambda?')
    print()
    print(f'  {"lam^2":>8} {"c1 (n=1)":>10} {"c2 (n=2)":>10} '
          f'{"c3 (n=3)":>10} {"c4 (n=4)":>10} {"2-mode %":>10}')
    print('  ' + '-' * 62)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        em, vm, P = odd_block_eigh(M, N)
        v = vm[:, -1]  # max eigenvalue eigenvector
        two_mode_pct = (v[0]**2 + v[1]**2) * 100
        print(f'  {lam_sq:>8d} {v[0]:>+10.6f} {v[1]:>+10.6f} '
              f'{v[2]:>+10.6f} {v[3]:>+10.6f} {two_mode_pct:>9.2f}%')
    sys.stdout.flush()

    # == Part 2: Fixed 2-mode test ==
    print(f'\n  === PART 2: FIXED 2-MODE VECTOR TEST ===')
    print(f'  Use FIXED c1=-0.54, c2=+0.84 (from lam^2=1000).')
    print(f'  Is M < 0 on this fixed direction at all lambda?')
    print()

    c1_fix, c2_fix = -0.54, 0.84

    print(f'  {"lam^2":>8} {"M_prime":>12} {"M_diag":>12} {"M_alpha":>12} '
          f'{"M_total":>14} {"sign":>6}')
    print('  ' + '-' * 72)

    for lam_sq in [10, 50, 200, 1000, 5000, 20000, 50000]:
        r = rayleigh_on_2mode(c1_fix, c2_fix, lam_sq)
        sign = '-' if r['rq_total'] < 0 else '+'
        print(f'  {lam_sq:>8d} {r["rq_prime"]:>+12.6f} {r["rq_diag"]:>+12.6f} '
              f'{r["rq_alpha"]:>+12.6f} {r["rq_total"]:>+14.8e} {sign:>6s}')
    sys.stdout.flush()

    # == Part 3: Explicit formulas ==
    print(f'\n  === PART 3: EXPLICIT FORMULA COMPONENTS ===')
    print(f'  At lam^2=1000:')

    r = rayleigh_on_2mode(c1_fix, c2_fix, 1000)
    print(f'  wr_diag[1] = {r["wr1"]:+.10f}')
    print(f'  wr_diag[2] = {r["wr2"]:+.10f}')
    print(f'  alpha[1]   = {r["a1"]:+.10f}')
    print(f'  alpha[2]   = {r["a2"]:+.10f}')
    print()
    print(f'  M_diag(v)  = c1^2 * wr[1] + c2^2 * wr[2]')
    print(f'             = {r["c1"]:.6f}^2 * {r["wr1"]:.6f} + '
          f'{r["c2"]:.6f}^2 * {r["wr2"]:.6f}')
    print(f'             = {r["c1"]**2 * r["wr1"]:+.6f} + '
          f'{r["c2"]**2 * r["wr2"]:+.6f}')
    print(f'             = {r["rq_diag"]:+.10f}')
    print()
    print(f'  M_prime(v) = sum over primes of trig weights')
    print(f'             = {r["rq_prime"]:+.10f}')
    print()
    print(f'  Top prime contributions:')
    for p, k, w, c in r['prime_terms']:
        print(f'    p={p:>3d}: weight={w:.4f}, contribution={c:+.6f}')
    print()
    print(f'  M_alpha(v) = {r["rq_alpha"]:+.10f}')
    print()
    print(f'  THE INEQUALITY (must hold for M_odd < 0 on this direction):')
    print(f'    M_diag(v) + M_alpha(v) > -M_prime(v)')
    print(f'    {r["rq_diag"]:+.6f} + {r["rq_alpha"]:+.6f} > '
          f'{-r["rq_prime"]:+.6f}')
    print(f'    {r["rq_diag"] + r["rq_alpha"]:+.6f} > '
          f'{-r["rq_prime"]:+.6f}')
    margin = (r['rq_diag'] + r['rq_alpha']) - (-r['rq_prime'])
    print(f'    margin: {margin:+.10e}')

    # == Part 4: How the inequality varies with lambda ==
    print(f'\n  === PART 4: INEQUALITY MARGIN vs LAMBDA ===')
    print(f'  {"lam^2":>8} {"M_diag+alpha":>14} {"-M_prime":>14} '
          f'{"margin":>16} {"relative":>12}')
    print('  ' + '-' * 70)

    for lam_sq in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                   10000, 20000, 50000]:
        r = rayleigh_on_2mode(c1_fix, c2_fix, lam_sq)
        lhs = r['rq_diag'] + r['rq_alpha']
        rhs = -r['rq_prime']
        margin = lhs - rhs
        rel = margin / rhs if rhs > 0 else 0
        print(f'  {lam_sq:>8d} {lhs:>+14.8f} {rhs:>+14.8f} '
              f'{margin:>+16.10e} {rel:>+12.2e}')
    sys.stdout.flush()

    # == Part 5: Simplification of M_prime on 2-mode ==
    print(f'\n  === PART 5: M_PRIME ON 2-MODE VECTOR (SIMPLIFIED) ===')
    print(f'  On v = c1*(|1>-|-1>)/sqrt(2) + c2*(|2>-|-2>)/sqrt(2),')
    print(f'  the M_prime Rayleigh quotient for a prime power at y is:')
    print(f'  R(y) = sum of trig terms involving sin(2*pi*n*y/L) for n=1,2')
    print()

    # The key simplification: since v has only 4 nonzero components
    # (at n = -2, -1, 1, 2), the Rayleigh quotient of the kernel Q(y)
    # involves only a few trig functions.
    # Let theta = 2*pi*y/L. Then:
    #   cos(2*pi*1*y/L) = cos(theta)
    #   cos(2*pi*2*y/L) = cos(2*theta)
    #   sin(2*pi*1*y/L) = sin(theta)
    #   sin(2*pi*2*y/L) = sin(2*theta)

    print(f'  Let theta = 2*pi*y/L, u = y/L.')
    print(f'  Then the diagonal part of Q(y) on v gives:')
    print(f'    2*(1-u)*[c1^2*cos(theta) + c2^2*cos(2*theta)]')
    print()
    print(f'  And the off-diagonal part involves:')
    print(f'    sin(theta), sin(2*theta) cross terms between n=+-1,+-2')
    print()

    # Compute the exact formula
    # v_full has: v[N+1]=c1/sqrt(2), v[N-1]=-c1/sqrt(2),
    #             v[N+2]=c2/sqrt(2), v[N-2]=-c2/sqrt(2)
    # The n-values: n=1,-1,2,-2
    # Q[n,n](y) = 2*(L-y)/L * cos(2*pi*n*y/L)
    # Q[n,m](y) = (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))  for n!=m

    # Diag:
    # v[N+1]^2 * Q[1,1] + v[N-1]^2 * Q[-1,-1] + v[N+2]^2 * Q[2,2] + v[N-2]^2 * Q[-2,-2]
    # = (c1^2/2)*2*(1-u)*cos(theta) + (c1^2/2)*2*(1-u)*cos(-theta) +
    #   (c2^2/2)*2*(1-u)*cos(2*theta) + (c2^2/2)*2*(1-u)*cos(-2*theta)
    # = 2*(1-u)*[c1^2*cos(theta) + c2^2*cos(2*theta)]

    # Off-diag (there are 4*3=12 terms, but many cancel by symmetry):
    # Need to enumerate all pairs from {(N+1,c1/sqrt2), (N-1,-c1/sqrt2),
    #                                    (N+2,c2/sqrt2), (N-2,-c2/sqrt2)}

    # Let me just compute it symbolically for a few representative y values
    # and verify against the matrix computation.
    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    c1n = c1_fix / np.sqrt(c1_fix**2 + c2_fix**2)
    c2n = c2_fix / np.sqrt(c1_fix**2 + c2_fix**2)

    print(f'  Verification: symbolic formula vs matrix at lam^2=1000')
    print(f'  c1={c1n:.6f}, c2={c2n:.6f}')
    print()

    for y in [np.log(2), np.log(3), np.log(5), np.log(7), L/2]:
        theta = 2 * np.pi * y / L
        u = y / L

        # Diagonal contribution
        diag = 2*(1-u)*(c1n**2 * np.cos(theta) + c2n**2 * np.cos(2*theta))

        # Off-diagonal: enumerate all 12 pairs
        components = [(1, c1n/np.sqrt(2)), (-1, -c1n/np.sqrt(2)),
                      (2, c2n/np.sqrt(2)), (-2, -c2n/np.sqrt(2))]
        offdiag = 0.0
        for i, (ni, vi) in enumerate(components):
            for j, (nj, vj) in enumerate(components):
                if i != j:
                    sin_i = np.sin(2*np.pi*ni*y/L)
                    sin_j = np.sin(2*np.pi*nj*y/L)
                    if abs(ni - nj) > 0.5:
                        offdiag += vi * vj * (sin_j - sin_i) / (np.pi * (ni - nj))

        formula_val = diag + offdiag

        # Matrix computation for comparison
        ns_arr = np.arange(-N, N+1, dtype=float)
        dim = 2*N+1
        v_full = np.zeros(dim)
        v_full[N+1] = c1n/np.sqrt(2)
        v_full[N-1] = -c1n/np.sqrt(2)
        v_full[N+2] = c2n/np.sqrt(2)
        v_full[N-2] = -c2n/np.sqrt(2)

        sin_arr = np.sin(2*np.pi*ns_arr*y/L)
        cos_arr = np.cos(2*np.pi*ns_arr*y/L)
        Qy = np.zeros((dim, dim))
        np.fill_diagonal(Qy, 2*(L-y)/L * cos_arr)
        nm_diff = ns_arr[:,None] - ns_arr[None,:]
        sin_diff = sin_arr[None,:] - sin_arr[:,None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off, 0.0)
        Qy += off
        Qy = (Qy + Qy.T) / 2
        matrix_val = float(v_full @ Qy @ v_full)

        print(f'  y=log({np.exp(y):.0f})={y:.4f}: formula={formula_val:+.8f}, '
              f'matrix={matrix_val:+.8f}, diff={abs(formula_val-matrix_val):.2e}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
