"""
SESSION 58 -- WHY IS M_ODD NEGATIVE DEFINITE?

Session 57 found M_odd < 0 (negative definite) at all tested lambda.
This is half of the Lorentzian reduction of RH.

M_odd = M_prime_odd + M_diag_odd + M_alpha_odd

Questions:
  1. Which component drives the negativity?
  2. Is M_diag_odd alone negative definite? (It's diagonal with
     entries wr_diag[n] for n=1..N. Some wr_diag[n] > 0 for small n.)
  3. How robust is the negativity? (Ratio of max eigenvalue to trace.)
  4. Is there a provable bound on the max eigenvalue of M_odd?

Plan:
  A. Decompose M_odd into its three components at several lambda.
  B. Check eigenvalue spectra of each component.
  C. Track max(eigenvalue of M_odd) / trace(M_odd) as lambda grows.
  D. Look for an ANALYTIC bound: e.g., Gershgorin, trace/dim, or
     operator-norm domination by M_diag_odd.
  E. Test: does M_prime_odd have a DEFINITE sign?
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _build_W02, _compute_alpha, _compute_wr_diag,
    _build_M_prime
)


def odd_block(M, N):
    """Extract the N x N odd-parity block of a (2N+1) x (2N+1) matrix."""
    dim = 2 * N + 1
    # Odd basis: (|n> - |-n>)/sqrt(2) for n=1..N
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def build_components(lam_sq):
    """Build M's three components and their odd blocks."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1

    Mp = _build_M_prime(L, N, lam_sq)
    Mp = (Mp + Mp.T) / 2

    wr = _compute_wr_diag(L, N)
    ns = np.arange(-N, N + 1)
    Md = np.diag([wr[abs(n)] for n in ns])

    alpha = _compute_alpha(L, N)
    ns_f = ns.astype(float)
    nm = ns_f[:, None] - ns_f[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        Ma = (alpha[None, :] - alpha[:, None]) / nm
    np.fill_diagonal(Ma, 0.0)
    Ma = (Ma + Ma.T) / 2

    Mt = Mp + Md + Ma

    # Odd blocks
    Mp_o = odd_block(Mp, N)
    Md_o = odd_block(Md, N)
    Ma_o = odd_block(Ma, N)
    Mt_o = odd_block(Mt, N)

    return dict(
        lam_sq=lam_sq, L=L, N=N, dim=dim,
        Mp=Mp, Md=Md, Ma=Ma, Mt=Mt,
        Mp_o=Mp_o, Md_o=Md_o, Ma_o=Ma_o, Mt_o=Mt_o,
        wr=wr, alpha=alpha
    )


def sig(A, tol=1e-10):
    e = np.linalg.eigvalsh(A)
    return (int(np.sum(e > tol)), int(np.sum(e < -tol)),
            int(np.sum(np.abs(e) <= tol)))


def run():
    print()
    print('#' * 76)
    print('  SESSION 58 -- WHY IS M_ODD NEGATIVE DEFINITE?')
    print('#' * 76)

    # == Part A: Component signatures on odd block ==
    print('\n  === PART A: ODD BLOCK COMPONENT SIGNATURES ===')
    print(f'  {"lam^2":>8} {"N":>4} {"Mp_odd":>14} {"Md_odd":>14} '
          f'{"Ma_odd":>14} {"Md+Ma_odd":>14} {"Mt_odd":>14}')
    print('  ' + '-' * 80)

    for lam_sq in [50, 200, 1000, 5000, 20000, 50000]:
        c = build_components(lam_sq)
        N = c['N']
        sp = sig(c['Mp_o'])
        sd = sig(c['Md_o'])
        sa = sig(c['Ma_o'])
        sda = sig(c['Md_o'] + c['Ma_o'])
        st = sig(c['Mt_o'])
        print(f'  {lam_sq:>8d} {N:>4d} '
              f'({sp[0]:>2d},{sp[1]:>2d},{sp[2]:>2d}) '
              f'({sd[0]:>2d},{sd[1]:>2d},{sd[2]:>2d}) '
              f'({sa[0]:>2d},{sa[1]:>2d},{sa[2]:>2d}) '
              f'({sda[0]:>2d},{sda[1]:>2d},{sda[2]:>2d}) '
              f'({st[0]:>2d},{st[1]:>2d},{st[2]:>2d})')
    sys.stdout.flush()

    # == Part B: Eigenvalue spectra at lam^2=1000 ==
    print('\n  === PART B: ODD BLOCK EIGENVALUE SPECTRA (lam^2=1000) ===')
    c = build_components(1000)
    N = c['N']

    for name, mat in [('M_prime_odd', c['Mp_o']),
                      ('M_diag_odd', c['Md_o']),
                      ('M_alpha_odd', c['Ma_o']),
                      ('M_diag+alpha_odd', c['Md_o'] + c['Ma_o']),
                      ('M_total_odd', c['Mt_o'])]:
        e = np.linalg.eigvalsh(mat)
        print(f'\n  {name} ({N}x{N}):')
        print(f'    top 3:    {e[-1]:+.6f} {e[-2]:+.6f} {e[-3]:+.6f}')
        print(f'    bottom 3: {e[0]:+.6f} {e[1]:+.6f} {e[2]:+.6f}')
        print(f'    trace:    {np.trace(mat):+.6f}')
        print(f'    max/trace: {e[-1]/np.trace(mat):.6f}' if np.trace(mat) != 0 else '')
    sys.stdout.flush()

    # == Part C: Robustness — max eigenvalue vs trace ==
    print('\n  === PART C: ROBUSTNESS OF M_ODD NEGATIVE DEFINITENESS ===')
    print(f'  If max_eig / trace << 1/N, the negativity is robust.')
    print()
    print(f'  {"lam^2":>8} {"N":>4} {"max_eig(Mt_o)":>16} {"trace(Mt_o)":>14} '
          f'{"max/trace":>10} {"1/N":>8} {"robust":>8}')
    print('  ' + '-' * 76)

    for lam_sq in [10, 50, 200, 1000, 5000, 20000, 50000]:
        c = build_components(lam_sq)
        N = c['N']
        e = np.linalg.eigvalsh(c['Mt_o'])
        tr = np.trace(c['Mt_o'])
        ratio = e[-1] / tr if tr != 0 else 0
        inv_N = 1.0 / N
        robust = abs(ratio) < inv_N
        print(f'  {lam_sq:>8d} {N:>4d} {e[-1]:>+16.8f} {tr:>+14.4f} '
              f'{ratio:>10.6f} {inv_N:>8.4f} {"YES" if robust else "NO"}')
    sys.stdout.flush()

    # == Part D: Diagonal analysis of M_diag_odd ==
    print('\n  === PART D: DIAGONAL ENTRIES wr_diag[n] ON ODD BLOCK ===')
    print(f'  M_diag restricted to odd subspace is diagonal with entries wr_diag[n].')
    print(f'  At lam^2=1000:')
    c = build_components(1000)
    wr = c['wr']
    N = c['N']
    L = c['L']
    print(f'    N = {N}, L = {L:.4f}')
    print(f'    {"n":>4} {"wr_diag[n]":>14} {"sign":>6}')
    print(f'    ' + '-' * 28)
    n_pos_wr = 0
    for n in range(1, min(N + 1, 25)):
        sign = '+' if wr[n] > 0 else '-'
        if wr[n] > 0:
            n_pos_wr += 1
        print(f'    {n:>4d} {wr[n]:>+14.6f}  {sign}')
    if N > 24:
        # Count how many are positive total
        n_pos_wr = sum(1 for n in range(1, N + 1) if wr[n] > 0)
        print(f'    ... ({N - 24} more entries)')
    print(f'    Total positive: {n_pos_wr} out of {N}')
    print(f'    Total negative: {N - n_pos_wr} out of {N}')
    print(f'    trace(M_diag_odd) = {sum(wr[n] for n in range(1, N+1)):+.4f}')

    # == Part E: Is M_prime_odd negative definite? ==
    print('\n  === PART E: M_PRIME ON ODD BLOCK ===')
    print(f'  Does M_prime_odd have definite sign?')
    print()

    for lam_sq in [50, 200, 1000, 5000, 20000]:
        c = build_components(lam_sq)
        N = c['N']
        e_p = np.linalg.eigvalsh(c['Mp_o'])
        e_da = np.linalg.eigvalsh(c['Md_o'] + c['Ma_o'])
        print(f'  lam^2={lam_sq:>6d}: M_prime_odd max={e_p[-1]:+.4f}, '
              f'min={e_p[0]:+.4f}  |  '
              f'M_diag+alpha_odd max={e_da[-1]:+.4f}, min={e_da[0]:+.4f}')
    sys.stdout.flush()

    # == Part F: Gershgorin bound on M_odd ==
    print('\n  === PART F: GERSHGORIN BOUND ON M_ODD ===')
    print(f'  Gershgorin: every eigenvalue lies in union of discs')
    print(f'  [M_ii - R_i, M_ii + R_i] where R_i = sum_j|!=i |M_ij|.')
    print(f'  If max(M_ii + R_i) < 0, M is negative definite.')
    print()

    for lam_sq in [200, 1000, 5000, 20000]:
        c = build_components(lam_sq)
        Mt_o = c['Mt_o']
        N = c['N']
        diag = np.diag(Mt_o)
        offdiag_sum = np.sum(np.abs(Mt_o), axis=1) - np.abs(diag)
        gershgorin_max = np.max(diag + offdiag_sum)
        gershgorin_min_diag = np.min(diag)
        max_offdiag = np.max(offdiag_sum)

        print(f'  lam^2={lam_sq:>6d}: diag range=[{np.min(diag):+.4f}, {np.max(diag):+.4f}], '
              f'max|offdiag_row|={max_offdiag:.4f}, '
              f'Gershgorin max={gershgorin_max:+.4f}')
    sys.stdout.flush()

    # == Part G: Trace bound (weaker but potentially provable) ==
    print('\n  === PART G: TRACE ANALYSIS ===')
    print(f'  trace(M_odd) = trace(M_prime_odd) + trace(M_diag_odd) + trace(M_alpha_odd)')
    print(f'  If we can bound max_eig(M_odd) <= trace(M_odd)/N + correction,')
    print(f'  then trace < 0 gives negativity.')
    print()

    print(f'  {"lam^2":>8} {"tr(Mp_o)":>12} {"tr(Md_o)":>12} {"tr(Ma_o)":>12} '
          f'{"tr(Mt_o)":>12} {"max_eig":>12} {"tr/N":>10}')
    print('  ' + '-' * 78)

    for lam_sq in [50, 200, 1000, 5000, 20000, 50000]:
        c = build_components(lam_sq)
        N = c['N']
        tr_p = np.trace(c['Mp_o'])
        tr_d = np.trace(c['Md_o'])
        tr_a = np.trace(c['Ma_o'])
        tr_t = np.trace(c['Mt_o'])
        e = np.linalg.eigvalsh(c['Mt_o'])
        print(f'  {lam_sq:>8d} {tr_p:>+12.4f} {tr_d:>+12.4f} {tr_a:>+12.4f} '
              f'{tr_t:>+12.4f} {e[-1]:>+12.8f} {tr_t/N:>+10.4f}')
    sys.stdout.flush()

    # == Part H: Asymptotic scaling ==
    print('\n  === PART H: ASYMPTOTIC SCALING ===')
    print(f'  How do traces and max eigenvalue scale with L?')
    print()

    Ls = []
    tr_tots = []
    max_eigs = []
    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        c = build_components(lam_sq)
        N = c['N']
        tr_t = np.trace(c['Mt_o'])
        e = np.linalg.eigvalsh(c['Mt_o'])
        Ls.append(c['L'])
        tr_tots.append(tr_t)
        max_eigs.append(e[-1])

    Ls = np.array(Ls)
    tr_tots = np.array(tr_tots)
    max_eigs = np.array(max_eigs)

    # Fit trace ~ a * L^b
    log_tr = np.log(-np.array(tr_tots))
    log_L = np.log(Ls)
    b_tr, log_a_tr = np.polyfit(log_L, log_tr, 1)
    print(f'  |trace(M_odd)| ~ {np.exp(log_a_tr):.2f} * L^{b_tr:.3f}')

    # Fit max_eig ~ a * L^b (if all negative)
    if np.all(np.array(max_eigs) < 0):
        log_me = np.log(-np.array(max_eigs))
        b_me, log_a_me = np.polyfit(log_L, log_me, 1)
        print(f'  |max_eig(M_odd)| ~ {np.exp(log_a_me):.4f} * L^{b_me:.3f}')
        print(f'  trace grows as L^{b_tr:.2f}, max_eig grows as L^{b_me:.2f}')
        if b_tr > b_me:
            print(f'  => trace grows FASTER than max_eig: negativity strengthens')
        else:
            print(f'  => max_eig grows faster: check if it stays bounded')
    else:
        print(f'  max_eig not always negative: {max_eigs}')

    # Ratio |max_eig|/|trace|
    ratios = np.abs(max_eigs) / np.abs(tr_tots)
    print(f'  |max_eig|/|trace| range: [{ratios.min():.6f}, {ratios.max():.6f}]')
    b_r, log_a_r = np.polyfit(log_L, np.log(ratios), 1)
    print(f'  ratio ~ {np.exp(log_a_r):.4f} * L^{b_r:.3f}')
    if b_r < -0.1:
        print(f'  => ratio DECREASING with L: negativity becomes more robust')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
