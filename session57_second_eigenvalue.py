"""
SESSION 57 -- WHY IS M's SECOND EIGENVALUE ZERO?

Session 56b found M_total's eigenvalues at lam^2=1000:
  +38.455, ~0, ~0, ~0, ~0, then -24.24, -7.60, ...

The second eigenvalue is essentially zero. This means:
  M_diag's contribution EXACTLY cancels M_prime's second eigenvalue.

If this is structural (not numerical accident), there must be an
identity or symmetry forcing it. Candidates:

  (A) Parity: M decomposes into even/odd blocks. The positive
      eigenvalue is even. The "second eigenvalue" might be the
      TOP eigenvalue of the ODD block, which could be zero by
      a trace/symmetry argument.

  (B) A functional identity: M's eigenvalues might be controlled
      by zeta at specific points, and the second eigenvalue
      vanishes because zeta has a specific value there.

  (C) The W02 structure: W02 has rank 2, with eigenvalues on
      the even and odd directions. M's near-zero eigenvalues
      might correspond to the W02 eigenvalue directions.

Plan:
  1. Decompose M into even/odd blocks. Check eigenvalues separately.
  2. Track the second eigenvalue of M across many lambda values.
     Is it always zero, or does it cross zero at specific lambda?
  3. Identify WHICH eigenvectors have near-zero eigenvalues.
  4. Check if the near-zero eigenvalue is the odd-direction
     eigenvalue (w_hat direction), which would connect to
     the barrier on range(W02).
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast, _build_W02


def parity_decomposition(M, N):
    """
    Decompose a (2N+1) x (2N+1) matrix into even and odd blocks.
    Even basis: |n> + |-n> for n=1..N, plus |0>   (dim N+1)
    Odd basis:  |n> - |-n> for n=1..N              (dim N)
    """
    dim = 2 * N + 1

    # Build transformation matrix P such that P^T M P is block diagonal
    # Even block: indices {0} union {n, -n symmetric combinations}
    # n=0 is index N in the array

    # Even basis vectors (normalized)
    even_vecs = []
    # |0> (already even)
    e0 = np.zeros(dim)
    e0[N] = 1.0
    even_vecs.append(e0)
    # (|n> + |-n>)/sqrt(2) for n=1..N
    for n in range(1, N + 1):
        v = np.zeros(dim)
        v[N + n] = 1.0 / np.sqrt(2)
        v[N - n] = 1.0 / np.sqrt(2)
        even_vecs.append(v)
    P_even = np.column_stack(even_vecs)  # (dim, N+1)

    # Odd basis vectors (normalized)
    odd_vecs = []
    # (|n> - |-n>)/sqrt(2) for n=1..N
    for n in range(1, N + 1):
        v = np.zeros(dim)
        v[N + n] = 1.0 / np.sqrt(2)
        v[N - n] = -1.0 / np.sqrt(2)
        odd_vecs.append(v)
    P_odd = np.column_stack(odd_vecs)  # (dim, N)

    M_even = P_even.T @ M @ P_even  # (N+1, N+1)
    M_odd = P_odd.T @ M @ P_odd     # (N, N)

    return M_even, M_odd, P_even, P_odd


def run():
    print()
    print('#' * 76)
    print('  SESSION 57 -- WHY IS M\'s SECOND EIGENVALUE ZERO?')
    print('#' * 76)

    # == Part 1: Even/odd decomposition ==
    print('\n  === PART 1: EVEN/ODD BLOCK DECOMPOSITION ===')
    print(f'  M should decompose by parity (n -> -n symmetry).')
    print(f'  The positive eigenvalue is EVEN (Session 56b).')
    print(f'  Question: is the near-zero eigenvalue in the EVEN or ODD block?')
    print()

    print(f'  {"lam^2":>8} {"dim":>5} {"even dim":>9} {"odd dim":>8} '
          f'{"M_even top2":>24} {"M_odd top2":>24} {"M_total top2":>24}')
    print('  ' + '-' * 110)

    for lam_sq in [50, 200, 1000, 5000, 20000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        dim = 2 * N + 1

        _, M, QW = build_all_fast(lam_sq, N)
        M_even, M_odd, P_even, P_odd = parity_decomposition(M, N)

        e_even = np.linalg.eigvalsh(M_even)
        e_odd = np.linalg.eigvalsh(M_odd)
        e_total = np.linalg.eigvalsh(M)

        print(f'  {lam_sq:>8d} {dim:>5d} {N+1:>9d} {N:>8d} '
              f'  ({e_even[-1]:+.4f},{e_even[-2]:+.4f}) '
              f'  ({e_odd[-1]:+.4f},{e_odd[-2]:+.4f}) '
              f'  ({e_total[-1]:+.4f},{e_total[-2]:+.4f})')
    sys.stdout.flush()

    # == Part 2: Detailed eigenvalue spectrum of each block ==
    print('\n  === PART 2: EIGENVALUE SPECTRA OF BLOCKS (lam^2=1000) ===')
    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    dim = 2 * N + 1

    W02, M, QW = build_all_fast(lam_sq, N)
    M_even, M_odd, P_even, P_odd = parity_decomposition(M, N)

    e_even = np.linalg.eigvalsh(M_even)
    e_odd = np.linalg.eigvalsh(M_odd)

    print(f'\n  EVEN block ({N+1} x {N+1}):')
    print(f'    top 5:    {e_even[-1]:+.6f} {e_even[-2]:+.6f} '
          f'{e_even[-3]:+.6f} {e_even[-4]:+.6f} {e_even[-5]:+.6f}')
    print(f'    bottom 3: {e_even[0]:+.6f} {e_even[1]:+.6f} {e_even[2]:+.6f}')
    n_pos_even = np.sum(e_even > 1e-10)
    print(f'    signature: ({n_pos_even}, {np.sum(e_even < -1e-10)}, '
          f'{np.sum(np.abs(e_even) <= 1e-10)})')

    print(f'\n  ODD block ({N} x {N}):')
    print(f'    top 5:    {e_odd[-1]:+.6f} {e_odd[-2]:+.6f} '
          f'{e_odd[-3]:+.6f} {e_odd[-4]:+.6f} {e_odd[-5]:+.6f}')
    print(f'    bottom 3: {e_odd[0]:+.6f} {e_odd[1]:+.6f} {e_odd[2]:+.6f}')
    n_pos_odd = np.sum(e_odd > 1e-10)
    print(f'    signature: ({n_pos_odd}, {np.sum(e_odd < -1e-10)}, '
          f'{np.sum(np.abs(e_odd) <= 1e-10)})')

    # == Part 3: Q_W even/odd blocks ==
    print(f'\n  === PART 3: Q_W EVEN/ODD BLOCKS ===')
    QW_even, QW_odd, _, _ = parity_decomposition(QW, N)
    eq_even = np.linalg.eigvalsh(QW_even)
    eq_odd = np.linalg.eigvalsh(QW_odd)

    print(f'\n  Q_W EVEN block:')
    print(f'    min eig: {eq_even[0]:+.6e}')
    print(f'    is PSD: {eq_even[0] > -1e-10}')

    print(f'\n  Q_W ODD block:')
    print(f'    min eig: {eq_odd[0]:+.6e}')
    print(f'    is PSD: {eq_odd[0] > -1e-10}')
    sys.stdout.flush()

    # == Part 4: Track second eigenvalue across lambda ==
    print(f'\n  === PART 4: SECOND EIGENVALUE OF M vs LAMBDA ===')
    print(f'  Track top eigenvalues of M_even and M_odd separately.')
    print()

    lam_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                  10000, 20000, 50000]

    print(f'  {"lam^2":>8} {"L":>6} {"M_even top":>12} {"M_even 2nd":>12} '
          f'{"M_odd top":>12} {"M_odd 2nd":>12} {"gap(odd)":>10}')
    print('  ' + '-' * 80)

    scan_data = []
    for lam_sq in lam_values:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        _, M, QW = build_all_fast(lam_sq, N)
        Me, Mo, _, _ = parity_decomposition(M, N)
        ee = np.linalg.eigvalsh(Me)
        eo = np.linalg.eigvalsh(Mo)

        # Also get Q_W blocks
        Qe, Qo, _, _ = parity_decomposition(QW, N)
        eqe = np.linalg.eigvalsh(Qe)
        eqo = np.linalg.eigvalsh(Qo)

        scan_data.append(dict(
            lam_sq=lam_sq, L=L, N=N,
            even_top=ee[-1], even_2nd=ee[-2],
            odd_top=eo[-1], odd_2nd=eo[-2],
            qw_even_min=eqe[0], qw_odd_min=eqo[0],
        ))
        # "gap(odd)" = how far the odd block top eigenvalue is from zero
        print(f'  {lam_sq:>8d} {L:>6.2f} {ee[-1]:>+12.4f} {ee[-2]:>+12.4f} '
              f'{eo[-1]:>+12.4f} {eo[-2]:>+12.4f} {eo[-1]:>+10.6f}')
    sys.stdout.flush()

    # == Part 5: Connection to barrier ==
    print(f'\n  === PART 5: CONNECTION TO BARRIER ===')
    print(f'  The "barrier" on range(W02) = Q_W restricted to the')
    print(f'  2D subspace spanned by the even and odd range vectors.')
    print(f'  Since parity decouples: even barrier + odd barrier.')
    print()
    print(f'  {"lam^2":>8} {"QW_even min":>14} {"QW_odd min":>14} '
          f'{"M_even sig":>12} {"M_odd sig":>12}')
    print('  ' + '-' * 66)

    for d in scan_data:
        lam_sq = d['lam_sq']
        L = d['L']
        N = d['N']
        _, M, QW = build_all_fast(lam_sq, N)
        Me, Mo, _, _ = parity_decomposition(M, N)
        Qe, Qo, _, _ = parity_decomposition(QW, N)
        ee = np.linalg.eigvalsh(Me)
        eo = np.linalg.eigvalsh(Mo)
        eqe = np.linalg.eigvalsh(Qe)
        eqo = np.linalg.eigvalsh(Qo)

        n_pos_e = np.sum(ee > 1e-10)
        n_neg_e = np.sum(ee < -1e-10)
        n_pos_o = np.sum(eo > 1e-10)
        n_neg_o = np.sum(eo < -1e-10)

        print(f'  {lam_sq:>8d} {eqe[0]:>+14.6e} {eqo[0]:>+14.6e} '
              f'  ({n_pos_e},{n_neg_e}) '
              f'  ({n_pos_o},{n_neg_o})')

    # == Part 6: Is M_odd negative definite? ==
    print(f'\n  === PART 6: IS M_ODD NEGATIVE DEFINITE? ===')
    print(f'  If M_even has signature (1, N) and M_odd is negative definite,')
    print(f'  then M_total has signature (1, 2N) = (1, d-1). Done.')
    print()

    all_odd_negdef = True
    for d in scan_data:
        is_nd = d['odd_top'] < 1e-10
        if not is_nd:
            all_odd_negdef = False

    if all_odd_negdef:
        print(f'  M_odd is NEGATIVE DEFINITE at all tested lambda!')
    else:
        print(f'  M_odd is NOT always negative definite.')
        print(f'  Top eigenvalue of M_odd:')
        for d in scan_data:
            if d['odd_top'] > 1e-10:
                print(f'    lam^2={d["lam_sq"]}: {d["odd_top"]:+.6e}')

    all_even_1pos = True
    for d in scan_data:
        lam_sq = d['lam_sq']
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        Me, _, _, _ = parity_decomposition(M, N)
        ee = np.linalg.eigvalsh(Me)
        n_pos = np.sum(ee > 1e-10)
        if n_pos != 1:
            all_even_1pos = False
            print(f'  M_even has {n_pos} positive eigenvalues at lam^2={lam_sq}')

    if all_even_1pos:
        print(f'  M_even has exactly 1 positive eigenvalue at all tested lambda!')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)

    if all_odd_negdef and all_even_1pos:
        print(f'\n  CLEAN DECOMPOSITION:')
        print(f'    M_even: signature (1, N)   -- Lorentzian on even block')
        print(f'    M_odd:  signature (0, N)   -- NEGATIVE DEFINITE')
        print(f'    M_total: signature (1, 2N) -- Lorentzian')
        print()
        print(f'  The "second eigenvalue = 0" from Session 56 was WRONG.')
        print(f'  It was an artifact of looking at M_total without parity')
        print(f'  separation. The actual structure is:')
        print(f'    - M_odd is negative definite (cleanly, no near-zero)')
        print(f'    - M_even has 1 positive eigenvalue (cleanly)')
        print(f'    - Combined: M has 1 positive eigenvalue')
        print()
        print(f'  PROOF PATH: Show M_odd < 0 and M_even has at most 1')
        print(f'  positive eigenvalue. These are INDEPENDENT problems')
        print(f'  on blocks of half the dimension.')
        print()
        print(f'  M_odd < 0 is equivalent to:')
        print(f'    Q_W > 0 restricted to the ODD subspace')
        print(f'    (since W02 has exactly 1 nonzero eigenvalue on odd,')
        print(f'     and that eigenvalue is NEGATIVE)')
        print()
        print(f'  M_even Lorentzian is equivalent to:')
        print(f'    Q_W > 0 restricted to the EVEN null(W02) subspace')
        print(f'    (the positive eigenvalue is in even range(W02))')
    elif all_odd_negdef:
        print(f'\n  M_odd is negative definite but M_even has multiple')
        print(f'  positive eigenvalues at some lambda. The Lorentzian')
        print(f'  property lives in the EVEN block.')
    else:
        print(f'\n  M_odd is NOT always negative definite.')
        print(f'  The problem is on the ODD block.')

    np.savez('session57_second_eigenvalue.npz',
             scan_data=scan_data)
    print(f'\n  Data saved to session57_second_eigenvalue.npz')


if __name__ == '__main__':
    run()
