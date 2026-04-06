"""
SESSION 62 -- PROVING THE LORENTZIAN PROPERTY

The structural argument:

  a_n^prime = sum w_pk * 2*(L-y)/L * cos(2*pi*n*y/L)

  At n=0: a_0^prime = sum w_pk * 2*(L-y)/L  (ALL POSITIVE, no cosine)
                    ~ 4*lambda by PNT (coherent sum)

  At n>=1: a_n^prime involves cos(2*pi*n*y/L) which oscillates.
           By equidistribution: sum is O(sqrt(sum w_pk^2)) << a_0

The n=0 mode gets coherent addition. All others see destructive
interference. M_diag ~ C - log(n) kills the small residual positives.

The odd block has NO n=0 mode -> no coherent sum -> negative definite.

Plan:
  1. Verify: a_0^prime >> a_n^prime quantitatively
  2. Test: is removing n=0 from even block enough to kill positivity?
  3. Derive: tight bound on a_n^prime for n >= 1
  4. Schur complement: can n=0's dominance be proved to sustain
     exactly 1 positive eigenvalue via a finite-dimensional argument?
  5. The "budget" argument: total positive trace from primes = sum a_n^prime.
     If a_0^prime captures most of this budget, the rest can't form
     a second positive eigenvalue against M_diag's opposition.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _build_M_prime, _compute_alpha, _compute_wr_diag
)


def even_block(M, N):
    dim = 2 * N + 1
    # Even basis: |0>, (|1>+|-1>)/sqrt(2), ..., (|N>+|-N>)/sqrt(2)
    P = np.zeros((dim, N + 1))
    P[N, 0] = 1.0  # |0>
    for n in range(1, N + 1):
        P[N + n, n] = 1.0 / np.sqrt(2)
        P[N - n, n] = 1.0 / np.sqrt(2)
    return P.T @ M @ P, P


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
    print('  SESSION 62 -- PROVING THE LORENTZIAN PROPERTY')
    print('#' * 76)

    # == Part 1: The n=0 coherence ==
    print('\n  === PART 1: n=0 COHERENCE vs n>=1 INTERFERENCE ===')
    print(f'  a_n^prime = sum w_pk * 2*(L-y)/L * cos(2*pi*n*y/L)')
    print(f'  At n=0: no cosine -> coherent sum ~ 4*lambda')
    print(f'  At n>=1: cosine oscillates -> partial cancellation')
    print()

    print(f'  {"lam^2":>8} {"a_0^prime":>12} {"a_1^prime":>12} '
          f'{"a_2^prime":>12} {"ratio 0/1":>10} {"sum a_n>0":>12}')
    print('  ' + '-' * 70)

    for lam_sq in [50, 200, 1000, 5000, 20000, 50000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        ns = np.arange(-N, N + 1, dtype=float)

        # Prime contribution to diagonal
        primes = sieve_primes(int(lam_sq))
        a_prime = np.zeros(2 * N + 1)
        for p in primes:
            pk = int(p)
            logp = np.log(p)
            while pk <= lam_sq:
                w = logp * pk ** (-0.5)
                y = np.log(pk)
                a_prime += w * 2 * (L - y) / L * np.cos(2 * np.pi * ns * y / L)
                pk *= int(p)

        a0 = a_prime[N]  # n=0
        a1 = a_prime[N + 1]  # n=1
        a2 = a_prime[N + 2]  # n=2
        # Count positive prime diagonal entries
        pos_sum = sum(max(0, a_prime[N + n]) for n in range(N + 1))
        ratio = a0 / a1 if abs(a1) > 1e-15 else float('inf')
        print(f'  {lam_sq:>8d} {a0:>+12.4f} {a1:>+12.4f} '
              f'{a2:>+12.4f} {ratio:>10.2f} {pos_sum:>+12.4f}')
    sys.stdout.flush()

    # == Part 2: Remove n=0 from even block ==
    print('\n  === PART 2: EVEN BLOCK WITHOUT n=0 ===')
    print(f'  If we remove the n=0 row/column from M_even,')
    print(f'  does the positive eigenvalue disappear?')
    print()

    for lam_sq in [200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        Me, _ = even_block(M, N)
        Mo, _ = odd_block(M, N)

        # Full even block
        ee = np.linalg.eigvalsh(Me)
        # Even block without n=0 (remove first row and column)
        Me_no0 = Me[1:, 1:]
        ee_no0 = np.linalg.eigvalsh(Me_no0)

        n_pos_full = np.sum(ee > 1e-10)
        n_pos_no0 = np.sum(ee_no0 > 1e-10)

        print(f'  lam^2={lam_sq:>6d}: M_even sig=({n_pos_full},{N+1-n_pos_full}), '
              f'M_even\\n=0 sig=({n_pos_no0},{N-n_pos_no0}), '
              f'max_eig: full={ee[-1]:+.4f}, no_n0={ee_no0[-1]:+.6f}')
    sys.stdout.flush()

    # == Part 3: The positive budget ==
    print('\n  === PART 3: POSITIVE EIGENVALUE BUDGET ===')
    print(f'  Positive eigenvalue of M_even = a_0 + corrections.')
    print(f'  If a_0 accounts for >90% of the positive eigenvalue,')
    print(f'  the Lorentzian property follows from n=0 dominance.')
    print()

    for lam_sq in [200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        Me, Pe = even_block(M, N)
        ee, ve = np.linalg.eigh(Me)

        lambda_max = ee[-1]
        v_max = ve[:, -1]  # positive eigenvector

        # How much of lambda_max comes from the n=0 diagonal?
        # Rayleigh: lambda_max = v^T Me v = sum_ij v_i Me_ij v_j
        # Diagonal contribution from n=0: v_0^2 * Me[0,0]
        diag_0_contrib = v_max[0]**2 * Me[0, 0]
        diag_other = sum(v_max[k]**2 * Me[k, k] for k in range(1, N + 1))
        offdiag = lambda_max - diag_0_contrib - diag_other

        print(f'  lam^2={lam_sq:>6d}: lambda_max={lambda_max:+.4f}')
        print(f'    v_max[0]^2 (n=0 weight): {v_max[0]**2:.6f}')
        print(f'    n=0 diagonal contrib:     {diag_0_contrib:+.4f} '
              f'({100*diag_0_contrib/lambda_max:.1f}%)')
        print(f'    other diagonal contrib:   {diag_other:+.4f} '
              f'({100*diag_other/lambda_max:.1f}%)')
        print(f'    off-diagonal contrib:     {offdiag:+.4f} '
              f'({100*offdiag/lambda_max:.1f}%)')
    sys.stdout.flush()

    # == Part 4: Schur complement argument ==
    print('\n  === PART 4: SCHUR COMPLEMENT ===')
    print(f'  Partition M_even = [a_0, c^T; c, B] where B = M_even[1:,1:].')
    print(f'  Positive eigenvalue count of M_even = that of B plus')
    print(f'  change from the rank-1 update a_0 - c^T B^{-1} c.')
    print()

    for lam_sq in [200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        Me, _ = even_block(M, N)

        a0_entry = Me[0, 0]
        c = Me[0, 1:]  # coupling vector
        B = Me[1:, 1:]  # submatrix without n=0

        eB = np.linalg.eigvalsh(B)
        n_pos_B = np.sum(eB > 1e-10)

        # Schur complement: s = a_0 - c^T B^{-1} c
        # If B is invertible:
        try:
            B_inv_c = np.linalg.solve(B, c)
            schur = a0_entry - float(c @ B_inv_c)
        except np.linalg.LinAlgError:
            schur = float('nan')

        print(f'  lam^2={lam_sq:>6d}:')
        print(f'    a_0 = {a0_entry:+.4f}')
        print(f'    B = M_even\\n=0: sig ({n_pos_B}, {N-n_pos_B}), '
              f'max_eig={eB[-1]:+.6f}')
        print(f'    Schur complement s = a_0 - c^T B^{{-1}} c = {schur:+.6f}')
        print(f'    s > 0: {schur > 0} '
              f'(if B<0 and s>0: M_even has exactly 1 positive eig)')
        if n_pos_B == 0 and schur > 0:
            print(f'    ** B < 0 AND s > 0: LORENTZIAN ON EVEN BLOCK PROVED '
                  f'(at this lambda) **')
    sys.stdout.flush()

    # == Part 5: Same for odd block — all-negative Schur ==
    print('\n  === PART 5: ODD BLOCK — NO n=0, NO POSITIVE ===')
    print(f'  M_odd starts at n=1. Partition: [a_1, c^T; c, B_rest].')
    print()

    for lam_sq in [200, 1000, 5000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        _, M, _ = build_all_fast(lam_sq, N)
        Mo, _ = odd_block(M, N)

        a1_entry = Mo[0, 0]
        c_o = Mo[0, 1:]
        B_o = Mo[1:, 1:]

        eB_o = np.linalg.eigvalsh(B_o)
        n_pos_Bo = np.sum(eB_o > 1e-10)

        try:
            Bo_inv_c = np.linalg.solve(B_o, c_o)
            schur_o = a1_entry - float(c_o @ Bo_inv_c)
        except np.linalg.LinAlgError:
            schur_o = float('nan')

        print(f'  lam^2={lam_sq:>6d}:')
        print(f'    a_1 = {a1_entry:+.4f}')
        print(f'    B_rest: sig ({n_pos_Bo}, {N-1-n_pos_Bo}), '
              f'max_eig={eB_o[-1]:+.6f}')
        print(f'    Schur s = a_1 - c^T B^{{-1}} c = {schur_o:+.6f}')
        print(f'    s <= 0: {schur_o <= 0} '
              f'(if B<0 and s<=0: M_odd negative definite)')
        if n_pos_Bo == 0 and schur_o <= 1e-10:
            print(f'    ** B_rest < 0 AND s <= 0: M_ODD NEG DEF PROVED '
                  f'(at this lambda) **')
    sys.stdout.flush()

    # == Part 6: Recursive Schur — peel off n=0,1,2,... ==
    print('\n  === PART 6: RECURSIVE SCHUR DESCENT ===')
    print(f'  Peel off rows one at a time. Track Schur complement sign.')
    print(f'  At lam^2=1000:')
    print()

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    _, M_full, _ = build_all_fast(lam_sq, N)
    Me, _ = even_block(M_full, N)
    Mo, _ = odd_block(M_full, N)

    print(f'  EVEN BLOCK (peel from n=0):')
    print(f'  {"step":>5} {"entry":>10} {"Schur s":>12} {"B neg def":>10} '
          f'{"conclusion":>20}')
    print('  ' + '-' * 62)

    R = Me.copy()
    for step in range(min(8, N)):
        a_top = R[0, 0]
        if R.shape[0] <= 1:
            print(f'  {step:>5d} {a_top:>+10.4f} {"—":>12} {"—":>10} '
                  f'{"1x1 matrix":>20}')
            break
        c_r = R[0, 1:]
        B_r = R[1:, 1:]
        eB_r = np.linalg.eigvalsh(B_r)
        neg_def_B = eB_r[-1] < 1e-10
        try:
            s_r = a_top - float(c_r @ np.linalg.solve(B_r, c_r))
        except:
            s_r = float('nan')
        if neg_def_B and s_r > 1e-10:
            conc = '1 pos eig from here'
        elif neg_def_B and s_r <= 1e-10:
            conc = 'neg def from here'
        else:
            conc = f'B has {np.sum(eB_r>1e-10)} pos'
        print(f'  {step:>5d} {a_top:>+10.4f} {s_r:>+12.6f} '
              f'{"YES" if neg_def_B else "NO":>10} {conc:>20}')

        # Schur-reduce: new matrix = B - c c^T / a_top (if a_top != 0)
        # Actually for tracking, just remove first row/col
        R = B_r

    print()
    print(f'  ODD BLOCK (peel from n=1):')
    print(f'  {"step":>5} {"entry":>10} {"Schur s":>12} {"B neg def":>10} '
          f'{"conclusion":>20}')
    print('  ' + '-' * 62)

    R = Mo.copy()
    for step in range(min(8, N)):
        a_top = R[0, 0]
        if R.shape[0] <= 1:
            print(f'  {step:>5d} {a_top:>+10.4f} {"—":>12} {"—":>10} '
                  f'{"1x1 matrix":>20}')
            break
        c_r = R[0, 1:]
        B_r = R[1:, 1:]
        eB_r = np.linalg.eigvalsh(B_r)
        neg_def_B = eB_r[-1] < 1e-10
        try:
            s_r = a_top - float(c_r @ np.linalg.solve(B_r, c_r))
        except:
            s_r = float('nan')
        if neg_def_B and s_r > 1e-10:
            conc = '1 pos eig from here'
        elif neg_def_B and s_r <= 1e-10:
            conc = 'neg def from here'
        else:
            conc = f'B has {np.sum(eB_r>1e-10)} pos'
        print(f'  {step:>5d} {a_top:>+10.4f} {s_r:>+12.6f} '
              f'{"YES" if neg_def_B else "NO":>10} {conc:>20}')
        R = B_r

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
