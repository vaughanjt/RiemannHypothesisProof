"""
SESSION 56b -- WHY DOES M HAVE LORENTZIAN SIGNATURE?

M = M_prime + M_diag + M_alpha, where:
  M_prime: sum over prime powers of rank-1-ish outer products
  M_diag:  diagonal matrix from wr_diag[|n|]
  M_alpha: off-diagonal from alpha[n] differences

Session 34 found M = (neg semidef) + rank-1 positive.
Session 56 confirmed signature (1, d-1) at all tested lambda.

Question: WHICH piece contributes the single positive eigenvalue?

Hypothesis 1: M_diag is negative definite, M_alpha is small,
and M_prime has exactly 1 positive eigenvalue (from the coherent
sum over primes).

Hypothesis 2: M_diag + M_alpha together are negative definite,
and M_prime is what lifts exactly one eigenvalue above zero.

Hypothesis 3: Each piece has multiple positive eigenvalues,
but the combination conspires to leave exactly 1. (Unlikely
if true -- would suggest fragile cancellation, not structure.)

Plan:
  1. Compute eigenvalue signatures of M_prime, M_diag, M_alpha separately
  2. Check M_diag + M_alpha signature
  3. Track how M_prime's signature depends on the NUMBER of primes
     (add primes one at a time and watch the signature evolve)
  4. Identify the structural reason for exactly 1 positive eigenvalue
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    _build_W02, _compute_alpha, _compute_wr_diag, _build_M_prime
)


def decompose_M(lam_sq):
    """Build M's three components separately."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1

    # M_prime (prime contributions only)
    M_prime = _build_M_prime(L, N, lam_sq)
    M_prime = (M_prime + M_prime.T) / 2

    # M_diag (wr_diag on diagonal)
    wr = _compute_wr_diag(L, N)  # shape (N+1,) for nv=0..N
    ns = np.arange(-N, N + 1)
    diag_vals = np.array([wr[abs(n)] for n in ns])
    M_diag = np.diag(diag_vals)

    # M_alpha (off-diagonal from alpha differences)
    alpha = _compute_alpha(L, N)  # shape (2N+1,)
    ns_f = ns.astype(float)
    a_m = alpha[None, :]
    a_n = alpha[:, None]
    nm = ns_f[:, None] - ns_f[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        M_alpha = (a_m - a_n) / nm
    np.fill_diagonal(M_alpha, 0.0)
    M_alpha = (M_alpha + M_alpha.T) / 2

    M_total = M_prime + M_diag + M_alpha

    return M_prime, M_diag, M_alpha, M_total, L, N, dim


def signature(A, tol=1e-10):
    """Return (n_pos, n_neg, n_zero) eigenvalue counts."""
    e = np.linalg.eigvalsh(A)
    return (int(np.sum(e > tol)), int(np.sum(e < -tol)),
            int(np.sum(np.abs(e) <= tol)))


def run():
    print()
    print('#' * 76)
    print('  SESSION 56b -- WHY DOES M HAVE LORENTZIAN SIGNATURE?')
    print('#' * 76)

    # == Part 1: Decompose M at several lambda ==
    print('\n  === PART 1: SIGNATURES OF M COMPONENTS ===')
    print(f'  {"lam^2":>8} {"dim":>5} {"M_prime":>14} {"M_diag":>14} '
          f'{"M_alpha":>14} {"diag+alpha":>14} {"M_total":>14}')
    print('  ' + '-' * 85)

    for lam_sq in [50, 200, 1000, 5000, 20000]:
        Mp, Md, Ma, Mt, L, N, dim = decompose_M(lam_sq)
        sp = signature(Mp)
        sd = signature(Md)
        sa = signature(Ma)
        sda = signature(Md + Ma)
        st = signature(Mt)
        print(f'  {lam_sq:>8d} {dim:>5d} '
              f'({sp[0]:>2d},{sp[1]:>2d},{sp[2]:>2d}) '
              f'({sd[0]:>2d},{sd[1]:>2d},{sd[2]:>2d}) '
              f'({sa[0]:>2d},{sa[1]:>2d},{sa[2]:>2d}) '
              f'({sda[0]:>2d},{sda[1]:>2d},{sda[2]:>2d}) '
              f'({st[0]:>2d},{st[1]:>2d},{st[2]:>2d})')
    sys.stdout.flush()

    # == Part 2: Deep dive at lam^2 = 1000 ==
    print('\n  === PART 2: EIGENVALUE SPECTRA AT lam^2 = 1000 ===')
    Mp, Md, Ma, Mt, L, N, dim = decompose_M(1000)

    for name, mat in [('M_prime', Mp), ('M_diag', Md), ('M_alpha', Ma),
                      ('M_diag+M_alpha', Md+Ma), ('M_total', Mt)]:
        e = np.linalg.eigvalsh(mat)
        print(f'\n  {name}: dim={dim}')
        print(f'    top 5:    {e[-1]:+.6f} {e[-2]:+.6f} {e[-3]:+.6f} '
              f'{e[-4]:+.6f} {e[-5]:+.6f}')
        print(f'    bottom 5: {e[0]:+.6f} {e[1]:+.6f} {e[2]:+.6f} '
              f'{e[3]:+.6f} {e[4]:+.6f}')
        print(f'    trace: {np.trace(mat):+.6f}  '
              f'sum(eig): {e.sum():+.6f}')

    # == Part 3: Add primes one at a time ==
    print('\n  === PART 3: PRIME-BY-PRIME SIGNATURE EVOLUTION ===')
    print(f'  At lam^2 = 1000, add primes one at a time to M_prime')
    print(f'  and track the number of positive eigenvalues.')
    print()

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    nm_diff = ns[:, None] - ns[None, :]

    primes = sieve_primes(lam_sq)
    M_accum = np.zeros((dim, dim))

    print(f'  {"primes":>8} {"last_p":>8} {"n_pos":>6} {"max_eig":>14} '
          f'{"2nd_eig":>14} {"ratio":>10}')
    print('  ' + '-' * 68)

    checkpoints = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, len(primes)]

    for i, p in enumerate(primes):
        # Add prime p (and its powers) to M_accum
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            weight = logp * pk ** (-0.5)
            yk = np.log(pk)
            sin_arr = np.sin(2 * np.pi * ns * yk / L)
            cos_arr = np.cos(2 * np.pi * ns * yk / L)
            diag = 2 * (L - yk) / L * cos_arr
            np.fill_diagonal(M_accum, M_accum.diagonal() + weight * diag)
            sin_diff = sin_arr[None, :] - sin_arr[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                off = sin_diff / (np.pi * nm_diff)
            np.fill_diagonal(off, 0.0)
            M_accum += weight * off
            pk *= int(p)

        if (i + 1) in checkpoints:
            M_sym = (M_accum + M_accum.T) / 2
            e = np.linalg.eigvalsh(M_sym)
            n_pos = int(np.sum(e > 1e-10))
            max_e = e[-1]
            second_e = e[-2]
            ratio = max_e / abs(second_e) if abs(second_e) > 1e-15 else float('inf')
            print(f'  {i+1:>8d} {int(p):>8d} {n_pos:>6d} {max_e:>+14.6f} '
                  f'{second_e:>+14.6f} {ratio:>10.1f}')

    sys.stdout.flush()

    # == Part 4: Is the positive eigenvalue from the "DC component"? ==
    print('\n  === PART 4: STRUCTURE OF THE POSITIVE EIGENVECTOR ===')
    print(f'  At lam^2 = 1000: what is M\'s positive eigenvector?')

    e_mt, v_mt = np.linalg.eigh(Mt)
    v_pos = v_mt[:, -1]  # eigenvector for largest eigenvalue

    # Check if it's the "constant mode" (all same sign)
    print(f'  Positive eigenvector components:')
    print(f'    norm: {np.linalg.norm(v_pos):.6f}')
    print(f'    min:  {v_pos.min():+.6f}  max: {v_pos.max():+.6f}')
    print(f'    mean: {v_pos.mean():+.6f}  std: {v_pos.std():.6f}')

    # Check parity: is it even or odd in n?
    ns_int = np.arange(-N, N + 1)
    v_even = np.array([(v_pos[N+n] + v_pos[N-n])/2 for n in range(N+1)])
    v_odd = np.array([(v_pos[N+n] - v_pos[N-n])/2 for n in range(N+1)])
    even_norm = np.linalg.norm(v_even)
    odd_norm = np.linalg.norm(v_odd)
    print(f'    even component norm: {even_norm:.6f}')
    print(f'    odd component norm:  {odd_norm:.6f}')
    if even_norm > 10 * odd_norm:
        print(f'    => EVEN (symmetric in n)')
    elif odd_norm > 10 * even_norm:
        print(f'    => ODD (antisymmetric in n)')
    else:
        print(f'    => MIXED parity')

    # Compare to W02's range vectors
    W02 = _build_W02(L, N)
    ew, vw = np.linalg.eigh(W02)
    # The two range vectors (largest |eigenvalue|)
    idx_sorted = np.argsort(np.abs(ew))
    v_w1 = vw[:, idx_sorted[-1]]
    v_w2 = vw[:, idx_sorted[-2]]
    overlap1 = abs(float(v_pos @ v_w1))
    overlap2 = abs(float(v_pos @ v_w2))
    print(f'    overlap with W02 range vec 1: {overlap1:.6f} '
          f'(eigenvalue {ew[idx_sorted[-1]]:+.4f})')
    print(f'    overlap with W02 range vec 2: {overlap2:.6f} '
          f'(eigenvalue {ew[idx_sorted[-2]]:+.4f})')

    # Check overlap with the "w_hat" test vector (odd direction)
    w = ns / (L**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    overlap_what = abs(float(v_pos @ w_hat))
    print(f'    overlap with w_hat (odd test): {overlap_what:.6f}')

    # Even test vector (u direction from Session 40)
    u = 1.0 / (L**2 + (4*np.pi)**2 * ns**2)
    u_hat = u / np.linalg.norm(u)
    overlap_uhat = abs(float(v_pos @ u_hat))
    print(f'    overlap with u_hat (even test): {overlap_uhat:.6f}')

    # == Part 5: Why exactly 1 positive eigenvalue? ==
    print('\n  === PART 5: STRUCTURAL EXPLANATION ===')

    # M_prime = sum_pk weight_pk * Q(y_pk)
    # where Q(y) is a rank-deficient matrix for each y.
    # If each Q(y) has signature (1, d-1, 0) or (1, d-2, 1),
    # then M_prime is a sum of "almost rank-1" matrices,
    # and the sum might preserve the Lorentzian property.

    # Check signature of individual prime contributions
    print(f'  Signatures of individual prime contributions at lam^2=1000:')
    print(f'  {"prime":>6} {"weight":>10} {"sig(Q_p)":>14} {"max_eig":>12}')
    print('  ' + '-' * 50)

    for p in [2, 3, 5, 7, 11, 23, 97, 167]:
        if p > lam_sq:
            continue
        Qp = np.zeros((dim, dim))
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            weight = logp * pk ** (-0.5)
            yk = np.log(pk)
            sin_arr = np.sin(2 * np.pi * ns * yk / L)
            cos_arr = np.cos(2 * np.pi * ns * yk / L)
            diag = 2 * (L - yk) / L * cos_arr
            np.fill_diagonal(Qp, Qp.diagonal() + weight * diag)
            sin_diff = sin_arr[None, :] - sin_arr[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                off = sin_diff / (np.pi * nm_diff)
            np.fill_diagonal(off, 0.0)
            Qp += weight * off
            pk *= int(p)
        Qp = (Qp + Qp.T) / 2
        sp = signature(Qp)
        ep = np.linalg.eigvalsh(Qp)
        w_tot = logp / np.sqrt(p)
        print(f'  {p:>6d} {w_tot:>10.4f} ({sp[0]:>2d},{sp[1]:>2d},{sp[2]:>2d}) '
              f'{ep[-1]:>+12.6f}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)
    print()
    print('  The structural question: WHY does M have exactly 1 positive')
    print('  eigenvalue? The answer should come from the prime-by-prime')
    print('  evolution (Part 3) and individual prime signatures (Part 5).')


if __name__ == '__main__':
    run()
