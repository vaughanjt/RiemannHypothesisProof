"""
SESSION 63d -- THE TRANSITION: when does M_odd go negative?

Test 6 showed the max eigenvalue of M_odd goes:
  arch only -> +2.73, rises to +7.0 at ~100 pp, falls to -1.6e-7 at all 193 pp.

Two critical questions:
  1. Fine-grained transition: where exactly does it cross zero?
  2. Does the transition shift with lambda? Is there a universal pattern?
  3. Connection to the explicit formula: can we identify the zero contribution?
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


def run():
    print()
    print('#' * 76)
    print('  SESSION 63d -- THE TRANSITION')
    print('#' * 76)

    # =================================================================
    # PART 1: Fine-grained transition at lam^2 = 1000
    # =================================================================
    print('\n  === PART 1: FINE-GRAINED TRANSITION (lam^2=1000) ===')
    print('  Add all prime powers, track max_eig at every step near crossing.\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # Build archimedean base
    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)
    M_arch = np.zeros((dim, dim))
    for n in range(-N, N + 1):
        M_arch[N + n, N + n] = wr[abs(n)]
    a_m = alpha[None, :]
    a_n_arr = alpha[:, None]
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        offdiag = (a_m - a_n_arr) / nm
    np.fill_diagonal(offdiag, 0.0)
    M_arch += offdiag
    M_arch = (M_arch + M_arch.T) / 2

    # Get prime powers sorted by y
    primes = list(sieve_primes(int(lam_sq)))
    pps = []
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            pps.append((logp * pk**(-0.5), np.log(pk), p, pk))
            pk *= int(p)
    pps.sort(key=lambda x: x[1])
    total = len(pps)

    # Full run tracking max_eig at every step from pp 150 onward
    M_running = M_arch.copy()
    max_eigs = []
    for i, (w, y, p, pk) in enumerate(pps):
        sin_arr = np.sin(2 * np.pi * ns * y / L)
        cos_arr = np.cos(2 * np.pi * ns * y / L)
        diag = 2 * (L - y) / L * cos_arr
        np.fill_diagonal(M_running, M_running.diagonal() + w * diag)
        nm_diff = ns[:, None] - ns[None, :]
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off, 0.0)
        M_running += w * off

        M_sym = (M_running + M_running.T) / 2
        Mo = odd_block(M_sym, N)
        max_eigs.append(np.linalg.eigvalsh(Mo)[-1])

    # Print around the transition
    print(f'  {"#pp":>5} {"pk":>8} {"p":>4} {"w":>8} '
          f'{"max_eig(M_odd)":>16} {"cumulative":>10}')
    print('  ' + '-' * 55)

    # Find crossing
    crossing_idx = None
    for i in range(len(max_eigs)):
        if max_eigs[i] < 0:
            crossing_idx = i
            break

    # Print from 20 before crossing to end
    start = max(0, (crossing_idx or total) - 25)
    for i in range(start, total):
        w, y, p, pk = pps[i]
        marker = ' <-- CROSSING' if i == crossing_idx else ''
        print(f'  {i+1:>5d} {pk:>8d} {p:>4d} {w:>8.4f} '
              f'{max_eigs[i]:>+16.8e} '
              f'{"NEG" if max_eigs[i] < 0 else "":>10}{marker}')
    sys.stdout.flush()

    # =================================================================
    # PART 2: Transition at different lambda values
    # =================================================================
    print('\n  === PART 2: TRANSITION POINT vs LAMBDA ===')
    print('  At what fraction of prime powers does M_odd cross zero?\n')

    print(f'  {"lam^2":>8} {"total pp":>9} {"cross at":>9} '
          f'{"frac":>8} {"last p before":>14} {"max before":>14}')
    print('  ' + '-' * 68)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))
        dim = 2 * N + 1
        ns = np.arange(-N, N + 1, dtype=float)

        wr = _compute_wr_diag(L, N)
        alpha = _compute_alpha(L, N)
        M_base = np.zeros((dim, dim))
        for n in range(-N, N + 1):
            M_base[N + n, N + n] = wr[abs(n)]
        a_m2 = alpha[None, :]
        a_n2 = alpha[:, None]
        nm2 = ns[:, None] - ns[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            od = (a_m2 - a_n2) / nm2
        np.fill_diagonal(od, 0.0)
        M_base += od
        M_base = (M_base + M_base.T) / 2

        prms = list(sieve_primes(int(lam_sq)))
        pp_list = []
        for p in prms:
            pk = int(p)
            logp = np.log(p)
            while pk <= lam_sq:
                pp_list.append((logp * pk**(-0.5), np.log(pk), p, pk))
                pk *= int(p)
        pp_list.sort(key=lambda x: x[1])
        n_pp = len(pp_list)

        M_run = M_base.copy()
        cross_at = n_pp  # default: only crosses at the very end
        max_before = 0
        last_p = 0
        for i, (w, y, p, pk) in enumerate(pp_list):
            sin_arr = np.sin(2 * np.pi * ns * y / L)
            cos_arr = np.cos(2 * np.pi * ns * y / L)
            d = 2 * (L - y) / L * cos_arr
            np.fill_diagonal(M_run, M_run.diagonal() + w * d)
            nm_diff2 = ns[:, None] - ns[None, :]
            sd = sin_arr[None, :] - sin_arr[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                of = sd / (np.pi * nm_diff2)
            np.fill_diagonal(of, 0.0)
            M_run += w * of

            # Only check near the end for speed
            if i >= n_pp - 30 or n_pp <= 50:
                M_s = (M_run + M_run.T) / 2
                Mo2 = odd_block(M_s, N)
                me = np.linalg.eigvalsh(Mo2)[-1]
                if me < 0 and cross_at == n_pp:
                    cross_at = i + 1
                if me >= 0:
                    max_before = me
                    last_p = p

        frac = cross_at / n_pp if n_pp > 0 else 0
        print(f'  {lam_sq:>8d} {n_pp:>9d} {cross_at:>9d} '
              f'{frac:>8.4f} {last_p:>14d} {max_before:>+14.6e}')
    sys.stdout.flush()

    # =================================================================
    # PART 3: THE EXPLICIT FORMULA VIEW
    # =================================================================
    print('\n  === PART 3: EXPLICIT FORMULA DECOMPOSITION ===')
    print('  The prime Fourier coefficients are:')
    print('    F(n) = Sum w_pk * exp(2*pi*i*n*y_k/L)')
    print('  By explicit formula: F(n) = -Sum_rho (contribution from zeros)')
    print('  On the critical line, these zeros create the cancellation.\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))

    prms = list(sieve_primes(int(lam_sq)))

    # Compute F(n) = Sum w_pk * exp(2*pi*i*n*y/L) for n = 1..N
    F_n = np.zeros(N, dtype=complex)
    for p in prms:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            for idx in range(N):
                n = idx + 1
                F_n[idx] += w * np.exp(2j * np.pi * n * y / L)
            pk *= int(p)

    print(f'  {"n":>4} {"|F(n)|":>12} {"Re F(n)":>12} {"Im F(n)":>12} '
          f'{"angle":>10}')
    print('  ' + '-' * 50)
    for idx in range(min(15, N)):
        n = idx + 1
        fn = F_n[idx]
        print(f'  {n:>4d} {abs(fn):>12.4f} {fn.real:>12.4f} {fn.imag:>12.4f} '
              f'{np.angle(fn):>10.4f}')

    # PNT prediction: F(0) ~ Sum w_pk = 2*sqrt(lam^2) approx
    F0 = sum(np.log(p) * p**(-0.5) for p in prms
             for _ in range(1))  # k=1 only approximation
    F0_full = 0
    for p in prms:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            F0_full += logp * pk**(-0.5)
            pk *= int(p)
    pnt_pred = 2 * np.sqrt(lam_sq)

    print(f'\n  F(0) = {F0_full:.4f} (PNT prediction: ~{pnt_pred:.1f})')
    print(f'  |F(1)| / F(0) = {abs(F_n[0]) / F0_full:.4f}')
    print(f'  |F(n)| / F(0) decreases, showing destructive interference at n>=1')
    sys.stdout.flush()

    # =================================================================
    # PART 4: HOW MUCH OF THE SHIFT IS FROM DIAGONAL vs OFF-DIAGONAL?
    # =================================================================
    print('\n  === PART 4: DIAGONAL vs OFF-DIAGONAL SHIFT ===')
    print('  When the last primes tip M_odd to negative:')
    print('  How much comes from shifting the diagonal vs the off-diagonal?\n')

    lam_sq = 1000
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns_full = np.arange(-N, N + 1, dtype=float)

    # Build M with only first 180 prime powers
    pp_list = []
    for p in sieve_primes(int(lam_sq)):
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            pp_list.append((logp * pk**(-0.5), np.log(pk), p, pk))
            pk *= int(p)
    pp_list.sort(key=lambda x: x[1])

    # Archimedean + first 180 pp
    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)
    M_180 = np.zeros((dim, dim))
    for n in range(-N, N + 1):
        M_180[N + n, N + n] = wr[abs(n)]
    a_m3 = alpha[None, :]
    a_n3 = alpha[:, None]
    nm3 = ns_full[:, None] - ns_full[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        od3 = (a_m3 - a_n3) / nm3
    np.fill_diagonal(od3, 0.0)
    M_180 += od3
    for i in range(min(180, len(pp_list))):
        w, y, p, pk = pp_list[i]
        sin_arr = np.sin(2 * np.pi * ns_full * y / L)
        cos_arr = np.cos(2 * np.pi * ns_full * y / L)
        d = 2 * (L - y) / L * cos_arr
        np.fill_diagonal(M_180, M_180.diagonal() + w * d)
        nm_d = ns_full[:, None] - ns_full[None, :]
        sd = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            of = sd / (np.pi * nm_d)
        np.fill_diagonal(of, 0.0)
        M_180 += w * of
    M_180 = (M_180 + M_180.T) / 2

    # Full M
    _, M_full, _ = build_all_fast(lam_sq, N)

    # Difference
    dM = M_full - M_180
    Mo_180 = odd_block(M_180, N)
    Mo_full = odd_block(M_full, N)
    Mo_diff = Mo_full - Mo_180

    # Decompose diff into diagonal and off-diagonal
    dM_diag = np.diag(np.diag(Mo_diff))
    dM_offdiag = Mo_diff - dM_diag

    # Rayleigh quotient of each on the max eigenvector of Mo_180
    eigs_180, vecs_180 = np.linalg.eigh(Mo_180)
    v_max_180 = vecs_180[:, -1]

    ray_diag = float(v_max_180 @ dM_diag @ v_max_180)
    ray_offdiag = float(v_max_180 @ dM_offdiag @ v_max_180)
    ray_total = float(v_max_180 @ Mo_diff @ v_max_180)

    print(f'  Mo_180 max_eig: {eigs_180[-1]:+.6f}')
    print(f'  Mo_full max_eig: {np.linalg.eigvalsh(Mo_full)[-1]:+.6e}')
    print(f'  Shift needed: {-eigs_180[-1]:+.6f}')
    print(f'  Actual shift (Rayleigh on v_max_180): {ray_total:+.6f}')
    print(f'    From diagonal:     {ray_diag:+.6f} ({100*ray_diag/ray_total:.1f}%)')
    print(f'    From off-diagonal: {ray_offdiag:+.6f} ({100*ray_offdiag/ray_total:.1f}%)')
    sys.stdout.flush()

    # =================================================================
    print()
    print('=' * 76)
    print('  SESSION 63d RESULTS')
    print('=' * 76)


if __name__ == '__main__':
    run()
