"""
SESSION 58b -- THE CRITICAL DIRECTION

M_odd's max eigenvalue is ~-1e-7. What direction achieves this
near-exact cancellation between M_prime and M_diag+M_alpha?

If the eigenvector has a recognizable structure (e.g., it's the
w_hat test function, or a low-frequency mode, or something related
to zeta), that reveals the identity forcing the cancellation.

Also: check N-convergence of this eigenvalue. Is ~1e-7 the true
value or a truncation artifact? If it converges to exactly 0,
then M_odd is negative SEMIDEFINITE, not negative definite, and
the barrier vanishes.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast, _build_W02


def odd_block_with_vecs(M, N):
    """Extract odd block and the projection matrix."""
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    M_odd = P.T @ M @ P
    return M_odd, P


def run():
    print()
    print('#' * 76)
    print('  SESSION 58b -- THE CRITICAL DIRECTION')
    print('#' * 76)

    # == Part 1: N-convergence of M_odd's max eigenvalue ==
    print('\n  === PART 1: N-CONVERGENCE ===')
    print(f'  Fix lam^2 = 1000, vary N to check if max eigenvalue')
    print(f'  converges to a definite value or to zero.')
    print()

    lam_sq = 1000
    L_base = np.log(lam_sq)
    N_base = max(15, round(6 * L_base))

    print(f'  {"N":>5} {"dim":>5} {"max_eig(M_odd)":>18} {"min_eig(QW_odd)":>18}')
    print('  ' + '-' * 52)

    for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        N = max(10, round(mult * N_base))
        W02, M, QW = build_all_fast(lam_sq, N)
        M_odd, P = odd_block_with_vecs(M, N)
        QW_odd, _ = odd_block_with_vecs(QW, N)
        em = np.linalg.eigvalsh(M_odd)
        eq = np.linalg.eigvalsh(QW_odd)
        print(f'  {N:>5d} {2*N+1:>5d} {em[-1]:>+18.10e} {eq[0]:>+18.10e}')
    sys.stdout.flush()

    # == Part 2: The critical eigenvector ==
    print('\n  === PART 2: CRITICAL EIGENVECTOR STRUCTURE ===')

    # Use a well-converged N
    N = round(2.0 * N_base)
    W02, M, QW = build_all_fast(lam_sq, N)
    M_odd, P = odd_block_with_vecs(M, N)

    em, vm = np.linalg.eigh(M_odd)
    v_crit = vm[:, -1]  # eigenvector for max eigenvalue

    print(f'  At lam^2={lam_sq}, N={N} (dim={2*N+1}):')
    print(f'  Max eigenvalue of M_odd: {em[-1]:+.10e}')
    print(f'  2nd eigenvalue:          {em[-2]:+.10e}')
    print(f'  Ratio |2nd/max|:         {abs(em[-2]/em[-1]):.1f}')
    print()

    # The eigenvector v_crit is in the odd subspace.
    # In this basis, index k corresponds to the odd function (|k+1> - |-(k+1)>)/sqrt(2).
    # So v_crit[k] is the coefficient of the (k+1)-th odd mode.
    print(f'  Critical eigenvector (odd basis, n = index+1):')
    print(f'    {"n":>4} {"coeff":>12} {"|coeff|":>10}')
    print(f'    ' + '-' * 30)
    for k in range(min(N, 20)):
        n = k + 1
        print(f'    {n:>4d} {v_crit[k]:>+12.6f} {abs(v_crit[k]):>10.6f}')
    if N > 20:
        # Find peak
        peak_k = np.argmax(np.abs(v_crit))
        print(f'    ...')
        print(f'    peak at n={peak_k+1}: {v_crit[peak_k]:+.6f}')
    print(f'    norm: {np.linalg.norm(v_crit):.6f}')

    # Compare to w_hat (the standard odd test vector)
    ns = np.arange(-N, N + 1, dtype=float)
    L = np.log(lam_sq)
    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    # Project w_hat to odd subspace
    w_hat_odd = P.T @ w_hat
    w_hat_odd_norm = np.linalg.norm(w_hat_odd)
    if w_hat_odd_norm > 1e-15:
        w_hat_odd /= w_hat_odd_norm

    overlap_what = abs(float(v_crit @ w_hat_odd))
    print(f'\n  Overlap with w_hat (odd projection): {overlap_what:.8f}')

    # Compare to simple basis vectors
    print(f'  Overlap with individual odd modes:')
    for n in [1, 2, 3, 4, 5]:
        print(f'    n={n}: |v_crit[{n-1}]| = {abs(v_crit[n-1]):.6f}')

    # Is it a smooth function of n?
    print(f'\n  Shape analysis:')
    # Fit v_crit to simple forms
    ns_odd = np.arange(1, N + 1, dtype=float)
    # Try: v ~ n / (a^2 + n^2)  (Lorentzian shape, like w_hat)
    # Normalize and compare
    v_lorentz = ns_odd / (L**2 / (4*np.pi)**2 + ns_odd**2)
    v_lorentz /= np.linalg.norm(v_lorentz)
    overlap_lorentz = abs(float(v_crit @ v_lorentz))
    print(f'  Overlap with n/(a^2+n^2) [Lorentzian]: {overlap_lorentz:.8f}')

    # Try: v ~ 1/n (harmonic)
    v_harmonic = 1.0 / ns_odd
    v_harmonic /= np.linalg.norm(v_harmonic)
    overlap_harmonic = abs(float(v_crit @ v_harmonic))
    print(f'  Overlap with 1/n [harmonic]:           {overlap_harmonic:.8f}')

    # Try: v ~ n * exp(-c*n) (damped linear)
    for c in [0.1, 0.05, 0.02]:
        v_damp = ns_odd * np.exp(-c * ns_odd)
        v_damp /= np.linalg.norm(v_damp)
        ov = abs(float(v_crit @ v_damp))
        print(f'  Overlap with n*exp(-{c}*n):              {ov:.8f}')

    # == Part 3: Per-component Rayleigh quotients on critical direction ==
    print(f'\n  === PART 3: RAYLEIGH QUOTIENTS ON CRITICAL DIRECTION ===')

    # Lift v_crit back to full space
    v_full = P @ v_crit

    # Build components
    from session49c_weil_residual import (
        _build_M_prime, _compute_alpha, _compute_wr_diag
    )
    Mp = _build_M_prime(L, N, lam_sq)
    Mp = (Mp + Mp.T) / 2

    wr = _compute_wr_diag(L, N)
    Md = np.diag([wr[abs(int(n))] for n in np.arange(-N, N+1)])

    alpha = _compute_alpha(L, N)
    ns_f = np.arange(-N, N+1, dtype=float)
    nm = ns_f[:, None] - ns_f[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        Ma = (alpha[None, :] - alpha[:, None]) / nm
    np.fill_diagonal(Ma, 0.0)
    Ma = (Ma + Ma.T) / 2

    rq_w02 = float(v_full @ W02 @ v_full)
    rq_mp = float(v_full @ Mp @ v_full)
    rq_md = float(v_full @ Md @ v_full)
    rq_ma = float(v_full @ Ma @ v_full)
    rq_m = float(v_full @ M @ v_full)
    rq_qw = float(v_full @ QW @ v_full)

    print(f'  Component Rayleigh quotients on the critical direction:')
    print(f'    W02:        {rq_w02:+.10e}')
    print(f'    M_prime:    {rq_mp:+.10e}')
    print(f'    M_diag:     {rq_md:+.10e}')
    print(f'    M_alpha:    {rq_ma:+.10e}')
    print(f'    M_total:    {rq_m:+.10e}')
    print(f'    Q_W:        {rq_qw:+.10e}')
    print(f'    Check: W02 - M = {rq_w02 - rq_m:+.10e} (should = Q_W)')
    print()
    print(f'    M_prime + M_diag + M_alpha = {rq_mp + rq_md + rq_ma:+.10e}')
    print(f'    The cancellation: M_prime ({rq_mp:+.6f})')
    print(f'                    + M_diag  ({rq_md:+.6f})')
    print(f'                    + M_alpha ({rq_ma:+.6f})')
    print(f'                    = M_total ({rq_m:+.10e})')

    # == Part 4: Lambda sweep of Rayleigh quotients on critical direction ==
    print(f'\n  === PART 4: RAYLEIGH QUOTIENTS vs LAMBDA ===')
    print(f'  Track the component contributions on the critical direction')
    print(f'  as lambda changes (recomputing eigenvector each time).')
    print()
    print(f'  {"lam^2":>8} {"W02":>12} {"M_prime":>12} {"M_diag":>12} '
          f'{"M_alpha":>12} {"M_total":>14} {"Q_W":>14}')
    print('  ' + '-' * 86)

    for lam_sq in [50, 200, 1000, 5000, 20000]:
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        W02, M_t, QW = build_all_fast(lam_sq, N)
        M_odd_t, P_t = odd_block_with_vecs(M_t, N)
        em_t, vm_t = np.linalg.eigh(M_odd_t)
        v_crit_t = P_t @ vm_t[:, -1]  # lift to full space

        Mp_t = _build_M_prime(L, N, lam_sq)
        Mp_t = (Mp_t + Mp_t.T) / 2
        wr_t = _compute_wr_diag(L, N)
        Md_t = np.diag([wr_t[abs(int(n))] for n in np.arange(-N, N+1)])
        alpha_t = _compute_alpha(L, N)
        ns_t = np.arange(-N, N+1, dtype=float)
        nm_t = ns_t[:, None] - ns_t[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            Ma_t = (alpha_t[None, :] - alpha_t[:, None]) / nm_t
        np.fill_diagonal(Ma_t, 0.0)
        Ma_t = (Ma_t + Ma_t.T) / 2

        rq = {
            'w02': float(v_crit_t @ W02 @ v_crit_t),
            'mp': float(v_crit_t @ Mp_t @ v_crit_t),
            'md': float(v_crit_t @ Md_t @ v_crit_t),
            'ma': float(v_crit_t @ Ma_t @ v_crit_t),
            'mt': float(v_crit_t @ M_t @ v_crit_t),
            'qw': float(v_crit_t @ QW @ v_crit_t),
        }
        print(f'  {lam_sq:>8d} {rq["w02"]:>+12.6f} {rq["mp"]:>+12.6f} '
              f'{rq["md"]:>+12.6f} {rq["ma"]:>+12.6f} '
              f'{rq["mt"]:>+14.8e} {rq["qw"]:>+14.8e}')
    sys.stdout.flush()

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)
    print(f'\n  The critical direction (M_odd max eigenvector) determines')
    print(f'  whether the 10^-7 barrier is real or a truncation artifact.')
    print(f'  N-convergence and Rayleigh quotient anatomy tell the story.')


if __name__ == '__main__':
    run()
