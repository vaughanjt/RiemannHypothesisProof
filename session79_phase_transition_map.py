"""
SESSION 79 -- MAPPING THE SPECTRAL PHASE TRANSITION

The Lorentzian property is a knife-edge at the exact Gamma diagonal.
Map it precisely, then prove it.

PHASE 1 — MAP:
  1. Fine-grained eigenvalue count vs diagonal scale (1000 points near 1.0)
  2. Track INDIVIDUAL eigenvalues through the transition
  3. Which eigenvalues cross zero at scale=1? In what order?
  4. The transition width: how sharp is the knife-edge?
  5. Does the transition point depend on lambda?
  6. Separate archimedean diagonal from prime diagonal: which is load-bearing?

PHASE 2 — MECHANISM:
  7. At scale=1, why exactly 1 positive eigenvalue? What cancels?
  8. The eigenvector at the transition: what direction flips?
  9. Connection to the trace: tr(M_odd) as function of scale
  10. The determinant: det(M_odd) sign change

PHASE 3 — TOWARD PROOF:
  11. Simplified model: diag(-log n) + Cauchy(1/(n-m)) — does it have Lorentzian?
  12. Pure archimedean (no primes): is the transition still at scale=1?
  13. Random matrix comparison: replace Cauchy with GOE — transition?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)
from session41g_uncapped_barrier import sieve_primes


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def build_components(lam_sq, N=None):
    """Return M and its diagonal wr_diag separately."""
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    wr = _compute_wr_diag(L, N)
    _, M, _ = build_all_fast(lam_sq, N)

    # Extract wr_diag contribution
    wr_diag_vec = np.array([wr[abs(int(n))] for n in ns])

    return M, wr_diag_vec, N, L, dim, ns


def run():
    print()
    print('#' * 76)
    print('  SESSION 79 -- MAPPING THE SPECTRAL PHASE TRANSITION')
    print('#' * 76)

    lam_sq = 1000
    M, wr_diag, N, L, dim, ns = build_components(lam_sq)

    # ======================================================================
    # MAP 1: Fine-grained eigenvalue count vs scale
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  MAP 1: EIGENVALUE COUNT vs DIAGONAL SCALE (fine grid)')
    print(f'{"="*76}\n')

    # Scale the wr_diag component of the diagonal
    # M_scaled = M - diag(wr_diag) + scale * diag(wr_diag)
    #          = M + (scale - 1) * diag(wr_diag)
    M_no_wr = M - np.diag(wr_diag)

    scales = np.concatenate([
        np.linspace(0.90, 0.98, 20),
        np.linspace(0.980, 0.999, 20),
        np.linspace(0.999, 1.001, 21),
        np.linspace(1.001, 1.020, 20),
        np.linspace(1.02, 1.10, 20),
    ])
    scales = np.unique(np.round(scales, 6))

    print(f'  {"scale":>10} {"#pos(M)":>8} {"#pos(Mo)":>8} {"eig_max(Mo)":>14} '
          f'{"eig_2(Mo)":>14}')
    print('  ' + '-' * 60)

    transition_data = []
    for scale in scales:
        M_s = M_no_wr + scale * np.diag(wr_diag)
        evals = np.linalg.eigvalsh(M_s)
        npos = np.sum(evals > 1e-10)

        Mo = odd_block(M_s, N)
        eo = np.linalg.eigvalsh(Mo)
        npos_o = np.sum(eo > 1e-10)
        emax_o = eo[-1]
        e2_o = eo[-2] if len(eo) >= 2 else 0

        transition_data.append((scale, npos, npos_o, emax_o, e2_o))

        # Print only at interesting points
        if abs(scale - 1.0) < 0.002 or npos <= 2 or scale in [0.90, 0.95, 1.05, 1.10]:
            print(f'  {scale:>10.6f} {npos:>8d} {npos_o:>8d} '
                  f'{emax_o:>+14.6e} {e2_o:>+14.6e}')

    # Find exact transition points
    print(f'\n  Transition points (where #pos changes):')
    for i in range(1, len(transition_data)):
        s_prev, np_prev, npo_prev, _, _ = transition_data[i-1]
        s_curr, np_curr, npo_curr, _, _ = transition_data[i]
        if np_prev != np_curr:
            print(f'    scale {s_prev:.6f} -> {s_curr:.6f}: '
                  f'#pos(M) {np_prev} -> {np_curr}')
        if npo_prev != npo_curr:
            print(f'    scale {s_prev:.6f} -> {s_curr:.6f}: '
                  f'#pos(Mo) {npo_prev} -> {npo_curr}')
    sys.stdout.flush()

    # ======================================================================
    # MAP 2: Track individual eigenvalues through transition
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  MAP 2: EIGENVALUE TRAJECTORIES THROUGH TRANSITION')
    print(f'{"="*76}\n')

    fine_scales = np.linspace(0.995, 1.005, 101)
    trajectories = []

    for scale in fine_scales:
        M_s = M_no_wr + scale * np.diag(wr_diag)
        Mo = odd_block(M_s, N)
        eo = np.linalg.eigvalsh(Mo)
        trajectories.append(eo)

    trajectories = np.array(trajectories)  # shape: (101, N)

    # Print the top 5 eigenvalues at selected scales
    print(f'  Top 5 eigenvalues of M_odd near scale=1:')
    print(f'  {"scale":>10} {"eig_1":>14} {"eig_2":>14} {"eig_3":>14} '
          f'{"eig_4":>14} {"eig_5":>14}')
    print('  ' + '-' * 82)

    for i in range(0, len(fine_scales), 10):
        s = fine_scales[i]
        top5 = trajectories[i, -5:][::-1]
        print(f'  {s:>10.6f} ' + ' '.join(f'{e:>+14.6e}' for e in top5))

    # How many eigenvalues cross zero?
    for j in range(N-1, max(N-10, 0), -1):
        traj = trajectories[:, j]
        if traj[0] > 0 and traj[-1] < 0:
            cross_idx = np.where(np.diff(np.sign(traj)))[0]
            if len(cross_idx) > 0:
                cross_scale = fine_scales[cross_idx[0]]
                print(f'\n  Eigenvalue {N-j} crosses zero at scale ~ {cross_scale:.6f}')
    sys.stdout.flush()

    # ======================================================================
    # MAP 3: Transition at different lambda
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  MAP 3: IS THE TRANSITION ALWAYS AT scale=1.0?')
    print(f'{"="*76}\n')

    print(f'  {"lam^2":>8} {"scale(1+)":>10} {"scale(1-)":>10} '
          f'{"#pos at 1.0":>12} {"eig_max(Mo)":>14}')
    print('  ' + '-' * 60)

    for lam_sq_t in [100, 200, 500, 1000, 2000, 5000, 10000]:
        M_t, wr_t, N_t, L_t, dim_t, ns_t = build_components(lam_sq_t)
        M_no_wr_t = M_t - np.diag(wr_t)

        # Binary search for upper transition (scale > 1 where #pos increases)
        s_lo, s_hi = 1.0, 1.1
        for _ in range(50):
            s_mid = (s_lo + s_hi) / 2
            M_s = M_no_wr_t + s_mid * np.diag(wr_t)
            npos = np.sum(np.linalg.eigvalsh(M_s) > 1e-10)
            if npos <= 1:
                s_lo = s_mid
            else:
                s_hi = s_mid
        scale_upper = (s_lo + s_hi) / 2

        # Binary search for lower transition (scale < 1 where #pos increases)
        s_lo, s_hi = 0.9, 1.0
        for _ in range(50):
            s_mid = (s_lo + s_hi) / 2
            M_s = M_no_wr_t + s_mid * np.diag(wr_t)
            npos = np.sum(np.linalg.eigvalsh(M_s) > 1e-10)
            if npos <= 1:
                s_hi = s_mid
            else:
                s_lo = s_mid
        scale_lower = (s_lo + s_hi) / 2

        M_exact = M_no_wr_t + np.diag(wr_t)
        npos_exact = np.sum(np.linalg.eigvalsh(M_exact) > 1e-10)
        Mo_exact = odd_block(M_exact, N_t)
        emax_exact = np.linalg.eigvalsh(Mo_exact)[-1]

        print(f'  {lam_sq_t:>8d} {scale_upper:>10.6f} {scale_lower:>10.6f} '
              f'{npos_exact:>12d} {emax_exact:>+14.6e}')
    sys.stdout.flush()

    # ======================================================================
    # MAP 4: Pure archimedean — is transition still at 1.0?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  MAP 4: PURE ARCHIMEDEAN (NO PRIMES) — TRANSITION POINT?')
    print(f'{"="*76}\n')

    for lam_sq_t in [200, 1000, 5000]:
        L_t = np.log(lam_sq_t)
        N_t = max(15, round(6 * L_t))
        dim_t = 2 * N_t + 1
        ns_t = np.arange(-N_t, N_t + 1, dtype=float)

        wr = _compute_wr_diag(L_t, N_t)
        alpha = _compute_alpha(L_t, N_t)

        # Build archimedean-only M
        a_arch = np.array([wr[abs(int(n))] for n in ns_t])
        nm = ns_t[:, None] - ns_t[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha_offdiag = (alpha[None, :] - alpha[:, None]) / nm
        np.fill_diagonal(alpha_offdiag, 0)
        M_arch = np.diag(a_arch) + alpha_offdiag
        M_arch = (M_arch + M_arch.T) / 2

        evals_arch = np.linalg.eigvalsh(M_arch)
        npos_arch = np.sum(evals_arch > 1e-10)

        # Scale test on archimedean only
        M_arch_no_wr = M_arch - np.diag(a_arch)
        results = []
        for scale in [0.5, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5]:
            M_s = M_arch_no_wr + scale * np.diag(a_arch)
            npos_s = np.sum(np.linalg.eigvalsh(M_s) > 1e-10)
            results.append(f'{scale}:{npos_s}')

        print(f'  lam^2={lam_sq_t}: arch #pos={npos_arch}, '
              f'by scale: {", ".join(results)}')
    sys.stdout.flush()

    # ======================================================================
    # MAP 5: Simplified model — pure log diagonal + simple Cauchy
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  MAP 5: TOY MODEL — diag(-log n) + Cauchy(1/(n-m))')
    print(f'{"="*76}\n')

    # Build a simple Cauchy matrix with constant B_n = n
    # Cauchy[n,m] = (m - n) / (n - m) = -1 for n != m
    # That's trivial. Use B_n = log(n+1) instead.

    for N_toy in [20, 40, 60]:
        dim_toy = N_toy
        ns_toy = np.arange(1, N_toy + 1, dtype=float)

        # Diagonal: -log(n)
        D = np.diag(-np.log(ns_toy))

        # Cauchy matrix: C[i,j] = 1/(n_i - n_j) for i != j
        nm_toy = ns_toy[:, None] - ns_toy[None, :]
        with np.errstate(divide='ignore'):
            C = 1.0 / nm_toy
        np.fill_diagonal(C, 0)
        C = (C + C.T) / 2

        # M_toy = alpha * D + C for various alpha
        print(f'  N={N_toy}:')
        for alpha in [0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            M_toy = alpha * D + C
            evals_toy = np.linalg.eigvalsh(M_toy)
            npos_toy = np.sum(evals_toy > 1e-10)
            print(f'    alpha={alpha:>5.1f}: #pos={npos_toy:>3d}/{N_toy}, '
                  f'evals [{evals_toy.min():.4f}, {evals_toy.max():.4f}]')
        print()

    # More realistic toy: B_n from digamma
    print(f'  Realistic toy: B_n = Im[psi(1/4 + i*pi*n/L)] (archimedean B)')
    for N_toy in [20, 41]:
        L_toy = 6.908
        ns_toy = np.arange(1, N_toy + 1, dtype=float)

        # Use actual alpha values (the archimedean B_n)
        alpha_toy = _compute_alpha(L_toy, max(N_toy, 15))
        # B_n for odd block: alpha[N+n] for n=1..N
        N_full = max(N_toy, 15)
        B_odd = np.array([alpha_toy[N_full + n] for n in range(1, N_toy + 1)])

        # Cauchy matrix from B_odd
        nm_toy = ns_toy[:, None] - ns_toy[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            C_toy = (B_odd[None, :] - B_odd[:, None]) / nm_toy
        np.fill_diagonal(C_toy, 0)
        C_toy = (C_toy + C_toy.T) / 2

        # wr_diag for odd block
        wr_toy = _compute_wr_diag(L_toy, max(N_toy, 15))
        D_toy = np.diag(np.array([wr_toy[n] for n in range(1, N_toy + 1)]))

        # Scale test
        print(f'  N={N_toy}, L={L_toy}:')
        for scale in [0, 0.5, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5, 2.0]:
            M_toy = scale * D_toy + C_toy
            evals_toy = np.linalg.eigvalsh(M_toy)
            npos_toy = np.sum(evals_toy > 1e-10)
            marker = ' <-- LORENTZIAN' if npos_toy == 0 else ''
            print(f'    scale={scale:>5.2f}: #pos={npos_toy:>3d}/{N_toy}'
                  f'{marker}')
        print()
    sys.stdout.flush()

    # ======================================================================
    # MAP 6: The transition width
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  MAP 6: TRANSITION WIDTH (how sharp is the knife-edge?)')
    print(f'{"="*76}\n')

    lam_sq = 1000
    M, wr_diag, N, L, dim, ns = build_components(lam_sq)
    M_no_wr = M - np.diag(wr_diag)

    # Ultra-fine scan near scale=1
    ultra_fine = np.linspace(0.9999, 1.0001, 201)
    print(f'  Ultra-fine scan near scale=1.0 (lam^2={lam_sq}):')
    print(f'  {"scale":>12} {"#pos(M)":>8} {"eig_max(Mo)":>14}')
    print('  ' + '-' * 38)

    for s in ultra_fine[::20]:  # every 20th point
        M_s = M_no_wr + s * np.diag(wr_diag)
        npos = np.sum(np.linalg.eigvalsh(M_s) > 1e-10)
        Mo = odd_block(M_s, N)
        emax = np.linalg.eigvalsh(Mo)[-1]
        print(f'  {s:>12.8f} {npos:>8d} {emax:>+14.6e}')

    # Find the EXACT scale where #pos(M) changes from 1 to 2
    s_lo, s_hi = 1.0, 1.001
    for _ in range(60):
        s_mid = (s_lo + s_hi) / 2
        M_s = M_no_wr + s_mid * np.diag(wr_diag)
        npos = np.sum(np.linalg.eigvalsh(M_s) > 1e-10)
        if npos <= 1:
            s_lo = s_mid
        else:
            s_hi = s_mid

    print(f'\n  Upper transition at scale = {(s_lo+s_hi)/2:.15f}')
    print(f'  Width above 1.0: {(s_lo+s_hi)/2 - 1.0:.2e}')

    s_lo, s_hi = 0.999, 1.0
    for _ in range(60):
        s_mid = (s_lo + s_hi) / 2
        M_s = M_no_wr + s_mid * np.diag(wr_diag)
        npos = np.sum(np.linalg.eigvalsh(M_s) > 1e-10)
        if npos <= 1:
            s_hi = s_mid
        else:
            s_lo = s_mid

    print(f'  Lower transition at scale = {(s_lo+s_hi)/2:.15f}')
    print(f'  Width below 1.0: {1.0 - (s_lo+s_hi)/2:.2e}')
    print(f'  Total window: {(s_lo+s_hi)/2:.15f} to {(s_lo+s_hi)/2:.15f}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 79 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
