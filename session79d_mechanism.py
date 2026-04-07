"""
SESSION 79d -- THE MECHANISM: WHY EXACTLY ONE SURVIVES

M(t) = L_cauchy + t * diag(wr)

At t=0: L_cauchy has ~50 positive eigenvalues.
At t=1: M has exactly 1 positive eigenvalue.

Hellmann-Feynman: d(lambda_k)/dt = v_k^T diag(wr) v_k = <wr>_k
  - Eigenvectors at large |n|: wr negative => eigenvalue decreases
  - Eigenvector at n=0: wr(0) = +4.16 => eigenvalue INCREASES

The ONE survivor is the eigenvector concentrated at n=0.
The 49 that die are concentrated at large |n| where wr < 0.

PROBES:
  1. Track all positive eigenvalues from t=0 to t=1
  2. At t=0: where are the eigenvectors concentrated? Mean |n|?
  3. Compute <wr>_k = v_k^T diag(wr) v_k for each positive eigenvector of L_cauchy
  4. The crossing time: t_cross = lambda_k(0) / |<wr>_k| (first-order estimate)
  5. Does the n=0 eigenvector ALWAYS have <wr> > 0?
  6. Do ALL other positive eigenvectors have <wr> < -lambda_k?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)
from session41g_uncapped_barrier import sieve_primes


def decompose_L_and_D(lam_sq, N=None):
    """Decompose M = L_cauchy + diag(wr_full) exactly."""
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    _, M, _ = build_all_fast(lam_sq, N)
    wr = _compute_wr_diag(L, N)
    wr_vec = np.array([wr[abs(int(n))] for n in ns])

    # L_cauchy = M - diag(wr_vec) approximately
    # But M also has prime diagonal contributions!
    # The FULL diagonal of M is: wr + prime_diag
    # L_cauchy = M - diag(M_diag) + diag(Cauchy_diag_limit)
    # For simplicity: define D_full = diag(M) and L = M - diag(D_full)

    D_full = np.diag(M).copy()
    L_offdiag = M - np.diag(D_full)

    return L_offdiag, D_full, M, wr_vec, N, L, dim, ns


def run():
    print()
    print('#' * 76)
    print('  SESSION 79d -- WHY EXACTLY ONE EIGENVALUE SURVIVES')
    print('#' * 76)

    lam_sq = 1000
    L_off, D_full, M, wr_vec, N, L, dim, ns = decompose_L_and_D(lam_sq)

    # ======================================================================
    # PROBE 1: Eigenvalue flow M(t) = L_off + t * diag(D_full)
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: EIGENVALUE FLOW FROM L_off TO M')
    print(f'{"="*76}\n')

    t_vals = np.linspace(0, 1, 51)
    n_pos_at_t = []

    for t in t_vals:
        Mt = L_off + t * np.diag(D_full)
        evals = np.linalg.eigvalsh(Mt)
        npos = np.sum(evals > 1e-10)
        n_pos_at_t.append(npos)

    print(f'  {"t":>6} {"#pos":>6}')
    print('  ' + '-' * 14)
    for i in range(0, len(t_vals), 5):
        print(f'  {t_vals[i]:>6.2f} {n_pos_at_t[i]:>6d}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Hellmann-Feynman diagnostic at t=0
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: HELLMANN-FEYNMAN AT t=0')
    print(f'{"="*76}\n')

    evals_L, evecs_L = np.linalg.eigh(L_off)

    # For each positive eigenvalue, compute the HF derivative
    pos_mask = evals_L > 1e-10
    pos_indices = np.where(pos_mask)[0]
    n_pos_L = len(pos_indices)

    print(f'  L_off has {n_pos_L} positive eigenvalues')
    print()
    print(f'  {"rank":>5} {"lambda_k":>12} {"<wr>_k":>12} {"mean|n|":>8} '
          f'{"n=0 wt":>8} {"t_cross":>10} {"survives?":>10}')
    print('  ' + '-' * 72)

    survivors = []
    for rank, idx in enumerate(reversed(pos_indices)):
        lam_k = evals_L[idx]
        v_k = evecs_L[:, idx]

        # HF derivative: <D>_k = v_k^T diag(D_full) v_k
        hf = np.sum(D_full * v_k**2)

        # Mean |n|
        mean_n = np.sum(np.abs(ns) * v_k**2)

        # n=0 weight
        n0_wt = v_k[N]**2

        # Crossing time estimate (first order)
        if hf < 0:
            t_cross = -lam_k / hf
        else:
            t_cross = float('inf')  # eigenvalue increases, never crosses

        survives = t_cross > 1.0 or hf >= 0

        survivors.append((rank, lam_k, hf, mean_n, n0_wt, t_cross, survives))

        if rank < 20 or survives:
            print(f'  {rank:>5d} {lam_k:>+12.6f} {hf:>+12.6f} {mean_n:>8.1f} '
                  f'{n0_wt:>8.4f} {t_cross:>10.4f} '
                  f'{"YES" if survives else "no":>10}')

    n_survive = sum(1 for s in survivors if s[6])
    print(f'\n  Predicted survivors (t_cross > 1 or <wr> >= 0): {n_survive}')
    print(f'  Actual survivors at t=1: {n_pos_at_t[-1]}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: The ONLY survivor — why does it have <wr> > 0?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: THE SURVIVOR\'S ANATOMY')
    print(f'{"="*76}\n')

    # Find the survivor(s)
    for rank, lam_k, hf, mean_n, n0_wt, t_cross, surv in survivors:
        if surv:
            idx = pos_indices[-(rank+1)]
            v = evecs_L[:, idx]

            print(f'  Survivor: rank {rank}, lambda = {lam_k:.6f}')
            print(f'    <D_full> = {hf:+.6f} (HF derivative)')
            print(f'    mean |n| = {mean_n:.2f}')
            print(f'    n=0 weight = {n0_wt:.4f}')
            print()

            # Decompose <D_full> = <wr> + <prime_diag>
            hf_wr = np.sum(wr_vec * v**2)
            hf_prime = hf - hf_wr
            print(f'    <wr_diag> = {hf_wr:+.6f} (Gamma contribution)')
            print(f'    <prime_diag> = {hf_prime:+.6f} (prime contribution)')
            print()

            # Weight distribution
            print(f'    Weight by |n| range:')
            for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, N)]:
                wt = sum(v[N+n]**2 + (v[N-n]**2 if n > 0 else 0)
                         for n in range(lo, min(hi+1, N+1)))
                wr_contrib = sum(wr_vec[N+n] * v[N+n]**2 +
                                 (wr_vec[N-n] * v[N-n]**2 if n > 0 else 0)
                                 for n in range(lo, min(hi+1, N+1)))
                print(f'      |n| in [{lo}, {hi}]: weight = {wt:.4f}, '
                      f'wr contrib = {wr_contrib:+.6f}')
            print()

            # Is this eigenvector essentially the n=0 mode?
            print(f'    Top 5 components:')
            sorted_idx = np.argsort(np.abs(v))[::-1]
            for j in range(5):
                i = sorted_idx[j]
                print(f'      n={int(ns[i]):>3d}: |v| = {abs(v[i]):.6f}, '
                      f'wr = {wr_vec[i]:+.4f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: WHY do the non-survivors have <wr> < -lambda?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: WHY NON-SURVIVORS DIE')
    print(f'{"="*76}\n')

    print(f'  For each non-surviving eigenvalue:')
    print(f'  {"rank":>5} {"lambda":>10} {"<D>":>10} {"<wr>":>10} '
          f'{"<prime_d>":>10} {"mean|n|":>8} {"n=0 wt":>8}')
    print('  ' + '-' * 66)

    for rank, lam_k, hf, mean_n, n0_wt, t_cross, surv in survivors[:30]:
        if not surv:
            idx = pos_indices[-(rank+1)]
            v = evecs_L[:, idx]
            hf_wr = np.sum(wr_vec * v**2)
            hf_prime = hf - hf_wr
            print(f'  {rank:>5d} {lam_k:>+10.4f} {hf:>+10.4f} {hf_wr:>+10.4f} '
                  f'{hf_prime:>+10.4f} {mean_n:>8.1f} {n0_wt:>8.4f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: The critical inequality — can we prove it?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: THE CRITICAL INEQUALITY')
    print(f'{"="*76}\n')

    # For M to be Lorentzian: ALL positive eigenvalues of L_off except
    # the top one must satisfy:
    #   <D_full>_k < -lambda_k
    # (so that lambda_k + t*<D>_k < 0 at t=1)
    #
    # This requires: sum_n D_full(n) * v_k(n)^2 < -lambda_k
    #
    # Since D_full(n) = wr(n) + prime_diag(n), and wr(n) ~ C - log(n):
    #   sum_n (C - log(n)) * v_k(n)^2 + sum_n prime_diag(n) * v_k(n)^2 < -lambda_k
    #
    # C - <log(n)>_k + <prime_diag>_k < -lambda_k
    #
    # Rearranging: <log(n)>_k > C + lambda_k + <prime_diag>_k
    #
    # For this to hold: the eigenvector must be concentrated at LARGE n
    # (large <log(n)>), and the prime diagonal must not compensate too much.

    print(f'  The inequality: <log|n|>_k > C + lambda_k + <prime_diag>_k')
    print(f'  where C = wr(0) = {wr_vec[N]:+.4f}')
    print()
    print(f'  {"rank":>5} {"lambda_k":>10} {"<log|n|>":>10} {"C+lam+<pd>":>12} '
          f'{"gap":>10} {"passes?":>8}')
    print('  ' + '-' * 60)

    C_val = wr_vec[N]  # wr(0)
    for rank, lam_k, hf, mean_n, n0_wt, t_cross, surv in survivors[:20]:
        idx = pos_indices[-(rank+1)]
        v = evecs_L[:, idx]
        hf_wr = np.sum(wr_vec * v**2)
        hf_prime = hf - hf_wr

        # <log|n|> weighted by v^2 (excluding n=0)
        log_n_avg = sum(np.log(abs(ns[i])) * v[i]**2
                        for i in range(dim) if ns[i] != 0)

        threshold = C_val + lam_k + hf_prime
        gap = log_n_avg - threshold
        passes = gap > 0

        print(f'  {rank:>5d} {lam_k:>+10.4f} {log_n_avg:>10.4f} '
              f'{threshold:>+12.4f} {gap:>+10.4f} '
              f'{"YES" if passes else "NO":>8}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: Lambda scaling — does the mechanism hold at all lambda?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: MECHANISM AT DIFFERENT LAMBDA')
    print(f'{"="*76}\n')

    print(f'  {"lam^2":>8} {"#pos(L)":>8} {"#survive":>8} {"survivor <wr>":>14} '
          f'{"n=0 wt":>8}')
    print('  ' + '-' * 52)

    for lam_sq_t in [100, 200, 500, 1000, 2000, 5000, 10000]:
        L_t, D_t, M_t, wr_t, N_t, Lval_t, dim_t, ns_t = decompose_L_and_D(lam_sq_t)

        evals_t, evecs_t = np.linalg.eigh(L_t)
        pos_mask_t = evals_t > 1e-10
        n_pos_t = np.sum(pos_mask_t)

        # Count survivors
        n_surv = 0
        surv_hf = 0
        surv_n0 = 0
        for idx in np.where(pos_mask_t)[0]:
            v = evecs_t[:, idx]
            hf = np.sum(D_t * v**2)
            lam_k = evals_t[idx]
            if hf >= 0 or lam_k + hf > 0:
                n_surv += 1
                surv_hf = np.sum(wr_t * v**2)
                surv_n0 = v[N_t]**2

        print(f'  {lam_sq_t:>8d} {n_pos_t:>8d} {n_surv:>8d} '
              f'{surv_hf:>+14.6f} {surv_n0:>8.4f}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 79d VERDICT')
    print('=' * 76)
    print()
    print('  THE MECHANISM:')
    print('  1. L_off (off-diagonal) has ~50 positive eigenvalues')
    print('  2. diag(D_full) has wr(0) = +4.16 at n=0 and wr(n) -> -inf at large n')
    print('  3. Eigenvectors at large |n| get killed (HF derivative < -lambda)')
    print('  4. The ONE eigenvector at n=0 gets BOOSTED (HF derivative > 0)')
    print('  5. Only the n=0 mode survives')
    print()
    print('  The proof requires showing:')
    print('  (A) L_off has a dominant n=0 eigenvector (from the explicit formula)')
    print('  (B) wr(0) > 0 (from Gamma: C(L) > 0 for all L)')
    print('  (C) For all other positive eigenvectors: <log|n|> > C + lambda + <prime>')
    print('  (C) is the hard part — it needs the eigenvector structure of L_off.')
    print()


if __name__ == '__main__':
    run()
