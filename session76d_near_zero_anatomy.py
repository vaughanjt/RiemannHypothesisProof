"""
SESSION 76d -- ANATOMY OF THE 73 CONVERGED NEAR-ZERO EIGENVALUES

76c found: 73 eigenvalues in [1e-8, 1e-4] are CONVERGED (stable from N=80+).
These are the TRUE spectral mirror -- the subspace where L_pure and D cancel.

Questions:
  1. Does the count 73 change with lam_sq?
  2. At different N, are these the SAME eigenvalues (tracking) or different ones?
  3. What's the eigenvector structure? Parity? Frequency content?
  4. Are they related to prolate spheroidal functions?
  5. How does the near-zero count depend on lam_sq?
  6. Is 73 = dim - something simple?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast


def count_by_band(evals):
    """Count eigenvalues in standard bands."""
    ae = np.abs(evals)
    bands = {
        '[10,inf)': np.sum(ae >= 10),
        '[1,10)': np.sum((ae >= 1) & (ae < 10)),
        '[0.1,1)': np.sum((ae >= 0.1) & (ae < 1)),
        '[0.01,0.1)': np.sum((ae >= 0.01) & (ae < 0.1)),
        '[1e-3,0.01)': np.sum((ae >= 1e-3) & (ae < 0.01)),
        '[1e-4,1e-3)': np.sum((ae >= 1e-4) & (ae < 1e-3)),
        '[1e-5,1e-4)': np.sum((ae >= 1e-5) & (ae < 1e-4)),
        '[1e-8,1e-5)': np.sum((ae >= 1e-8) & (ae < 1e-5)),
        '[0,1e-8)': np.sum(ae < 1e-8),
    }
    return bands


def run():
    print()
    print('#' * 76)
    print('  SESSION 76d -- THE 73 CONVERGED NEAR-ZERO EIGENVALUES')
    print('#' * 76)

    # ======================================================================
    # TEST 1: Does the near-zero count depend on lam_sq?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 1: NEAR-ZERO COUNT vs LAM_SQ (at large N=120)')
    print(f'{"="*76}\n')

    # Use N=120 (large enough for convergence) at various lam_sq
    N_big = 120
    print(f'  {"lam^2":>8} {"L":>8} {"[1e-5,1e-4)":>12} {"[1e-8,1e-5)":>12} '
          f'{"total_nz":>10} {"dim":>5}')
    print('  ' + '-' * 62)

    nz_by_lam = {}
    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        try:
            _, M, _ = build_all_fast(lam_sq, N_big)
            evals = np.linalg.eigvalsh(M)
            bands = count_by_band(evals)
            total_nz = bands['[1e-5,1e-4)'] + bands['[1e-8,1e-5)'] + bands['[0,1e-8)']
            dim = 2 * N_big + 1

            nz_by_lam[lam_sq] = evals
            print(f'  {lam_sq:>8d} {np.log(lam_sq):>8.3f} {bands["[1e-5,1e-4)"]:>12d} '
                  f'{bands["[1e-8,1e-5)"]:>12d} {total_nz:>10d} {dim:>5d}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 2: Eigenvalue tracking across N at fixed lam_sq
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 2: DO THE NEAR-ZERO EIGENVALUES TRACK ACROSS N?')
    print(f'{"="*76}\n')

    lam_sq = 1000
    N_vals = [60, 80, 100, 120, 150]
    evals_by_N = {}

    for N in N_vals:
        _, M, _ = build_all_fast(lam_sq, N)
        evals = np.linalg.eigvalsh(M)
        evals_by_N[N] = evals

    # The near-zero eigenvalues at N=150
    ref_evals = evals_by_N[150]
    ref_nz = ref_evals[(np.abs(ref_evals) >= 1e-8) & (np.abs(ref_evals) < 1e-4)]
    ref_nz_sorted = np.sort(ref_nz)

    print(f'  Reference: N=150, {len(ref_nz)} near-zero eigenvalues in [1e-8, 1e-4)')
    print(f'  Comparing to smaller N:')
    print()

    for N in N_vals[:-1]:
        evals_N = evals_by_N[N]
        nz_N = evals_N[(np.abs(evals_N) >= 1e-8) & (np.abs(evals_N) < 1e-4)]
        nz_N_sorted = np.sort(nz_N)

        # Match by nearest eigenvalue
        if len(nz_N) > 0 and len(ref_nz) > 0:
            # For each nz eigenvalue at N, find closest at N=150
            matched = 0
            max_diff = 0
            for e in nz_N_sorted:
                closest = ref_nz_sorted[np.argmin(np.abs(ref_nz_sorted - e))]
                diff = abs(e - closest)
                if diff < 1e-4:
                    matched += 1
                max_diff = max(max_diff, diff)

            print(f'  N={N:>3d}: {len(nz_N):>3d} near-zero, '
                  f'{matched:>3d} matched to N=150 (max_diff={max_diff:.4e})')
        else:
            print(f'  N={N:>3d}: {len(nz_N):>3d} near-zero')
    sys.stdout.flush()

    # ======================================================================
    # TEST 3: Full band distribution at large N across lam_sq
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 3: FULL BAND DISTRIBUTION (N=120) vs LAM_SQ')
    print(f'{"="*76}\n')

    print(f'  {"lam^2":>8} {"[10,inf)":>8} {"[1,10)":>7} {"[.1,1)":>7} '
          f'{"[.01,.1)":>8} {"[1e-3,.01)":>10} {"[1e-4,1e-3)":>12} '
          f'{"[1e-5,1e-4)":>12} {"<1e-5":>7}')
    print('  ' + '-' * 90)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        try:
            _, M, _ = build_all_fast(lam_sq, N_big)
            evals = np.linalg.eigvalsh(M)
            b = count_by_band(evals)
            below = b['[1e-8,1e-5)'] + b['[0,1e-8)']
            print(f'  {lam_sq:>8d} {b["[10,inf)"] :>8d} {b["[1,10)"]:>7d} '
                  f'{b["[0.1,1)"]:>7d} {b["[0.01,0.1)"]:>8d} '
                  f'{b["[1e-3,0.01)"]:>10d} {b["[1e-4,1e-3)"]:>12d} '
                  f'{b["[1e-5,1e-4)"]:>12d} {below:>7d}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 4: Eigenvector structure of the near-zero modes
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 4: NEAR-ZERO EIGENVECTOR STRUCTURE (lam^2=1000, N=120)')
    print(f'{"="*76}\n')

    lam_sq = 1000
    N = 120
    _, M, _ = build_all_fast(lam_sq, N)
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    evals, evecs = np.linalg.eigh(M)

    # Select near-zero eigenvalues
    nz_mask = (np.abs(evals) >= 1e-8) & (np.abs(evals) < 1e-4)
    nz_evals = evals[nz_mask]
    nz_evecs = evecs[:, nz_mask]
    n_nz = nz_evecs.shape[1]

    print(f'  {n_nz} near-zero eigenvectors')
    print()

    # Parity analysis
    parities = []
    mean_freqs = []
    participation_ratios = []

    for j in range(n_nz):
        v = nz_evecs[:, j]
        # Parity
        even_e = v[N]**2 + sum((v[N+k]+v[N-k])**2/2 for k in range(1, N+1))
        parities.append(even_e)
        # Mean |n|
        mean_freqs.append(np.sum(np.abs(ns) * v**2))
        # Participation ratio
        participation_ratios.append(1.0 / np.sum(v**4))

    parities = np.array(parities)
    mean_freqs = np.array(mean_freqs)
    participation_ratios = np.array(participation_ratios)

    n_even = np.sum(parities > 0.9)
    n_odd = np.sum(parities < 0.1)
    n_mixed = n_nz - n_even - n_odd

    print(f'  Parity: {n_even} even, {n_odd} odd, {n_mixed} mixed')
    print(f'  Mean |n|: min={mean_freqs.min():.1f}, max={mean_freqs.max():.1f}, '
          f'mean={mean_freqs.mean():.1f} (N={N})')
    print(f'  Participation ratio: min={participation_ratios.min():.1f}, '
          f'max={participation_ratios.max():.1f}, '
          f'mean={participation_ratios.mean():.1f} (dim={dim})')

    # Are they HIGH frequency? (concentrated at |n| > N/2)
    high_freq_weight = []
    for j in range(n_nz):
        v = nz_evecs[:, j]
        hf = np.sum(v[np.abs(ns) > N/2]**2)
        high_freq_weight.append(hf)
    hf_arr = np.array(high_freq_weight)
    print(f'  Weight at |n| > N/2: min={hf_arr.min():.4f}, max={hf_arr.max():.4f}, '
          f'mean={hf_arr.mean():.4f}')
    print(f'  Weight at |n| > 3N/4: ', end='')
    vhf = []
    for j in range(n_nz):
        v = nz_evecs[:, j]
        vhf.append(np.sum(v[np.abs(ns) > 3*N/4]**2))
    vhf_arr = np.array(vhf)
    print(f'min={vhf_arr.min():.4f}, max={vhf_arr.max():.4f}, mean={vhf_arr.mean():.4f}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 5: Compare near-zero subspace to prolate null space
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 5: NEAR-ZERO vs PROLATE NULL SPACE')
    print(f'{"="*76}\n')

    # Build Slepian prolate concentration matrix for various c values
    # and compare its null space (eigenvalues < 0.5) to M's near-zero space

    for c in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        C_mat = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    C_mat[i, j] = c
                else:
                    diff = ns[i] - ns[j]
                    C_mat[i, j] = np.sin(np.pi * diff * c) / (np.pi * diff)

        prol_evals, prol_evecs = np.linalg.eigh(C_mat)

        # Prolate "null space" = eigenvectors with eigenvalue < 0.01
        prol_null_mask = prol_evals < 0.01
        prol_null = prol_evecs[:, prol_null_mask]
        n_prol_null = prol_null.shape[1]

        # Overlap between M's near-zero and prolate null
        if n_prol_null > 0 and n_nz > 0:
            sv = np.linalg.svd(nz_evecs.T @ prol_null, compute_uv=False)
            overlap = np.sum(sv**2) / min(n_nz, n_prol_null)
        else:
            overlap = 0

        n_prol_signal = np.sum(prol_evals > 0.5)
        print(f'  c={c:.3f}: prolate signal={n_prol_signal}, prolate null={n_prol_null}, '
              f'overlap with M near-zero: {overlap:.6f}')
    sys.stdout.flush()

    # ======================================================================
    # TEST 6: Relationship between near-zero count and lam_sq at fixed N
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  TEST 6: NEAR-ZERO COUNT vs LAM_SQ (multiple thresholds)')
    print(f'{"="*76}\n')

    N = 120
    print(f'  N={N} fixed:')
    print(f'  {"lam^2":>8} {"<1e-3":>7} {"<1e-4":>7} {"<1e-5":>7} {"<1e-6":>7} '
          f'{"<1e-7":>7} {"<1e-8":>7}')
    print('  ' + '-' * 50)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
        try:
            _, M, _ = build_all_fast(lam_sq, N)
            evals = np.linalg.eigvalsh(M)
            ae = np.abs(evals)
            counts = [np.sum(ae < t) for t in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]
            print(f'  {lam_sq:>8d} ' + ' '.join(f'{c:>7d}' for c in counts))
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 76d VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
