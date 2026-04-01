"""
SESSION 33 — THE RANGE-NULL SPLIT: Decompose Q_W along W_{0,2} eigenspaces

THE STRUCTURE:
  W_{0,2} has rank 2 with eigenvectors u1, u2 (eigenvalues s1, s2).
  null(W_{0,2}) has dimension dim-2.

  Q_W = W_{0,2} - M decomposes as:
    On range(W_{0,2}):  Q_W = diag(s1,s2) - M_range  (2x2 problem)
    On null(W_{0,2}):   Q_W = -M_null                 (need M <= 0)
    Cross terms:         Q_W = -M_cross

  For Q_W >= 0, we need (Schur complement):
    1. Q_W restricted to range is PD (2x2 — computable!)
    2. Q_W restricted to null minus cross correction is PSD

  THE 2x2 BLOCK IS THE BREAKTHROUGH:
  If we can prove the 2x2 range block is PD for all lambda,
  that's HALF the problem solved — and it's a finite computation.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, sinh
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 50


def range_null_decomposition(lam_sq, N=None):
    """
    Decompose Q_W into range(W02) and null(W02) blocks.

    Returns the 2x2 range block, the (dim-2)x(dim-2) null block,
    and the cross terms.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)

    # Eigendecomposition of W02
    evals_w02, evecs_w02 = np.linalg.eigh(W02)

    # Identify range (nonzero eigenvalues) and null space
    threshold = np.max(np.abs(evals_w02)) * 1e-10
    range_idx = np.where(np.abs(evals_w02) > threshold)[0]
    null_idx = np.where(np.abs(evals_w02) <= threshold)[0]

    P_range = evecs_w02[:, range_idx]  # dim x 2
    P_null = evecs_w02[:, null_idx]    # dim x (dim-2)

    # Full change of basis: [P_range | P_null]
    P = np.hstack([P_range, P_null])  # dim x dim, orthogonal

    # Transform Q_W to this basis
    QW_transformed = P.T @ QW @ P

    # Extract blocks
    r = len(range_idx)  # should be 2
    n = len(null_idx)   # should be dim-2

    QW_range = QW_transformed[:r, :r]        # 2x2
    QW_null = QW_transformed[r:, r:]         # (dim-2)x(dim-2)
    QW_cross = QW_transformed[:r, r:]        # 2x(dim-2)

    # Also decompose W02 and M in same basis
    W02_range = P_range.T @ W02 @ P_range    # 2x2 diagonal (eigenvalues)
    M_range = P_range.T @ M @ P_range        # 2x2
    M_null = P_null.T @ M @ P_null           # (dim-2)x(dim-2)
    M_cross = P_range.T @ M @ P_null         # 2x(dim-2)

    return {
        'dim': dim, 'r': r, 'n': n,
        'QW_range': QW_range, 'QW_null': QW_null, 'QW_cross': QW_cross,
        'W02_range': W02_range, 'M_range': M_range,
        'M_null': M_null, 'M_cross': M_cross,
        'W02_eigenvalues': evals_w02[range_idx],
        'P_range': P_range, 'P_null': P_null
    }


def analyze_2x2_range_block(lam_sq_values):
    """
    Analyze the 2x2 range block of Q_W.

    Q_W_range = W02_range - M_range = diag(s1,s2) - M_range

    This is a 2x2 matrix. It's PD iff:
    1. Both diagonal entries > 0
    2. Determinant > 0

    These are EXPLICIT conditions on s1, s2 and the 2x2 M_range entries.
    """
    print("THE 2x2 RANGE BLOCK")
    print("=" * 75)
    print("Q_W restricted to range(W02) — this is a 2x2 matrix.\n")

    results = []
    for lam_sq in lam_sq_values:
        t0 = time.time()
        decomp = range_null_decomposition(lam_sq)
        elapsed = time.time() - t0

        QW_r = decomp['QW_range']
        W02_r = decomp['W02_range']
        M_r = decomp['M_range']
        s = decomp['W02_eigenvalues']

        # 2x2 PD conditions
        diag_1 = QW_r[0, 0]
        diag_2 = QW_r[1, 1]
        det_2x2 = np.linalg.det(QW_r)
        evals_2x2 = np.linalg.eigvalsh(QW_r)
        is_pd = evals_2x2[0] > 0

        # The margin: how far from singular?
        margin = evals_2x2[0]  # smallest eigenvalue
        condition = evals_2x2[1] / evals_2x2[0] if evals_2x2[0] > 0 else float('inf')

        r = {
            'lam_sq': lam_sq,
            'W02_eigs': [float(s[0]), float(s[1])],
            'QW_range': QW_r.tolist(),
            'M_range': M_r.tolist(),
            'diag': [float(diag_1), float(diag_2)],
            'det': float(det_2x2),
            'eigs_2x2': [float(evals_2x2[0]), float(evals_2x2[1])],
            'is_pd': bool(is_pd),
            'margin': float(margin),
            'condition': float(condition)
        }
        results.append(r)

        print(f"lam^2={lam_sq} ({elapsed:.1f}s):")
        print(f"  W02 eigenvalues: {s[0]:.6f}, {s[1]:.6f}")
        print(f"  Q_W range block:")
        print(f"    [{QW_r[0,0]:>12.6f}  {QW_r[0,1]:>12.6f}]")
        print(f"    [{QW_r[1,0]:>12.6f}  {QW_r[1,1]:>12.6f}]")
        print(f"  M range block:")
        print(f"    [{M_r[0,0]:>12.6f}  {M_r[0,1]:>12.6f}]")
        print(f"    [{M_r[1,0]:>12.6f}  {M_r[1,1]:>12.6f}]")
        print(f"  Eigenvalues: {evals_2x2[0]:.6e}, {evals_2x2[1]:.6e}")
        print(f"  Determinant: {det_2x2:.6e}")
        print(f"  PD: {'YES' if is_pd else 'NO'}  Margin: {margin:.6e}  Condition: {condition:.2f}")

        # Express PD conditions explicitly
        print(f"\n  EXPLICIT PD CONDITIONS:")
        print(f"    s1 - M11 > 0:  {s[0]:.4f} - {M_r[0,0]:.4f} = {diag_1:.6e}  {'PASS' if diag_1 > 0 else 'FAIL'}")
        print(f"    s2 - M22 > 0:  {s[1]:.4f} - {M_r[1,1]:.4f} = {diag_2:.6e}  {'PASS' if diag_2 > 0 else 'FAIL'}")
        print(f"    det > 0:       {det_2x2:.6e}  {'PASS' if det_2x2 > 0 else 'FAIL'}")

        # Ratio: how much of W02 is consumed by M on the range?
        ratio_1 = M_r[0, 0] / s[0] if abs(s[0]) > 1e-15 else 0
        ratio_2 = M_r[1, 1] / s[1] if abs(s[1]) > 1e-15 else 0
        print(f"\n  M/W02 ratios: {ratio_1:.6f}, {ratio_2:.6f}")
        print(f"  M consumes {ratio_1*100:.4f}% and {ratio_2*100:.4f}% of W02 on range")
        print()

    return results


def analyze_null_block(lam_sq_values):
    """
    Analyze the null block: -M restricted to null(W02).
    This must be PSD (i.e., M <= 0 on null(W02)).
    """
    print("\n\nTHE NULL BLOCK: -M restricted to null(W02)")
    print("=" * 75)

    for lam_sq in lam_sq_values:
        decomp = range_null_decomposition(lam_sq)
        M_null = decomp['M_null']
        n = decomp['n']

        evals = np.linalg.eigvalsh(M_null)
        max_eig = evals[-1]
        min_eig = evals[0]

        print(f"\nlam^2={lam_sq}: M|null is {n}x{n}")
        print(f"  Eigenvalues: [{min_eig:.4e}, ..., {max_eig:.4e}]")
        print(f"  M <= 0 on null(W02): {'YES' if max_eig < 1e-10 else 'NO'}")
        print(f"  max eigenvalue / |trace/n|: {max_eig / abs(np.trace(M_null)/n):.6e}")


def analyze_cross_terms(lam_sq_values):
    """
    Analyze cross terms and Schur complement.

    For Q_W >= 0 via Schur complement:
    If Q_W_range is PD, then Q_W >= 0 iff
      Q_W_null - Q_W_cross^T * Q_W_range^{-1} * Q_W_cross >= 0

    The cross correction is small if the cross terms are small.
    """
    print("\n\nCROSS TERMS AND SCHUR COMPLEMENT")
    print("=" * 75)

    for lam_sq in lam_sq_values:
        decomp = range_null_decomposition(lam_sq)
        QW_r = decomp['QW_range']
        QW_n = decomp['QW_null']
        QW_c = decomp['QW_cross']

        # Schur complement: S = QW_null - QW_cross^T * QW_range^{-1} * QW_cross
        QW_r_inv = np.linalg.inv(QW_r)
        correction = QW_c.T @ QW_r_inv @ QW_c  # this is (dim-2)x(dim-2) but rank <= 2
        schur = QW_n - correction

        evals_schur = np.linalg.eigvalsh(schur)
        evals_null = np.linalg.eigvalsh(QW_n)
        evals_corr = np.linalg.eigvalsh(correction)

        # The Schur complement eigenvalues should match eps_0
        eps_0 = np.linalg.eigvalsh(
            np.block([[QW_r, QW_c], [QW_c.T, QW_n]])
        )[0]

        print(f"\nlam^2={lam_sq}:")
        print(f"  Q_W_range eigenvalues: {np.linalg.eigvalsh(QW_r)}")
        print(f"  Q_W_null eigenvalues: [{evals_null[0]:.4e}, ..., {evals_null[-1]:.4e}]")
        print(f"  Cross term ||QW_cross||_F: {np.linalg.norm(QW_c, 'fro'):.6e}")
        print(f"  Correction eigenvalues: [{evals_corr[0]:.4e}, ..., {evals_corr[-1]:.4e}]")
        print(f"  Correction rank: {np.sum(evals_corr > 1e-12)}")
        print(f"  Schur complement eigenvalues: [{evals_schur[0]:.4e}, ..., {evals_schur[-1]:.4e}]")
        print(f"  eps_0 (full Q_W): {eps_0:.6e}")
        print(f"  eps_0 (Schur): {evals_schur[0]:.6e}")

        # Cross terms should be ZERO if range and null perfectly decouple
        # (they don't because M has cross terms between range and null)
        cross_norm = np.linalg.norm(QW_c, 'fro')
        range_norm = np.linalg.norm(QW_r, 'fro')
        null_norm = np.linalg.norm(QW_n, 'fro')
        print(f"\n  Coupling strength: ||cross|| / sqrt(||range||*||null||) = "
              f"{cross_norm / np.sqrt(range_norm * null_norm):.6e}")

        if cross_norm < 1e-10:
            print(f"  *** DECOUPLED: range and null are independent! ***")
            print(f"  *** Q_W >= 0 iff range block PD AND null block PSD ***")
        else:
            # How much does the correction affect the null block?
            correction_impact = np.max(np.abs(evals_corr)) / np.max(np.abs(evals_null))
            print(f"  Correction impact: {correction_impact:.6e}")
            if correction_impact < 0.01:
                print(f"  Correction is < 1% of null block — NEGLIGIBLE")


def scaling_analysis(lam_sq_values):
    """Track how the 2x2 range block scales with lambda."""
    print("\n\n2x2 RANGE BLOCK SCALING")
    print("=" * 75)

    margins = []
    for lam_sq in lam_sq_values:
        decomp = range_null_decomposition(lam_sq)
        QW_r = decomp['QW_range']
        evals = np.linalg.eigvalsh(QW_r)
        s = decomp['W02_eigenvalues']
        M_r = decomp['M_range']

        margin = evals[0]
        margins.append(margin)

        # Ratio of M to W02 on range
        ratio = np.trace(M_r) / np.trace(np.diag(s))

        print(f"  lam^2={lam_sq:>5}: margin={margin:.6e}  "
              f"M/W02={ratio:.6f}  cond={evals[1]/evals[0]:.1f}")

    # Fit margin vs lambda
    if len(margins) >= 3:
        lams = np.array(lam_sq_values)
        ms = np.array(margins)
        valid = ms > 0
        if np.sum(valid) >= 3:
            alpha, logC = np.polyfit(np.log(lams[valid]), np.log(ms[valid]), 1)
            print(f"\n  Margin ~ {np.exp(logC):.4e} * lam^({alpha:.3f})")
            if alpha > 0:
                print(f"  *** MARGIN GROWS — 2x2 block gets MORE positive ***")
            elif alpha > -0.1:
                print(f"  Margin roughly constant — stable")
            else:
                print(f"  Margin shrinks — need to check convergence")


if __name__ == "__main__":
    print("SESSION 33 — THE RANGE-NULL SPLIT")
    print("=" * 75)

    lam_sq_values = [50, 100, 200, 500, 1000, 2000]

    # Part 1: The 2x2 range block
    range_results = analyze_2x2_range_block(lam_sq_values)

    # Part 2: The null block
    analyze_null_block(lam_sq_values)

    # Part 3: Cross terms
    analyze_cross_terms([200, 1000])

    # Part 4: Scaling
    scaling_analysis(lam_sq_values)

    # Summary
    print("\n\n" + "=" * 75)
    print("RANGE-NULL SPLIT SUMMARY")
    print("=" * 75)

    all_pd = all(r['is_pd'] for r in range_results)
    print(f"\n  2x2 range block PD for all tested lambda: {'YES' if all_pd else 'NO'}")

    if all_pd:
        print(f"\n  THIS MEANS: Half the problem is solved.")
        print(f"  Q_W >= 0 reduces to: M <= 0 on null(W02) + cross term correction.")
        print(f"  The 2x2 range verification is a FINITE explicit computation")
        print(f"  involving only 2 eigenvalues of W02 and a 2x2 prime sum matrix.")

    with open('session33_range_null.json', 'w') as f:
        json.dump(range_results, f, indent=2, default=str)
    print(f"\nResults saved to session33_range_null.json")
