"""
SESSION 32 — CONNES Q_W: CRITICALITY INSIGHT ATTACK

Key insight from Session 31: RH is a critical phenomenon (Lambda=0).
The system is at the exact phase transition — no safety margin.

This means for Connes Q_W:
  eps_0 → 0+ as lambda → infinity (consistent with Lambda=0 criticality)
  The RATE of vanishing encodes the criticality structure

THE ATTACK:
1. Map eps_0(lambda) scaling law precisely: is it 1/lambda? 1/log(lambda)? 1/lambda^alpha?
2. Decompose: does the cancellation factor grow at the SAME rate as Signal or Null?
   If cancel ~ Signal, then eps_0 ~ Signal/cancel ~ O(1). CONTRADICTION.
   So the growth must be FASTER than Signal.
3. Connect to GUE: Tracy-Widom edge statistics predict the smallest eigenvalue of
   certain random matrices scales as N^{-2/3}. Does eps_0 follow this?
4. Critical exponent: If eps_0 ~ C * lambda^{-alpha}, what is alpha?
   This alpha IS the critical exponent of the RH phase transition.
5. NEW: Test if eps_0 * D_null^{2/3} converges — this would mean eps_0 follows
   Tracy-Widom scaling with respect to null space dimension.

If we can identify the scaling law, we can:
- Compare with Connes 2026 finite approximation theorem
- Use GUE universality to argue the scaling is forced
- Bridge to the Rodgers-Tao barrier analysis
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


def analyze_scaling(lam_sq_values, N_factor=8):
    """Comprehensive scaling analysis of eps_0 and its decomposition."""
    results = []

    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        N = max(21, round(N_factor * L_f))
        dim = 2 * N + 1

        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N)

        # Full eigensystem
        evals_qw, evecs_qw = np.linalg.eigh(QW)
        evals_m, evecs_m = np.linalg.eigh(M)

        xi_0 = evecs_qw[:, 0]
        eps_0 = evals_qw[0]
        eps_1 = evals_qw[1]
        spectral_gap = eps_1 - eps_0

        if eps_0 < 0:
            print(f"  lam^2={lam_sq}: NEGATIVE eps_0 = {eps_0:.4e}, skipping")
            continue

        # Signal/null decomposition
        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        signal_idx = np.where(abs_evals >= threshold)[0]
        null_idx = np.where(abs_evals < threshold)[0]
        P_signal = evecs_m[:, signal_idx]
        P_null = evecs_m[:, null_idx]
        D_null = len(null_idx)
        D_signal = len(signal_idx)

        # Project xi_0
        xi_s = P_signal @ (P_signal.T @ xi_0)
        xi_n = P_null @ (P_null.T @ xi_0)
        alpha_sq = np.linalg.norm(xi_s)**2
        beta_sq = np.linalg.norm(xi_n)**2

        # Three-way decomposition
        term_signal = xi_s @ QW @ xi_s
        term_null = xi_n @ QW @ xi_n
        term_cross = 2 * xi_s @ QW @ xi_n
        cancel_factor = max(abs(term_signal), abs(term_null), abs(term_cross)) / max(eps_0, 1e-30)

        # Tracy-Widom test: eps_0 * D_null^{2/3}
        tw_scaled = eps_0 * D_null**(2/3) if D_null > 0 else 0

        # M spectral gap (between signal eigenvalues)
        m_signal_evals = sorted(abs_evals[signal_idx])
        m_spectral_gap = m_signal_evals[1] - m_signal_evals[0] if len(m_signal_evals) > 1 else 0

        # W02 restricted to null space
        W02_null = P_null.T @ W02 @ P_null
        w02_null_evals = np.linalg.eigvalsh(W02_null)

        elapsed = time.time() - t0

        r = {
            'lam_sq': lam_sq,
            'L': L_f,
            'N': N,
            'dim': dim,
            'D_null': D_null,
            'D_signal': D_signal,
            'eps_0': float(eps_0),
            'eps_1': float(eps_1),
            'spectral_gap': float(spectral_gap),
            'alpha_sq': float(alpha_sq),
            'beta_sq': float(beta_sq),
            'term_signal': float(term_signal),
            'term_null': float(term_null),
            'term_cross': float(term_cross),
            'cancel_factor': float(cancel_factor),
            'tw_scaled': float(tw_scaled),
            'm_spectral_gap': float(m_spectral_gap),
            'w02_null_min': float(w02_null_evals[0]),
            'w02_null_max': float(w02_null_evals[-1]),
            'elapsed': elapsed
        }
        results.append(r)

        print(f"  lam^2={lam_sq:>6}: eps_0={eps_0:.4e}  D_null={D_null:>3}  "
              f"cancel={cancel_factor:.0f}  TW={tw_scaled:.4e}  ({elapsed:.1f}s)")

    return results


def fit_power_law(x, y):
    """Fit y = C * x^alpha using log-log regression."""
    valid = [(xi, yi) for xi, yi in zip(x, y) if yi > 0 and xi > 0]
    if len(valid) < 2:
        return 0, 0, 0
    x_log = np.log([v[0] for v in valid])
    y_log = np.log([v[1] for v in valid])
    alpha, log_C = np.polyfit(x_log, y_log, 1)
    C = np.exp(log_C)
    residuals = y_log - (alpha * x_log + log_C)
    r_squared = 1 - np.var(residuals) / np.var(y_log) if np.var(y_log) > 0 else 0
    return alpha, C, r_squared


if __name__ == "__main__":
    print("CONNES Q_W — CRITICALITY ATTACK")
    print("=" * 75)
    print("Testing: does eps_0 scaling reveal the RH critical exponent?")
    print()

    # ================================================================
    # PART 1: eps_0 scaling with lambda
    # ================================================================
    print("PART 1: eps_0 vs lambda (fixed N/L ratio = 8)")
    print("-" * 75)

    lam_sq_values = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    results = analyze_scaling(lam_sq_values)

    if len(results) >= 3:
        lambdas = [r['lam_sq'] for r in results]
        eps0s = [r['eps_0'] for r in results]
        cancels = [r['cancel_factor'] for r in results]
        D_nulls = [r['D_null'] for r in results]
        tw_scaleds = [r['tw_scaled'] for r in results]
        Ls = [r['L'] for r in results]

        # Fit eps_0 vs lambda
        alpha_lam, C_lam, r2_lam = fit_power_law(lambdas, eps0s)
        print(f"\n  eps_0 ~ {C_lam:.4e} * lambda^({alpha_lam:.3f})  [R2={r2_lam:.4f}]")

        # Fit eps_0 vs L = log(lambda)
        alpha_L, C_L, r2_L = fit_power_law(Ls, eps0s)
        print(f"  eps_0 ~ {C_L:.4e} * L^({alpha_L:.3f})  [R2={r2_L:.4f}]")

        # Fit cancel factor vs lambda
        alpha_c, C_c, r2_c = fit_power_law(lambdas, cancels)
        print(f"  cancel ~ {C_c:.4e} * lambda^({alpha_c:.3f})  [R2={r2_c:.4f}]")

        # Fit cancel factor vs D_null
        alpha_d, C_d, r2_d = fit_power_law(D_nulls, cancels)
        print(f"  cancel ~ {C_d:.4e} * D_null^({alpha_d:.3f})  [R2={r2_d:.4f}]")

        # Tracy-Widom test: does eps_0 * D_null^{2/3} converge?
        print(f"\n  Tracy-Widom test: eps_0 * D_null^(2/3)")
        for r in results:
            print(f"    lam^2={r['lam_sq']:>5}: TW_scaled = {r['tw_scaled']:.6e}  "
                  f"(D_null={r['D_null']})")
        tw_alpha, tw_C, tw_r2 = fit_power_law(D_nulls, tw_scaleds)
        if abs(tw_alpha) < 0.1 and tw_r2 > 0.5:
            print(f"  *** TW_scaled ~ CONSTANT (alpha={tw_alpha:.3f}) => "
                  f"eps_0 ~ D_null^(-2/3) [Tracy-Widom confirmed!] ***")
        else:
            print(f"  TW_scaled trend: alpha={tw_alpha:.3f}, R²={tw_r2:.4f}")

    # ================================================================
    # PART 2: N-dependence at fixed lambda
    # ================================================================
    print("\n\nPART 2: eps_0 vs N at fixed lambda (convergence test)")
    print("-" * 75)
    print("If eps_0(N) converges as N->inf at fixed lambda, the limit IS the true eps_0.")
    print()

    for lam_sq in [200, 1000]:
        L_f = np.log(lam_sq)
        N_base = round(4 * L_f)
        N_values = [N_base, round(6*L_f), round(8*L_f), round(10*L_f), round(12*L_f)]
        N_values = sorted(set(max(15, n) for n in N_values))

        print(f"  lam^2={lam_sq} (L={L_f:.2f})")
        prev_eps = None
        for N in N_values:
            dim = 2*N+1
            W02, M, QW = build_all(lam_sq, N)
            evals = np.linalg.eigvalsh(QW)
            eps_0 = evals[0]
            delta = f"  d={eps_0-prev_eps:+.2e}" if prev_eps is not None else ""
            print(f"    N={N:>3} (dim={dim:>4}): eps_0 = {eps_0:.8e}{delta}")
            prev_eps = eps_0
        print()

    # ================================================================
    # PART 3: Cross-term mechanism deep dive
    # ================================================================
    print("\nPART 3: CROSS-TERM STRUCTURE")
    print("-" * 75)
    print("The cross term 2<xi_s|QW|xi_n> = 2<xi_s|W02|xi_n> - 2<xi_s|M|xi_n>")
    print("M term should be ~0 (xi_n in null(M)). W02 term is the killer.")
    print()

    for lam_sq in [200, 1000, 5000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        dim = 2*N+1

        W02, M, QW = build_all(lam_sq, N)
        evals_qw, evecs_qw = np.linalg.eigh(QW)
        evals_m, evecs_m = np.linalg.eigh(M)

        xi_0 = evecs_qw[:, 0]
        eps_0 = evals_qw[0]

        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        signal_idx = np.where(abs_evals >= threshold)[0]
        null_idx = np.where(abs_evals < threshold)[0]
        P_signal = evecs_m[:, signal_idx]
        P_null = evecs_m[:, null_idx]

        xi_s = P_signal @ (P_signal.T @ xi_0)
        xi_n = P_null @ (P_null.T @ xi_0)

        # Cross term from W02 vs M
        cross_w02 = 2 * xi_s @ W02 @ xi_n
        cross_M = 2 * xi_s @ M @ xi_n  # should be ~0

        # W02 has rank 2. Decompose into its two eigenvectors
        w02_evals, w02_evecs = np.linalg.eigh(W02)
        top2_idx = np.argsort(np.abs(w02_evals))[-2:]
        u1, u2 = w02_evecs[:, top2_idx[0]], w02_evecs[:, top2_idx[1]]
        lam1, lam2 = w02_evals[top2_idx[0]], w02_evals[top2_idx[1]]

        # How much of u1, u2 overlaps with signal vs null space?
        u1_s = np.linalg.norm(P_signal.T @ u1)**2
        u1_n = np.linalg.norm(P_null.T @ u1)**2
        u2_s = np.linalg.norm(P_signal.T @ u2)**2
        u2_n = np.linalg.norm(P_null.T @ u2)**2

        # Critical ratio: the fraction of W02's dominant eigenvectors in null space
        null_capture = (u1_n * abs(lam1) + u2_n * abs(lam2)) / (abs(lam1) + abs(lam2))

        print(f"  lam^2={lam_sq}: eps_0={eps_0:.4e}")
        print(f"    Cross(W02) = {cross_w02:+.4e}  Cross(M) = {cross_M:+.4e}")
        print(f"    W02 rank-2: evals = {lam1:.4e}, {lam2:.4e}")
        print(f"    u1 overlap: signal={u1_s:.4f} null={u1_n:.4f}")
        print(f"    u2 overlap: signal={u2_s:.4f} null={u2_n:.4f}")
        print(f"    Null capture of W02 = {null_capture:.4f}")
        print()

    # ================================================================
    # PART 4: Spectral gap ratio
    # ================================================================
    print("\nPART 4: SPECTRAL GAP RATIO (eps_1/eps_0)")
    print("-" * 75)
    print("At criticality, gap ratio should diverge (critical slowing down).")
    print()

    for r in results:
        ratio = r['eps_1'] / r['eps_0'] if r['eps_0'] > 0 else float('inf')
        print(f"  lam^2={r['lam_sq']:>5}: eps_0={r['eps_0']:.4e}  "
              f"eps_1={r['eps_1']:.4e}  ratio={ratio:.1f}")

    if len(results) >= 3:
        ratios = [r['eps_1']/r['eps_0'] for r in results if r['eps_0'] > 0]
        alpha_r, C_r, r2_r = fit_power_law(lambdas[:len(ratios)], ratios)
        print(f"\n  gap_ratio ~ {C_r:.2f} * lambda^({alpha_r:.3f})  [R2={r2_r:.4f}]")
        if alpha_r > 0.05:
            print(f"  *** Gap ratio DIVERGES => critical slowing down confirmed ***")

    # ================================================================
    # SAVE
    # ================================================================
    output = {
        'scaling_results': results,
        'fits': {}
    }
    if len(results) >= 3:
        output['fits'] = {
            'eps0_vs_lambda': {'alpha': alpha_lam, 'C': C_lam, 'R2': r2_lam},
            'eps0_vs_L': {'alpha': alpha_L, 'C': C_L, 'R2': r2_L},
            'cancel_vs_lambda': {'alpha': alpha_c, 'C': C_c, 'R2': r2_c},
            'cancel_vs_D_null': {'alpha': alpha_d, 'C': C_d, 'R2': r2_d},
        }

    with open('connes_criticality.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to connes_criticality.json")
