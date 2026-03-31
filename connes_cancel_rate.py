"""
Session 30 iteration 6: Cancellation factor vs D (null space dimension).

Fix lambda, vary N. As N grows, D = null(M) grows.
Track: cancellation factor = max(|Signal|,|Null|,|Cross|) / eps_0.

If cancel ~ D^alpha (alpha > 0): eps_0 ~ C/D^alpha -> 0. QED.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 50


if __name__ == "__main__":
    print("CANCELLATION FACTOR vs NULL SPACE DIMENSION D")
    print("=" * 70)

    lam_sq = 200  # fixed lambda

    print(f"\nlam^2={lam_sq} (fixed), varying N")
    print(f"{'N':>4} {'dim':>5} {'D_null':>6} {'eps_0':>12} {'Signal':>12} "
          f"{'Null':>12} {'Cross':>12} {'Cancel':>8}")
    print("-" * 80)

    Ds = []
    cancels = []
    eps0s = []

    for N in [18, 21, 25, 30, 35, 40, 45, 50]:
        dim = 2*N+1
        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N)

        evals_qw, evecs_qw = np.linalg.eigh(QW)
        evals_m, evecs_m = np.linalg.eigh(M)
        xi_0 = evecs_qw[:, 0]
        eps_0 = evals_qw[0]

        if eps_0 < 0:
            print(f"{N:>4} {dim:>5}  --- NEGATIVE eps_0 = {eps_0:.4e} ---")
            continue

        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        signal_idx = np.where(abs_evals >= threshold)[0]
        null_idx = np.where(abs_evals < threshold)[0]
        P_signal = evecs_m[:, signal_idx]
        P_null = evecs_m[:, null_idx]
        D_null = len(null_idx)

        xi_s = P_signal @ (P_signal.T @ xi_0)
        xi_n = P_null @ (P_null.T @ xi_0)

        term_signal = xi_s @ QW @ xi_s
        term_null = xi_n @ QW @ xi_n
        term_cross = 2 * xi_s @ QW @ xi_n

        max_term = max(abs(term_signal), abs(term_null), abs(term_cross))
        cancel = max_term / eps_0 if eps_0 > 1e-20 else 0

        print(f"{N:>4} {dim:>5} {D_null:>6} {eps_0:>12.4e} {term_signal:>12.4e} "
              f"{term_null:>12.4e} {term_cross:>12.4e} {cancel:>8.0f}")

        Ds.append(D_null)
        cancels.append(cancel)
        eps0s.append(eps_0)

    # Fit: cancel ~ D^alpha
    if len(Ds) > 3:
        Ds = np.array(Ds)
        cancels = np.array(cancels)
        eps0s = np.array(eps0s)

        log_D = np.log(Ds)
        log_cancel = np.log(cancels)
        alpha, log_C = np.polyfit(log_D, log_cancel, 1)

        print(f"\nFIT: cancellation ~ {np.exp(log_C):.2f} * D^{alpha:.4f}")
        print(f"  => eps_0 ~ C / D^{alpha:.2f}")

        # Also fit eps_0 vs D directly
        log_eps = np.log(eps0s)
        beta, log_E = np.polyfit(log_D, log_eps, 1)
        print(f"  Direct: eps_0 ~ {np.exp(log_E):.2e} * D^({beta:.4f})")

        print(f"\n  {'D':>6} {'cancel(meas)':>12} {'cancel(fit)':>12} "
              f"{'eps_0(meas)':>12} {'eps_0(fit)':>12}")
        print("  " + "-" * 55)
        for i in range(len(Ds)):
            c_fit = np.exp(log_C) * Ds[i]**alpha
            e_fit = np.exp(log_E) * Ds[i]**beta
            print(f"  {Ds[i]:>6} {cancels[i]:>12.0f} {c_fit:>12.0f} "
                  f"{eps0s[i]:>12.4e} {e_fit:>12.4e}")

    # ================================================================
    # PART 2: Same analysis varying lambda (N=8*L, D grows with L)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: CANCELLATION vs D (varying lambda, N=8*L)")
    print("-" * 70)

    print(f"{'lam^2':>6} {'D_null':>6} {'eps_0':>12} {'Cancel':>8} {'S/N ratio':>10}")
    print("-" * 50)

    Ds2 = []
    cancels2 = []
    eps0s2 = []

    for lam_sq_test in [30, 50, 100, 200, 500, 1000, 2000]:
        L_f = np.log(lam_sq_test)
        N = round(8 * L_f)
        dim = 2*N+1

        W02, M, QW = build_all(lam_sq_test, N)
        evals_qw, evecs_qw = np.linalg.eigh(QW)
        evals_m, evecs_m = np.linalg.eigh(M)
        xi_0 = evecs_qw[:, 0]
        eps_0 = evals_qw[0]

        if eps_0 < 0: continue

        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        null_idx = np.where(abs_evals < threshold)[0]
        signal_idx = np.where(abs_evals >= threshold)[0]
        P_s = evecs_m[:, signal_idx]
        P_n = evecs_m[:, null_idx]
        D_null = len(null_idx)

        xi_s = P_s @ (P_s.T @ xi_0)
        xi_n = P_n @ (P_n.T @ xi_0)

        S = xi_s @ QW @ xi_s
        N_val = xi_n @ QW @ xi_n
        C_val = 2 * xi_s @ QW @ xi_n

        max_term = max(abs(S), abs(N_val), abs(C_val))
        cancel = max_term / eps_0
        sn_ratio = S / N_val if abs(N_val) > 1e-20 else 0

        print(f"{lam_sq_test:>6} {D_null:>6} {eps_0:>12.4e} {cancel:>8.0f} {sn_ratio:>10.6f}")

        Ds2.append(D_null)
        cancels2.append(cancel)
        eps0s2.append(eps_0)

    if len(Ds2) > 3:
        Ds2 = np.array(Ds2)
        cancels2 = np.array(cancels2)
        eps0s2 = np.array(eps0s2)

        log_D2 = np.log(Ds2)
        log_c2 = np.log(cancels2)
        alpha2, log_C2 = np.polyfit(log_D2, log_c2, 1)

        log_e2 = np.log(eps0s2)
        beta2, log_E2 = np.polyfit(log_D2, log_e2, 1)

        print(f"\nFIT (varying lambda):")
        print(f"  cancellation ~ D^{alpha2:.4f}")
        print(f"  eps_0 ~ D^({beta2:.4f})")
        print(f"  If beta < 0: eps_0 -> 0 as D -> inf => RH consistent")

    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)
