"""
Session 30 iteration 7: The secular equation in the signal subspace.

QW|signal = W02|signal - diag(mu_k) is a 26x26 matrix.
Its min eigenvalue determines the signal contribution to eps_0.

Track: mu_k (signal M eigenvalues), s1/s2 (W02 eigenvalues),
and min(QW|signal) as functions of L = log(lam^2).

The secular equation: rank-2 perturbation of a diagonal.
How do its roots move with L?
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
    print("SECULAR EQUATION: SIGNAL EIGENVALUES vs L")
    print("=" * 70)

    # ================================================================
    # PART 1: Track ALL signal eigenvalues + W02 eigenvalues vs L
    # ================================================================
    print("\nPART 1: SIGNAL SPECTRUM vs LAMBDA")
    print("-" * 70)

    results = []

    for lam_sq in [14, 30, 50, 100, 200, 500, 1000, 2000]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N+1
        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N)

        evals_m, evecs_m = np.linalg.eigh(M)
        evals_w02 = np.linalg.eigvalsh(W02)
        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        signal_idx = np.where(abs_evals >= threshold)[0]
        P_sig = evecs_m[:, signal_idx]

        # Signal eigenvalues of M (sorted)
        mu_signal = evals_m[signal_idx]
        mu_sorted = np.sort(mu_signal)

        # W02 eigenvalues
        w02_nz = evals_w02[np.abs(evals_w02) > 1e-10]
        s_even = max(w02_nz)
        s_odd = min(w02_nz)

        # QW restricted to signal
        QW_sig = P_sig.T @ QW @ P_sig
        evals_qw_sig = np.linalg.eigvalsh(QW_sig)

        # Full eps_0
        eps_0 = np.linalg.eigvalsh(QW)[0]

        dt = time.time() - t0
        n_sig = len(signal_idx)

        results.append({
            'lam_sq': lam_sq, 'L': L_f, 'n_sig': n_sig,
            's_even': s_even, 's_odd': s_odd,
            'mu_max': mu_sorted[-1], 'mu_min': mu_sorted[0],
            'mu_2nd': mu_sorted[-2] if n_sig > 1 else 0,
            'min_qw_sig': evals_qw_sig[0], 'eps_0': eps_0,
            'mu_signal': mu_sorted
        })

        print(f"lam^2={lam_sq:>5} L={L_f:.3f} n_sig={n_sig:>3}: "
              f"s_even={s_even:.2f} s_odd={s_odd:.2f} "
              f"mu_max={mu_sorted[-1]:.2f} mu_min={mu_sorted[0]:.2f} "
              f"min(QW|sig)={evals_qw_sig[0]:.4e} eps_0={eps_0:.4e} ({dt:.0f}s)")

    # ================================================================
    # PART 2: How do s_even, s_odd, mu_max scale with L?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: SCALING WITH L")
    print("-" * 70)

    Ls = np.array([r['L'] for r in results])
    s_evens = np.array([r['s_even'] for r in results])
    s_odds = np.array([r['s_odd'] for r in results])
    mu_maxs = np.array([r['mu_max'] for r in results])

    # Fit: s_even ~ a * L^b
    log_L = np.log(Ls)
    log_se = np.log(s_evens)
    b_se, log_a_se = np.polyfit(log_L, log_se, 1)

    log_so = np.log(np.abs(s_odds))
    b_so, log_a_so = np.polyfit(log_L, log_so, 1)

    log_mm = np.log(mu_maxs)
    b_mm, log_a_mm = np.polyfit(log_L, log_mm, 1)

    print(f"  s_even ~ {np.exp(log_a_se):.2f} * L^{b_se:.3f}")
    print(f"  |s_odd| ~ {np.exp(log_a_so):.2f} * L^{b_so:.3f}")
    print(f"  mu_max ~ {np.exp(log_a_mm):.2f} * L^{b_mm:.3f}")

    # The KEY ratio: (s_even - mu_max) / s_even
    print(f"\n  {'L':>6} {'s_even':>10} {'mu_max':>10} {'s-mu':>10} {'(s-mu)/s':>10}")
    print("  " + "-" * 45)
    for r in results:
        diff = r['s_even'] - r['mu_max']
        ratio = diff / r['s_even']
        print(f"  {r['L']:>6.3f} {r['s_even']:>10.4f} {r['mu_max']:>10.4f} "
              f"{diff:>10.6f} {ratio:>10.6f}")

    # ================================================================
    # PART 3: The min eigenvalue of QW|signal — what controls it?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: min(QW|signal) vs L")
    print("-" * 70)

    min_qws = np.array([r['min_qw_sig'] for r in results])
    eps0s = np.array([r['eps_0'] for r in results])

    print(f"  {'L':>6} {'min(QW|sig)':>12} {'eps_0':>12} {'ratio':>8} {'min*L':>12}")
    print("  " + "-" * 50)
    for r in results:
        ratio = r['min_qw_sig'] / r['eps_0'] if r['eps_0'] > 0 else 0
        print(f"  {r['L']:>6.3f} {r['min_qw_sig']:>12.4e} {r['eps_0']:>12.4e} "
              f"{ratio:>8.1f} {r['min_qw_sig']*r['L']:>12.4e}")

    # Fit min(QW|signal) vs L
    log_mqs = np.log(min_qws[min_qws > 0])
    Ls_valid = Ls[min_qws > 0]
    if len(log_mqs) > 3:
        # Try power law: min ~ L^gamma
        gamma, log_A = np.polyfit(np.log(Ls_valid), log_mqs, 1)
        print(f"\n  Power law fit: min(QW|sig) ~ {np.exp(log_A):.2e} * L^({gamma:.4f})")

        # Try: min ~ C / L
        print(f"  If min ~ C/L: C = {np.mean(min_qws[min_qws>0] * Ls[min_qws>0]):.4e}")

    # ================================================================
    # PART 4: The secular equation roots — how many are small?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: QW|SIGNAL EIGENVALUE SPECTRUM")
    print("-" * 70)

    for r in results:
        if r['lam_sq'] in [50, 200, 1000]:
            lam_sq = r['lam_sq']
            L_f = r['L']
            N = round(8 * L_f)

            W02, M, QW = build_all(lam_sq, N)
            evals_m, evecs_m = np.linalg.eigh(M)
            abs_evals = np.abs(evals_m)
            threshold = np.max(abs_evals) * 1e-4
            signal_idx = np.where(abs_evals >= threshold)[0]
            P_sig = evecs_m[:, signal_idx]
            QW_sig = P_sig.T @ QW @ P_sig
            evals_qw_sig = np.sort(np.linalg.eigvalsh(QW_sig))

            print(f"\n  lam^2={lam_sq}: QW|signal eigenvalues:")
            for i in range(min(10, len(evals_qw_sig))):
                print(f"    [{i+1}] {evals_qw_sig[i]:.6e}")
            print(f"    ...")
            print(f"    [{len(evals_qw_sig)}] {evals_qw_sig[-1]:.6e}")

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print("=" * 70)
