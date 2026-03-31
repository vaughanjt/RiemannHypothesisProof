"""
Session 30 iteration 5: EXACT decomposition of eps_0 into signal + null + cross.

For the actual min eigenvector xi_0:
  eps_0 = <xi|QW|xi> = <xi_s|QW|xi_s> + <xi_n|QW|xi_n> + 2*<xi_s|QW|xi_n>

where xi_s = P_signal * xi, xi_n = P_null * xi.

Compute each term EXACTLY. This reveals the mechanism.
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
    print("EXACT eps_0 DECOMPOSITION")
    print("=" * 70)

    for lam_sq in [50, 200, 1000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        dim = 2*N + 1

        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N)

        # Eigensystems
        evals_qw, evecs_qw = np.linalg.eigh(QW)
        evals_m, evecs_m = np.linalg.eigh(M)
        xi_0 = evecs_qw[:, 0]
        eps_0 = evals_qw[0]

        # Signal/null split
        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        signal_idx = np.where(abs_evals >= threshold)[0]
        null_idx = np.where(abs_evals < threshold)[0]
        P_signal = evecs_m[:, signal_idx]
        P_null = evecs_m[:, null_idx]

        # Project xi_0
        xi_s = P_signal @ (P_signal.T @ xi_0)  # signal component
        xi_n = P_null @ (P_null.T @ xi_0)      # null component
        alpha_sq = np.linalg.norm(xi_s)**2
        beta_sq = np.linalg.norm(xi_n)**2

        # Three contributions
        term_signal = xi_s @ QW @ xi_s
        term_null = xi_n @ QW @ xi_n
        term_cross = 2 * xi_s @ QW @ xi_n
        total = term_signal + term_null + term_cross

        # Further decompose cross term into W02 and M parts
        cross_w02 = 2 * xi_s @ W02 @ xi_n
        cross_M = -2 * xi_s @ M @ xi_n  # should be ~0 since M block-diagonal

        # Signal term decomposition
        sig_w02 = xi_s @ W02 @ xi_s
        sig_M = xi_s @ M @ xi_s
        sig_QW = sig_w02 - sig_M  # = term_signal

        # Null term decomposition
        null_w02 = xi_n @ W02 @ xi_n
        null_M = xi_n @ M @ xi_n
        null_QW = null_w02 - null_M  # = term_null

        print(f"\nlam^2={lam_sq}, N={N}, dim={dim} ({time.time()-t0:.0f}s)")
        print(f"  eps_0 = {eps_0:.6e}")
        print(f"  ||xi_s||^2 = {alpha_sq:.6f}, ||xi_n||^2 = {beta_sq:.6f}")
        print(f"\n  === DECOMPOSITION ===")
        print(f"  Signal:  <xi_s|QW|xi_s> = {term_signal:+.6e}")
        print(f"    (W02: {sig_w02:+.6e}, M: {sig_M:+.6e})")
        print(f"  Null:    <xi_n|QW|xi_n> = {term_null:+.6e}")
        print(f"    (W02: {null_w02:+.6e}, M: {null_M:+.6e})")
        print(f"  Cross:  2<xi_s|QW|xi_n> = {term_cross:+.6e}")
        print(f"    (W02: {cross_w02:+.6e}, M: {cross_M:+.6e})")
        print(f"  TOTAL:                    {total:+.6e}")
        print(f"  Check: total - eps_0 =    {total - eps_0:.4e}")

        # The SIGN STRUCTURE tells us the mechanism:
        print(f"\n  SIGN STRUCTURE:")
        signs = []
        for name, val in [("Signal", term_signal), ("Null", term_null), ("Cross", term_cross)]:
            sign = "+" if val > 0 else "-"
            signs.append(f"{name}({sign})")
        print(f"    {' '.join(signs)}")

        # How much cancellation?
        max_term = max(abs(term_signal), abs(term_null), abs(term_cross))
        print(f"  Max |term| = {max_term:.4e}, eps_0 = {eps_0:.4e}")
        print(f"  Cancellation factor: {max_term/eps_0:.0f}x")

    print(f"\n{'='*70}")
    print("THE MECHANISM")
    print("=" * 70)
