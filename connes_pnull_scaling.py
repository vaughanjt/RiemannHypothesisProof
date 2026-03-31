"""
THE LAST PIECE: How does ||P_null u1||^2 depend on L?

If ||P_null u1||^2 ~ f(L): the cross-term ~ s1 * sqrt(f(L)) ~ L^2 * sqrt(f(L)).
The balance eps_0 ~ 1/L requires understanding this scaling.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import sys, time
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 50

if __name__ == "__main__":
    print("||P_null u||^2 SCALING WITH L")
    print("=" * 60)

    print(f"{'lam^2':>7} {'L':>6} {'N':>4} {'D_null':>6} "
          f"{'||Pn*u_e||^2':>14} {'||Pn*u_o||^2':>14} {'s1*||Pn||':>12}")
    print("-" * 75)

    for lam_sq in [14, 30, 50, 100, 200, 500, 1000, 2000]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N+1

        W02, M, QW = build_all(lam_sq, N, n_quad=10000)

        evals_m, evecs_m = np.linalg.eigh(M)
        evals_w02, evecs_w02 = np.linalg.eigh(W02)

        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        null_idx = np.where(abs_evals < threshold)[0]
        P_null = evecs_m[:, null_idx]
        D_null = len(null_idx)

        idx_w02 = np.where(np.abs(evals_w02) > 1e-10)[0]
        center = N

        pn_even = pn_odd = 0
        s1 = 0
        for idx in idx_w02:
            u = evecs_w02[:, idx]
            s = evals_w02[idx]
            pn = np.linalg.norm(P_null.T @ u)**2
            even = sum(abs(u[center+k] - u[center-k]) for k in range(1,N+1))
            odd = sum(abs(u[center+k] + u[center-k]) for k in range(1,N+1))
            if even < odd:
                pn_even = pn; s1 = s
            else:
                pn_odd = pn

        cross = s1 * np.sqrt(pn_even)
        print(f"{lam_sq:>7} {L_f:>6.3f} {N:>4} {D_null:>6} "
              f"{pn_even:>14.6e} {pn_odd:>14.6e} {cross:>12.4e}")

    print(f"\nKey: if ||Pn*u_even||^2 ~ 1/L^a, then cross ~ L^(2-a/2)")
