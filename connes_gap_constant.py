"""
Session 30 iteration 9: The mysterious constant gap s_even - mu_max ~ 0.03.

Both s_even and mu_max grow as ~0.74*L^2. Their difference stays ~0.03.
Is this a known constant? Does it converge to a limit?

Also: collect ALL numerical data for the paper tables.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 50


if __name__ == "__main__":
    print("THE GAP CONSTANT + PAPER DATA")
    print("=" * 70)

    # Fine scan of the gap at many lambda values
    print(f"{'lam^2':>7} {'L':>6} {'N':>4} {'s_even':>10} {'mu_max':>10} "
          f"{'gap':>10} {'gap*L^2':>10} {'eps_0':>12} {'gap_r':>7}")
    print("-" * 90)

    all_data = []

    for lam_sq in [10, 14, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N+1

        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N, n_quad=10000)

        evals_qw = np.linalg.eigvalsh(QW)
        evals_m = np.linalg.eigvalsh(M)
        evals_w02 = np.linalg.eigvalsh(W02)

        eps_0 = evals_qw[0]
        eps_1 = evals_qw[1]
        gap_r = eps_1/eps_0 if abs(eps_0) > 1e-20 else 0

        # W02 eigenvalues
        w02_nz = evals_w02[np.abs(evals_w02) > 1e-10]
        s_even = max(w02_nz)
        s_odd = min(w02_nz)

        # M max eigenvalue
        mu_max = evals_m[-1]

        gap = s_even - mu_max
        gap_L2 = gap * L_f**2

        dt = time.time() - t0

        all_data.append({
            'lam_sq': lam_sq, 'L': L_f, 'N': N, 'dim': dim,
            's_even': s_even, 's_odd': s_odd, 'mu_max': mu_max,
            'gap': gap, 'eps_0': eps_0, 'gap_ratio': gap_r
        })

        print(f"{lam_sq:>7} {L_f:>6.3f} {N:>4} {s_even:>10.4f} {mu_max:>10.4f} "
              f"{gap:>10.6f} {gap_L2:>10.4f} {eps_0:>12.4e} {gap_r:>7.1f}")

    # Analyze the gap
    gaps = np.array([d['gap'] for d in all_data])
    Ls = np.array([d['L'] for d in all_data])

    print(f"\nGap statistics:")
    print(f"  mean = {np.mean(gaps):.6f}")
    print(f"  std  = {np.std(gaps):.6f}")
    print(f"  min  = {np.min(gaps):.6f} at lam^2={all_data[np.argmin(gaps)]['lam_sq']}")
    print(f"  max  = {np.max(gaps):.6f} at lam^2={all_data[np.argmax(gaps)]['lam_sq']}")

    # Does the gap converge? Check last 5 values
    last5 = gaps[-5:]
    print(f"  Last 5 gaps: {', '.join(f'{g:.6f}' for g in last5)}")

    # Known constants for comparison
    print(f"\n  Candidate constants:")
    print(f"    1/(4*pi) = {1/(4*np.pi):.6f}")
    print(f"    1/(8*pi) = {1/(8*np.pi):.6f}")
    print(f"    ln(2)/pi^2 = {np.log(2)/np.pi**2:.6f}")
    print(f"    gamma/(2*pi) = {0.5772/(2*np.pi):.6f}")
    print(f"    1/(2*pi*e) = {1/(2*np.pi*np.e):.6f}")
    print(f"    6/pi^2 - 1/2 = {6/np.pi**2 - 0.5:.6f}")

    # PAPER TABLE: Complete numerical evidence
    print(f"\n{'='*70}")
    print("PAPER TABLE: COMPLETE NUMERICAL EVIDENCE")
    print("=" * 70)
    print(f"{'lam^2':>7} {'N':>4} {'dim':>5} {'eps_0':>14} {'eps_1':>14} "
          f"{'gap_ratio':>9} {'all_pos':>7}")
    print("-" * 65)

    for d in all_data:
        pos = "YES" if d['eps_0'] > 0 else "NO"
        print(f"{d['lam_sq']:>7} {d['N']:>4} {d['dim']:>5} {d['eps_0']:>14.6e} "
              f"{d['eps_0']*d['gap_ratio']:>14.6e} {d['gap_ratio']:>9.1f} {pos:>7}")

    print(f"\n{'='*70}")
