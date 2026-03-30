"""
Session 29i: Connes H1/H2 with CORRECT Q_W matrix.

The correct Weil quadratic form Q_W = W02 - WR - Wp
(from connes_attack.py, session 22) — NOT the simplified psi-based tau.

Focus: verify H1 (eigenvector freezing) and H2 (spectral gap) with the
actual tiny eigenvalue eps_0 ~ exp(-cL).
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time

mp.dps = 50


def build_QW(lam_sq, N_val):
    """Build the correct Weil QW matrix from connes_attack.py."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    # Prime powers
    vM = []
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]:
        if p > lam_sq:
            break
        lp_f = np.log(p)
        pk = p
        while pk <= lam_sq:
            vM.append((pk, lp_f, np.log(pk)))
            pk *= p

    # q kernel for Wp
    def q_func(n, m, y):
        if n != m:
            return (np.sin(2*np.pi*m*y/L_f) - np.sin(2*np.pi*n*y/L_f)) / (np.pi*(n-m))
        else:
            return 2*(L_f - y)/L_f * np.cos(2*np.pi*n*y/L_f)

    def Wp_element(n, m):
        total = 0.0
        for k, lk, logk in vM:
            total += lk * k**(-0.5) * q_func(n, m, logk)
        return total

    # W02
    L2_f = L_f**2
    p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    def W02_element(n, m):
        return pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))

    # WR off-diagonal (exact Prop 4.3)
    def alpha_L_exact(n):
        if n == 0:
            return mpf(0)
        z = exp(-2*L)
        a = pi*mpc(0, n)/L + mpf(1)/4
        h = hyp2f1(1, a, a+1, z)
        f1 = exp(-L/2) * (2*L/(L + 4*pi*mpc(0, n)) * h).imag
        d = digamma(a).imag / 2
        return (f1 + d) / pi

    # WR diagonal (eq 4.4)
    def WR_diag(n, n_quad=20000):
        omega_0 = mpf(2)
        def omega(x):
            return 2*(1 - x/L)*cos(2*pi*n*x/L)
        w_const = (omega_0/2)*(euler + log(4*pi*(eL-1)/(eL+1)))
        dx = L/n_quad
        integral = mpf(0)
        for k in range(n_quad):
            x = dx*(k + mpf(1)/2)
            numer = exp(x/2)*omega(x) - omega_0
            denom = exp(x) - exp(-x)
            if abs(denom) > mpf(10)**(-40):
                integral += numer/denom
        integral *= dx
        return float(w_const + integral)

    # Precompute alpha_L
    alpha = {}
    for n in range(-N_val, N_val+1):
        alpha[n] = float(alpha_L_exact(abs(n)))
        if n < 0:
            alpha[n] = -alpha[n]

    # Diagonal WR
    wr_diag = {}
    for n_val in range(N_val+1):
        wr_diag[n_val] = WR_diag(n_val)
        wr_diag[-n_val] = wr_diag[n_val]

    # Assemble QW
    QW = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(i, dim):
            m = j - N_val
            w02 = W02_element(n, m)
            wp = Wp_element(n, m)
            if n == m:
                wr = wr_diag[n]
            else:
                wr = (alpha[m] - alpha[n]) / (n - m)
            QW[i, j] = w02 - wr - wp
            QW[j, i] = QW[i, j]
    QW = (QW + QW.T) / 2

    return QW


if __name__ == "__main__":
    print("CONNES H1/H2 WITH CORRECT Q_W MATRIX")
    print("=" * 70)

    # ================================================================
    # PART 1: Build Q_W and verify tiny eps_0
    # ================================================================
    print("\nPART 1: EIGENVALUE SPECTRUM OF Q_W")
    print("-" * 70)

    N = 30

    for lam_sq in [14, 30, 50, 100]:
        t0 = time.time()
        L = np.log(lam_sq)
        QW = build_QW(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        dt = time.time() - t0

        n_pos = np.sum(evals > 0)
        n_neg = np.sum(evals < 0)
        eps_0 = evals[0]
        eps_1 = evals[1]
        gap = eps_1 - eps_0
        gap_ratio = gap / abs(eps_0) if abs(eps_0) > 1e-20 else float('inf')

        print(f"  lam^2={lam_sq:>4} (L={L:.3f}): eps_0={eps_0:+.6e}, eps_1={eps_1:+.6e}, "
              f"gap_ratio={gap_ratio:.1f}, pos={n_pos}, neg={n_neg} ({dt:.0f}s)")

    # ================================================================
    # PART 2: H1 — Eigenvector decay at correct eps_0
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: EIGENVECTOR DECAY (H1) — CORRECT Q_W")
    print("-" * 70)

    for lam_sq in [14, 50, 100]:
        QW = build_QW(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi_0 = evecs[:, 0]
        eps_0 = evals[0]
        center = N  # index of n=0

        abs_xi = np.abs(xi_0)

        # Eigenvector components
        print(f"\n  lam^2={lam_sq}: eps_0={eps_0:+.6e}")
        print(f"    |xi_n| for n=0..15: {', '.join(f'{abs_xi[center+k]:.4e}' for k in range(min(16, N+1)))}")

        # Geometric decay fit: |xi_n| ~ C * r^|n| for |n| > 3
        ns = np.arange(3, N+1)
        log_xi_pos = np.log(abs_xi[center + ns] + 1e-300)
        log_xi_neg = np.log(abs_xi[center - ns] + 1e-300)
        log_xi_avg = (log_xi_pos + log_xi_neg) / 2
        valid = log_xi_avg > -50
        if np.sum(valid) > 3:
            slope, intercept = np.polyfit(ns[valid], log_xi_avg[valid], 1)
            r = np.exp(slope)
            L_val = np.log(lam_sq)
            r_BT = np.exp(-L_val / (8*np.pi*N))
            print(f"    Decay rate: r = {r:.6f} (BT prediction: {r_BT:.6f})")
            print(f"    Effective analyticity width: d_eff = {-1/slope:.4f} vs L/(4pi) = {L_val/(4*np.pi):.4f}")

    # ================================================================
    # PART 3: H1 — Freezing as N grows
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: EIGENVECTOR FREEZING AS N GROWS")
    print("-" * 70)

    lam_sq = 50

    prev_xi = None
    prev_eps = None
    for N_test in [10, 15, 20, 25, 30]:
        t0 = time.time()
        QW = build_QW(lam_sq, N_test)
        evals, evecs = np.linalg.eigh(QW)
        xi_0 = evecs[:, 0]
        eps_0 = evals[0]
        dt = time.time() - t0

        if prev_xi is not None:
            prev_N = (len(prev_xi) - 1) // 2
            common = min(prev_N, N_test)
            curr_central = xi_0[N_test-common:N_test+common+1]
            prev_central = prev_xi[prev_N-common:prev_N+common+1]
            # Align sign
            if np.dot(curr_central, prev_central) < 0:
                curr_central = -curr_central
            diff = np.linalg.norm(curr_central - prev_central)
            eps_diff = abs(eps_0 - prev_eps)
            print(f"  N={N_test:>3}: eps_0={eps_0:+.6e}, ||xi-xi_prev||={diff:.4e}, "
                  f"|eps-eps_prev|={eps_diff:.4e} ({dt:.0f}s)")
        else:
            print(f"  N={N_test:>3}: eps_0={eps_0:+.6e} ({dt:.0f}s)")

        prev_xi = xi_0.copy()
        prev_eps = eps_0

    # ================================================================
    # PART 4: H2 — Spectral gap vs lambda
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: SPECTRAL GAP (H2) vs LAMBDA")
    print("-" * 70)

    N = 30
    print(f"\n{'lam^2':>6} {'L':>6} {'eps_0':>12} {'eps_1':>12} {'gap':>12} "
          f"{'gap/|eps_0|':>12} {'exp(-cL)':>10}")
    print("-" * 75)

    for lam_sq in [10, 14, 20, 30, 50, 75, 100, 150, 200]:
        t0 = time.time()
        L = np.log(lam_sq)
        QW = build_QW(lam_sq, N)
        evals = np.linalg.eigvalsh(QW)
        eps_0, eps_1 = evals[0], evals[1]
        gap = eps_1 - eps_0
        gap_ratio = gap / abs(eps_0) if abs(eps_0) > 1e-20 else float('inf')
        exp_cL = np.exp(-0.25 * L)  # rough c estimate

        print(f"{lam_sq:>6} {L:>6.3f} {eps_0:>12.4e} {eps_1:>12.4e} {gap:>12.4e} "
              f"{gap_ratio:>12.1f} {exp_cL:>10.4e}")

    # ================================================================
    # PART 5: Displacement rank of Q_W
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: DISPLACEMENT RANK OF Q_W")
    print("-" * 70)

    N = 20
    for lam_sq in [14, 50, 100]:
        QW = build_QW(lam_sq, N)
        dim = 2*N + 1
        D = np.diag(np.arange(-N, N+1, dtype=float))
        disp = D @ QW - QW @ D
        _, svd_vals, _ = np.linalg.svd(disp)

        print(f"  lam^2={lam_sq}: sigma_1={svd_vals[0]:.4e}, sigma_2={svd_vals[1]:.4e}, "
              f"sigma_3={svd_vals[2]:.4e}, rank2={svd_vals[2]/svd_vals[0]:.2e}")

    print(f"\n{'='*70}")
    print("H1/H2 VERIFICATION COMPLETE")
    print("=" * 70)
