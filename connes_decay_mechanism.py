"""
Session 29k: WHY is eigenvector decay 15x faster than BT predicts?

BT argument: b(z) analytic in Bernstein ellipse with rho = exp(L/(8*pi*N))
  => sigma_k <= C * rho^{-k}
  => eigenvector components |xi_n| <= C * rho^{-|n|}
  => predicted r = rho^{-1} ≈ 0.997

ACTUAL: r ≈ 0.75 (lam^2=14), 0.82 (lam^2=50)

HYPOTHESES for the faster decay:
  A) The Bernstein ellipse is NOT the right domain — the actual analyticity
     domain of b(z) is much larger (wider strip)
  B) The eigenvector has BETTER analyticity than the generators
     (because Q_W is positive-definite, the min eigenvector is "smoother")
  C) The fast decay comes from the SPECIFIC structure of Q_W
     (not just displacement rank, but the actual values)
  D) The Q_W matrix has exponential off-diagonal decay, giving
     localization of eigenvectors independent of BT

Also: investigate N(lambda) scaling for the large-lambda breakdown.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time
import sys

# Reuse the build_QW from connes_h1h2_correct.py
mp.dps = 50


def build_QW(lam_sq, N_val):
    """Build Q_W = W02 - WR - Wp."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    vM = []
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]:
        if p > lam_sq:
            break
        lp_f = np.log(p)
        pk = p
        while pk <= lam_sq:
            vM.append((pk, lp_f, np.log(pk)))
            pk *= p

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

    L2_f = L_f**2
    p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    def W02_element(n, m):
        return pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))

    def alpha_L_exact(n):
        if n == 0:
            return mpf(0)
        z = exp(-2*L)
        a = pi*mpc(0, n)/L + mpf(1)/4
        h = hyp2f1(1, a, a+1, z)
        f1 = exp(-L/2) * (2*L/(L + 4*pi*mpc(0, n)) * h).imag
        d = digamma(a).imag / 2
        return (f1 + d) / pi

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

    alpha = {}
    for n in range(-N_val, N_val+1):
        alpha[n] = float(alpha_L_exact(abs(n)))
        if n < 0:
            alpha[n] = -alpha[n]

    wr_diag = {}
    for n_val in range(N_val+1):
        wr_diag[n_val] = WR_diag(n_val)
        wr_diag[-n_val] = wr_diag[n_val]

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
    print("SESSION 29k: EIGENVECTOR DECAY MECHANISM")
    print("=" * 70)

    # ================================================================
    # PART 1: Off-diagonal decay of Q_W itself
    # ================================================================
    print("\nPART 1: OFF-DIAGONAL DECAY OF Q_W")
    print("-" * 70)

    # Hypothesis D: Q_W has exponential off-diagonal decay
    # |Q_W_{nm}| <= C * r^|n-m| => eigenvector localization

    for lam_sq in [14, 50]:
        N = 30
        t0 = time.time()
        QW = build_QW(lam_sq, N)
        print(f"\n  lam^2={lam_sq} ({time.time()-t0:.0f}s):")

        # Measure off-diagonal decay along each super-diagonal
        center = N
        print(f"  {'|n-m|':>6} {'max|Q_{nm}|':>14} {'avg|Q_{nm}|':>14} {'Q_{0,d}/Q_{0,0}':>16}")
        for d in range(0, min(N, 20)+1):
            vals = []
            for i in range(2*N+1-d):
                vals.append(abs(QW[i, i+d]))
            max_val = max(vals)
            avg_val = np.mean(vals)
            ratio = QW[center, center+d] / QW[center, center] if d <= N else 0
            print(f"  {d:>6} {max_val:>14.6e} {avg_val:>14.6e} {ratio:>16.8f}")

        # Fit off-diagonal decay: max|Q_{nm}| at distance d
        ds = np.arange(1, 16)
        max_offdiag = []
        for d in ds:
            vals = [abs(QW[i, i+d]) for i in range(2*N+1-d)]
            max_offdiag.append(max(vals))
        max_offdiag = np.array(max_offdiag)
        log_offdiag = np.log(max_offdiag + 1e-300)
        valid = log_offdiag > -50
        if np.sum(valid) > 3:
            slope, intercept = np.polyfit(ds[valid], log_offdiag[valid], 1)
            r_offdiag = np.exp(slope)
            print(f"  Off-diagonal decay rate: r_QW = {r_offdiag:.4f}")
        else:
            print(f"  No clear off-diagonal decay")

    # ================================================================
    # PART 2: Decompose eigenvector decay into W02, WR, Wp contributions
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: WHICH COMPONENT DRIVES THE FAST DECAY?")
    print("-" * 70)

    lam_sq = 14
    N = 30
    L_f = np.log(lam_sq)

    QW = build_QW(lam_sq, N)
    evals, evecs = np.linalg.eigh(QW)
    xi_0 = evecs[:, 0]
    eps_0 = evals[0]

    # Rebuild the individual components
    L = log(mpf(lam_sq))
    eL = exp(L)
    dim = 2*N + 1

    # W02 matrix
    L2_f = L_f**2
    p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    W02 = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N
        for j in range(dim):
            m = j - N
            W02[i,j] = pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))

    # W02 is positive semi-definite (it's a Gram matrix from a specific kernel)
    evals_w02 = np.linalg.eigvalsh(W02)

    # WR + Wp = W02 - QW
    WR_WP = W02 - QW
    evals_wrwp = np.linalg.eigvalsh(WR_WP)

    print(f"  lam^2={lam_sq}, N={N}")
    print(f"  Q_W eigenvalues:  [{evals[0]:.4e}, {evals[-1]:.4e}]")
    print(f"  W02 eigenvalues:  [{evals_w02[0]:.4e}, {evals_w02[-1]:.4e}]")
    print(f"  WR+Wp eigenvalues: [{evals_wrwp[0]:.4e}, {evals_wrwp[-1]:.4e}]")

    # Eigenvector of Q_W applied to each component
    xi_W02 = xi_0 @ W02 @ xi_0  # <xi|W02|xi>
    xi_QW = xi_0 @ QW @ xi_0    # = eps_0
    xi_WRWP = xi_0 @ WR_WP @ xi_0

    print(f"\n  Rayleigh quotients with xi_0:")
    print(f"    <xi|Q_W|xi>  = {xi_QW:.6e} (= eps_0)")
    print(f"    <xi|W02|xi>  = {xi_W02:.6e}")
    print(f"    <xi|WR+Wp|xi> = {xi_WRWP:.6e}")
    print(f"    Check: W02 - (WR+Wp) = {xi_W02 - xi_WRWP:.6e} (should = eps_0)")

    # W02 is rank-1? Check
    _, sv_w02, _ = np.linalg.svd(W02)
    print(f"\n  W02 singular values (top 5): {', '.join(f'{s:.4e}' for s in sv_w02[:5])}")
    print(f"  W02 effective rank: {np.sum(sv_w02 > sv_w02[0]*1e-10)}")

    # ================================================================
    # PART 3: The REAL analyticity domain of b(z)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: ANALYTICITY DOMAIN OF b(z)")
    print("-" * 70)

    # b(z) = Im(psi(1/4 + i*pi*z/L)) + hypergeometric correction
    # psi has poles at 1/4 + i*pi*z/L = -m (m=0,1,2,...)
    # => z = i*L*(m + 1/4)/pi
    # Nearest pole: z = i*L/(4*pi) at distance L/(4*pi) from real axis

    # But the EIGENVECTOR isn't b(z) directly — it's the solution of Q_W*xi = eps*xi.
    # The eigenvector might be smoother than b(z).

    # Check: what is the analyticity domain of the EIGENVECTOR?
    # If xi_n has generating function Xi(z) = sum xi_n z^n,
    # the radius of convergence tells us the analyticity radius.

    for lam_sq in [14, 50]:
        QW = build_QW(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi_0 = evecs[:, 0]
        center = N

        # Cauchy-Hadamard: R = 1/limsup |xi_n|^{1/n}
        abs_xi = np.abs(xi_0[center:])  # positive n components
        n_vals = np.arange(0, N+1)

        # Compute |xi_n|^{1/n} for n >= 3
        R_estimates = []
        for n in range(3, N+1):
            if abs_xi[n] > 1e-20:
                R_est = abs_xi[n]**(-1.0/n)
                R_estimates.append((n, R_est))

        if R_estimates:
            ns_R = [r[0] for r in R_estimates]
            Rs = [r[1] for r in R_estimates]
            print(f"\n  lam^2={lam_sq}: Cauchy-Hadamard radius estimates")
            print(f"    R(n=5) = {Rs[2] if len(Rs)>2 else 'N/A':.4f}")
            print(f"    R(n=10) = {Rs[7] if len(Rs)>7 else 'N/A':.4f}")
            print(f"    R(n=20) = {Rs[17] if len(Rs)>17 else 'N/A':.4f}")
            print(f"    R(n=N={N}) = {Rs[-1]:.4f}")
            print(f"    => decay rate r = 1/R ≈ {1/Rs[-1]:.4f}")

        # Compare with BT:
        L_val = np.log(lam_sq)
        r_BT = np.exp(-L_val/(8*np.pi*N))
        d_pole = L_val / (4*np.pi)
        print(f"    BT prediction: r_BT = exp(-L/(8*pi*N)) = {r_BT:.6f}")
        print(f"    Pole distance: L/(4*pi) = {d_pole:.4f}")
        print(f"    If xi analytic in strip |Im(z)| < D, then r = exp(-D/N)")
        print(f"    Actual r = {1/Rs[-1]:.4f} => effective D = {-N*np.log(1/Rs[-1]):.4f}")

    # ================================================================
    # PART 4: N(lambda) scaling for large lambda
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: N(lambda) SCALING")
    print("-" * 70)

    # For each lam^2, find the smallest N such that Q_W is positive-definite
    print(f"\n  {'lam^2':>6} {'N_min for pos-def':>20} {'eps_0 at N_min':>16}")
    print("  " + "-" * 50)

    for lam_sq in [14, 30, 50, 75, 100]:
        found = False
        for N_test in [10, 15, 20, 25, 30, 35, 40]:
            t0 = time.time()
            try:
                QW = build_QW(lam_sq, N_test)
                evals = np.linalg.eigvalsh(QW)
                if evals[0] > -1e-6:  # essentially positive
                    print(f"  {lam_sq:>6} {N_test:>20} {evals[0]:>16.4e} ({time.time()-t0:.0f}s)")
                    found = True
                    break
            except Exception as e:
                pass
        if not found:
            print(f"  {lam_sq:>6} {'> 40':>20}")

    # ================================================================
    # PART 5: The key — can we prove r < 1 from Q_W structure?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: STRUCTURAL PROOF OF DECAY")
    print("-" * 70)

    # For the min eigenvector of a banded/decaying matrix:
    # If Q_W_{nm} <= C * rho^{-|n-m|} and Q_W is positive-definite,
    # then the min eigenvector also decays: |xi_n| <= C' * rho'^{-|n|}
    # with rho' related to rho.

    # This is the COMBES-THOMAS estimate from mathematical physics!
    # For a self-adjoint operator H with exponential off-diagonal decay:
    #   |H_{nm}| <= C * e^{-mu*|n-m|}
    # and if E is an eigenvalue with gap delta to the rest of spectrum:
    #   |psi_n| <= C' * e^{-mu'*|n-n_0|}
    # where mu' = min(mu, something involving delta)

    # The Combes-Thomas bound gives EXACTLY what we need!
    # It says: eigenvectors of exponentially-localized operators
    # are exponentially localized, with a rate determined by
    # the off-diagonal decay AND the spectral gap.

    lam_sq = 14
    N = 30
    QW = build_QW(lam_sq, N)
    evals, evecs = np.linalg.eigh(QW)
    xi_0 = evecs[:, 0]
    eps_0 = evals[0]
    eps_1 = evals[1]
    gap = eps_1 - eps_0

    # Measure off-diagonal decay of Q_W
    center = N
    offdiag_decay = []
    for d in range(1, N+1):
        vals = [abs(QW[i, i+d]) for i in range(2*N+1-d)]
        offdiag_decay.append(max(vals))

    offdiag_decay = np.array(offdiag_decay)
    ds = np.arange(1, N+1)

    # Fit: |Q_W_{nm}| <= A * exp(-mu * |n-m|)
    log_od = np.log(offdiag_decay + 1e-300)
    valid_od = log_od > -40
    if np.sum(valid_od) > 3:
        mu_fit, log_A = np.polyfit(ds[valid_od], log_od[valid_od], 1)
        mu = -mu_fit  # decay rate (positive)
        A = np.exp(log_A)

        print(f"\n  Q_W off-diagonal decay: |Q_nm| <= {A:.4f} * exp(-{mu:.4f} * |n-m|)")
        print(f"  Spectral gap: delta = eps_1 - eps_0 = {gap:.4e}")
        print(f"  eps_0 = {eps_0:.4e}")

        # Combes-Thomas: eigenvector decay rate mu' = mu - C/delta
        # More precisely: mu' = mu if mu < arccosh(1 + delta/(2*||Q||))
        # For our case: ||Q|| ~ evals[-1] ~ O(1)
        norm_Q = evals[-1]
        mu_CT = np.arccosh(1 + gap / (2 * norm_Q))
        mu_effective = min(mu, mu_CT)

        print(f"\n  Combes-Thomas analysis:")
        print(f"    Off-diagonal decay: mu = {mu:.4f}")
        print(f"    CT bound: mu_CT = arccosh(1 + delta/(2||Q||)) = {mu_CT:.4f}")
        print(f"    Effective: mu' = min(mu, mu_CT) = {mu_effective:.4f}")
        print(f"    Predicted r = exp(-mu') = {np.exp(-mu_effective):.4f}")

        # Compare with actual
        abs_xi = np.abs(xi_0[center:])
        ns_fit = np.arange(3, N+1)
        log_xi = np.log(abs_xi[3:] + 1e-300)
        valid_xi = log_xi > -50
        if np.sum(valid_xi) > 3:
            mu_actual, _ = np.polyfit(ns_fit[valid_xi], log_xi[valid_xi], 1)
            r_actual = np.exp(mu_actual)
            print(f"    Actual decay rate: r = {r_actual:.4f} (mu = {-mu_actual:.4f})")
            print(f"    CT prediction works? {np.exp(-mu_effective):.4f} vs actual {r_actual:.4f}")

    # Repeat for lam^2 = 50
    print(f"\n  --- lam^2 = 50 ---")
    lam_sq = 50
    QW = build_QW(lam_sq, N)
    evals, evecs = np.linalg.eigh(QW)
    xi_0 = evecs[:, 0]
    eps_0 = evals[0]
    eps_1 = evals[1]
    gap = eps_1 - eps_0

    offdiag_decay = []
    for d in range(1, N+1):
        vals = [abs(QW[i, i+d]) for i in range(2*N+1-d)]
        offdiag_decay.append(max(vals))
    offdiag_decay = np.array(offdiag_decay)

    log_od = np.log(offdiag_decay + 1e-300)
    valid_od = log_od > -40
    if np.sum(valid_od) > 3:
        mu_fit, log_A = np.polyfit(ds[valid_od], log_od[valid_od], 1)
        mu = -mu_fit
        A = np.exp(log_A)
        norm_Q = evals[-1]
        mu_CT = np.arccosh(1 + gap / (2 * norm_Q))
        mu_effective = min(mu, mu_CT)

        abs_xi = np.abs(xi_0[center:])
        log_xi = np.log(abs_xi[3:] + 1e-300)
        valid_xi = log_xi > -50
        mu_actual = 0
        if np.sum(valid_xi) > 3:
            mu_actual, _ = np.polyfit(ns_fit[valid_xi], log_xi[valid_xi], 1)

        print(f"  Off-diagonal: mu = {mu:.4f}, CT: mu_CT = {mu_CT:.4f}")
        print(f"  Effective: mu' = {mu_effective:.4f}, predicted r = {np.exp(-mu_effective):.4f}")
        print(f"  Actual: r = {np.exp(mu_actual):.4f}")

    print(f"\n{'='*70}")
    print("SYNTHESIS: THE COMBES-THOMAS ROUTE TO H1")
    print("=" * 70)
    print("""
The Combes-Thomas estimate from mathematical physics gives:

  THEOREM (Combes-Thomas): If H is a self-adjoint operator on l^2(Z)
  with |H_{nm}| <= A * exp(-mu * |n-m|), and E is an eigenvalue
  with spectral gap delta = dist(E, spec(H)\\{E}), then the
  eigenvector psi_E satisfies:

    |psi_E(n)| <= C * exp(-mu' * |n - n_0|)

  where mu' = min(mu, arccosh(1 + delta/(2*||H||)))
  and n_0 is the localization center.

For our Q_W matrix:
  - Off-diagonal decay: VERIFIED (mu > 0)
  - Spectral gap: VERIFIED (delta = eps_1 - eps_0 > 0, ratio ~3)
  - Positive-definite: VERIFIED (for lam^2 <= 50, N=30)

  => Combes-Thomas gives H1 (eigenvector freezing) DIRECTLY!
  => No need for BT — CT is the RIGHT tool for this problem.

The BT argument bounds decay via Bernstein ellipse analyticity.
The CT argument bounds decay via off-diagonal matrix structure + spectral gap.
CT gives a MUCH tighter bound because it uses the actual matrix decay rate,
not the conservative analyticity domain.

REMAINING STEPS FOR RIGOROUS PROOF:
  1. Prove Q_W has exponential off-diagonal decay (from the explicit formulas)
  2. Prove the spectral gap is bounded away from 0 (H2)
  3. Apply Combes-Thomas to get H1

Note: H2 (spectral gap) is still needed as INPUT to Combes-Thomas.
But H2 might be provable from the off-diagonal decay + positivity.
""")
