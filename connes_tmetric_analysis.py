"""
Session 30b: T-metric analysis — understand the constant C in |xi_hat| <= C*eps_0.

From the variational equation:
  Q_W(xi, f) = eps_0 * <xi, f>_T

And xi_hat(gamma_k) ~ Q_W(xi, f_k) where f_k is a bump at gamma_k.

So: |xi_hat(gamma_k)| = eps_0 * |<xi, f_k>_T|

The ratio |xi_hat|/eps_0 = |<xi, f_k>_T| ~ 10^3 to 10^5 (from data).

QUESTION: What is <xi, f_k>_T?

APPROACH 1: Compute Q_W(xi, f_k) directly and divide by eps_0.
This gives <xi, f_k>_T = Q_W(xi, f_k) / eps_0.

APPROACH 2: The Weil explicit formula gives:
Q_W(xi, f) = sum_rho xi_hat(rho) * conj(f_hat(rho)) + (pole terms)

For f_k = bump at gamma_k:
Q_W(xi, f_k) = xi_hat(gamma_k) + sum_{rho != gamma_k} xi_hat(rho)*conj(f_hat(rho)) + poles

If xi_hat at OTHER zeros is also ~10^{-5}:
Q_W(xi, f_k) ~ xi_hat(gamma_k) + leakage ~ xi_hat(gamma_k) * (1 + small)

So <xi, f_k>_T ~ xi_hat(gamma_k) / eps_0 ~ 10^3 to 10^5.

THE KEY INSIGHT: Q_W(xi, f) = eps_0 * <xi, f>_T is the EXACT variational equation.
There's NO approximation here. The T-inner product IS defined by this equation.

So for ANY f in E_N: <xi, f>_T = Q_W(xi, f) / eps_0.

Now Q_W = W02 - WR - Wp. Each component contributes:
<xi, f>_T = [W02(xi,f) - WR(xi,f) - Wp(xi,f)] / eps_0

Since W02 ~ O(1), WR ~ O(1), Wp ~ O(1), and eps_0 ~ 10^{-10}:
<xi, f>_T ~ O(1)/10^{-10} = 10^{10} ... but with MASSIVE CANCELLATION.

The cancellation gives <xi, f>_T ~ 10^3 to 10^5 (not 10^{10}).
The cancellation quality depends on how well W02(xi,f) ≈ WR(xi,f) + Wp(xi,f).

For the PROOF: we need |<xi, f>_T| bounded as lambda -> inf.
This means W02(xi,f), WR(xi,f), Wp(xi,f) must cancel to within eps_0 * C
for some bounded C.

COMPUTE THIS DIRECTLY: break Q_W(xi, f_k) into its three components.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time
import sys
sys.path.insert(0, '.')

mp.dps = 50


def build_components(lam_sq, N_val, n_quad=20000):
    """Build W02, WR, Wp separately."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    vM = []
    for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]:
        if p > lam_sq: break
        pk = p
        while pk <= lam_sq:
            vM.append((pk, np.log(p), np.log(pk)))
            pk *= p

    def q_func(n, m, y):
        if n != m:
            return (np.sin(2*np.pi*m*y/L_f) - np.sin(2*np.pi*n*y/L_f)) / (np.pi*(n-m))
        else:
            return 2*(L_f - y)/L_f * np.cos(2*np.pi*n*y/L_f)

    L2_f = L_f**2; p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    W02 = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(dim):
            m = j - N_val
            W02[i,j] = pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))

    alpha = {}
    for n in range(-N_val, N_val+1):
        if n == 0:
            alpha[n] = 0.0
        else:
            z = exp(-2*L)
            a = pi*mpc(0,abs(n))/L + mpf(1)/4
            h = hyp2f1(1,a,a+1,z)
            f1 = exp(-L/2) * (2*L/(L+4*pi*mpc(0,abs(n)))*h).imag
            d = digamma(a).imag/2
            val = float((f1+d)/pi)
            alpha[n] = val if n>0 else -val

    wr_diag = {}
    omega_0 = mpf(2)
    for nv in range(N_val+1):
        def omega(x, nv=nv):
            return 2*(1-x/L)*cos(2*pi*nv*x/L)
        w_const = (omega_0/2)*(euler+log(4*pi*(eL-1)/(eL+1)))
        dx = L/n_quad; integral = mpf(0)
        for k in range(n_quad):
            x = dx*(k+mpf(1)/2)
            numer = exp(x/2)*omega(x)-omega_0
            denom = exp(x)-exp(-x)
            if abs(denom) > mpf(10)**(-40): integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const+integral)
        wr_diag[-nv] = wr_diag[nv]

    WR = np.zeros((dim, dim))
    Wp = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        WR[i,i] = wr_diag[n]
        for j in range(dim):
            m = j - N_val
            if n != m:
                WR[i,j] = (alpha[m]-alpha[n])/(n-m)
            Wp[i,j] = sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
    Wp = (Wp + Wp.T)/2
    QW = W02 - WR - Wp
    QW = (QW + QW.T)/2
    return W02, WR, Wp, QW


def make_bump_vector(gamma, N, L_f, Delta=2.0):
    """Create a discrete bump function at frequency gamma with width Delta.

    f_n = sinc-like weight centered at n_peak = gamma*L/(2*pi).
    """
    dim = 2*N + 1
    f = np.zeros(dim)
    for idx in range(dim):
        n = idx - N
        freq_n = 2*np.pi*n/L_f
        # Bump: concentrated at freq = gamma, width Delta
        x = (freq_n - gamma) / Delta
        if abs(x) < 10:
            f[idx] = np.exp(-x**2/2)  # Gaussian bump
    # Normalize
    norm = np.linalg.norm(f)
    if norm > 1e-20:
        f /= norm
    return f


if __name__ == "__main__":
    print("T-METRIC ANALYSIS")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # ================================================================
    # PART 1: Decompose Q_W(xi, f_k) into components
    # ================================================================
    print("\nPART 1: Q_W(xi, f_k) = W02(xi,f) - WR(xi,f) - Wp(xi,f)")
    print("-" * 70)

    for lam_sq in [14, 50, 100, 200]:
        N = max(21, round(8 * np.log(lam_sq)))
        L_f = np.log(lam_sq)

        t0 = time.time()
        W02, WR, Wp, QW = build_components(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi = evecs[:, 0]
        eps_0 = evals[0]

        # Normalize
        xs = np.sum(xi)
        if abs(xs) > 1e-30:
            xi_n = xi * np.sqrt(L_f) / xs
        else:
            xi_n = xi

        # Bump at gamma_1
        f_k = make_bump_vector(gammas[0], N, L_f, Delta=2.0)

        # Component-wise inner products with xi (unnormalized eigenvector)
        qw_xf = xi @ QW @ f_k
        w02_xf = xi @ W02 @ f_k
        wr_xf = xi @ WR @ f_k
        wp_xf = xi @ Wp @ f_k

        # The T-inner product
        t_inner = qw_xf / eps_0 if abs(eps_0) > 1e-20 else float('inf')

        print(f"\n  lam^2={lam_sq}, N={N} ({time.time()-t0:.0f}s):")
        print(f"    eps_0         = {eps_0:.6e}")
        print(f"    W02(xi, f)    = {w02_xf:.6e}")
        print(f"    WR(xi, f)     = {wr_xf:.6e}")
        print(f"    Wp(xi, f)     = {wp_xf:.6e}")
        print(f"    QW(xi, f)     = {qw_xf:.6e}  (= W02 - WR - Wp)")
        print(f"    <xi, f>_T     = {t_inner:.6e}  (= QW/eps_0)")
        print(f"    Cancellation: {abs(w02_xf)/abs(qw_xf):.0f}x")

    # ================================================================
    # PART 2: How does the cancellation scale with lambda?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: CANCELLATION SCALING vs LAMBDA")
    print("-" * 70)

    print(f"\n{'lam^2':>6} {'eps_0':>12} {'|W02(xi,f)|':>12} {'|QW(xi,f)|':>12} "
          f"{'cancel':>8} {'<xi,f>_T':>12}")
    print("-" * 70)

    for lam_sq in [14, 30, 50, 100, 200, 300]:
        N = max(21, round(8 * np.log(lam_sq)))
        L_f = np.log(lam_sq)

        W02, WR, Wp, QW = build_components(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi = evecs[:, 0]
        eps_0 = evals[0]

        f_k = make_bump_vector(gammas[0], N, L_f)

        w02_xf = abs(xi @ W02 @ f_k)
        qw_xf = abs(xi @ QW @ f_k)
        cancel = w02_xf / qw_xf if qw_xf > 1e-20 else float('inf')
        t_inner = qw_xf / abs(eps_0) if abs(eps_0) > 1e-20 else float('inf')

        print(f"{lam_sq:>6} {eps_0:>12.4e} {w02_xf:>12.4e} {qw_xf:>12.4e} "
              f"{cancel:>8.0f} {t_inner:>12.4e}")

    # ================================================================
    # PART 3: The convergence mechanism
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: THE CONVERGENCE MECHANISM")
    print("-" * 70)

    # For the proof: we need eps_0 * <xi, f>_T -> 0.
    # <xi, f>_T = QW(xi, f) / eps_0
    # So eps_0 * <xi, f>_T = QW(xi, f)
    #
    # THE QUESTION REDUCES TO: does QW(xi, f_k) -> 0?
    #
    # QW(xi, f_k) = W02(xi, f_k) - WR(xi, f_k) - Wp(xi, f_k)
    #
    # Each term is O(1) but they cancel. The residual QW(xi, f_k) is:
    # = eps_0 * <xi, f_k>_T
    #
    # This is EXACTLY |xi_hat(gamma_k)| (up to normalization factors).
    # So the question "does xi_hat -> 0?" is the same as "does QW(xi,f) -> 0?"
    # is the same as "does the cancellation improve with lambda?"

    # If the cancellation is perfect in the limit (as lambda -> inf, the three
    # terms all approach the SAME value), then QW(xi, f) -> 0.

    # The cancellation quality depends on how well the even/odd W02 eigenvectors
    # are "matched" by WR and Wp. Since W02, WR, Wp all depend on L = log(lam^2),
    # and as L -> inf they all grow, the RELATIVE cancellation might stay constant
    # or improve.

    # Test: compute W02(xi,f), WR(xi,f), Wp(xi,f) and their ratios
    print(f"\n{'lam^2':>6} {'W02/QW':>10} {'WR/QW':>10} {'Wp/QW':>10} "
          f"{'(WR+Wp)/W02':>12}")
    print("-" * 55)

    for lam_sq in [14, 50, 100, 200, 300]:
        N = max(21, round(8 * np.log(lam_sq)))
        L_f = np.log(lam_sq)

        W02, WR, Wp, QW = build_components(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi = evecs[:, 0]

        f_k = make_bump_vector(gammas[0], N, L_f)

        w02 = xi @ W02 @ f_k
        wr = xi @ WR @ f_k
        wp = xi @ Wp @ f_k
        qw = xi @ QW @ f_k
        ratio = (wr + wp) / w02 if abs(w02) > 1e-20 else float('inf')

        print(f"{lam_sq:>6} {w02/qw:>10.1f} {wr/qw:>10.1f} {wp/qw:>10.1f} "
              f"{ratio:>12.10f}")

    print(f"\n  If (WR+Wp)/W02 -> 1 as lambda -> inf: cancellation improves -> QW(xi,f) -> 0 -> RH")
    print(f"  If (WR+Wp)/W02 stays bounded away from 1: |xi_hat| stays ~10^{{-5}} -> NOT converging")
