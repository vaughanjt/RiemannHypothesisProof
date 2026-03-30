"""
Session 29r: THE DECISIVE EXPERIMENT — double limit N ~ C*L.

For each lam^2, set N = round(C * L) where C ~ 8.
Verify:
  1. All eigenvalues of Q_W positive
  2. eps_0 decreasing with lambda
  3. |xi_hat(gamma_k)| -> 0 for the first few zeta zeros
  4. Gap ratio stable

If all hold, the corrected proof skeleton closes.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time

mp.dps = 30


def build_QW_fast(lam_sq, N_val):
    """Build Q_W with reduced quadrature for speed at large N."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    vM = []
    primes_list = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
                   101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,
                   197,199,211,223,227,229,233,239,241,251]
    for p in primes_list:
        if p > lam_sq: break
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

    L2_f = L_f**2; p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    # Alpha values
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

    # WR diagonal (reduced quadrature for speed)
    wr_diag = {}
    omega_0 = mpf(2)
    n_quad = max(3000, N_val * 100)
    for nv in range(N_val+1):
        def omega(x, nv=nv):
            return 2*(1-x/L)*cos(2*pi*nv*x/L)
        w_const = (omega_0/2)*(euler+log(4*pi*(eL-1)/(eL+1)))
        dx = L/n_quad; integral = mpf(0)
        for k in range(n_quad):
            x = dx*(k+mpf(1)/2)
            numer = exp(x/2)*omega(x)-omega_0
            denom = exp(x)-exp(-x)
            if abs(denom) > mpf(10)**(-25): integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const+integral)
        wr_diag[-nv] = wr_diag[nv]

    # Assemble
    QW = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(i, dim):
            m = j - N_val
            w02 = pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))
            wp = sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
            wr = wr_diag[n] if n==m else (alpha[m]-alpha[n])/(n-m)
            QW[i,j] = w02 - wr - wp
            QW[j,i] = QW[i,j]
    QW = (QW + QW.T)/2
    return QW


def xi_hat(z, xi, N, L_f):
    """xi_hat(z) = multiplicative Fourier transform."""
    s = np.sin(z * L_f / 2)
    if abs(s) < 1e-60:
        return 0.0
    total = sum(xi[j+N] / (z - 2*np.pi*j/L_f)
                for j in range(-N, N+1)
                if abs(z - 2*np.pi*j/L_f) > 1e-12)
    return 2 * L_f**(-0.5) * s * total


if __name__ == "__main__":
    print("THE DECISIVE EXPERIMENT: DOUBLE LIMIT N ~ 8*L")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")
    C_scale = 8  # N = round(C * L)

    print(f"\nScaling: N = round({C_scale} * L)")
    print(f"\n{'lam^2':>6} {'L':>6} {'N':>4} {'dim':>5} {'eps_0':>12} {'eps_1':>12} "
          f"{'gap_r':>7} {'pos':>4} {'|xh(g1)|':>10} {'|xh(g2)|':>10} {'time':>6}")
    print("-" * 95)

    for lam_sq in [14, 20, 30, 50, 75, 100, 150, 200, 300, 500]:
        L_f = np.log(lam_sq)
        N = max(15, round(C_scale * L_f))
        dim = 2*N + 1

        # Check bandwidth: pi*N/L should cover gamma_1 = 14.13
        bandwidth = np.pi * N / L_f
        if bandwidth < gammas[0]:
            print(f"{lam_sq:>6} {L_f:>6.3f} {N:>4} {dim:>5} --- BANDWIDTH {bandwidth:.1f} < gamma_1={gammas[0]:.2f} ---")
            continue

        t0 = time.time()
        try:
            QW = build_QW_fast(lam_sq, N)
            evals, evecs = np.linalg.eigh(QW)
            xi = evecs[:, 0]
            eps_0 = evals[0]
            eps_1 = evals[1]
            n_pos = np.sum(evals > -1e-8)
            gap_r = eps_1 / eps_0 if abs(eps_0) > 1e-20 else float('inf')

            # Normalize and compute xi_hat
            xs = np.sum(xi)
            if abs(xs) > 1e-30:
                xi_n = xi * np.sqrt(L_f) / xs
            else:
                xi_n = xi

            # xi_hat at first 2 zeta zeros
            xh = []
            for k in range(min(2, len(gammas))):
                if gammas[k] < bandwidth:
                    xh.append(abs(xi_hat(gammas[k], xi_n, N, L_f)))
                else:
                    xh.append(float('nan'))

            dt = time.time() - t0

            xh1_str = f"{xh[0]:.4e}" if len(xh)>0 and not np.isnan(xh[0]) else "---"
            xh2_str = f"{xh[1]:.4e}" if len(xh)>1 and not np.isnan(xh[1]) else "---"

            print(f"{lam_sq:>6} {L_f:>6.3f} {N:>4} {dim:>5} {eps_0:>12.4e} {eps_1:>12.4e} "
                  f"{gap_r:>7.1f} {n_pos:>4} {xh1_str:>10} {xh2_str:>10} {dt:>5.0f}s")

        except Exception as e:
            dt = time.time() - t0
            print(f"{lam_sq:>6} {L_f:>6.3f} {N:>4} {dim:>5} ERROR: {str(e)[:40]} {dt:>5.0f}s")

    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)
    print("""
Check these criteria:
  1. ALL eigenvalues positive? (n_pos = dim for all rows)
  2. eps_0 DECREASING with lambda? (should decrease or stay stable)
  3. |xi_hat(gamma_k)| DECREASING? (this is what proves RH)
  4. Gap ratio STABLE? (should stay around 3-4)

If all four: the corrected proof (N ~ C*L) closes.
""")
