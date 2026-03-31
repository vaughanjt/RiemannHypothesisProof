"""
Session 30 autonomous iteration 3: Cross-term structure.

The min eigenvector optimizes over signal (rank-26) and null space of M.
W02 restricted to null(M) determines the optimization landscape.

COMPUTE:
1. W02 projection onto null(M): how much of u1, u2 lives in null(M)?
2. W02 eigenvalues on null(M): positive? negative?
3. Scaling with D (null space dimension)
4. The actual cross-term sum that achieves cancellation
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time

mp.dps = 50


def build_all(lam_sq, N_val, n_quad=10000):
    """Build W02, M, QW."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

    vM = []
    limit = min(lam_sq, 10000)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5)+2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    for p in range(2, limit+1):
        if sieve[p] and p <= lam_sq:
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
        if n == 0: alpha[n] = 0.0
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

    M = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        M[i,i] = wr_diag[n]
        for j in range(dim):
            m = j - N_val
            if n != m: M[i,j] += (alpha[m]-alpha[n])/(n-m)
            M[i,j] += sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
    M = (M + M.T)/2
    QW = W02 - M; QW = (QW + QW.T)/2
    return W02, M, QW


if __name__ == "__main__":
    print("CROSS-TERM STRUCTURE ANALYSIS")
    print("=" * 70)

    for lam_sq in [50, 200, 1000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        dim = 2*N + 1

        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N)
        print(f"\nlam^2={lam_sq}, N={N}, dim={dim} ({time.time()-t0:.0f}s)")
        print("-" * 60)

        # M eigensystem
        evals_m, evecs_m = np.linalg.eigh(M)
        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4

        # Signal and null subspaces
        signal_idx = np.where(abs_evals >= threshold)[0]
        null_idx = np.where(abs_evals < threshold)[0]
        D_signal = len(signal_idx)
        D_null = len(null_idx)

        # Projectors
        P_signal = evecs_m[:, signal_idx]  # dim x D_signal
        P_null = evecs_m[:, null_idx]      # dim x D_null

        # W02 eigenvectors
        evals_w02, evecs_w02 = np.linalg.eigh(W02)
        idx_w02 = np.where(np.abs(evals_w02) > 1e-10)[0]
        center = N

        print(f"  Signal: {D_signal} vectors, Null: {D_null} vectors")

        for idx in idx_w02:
            u = evecs_w02[:, idx]
            s = evals_w02[idx]
            even = sum(abs(u[center+k] - u[center-k]) for k in range(1,N+1))
            odd = sum(abs(u[center+k] + u[center-k]) for k in range(1,N+1))
            parity = "EVEN" if even < odd else "ODD"

            # Projection onto signal and null
            proj_signal = np.linalg.norm(P_signal.T @ u)**2
            proj_null = np.linalg.norm(P_null.T @ u)**2

            print(f"\n  W02 {parity} eigvec (s={s:.4f}):")
            print(f"    ||P_signal u||^2 = {proj_signal:.8f}")
            print(f"    ||P_null u||^2   = {proj_null:.8f}")
            print(f"    Ratio: {proj_signal/proj_null:.1f}x in signal")

        # W02 restricted to null(M)
        W02_null = P_null.T @ W02 @ P_null  # D_null x D_null
        evals_w02_null = np.linalg.eigvalsh(W02_null)

        print(f"\n  W02 restricted to null(M) ({D_null}x{D_null}):")
        print(f"    Eigenvalues: min={evals_w02_null[0]:.6e}, max={evals_w02_null[-1]:.6e}")
        print(f"    Positive: {np.sum(evals_w02_null > 0)}, "
              f"Negative: {np.sum(evals_w02_null < -1e-12)}, "
              f"Zero: {np.sum(np.abs(evals_w02_null) <= 1e-12)}")
        print(f"    Trace: {np.sum(evals_w02_null):.6e}")

        # Q_W restricted to null(M)
        QW_null = P_null.T @ QW @ P_null
        evals_qw_null = np.linalg.eigvalsh(QW_null)

        print(f"\n  Q_W restricted to null(M):")
        print(f"    min eig = {evals_qw_null[0]:.6e}")
        print(f"    max eig = {evals_qw_null[-1]:.6e}")
        print(f"    All positive? {np.all(evals_qw_null > -1e-12)}")

        # FULL Q_W eigenvalues for comparison
        evals_qw = np.linalg.eigvalsh(QW)
        eps_0 = evals_qw[0]

        print(f"\n  Full Q_W: eps_0 = {eps_0:.6e}")
        print(f"  Ratio eps_0 / min(QW_null) = {eps_0/evals_qw_null[0]:.4f}")

    # ================================================================
    # PART 2: How do W02 null projections scale with D?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: W02 NULL PROJECTIONS vs D")
    print("-" * 70)

    lam_sq = 200

    print(f"\n{'N':>4} {'D_null':>6} {'||Pn*u1||^2':>12} {'||Pn*u2||^2':>12} "
          f"{'W02null_min':>12} {'W02null_max':>12} {'QWnull_min':>12}")
    print("-" * 80)

    for N in [20, 25, 30, 35, 40, 50]:
        W02, M, QW = build_all(lam_sq, N, n_quad=8000)
        dim = 2*N+1

        evals_m, evecs_m = np.linalg.eigh(M)
        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        null_idx = np.where(abs_evals < threshold)[0]
        P_null = evecs_m[:, null_idx]
        D_null = len(null_idx)

        evals_w02, evecs_w02 = np.linalg.eigh(W02)
        idx_w02 = np.where(np.abs(evals_w02) > 1e-10)[0]

        proj_nulls = []
        for idx in idx_w02:
            u = evecs_w02[:, idx]
            proj_nulls.append(np.linalg.norm(P_null.T @ u)**2)

        W02_null = P_null.T @ W02 @ P_null
        evals_w02n = np.linalg.eigvalsh(W02_null)

        QW_null = P_null.T @ QW @ P_null
        evals_qwn = np.linalg.eigvalsh(QW_null)

        print(f"{N:>4} {D_null:>6} {proj_nulls[0]:>12.6e} {proj_nulls[1]:>12.6e} "
              f"{evals_w02n[0]:>12.4e} {evals_w02n[-1]:>12.4e} {evals_qwn[0]:>12.4e}")

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print("=" * 70)
