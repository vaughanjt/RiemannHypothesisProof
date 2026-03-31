"""
Session 30e: COMPLEMENT STRUCTURE — how does the 16% achieve 10^8 cancellation?

The min eigenvector xi_0 is 84% in the W02 rank-2 subspace (giving 10^2 cancellation)
and 16% in the (2N-1)-dimensional complement of M eigenvectors (giving 10^8 more).

QUESTIONS:
1. How is the 16% distributed across M eigenvectors?
2. Which M eigenvectors contribute most to the extra cancellation?
3. Is there a pattern (low eigenvalues? high eigenvalues? specific structure?)
4. How does this distribution change with lambda?
5. Can we predict the 10^8 factor analytically?
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time

mp.dps = 80


def build_all(lam_sq, N_val, n_quad=20000):
    """Build W02, M, QW and return everything."""
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
            if abs(denom) > mpf(10)**(-60): integral += numer/denom
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
    print("COMPLEMENT STRUCTURE PROBE")
    print("=" * 70)

    for lam_sq in [50, 500, 2000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        dim = 2*N + 1

        print(f"\n{'='*70}")
        print(f"lam^2={lam_sq}, N={N}, dim={dim}")
        print("-" * 70)

        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N)
        print(f"Built in {time.time()-t0:.0f}s")

        # Eigensystems
        evals_qw, evecs_qw = np.linalg.eigh(QW)
        evals_m, evecs_m = np.linalg.eigh(M)
        evals_w02, evecs_w02 = np.linalg.eigh(W02)

        xi_0 = evecs_qw[:, 0]  # min eigvec of QW
        eps_0 = evals_qw[0]

        # Decompose xi_0 in M eigenbasis
        coeffs_m = evecs_m.T @ xi_0  # c_k = <phi_k | xi_0>

        # How much of xi_0 is in each M eigenvector?
        print(f"\neps_0 = {eps_0:.6e}")
        print(f"\nxi_0 decomposition in M eigenbasis:")
        print(f"  {'k':>4} {'mu_k (M eig)':>14} {'|c_k|^2':>12} {'c_k^2*QW_kk':>14} {'cum%':>7}")

        # Sort by |c_k|^2 descending
        energy = coeffs_m**2
        order = np.argsort(-energy)

        cum = 0
        total_energy = np.sum(energy)
        shown = 0
        for rank, k in enumerate(order):
            cum += energy[k]
            pct = 100 * cum / total_energy
            # What is this M eigenvector's Q_W Rayleigh quotient?
            qw_rq = evecs_m[:, k] @ QW @ evecs_m[:, k]

            if shown < 20 or pct - 100*((cum-energy[k])/total_energy) > 0.5:
                print(f"  {rank+1:>4} {evals_m[k]:>14.6f} {energy[k]:>12.4e} "
                      f"{energy[k]*qw_rq:>14.4e} {pct:>6.1f}%")
                shown += 1
            if pct > 99.9:
                break

        # The CANCELLATION mechanism:
        # xi_0 = sum c_k phi_k (M eigenvectors)
        # <xi_0|QW|xi_0> = sum c_k c_l <phi_k|QW|phi_l>
        # = sum c_k c_l <phi_k|(W02-M)|phi_l>
        # = sum c_k c_l [<phi_k|W02|phi_l> - mu_k delta_{kl}]
        # = sum_k c_k^2 (<phi_k|W02|phi_k> - mu_k) + sum_{k!=l} c_k c_l <phi_k|W02|phi_l>

        # Diagonal contribution: sum c_k^2 * (W02_{kk} - mu_k)
        diag_contrib = sum(coeffs_m[k]**2 * (evecs_m[:,k] @ W02 @ evecs_m[:,k] - evals_m[k])
                          for k in range(dim))
        # Off-diagonal contribution (from W02 coupling between M eigenvectors)
        offdiag_contrib = eps_0 - diag_contrib

        print(f"\n  Cancellation mechanism:")
        print(f"    Diagonal:     sum c_k^2 * (W02_kk - mu_k) = {diag_contrib:.6e}")
        print(f"    Off-diagonal: cross terms from W02        = {offdiag_contrib:.6e}")
        print(f"    Total:        eps_0                        = {eps_0:.6e}")

        # Which W02 cross-couplings matter most?
        # W02 has rank 2, so <phi_k|W02|phi_l> = s1*a_k*a_l + s2*b_k*b_l
        # where a_k = <phi_k|u1>, b_k = <phi_k|u2>
        idx_w02 = np.where(np.abs(evals_w02) > 1e-10)[0]
        print(f"\n  W02 rank-2 coupling in M basis:")
        for idx in idx_w02:
            u = evecs_w02[:, idx]
            s = evals_w02[idx]
            # Project onto M eigenbasis
            proj = evecs_m.T @ u
            # Top couplings
            top_k = np.argsort(-np.abs(proj * coeffs_m))[:5]
            center = N
            even = sum(abs(u[center+k] - u[center-k]) for k in range(1,N+1))
            odd = sum(abs(u[center+k] + u[center-k]) for k in range(1,N+1))
            parity = "EVEN" if even < odd else "ODD"
            print(f"    W02 eigvec (s={s:.4f}, {parity}): top M-coupled components:")
            for kk in top_k:
                print(f"      M eigvec {kk+1} (mu={evals_m[kk]:.4f}): "
                      f"coupling={proj[kk]:.4f}, xi_coeff={coeffs_m[kk]:.4e}, "
                      f"product={proj[kk]*coeffs_m[kk]:.4e}")

        # KEY: What M eigenvalues contribute most to xi_0?
        # Plot the "spectral weight" of xi_0 in M basis
        print(f"\n  Spectral weight |c_k|^2 vs M eigenvalue mu_k:")
        # Bin by M eigenvalue ranges
        mu_bins = [(-100, -10), (-10, -1), (-1, 0), (0, 1), (1, 10), (10, 100)]
        for lo, hi in mu_bins:
            mask = (evals_m >= lo) & (evals_m < hi)
            weight = np.sum(energy[mask])
            n_in = np.sum(mask)
            if n_in > 0:
                print(f"    mu in [{lo:>5},{hi:>5}): {n_in:>3} vectors, "
                      f"weight = {weight:.4e} ({100*weight/total_energy:.1f}%)")

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print("=" * 70)
