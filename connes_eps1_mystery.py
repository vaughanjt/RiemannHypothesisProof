"""
Session 30d: WHY is eps_1 ~ 7.5e-10 rock-stable?

eps_0 (even mode) plunges from 1.08e-10 to 4.82e-12 over lam^2=1000..5000.
eps_1 (odd mode) stays at 7.2e-10 to 8.3e-10 — barely moves.

eps_1 = <v_odd | Q_W | v_odd> where v_odd is the odd W02 eigenvector.
     = <v_odd | W02 | v_odd> - <v_odd | WR+Wp | v_odd>
     = W02_eigenvalue(odd) - <v_odd | M | v_odd>

W02_eigenvalue(odd) changes with lambda (W02 depends on L).
<v_odd | M | v_odd> also changes.
But their DIFFERENCE stays at ~7.5e-10.

This is a DEEPER cancellation than eps_0.
For eps_0: the cancellation is 10^10-fold and improving.
For eps_1: the cancellation is 10^9-fold and STABLE.

QUESTIONS:
1. What are the W02 eigenvalues as functions of L?
2. What are the M Rayleigh quotients on even/odd vectors as functions of L?
3. Why does the odd difference stabilize while the even difference plunges?
4. Is there a closed-form expression for eps_1 in the limit?
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time

mp.dps = 80


def build_decomposed(lam_sq, N_val, n_quad=20000):
    """Build W02, M=WR+Wp, QW and return eigendecompositions."""
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
            if abs(denom) > mpf(10)**(-60): integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const+integral)
        wr_diag[-nv] = wr_diag[nv]

    M = np.zeros((dim, dim))  # WR + Wp combined
    for i in range(dim):
        n = i - N_val
        M[i,i] = wr_diag[n]
        for j in range(dim):
            m = j - N_val
            if n != m:
                M[i,j] += (alpha[m]-alpha[n])/(n-m)
            M[i,j] += sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
    M = (M + M.T)/2

    QW = W02 - M
    QW = (QW + QW.T)/2
    return W02, M, QW, L_f


if __name__ == "__main__":
    print("THE eps_1 MYSTERY")
    print("=" * 70)

    # ================================================================
    # PART 1: Track W02 eigenvalues and M Rayleigh quotients vs lambda
    # ================================================================
    print("\nPART 1: DECOMPOSITION vs LAMBDA")
    print("-" * 70)

    print(f"\n{'lam^2':>6} {'W02_even':>10} {'W02_odd':>10} {'M_even':>12} {'M_odd':>12} "
          f"{'eps_even':>12} {'eps_odd':>12}")
    print("-" * 80)

    for lam_sq in [14, 50, 100, 500, 1000, 2000, 3000, 5000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))

        t0 = time.time()
        W02, M, QW, _ = build_decomposed(lam_sq, N, n_quad=20000)

        # W02 eigenvectors (even and odd)
        evals_w02, evecs_w02 = np.linalg.eigh(W02)
        idx_nz = np.where(np.abs(evals_w02) > 1e-10)[0]

        center = N
        results = {}
        for idx in idx_nz:
            ev = evecs_w02[:, idx]
            even_score = sum(abs(ev[center+k] - ev[center-k]) for k in range(1, N+1))
            odd_score = sum(abs(ev[center+k] + ev[center-k]) for k in range(1, N+1))
            parity = "even" if even_score < odd_score else "odd"

            w02_eig = evals_w02[idx]
            m_rq = ev @ M @ ev
            qw_rq = ev @ QW @ ev

            results[parity] = {'w02': w02_eig, 'm': m_rq, 'qw': qw_rq}

        if 'even' in results and 'odd' in results:
            r = results
            print(f"{lam_sq:>6} {r['even']['w02']:>10.4f} {r['odd']['w02']:>10.4f} "
                  f"{r['even']['m']:>12.6f} {r['odd']['m']:>12.6f} "
                  f"{r['even']['qw']:>12.4e} {r['odd']['qw']:>12.4e}  ({time.time()-t0:.0f}s)")

    # ================================================================
    # PART 2: What controls eps_1 (odd mode)?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: ODD MODE RAYLEIGH QUOTIENT DECOMPOSITION")
    print("-" * 70)

    # eps_odd = W02_odd - M_odd = W02_eigenvalue(odd) - <v_odd|M|v_odd>
    # What determines <v_odd|M|v_odd>?
    # M = WR + Wp, so <v_odd|M|v_odd> = <v_odd|WR|v_odd> + <v_odd|Wp|v_odd>

    print(f"\n{'lam^2':>6} {'W02_odd':>10} {'WR_odd':>12} {'Wp_odd':>12} "
          f"{'M_odd':>12} {'eps_odd':>12} {'diff_to_7.5e-10':>16}")
    print("-" * 90)

    for lam_sq in [14, 100, 500, 1000, 2000, 5000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        L = log(mpf(lam_sq))
        eL = exp(L)

        W02, M, QW, _ = build_decomposed(lam_sq, N, n_quad=20000)

        # Get odd eigenvector
        evals_w02, evecs_w02 = np.linalg.eigh(W02)
        idx_nz = np.where(np.abs(evals_w02) > 1e-10)[0]
        center = N
        v_odd = None
        w02_odd_eig = None
        for idx in idx_nz:
            ev = evecs_w02[:, idx]
            even_score = sum(abs(ev[center+k] - ev[center-k]) for k in range(1, N+1))
            odd_score = sum(abs(ev[center+k] + ev[center-k]) for k in range(1, N+1))
            if even_score > odd_score:  # odd
                v_odd = ev
                w02_odd_eig = evals_w02[idx]

        if v_odd is not None:
            # Split M into WR and Wp
            # Build WR separately
            vM = []
            limit = min(lam_sq, 10000)
            sieve_arr = [True] * (limit + 1)
            sieve_arr[0] = sieve_arr[1] = False
            for i in range(2, int(limit**0.5)+2):
                if i <= limit and sieve_arr[i]:
                    for j_s in range(i*i, limit+1, i):
                        sieve_arr[j_s] = False
            for p in range(2, limit+1):
                if sieve_arr[p] and p <= lam_sq:
                    pk = p
                    while pk <= lam_sq:
                        vM.append((pk, np.log(p), np.log(pk)))
                        pk *= p

            dim = 2*N+1
            L_f_val = float(L)
            def q_func_inner(n, m, y):
                if n != m:
                    return (np.sin(2*np.pi*m*y/L_f_val) - np.sin(2*np.pi*n*y/L_f_val)) / (np.pi*(n-m))
                else:
                    return 2*(L_f_val - y)/L_f_val * np.cos(2*np.pi*n*y/L_f_val)

            Wp = np.zeros((dim, dim))
            for i in range(dim):
                n = i - N
                for j in range(dim):
                    m_idx = j - N
                    Wp[i,j] = sum(lk*k**(-0.5)*q_func_inner(n,m_idx,logk) for k,lk,logk in vM)
            Wp = (Wp + Wp.T)/2

            WR = M - Wp

            wr_rq = v_odd @ WR @ v_odd
            wp_rq = v_odd @ Wp @ v_odd
            m_rq = v_odd @ M @ v_odd
            qw_rq = v_odd @ QW @ v_odd

            diff = qw_rq - 7.5e-10

            print(f"{lam_sq:>6} {w02_odd_eig:>10.4f} {wr_rq:>12.6f} {wp_rq:>12.6f} "
                  f"{m_rq:>12.6f} {qw_rq:>12.4e} {diff:>16.4e}")

    # ================================================================
    # PART 3: The even/odd RATIO and what drives the plunge
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: EVEN vs ODD — WHAT DRIVES THE PLUNGE?")
    print("-" * 70)

    print(f"\n{'lam^2':>6} {'eps_even':>12} {'eps_odd':>12} {'ratio':>8} "
          f"{'W02_even':>10} {'W02_odd':>10}")
    print("-" * 60)

    for lam_sq in [14, 100, 500, 1000, 2000, 3000, 5000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))

        W02, M, QW, _ = build_decomposed(lam_sq, N, n_quad=15000)

        evals_w02, evecs_w02 = np.linalg.eigh(W02)
        idx_nz = np.where(np.abs(evals_w02) > 1e-10)[0]
        center = N

        for idx in idx_nz:
            ev = evecs_w02[:, idx]
            even_score = sum(abs(ev[center+k] - ev[center-k]) for k in range(1, N+1))
            odd_score = sum(abs(ev[center+k] + ev[center-k]) for k in range(1, N+1))
            parity = "even" if even_score < odd_score else "odd"
            if parity == "even":
                eps_even = ev @ QW @ ev
                w02_even = evals_w02[idx]
            else:
                eps_odd = ev @ QW @ ev
                w02_odd = evals_w02[idx]

        ratio = eps_odd / eps_even if abs(eps_even) > 1e-20 else float('inf')
        print(f"{lam_sq:>6} {eps_even:>12.4e} {eps_odd:>12.4e} {ratio:>8.1f} "
              f"{w02_even:>10.4f} {w02_odd:>10.4f}")

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("=" * 70)
