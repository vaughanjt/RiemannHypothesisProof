"""
Session 26h: Bridge between Li coefficients and the Weil matrix tau.

The Li coefficients lambda_n = lambda_n^Gamma + lambda_n^zeta.
The Weil matrix tau encodes the Weil distribution in the V_n basis.

Key computations:
1. tr(tau^k) for k=1..20 — moments of the tau eigenvalue distribution
2. Known Li coefficients lambda_n from the exact g_k + d_k formulas
3. Compare: is there a direct relationship lambda_n <-> tr(f(tau))?
4. Decompose lambda_n^zeta in the tau eigenbasis

The displacement rank 2 of tau constrains tr(tau^k) via Newton's identities:
if the characteristic polynomial has low-rank structure, the power sums
(traces) satisfy a recurrence of bounded order.
"""

import numpy as np
import mpmath
import sympy
import time
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad,
                    loggamma, power, fac, nstr, fabs, zeta)

mp.dps = 100


def primes_up_to(n): return list(sympy.primerange(2, int(n) + 1))


def build_tau_mp(lam_sq, N=30):
    """Build tau matrix at full mpmath precision. Return tau and eigenvalues."""
    L = log(mpf(lam_sq)); eL = exp(L); vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq): vM.append((pk, lp, log(pk))); pk *= p
    dim = 2*N+1; al = {}
    for n in range(-N, N+1):
        nn = abs(n)
        if nn == 0: al[n] = mpf(0); continue
        z = exp(-2*L); a = pi*mpc(0,nn)/L + mpf(1)/4
        h = hyp2f1(1,a,a+1,z); al[n] = (exp(-L/2)*(2*L/(L+4*pi*mpc(0,nn))*h).imag + digamma(a).imag/2)/pi
        if n < 0: al[n] = -al[n]
    wr_d = {}
    for nv in range(N+1):
        w_c = euler + log(4*pi*(eL-1)/(eL+1))
        def ig(x, nv=nv): return (exp(x/2)*2*(1-x/L)*cos(2*pi*nv*x/L)-2)/(exp(x)-exp(-x))
        wr_d[nv] = w_c + quad(ig,[mpf(0),L]); wr_d[-nv] = wr_d[nv]
    tau = mpmatrix(dim,dim); L2=L*L; p2=16*pi*pi; pf=32*L*sinh(L/4)**2
    def q_mp(n,m,y):
        if n!=m: return (sin(2*pi*m*y/L)-sin(2*pi*n*y/L))/(pi*(n-m))
        else: return 2*(L-y)/L*cos(2*pi*n*y/L)
    for i in range(dim):
        n=i-N
        for j in range(i,dim):
            m=j-N
            w02=pf*(L2-p2*m*n)/((L2+p2*m**2)*(L2+p2*n**2))
            wp=sum(lk*pkv**(-mpf(1)/2)*q_mp(n,m,logk) for pkv,lk,logk in vM)
            wr=wr_d[n] if n==m else (al[m]-al[n])/(n-m)
            tau[i,j]=w02-wr-wp; tau[j,i]=tau[i,j]

    E_vals = eig(tau, left=False, right=False)
    evals = sorted([E_vals[i].real for i in range(dim)], key=lambda x: float(x))
    return tau, evals, float(L), N


def compute_li_coefficients(n_max):
    """Compute Li coefficients lambda_n from the exact Gamma/Zeta split."""
    # Gamma coefficients g_k (exact formula)
    g = [mpf(0)] * (n_max + 2)
    g[0] = log(mpf(0.5)) - log(pi)/2 + loggamma(mpf(0.5))
    g[1] = mpf(1) - log(pi)/2 + mpmath.polygamma(0, mpf(0.5)) / 2
    for k in range(2, n_max + 2):
        g[k] = power(-1, k) * ((1 - power(2, -k)) * zeta(k) - 1) / k

    # Zeta coefficients d_k via Cauchy integral (FFT)
    n_fft = 1024; r = mpf(2)
    angles = [2 * pi * j / n_fft for j in range(n_fft)]
    vals = []
    for theta in angles:
        s = 1 + r * exp(mpc(0, 1) * theta)
        # log((s-1)*zeta(s))
        val = log((s - 1) * zeta(s))
        vals.append(val)

    d = [mpf(0)] * (n_max + 2)
    for k in range(n_max + 2):
        total = mpc(0, 0)
        for j in range(n_fft):
            total += vals[j] * exp(mpc(0, -2 * pi * k * j / n_fft))
        d[k] = (total / n_fft / power(r, k)).real

    # Li coefficients from binomial sums
    li_gamma = [mpf(0)] * (n_max + 1)
    li_zeta = [mpf(0)] * (n_max + 1)
    li_total = [mpf(0)] * (n_max + 1)

    for n in range(1, n_max + 1):
        sg = mpf(0); sz = mpf(0)
        for j in range(n):
            k = n - j
            if k < len(g):
                sg += mpmath.binomial(n - 1, j) * g[k]
            if k < len(d):
                sz += mpmath.binomial(n - 1, j) * d[k]
        li_gamma[n] = n * sg
        li_zeta[n] = n * sz
        li_total[n] = li_gamma[n] + li_zeta[n]

    return li_gamma, li_zeta, li_total


if __name__ == "__main__":
    print("LI COEFFICIENTS <-> WEIL MATRIX BRIDGE")
    print("=" * 80)

    # 1. Compute Li coefficients
    print("\nComputing Li coefficients (n=1..20)...", flush=True)
    li_g, li_z, li_t = compute_li_coefficients(20)

    print(f"\n{'n':>4} {'lambda_n':>15} {'lambda_n^G':>15} {'lambda_n^z':>15}")
    print("-" * 55)
    for n in range(1, 21):
        print(f"{n:>4} {nstr(li_t[n], 10):>15} {nstr(li_g[n], 10):>15} {nstr(li_z[n], 10):>15}")

    # 2. Build tau and compute moments tr(tau^k)
    for lam_sq in [14, 50]:
        print(f"\n{'='*80}")
        print(f"lam^2 = {lam_sq}: Building tau...", flush=True)
        t0 = time.time()
        tau, evals, L, N = build_tau_mp(lam_sq, N=30)
        dim = 2 * N + 1
        print(f"  Built in {time.time()-t0:.0f}s. Eigenvalue range: [{nstr(evals[0],8)}, {nstr(evals[-1],8)}]")

        # Moments tr(tau^k)
        print(f"\n  Moments tr(tau^k):")
        # Compute from eigenvalues (faster than matrix powers)
        print(f"  {'k':>4} {'tr(tau^k)':>25} {'tr(|tau|^k)':>25}")
        for k in range(1, 11):
            tr_k = sum(ev**k for ev in evals)
            tr_abs_k = sum(fabs(ev)**k for ev in evals)
            print(f"  {k:>4} {nstr(tr_k, 15):>25} {nstr(tr_abs_k, 15):>25}")

        # 3. Key test: relate moments to Li coefficients
        # The Weil explicit formula: sum_rho f_hat(rho) = Q_W(f)
        # For Li's test functions: f_n corresponds to (1-(1-1/rho))^n
        # In the V_n basis, the test vector for lambda_n is related to
        # the Taylor coefficients of the generating function

        # Try: lambda_n^zeta vs eigenvalue sums
        print(f"\n  Comparison: lambda_n^zeta vs tau eigenvalue sums")
        print(f"  {'n':>4} {'lambda_n^z':>15} {'sum e_i^n':>15} {'sum |e_i|^n':>15}")
        for n in range(1, 11):
            lz = li_z[n]
            se = sum(ev**n for ev in evals)
            sae = sum(fabs(ev)**n for ev in evals)
            print(f"  {n:>4} {nstr(lz, 10):>15} {nstr(se, 10):>15} {nstr(sae, 10):>15}")

        # 4. Displacement rank -> recurrence for tr(tau^k)?
        # Newton's identity: k*e_k = sum_{i=1}^k (-1)^{i+1} p_i * e_{k-i}
        # where p_k = tr(tau^k) and e_k are elementary symmetric polynomials
        # For displacement rank 2: the char poly has special structure
        print(f"\n  Newton's identity check (recurrence from displacement):")
        p = [mpf(0)] + [sum(ev**k for ev in evals) for k in range(1, 21)]
        # Check: do the p_k satisfy a low-order recurrence?
        # Try order 2: p_k = a*p_{k-1} + b*p_{k-2}
        for start in [3, 5, 10]:
            if start + 4 <= 20:
                # Fit a, b from two equations
                # p_{start} = a*p_{start-1} + b*p_{start-2}
                # p_{start+1} = a*p_{start} + b*p_{start-1}
                A = mpmatrix(2, 2)
                A[0,0] = p[start-1]; A[0,1] = p[start-2]
                A[1,0] = p[start]; A[1,1] = p[start-1]
                rhs = mpmatrix(2, 1)
                rhs[0,0] = p[start]; rhs[1,0] = p[start+1]
                try:
                    coeffs = mpmath.lu_solve(A, rhs)
                    a_fit, b_fit = coeffs[0], coeffs[1]
                    # Check prediction
                    errs = []
                    for k in range(start+2, min(start+5, 21)):
                        pred = a_fit * p[k-1] + b_fit * p[k-2]
                        err = fabs(pred - p[k]) / fabs(p[k]) if fabs(p[k]) > 0 else fabs(pred)
                        errs.append(float(err))
                    print(f"  Order-2 recurrence from k={start}: "
                          f"a={nstr(a_fit,8)}, b={nstr(b_fit,8)}, "
                          f"pred errors: {[f'{e:.2e}' for e in errs]}")
                except:
                    print(f"  Order-2 recurrence from k={start}: singular")

    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print("  If tr(tau^k) ~ lambda_k^zeta: direct moment connection")
    print("  If order-2 recurrence holds for tr(tau^k): displacement constrains moments")
    print("  If neither: need different test vectors (not identity)")
