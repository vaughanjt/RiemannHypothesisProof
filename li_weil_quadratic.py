"""
Session 26i: The correct Li-Weil bridge — QUADRATIC form, not trace.

lambda_n^trunc = v_n^T * tau * v_n

where v_n = V_j-projection of f_n(y) = y^n on [-L/2, L/2].

v_j^(n) = (1/L) * integral_{-L/2}^{L/2} y^n * exp(-2*pi*i*j*y/L) dy

Then test: does v_n^T tau v_n match lambda_n or lambda_n^zeta?
"""

import numpy as np
import mpmath
import sympy
import time
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad,
                    loggamma, power, fac, nstr, fabs, zeta, mpmathify)

mp.dps = 100


def primes_up_to(n): return list(sympy.primerange(2, int(n) + 1))


def build_tau(lam_sq, N=30):
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
    return tau, float(L), N


def project_yn(k, L, N):
    """Project f(y) = y^k onto V_j basis.

    v_j = (1/L) integral_{-L/2}^{L/2} y^k exp(-2*pi*i*j*y/L) dy

    Returns complex vector of coefficients (dimension 2N+1).
    """
    dim = 2 * N + 1
    L_mp = mpf(L)
    v = [mpc(0, 0)] * dim

    for idx in range(dim):
        j = idx - N
        if j == 0:
            # (1/L) integral y^k dy on [-L/2, L/2]
            if k % 2 == 0:
                v[idx] = mpc(2 * (L_mp / 2) ** (k + 1) / ((k + 1) * L_mp), 0)
            else:
                v[idx] = mpc(0, 0)  # odd power on symmetric interval
        else:
            # Numerical integration
            omega = 2 * pi * j / L_mp
            def re_integrand(y):
                return y ** k * cos(omega * y) / L_mp
            def im_integrand(y):
                return -y ** k * sin(omega * y) / L_mp
            re_part = quad(re_integrand, [-L_mp / 2, L_mp / 2])
            im_part = quad(im_integrand, [-L_mp / 2, L_mp / 2])
            v[idx] = mpc(re_part, im_part)

    return v


def quadratic_form(v, tau, dim):
    """Compute v^H * tau * v (Hermitian quadratic form)."""
    result = mpc(0, 0)
    for i in range(dim):
        for j in range(dim):
            result += v[i].conjugate() * tau[i, j] * v[j]
    return result


def compute_li(n_max):
    """Compute Li coefficients from exact formulas."""
    g = [mpf(0)] * (n_max + 2)
    g[0] = log(mpf(0.5)) - log(pi) / 2 + loggamma(mpf(0.5))
    g[1] = mpf(1) - log(pi) / 2 + mpmath.polygamma(0, mpf(0.5)) / 2
    for k in range(2, n_max + 2):
        g[k] = power(-1, k) * ((1 - power(2, -k)) * zeta(k) - 1) / k

    n_fft = 1024; r = mpf(2)
    vals = []
    for jj in range(n_fft):
        theta = 2 * pi * jj / n_fft
        s = 1 + r * exp(mpc(0, 1) * theta)
        vals.append(log((s - 1) * zeta(s)))
    d = [mpf(0)] * (n_max + 2)
    for k in range(n_max + 2):
        total = mpc(0, 0)
        for jj in range(n_fft):
            total += vals[jj] * exp(mpc(0, -2 * pi * k * jj / n_fft))
        d[k] = (total / n_fft / power(r, k)).real

    li_g = [mpf(0)] * (n_max + 1)
    li_z = [mpf(0)] * (n_max + 1)
    for n in range(1, n_max + 1):
        sg = sz = mpf(0)
        for j in range(n):
            k = n - j
            if k < len(g): sg += mpmath.binomial(n - 1, j) * g[k]
            if k < len(d): sz += mpmath.binomial(n - 1, j) * d[k]
        li_g[n] = n * sg; li_z[n] = n * sz
    return li_g, li_z


if __name__ == "__main__":
    print("LI-WEIL QUADRATIC BRIDGE: lambda_n = v_n^T tau v_n")
    print("=" * 80)

    N_MAX_LI = 15

    # Li coefficients
    print("Computing Li coefficients...", flush=True)
    li_g, li_z = compute_li(N_MAX_LI)

    for lam_sq in [14, 50]:
        print(f"\n{'='*80}")
        print(f"lam^2 = {lam_sq}", flush=True)

        t0 = time.time()
        tau, L, N = build_tau(lam_sq, 30)
        dim = 2 * N + 1
        print(f"  tau built ({time.time()-t0:.0f}s), L = {L:.4f}")

        print(f"\n  {'n':>4} {'v^T tau v':>18} {'lambda_n':>18} {'lambda_n^G':>15} {'lambda_n^z':>15} {'match?':>10}")
        print(f"  {'-'*4} {'-'*18} {'-'*18} {'-'*15} {'-'*15} {'-'*10}")

        for n in range(1, N_MAX_LI + 1):
            # Project y^n onto V_j basis
            v = project_yn(n, L, N)

            # Quadratic form
            qf = quadratic_form(v, tau, dim)
            qf_real = qf.real  # should be real for Hermitian form

            # Li values
            li_total = li_g[n] + li_z[n]

            # Check what it matches
            diff_total = float(fabs(qf_real - li_total))
            diff_zeta = float(fabs(qf_real - li_z[n]))
            diff_gamma = float(fabs(qf_real - li_g[n]))

            if diff_total < 0.01:
                match = "TOTAL"
            elif diff_zeta < 0.01:
                match = "ZETA"
            elif diff_gamma < 0.01:
                match = "GAMMA"
            else:
                match = f"none({diff_total:.2f})"

            print(f"  {n:>4} {nstr(qf_real, 12):>18} {nstr(li_total, 12):>18} "
                  f"{nstr(li_g[n], 10):>15} {nstr(li_z[n], 10):>15} {match:>10}")

        print(f"\n  Time: {time.time()-t0:.0f}s")

    print(f"\n{'='*80}")
    print("If match=TOTAL: v^T tau v = lambda_n (full Li coefficient)")
    print("If match=ZETA:  v^T tau v = lambda_n^zeta (arithmetic part only)")
    print("If match=none:  need different normalization or test function")
