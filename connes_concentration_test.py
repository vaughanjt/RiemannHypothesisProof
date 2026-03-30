"""
Session 26g: Information-theoretic concentration tests.

Test 1: Concentration ratio R(xi) = energy on critical line / total energy
  R = integral |xi_hat(1/2+it)|^2 dt / integral |xi_hat(sigma+it)|^2 dt d_sigma
  If R(xi) ~ R(prolate) ~ 1, xi IS the concentration maximizer.

Test 2: Leakage outside critical strip
  L_out = integral_{|sigma-1/2|>1/2-delta} |xi_hat(sigma+it)|^2 dt d_sigma
  If L_out ~ O(eps_0), leakage is controlled by the eigenvalue.

Practical computation:
  xi_hat(sigma+it) = integral xi(u) u^{sigma-1+it} du/u
  In V_n basis: xi_hat(sigma+it) = sum_n xi_n * integral V_n(u) u^{sigma-1+it} du/u
  V_n(u) = u^{2*pi*i*n/L}, so:
  integral V_n(u) u^{sigma-1+it} du/u = integral e^{(2*pi*i*n/L + sigma-1/2 + it)y} dy
  on [-L/2, L/2].

  Let alpha_n(sigma,t) = 2*pi*i*n/L + (sigma-1/2) + it
  Then the integral = 2*sinh(alpha_n * L/2) / alpha_n

  xi_hat(sigma+it) = sum_n xi_n * 2*sinh(alpha_n * L/2) / alpha_n
"""

import numpy as np
import mpmath
import sympy
import time
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad)

mp.dps = 50


def primes_up_to(n): return list(sympy.primerange(2, int(n) + 1))


def build_xi(lam_sq, N=30):
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
    E,ER=eig(tau,left=False,right=True)
    evals=sorted([(E[i].real,i) for i in range(dim)],key=lambda x:float(x[0]))
    eps=evals[0][0]; idx=evals[0][1]
    xi=[float(ER[j,idx].real) for j in range(dim)]
    xs=sum(xi); sqL=float(mpmath.sqrt(L))
    if abs(xs)>1e-20: xi=[x*sqL/xs for x in xi]
    return np.array(xi)/np.linalg.norm(xi), float(eps), float(L), N


def mellin_energy(xi, L, N, sigma, t_max=100, n_t=2000):
    """Compute integral_{-t_max}^{t_max} |xi_hat(sigma+it)|^2 dt.

    xi_hat(sigma+it) = sum_n xi_n * 2*sinh(alpha_n*L/2)/alpha_n
    where alpha_n = (sigma-1/2) + i*(2*pi*n/L + t)
    """
    dim = 2 * N + 1
    t_grid = np.linspace(-t_max, t_max, n_t)
    dt = t_grid[1] - t_grid[0]

    energy = 0.0
    for it in range(n_t):
        t = t_grid[it]
        val = 0.0 + 0.0j
        for j in range(dim):
            n = j - N
            alpha = complex(sigma - 0.5, 2 * np.pi * n / L + t)
            if abs(alpha) < 1e-15:
                val += xi[j] * L  # limit of 2*sinh(alpha*L/2)/alpha as alpha->0
            else:
                val += xi[j] * 2 * np.sinh(alpha * L / 2) / alpha
        energy += abs(val)**2

    return energy * dt


if __name__ == "__main__":
    print("CONCENTRATION & LEAKAGE TESTS")
    print("=" * 70)

    N = 30

    for lam_sq in [14, 50, 100, 200]:
        t0 = time.time()
        xi, eps, L, _ = build_xi(lam_sq, N)

        print(f"\nlam^2 = {lam_sq}, L = {L:.4f}, eps_0 = {eps:.4e}")

        # Energy on the critical line (sigma = 1/2, i.e., Im z = 0)
        E_crit = mellin_energy(xi, L, N, sigma=0.5, t_max=50, n_t=1000)

        # Energy at boundary of critical strip (sigma = 0, i.e., Im z = 1/2)
        E_edge = mellin_energy(xi, L, N, sigma=0.0, t_max=50, n_t=1000)

        # Energy well outside (sigma = -1, i.e., Im z = 3/2)
        E_out = mellin_energy(xi, L, N, sigma=-1.0, t_max=50, n_t=1000)

        # Also at sigma = 1 (symmetric to sigma = 0)
        E_edge2 = mellin_energy(xi, L, N, sigma=1.0, t_max=50, n_t=1000)

        # Concentration ratio: critical line vs edge
        R = E_crit / (E_crit + E_edge + E_edge2) if (E_crit + E_edge + E_edge2) > 0 else 0

        dt = time.time() - t0
        print(f"  Energy on critical line (sigma=1/2): {E_crit:.6e}")
        print(f"  Energy at strip edge (sigma=0):      {E_edge:.6e}")
        print(f"  Energy at strip edge (sigma=1):      {E_edge2:.6e}")
        print(f"  Energy outside (sigma=-1):           {E_out:.6e}")
        print(f"  Concentration ratio R:               {R:.6f}")
        print(f"  E_edge / E_crit:                     {E_edge/E_crit:.6e}")
        print(f"  E_out / E_crit:                      {E_out/E_crit:.6e}")
        print(f"  E_edge / |eps_0|:                    {E_edge/abs(eps):.6e}")
        print(f"  ({dt:.0f}s)")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  If R ~ 1: xi is maximally concentrated on critical line")
    print("  If E_edge/E_crit ~ O(eps_0): leakage tracks eigenvalue")
    print("  If E_edge/E_crit ~ O(1): leakage is NOT controlled by eps_0")
