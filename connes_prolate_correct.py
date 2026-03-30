"""Session 24: Correct prolate via Sturm-Liouville with gamma=2*pi*lambda^2."""

import numpy as np, mpmath, sympy, time
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad)
from scipy.linalg import eigh_tridiagonal

mp.dps = 50

def primes_up_to(n): return list(sympy.primerange(2, n+1))

def build_xi(lam_sq, N=30):
    L=log(mpf(lam_sq));eL=exp(L);vM=[]
    for p in primes_up_to(lam_sq):
        lp=log(mpf(p));pk=mpf(p)
        while pk<=mpf(lam_sq): vM.append((pk,lp,log(pk)));pk*=p
    dim=2*N+1;al={}
    for n in range(-N,N+1):
        nn=abs(n)
        if nn==0: al[n]=mpf(0);continue
        z=exp(-2*L);a=pi*mpc(0,nn)/L+mpf(1)/4
        h=hyp2f1(1,a,a+1,z);al[n]=(exp(-L/2)*(2*L/(L+4*pi*mpc(0,nn))*h).imag+digamma(a).imag/2)/pi
        if n<0: al[n]=-al[n]
    wr_d={}
    for nv in range(N+1):
        w_c=euler+log(4*pi*(eL-1)/(eL+1))
        def ig(x,nv=nv): return (exp(x/2)*2*(1-x/L)*cos(2*pi*nv*x/L)-2)/(exp(x)-exp(-x))
        wr_d[nv]=w_c+quad(ig,[mpf(0),L]);wr_d[-nv]=wr_d[nv]
    tau=mpmatrix(dim,dim);L2=L*L;p2=16*pi*pi;pf=32*L*sinh(L/4)**2
    def q_mp(n,m,y):
        if n!=m: return (sin(2*pi*m*y/L)-sin(2*pi*n*y/L))/(pi*(n-m))
        else: return 2*(L-y)/L*cos(2*pi*n*y/L)
    for i in range(dim):
        n=i-N
        for j in range(i,dim):
            m=j-N
            w02=pf*(L2-p2*m*n)/((L2+p2*m**2)*(L2+p2*n**2))
            wp=sum(lk*pk**(-mpf(1)/2)*q_mp(n,m,logk) for pk,lk,logk in vM)
            wr=wr_d[n] if n==m else (al[m]-al[n])/(n-m)
            tau[i,j]=w02-wr-wp;tau[j,i]=tau[i,j]
    E,ER=eig(tau,left=False,right=True)
    evals=sorted([(E[i].real,i) for i in range(dim)],key=lambda x:float(x[0]))
    eps=evals[0][0];idx=evals[0][1]
    xi=[float(ER[j,idx].real) for j in range(dim)]
    xs=sum(xi);sqL=float(mpmath.sqrt(L))
    if abs(xs)>1e-20: xi=[x*sqL/xs for x in xi]
    return np.array(xi),float(eps),float(L),N

def solve_SL(lam, n_grid=500):
    """Sturm-Liouville: F_gamma y = d/dz[(1-z^2)dy/dz] + gamma^2(1-z^2)y on [-1,1]."""
    gamma = 2*np.pi*lam**2
    z = np.linspace(-1,1,n_grid+2)[1:-1]
    dz = z[1]-z[0]
    d = np.zeros(n_grid)
    od = np.zeros(n_grid-1)
    for i in range(n_grid):
        zi=z[i]; qi=gamma**2*(1-zi**2)
        if i>0: pl=1-((z[i-1]+z[i])/2)**2
        else: pl=1-((-1+z[0])/2)**2
        if i<n_grid-1: pr=1-((z[i]+z[i+1])/2)**2
        else: pr=1-((z[-1]+1)/2)**2
        d[i]=-(pl+pr)/dz**2+qi
        if i<n_grid-1: od[i]=pr/dz**2
    evals,evecs=eigh_tridiagonal(d,od)
    even=[];eev=[]
    for i in range(min(20,n_grid)):
        psi=evecs[:,i];pf2=psi[::-1]
        if np.linalg.norm(psi-pf2)<np.linalg.norm(psi+pf2):
            even.append(psi/np.linalg.norm(psi));eev.append(evals[i])
            if len(even)>=3: break
    return z,even,eev

def apply_E_map(h_func_interp, z_grid, psi, lam, L_f, N):
    """Apply the map E(h)(u) = u^{1/2} * sum_{n=1}^{Nmax} h(n*u) on [lam^{-1}, lam].

    h is defined on the additive interval [-lam, lam] (via z_grid on [-1,1]).
    E folds it into the multiplicative interval [lam^{-1}, lam].
    Then project onto V_n basis.
    """
    dim = 2 * N + 1

    # Multiplicative grid u in [lam^{-1}, lam], using log-uniform spacing
    n_u = 2000
    u_grid = np.exp(np.linspace(-L_f/2, L_f/2, n_u))  # u in [lam^{-1}, lam]

    # For each u, compute E(h)(u) = u^{1/2} * sum_{n>=1} h(n*u)
    # h(x) is defined on [-lam, lam]; h(x) = 0 for |x| > lam
    # In scaled coords: h(x) = psi(x/lam) for x in [-lam, lam]
    k_vals = np.zeros(n_u)
    for i, u in enumerate(u_grid):
        total = 0.0
        n = 1
        while n * u <= lam:
            x = n * u  # additive argument
            z = x / lam  # scale to [-1, 1]
            if -1 <= z <= 1:
                total += np.interp(z, z_grid, psi)
            # Also include negative: h is even, so h(-n*u) = h(n*u)
            # But n*u > 0 always (n>=1, u>0), and h is even so h(n*u) already counts
            n += 1
        k_vals[i] = np.sqrt(u) * total

    # Project k(u) onto V_n basis: c_j = integral k(u) * conj(V_j(u)) du/u
    # V_j(u) = u^{2*pi*i*j/L}, conj = u^{-2*pi*i*j/L}
    # In log coords y = log(u): c_j = integral k(e^y) * e^{-2*pi*i*j*y/L} dy
    y_grid = np.log(u_grid)
    dy = y_grid[1] - y_grid[0]

    coeffs = np.zeros(dim)
    for j in range(-N, N + 1):
        # Real part since k and V_j projection should give real coefficients for even k
        integrand = k_vals * np.cos(2 * np.pi * j * y_grid / L_f)
        coeffs[j + N] = np.sum(integrand) * dy

    nrm = np.linalg.norm(coeffs)
    if nrm > 0:
        coeffs /= nrm
    return coeffs, k_vals, u_grid


print("CORRECT PROLATE WITH E MAP (Session 24)", flush=True)
print("E(h)(u) = u^{1/2} * sum_{n>=1} h(n*u)", flush=True)
print("=" * 70, flush=True)

for lam_sq in [14, 50, 100, 200, 500, 1000]:
    t0 = time.time()
    lam = np.sqrt(lam_sq); L_f = np.log(lam_sq); N = 30; dim = 2 * N + 1
    xi, eps, _, _ = build_xi(lam_sq, N)
    xi_n = xi / np.linalg.norm(xi)
    z_grid, efuncs, eevals = solve_SL(lam, n_grid=500)

    # Apply E map to each even prolate eigenfunction
    ovs = []; projs = []
    for psi in efuncs:
        coeffs, _, _ = apply_E_map(None, z_grid, psi, lam, L_f, N)
        ov = abs(np.dot(xi_n, coeffs))
        ovs.append(ov)
        projs.append(coeffs)

    # Build k_lambda from paper's combination:
    # h_lambda = (sqrt(3)/2^{11/4}) h_4 - (3/2^{17/4}) h_0
    # But we need to first apply E to h_0 and h_4 separately, then combine
    ov_k = -1
    if len(projs) >= 2:
        h0c = projs[0]; h4c = projs[1]
        # Integral-vanishing: a*h0[N] + b*h4[N] = 0 => b/a = -h0[N]/h4[N]
        # (The paper's exact coefficients may differ; use vanishing-integral condition)
        if abs(h4c[N]) > 1e-15:
            r = -h0c[N] / h4c[N]
            kl = h0c + r * h4c
            nrm = np.linalg.norm(kl)
            if nrm > 0:
                kl /= nrm
                ov_k = abs(np.dot(xi_n, kl))

    dt = time.time() - t0
    gam = 2 * np.pi * lam_sq
    print(f"lam^2={lam_sq:5d} lam={lam:.2f} gamma={gam:.0f}", flush=True)
    for i in range(min(3, len(ovs))):
        print(f"  overlap(xi, E(h_{2*i})) = {ovs[i]:.6f} (eval={eevals[i]:.2e})", flush=True)
    if ov_k >= 0:
        print(f"  overlap(xi, k_lam)     = {ov_k:.6f} <-- KEY", flush=True)
    print(f"  ({dt:.0f}s)\n", flush=True)
