"""
Session 26e: Large sweep to test Grok's prediction.

Prediction: good windows (overlap > 0.5) become WIDER and MORE FREQUENT
as lambda grows, because prime density increases.

Sweep lam^2 = 10..500 step 5, compute SPAN overlap (not just h0).
Mark where new primes enter.
"""

import numpy as np
import mpmath
import sympy
import time
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad)
from scipy.linalg import eigh_tridiagonal

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
    idx=evals[0][1]; xi=[float(ER[j,idx].real) for j in range(dim)]
    xs=sum(xi); sqL=float(mpmath.sqrt(L))
    if abs(xs)>1e-20: xi=[x*sqL/xs for x in xi]
    return np.array(xi)/np.linalg.norm(xi)

def solve_SL_fixed(lam, max_grid=100000):
    gamma=2*np.pi*lam**2
    n_grid=min(max_grid, max(500, int(np.ceil(4*gamma/np.pi))+100))
    z=np.linspace(-1,1,n_grid+2)[1:-1]; dz=z[1]-z[0]
    zh=np.empty(n_grid+1); zh[0]=(-1+z[0])/2; zh[1:-1]=(z[:-1]+z[1:])/2; zh[-1]=(z[-1]+1)/2
    ph=1-zh**2; q=gamma**2*(1-z**2)
    d_diag=-(ph[:-1]+ph[1:])/dz**2+q; od=ph[1:-1]/dz**2
    ne=min(20,n_grid)
    evals,evecs=eigh_tridiagonal(d_diag,od,select='i',select_range=(n_grid-ne,n_grid-1))
    even_psi=[]
    for i in range(ne-1,-1,-1):
        psi=evecs[:,i]; pf2=psi[::-1]
        if np.linalg.norm(psi-pf2)<np.linalg.norm(psi+pf2):
            pn=psi/np.linalg.norm(psi)
            if len(even_psi)==0 and pn[len(pn)//2]<0: pn=-pn
            even_psi.append(pn)
            if len(even_psi)>=3: break
    return z, even_psi, n_grid

def apply_E(z_grid, psi, lam, L_f, N, n_u=3000):
    dim=2*N+1; yg=np.linspace(-L_f/2,L_f/2,n_u); ug=np.exp(yg); dy=yg[1]-yg[0]
    mid=len(z_grid)//2; zp=z_grid[mid:]; pp=psi[mid:]
    z0=zp[0]; dzg=zp[1]-zp[0]; npt=len(zp)
    kv=np.zeros(n_u)
    for i in range(n_u):
        u=ug[i]; nm=int(lam/u)
        if nm<1: continue
        ns=np.arange(1,nm+1,dtype=np.float64); zs=ns*u/lam
        valid=(zs>=zp[0])&(zs<=zp[-1])
        if not np.any(valid): continue
        idf=(zs[valid]-z0)/dzg; ilo=np.floor(idf).astype(np.intp)
        ilo=np.clip(ilo,0,npt-2); fr=idf-ilo; fr=np.clip(fr,0,1)
        vals=pp[ilo]*(1-fr)+pp[ilo+1]*fr
        kv[i]=np.sqrt(u)*np.sum(vals)
    jv=np.arange(-N,N+1)
    cm=np.cos(2*np.pi*np.outer(jv,yg)/L_f)
    return cm@kv*dy  # raw coefficients

if __name__ == "__main__":
    N = 30
    primes_list = list(sympy.primerange(2, 510))

    print("LARGE SWEEP: SPAN OVERLAP vs lam^2 (step 5)")
    print("P = new prime at this lam^2")
    print("=" * 70)

    results = []
    for lam_sq in range(10, 505, 5):
        t0 = time.time()
        lam = np.sqrt(lam_sq); L_f = np.log(lam_sq)

        # Mark new primes
        new_prime = any(p == lam_sq or (lam_sq - 5 < p <= lam_sq) for p in primes_list)
        marker = "P" if new_prime else " "

        xi_n = build_xi(lam_sq, N)
        z_grid, efuncs, _ = solve_SL_fixed(lam)

        # Span projection
        vecs = [apply_E(z_grid, psi, lam, L_f, N, n_u=3000) for psi in efuncs]
        V = np.column_stack(vecs)
        try:
            coeffs = np.linalg.lstsq(V, xi_n, rcond=None)[0]
            proj = V @ coeffs
            span_ov = np.linalg.norm(proj)
        except:
            span_ov = -1

        # h0 overlap
        h0_nrm = np.linalg.norm(vecs[0])
        ov_h0 = abs(np.dot(xi_n, vecs[0] / h0_nrm)) if h0_nrm > 0 else 0

        dt = time.time() - t0
        bar = '#' * int(span_ov * 30) if span_ov >= 0 else ''
        print(f" {marker} {lam_sq:>4}: h0={ov_h0:.3f} SPAN={span_ov:.3f} {bar} ({dt:.0f}s)")
        results.append((lam_sq, ov_h0, span_ov))

    # Summary statistics
    print("\n" + "=" * 70)
    good = [(ls, s) for ls, _, s in results if s > 0.5]
    total = len(results)
    print(f"Good windows (span>0.5): {len(good)}/{total} = {100*len(good)/total:.0f}%")
    print(f"  Locations: {[ls for ls, _ in good]}")

    # Check if windows get wider at larger lambda
    good_below_100 = sum(1 for ls, s in good if ls <= 100)
    good_above_100 = sum(1 for ls, s in good if ls > 100)
    count_below = sum(1 for ls, _, _ in results if ls <= 100)
    count_above = sum(1 for ls, _, _ in results if ls > 100)
    print(f"\n  Below lam^2=100: {good_below_100}/{count_below} good ({100*good_below_100/max(1,count_below):.0f}%)")
    print(f"  Above lam^2=100: {good_above_100}/{count_above} good ({100*good_above_100/max(1,count_above):.0f}%)")
    print()
    if good_above_100 > good_below_100:
        print("GROK'S PREDICTION CONFIRMED: good windows more frequent at large lambda")
    else:
        print("GROK'S PREDICTION NOT CONFIRMED: good windows NOT more frequent")
