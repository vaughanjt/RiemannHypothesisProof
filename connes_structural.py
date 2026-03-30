"""
Session 26f: Structural proof path — decompose tau into free + prime parts.

tau = W_{0,2} - W_r - W_p = (W_{0,2} - W_r) + (-W_p) = tau_free + tau_prime

Key questions:
1. What is the minimal eigenvector of tau_free (no primes)?
2. Is it the prolate function E(h_0)?
3. How much does W_p perturb the eigenvector direction?
4. As lambda grows (more primes), does the perturbation decrease?

If tau_free has the prolate direction as its null vector, and W_p is a
perturbation whose effect on the eigenvector direction vanishes as
lambda -> infinity, the educated guess follows.
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

def build_decomposed(lam_sq, N=30):
    """Build tau_free (W02-Wr) and tau_prime (-Wp) separately."""
    L = log(mpf(lam_sq)); eL = exp(L); vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq): vM.append((pk, lp, log(pk))); pk *= p
    dim = 2*N+1; al = {}
    for n in range(-N, N+1):
        nn = abs(n)
        if nn == 0: al[n] = mpf(0); continue
        z = exp(-2*L); a = pi*mpc(0,nn)/L + mpf(1)/4
        h = hyp2f1(1,a,a+1,z)
        al[n] = (exp(-L/2)*(2*L/(L+4*pi*mpc(0,nn))*h).imag + digamma(a).imag/2)/pi
        if n < 0: al[n] = -al[n]
    wr_d = {}
    for nv in range(N+1):
        w_c = euler + log(4*pi*(eL-1)/(eL+1))
        def ig(x, nv=nv): return (exp(x/2)*2*(1-x/L)*cos(2*pi*nv*x/L)-2)/(exp(x)-exp(-x))
        wr_d[nv] = w_c + quad(ig,[mpf(0),L]); wr_d[-nv] = wr_d[nv]

    tau_free = mpmatrix(dim, dim)
    tau_prime = mpmatrix(dim, dim)
    L2 = L*L; p2 = 16*pi*pi; pf = 32*L*sinh(L/4)**2

    def q_mp(n, m, y):
        if n != m: return (sin(2*pi*m*y/L)-sin(2*pi*n*y/L))/(pi*(n-m))
        else: return 2*(L-y)/L*cos(2*pi*n*y/L)

    for i in range(dim):
        n = i-N
        for j in range(i, dim):
            m = j-N
            w02 = pf*(L2-p2*m*n)/((L2+p2*m**2)*(L2+p2*n**2))
            wp = sum(lk*pkv**(-mpf(1)/2)*q_mp(n,m,logk) for pkv,lk,logk in vM)
            wr = wr_d[n] if n==m else (al[m]-al[n])/(n-m)

            tau_free[i,j] = w02 - wr
            tau_free[j,i] = tau_free[i,j]
            tau_prime[i,j] = -wp
            tau_prime[j,i] = tau_prime[i,j]

    return tau_free, tau_prime, float(L), N


def get_min_eigvec(M, dim):
    """Get minimum eigenvector of mpmath matrix."""
    E, ER = eig(M, left=False, right=True)
    evals = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    eps = evals[0][0]; idx = evals[0][1]
    xi = np.array([float(ER[j, idx].real) for j in range(dim)])
    xi /= np.linalg.norm(xi)
    return xi, float(eps), float(evals[1][0])


def solve_SL_fixed(lam, max_grid=100000):
    gamma = 2*np.pi*lam**2
    n_grid = min(max_grid, max(500, int(np.ceil(4*gamma/np.pi))+100))
    z = np.linspace(-1,1,n_grid+2)[1:-1]; dz = z[1]-z[0]
    zh = np.empty(n_grid+1); zh[0]=(-1+z[0])/2; zh[1:-1]=(z[:-1]+z[1:])/2; zh[-1]=(z[-1]+1)/2
    ph = 1-zh**2; q = gamma**2*(1-z**2)
    d_diag = -(ph[:-1]+ph[1:])/dz**2+q; od = ph[1:-1]/dz**2
    ne = min(20, n_grid)
    evals, evecs = eigh_tridiagonal(d_diag, od, select='i', select_range=(n_grid-ne, n_grid-1))
    even_psi = []
    for i in range(ne-1,-1,-1):
        psi = evecs[:,i]; pf2 = psi[::-1]
        if np.linalg.norm(psi-pf2) < np.linalg.norm(psi+pf2):
            pn = psi/np.linalg.norm(psi)
            if len(even_psi)==0 and pn[len(pn)//2]<0: pn=-pn
            even_psi.append(pn)
            if len(even_psi)>=3: break
    return z, even_psi, n_grid


def apply_E(z_grid, psi, lam, L_f, N, n_u=3000):
    dim = 2*N+1; yg = np.linspace(-L_f/2,L_f/2,n_u); ug = np.exp(yg); dy = yg[1]-yg[0]
    mid = len(z_grid)//2; zp = z_grid[mid:]; pp = psi[mid:]
    z0 = zp[0]; dzg = zp[1]-zp[0]; npt = len(zp)
    kv = np.zeros(n_u)
    for i in range(n_u):
        u = ug[i]; nm = int(lam/u)
        if nm < 1: continue
        ns = np.arange(1,nm+1,dtype=np.float64); zs = ns*u/lam
        valid = (zs>=zp[0])&(zs<=zp[-1])
        if not np.any(valid): continue
        idf = (zs[valid]-z0)/dzg; ilo = np.floor(idf).astype(np.intp)
        ilo = np.clip(ilo,0,npt-2); fr = idf-ilo; fr = np.clip(fr,0,1)
        vals = pp[ilo]*(1-fr)+pp[ilo+1]*fr
        kv[i] = np.sqrt(u)*np.sum(vals)
    jv = np.arange(-N,N+1)
    cm = np.cos(2*np.pi*np.outer(jv,yg)/L_f)
    coeffs = cm@kv*dy
    nrm = np.linalg.norm(coeffs)
    if nrm > 0: coeffs /= nrm
    return coeffs


if __name__ == "__main__":
    print("STRUCTURAL DECOMPOSITION: tau = tau_free + tau_prime")
    print("=" * 80)
    N = 30; dim = 2*N+1

    for lam_sq in [14, 50, 100, 200, 500]:
        t0 = time.time()
        lam = np.sqrt(lam_sq); L_f = np.log(lam_sq)
        print(f"\nlam^2 = {lam_sq}", flush=True)

        # Build decomposed matrices
        tau_free, tau_prime, L, _ = build_decomposed(lam_sq, N)
        tau_full = mpmatrix(dim, dim)
        for i in range(dim):
            for j in range(dim):
                tau_full[i,j] = tau_free[i,j] + tau_prime[i,j]

        # Eigenvectors
        xi_full, eps_full, eps1_full = get_min_eigvec(tau_full, dim)
        xi_free, eps_free, eps1_free = get_min_eigvec(tau_free, dim)

        # Prolate E(h_0)
        z_grid, efuncs, _ = solve_SL_fixed(lam)
        prolate = apply_E(z_grid, efuncs[0], lam, L_f, N) if efuncs else np.zeros(dim)

        # Overlaps
        ov_full_free = abs(np.dot(xi_full, xi_free))
        ov_full_prolate = abs(np.dot(xi_full, prolate))
        ov_free_prolate = abs(np.dot(xi_free, prolate))

        # Matrix norms
        tau_free_np = np.array([[float(tau_free[i,j]) for j in range(dim)] for i in range(dim)])
        tau_prime_np = np.array([[float(tau_prime[i,j]) for j in range(dim)] for i in range(dim)])
        norm_free = np.linalg.norm(tau_free_np, ord=2)
        norm_prime = np.linalg.norm(tau_prime_np, ord=2)

        dt = time.time() - t0
        print(f"  Eigenvalues:")
        print(f"    tau_full:  eps_0 = {eps_full:.4e}, eps_1 = {eps1_full:.4e}")
        print(f"    tau_free:  eps_0 = {eps_free:.4e}, eps_1 = {eps1_free:.4e}")
        print(f"  Matrix norms:")
        print(f"    ||tau_free|| = {norm_free:.4f}")
        print(f"    ||tau_prime|| (= ||W_p||) = {norm_prime:.4f}")
        print(f"    ratio ||W_p||/||tau_free|| = {norm_prime/norm_free:.4f}")
        print(f"  Overlaps:")
        print(f"    <xi_full, xi_free>    = {ov_full_free:.6f}  (full vs free eigenvectors)")
        print(f"    <xi_full, E(h_0)>     = {ov_full_prolate:.6f}  (full vs prolate)")
        print(f"    <xi_free, E(h_0)>     = {ov_free_prolate:.6f}  (free vs prolate)")
        print(f"  ({dt:.0f}s)")

    print("\n" + "=" * 80)
    print("KEY QUESTION:")
    print("  If <xi_free, E(h_0)> ~ 1: tau_free has prolate as its null vector")
    print("  If <xi_full, xi_free> ~ 1: W_p doesn't change the eigenvector direction")
    print("  Both together => educated guess follows from perturbation theory")
