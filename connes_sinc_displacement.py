"""
Session 26j: Does the prolate concentration operator (sinc kernel) have
the same displacement structure as the Weil matrix tau?

The sinc kernel on [-L/2, L/2] with bandwidth c = gamma = 2*pi*lam^2:
  K(s,t) = sin(c(s-t)) / (pi(s-t))

In the V_n basis (e^{2*pi*i*n*y/L} on [-L/2, L/2]):
  K_{nm} = (1/L^2) integral integral K(s,t) e^{-2*pi*i*n*s/L} e^{2*pi*i*m*t/L} ds dt

For the sinc kernel, this should simplify because K is a convolution:
  K_{nm} = delta_{nm} * (fraction of the bandwidth occupied by mode n)

Actually, K_{nm} = integral K(s,t) V_n(s) V_m(t)^* ds dt / L^2
         = L * delta_{nm} * min(1, c*L/(2*pi*|n|)) ... no, let me compute numerically.

Key test: does the displacement D*K - K*D have rank <= 2?
And if so, are the generators the same as tau's generators (the b_n sequence)?
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, pi, log, sin, cos, nstr, fabs
from scipy.linalg import svd as np_svd
import time

mp.dps = 50


def build_sinc_kernel_Vn(lam_sq, N=30, n_quad=500):
    """Build the sinc kernel matrix in the V_n basis via numerical quadrature."""
    L = float(log(mpf(lam_sq)))
    gamma = 2 * np.pi * lam_sq  # bandwidth parameter
    dim = 2 * N + 1

    # Quadrature grid on [-L/2, L/2]
    s_grid = np.linspace(-L/2, L/2, n_quad)
    ds = s_grid[1] - s_grid[0]

    # Build the sinc kernel matrix on the spatial grid
    K_spatial = np.zeros((n_quad, n_quad))
    for i in range(n_quad):
        for j in range(n_quad):
            diff = s_grid[i] - s_grid[j]
            if abs(diff) < 1e-15:
                K_spatial[i, j] = gamma / np.pi
            else:
                K_spatial[i, j] = np.sin(gamma * diff) / (np.pi * diff)
    K_spatial *= ds  # quadrature weight

    # Project onto V_n basis: K_{nm} = (1/L^2) sum_{ij} K(s_i,t_j) V_n(s_i)^* V_m(t_j) ds^2
    # V_n(s) = exp(2*pi*i*n*s/L)
    # K_{nm} = (1/L^2) * V^H K_spatial V * ds
    V_mat = np.zeros((n_quad, dim), dtype=complex)
    for idx in range(dim):
        n = idx - N
        V_mat[:, idx] = np.exp(2j * np.pi * n * s_grid / L)

    # K_Vn = V^H K V / L^2 (the V^H includes a ds from the quadrature)
    K_Vn = V_mat.conj().T @ K_spatial @ V_mat * ds / (L * L)

    return K_Vn.real, L  # Should be real for even kernel


def build_tau_float(lam_sq, N=30):
    """Build tau matrix (float64 for speed)."""
    import sympy
    L_mp = log(mpf(lam_sq)); L = float(L_mp)
    eL = float(mpmath.exp(L_mp))
    primes = list(sympy.primerange(2, int(lam_sq) + 1))
    vM = []
    for p in primes:
        pk = p
        while pk <= lam_sq:
            vM.append((pk, np.log(p), np.log(pk)))
            pk *= p
    dim = 2 * N + 1

    # Build tau entries (float64 approximation)
    from mpmath import euler as mp_euler, sinh as mp_sinh, hyp2f1 as mp_hyp, digamma as mp_digamma
    mp.dps = 50

    al = {}
    for n in range(-N, N+1):
        nn = abs(n)
        if nn == 0: al[n] = 0.0; continue
        z = float(mpmath.exp(-2*L_mp)); a = mpmath.pi*mpmath.mpc(0,nn)/L_mp + mpf(1)/4
        h = mp_hyp(1, a, a+1, mpf(z))
        al[n] = float((mpmath.exp(-L_mp/2)*(2*L_mp/(L_mp+4*mpmath.pi*mpmath.mpc(0,nn))*h).imag + mp_digamma(a).imag/2)/mpmath.pi)
        if n < 0: al[n] = -al[n]

    wr_d = {}
    for nv in range(N+1):
        w_c = float(mp_euler + mpmath.log(4*mpmath.pi*(mpmath.exp(L_mp)-1)/(mpmath.exp(L_mp)+1)))
        def ig(x, nv=nv):
            return (mpmath.exp(x/2)*2*(1-x/L_mp)*mpmath.cos(2*mpmath.pi*nv*x/L_mp)-2)/(mpmath.exp(x)-mpmath.exp(-x))
        wr_d[nv] = float(w_c + mpmath.quad(ig, [mpf(0), L_mp]))
        wr_d[-nv] = wr_d[nv]

    pf = float(32*L_mp*mp_sinh(L_mp/4)**2)
    p2 = 16*np.pi**2

    tau = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N
        for j in range(i, dim):
            m = j - N
            w02 = pf * (L**2 - p2*m*n) / ((L**2 + p2*m**2) * (L**2 + p2*n**2))
            wp = sum(lk * pk**(-0.5) * (np.sin(2*np.pi*m*logk/L) - np.sin(2*np.pi*n*logk/L))/(np.pi*(n-m))
                     if n != m else
                     lk * pk**(-0.5) * 2*(L-logk)/L * np.cos(2*np.pi*n*logk/L)
                     for pk, lk, logk in vM)
            wr = wr_d[n] if n == m else (al[m] - al[n]) / (n - m)
            tau[i, j] = w02 - wr - wp
            tau[j, i] = tau[i, j]

    return tau, L


if __name__ == "__main__":
    N = 30; dim = 2 * N + 1

    print("SINC KERNEL vs TAU: DISPLACEMENT RANK COMPARISON")
    print("=" * 70)

    for lam_sq in [14, 50, 100]:
        print(f"\nlam^2 = {lam_sq}")
        t0 = time.time()

        # Build sinc kernel in V_n basis
        K, L = build_sinc_kernel_Vn(lam_sq, N, n_quad=500)

        # Build tau
        tau, _ = build_tau_float(lam_sq, N)

        # Displacement of sinc kernel: D*K - K*D
        D = np.diag(np.arange(-N, N+1, dtype=float))
        disp_K = D @ K - K @ D
        U_K, S_K, Vt_K = np_svd(disp_K)

        # Displacement of tau: D*tau - tau*D
        disp_tau = D @ tau - tau @ D
        U_tau, S_tau, Vt_tau = np_svd(disp_tau)

        print(f"  SINC KERNEL displacement SVD (top 6):")
        print(f"    {', '.join(f'{s:.4e}' for s in S_K[:6])}")
        print(f"    Effective rank: {np.sum(S_K > S_K[0] * 1e-10)}")

        print(f"  TAU displacement SVD (top 6):")
        print(f"    {', '.join(f'{s:.4e}' for s in S_tau[:6])}")
        print(f"    Effective rank: {np.sum(S_tau > S_tau[0] * 1e-10)}")

        # Extract b_n from each (using the 0-th column method)
        b_tau = np.zeros(dim)
        b_sinc = np.zeros(dim)
        for n in range(-N, N+1):
            if n != 0:
                b_tau[n+N] = n * tau[n+N, N]
                b_sinc[n+N] = n * K[n+N, N]

        # Compare generators
        # Normalize for comparison
        bt_norm = np.linalg.norm(b_tau)
        bs_norm = np.linalg.norm(b_sinc)
        if bt_norm > 0 and bs_norm > 0:
            overlap = abs(np.dot(b_tau/bt_norm, b_sinc/bs_norm))
        else:
            overlap = 0

        print(f"  Generator comparison:")
        print(f"    ||b_tau|| = {bt_norm:.6f}")
        print(f"    ||b_sinc|| = {bs_norm:.6f}")
        print(f"    overlap(b_tau, b_sinc) = {overlap:.6f}")
        print(f"    ||b_tau - b_sinc|| = {np.linalg.norm(b_tau - b_sinc):.6f}")

        # Eigenvector comparison
        evals_K, evecs_K = np.linalg.eigh(K)
        evals_tau, evecs_tau = np.linalg.eigh(tau)

        # Largest eigenvector of K (prolate h_0) vs smallest eigenvector of tau (xi)
        xi_tau = evecs_tau[:, 0]  # smallest eigenvalue
        h0_K = evecs_K[:, -1]    # largest eigenvalue (most concentrated)

        ov_eigvec = abs(np.dot(xi_tau, h0_K))
        print(f"  Eigenvector overlap (xi_tau vs h0_sinc): {ov_eigvec:.6f}")
        print(f"  tau min eigenvalue: {evals_tau[0]:.4e}")
        print(f"  sinc max eigenvalue: {evals_K[-1]:.6f}")

        print(f"  ({time.time()-t0:.0f}s)")

    print(f"\n{'='*70}")
    print("If both have displacement rank 2 AND generators match:")
    print("  -> Weil matrix and concentration operator are the SAME in the limit")
    print("  -> educated guess follows")
    print("If displacement ranks differ or generators mismatch:")
    print("  -> different operators, no direct connection")
