"""
Session 26d: Dense sweep of prolate overlap to characterize the oscillation.

With fixed SL eigenvalues: overlap = 0.955 at lam^2=50 but 0.04 at lam^2=30.
Is this a sharp resonance? Does it converge at a finer scale?

Dense sweep: lam^2 = 10, 12, 14, ..., 100 (step 2)
Also test n_u dependence for a good case and a bad case.
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
    dim = 2 * N + 1; al = {}
    for n in range(-N, N + 1):
        nn = abs(n)
        if nn == 0: al[n] = mpf(0); continue
        z = exp(-2 * L); a = pi * mpc(0, nn) / L + mpf(1) / 4
        h = hyp2f1(1, a, a + 1, z)
        al[n] = (exp(-L / 2) * (2 * L / (L + 4 * pi * mpc(0, nn)) * h).imag + digamma(a).imag / 2) / pi
        if n < 0: al[n] = -al[n]
    wr_d = {}
    for nv in range(N + 1):
        w_c = euler + log(4 * pi * (eL - 1) / (eL + 1))
        def ig(x, nv=nv): return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        wr_d[nv] = w_c + quad(ig, [mpf(0), L]); wr_d[-nv] = wr_d[nv]
    tau = mpmatrix(dim, dim); L2 = L * L; p2 = 16 * pi * pi; pf = 32 * L * sinh(L / 4) ** 2
    def q_mp(n, m, y):
        if n != m: return (sin(2 * pi * m * y / L) - sin(2 * pi * n * y / L)) / (pi * (n - m))
        else: return 2 * (L - y) / L * cos(2 * pi * n * y / L)
    for i in range(dim):
        n = i - N
        for j in range(i, dim):
            m = j - N
            w02 = pf * (L2 - p2 * m * n) / ((L2 + p2 * m ** 2) * (L2 + p2 * n ** 2))
            wp = sum(lk * pkv ** (-mpf(1) / 2) * q_mp(n, m, logk) for pkv, lk, logk in vM)
            wr = wr_d[n] if n == m else (al[m] - al[n]) / (n - m)
            tau[i, j] = w02 - wr - wp; tau[j, i] = tau[i, j]
    E, ER = eig(tau, left=False, right=True)
    evals = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    idx = evals[0][1]; xi = [float(ER[j, idx].real) for j in range(dim)]
    xs = sum(xi); sqL = float(mpmath.sqrt(L))
    if abs(xs) > 1e-20: xi = [x * sqL / xs for x in xi]
    return np.array(xi) / np.linalg.norm(xi)

def solve_SL_fixed(lam, max_grid=100000):
    gamma = 2 * np.pi * lam**2
    n_grid = min(max_grid, max(500, int(np.ceil(4 * gamma / np.pi)) + 100))
    z = np.linspace(-1, 1, n_grid + 2)[1:-1]; dz = z[1] - z[0]
    z_half = np.empty(n_grid + 1)
    z_half[0] = (-1 + z[0]) / 2; z_half[1:-1] = (z[:-1] + z[1:]) / 2; z_half[-1] = (z[-1] + 1) / 2
    p_half = 1 - z_half**2; q = gamma**2 * (1 - z**2)
    d_diag = -(p_half[:-1] + p_half[1:]) / dz**2 + q
    od = p_half[1:-1] / dz**2
    n_eig = min(20, n_grid)
    evals, evecs = eigh_tridiagonal(d_diag, od, select='i', select_range=(n_grid - n_eig, n_grid - 1))
    even_psi = []
    for i in range(n_eig - 1, -1, -1):
        psi = evecs[:, i]; psi_flip = psi[::-1]
        if np.linalg.norm(psi - psi_flip) < np.linalg.norm(psi + psi_flip):
            psi_norm = psi / np.linalg.norm(psi)
            if len(even_psi) == 0 and psi_norm[len(psi_norm) // 2] < 0: psi_norm = -psi_norm
            even_psi.append(psi_norm)
            if len(even_psi) >= 3: break
    return z, even_psi, n_grid

def apply_E_map(z_grid, psi, lam, L_f, N, n_u=3000):
    dim = 2 * N + 1
    y_grid = np.linspace(-L_f / 2, L_f / 2, n_u)
    u_grid = np.exp(y_grid); dy = y_grid[1] - y_grid[0]
    mid = len(z_grid) // 2
    z_pos = z_grid[mid:]; psi_pos = psi[mid:]
    z0 = z_pos[0]; dz_g = z_pos[1] - z_pos[0]; n_pts = len(z_pos)
    k_vals = np.zeros(n_u)
    for i in range(n_u):
        u = u_grid[i]; n_max = int(lam / u)
        if n_max < 1: continue
        ns = np.arange(1, n_max + 1, dtype=np.float64)
        zs = ns * u / lam
        valid = (zs >= z_pos[0]) & (zs <= z_pos[-1])
        if not np.any(valid): continue
        idx_f = (zs[valid] - z0) / dz_g
        idx_lo = np.floor(idx_f).astype(np.intp)
        idx_lo = np.clip(idx_lo, 0, n_pts - 2)
        frac = idx_f - idx_lo; frac = np.clip(frac, 0, 1)
        vals = psi_pos[idx_lo] * (1 - frac) + psi_pos[idx_lo + 1] * frac
        k_vals[i] = np.sqrt(u) * np.sum(vals)
    j_vals = np.arange(-N, N + 1)
    cos_mat = np.cos(2 * np.pi * np.outer(j_vals, y_grid) / L_f)
    coeffs = cos_mat @ k_vals * dy
    nrm = np.linalg.norm(coeffs)
    if nrm > 0: coeffs /= nrm
    return coeffs


if __name__ == "__main__":
    print("DENSE PROLATE OVERLAP SWEEP (fixed SL)")
    print("=" * 60)

    N = 30
    results = []

    for lam_sq in range(10, 102, 2):
        t0 = time.time()
        lam = np.sqrt(lam_sq); L_f = np.log(lam_sq)

        xi_n = build_xi(lam_sq, N)
        z_grid, efuncs, n_grid = solve_SL_fixed(lam)

        ov_h0 = -1
        if efuncs:
            c = apply_E_map(z_grid, efuncs[0], lam, L_f, N, n_u=3000)
            ov_h0 = abs(np.dot(xi_n, c))

        dt = time.time() - t0
        bar = '#' * int(ov_h0 * 40) if ov_h0 >= 0 else ''
        print(f"  lam^2={lam_sq:>4}: overlap(xi, E(h0)) = {ov_h0:.4f} {bar}  ({dt:.0f}s)")
        results.append((lam_sq, ov_h0))

    print("\n" + "=" * 60)
    print("TREND:")
    highs = [(ls, ov) for ls, ov in results if ov > 0.5]
    lows = [(ls, ov) for ls, ov in results if ov < 0.1]
    print(f"  High overlap (>0.5): {[f'{ls}:{ov:.3f}' for ls, ov in highs]}")
    print(f"  Low overlap (<0.1):  {[f'{ls}' for ls, ov in lows]}")
