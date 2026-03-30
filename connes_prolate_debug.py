"""
Session 26b: Debug prolate overlap computation.

The overlap oscillates at 0.005-0.315 instead of converging to 1.
Possible issues:
1. SL eigenfunctions wrong (wrong gamma, wrong eigenfunctions)
2. E map formula wrong (wrong scaling, missing terms)
3. V_n projection wrong (missing sine components, wrong normalization)
4. Grid resolution artifacts (n_grid or n_u too coarse)
5. Fundamental: prolate functions CAN'T approximate xi because W_p distorts it

Diagnostics:
A. Verify SL h_0 is smooth, positive, peaked at z=0 (ground state)
B. Check E(h_0) in V_n basis — is it even? Does it have right symmetry?
C. Vary n_u from 500 to 10000 — does overlap change?
D. Compute overlap of xi with RANDOM vectors for baseline
E. Check the V_n projection includes BOTH cos and sin components
F. Compute W_p contribution to xi separately
"""

import numpy as np
import mpmath
import sympy
import time
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad)
from scipy.linalg import eigh_tridiagonal

mp.dps = 50


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def build_xi(lam_sq, N=30):
    L = log(mpf(lam_sq)); eL = exp(L)
    vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq):
            vM.append((pk, lp, log(pk))); pk *= p
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
        def ig(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
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
    eps = evals[0][0]; idx = evals[0][1]
    xi = [float(ER[j, idx].real) for j in range(dim)]
    xs = sum(xi); sqL = float(mpmath.sqrt(L))
    if abs(xs) > 1e-20: xi = [x * sqL / xs for x in xi]
    return np.array(xi), float(eps), float(L), N


def solve_SL(lam, max_grid=100000):
    gamma = 2 * np.pi * lam**2
    n_grid = min(max_grid, max(500, int(np.ceil(4 * gamma / np.pi)) + 100))
    z = np.linspace(-1, 1, n_grid + 2)[1:-1]
    dz = z[1] - z[0]
    z_half = np.empty(n_grid + 1)
    z_half[0] = (-1 + z[0]) / 2
    z_half[1:-1] = (z[:-1] + z[1:]) / 2
    z_half[-1] = (z[-1] + 1) / 2
    p_half = 1 - z_half**2
    q = gamma**2 * (1 - z**2)
    d = -(p_half[:-1] + p_half[1:]) / dz**2 + q
    od = p_half[1:-1] / dz**2
    n_eig = min(20, n_grid)
    evals, evecs = eigh_tridiagonal(d, od, select='i', select_range=(0, n_eig - 1))
    even_psi = []; even_evals = []; even_idx = 0
    for i in range(n_eig):
        psi = evecs[:, i]; psi_flip = psi[::-1]
        if np.linalg.norm(psi - psi_flip) < np.linalg.norm(psi + psi_flip):
            psi_norm = psi / np.linalg.norm(psi)
            even_psi.append(psi_norm); even_evals.append(evals[i])
            even_idx += 1
            if len(even_psi) >= 3: break
    return z, even_psi, even_evals, n_grid


def apply_E_map_full(z_grid, psi, lam, L_f, N, n_u=2000):
    """E map with FULL complex V_n projection (not just cosine)."""
    dim = 2 * N + 1
    y_grid = np.linspace(-L_f / 2, L_f / 2, n_u)
    u_grid = np.exp(y_grid)
    dy = y_grid[1] - y_grid[0]

    mid = len(z_grid) // 2
    z_pos = z_grid[mid:]
    psi_pos = psi[mid:]
    z0 = z_pos[0]; dz_g = z_pos[1] - z_pos[0]; n_pts = len(z_pos)

    k_vals = np.zeros(n_u)
    for i in range(n_u):
        u = u_grid[i]
        n_max = int(lam / u)
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

    # FULL complex projection (not just cosine)
    j_vals = np.arange(-N, N + 1)
    coeffs_cos = np.zeros(dim)
    coeffs_sin = np.zeros(dim)
    for j in range(-N, N + 1):
        cos_int = np.sum(k_vals * np.cos(2 * np.pi * j * y_grid / L_f)) * dy
        sin_int = np.sum(k_vals * np.sin(2 * np.pi * j * y_grid / L_f)) * dy
        coeffs_cos[j + N] = cos_int
        coeffs_sin[j + N] = sin_int

    # Full complex coefficients: c_j = cos_int - i*sin_int (conjugate Fourier)
    # For overlap with real even xi: only cos part matters
    # But let's check how big the sin part is
    cos_norm = np.linalg.norm(coeffs_cos)
    sin_norm = np.linalg.norm(coeffs_sin)

    # Normalize cosine coefficients (what we've been using)
    if cos_norm > 0:
        coeffs_norm = coeffs_cos / cos_norm
    else:
        coeffs_norm = coeffs_cos

    return coeffs_norm, coeffs_cos, coeffs_sin, cos_norm, sin_norm, k_vals, y_grid, u_grid


if __name__ == "__main__":
    print("PROLATE OVERLAP DIAGNOSTICS")
    print("=" * 80)

    lam_sq = 50
    lam = np.sqrt(lam_sq)
    L_f = np.log(lam_sq)
    N = 30; dim = 2 * N + 1

    print(f"\nlam^2 = {lam_sq}, lam = {lam:.2f}, L = {L_f:.4f}")
    print(f"gamma = 2*pi*{lam_sq} = {2*np.pi*lam_sq:.1f}")

    # Build xi
    print("\nBuilding xi...", end="", flush=True)
    xi, eps, _, _ = build_xi(lam_sq, N)
    xi_n = xi / np.linalg.norm(xi)
    print(f" eps = {eps:.4e}")

    # Check xi symmetry
    sym_err = max(abs(xi_n[N+k] - xi_n[N-k]) for k in range(1, N+1))
    print(f"xi symmetry (max |xi_n - xi_{{-n}}|): {sym_err:.2e}")
    print(f"xi[0] (n=0 component): {xi_n[N]:.6f}")
    print(f"sum(xi): {sum(xi_n):.6e}")

    # DIAGNOSTIC A: SL eigenfunctions
    print("\n--- DIAGNOSTIC A: SL eigenfunctions ---")
    z_grid, efuncs, eevals, n_grid = solve_SL(lam)
    print(f"SL n_grid = {n_grid}, gamma = {2*np.pi*lam_sq:.1f}")
    for i, (psi, ev) in enumerate(zip(efuncs, eevals)):
        psi_max = np.max(np.abs(psi))
        psi_at_0 = psi[len(psi)//2]
        print(f"  h_{2*i}: eval={ev:.4e}, max|psi|={psi_max:.4f}, "
              f"psi(0)={psi_at_0:.4f}, positive={np.all(psi > -0.01)}")

    # DIAGNOSTIC B: E map — check cos vs sin content
    print("\n--- DIAGNOSTIC B: E map symmetry (cos vs sin) ---")
    for i, psi in enumerate(efuncs):
        cn, cc, cs, c_nrm, s_nrm, kv, yg, ug = apply_E_map_full(
            z_grid, psi, lam, L_f, N, n_u=2000)
        print(f"  E(h_{2*i}): ||cos||={c_nrm:.4f}, ||sin||={s_nrm:.4f}, "
              f"ratio sin/cos={s_nrm/c_nrm:.4f}")
        # Check if E(h) coefficients are even: c_n = c_{-n}?
        even_err = max(abs(cc[N+k] - cc[N-k]) for k in range(1, N+1))
        odd_err = max(abs(cc[N+k] + cc[N-k]) for k in range(1, N+1))
        print(f"         cos coeffs even err: {even_err:.4e}, odd err: {odd_err:.4e}")

    # DIAGNOSTIC C: Vary n_u grid resolution
    print("\n--- DIAGNOSTIC C: Overlap vs n_u resolution ---")
    psi_h0 = efuncs[0]
    for n_u in [500, 1000, 2000, 5000, 10000]:
        cn, _, _, _, _, _, _, _ = apply_E_map_full(z_grid, psi_h0, lam, L_f, N, n_u=n_u)
        ov = abs(np.dot(xi_n, cn))
        print(f"  n_u={n_u:>6}: overlap(xi, E(h_0)) = {ov:.6f}")

    # DIAGNOSTIC D: Random baseline
    print("\n--- DIAGNOSTIC D: Random baseline overlaps ---")
    for trial in range(5):
        v = np.random.randn(dim)
        v /= np.linalg.norm(v)
        ov = abs(np.dot(xi_n, v))
        print(f"  random vector #{trial}: overlap = {ov:.6f}")
    print(f"  Expected for random: ~1/sqrt({dim}) = {1/np.sqrt(dim):.4f}")

    # DIAGNOSTIC E: Compare k_vals structure to xi structure
    print("\n--- DIAGNOSTIC E: E(h_0) vs xi in V_n basis ---")
    cn, cc, _, _, _, _, _, _ = apply_E_map_full(z_grid, psi_h0, lam, L_f, N, n_u=5000)
    print(f"  {'n':>4} {'xi_n':>12} {'E(h0)_n':>12} {'ratio':>12}")
    for n in range(-5, 6):
        x = xi_n[n + N]
        e = cn[n + N]
        r = x / e if abs(e) > 1e-10 else float('nan')
        print(f"  {n:>4} {x:>12.6f} {e:>12.6f} {r:>12.4f}")

    # DIAGNOSTIC F: What does k(u) look like in the u-domain?
    print("\n--- DIAGNOSTIC F: k(u) = E(h_0)(u) structure ---")
    _, _, _, _, _, kv, yg, ug = apply_E_map_full(z_grid, psi_h0, lam, L_f, N, n_u=2000)
    print(f"  k(u) range: [{np.min(kv):.4f}, {np.max(kv):.4f}]")
    print(f"  k(u) at u=1 (y=0): {kv[len(kv)//2]:.6f}")
    print(f"  k(u) at u=lam^{{-1}} (y=-L/2): {kv[0]:.6f}")
    print(f"  k(u) at u=lam (y=+L/2): {kv[-1]:.6f}")
    n_nonzero = np.sum(np.abs(kv) > 1e-10)
    print(f"  Nonzero points: {n_nonzero}/{len(kv)}")
