"""
Session 25: Scale prolate overlap to lam^2 = 10,000+

Key fixes and improvements over Session 24 (connes_prolate_correct.py):
1. BUGFIX: h_4 = even[2] (3rd even eigenfunction), NOT even[1] (which is h_2)
   Session 24 used projs[1] calling it "h_4" but it was actually h_2.
2. SL grid scales with gamma: n_grid >= 4*gamma/pi (Nyquist criterion)
3. Paper's exact coefficients: (sqrt(3)/2^{11/4})*h_4 - (3/2^{17/4})*h_0
4. Fast regular-grid interpolation in E map (O(1) per point vs O(log n))
5. Selective eigensolve (first 20 only) via select='i' for large grids
6. Vectorized tridiagonal construction and V_n projection

Prolate indexing convention:
  SL eigenvalues in ascending order: chi_0, chi_1, chi_2, chi_3, chi_4, ...
  Even eigenfunctions (psi(-z)=psi(z)): h_0(chi_0), h_2(chi_2), h_4(chi_4), ...
  So even[0]=h_0, even[1]=h_2, even[2]=h_4
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
    """Build the Weil eigenvector xi_lambda in the V_n basis.

    This is the target vector: the prolate overlap should converge to it.
    Uses mpmath at dps=50 for the tau matrix construction and eigensolve.
    """
    L = log(mpf(lam_sq)); eL = exp(L)

    # Collect prime powers up to lam^2
    vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq):
            vM.append((pk, lp, log(pk))); pk *= p

    dim = 2 * N + 1

    # Alpha coefficients (off-diagonal W_r)
    al = {}
    for n in range(-N, N + 1):
        nn = abs(n)
        if nn == 0:
            al[n] = mpf(0); continue
        z = exp(-2 * L); a = pi * mpc(0, nn) / L + mpf(1) / 4
        h = hyp2f1(1, a, a + 1, z)
        al[n] = (exp(-L / 2) * (2 * L / (L + 4 * pi * mpc(0, nn)) * h).imag
                 + digamma(a).imag / 2) / pi
        if n < 0:
            al[n] = -al[n]

    # Diagonal W_r
    wr_d = {}
    for nv in range(N + 1):
        w_c = euler + log(4 * pi * (eL - 1) / (eL + 1))
        def ig(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        wr_d[nv] = w_c + quad(ig, [mpf(0), L])
        wr_d[-nv] = wr_d[nv]

    # Build tau = W_0,2 - W_r - W_p
    tau = mpmatrix(dim, dim)
    L2 = L * L; p2 = 16 * pi * pi; pf = 32 * L * sinh(L / 4) ** 2

    def q_mp(n, m, y):
        if n != m:
            return (sin(2 * pi * m * y / L) - sin(2 * pi * n * y / L)) / (pi * (n - m))
        else:
            return 2 * (L - y) / L * cos(2 * pi * n * y / L)

    for i in range(dim):
        n = i - N
        for j in range(i, dim):
            m = j - N
            w02 = pf * (L2 - p2 * m * n) / ((L2 + p2 * m ** 2) * (L2 + p2 * n ** 2))
            wp = sum(lk * pk ** (-mpf(1) / 2) * q_mp(n, m, logk) for pk, lk, logk in vM)
            wr = wr_d[n] if n == m else (al[m] - al[n]) / (n - m)
            tau[i, j] = w02 - wr - wp
            tau[j, i] = tau[i, j]

    E, ER = eig(tau, left=False, right=True)
    evals = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    eps = evals[0][0]; idx = evals[0][1]
    xi = [float(ER[j, idx].real) for j in range(dim)]
    xs = sum(xi); sqL = float(mpmath.sqrt(L))
    if abs(xs) > 1e-20:
        xi = [x * sqL / xs for x in xi]
    return np.array(xi), float(eps), float(L), N


def solve_SL_scaled(lam, max_grid=100000):
    """Sturm-Liouville with grid scaled to gamma = 2*pi*lam^2.

    F_gamma y = d/dz[(1-z^2)dy/dz] + gamma^2(1-z^2)y = chi*y on [-1,1]

    Vectorized tridiagonal construction. Selective eigensolve for first 20.
    Returns first 3 even eigenfunctions: h_0, h_2, h_4.
    """
    gamma = 2 * np.pi * lam**2
    n_grid = min(max_grid, max(500, int(np.ceil(4 * gamma / np.pi)) + 100))

    z = np.linspace(-1, 1, n_grid + 2)[1:-1]  # interior points
    dz = z[1] - z[0]

    # p(z) = 1-z^2 at half-points (vectorized)
    z_half = np.empty(n_grid + 1)
    z_half[0] = (-1 + z[0]) / 2
    z_half[1:-1] = (z[:-1] + z[1:]) / 2
    z_half[-1] = (z[-1] + 1) / 2
    p_half = 1 - z_half**2

    # q(z) = gamma^2 * (1-z^2)
    q = gamma**2 * (1 - z**2)

    # Tridiagonal: d[i] = -(p_{i-1/2}+p_{i+1/2})/dz^2 + q[i]
    d = -(p_half[:-1] + p_half[1:]) / dz**2 + q
    od = p_half[1:-1] / dz**2

    # Get first 20 eigenvalues/vectors (smallest)
    n_eig = min(20, n_grid)
    evals, evecs = eigh_tridiagonal(d, od, select='i', select_range=(0, n_eig - 1))

    # Filter for even eigenfunctions: need h_0, h_2, h_4 (3 even)
    even_psi = []
    even_evals = []
    even_labels = []
    even_idx = 0
    for i in range(n_eig):
        psi = evecs[:, i]
        psi_flip = psi[::-1]
        if np.linalg.norm(psi - psi_flip) < np.linalg.norm(psi + psi_flip):
            psi_norm = psi / np.linalg.norm(psi)
            even_psi.append(psi_norm)
            even_evals.append(evals[i])
            even_labels.append(f"h_{2*even_idx}")
            even_idx += 1
            if len(even_psi) >= 3:  # h_0, h_2, h_4
                break

    return z, even_psi, even_evals, even_labels, n_grid


def apply_E_map(z_grid, psi, lam, L_f, N, n_u=2000):
    """Apply E(h)(u) = u^{1/2} * sum_{n>=1} h(n*u), project to V_n basis.

    Uses fast regular-grid interpolation (O(1) per point).
    h is even, so only evaluates on z >= 0.
    """
    dim = 2 * N + 1

    # Log-uniform grid for u in [lam^{-1}, lam]
    y_grid = np.linspace(-L_f / 2, L_f / 2, n_u)
    u_grid = np.exp(y_grid)
    dy = y_grid[1] - y_grid[0]

    # Even function: use z >= 0 half only
    mid = len(z_grid) // 2
    z_pos = z_grid[mid:]
    psi_pos = psi[mid:]
    z0 = z_pos[0]
    dz = z_pos[1] - z_pos[0]
    n_pts = len(z_pos)

    # E map with fast interpolation
    k_vals = np.zeros(n_u)
    for i in range(n_u):
        u = u_grid[i]
        n_max = int(lam / u)
        if n_max < 1:
            continue
        ns = np.arange(1, n_max + 1, dtype=np.float64)
        zs = ns * u / lam  # in [0, 1]

        # Fast regular-grid interpolation (no binary search)
        idx_f = (zs - z0) / dz
        idx_lo = np.floor(idx_f).astype(np.intp)
        # Clip to valid range
        valid = (idx_lo >= 0) & (idx_lo < n_pts - 1)
        if not np.any(valid):
            continue
        idx_lo_v = idx_lo[valid]
        frac = idx_f[valid] - idx_lo_v
        vals = psi_pos[idx_lo_v] * (1 - frac) + psi_pos[idx_lo_v + 1] * frac
        k_vals[i] = np.sqrt(u) * np.sum(vals)

    # Vectorized V_n projection: c_j = integral k(e^y) cos(2*pi*j*y/L) dy
    j_vals = np.arange(-N, N + 1)
    cos_mat = np.cos(2 * np.pi * np.outer(j_vals, y_grid) / L_f)  # (dim, n_u)
    coeffs = cos_mat @ k_vals * dy

    nrm = np.linalg.norm(coeffs)
    if nrm > 0:
        coeffs /= nrm
    return coeffs


# ======================================================================
# Main sweep
# ======================================================================
if __name__ == "__main__":
    print("PROLATE OVERLAP SCALING (Session 25)")
    print("E(h)(u) = u^{1/2} * sum_{n>=1} h(n*u)")
    print("gamma = 2*pi*lam^2  |  SL grid scales with gamma")
    print("BUGFIX: h_4 = even[2] (was using even[1]=h_2 in Session 24)")
    print("=" * 80)
    print()

    # Paper's exact coefficients for k_lambda
    # k_lambda = (sqrt(3)/2^{11/4}) * h_4 - (3/2^{17/4}) * h_0
    a_h4 = np.sqrt(3) / 2**(11/4)   # coeff of h_4 ~ 0.2576
    b_h0 = -3 / 2**(17/4)            # coeff of h_0 ~ -0.1577

    results = []

    for lam_sq in [14, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        t0 = time.time()
        lam = np.sqrt(lam_sq)
        L_f = np.log(lam_sq)
        N = 30
        dim = 2 * N + 1
        gamma = 2 * np.pi * lam_sq

        # 1. Build Weil eigenvector (mpmath, slow for large lam_sq)
        print(f"lam^2={lam_sq:6d}  building xi...", end="", flush=True)
        xi, eps, _, _ = build_xi(lam_sq, N)
        xi_n = xi / np.linalg.norm(xi)
        t_xi = time.time() - t0
        print(f" ({t_xi:.0f}s)  SL...", end="", flush=True)

        # 2. Solve SL with scaled grid
        z_grid, efuncs, eevals, elabels, n_grid = solve_SL_scaled(lam)
        t_sl = time.time() - t0 - t_xi
        print(f" ({t_sl:.1f}s, n={n_grid})  E map...", end="", flush=True)

        # 3. Apply E map to each even prolate eigenfunction
        projs = []
        ovs = []
        for psi in efuncs:
            coeffs = apply_E_map(z_grid, psi, lam, L_f, N)
            ov = abs(np.dot(xi_n, coeffs))
            ovs.append(ov)
            projs.append(coeffs)
        t_E = time.time() - t0 - t_xi - t_sl
        print(f" ({t_E:.0f}s)", flush=True)

        # 4a. k_lambda via integral-vanishing condition using h_0 + h_4
        #     (FIXED: use projs[2]=h_4, not projs[1]=h_2)
        ov_vanish_h4 = -1.0
        if len(projs) >= 3:
            h0c, h4c = projs[0], projs[2]  # h_0 and h_4
            if abs(h4c[N]) > 1e-15:
                r = -h0c[N] / h4c[N]
                kl = h0c + r * h4c
                nrm = np.linalg.norm(kl)
                if nrm > 0:
                    kl /= nrm
                    ov_vanish_h4 = abs(np.dot(xi_n, kl))

        # 4b. k_lambda via integral-vanishing with h_0 + h_2 (Session 24 version for comparison)
        ov_vanish_h2 = -1.0
        if len(projs) >= 2:
            h0c, h2c = projs[0], projs[1]  # h_0 and h_2
            if abs(h2c[N]) > 1e-15:
                r = -h0c[N] / h2c[N]
                kl = h0c + r * h2c
                nrm = np.linalg.norm(kl)
                if nrm > 0:
                    kl /= nrm
                    ov_vanish_h2 = abs(np.dot(xi_n, kl))

        # 5. k_lambda via paper's exact coefficients: b_h0*h_0 + a_h4*h_4
        ov_paper = -1.0
        if len(projs) >= 3:
            h0c, h4c = projs[0], projs[2]
            kl_p = b_h0 * h0c + a_h4 * h4c
            nrm = np.linalg.norm(kl_p)
            if nrm > 0:
                kl_p /= nrm
                ov_paper = abs(np.dot(xi_n, kl_p))

        dt = time.time() - t0

        results.append({
            'lam_sq': lam_sq, 'gamma': gamma, 'n_grid': n_grid,
            'eps': eps,
            'ov_h0': ovs[0] if len(ovs) > 0 else -1,
            'ov_h2': ovs[1] if len(ovs) > 1 else -1,
            'ov_h4': ovs[2] if len(ovs) > 2 else -1,
            'ov_vanish_h2': ov_vanish_h2,
            'ov_vanish_h4': ov_vanish_h4,
            'ov_paper': ov_paper,
            'dt': dt
        })

        # Print results for this lambda
        print(f"  eps(xi) = {eps:.4e}  gamma={gamma:.0f}  n_grid={n_grid}")
        for i in range(min(3, len(ovs))):
            tag = elabels[i] if i < len(elabels) else f"h_{2*i}"
            print(f"  overlap(xi, E({tag}))   = {ovs[i]:.6f}")
        if ov_vanish_h2 >= 0:
            print(f"  k_vanish(h0+h2)       = {ov_vanish_h2:.6f}  (Session 24 method)")
        if ov_vanish_h4 >= 0:
            print(f"  k_vanish(h0+h4)       = {ov_vanish_h4:.6f}  (CORRECTED)")
        if ov_paper >= 0:
            print(f"  k_paper(h0+h4)        = {ov_paper:.6f}  (exact coefficients)")
        print(f"  total: {dt:.0f}s")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY: overlap trend vs lam^2")
    print("-" * 80)
    print(f"{'lam^2':>8} {'gamma':>10} {'n_grid':>7} | {'E(h0)':>8} {'E(h2)':>8} {'E(h4)':>8} | {'k_h0h2':>8} {'k_h0h4':>8} {'k_paper':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['lam_sq']:>8} {r['gamma']:>10.0f} {r['n_grid']:>7} | "
              f"{r['ov_h0']:>8.5f} {r['ov_h2']:>8.5f} {r['ov_h4']:>8.5f} | "
              f"{r['ov_vanish_h2']:>8.5f} {r['ov_vanish_h4']:>8.5f} {r['ov_paper']:>8.5f}")

    print()
    print("k_h0h2 = Session 24 method (integral-vanishing with h_0+h_2, mislabeled h_4)")
    print("k_h0h4 = Corrected (integral-vanishing with actual h_0+h_4)")
    print("k_paper = Paper's exact coefficients: (sqrt(3)/2^{11/4})*h_4 - (3/2^{17/4})*h_0")
    print()
    print("If any overlap column -> 1 as lam^2 -> inf: educated guess VERIFIED.")
