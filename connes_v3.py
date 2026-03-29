"""
Session 20b: Connes spectral operator — EXACT formulas from arxiv:2511.22755.

Key formula from Proposition 4.3:
  W_R(V_n, V_m) = (alpha_L(m) - alpha_L(n)) / (n - m)    for n != m
  W_R(V_n, V_n) = 2*gamma_L(n) - 2*beta_L(n)              for n = n

where:
  alpha_L(n) = (1/pi) * integral_0^L sin(2*pi*n*x/L) * rho(x) dx
  beta_L(n)  = (1/L)  * integral_0^L x*cos(2*pi*n*x/L) * rho(x) dx
  gamma_L(n) = integral_0^L (cos(2*pi*n*x/L) - e^{-x/2}) * rho(x) dx + c(L) + w(L)
  rho(x) = e^{x/2} / (e^x - e^{-x})

  c(L) + w(L) = (1/2)*log((e^{L/2}-1)/(e^{L/2}+1)) + arctan(e^{L/2}) - pi/4 + gamma/2 + (1/2)*log(8*pi)

Normalization: xi is normalized by delta_N(xi) = 1, i.e. sum(xi_j) = sqrt(L).

Secular equation: zeros of xi_hat(z) = 2/sqrt(L) * sin(zL/2) * sum xi_j/(z - 2*pi*j/L)
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, sinh, exp, cos, sin, atan

mp.dps = 50


def von_mangoldt_sieve(k_max):
    is_prime = [False, False] + [True] * (k_max - 1)
    for i in range(2, int(k_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, k_max + 1, i): is_prime[j] = False
    vM = {}
    for i in range(2, k_max + 1):
        if is_prime[i]:
            pk = i
            lp = float(np.log(i))
            while pk <= k_max:
                vM[pk] = lp
                pk *= i
    return vM


# ─── Compute alpha_L, beta_L, gamma_L via numerical quadrature ───

def compute_rho_integrals(n_val, L, n_quad=10000):
    """
    Compute alpha_L(n), beta_L(n), gamma_L(n) by numerical integration.

    rho(x) = e^{x/2} / (e^x - e^{-x}) = e^{x/2} / (2*sinh(x))
    """
    L_mp = mpf(L)
    n_mp = mpf(n_val)

    # Quadrature points (avoid x=0 where rho ~ 1/(2x))
    # Use Gauss-Legendre or midpoint rule
    dx = L_mp / n_quad
    x_vals = [dx * (k + mpf(0.5)) for k in range(n_quad)]

    # rho(x) = e^{x/2} / (2*sinh(x))
    alpha_sum = mpf(0)
    beta_sum = mpf(0)
    cos_minus_exp_sum = mpf(0)  # integral of (cos - e^{-x/2}) * rho

    for x in x_vals:
        rho_x = exp(x / 2) / (2 * sinh(x))
        phase = 2 * pi * n_mp * x / L_mp
        sin_val = sin(phase)
        cos_val = cos(phase)

        alpha_sum += sin_val * rho_x
        beta_sum += x * cos_val * rho_x
        cos_minus_exp_sum += (cos_val - exp(-x / 2)) * rho_x

    alpha_sum *= dx / pi  # alpha_L(n) = (1/pi) * integral
    beta_sum *= dx / L_mp  # beta_L(n) = (1/L) * integral
    cos_minus_exp_sum *= dx

    # c(L) + w(L)
    eL2 = exp(L_mp / 2)
    cw = (log((eL2 - 1) / (eL2 + 1)) / 2
          + atan(eL2) - pi / 4
          + euler / 2 + log(8 * pi) / 2)

    gamma_val = cos_minus_exp_sum + cw

    return float(alpha_sum), float(beta_sum), float(gamma_val)


# ─── W_R matrix using Proposition 4.3 ───

def build_WR_prop43(N, L, n_quad=5000):
    """W_R from Proposition 4.3 with exact n-dependence."""
    dim = 2 * N + 1

    # Precompute alpha_L, beta_L, gamma_L for all n
    print("    Precomputing alpha, beta, gamma for each n...")
    alphas = {}
    betas = {}
    gammas_L = {}
    for idx in range(dim):
        n = idx - N
        a, b, g = compute_rho_integrals(n, L, n_quad=n_quad)
        alphas[n] = a
        betas[n] = b
        gammas_L[n] = g
        if idx % 20 == 0:
            print(f"      n={n}: alpha={a:.8f}, beta={b:.8f}, gamma={g:.8f}")

    W = np.zeros((dim, dim))
    for idx_n in range(dim):
        n = idx_n - N
        for idx_m in range(idx_n, dim):
            m = idx_m - N
            if n == m:
                val = 2 * gammas_L[n] - 2 * betas[n]
            else:
                val = (alphas[m] - alphas[n]) / (n - m)
            W[idx_n, idx_m] = val
            W[idx_m, idx_n] = val

    return W


# ─── W_{0,2} and W_p (same as before) ───

def build_W02(N, L):
    dim = 2 * N + 1
    L2 = L**2; p2 = (4*np.pi)**2
    prefactor = 32 * L * np.sinh(L/4)**2
    W = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N
        for j in range(i, dim):
            m = j - N
            val = prefactor * (L2 - p2*m*n) / ((L2 + p2*m*m) * (L2 + p2*n*n))
            W[i,j] = val; W[j,i] = val
    return W


def build_Wp(N, L, vM):
    dim = 2 * N + 1
    k_max = int(np.exp(L)) + 1
    W = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N
        for j in range(i, dim):
            m = j - N
            d = n - m
            total = 0.0
            for k, lk in vM.items():
                if k > k_max: continue
                total += lk * k**(-0.5) * np.cos(2*np.pi*d*np.log(k)/L)
            W[i,j] = total; W[j,i] = total
    return W


# ─── xi_hat and zero finding ───

def xi_hat_val(z, xi_vec, N, L):
    sin_part = np.sin(z * L / 2)
    if abs(sin_part) < 1e-30: return 0.0
    total = 0.0
    for idx in range(2 * N + 1):
        j = idx - N
        dj = 2 * np.pi * j / L
        denom = z - dj
        if abs(denom) < 1e-12: return float('inf')
        total += xi_vec[idx] / denom
    return 2 * L**(-0.5) * sin_part * total


def find_zeros(xi_vec, N, L, z_min=1.0, z_max=100.0, n_scan=200000):
    z_range = np.linspace(z_min, z_max, n_scan)
    roots = []
    prev = xi_hat_val(z_range[0], xi_vec, N, L)
    for i in range(1, len(z_range)):
        val = xi_hat_val(z_range[i], xi_vec, N, L)
        if np.isfinite(val) and np.isfinite(prev) and prev * val < 0 and abs(val) < 1e10:
            lo, hi = z_range[i-1], z_range[i]
            for _ in range(100):
                mid = (lo + hi) / 2
                fm = xi_hat_val(mid, xi_vec, N, L)
                if not np.isfinite(fm): break
                if fm * xi_hat_val(lo, xi_vec, N, L) < 0: hi = mid
                else: lo = mid
            root = (lo + hi) / 2
            if not any(abs(root - 2*np.pi*j/L) < 0.05 for j in range(-N, N+1)):
                roots.append(root)
        prev = val
    return roots


# ─── Main ───

if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 20b: Connes — Proposition 4.3 Exact W_R")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    lam_sq = 14
    L = float(np.log(lam_sq))
    N = 60
    dim = 2 * N + 1

    print(f"\n  lambda^2={lam_sq}, L={L:.6f}, N={N}, dim={dim}")

    # --- Build QW ---
    print("\n--- Building QW components ---")
    vM = von_mangoldt_sieve(int(np.exp(L)) + 1)

    print("  W_{0,2}...")
    W02 = build_W02(N, L)

    print("  W_p...")
    Wp = build_Wp(N, L, vM)

    print("  W_R (Proposition 4.3)...")
    WR = build_WR_prop43(N, L, n_quad=3000)

    # Assemble
    QW = W02 - WR - Wp
    QW = (QW + QW.T) / 2

    print(f"\n  QW diagonal check:")
    for n in [0, 1, 5, 10, 20]:
        idx = n + N
        print(f"    QW[{n},{n}] = {QW[idx,idx]:+.8f}  (W02={W02[idx,idx]:+.6f}, WR={WR[idx,idx]:+.6f}, Wp={Wp[idx,idx]:+.6f})")

    eigvals, eigvecs = np.linalg.eigh(QW)
    print(f"\n  Eigenvalues: min={eigvals[0]:+.8f}, max={eigvals[-1]:+.8f}")
    print(f"  Positive: {np.sum(eigvals > 0)}, Negative: {np.sum(eigvals < 0)}")
    print(f"  Near-zero (|e|<0.01): {np.sum(np.abs(eigvals) < 0.01)}")

    # Min eigenvector, normalized by delta_N(xi) = 1 => sum(xi) = sqrt(L)
    xi_raw = eigvecs[:, 0]
    xi_sum = np.sum(xi_raw)
    if abs(xi_sum) > 1e-10:
        xi = xi_raw * np.sqrt(L) / xi_sum
    else:
        xi = xi_raw  # fallback
    print(f"  xi normalization: sum(xi) = {np.sum(xi):.6f} (should be {np.sqrt(L):.6f})")

    # --- Find zeros of xi_hat ---
    print("\n--- Zeros of xi_hat(z) ---")
    roots = find_zeros(xi, N, L, z_min=1.0, z_max=80.0, n_scan=200000)
    print(f"  Found {len(roots)} zeros")

    print(f"\n  {'#':>3s}  {'xi_hat zero':>14s}  {'zeta zero':>14s}  {'diff':>12s}  {'rel%':>8s}")
    for i in range(min(25, len(roots), len(gammas))):
        r = roots[i]
        g = gammas[i]
        diff = r - g
        rel = abs(diff / g) * 100
        print(f"  {i+1:3d}  {r:14.6f}  {g:14.6f}  {diff:+12.6f}  {rel:7.3f}%")

    print("\n" + "=" * 70)
