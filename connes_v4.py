"""
Session 20c: Connes — DIRECT from equation (4.4).

W_R(V_n, V_m) = (omega(0)/2)*[gamma + log(4*pi*(e^L-1)/(e^L+1))]
                + integral_0^L [e^{x/2}*omega(x) - omega(0)] / (e^x - e^{-x}) dx

where omega(x) = q(U_n, U_m)(x):
  n != m: omega(x) = [sin(2*pi*m*x/L) - sin(2*pi*n*x/L)] / [pi*(n-m)], omega(0) = 0
  n == m: omega(x) = 2*(1-x/L)*cos(2*pi*n*x/L), omega(0) = 2

The integrand [e^{x/2}*omega(x) - omega(0)] / (e^x-e^{-x}) is BOUNDED at x=0.
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
            pk = i; lp = float(np.log(i))
            while pk <= k_max: vM[pk] = lp; pk *= i
    return vM


def WR_element_eq44(n, m, L, n_quad=8000):
    """
    W_R(V_n, V_m) directly from equation (4.4).
    Numerically integrates the regularized kernel.
    """
    L_mp = mpf(L)

    # omega(x) and omega(0)
    if n == m:
        omega_0 = mpf(2)
        def omega(x):
            return 2 * (1 - x / L_mp) * cos(2 * pi * n * x / L_mp)
    else:
        omega_0 = mpf(0)
        denom = pi * (n - m)
        def omega(x):
            return (sin(2 * pi * m * x / L_mp) - sin(2 * pi * n * x / L_mp)) / denom

    # Constant term
    eL = exp(L_mp)
    w_const = (omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1)))

    # Integral: [e^{x/2}*omega(x) - omega(0)] / (e^x - e^{-x}) from 0 to L
    dx = L_mp / n_quad
    integral = mpf(0)
    for k in range(n_quad):
        x = dx * (k + mpf(0.5))  # midpoint rule
        numerator = exp(x / 2) * omega(x) - omega_0
        denominator = exp(x) - exp(-x)  # = 2*sinh(x)
        if abs(denominator) > mpf(10)**(-40):
            integral += numerator / denominator
    integral *= dx

    return float(w_const + integral)


def build_QW_eq44(N, L, vM, n_quad=5000):
    """Build full QW using eq (4.4) for W_R."""
    dim = 2 * N + 1
    L2 = L**2; p2 = (4*np.pi)**2
    pf02 = 32 * L * np.sinh(L/4)**2
    k_max = int(np.exp(L)) + 1

    QW = np.zeros((dim, dim))
    total = dim * (dim + 1) // 2
    count = 0

    for idx_n in range(dim):
        n = idx_n - N
        for idx_m in range(idx_n, dim):
            m = idx_m - N

            # W_{0,2} (eq 4.2)
            w02 = pf02 * (L2 - p2*m*n) / ((L2 + p2*m*m) * (L2 + p2*n*n))

            # W_p (eq 4.3)
            d = n - m
            wp = sum(lk * k**(-0.5) * np.cos(2*np.pi*d*np.log(k)/L) for k, lk in vM.items() if k <= k_max)

            # W_R (eq 4.4)
            wr = WR_element_eq44(n, m, L, n_quad=n_quad)

            val = w02 - wr - wp
            QW[idx_n, idx_m] = val
            QW[idx_m, idx_n] = val

            count += 1
            if count % 200 == 0:
                print(f"    {count}/{total}...", flush=True)

    return QW


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


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 20c: Connes — Direct Equation (4.4)")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")
    lam_sq = 14; L = float(np.log(lam_sq)); N = 40; dim = 2*N+1

    print(f"\n  lambda^2={lam_sq}, L={L:.6f}, N={N}, dim={dim}")

    vM = von_mangoldt_sieve(int(np.exp(L)) + 1)
    print(f"  {len(vM)} prime powers")

    # Verify W_R against backed-out values
    print("\n--- W_R verification (eq 4.4 vs backed-out from zeros) ---")
    for n_test, m_test in [(0,0), (1,1), (5,5), (10,10), (0,1), (1,2)]:
        wr = WR_element_eq44(n_test, m_test, L, n_quad=5000)
        print(f"  WR[{n_test},{m_test}] = {wr:+.8f}")

    # Build QW
    print(f"\n--- Building QW ({dim}x{dim}) ---")
    QW = build_QW_eq44(N, L, vM, n_quad=3000)
    QW = (QW + QW.T) / 2

    eigvals, eigvecs = np.linalg.eigh(QW)
    print(f"\n  Eigenvalues: min={eigvals[0]:+.8f}, max={eigvals[-1]:+.8f}")
    print(f"  Positive: {np.sum(eigvals > 0)}, Negative: {np.sum(eigvals < 0)}")

    # Find even eigenvectors
    even_eigs = []
    for i in range(dim):
        xi = eigvecs[:, i]
        even_score = sum(abs(xi[N+n] - xi[N-n]) for n in range(1, N+1))
        odd_score = sum(abs(xi[N+n] + xi[N-n]) for n in range(1, N+1))
        if even_score < odd_score:
            even_eigs.append((i, eigvals[i]))

    print(f"\n  Even eigenvectors: {len(even_eigs)}")
    print(f"  Smallest even eigenvalue: {even_eigs[0][1]:+.8f}")

    # Normalize: sum(xi) = sqrt(L)
    best_idx = even_eigs[0][0]
    xi = eigvecs[:, best_idx]
    xi_sum = np.sum(xi)
    if abs(xi_sum) > 1e-8:
        xi = xi * np.sqrt(L) / xi_sum
    print(f"  xi sum = {np.sum(xi):.6f} (target {np.sqrt(L):.6f})")

    # Find zeros
    print("\n--- Zeros of xi_hat ---")
    roots = find_zeros(xi, N, L, z_min=1.0, z_max=60.0, n_scan=200000)
    print(f"  Found {len(roots)} zeros\n")

    print(f"  {'#':>3s}  {'xi_hat zero':>14s}  {'zeta zero':>14s}  {'diff':>12s}  {'rel%':>8s}")
    for i in range(min(15, len(roots), len(gammas))):
        r = roots[i]; g = gammas[i]
        print(f"  {i+1:3d}  {r:14.6f}  {g:14.6f}  {r-g:+12.6f}  {abs(r-g)/g*100:7.3f}%")

    print("\n" + "=" * 70)
