"""
Session 20: Connes spectral operator — exact W_R via series expansion.

Key insight: 1/(2*sinh(t)) = sum_{k=0}^inf e^{-(2k+1)t}, so the regularized
integral becomes a fast-converging series:

I(alpha) = integral_0^A (e^{t*alpha} - 1)/(2*sinh(t)) dt
         = sum_{k=0}^K [(e^{A*(alpha-2k-1)} - 1)/(alpha-2k-1)
                        + (1 - e^{-A*(2k+1)})/(2k+1)]

where A = L/2 = log(lambda), alpha = 1/2 + 2*pi*i*d/L, d = n-m.

Converges exponentially for k >= 1 (factor e^{-2kA}).
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, sinh, exp, cos, sin, hyp2f1, digamma

mp.dps = 60  # 60 digits for initial test


def von_mangoldt_sieve(k_max):
    """Precompute von Mangoldt function Lambda(k) for k=2..k_max."""
    # Sieve of Eratosthenes to find primes
    is_prime = [False, False] + [True] * (k_max - 1)
    for i in range(2, int(k_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, k_max + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, k_max + 1) if is_prime[i]]

    # Lambda(k) = log(p) if k = p^m, else 0
    vM = {}
    for p in primes:
        pk = p
        lp = float(np.log(p))
        while pk <= k_max:
            vM[pk] = lp
            pk *= p
    return vM


# ─── W_{0,2}: exact closed form (eq 4.2) ───

def W02_element(n, m, L):
    """W_{0,2}(V_n, V_m) from eq 4.2."""
    L2 = L * L
    p2 = mpf(16) * pi * pi
    prefactor = 32 * L * sinh(L / 4)**2
    numer = L2 - p2 * m * n
    denom = (L2 + p2 * m * m) * (L2 + p2 * n * n)
    return prefactor * numer / denom


# ─── W_p: von Mangoldt sum (eq 4.3) ───

def Wp_element(n, m, L, vM_dict):
    """sum_p W_p(V_n, V_m) = sum_{k} Lambda(k) * k^{-1/2} * cos(2*pi*(n-m)*log(k)/L)"""
    d = n - m
    k_max = int(mpmath.exp(L)) + 1
    total = mpf(0)
    for k, lk in vM_dict.items():
        if k > k_max:
            continue
        total += mpf(lk) * mpmath.power(k, mpf(-0.5)) * mpmath.cos(2 * pi * d * mpmath.log(k) / L)
    return total


# ─── W_R: exact series expansion ───

def WR_element(n, m, L, K_terms=50):
    """
    W_R(V_n, V_m) using series expansion.

    W_R#(F) = c0 + integral_0^A (e^{t*alpha} - 1)/(2*sinh(t)) dt

    where c0 = (1/2)*(log(4*pi) + gamma), A = L/2, alpha = 1/2 + 2*pi*i*d/L.

    Series: I(alpha) = sum_{k=0}^K [(e^{A*(alpha-2k-1)} - 1)/(alpha-2k-1)
                                    + (1 - e^{-A*(2k+1)})/(2k+1)]
    """
    d = n - m
    A = L / 2
    alpha = mpf(0.5) + 2 * pi * mpc(0, 1) * d / L
    c0 = (mpmath.log(4 * pi) + euler) / 2

    total = mpc(0, 0)
    for k in range(K_terms):
        beta = alpha - (2 * k + 1)  # alpha - (2k+1)
        # Term 1: (e^{A*beta} - 1) / beta
        if abs(beta) < mpf(10)**(-40):
            t1 = A  # limit as beta -> 0
        else:
            t1 = (exp(A * beta) - 1) / beta
        # Term 2: (e^{-A*(2k+1)} - 1) / (2k+1)  [integral of -e^{-(2k+1)t}]
        t2 = (exp(-A * (2 * k + 1)) - 1) / (2 * k + 1)
        total += t1 + t2

        # Check convergence (k >= 2 terms decay exponentially)
        if k >= 2 and abs(t1) < mpf(10)**(-mp.dps + 5):
            break

    return float((c0 + total).real)


# ─── Build full QW matrix ───

def build_QW(N, L, vM_dict):
    """Build complete QW = W_{0,2} - W_R - sum W_p."""
    dim = 2 * N + 1
    QW = np.zeros((dim, dim))

    L_mp = mpf(L)
    total_elements = dim * (dim + 1) // 2
    computed = 0

    for idx_n in range(dim):
        n = idx_n - N
        for idx_m in range(idx_n, dim):
            m = idx_m - N
            w02 = float(W02_element(n, m, L_mp))
            wp = float(Wp_element(n, m, L_mp, vM_dict))
            wr = WR_element(n, m, L_mp)

            val = w02 - wr - wp
            QW[idx_n, idx_m] = val
            QW[idx_m, idx_n] = val

            computed += 1
            if computed % 500 == 0:
                print(f"    {computed}/{total_elements} elements...", flush=True)

    return QW


# ─── xi_hat and zero finding ───

def xi_hat_val(z, xi_vec, N, L):
    """xi_hat(z) = 2/sqrt(L) * sin(z*L/2) * sum xi_j/(z - 2*pi*j/L)"""
    sin_part = np.sin(z * L / 2)
    if abs(sin_part) < 1e-30:
        return 0.0
    total = 0.0
    for idx in range(2 * N + 1):
        j = idx - N
        d_j = 2 * np.pi * j / L
        denom = z - d_j
        if abs(denom) < 1e-12:
            return float('inf')  # pole
        total += xi_vec[idx] / denom
    return 2 * L**(-0.5) * sin_part * total


def find_zeros(xi_vec, N, L, z_min=1.0, z_max=100.0, n_scan=200000):
    """Find zeros of xi_hat by bisection."""
    z_range = np.linspace(z_min, z_max, n_scan)
    roots = []
    scaling_eigs = set()
    for j in range(-N, N+1):
        scaling_eigs.add(round(2 * np.pi * j / L, 8))

    prev = xi_hat_val(z_range[0], xi_vec, N, L)
    for i in range(1, len(z_range)):
        val = xi_hat_val(z_range[i], xi_vec, N, L)
        if not (np.isfinite(val) and np.isfinite(prev)):
            prev = val
            continue
        if prev * val < 0 and abs(val) < 1e10 and abs(prev) < 1e10:
            lo, hi = z_range[i-1], z_range[i]
            for _ in range(100):
                mid = (lo + hi) / 2
                fmid = xi_hat_val(mid, xi_vec, N, L)
                if not np.isfinite(fmid):
                    break
                if fmid * xi_hat_val(lo, xi_vec, N, L) < 0:
                    hi = mid
                else:
                    lo = mid
            root = (lo + hi) / 2
            # Filter out scaling eigenvalues (poles)
            is_pole = any(abs(root - 2*np.pi*j/L) < 0.05 for j in range(-N, N+1))
            if not is_pole:
                roots.append(root)
        prev = val
    return roots


# ─── Main ───

if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 20: Connes Spectral Operator — Exact W_R")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # Parameters from paper
    lam_sq = 14
    L = float(np.log(lam_sq))
    N = 80  # 2N+1 = 161

    print(f"\n  lambda^2 = {lam_sq}, L = {L:.6f}")
    print(f"  N = {N}, dim = {2*N+1}")
    print(f"  Scaling spacing: 2*pi/L = {2*np.pi/L:.6f}")
    print(f"  Precision: {mp.dps} digits")

    # Precompute von Mangoldt
    k_max = int(np.exp(L)) + 1
    print(f"\n  Precomputing von Mangoldt up to {k_max}...")
    vM = von_mangoldt_sieve(k_max)
    print(f"  {len(vM)} prime powers found")

    # --- Step 1: Verify W_R series ---
    print("\n--- Step 1: Verify W_R series convergence ---")
    L_mp = mpf(L)
    for d in [0, 1, 5, 10]:
        wr = WR_element(d, 0, L_mp, K_terms=50)
        print(f"  W_R(d={d:2d}): {wr:+.15f}")

    # --- Step 2: Build QW ---
    print(f"\n--- Step 2: Building QW ({2*N+1}x{2*N+1}) ---")
    QW = build_QW(N, L, vM)
    QW = (QW + QW.T) / 2  # ensure exact symmetry

    eigvals, eigvecs = np.linalg.eigh(QW)
    print(f"  Min eigenvalue: {eigvals[0]:+.10f}")
    print(f"  Max eigenvalue: {eigvals[-1]:+.10f}")
    print(f"  Positive: {np.sum(eigvals > 0)}, Negative: {np.sum(eigvals < 0)}")

    # Min eigenvector
    xi = eigvecs[:, 0]
    print(f"  Min eigenvector: xi[0]={xi[N]:+.8f}")

    # --- Step 3: Find zeros of xi_hat ---
    print(f"\n--- Step 3: xi_hat zeros ---")
    roots = find_zeros(xi, N, L, z_min=1.0, z_max=80.0, n_scan=200000)
    print(f"  Found {len(roots)} zeros")

    print(f"\n  {'#':>3s}  {'xi_hat zero':>14s}  {'zeta zero':>14s}  {'diff':>12s}  {'rel%':>8s}")
    for i in range(min(25, len(roots), len(gammas))):
        r = roots[i]
        g = gammas[i]
        diff = r - g
        rel = abs(diff / g) * 100
        print(f"  {i+1:3d}  {r:14.6f}  {g:14.6f}  {diff:+12.6f}  {rel:7.3f}%")

    # --- Step 4: Diagonal check ---
    print(f"\n--- Step 4: QW diagonal structure ---")
    print(f"  {'n':>4s}  {'W02':>12s}  {'Wp':>12s}  {'WR':>12s}  {'QW':>12s}")
    for n_val in [0, 1, 5, 10, 20, 40]:
        w02 = float(W02_element(n_val, n_val, L_mp))
        wp = float(Wp_element(n_val, n_val, L_mp, vM))
        wr = WR_element(n_val, n_val, L_mp)
        qw = w02 - wr - wp
        print(f"  {n_val:4d}  {w02:+12.6f}  {wp:+12.6f}  {wr:+12.6f}  {qw:+12.6f}")

    print("\n" + "=" * 70)
