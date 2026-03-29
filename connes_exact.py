"""
Session 19e: Connes-Consani-Moscovici spectral operator — exact formulas.

From arxiv:2511.22755 "Zeta Spectral Triples" (Nov 2025).

Key formulas:
  Basis: V_n(u) = u^{2*pi*i*n/L} on [lambda^{-1}, lambda], L = 2*log(lambda)
  Scaling operator: D_log V_n = (2*pi*n/L) * V_n

  QW = W_{0,2} - W_R - sum W_p

  W_{0,2}(V_n,V_m) = 32L*sinh^2(L/4)*(L^2-16*pi^2*m*n) /
                      [(L^2+16*pi^2*m^2)*(L^2+16*pi^2*n^2)]  [eq 4.2]

  sum W_p(V_n,V_m) = sum_{1<k<=e^L} Lambda(k)*k^{-1/2} * exp(2*pi*i*(n-m)*log(k)/L)
                                                              [eq 4.3]

  W_R: involves rho(x) = e^{x/2}/(e^x-e^{-x}), digamma, 2F1  [eqs 4.5-4.7]

  delta_N = (1/sqrt(L)) * sum_{n=-N}^{N} V_n   [eq 5.9]

  xi_hat(z) = 2*L^{-1/2} * sin(z*L/2) * sum xi_j / (z - 2*pi*j/L)  [eq 5.25]

  Zeros of xi_hat(z) = eigenvalues of D_log^{(lambda,N)} = zeta zeros!
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, power, sinh, sin, cos, exp

mp.dps = 30


def von_mangoldt(k):
    """Lambda(k): log(p) if k=p^m, else 0."""
    if k < 2:
        return 0.0
    n = k
    # Check if k is a prime power
    for p in range(2, int(k**0.5) + 2):
        if n % p == 0:
            # k is divisible by p. Check if k = p^m
            while n % p == 0:
                n //= p
            if n == 1:
                return np.log(p)
            else:
                return 0.0
    # k itself is prime
    return np.log(k)


# ─── W_{0,2}: equation (4.2) ───

def W02_matrix(N, L):
    """
    W_{0,2}(V_n, V_m) = 32L * sinh^2(L/4) * (L^2 - 16*pi^2*m*n) /
                         [(L^2 + 16*pi^2*m^2) * (L^2 + 16*pi^2*n^2)]
    """
    dim = 2 * N + 1
    L2 = L**2
    p2 = (4 * np.pi)**2  # 16*pi^2
    prefactor = 32 * L * np.sinh(L / 4)**2
    W = np.zeros((dim, dim))
    for idx_n in range(dim):
        n = idx_n - N
        denom_n = L2 + p2 * n**2
        for idx_m in range(idx_n, dim):
            m = idx_m - N
            denom_m = L2 + p2 * m**2
            numer = L2 - p2 * m * n
            val = prefactor * numer / (denom_n * denom_m)
            W[idx_n, idx_m] = val
            W[idx_m, idx_n] = val
    return W


# ─── W_p: equation (4.3) ───

def Wp_matrix(N, L):
    """
    sum_p W_p(V_n, V_m) = sum_{1 < k <= exp(L)} Lambda(k) * k^{-1/2} *
                           exp(2*pi*i*(n-m)*log(k)/L)

    Taking real part (QW is real-symmetric):
    = sum_{k} Lambda(k) * k^{-1/2} * cos(2*pi*(n-m)*log(k)/L)
    """
    dim = 2 * N + 1
    k_max = int(np.exp(L)) + 1
    W = np.zeros((dim, dim))

    # Precompute von Mangoldt weights
    vM = []
    for k in range(2, k_max + 1):
        lk = von_mangoldt(k)
        if lk > 0:
            vM.append((k, lk))

    for idx_n in range(dim):
        n = idx_n - N
        for idx_m in range(idx_n, dim):
            m = idx_m - N
            diff = n - m
            total = 0.0
            for k, lk in vM:
                total += lk * k**(-0.5) * np.cos(2 * np.pi * diff * np.log(k) / L)
            W[idx_n, idx_m] = total
            W[idx_m, idx_n] = total

    return W


# ─── W_R: archimedean (simplified leading terms) ───

def WR_matrix(N, L):
    """
    W_R from paper's formula (3.14):
    W_R#(F) = (1/2)(log(4*pi) + gamma)*F(1) + integral_1^inf (x^{1/2}*F(x)-F(1))/(x-x^{-1}) d*x

    For F(x) = x^{2*pi*i*d/L} where d = n-m, F(1) = 1.

    In log coordinates t = log(x), x = e^t, t in [0, L/2]:
    integral = integral_0^{L/2} (e^{t/2+2*pi*i*d*t/L} - 1) / (2*sinh(t)) dt

    The integrand is BOUNDED near t=0:
    (e^{t*(1/2+2*pi*i*d/L)} - 1)/(2*sinh(t)) -> (1/2+2*pi*i*d/L)/2 as t->0

    For the SYMMETRIC form (both t>0 and t<0):
    W_R(n,m) = (log(4*pi)+gamma)/2 + integral_0^{L/2} Re[(e^{t*alpha}-1)/(2*sinh(t))] dt
             + integral_0^{L/2} Re[(e^{t*alpha_bar}-1)/(2*sinh(t))] dt

    where alpha = 1/2 + 2*pi*i*d/L and alpha_bar = 1/2 - 2*pi*i*d/L.

    Simplifying: 2*Re[...] = integral_0^{L/2} (e^{t/2}*cos(2*pi*d*t/L)-1+e^{-t/2}*cos(2*pi*d*t/L)-1) / (2*sinh(t))
    Hmm, the -t part contributes differently. Let me just compute the +t integral.

    Actually for W_R# (the "positive part"), only t > 0 is integrated:
    W_R#(V_n,V_m) = c0 + integral_0^{L/2} [e^{t*(1/2+2*pi*i*d/L)} - 1] / (2*sinh(t)) dt

    Taking real part:
    Re[...] = integral_0^{L/2} [e^{t/2}*cos(2*pi*d*t/L) - 1] / (2*sinh(t)) dt
    """
    dim = 2 * N + 1
    W = np.zeros((dim, dim))

    c0 = 0.5 * (np.log(4 * np.pi) + float(euler))
    half_L = L / 2.0

    # Quadrature: fine grid avoiding t=0 singularity
    n_quad = 5000
    t_vals = np.linspace(1e-10, half_L, n_quad)
    dt = t_vals[1] - t_vals[0]
    sinh_t = 2 * np.sinh(t_vals)
    exp_half_t = np.exp(t_vals / 2)

    for idx_n in range(dim):
        n = idx_n - N
        for idx_m in range(idx_n, dim):
            m = idx_m - N
            d = n - m

            # Integrand: [e^{t/2}*cos(2*pi*d*t/L) - 1] / (2*sinh(t))
            if d == 0:
                integrand = (exp_half_t - 1) / sinh_t
            else:
                integrand = (exp_half_t * np.cos(2 * np.pi * d * t_vals / L) - 1) / sinh_t

            val = c0 + np.sum(integrand) * dt
            W[idx_n, idx_m] = val
            W[idx_m, idx_n] = val

    return W


# ─── Secular equation for eigenvalues ───

def xi_hat(z, xi_vec, N, L):
    """
    xi_hat(z) = 2*L^{-1/2} * sin(z*L/2) * sum_{j=-N}^{N} xi_j / (z - 2*pi*j/L)

    Zeros of xi_hat give the eigenvalues (= zeta zeros).
    """
    total = 0.0
    for idx in range(2 * N + 1):
        j = idx - N
        d_j = 2 * np.pi * j / L
        if abs(z - d_j) < 1e-12:
            return 0.0  # pole
        total += xi_vec[idx] / (z - d_j)
    return 2 * L**(-0.5) * np.sin(z * L / 2) * total


def find_xi_hat_zeros(xi_vec, N, L, z_min=0.5, z_max=100, n_scan=50000):
    """Find zeros of xi_hat by bisection."""
    z_range = np.linspace(z_min, z_max, n_scan)
    roots = []
    prev = xi_hat(z_range[0], xi_vec, N, L)
    for i in range(1, len(z_range)):
        val = xi_hat(z_range[i], xi_vec, N, L)
        if not np.isfinite(val) or not np.isfinite(prev):
            prev = val
            continue
        if prev * val < 0:
            # Bisect
            lo, hi = z_range[i-1], z_range[i]
            for _ in range(80):
                mid = (lo + hi) / 2
                fmid = xi_hat(mid, xi_vec, N, L)
                if not np.isfinite(fmid):
                    break
                if fmid * xi_hat(lo, xi_vec, N, L) < 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        prev = val
    return roots


# ─── Main ───

if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 19e: Connes Spectral Operator — Exact Formulas")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # Parameters matching the paper
    lam_sq = 14  # lambda^2 = 14, so lambda = sqrt(14)
    L = np.log(lam_sq)  # L = log(lambda^2) = 2*log(lambda)
    N = 60

    print(f"\n  lambda^2 = {lam_sq}, L = {L:.6f}")
    print(f"  N = {N}, dim = {2*N+1}")
    print(f"  Scaling eigenvalues: 2*pi*j/L, spacing = {2*np.pi/L:.6f}")
    print(f"  Max eigenvalue: {2*np.pi*N/L:.4f}")

    # --- Step 1: Build QW components ---
    print("\n--- Step 1: Building QW components ---")

    print("  W_{0,2} (closed form, eq 4.2)...")
    W02 = W02_matrix(N, L)
    print(f"    W02[0,0] = {W02[N,N]:.10f}")
    print(f"    Rank: {np.linalg.matrix_rank(W02, tol=1e-8)}")

    print("  sum W_p (von Mangoldt, eq 4.3)...")
    Wp = Wp_matrix(N, L)
    print(f"    Wp[0,0] = {Wp[N,N]:.10f}")

    print("  W_R (archimedean, numerical integration)...")
    WR = WR_matrix(N, L)
    print(f"    WR[0,0] = {WR[N,N]:.10f}")

    # --- Step 2: Assemble QW ---
    print("\n--- Step 2: QW = W_{0,2} - W_R - sum W_p ---")
    QW = W02 - WR - Wp
    QW = (QW + QW.T) / 2  # ensure symmetry

    eigvals, eigvecs = np.linalg.eigh(QW)
    print(f"  Eigenvalues: min = {eigvals[0]:+.8f}, max = {eigvals[-1]:+.8f}")
    print(f"  Positive: {np.sum(eigvals > 0)}, Negative: {np.sum(eigvals < 0)}")

    # Min eigenvector
    xi = eigvecs[:, 0]
    eps_min = eigvals[0]
    print(f"  Min eigenvalue epsilon_N = {eps_min:+.10f}")
    print(f"  Min eigenvector norm: {np.linalg.norm(xi):.10f}")

    # --- Step 3: Find zeros of xi_hat ---
    print("\n--- Step 3: Zeros of xi_hat(z) ---")
    print("  xi_hat(z) = 2/sqrt(L) * sin(zL/2) * sum xi_j/(z - 2*pi*j/L)")
    print("  Searching for zeros in [0.5, 80]...")

    roots = find_xi_hat_zeros(xi, N, L, z_min=0.5, z_max=80.0, n_scan=100000)

    # Filter: remove roots that are close to scaling eigenvalues (poles)
    scaling_eigs = [2 * np.pi * j / L for j in range(-N, N+1)]
    filtered_roots = []
    for r in roots:
        is_pole = any(abs(r - se) < 0.01 for se in scaling_eigs)
        if not is_pole:
            filtered_roots.append(r)

    print(f"  Found {len(roots)} raw roots, {len(filtered_roots)} after filtering poles")

    # Compare with zeta zeros
    print(f"\n  {'#':>3s}  {'xi_hat zero':>14s}  {'zeta zero':>14s}  {'diff':>12s}  {'rel%':>8s}")
    for i in range(min(20, len(filtered_roots), len(gammas))):
        r = filtered_roots[i]
        g = gammas[i]
        diff = r - g
        rel = abs(diff / g) * 100
        print(f"  {i+1:3d}  {r:14.6f}  {g:14.6f}  {diff:+12.6f}  {rel:7.3f}%")

    # --- Step 4: Try with QW from zeros (circular, as baseline) ---
    print("\n--- Step 4: Comparison with QW from zeros (circular) ---")

    def mellin_Vn(s, n, L):
        """Mellin transform with correct 2*pi*n/L normalization."""
        alpha = complex(s) + 2j * np.pi * n / L
        log_lam = L / 2
        if abs(alpha) < 1e-15:
            return 2 * log_lam
        return 2 * np.sinh(alpha * log_lam) / alpha

    dim = 2 * N + 1
    QW_zeros = np.zeros((dim, dim))
    for gamma in gammas[:200]:
        rho = 0.5 + 1j * gamma
        M = np.array([mellin_Vn(rho, idx - N, L) for idx in range(dim)])
        QW_zeros += 2 * np.real(np.outer(M, np.conj(M)))
    QW_zeros = (QW_zeros + QW_zeros.T) / 2

    eigvals_z, eigvecs_z = np.linalg.eigh(QW_zeros)
    xi_zeros = eigvecs_z[:, 0]
    print(f"  QW_zeros: min eigenvalue = {eigvals_z[0]:+.8f}")

    roots_z = find_xi_hat_zeros(xi_zeros, N, L, z_min=0.5, z_max=80.0, n_scan=100000)
    filtered_z = [r for r in roots_z if not any(abs(r - 2*np.pi*j/L) < 0.01 for j in range(-N, N+1))]

    print(f"  Found {len(filtered_z)} zeros from QW_zeros")
    print(f"\n  {'#':>3s}  {'from QW_zeros':>14s}  {'from QW_exact':>14s}  {'zeta zero':>14s}")
    for i in range(min(10, len(filtered_z), len(filtered_roots), len(gammas))):
        rz = filtered_z[i] if i < len(filtered_z) else float('nan')
        re = filtered_roots[i] if i < len(filtered_roots) else float('nan')
        g = gammas[i]
        print(f"  {i+1:3d}  {rz:14.6f}  {re:14.6f}  {g:14.6f}")

    print("\n" + "=" * 70)
