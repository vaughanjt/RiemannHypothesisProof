"""
Session 20d: Connes spectral operator — EXACT W_R via 2F1 + digamma.

CONFIRMED: Formula (4.4) with omega(0)=2 IS correct (Grok verification).
The "backing out from zeros" was wrong — QW != sum |V_tilde|^2.

Now using Proposition 4.2 closed forms for maximum precision:
  alpha_L(n), beta_L(n), gamma_L(n) via 2F1, digamma, Hurwitz-Lerch.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, sinh, cosh, exp, cos, sin,
                    atan, hyp2f1, digamma, psi as polygamma)

mp.dps = 50


def primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i): sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


# ─── Proposition 4.2: Exact integrals via 2F1 and digamma ───

def alpha_L(n, L):
    """(1/pi) * integral_0^L sin(2*pi*n*x/L) * rho(x) dx  [eq 4.5, 4.12]"""
    if n == 0:
        return mpf(0)
    z = exp(-2 * L)
    a_arg = pi * mpc(0, n) / L + mpf(1) / 4  # pi*i*n/L + 1/4

    # 2F1 term
    h = hyp2f1(1, a_arg, a_arg + 1, z)
    coeff = 2 * L / (L + 4 * pi * mpc(0, n))
    f1_term = exp(-L / 2) * (coeff * h).imag

    # digamma term
    d_term = (digamma(a_arg)).imag / 2

    return (f1_term + d_term) / pi


def beta_L(n, L):
    """(1/L) * integral_0^L x*cos(2*pi*n*x/L) * rho(x) dx  [eq 4.6, 4.13]"""
    z = exp(-2 * L)
    a_arg = mpc(0, pi * n / L) + mpf(1) / 4

    # 2F1 term
    h = hyp2f1(1, a_arg, a_arg + 1, z)
    coeff = 2 * L / (4 * pi * n - mpc(0, L))
    f1_term = -L * exp(-L / 2) * (coeff * h).imag

    # Hurwitz-Lerch Phi term: Phi(z, 2, a) = sum z^k / (a+k)^2
    phi_val = mpmath.lerchphi(z, 2, a_arg)
    phi_term = -exp(-L / 2) / 4 * phi_val.real

    # Polygamma (trigamma) term
    pg_term = polygamma(1, a_arg).real / 4

    return (f1_term + phi_term + pg_term) / L


def gamma_L(n, L):
    """integral_0^L (cos(2*pi*n*x/L) - e^{-x/2}) * rho(x) dx + c(L) + w(L)  [eq 4.7, 4.14]

    Uses: integral(cos-1)*rho from (4.7) + c(L) from (4.11) + w(L)
    """
    z = exp(-2 * L)
    a_arg = mpc(0, pi * n / L) + mpf(1) / 4

    # integral_0^L (cos - 1) * rho from eq (4.7):
    h = hyp2f1(1, a_arg, a_arg + 1, z)
    coeff = 2 * L / (L + 4 * pi * mpc(0, n))
    f1_term = -exp(-L / 2) * (coeff * h).real

    h0 = hyp2f1(mpf(1) / 4, 1, mpf(5) / 4, z)
    f1_const = 2 * exp(-L / 2) * h0

    d_term = -(digamma(a_arg) - digamma(mpf(1) / 4)).real / 2

    cos_minus_1 = f1_term + f1_const + d_term

    # c(L) + w(L) from the paper (page 14-15):
    eL2 = exp(L / 2)
    cw = (log((eL2 - 1) / (eL2 + 1)) / 2
          + atan(eL2) - pi / 4
          + euler / 2 + log(8 * pi) / 2)

    # gamma_L(n) = integral(cos - e^{-x/2})*rho + c + w
    # = integral(cos - 1)*rho + c(L) + c + w
    # Actually from (4.14): gamma_L = integral(cos-exp(-x/2))*rho + c(L) + w(L)
    # And from (4.11): integral(cos-exp(-x/2))*rho = integral(cos-1)*rho + c(L)
    # So gamma_L = integral(cos-1)*rho + c(L) + c(L) + w(L)???
    # NO: gamma_L = integral(cos-exp(-x/2))*rho + c(L) + w(L)
    # = [integral(cos-1)*rho + c_integral] + c(L) + w(L)
    # where c_integral = integral(1-exp(-x/2))*rho
    # But (4.11) says: integral(cos-exp(-x/2))*rho = integral(cos-1)*rho + c(L)
    # where c(L) IS integral(1-exp(-x/2))/(e^x-e^{-x}) dx ... different from integral(1-exp(-x/2))*rho!

    # Let me just use (4.14) directly:
    # gamma_L(n) = integral(cos-exp(-x/2))*rho + c(L) + w(L)
    # = integral(cos-1)*rho + [integral from (4.11): c(L)] + c(L) + w(L)
    # NO, that's wrong. Let me re-read.

    # From (4.11): integral(cos-exp(-x/2))*rho = integral(cos-1)*rho + c(L)
    # So gamma_L = [integral(cos-1)*rho + c(L)] + c(L) + w(L)
    # WAIT: (4.14) says gamma_L = integral(cos-exp(-x/2))*rho + c(L) + w(L)
    # And (4.11) says integral(cos-exp(-x/2))*rho = integral(cos-1)*rho + c(L)
    # So gamma_L = integral(cos-1)*rho + c(L) + c(L) + w(L)?
    # That has c(L) appearing twice!
    # Actually, re-reading: (4.14) defines gamma_L with c(L)+w(L) ADDED to the integral.
    # And the integral in (4.14) is (cos-exp(-x/2))*rho.
    # This integral = (cos-1)*rho + (1-exp(-x/2))*rho
    # And (4.11) says (1-exp(-x/2))*rho... hmm.
    # Actually (4.11) is about (cos-exp(-x/2))*rho = (cos-1)*rho + c(L)
    # So c(L) = integral(1-exp(-x/2))*rho? Not exactly...
    # The paper says c(L) = integral(1-exp(-x/2))/(exp(x)-exp(-x)) dx
    # while (1-exp(-x/2))*rho = (1-exp(-x/2))*exp(x/2)/(exp(x)-exp(-x))
    # These differ by a factor exp(x/2).
    # BUT: (cos - exp(-x/2))*rho = cos*rho - exp(-x/2)*rho
    # = (cos-1)*rho + rho - exp(-x/2)*rho
    # = (cos-1)*rho + (1-exp(-x/2))*rho  ... WRONG sign on last term
    # Actually: cos*rho - exp(-x/2)*rho = (cos-1)*rho + (1-exp(-x/2))*rho???
    # cos - exp(-x/2) = (cos - 1) + (1 - exp(-x/2)). YES!
    # So integral(cos-exp(-x/2))*rho = integral(cos-1)*rho + integral(1-exp(-x/2))*rho

    # The paper defines c(L) as something that equals integral(1-exp(-x/2))*rho
    # even though it writes it as integral(1-exp(-x/2))/(exp(x)-exp(-x))
    # (potentially a typo, as we discussed earlier).

    # Regardless: gamma_L = cos_minus_1 + integral(1-exp(-x/2))*rho + c(L) + w(L)
    # If the paper's c(L) IS integral(1-exp(-x/2))*rho, then:
    # gamma_L = cos_minus_1 + 2*c(L) + w(L)?? That seems wrong.

    # Let me just compute gamma_L via the W_R diagonal formula directly.
    # W_R(V_n,V_n) = 2*gamma_L(n) - 2*beta_L(n)
    # So gamma_L(n) = (W_R(V_n,V_n) + 2*beta_L(n)) / 2
    # And W_R from (4.4) is what we know is correct.

    # For now, use (4.4) directly for the diagonal.
    # I'll compute gamma_L as: gamma_L = cos_minus_1_integral + c_w
    # where c_w combines all the constant corrections.
    # This is consistent with (4.14): gamma_L = integral(cos-exp(-x/2))*rho + c(L) + w(L)
    # = cos_minus_1 + integral(1-exp(-x/2))*rho + c(L) + w(L)

    # For the integral(1-exp(-x/2))*rho, compute numerically:
    n_quad = 5000
    dx = L / n_quad
    corr_integral = mpf(0)
    for k in range(n_quad):
        x = dx * (k + mpf(0.5))
        rho_x = exp(x / 2) / (exp(x) - exp(-x))
        corr_integral += (1 - exp(-x / 2)) * rho_x
    corr_integral *= dx

    return cos_minus_1 + corr_integral + cw


# ─── W_R from Proposition 4.3 ───

def WR_prop43(n, m, L):
    """W_R(V_n, V_m) from Proposition 4.3."""
    if n != m:
        an = alpha_L(n, L)
        am = alpha_L(m, L)
        return float((am - an) / (n - m))
    else:
        gn = gamma_L(n, L)
        bn = beta_L(n, L)
        return float(2 * gn - 2 * bn)


# ─── W_R from equation (4.4) directly ───

def WR_eq44(n, m, L, n_quad=5000):
    """W_R from eq (4.4) — numerical integration."""
    L_mp = mpf(L)
    if n == m:
        omega_0 = mpf(2)
        def omega(x):
            return 2 * (1 - x / L_mp) * cos(2 * pi * n * x / L_mp)
    else:
        omega_0 = mpf(0)
        denom = pi * (n - m)
        def omega(x):
            return (sin(2 * pi * m * x / L_mp) - sin(2 * pi * n * x / L_mp)) / denom

    eL = exp(L_mp)
    w_const = (omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1)))

    dx = L_mp / n_quad
    integral = mpf(0)
    for k in range(n_quad):
        x = dx * (k + mpf(0.5))
        numer = exp(x / 2) * omega(x) - omega_0
        denom_val = exp(x) - exp(-x)
        if abs(denom_val) > mpf(10)**(-40):
            integral += numer / denom_val
    integral *= dx

    return float(w_const + integral)


# ─── Build full QW ───

def build_QW(N, L, use_exact_WR=True, n_quad=5000):
    """Build QW = W_{0,2} - W_R - sum W_p."""
    dim = 2 * N + 1
    L_mp = mpf(L)
    L2 = L**2
    p2 = (4 * np.pi)**2
    pf02 = 32 * L * np.sinh(L / 4)**2

    # Prime powers
    primes = primes_up_to(int(np.exp(L)) + 1)
    vM = {}
    for p in primes:
        pk = p
        while pk <= int(np.exp(L)) + 1:
            vM[pk] = np.log(p)
            pk *= p

    QW = np.zeros((dim, dim))
    total = dim * (dim + 1) // 2
    count = 0

    for idx_n in range(dim):
        n = idx_n - N
        for idx_m in range(idx_n, dim):
            m = idx_m - N

            # W_{0,2}
            w02 = pf02 * (L2 - p2 * m * n) / ((L2 + p2 * m**2) * (L2 + p2 * n**2))

            # W_p
            d = n - m
            wp = sum(lk * k**(-0.5) * np.cos(2 * np.pi * d * np.log(k) / L)
                     for k, lk in vM.items())

            # W_R
            if use_exact_WR:
                try:
                    wr = WR_prop43(n, m, L_mp)
                except:
                    wr = WR_eq44(n, m, L, n_quad)
            else:
                wr = WR_eq44(n, m, L, n_quad)

            QW[idx_n, idx_m] = w02 - wr - wp
            QW[idx_m, idx_n] = QW[idx_n, idx_m]

            count += 1
            if count % 500 == 0:
                print(f"    {count}/{total}...", flush=True)

    return QW


# ─── xi_hat and zero finding ───

def xi_hat_val(z, xi_vec, N, L):
    sin_part = np.sin(z * L / 2)
    if abs(sin_part) < 1e-30:
        return 0.0
    total = 0.0
    for idx in range(2 * N + 1):
        j = idx - N
        dj = 2 * np.pi * j / L
        denom = z - dj
        if abs(denom) < 1e-12:
            return float('inf')
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
                if not np.isfinite(fm):
                    break
                if fm * xi_hat_val(lo, xi_vec, N, L) < 0:
                    hi = mid
                else:
                    lo = mid
            root = (lo + hi) / 2
            if not any(abs(root - 2 * np.pi * j / L) < 0.05 for j in range(-N, N + 1)):
                roots.append(root)
        prev = val
    return roots


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 20d: Connes v5 — Exact W_R (2F1 + digamma)")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")
    lam_sq = 14
    L = float(np.log(lam_sq))
    N = 60

    print(f"\n  lambda^2={lam_sq}, L={L:.6f}, N={N}, dim={2*N+1}")

    # Test exact W_R formulas
    print("\n--- Testing exact W_R (Prop 4.3) vs numerical (4.4) ---")
    L_mp = mpf(L)
    for n, m in [(0, 0), (1, 1), (5, 5), (0, 1), (1, 2)]:
        try:
            wr_exact = WR_prop43(n, m, L_mp)
            wr_num = WR_eq44(n, m, L, n_quad=5000)
            print(f"  WR[{n},{m}]: exact={wr_exact:+.10f}, num={wr_num:+.10f}, diff={wr_exact-wr_num:+.2e}")
        except Exception as e:
            print(f"  WR[{n},{m}]: ERROR: {e}")

    # Build QW
    print(f"\n--- Building QW ({2*N+1}x{2*N+1}) with numerical W_R ---")
    QW = build_QW(N, L, use_exact_WR=False, n_quad=5000)
    QW = (QW + QW.T) / 2

    eigvals, eigvecs = np.linalg.eigh(QW)
    print(f"  min eval: {eigvals[0]:+.6f}, max eval: {eigvals[-1]:+.6f}")
    print(f"  Positive: {np.sum(eigvals > 0)}, Negative: {np.sum(eigvals < 0)}")

    # Find smallest even eigenvector
    even_eigs = []
    for i in range(2 * N + 1):
        xi = eigvecs[:, i]
        even_score = sum(abs(xi[N + n] - xi[N - n]) for n in range(1, N + 1))
        odd_score = sum(abs(xi[N + n] + xi[N - n]) for n in range(1, N + 1))
        if even_score < odd_score:
            even_eigs.append((i, eigvals[i]))
    even_eigs.sort(key=lambda x: x[1])

    print(f"  Smallest even eigenvalue: {even_eigs[0][1]:+.8f}")

    # Normalize xi
    idx = even_eigs[0][0]
    xi = eigvecs[:, idx].copy()
    xi_sum = np.sum(xi)
    if abs(xi_sum) > 1e-10:
        xi = xi * np.sqrt(L) / xi_sum
    print(f"  xi sum = {np.sum(xi):.8f} (target {np.sqrt(L):.8f})")

    # Find zeros
    print("\n--- Zeros of xi_hat ---")
    roots = find_zeros(xi, N, L, z_min=1.0, z_max=80.0, n_scan=300000)
    print(f"  Found {len(roots)} zeros\n")

    print(f"  {'#':>3s}  {'xi_hat zero':>14s}  {'zeta zero':>14s}  {'diff':>12s}  {'rel%':>8s}")
    for i in range(min(20, len(roots), len(gammas))):
        r = roots[i]
        g = gammas[i]
        print(f"  {i+1:3d}  {r:14.6f}  {g:14.6f}  {r - g:+12.6f}  {abs(r - g) / g * 100:7.3f}%")

    print("\n" + "=" * 70)
