"""
Session 22: N=120 Connes operator — full QW matrix with exact formulas.

Key fixes:
1. Build full QW matrix (not tau) — off-diagonal = Prop 4.3 exact formula
2. Closed-form for gamma_L correction integral (no quadrature)
3. beta_L(0) from general formula (no quadrature)
4. Eigenvector selection: min |eigenvalue| among even eigenvectors
5. OFFSET computed with 50000-pt eq44 at n=0

Matrix structure:
  QW(V_n, V_m) = W02(n,m) - WR(n,m) - Wp(n,m)
  Diagonal WR: 2*gamma_L(n) - 2*beta_L(n) - OFFSET
  Off-diagonal WR: (alpha_L(m) - alpha_L(n))/(n-m)  [exact Prop 4.3]
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler,
                    sinh, exp, cos, sin, hyp2f1, digamma)
from mpmath import psi as polygamma
import time

mp.dps = 200

L = log(mpf(14))
L_float = float(L)
eL = exp(L)
eL_float = float(eL)
primes_list = [2, 3, 5, 7, 11, 13]

print("=" * 70, flush=True)
print("SESSION 22: Full QW matrix, exact formulas, N=120", flush=True)
print("=" * 70, flush=True)


# === Exact functions ===

def corr_closed_form(L_val):
    v = exp(L_val / 2)
    Fv = -log(v + 1) + log(v**2 + 1) / 2 + mpmath.atan(v)
    F1 = -log(mpf(2)) / 2 + pi / 4
    return Fv - F1

def alpha_L_exact(n, L_val):
    if n == 0: return mpf(0)
    z = exp(-2 * L_val)
    a = pi * mpc(0, n) / L_val + mpf(1) / 4
    h = hyp2f1(1, a, a + 1, z)
    f1 = exp(-L_val / 2) * (2 * L_val / (L_val + 4 * pi * mpc(0, n)) * h).imag
    d = digamma(a).imag / 2
    return (f1 + d) / pi

def beta_L_exact(n, L_val):
    z = exp(-2 * L_val)
    a = mpc(0, pi * n / L_val) + mpf(1) / 4
    h = hyp2f1(1, a, a + 1, z)
    coeff = 2 * L_val / (4 * pi * n - mpc(0, L_val))
    f1_term = -L_val * exp(-L_val / 2) * (coeff * h).imag
    phi_val = mpmath.lerchphi(z, 2, a)
    phi_term = -exp(-L_val / 2) / 4 * phi_val.real
    pg_term = polygamma(1, a).real / 4
    return (f1_term + phi_term + pg_term) / L_val

def gamma_L_exact(n, L_val):
    z = exp(-2 * L_val)
    a = mpc(0, pi * n / L_val) + mpf(1) / 4
    if n == 0:
        cos_minus_1 = mpf(0)
    else:
        h = hyp2f1(1, a, a + 1, z)
        coeff = 2 * L_val / (L_val + 4 * pi * mpc(0, n))
        f1_term = -exp(-L_val / 2) * (coeff * h).real
        h0 = hyp2f1(mpf(1) / 4, 1, mpf(5) / 4, z)
        f1_const = 2 * exp(-L_val / 2) * h0
        d_term = -(digamma(a) - digamma(mpf(1) / 4)).real / 2
        cos_minus_1 = f1_term + f1_const + d_term
    corr = corr_closed_form(L_val)
    eL2 = exp(L_val / 2)
    cw = (log((eL2 - 1) / (eL2 + 1)) / 2
          + mpmath.atan(eL2) - pi / 4
          + euler / 2 + log(8 * pi) / 2)
    return cos_minus_1 + corr + cw


# === OFFSET ===
print("\nComputing OFFSET (50000-pt eq44 at n=0)...", flush=True)
t0 = time.time()
g0 = gamma_L_exact(0, L)
b0 = beta_L_exact(0, L)
wr_prop43_0 = 2 * g0 - 2 * b0
n_quad = 50000
dx_q = L / n_quad
w_const_0 = euler + log(4 * pi * (eL - 1) / (eL + 1))
integral_44 = mpf(0)
for k in range(n_quad):
    x = dx_q * (k + mpf(1) / 2)
    omega_x = 2 * (1 - x / L)
    numer = exp(x / 2) * omega_x - 2
    denom_val = exp(x) - exp(-x)
    if abs(denom_val) > mpf(10)**(-40):
        integral_44 += numer / denom_val
integral_44 *= dx_q
wr_eq44_0 = w_const_0 + integral_44
OFFSET = float(wr_prop43_0 - wr_eq44_0)
print(f"  OFFSET = {OFFSET:.15f}  ({time.time()-t0:.0f}s)", flush=True)


# === Precompute alpha_L for all n ===
N = 120
dim = 2 * N + 1
t_start = time.time()
print(f"\nN={N}, dim={dim}", flush=True)

print("\nPrecomputing alpha_L...", flush=True)
alpha_cache = {}
for n in range(-N, N + 1):
    alpha_cache[n] = float(alpha_L_exact(abs(n), L))
    if n < 0:
        alpha_cache[n] = -alpha_cache[n]  # alpha_L is odd
    if abs(n) % 30 == 0:
        print(f"  alpha_L({n:+4d}) = {alpha_cache[n]:+.12f}", flush=True)
print(f"  Done ({time.time()-t_start:.0f}s)", flush=True)

# === Prime power table ===
vM = {}
for p in primes_list:
    pk = p
    lp = np.log(p)
    while pk <= eL_float + 1:
        vM[pk] = lp
        pk *= p

# === Build full QW matrix ===
print(f"\nBuilding QW ({dim}x{dim})...", flush=True)
L2_f = L_float**2
p2_f = (4 * np.pi)**2
pf_f = 32 * L_float * np.sinh(L_float / 4)**2

QW = np.zeros((dim, dim))
total = dim * (dim + 1) // 2
count = 0

for idx_n in range(dim):
    n = idx_n - N
    for idx_m in range(idx_n, dim):
        m = idx_m - N

        # W_{0,2}
        w02 = pf_f * (L2_f - p2_f * m * n) / (
            (L2_f + p2_f * m**2) * (L2_f + p2_f * n**2))

        # W_p (depends on n-m)
        d = n - m
        wp = sum(lk * k**(-0.5) * np.cos(2 * np.pi * d * np.log(k) / L_float)
                 for k, lk in vM.items())

        # W_R
        if n == m:
            # Diagonal: exact Prop 4.3 with OFFSET
            gamma_n = float(gamma_L_exact(abs(n), L))
            beta_n = float(beta_L_exact(abs(n), L))
            wr = 2 * gamma_n - 2 * beta_n - OFFSET
        else:
            # Off-diagonal: exact Prop 4.3
            wr = (alpha_cache[m] - alpha_cache[n]) / (n - m)

        QW[idx_n, idx_m] = w02 - wr - wp
        QW[idx_m, idx_n] = QW[idx_n, idx_m]

        count += 1
        if count % 5000 == 0:
            print(f"  {count}/{total} ({time.time()-t_start:.0f}s)", flush=True)

# Symmetrize
QW = (QW + QW.T) / 2
print(f"  Done ({time.time()-t_start:.0f}s)", flush=True)

# === Eigenvalues ===
print(f"\nEigenvalues (numpy eigh)...", flush=True)
eigvals, eigvecs = np.linalg.eigh(QW)
print(f"  Done ({time.time()-t_start:.0f}s)", flush=True)

print(f"\n  Spectrum: [{eigvals[0]:+.6f}, {eigvals[-1]:+.6f}]", flush=True)
print(f"  Positive: {np.sum(eigvals > 0)}, Negative: {np.sum(eigvals < 0)}", flush=True)

# 10 nearest 0
near_zero_idx = np.argsort(np.abs(eigvals))[:10]
print(f"\n  10 eigenvalues nearest 0:", flush=True)
for rank, idx in enumerate(near_zero_idx):
    ev = eigvals[idx]
    xi_test = eigvecs[:, idx]
    es = sum(abs(xi_test[N + k] - xi_test[N - k]) for k in range(1, N + 1))
    os = sum(abs(xi_test[N + k] + xi_test[N - k]) for k in range(1, N + 1))
    parity = "EVEN" if es < os else "odd"
    print(f"    {rank+1:2d}. eval={ev:+.8f}  {parity}", flush=True)


# === Find best eigenvectors ===
gammas = np.load("_zeros_500.npy")

def test_eigvec(xi_raw, label):
    xi = xi_raw.copy()
    xs = np.sum(xi)
    if abs(xs) > 1e-30:
        xi = xi * np.sqrt(L_float) / xs
    def xi_hat(z):
        s = np.sin(z * L_float / 2)
        if abs(s) < 1e-60: return 0
        t = sum(xi[j + N] / (z - 2 * np.pi * j / L_float)
                for j in range(-N, N + 1)
                if abs(z - 2 * np.pi * j / L_float) > 1e-12)
        return 2 * L_float**(-0.5) * s * t

    print(f"\n  --- {label} ---", flush=True)
    for i in range(min(5, len(gammas))):
        val = xi_hat(gammas[i])
        print(f"    xi_hat(g{i+1}={gammas[i]:.4f}) = {val:+.6e}", flush=True)

    # Zero scan
    zr = np.linspace(0.5, 80, 500000)
    roots = []
    prev = xi_hat(zr[0])
    for i in range(1, len(zr)):
        val = xi_hat(zr[i])
        if np.isfinite(val) and np.isfinite(prev) and prev * val < 0 and abs(val) < 1e10:
            lo, hi = zr[i-1], zr[i]
            for _ in range(100):
                mid = (lo + hi) / 2
                fm = xi_hat(mid)
                if np.isfinite(fm) and fm * xi_hat(lo) < 0: hi = mid
                else: lo = mid
            r = (lo + hi) / 2
            if not any(abs(r - 2 * np.pi * j / L_float) < 0.03
                       for j in range(-N, N + 1)):
                roots.append(r)
        prev = val

    print(f"\n  {len(roots)} zeros found:", flush=True)
    print(f"  {'#':>3s}  {'xi_hat':>14s}  {'zeta':>14s}  {'diff':>12s}  {'%':>8s}", flush=True)
    for i in range(min(20, len(roots), len(gammas))):
        r = roots[i]; g = gammas[i]
        print(f"  {i+1:3d}  {r:14.6f}  {g:14.6f}  {r-g:+12.6f}  {abs(r-g)/g*100:7.3f}%", flush=True)
    if roots:
        n_c = min(10, len(roots), len(gammas))
        diffs = [abs(roots[i] - gammas[i]) for i in range(n_c)]
        print(f"  g1 accuracy: {abs(roots[0]-gammas[0]):.6f}", flush=True)
        print(f"  Mean |diff| ({n_c}): {np.mean(diffs):.6f}", flush=True)
    return roots


# Scan top-5 near-zero even eigenvectors
print(f"\n{'='*70}", flush=True)
print("Scanning near-zero even eigenvectors:", flush=True)
even_eigs = []
for i in range(dim):
    xi_test = eigvecs[:, i]
    es = sum(abs(xi_test[N + k] - xi_test[N - k]) for k in range(1, N + 1))
    os = sum(abs(xi_test[N + k] + xi_test[N - k]) for k in range(1, N + 1))
    if es < os:
        even_eigs.append((abs(eigvals[i]), eigvals[i], i))
even_eigs.sort()

# Quick scan: just evaluate at gamma_1
print("\n  Quick scan (xi_hat at g1):", flush=True)
for rank, (absev, ev, idx) in enumerate(even_eigs[:10]):
    xi_test = eigvecs[:, idx].copy()
    xs = np.sum(xi_test)
    if abs(xs) > 1e-30:
        xi_test = xi_test * np.sqrt(L_float) / xs
    s = np.sin(gammas[0] * L_float / 2)
    t = sum(xi_test[j + N] / (gammas[0] - 2 * np.pi * j / L_float)
            for j in range(-N, N + 1)
            if abs(gammas[0] - 2 * np.pi * j / L_float) > 1e-12)
    v1 = 2 * L_float**(-0.5) * s * t
    print(f"    #{rank+1}: eval={ev:+.8f}  xi_hat(g1)={v1:+.6e}", flush=True)

# Full test on the best candidate
best_rank = min(range(min(10, len(even_eigs))),
                key=lambda r: abs(eval(f"v1") if r == 0 else 0))

# Actually just test the top 3
for rank in range(min(3, len(even_eigs))):
    _, ev, idx = even_eigs[rank]
    test_eigvec(eigvecs[:, idx], f"Even #{rank+1} (eval={ev:+.6f})")

print(f"\nTotal: {time.time()-t_start:.0f}s", flush=True)
print("=" * 70, flush=True)
