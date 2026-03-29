"""
Connes N=120: Full reproduction of paper's Table 1.
lambda=sqrt(14), N=120, 200dp.

Optimization: compute a_n only for n=0..120 (symmetry a_{-n}=a_n).
Use exact 2F1+digamma for b_n (fast). Numerical integral for a_n (slow but parallelizable).
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    sinh, exp, cos, sin, hyp2f1, digamma, eig)
import time

mp.dps = 200

L = log(mpf(14))
N = 120
dim = 2 * N + 1
L_val = float(L)
eL = exp(L)
L2 = L * L
p2 = 16 * pi * pi
pf = 32 * L * sinh(L / 4)**2

primes_list = [2, 3, 5, 7, 11, 13]  # primes <= 14

print(f"N={N}, dim={dim}, L={L_val:.6f}, 200dp")
t0 = time.time()

# ═══ STEP 1: Compute b_n (fast: exact special functions) ═══
print("\n[1/4] Computing b_n (exact 2F1+digamma)...")
b = {}
for n in range(-N, N + 1):
    if n == 0:
        b[0] = mpf(0)
        continue
    z = exp(-2 * L)
    a_arg = pi * mpc(0, n) / L + mpf(1) / 4
    h = hyp2f1(1, a_arg, a_arg + 1, z)
    f1 = exp(-L / 2) * (2 * L / (L + 4 * pi * mpc(0, n)) * h).imag
    d = digamma(a_arg).imag / 2
    alpha_WR = (f1 + d) / pi

    W02_n0 = pf * L2 / (L2 * (L2 + p2 * n * n))
    WR_n0 = -alpha_WR / n
    Wp_n0 = mpf(0)
    for p_ in primes_list:
        lp = log(mpf(p_)); pk = mpf(p_)
        while pk <= eL:
            Wp_n0 += lp * pk**(-mpf(1)/2) * cos(2*pi*n*log(pk)/L)
            pk *= p_
    b[n] = n * (W02_n0 - WR_n0 - Wp_n0)

    if abs(n) % 30 == 0:
        print(f"  b[{n:+4d}] = {float(b[n]):+.10f}  ({time.time()-t0:.0f}s)")

print(f"  b_n done ({time.time()-t0:.0f}s)")

# ═══ STEP 2: Compute a_n (slow: 200dp numerical integral) ═══
print("\n[2/4] Computing a_n (eq 4.4 at 200dp)...")
a = {}
w_const_base = (euler + log(4 * pi * (eL - 1) / (eL + 1)))
Wp_diag = mpf(0)
for p_ in primes_list:
    lp = log(mpf(p_)); pk = mpf(p_)
    while pk <= eL: Wp_diag += lp * pk**(-mpf(1)/2); pk *= p_

n_quad = 5000  # quadrature points

for n_val in range(N + 1):
    dx = L / n_quad
    integral = mpf(0)
    for k in range(n_quad):
        x = dx * (k + mpf(1) / 2)
        omega_x = 2 * (1 - x / L) * cos(2 * pi * n_val * x / L)
        numer = exp(x / 2) * omega_x - 2  # omega_0 = 2
        denom = exp(x) - exp(-x)
        if abs(denom) > mpf(10)**(-180):
            integral += numer / denom
    integral *= dx
    WR_nn = w_const_base + integral  # omega_0/2 * w_const = 1 * w_const

    W02_nn = pf * (L2 - p2 * n_val * n_val) / ((L2 + p2 * n_val * n_val)**2)
    a[n_val] = W02_nn - WR_nn - Wp_diag
    a[-n_val] = a[n_val]

    if n_val % 20 == 0:
        print(f"  a[{n_val:3d}] = {float(a[n_val]):+.10f}  ({time.time()-t0:.0f}s)")

print(f"  a_n done ({time.time()-t0:.0f}s)")

# ═══ STEP 3: Build tau matrix ═══
print("\n[3/4] Building tau matrix...")
tau = mpmatrix(dim, dim)
for i in range(dim):
    ni = i - N
    for j in range(dim):
        nj = j - N
        if ni == nj:
            tau[i, j] = a[ni]
        else:
            tau[i, j] = (b[ni] - b[nj]) / (ni - nj)

print(f"  Matrix built ({time.time()-t0:.0f}s)")

# ═══ STEP 4: Eigenvalue computation ═══
print("\n[4/4] Computing eigenvalues (241x241 at 200dp)...")
try:
    E, ER, EL_mat = eig(tau, left=True, right=True)
    print(f"  Eigenvalues computed ({time.time()-t0:.0f}s)")

    eigenvalues = sorted([(float(E[i].real), i) for i in range(dim)])
    print(f"  Min: {eigenvalues[0][0]:+.6f}")
    print(f"  Max: {eigenvalues[-1][0]:+.6f}")
    print(f"  Pos: {sum(1 for e,_ in eigenvalues if e>0)}")
    print(f"  Neg: {sum(1 for e,_ in eigenvalues if e<0)}")

    # Find smallest even eigenvector
    best_even_eval = float('inf')
    best_even_xi = None
    for ev, idx in eigenvalues:
        xi = np.array([float(ER[j, idx].real) for j in range(dim)])
        es = sum(abs(xi[N+k]-xi[N-k]) for k in range(1, N+1))
        os = sum(abs(xi[N+k]+xi[N-k]) for k in range(1, N+1))
        if es < os and ev < best_even_eval:
            best_even_eval = ev
            best_even_xi = xi

    if best_even_xi is not None:
        print(f"  Smallest even eigenvalue: {best_even_eval:+.6f}")
        xi = best_even_xi
    else:
        print("  No even eigenvector found, using smallest overall")
        xi = np.array([float(ER[j, eigenvalues[0][1]].real) for j in range(dim)])

    # Normalize: sum(xi) = sqrt(L)
    xs = np.sum(xi)
    if abs(xs) > 1e-30:
        xi = xi * np.sqrt(L_val) / xs

    # Find xi_hat zeros
    gammas = np.load("_zeros_500.npy")

    def xi_hat(z):
        s = np.sin(z * L_val / 2)
        if abs(s) < 1e-60: return 0
        t = sum(xi[j+N]/(z-2*np.pi*j/L_val) for j in range(-N, N+1) if abs(z-2*np.pi*j/L_val) > 1e-12)
        return 2 * L_val**(-0.5) * s * t

    print("\n  Scanning for xi_hat zeros...")
    zr = np.linspace(0.5, 80, 400000)
    roots = []
    prev = xi_hat(zr[0])
    for i in range(1, len(zr)):
        val = xi_hat(zr[i])
        if np.isfinite(val) and np.isfinite(prev) and prev*val < 0 and abs(val) < 1e10:
            lo, hi = zr[i-1], zr[i]
            for _ in range(100):
                mid = (lo+hi)/2
                fm = xi_hat(mid)
                if np.isfinite(fm) and fm*xi_hat(lo) < 0: hi = mid
                else: lo = mid
            root = (lo+hi)/2
            if not any(abs(root-2*np.pi*j/L_val) < 0.03 for j in range(-N, N+1)):
                roots.append(root)
        prev = val

    print(f"\n  {len(roots)} zeros found:")
    print(f"  {'#':>3}  {'xi_hat':>14}  {'zeta zero':>14}  {'diff':>12}  {'rel%':>8}")
    for i in range(min(25, len(roots), len(gammas))):
        r = roots[i]; g = gammas[i]
        print(f"  {i+1:3d}  {r:14.6f}  {g:14.6f}  {r-g:+12.6f}  {abs(r-g)/g*100:7.3f}%")

except Exception as ex:
    print(f"  ERROR: {ex}")
    import traceback
    traceback.print_exc()

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("=" * 70)
