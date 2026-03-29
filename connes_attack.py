"""
Session 22 — Attack script: reproduce paper + Rayleigh quotient test.

Correct W_p formula: q(U_n,U_m)(log k), NOT cos(2pi(n-m)lk/L).
  n!=m: q(y) = [sin(2*pi*m*y/L) - sin(2*pi*n*y/L)] / [pi*(n-m)]
  n==m: q(y) = 2*(L-y)/L * cos(2*pi*n*y/L)

Architecture:
  - Off-diagonal WR: exact Prop 4.3 (alpha_L via 2F1/digamma)
  - Diagonal WR: eq (4.4) quadrature at full precision
  - W02: closed formula
  - Wp: correct q kernel (fast arithmetic)
  - Eigenproblem: numpy float64 (sufficient for eigenvalue/eigenvector)
    For 200dp zeros: use mpmath secular equation evaluation
"""

import sys
import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
from mpmath import psi as polygamma
import time

# ====== Configuration ======
DPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_VAL = int(sys.argv[2]) if len(sys.argv) > 2 else 30
LAM_SQ = int(sys.argv[3]) if len(sys.argv) > 3 else 14

mp.dps = DPS
L = log(mpf(LAM_SQ))
eL = exp(L)
L_f = float(L)

print(f"lambda^2={LAM_SQ}, L={L_f:.6f}, N={N_VAL}, dim={2*N_VAL+1}, {DPS}dp", flush=True)

# ====== Prime powers ======
vM = []
for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
    if p > LAM_SQ:
        break
    lp_f = np.log(p)
    pk = p
    while pk <= LAM_SQ:
        vM.append((pk, lp_f, np.log(pk)))
        pk *= p
print(f"  {len(vM)} prime powers <= {LAM_SQ}", flush=True)


# ====== W_p with correct q kernel ======
def q_func(n, m, y):
    if n != m:
        return (np.sin(2*np.pi*m*y/L_f) - np.sin(2*np.pi*n*y/L_f)) / (np.pi*(n-m))
    else:
        return 2*(L_f - y)/L_f * np.cos(2*np.pi*n*y/L_f)

def Wp_element(n, m):
    total = 0.0
    for k, lk, logk in vM:
        total += lk * k**(-0.5) * q_func(n, m, logk)
    return total


# ====== W02 ======
L2_f = L_f**2
p2_f = (4*np.pi)**2
pf_f = 32*L_f*np.sinh(L_f/4)**2

def W02_element(n, m):
    return pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))


# ====== WR off-diagonal: exact Prop 4.3 ======
def alpha_L_exact(n):
    if n == 0:
        return mpf(0)
    z = exp(-2*L)
    a = pi*mpc(0, n)/L + mpf(1)/4
    h = hyp2f1(1, a, a+1, z)
    f1 = exp(-L/2) * (2*L/(L + 4*pi*mpc(0, n)) * h).imag
    d = digamma(a).imag / 2
    return (f1 + d) / pi


# ====== WR diagonal: eq (4.4) ======
def WR_diag(n, n_quad=20000):
    omega_0 = mpf(2)
    def omega(x):
        return 2*(1 - x/L)*cos(2*pi*n*x/L)
    w_const = (omega_0/2)*(euler + log(4*pi*(eL-1)/(eL+1)))
    dx = L/n_quad
    integral = mpf(0)
    for k in range(n_quad):
        x = dx*(k + mpf(1)/2)
        numer = exp(x/2)*omega(x) - omega_0
        denom = exp(x) - exp(-x)
        if abs(denom) > mpf(10)**(-40):
            integral += numer/denom
    integral *= dx
    return float(w_const + integral)


# ====== Build QW ======
N = N_VAL
dim = 2*N + 1
t0 = time.time()

# Precompute alpha_L
print("\nPrecomputing alpha_L...", flush=True)
alpha = {}
for n in range(-N, N+1):
    alpha[n] = float(alpha_L_exact(abs(n)))
    if n < 0:
        alpha[n] = -alpha[n]
print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

# Diagonal WR
print("Computing diagonal WR (eq 4.4)...", flush=True)
wr_diag = {}
for n_val in range(N+1):
    wr_diag[n_val] = WR_diag(n_val)
    wr_diag[-n_val] = wr_diag[n_val]
    if n_val % 20 == 0:
        print(f"  WR_diag[{n_val}] = {wr_diag[n_val]:+.8f}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

# Assemble QW
print("Assembling QW...", flush=True)
QW = np.zeros((dim, dim))
for i in range(dim):
    n = i - N
    for j in range(i, dim):
        m = j - N
        w02 = W02_element(n, m)
        wp = Wp_element(n, m)
        if n == m:
            wr = wr_diag[n]
        else:
            wr = (alpha[m] - alpha[n]) / (n - m)
        QW[i, j] = w02 - wr - wp
        QW[j, i] = QW[i, j]
QW = (QW + QW.T) / 2
print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

# Eigenvalues
print("\nEigenvalues...", flush=True)
eigvals, eigvecs = np.linalg.eigh(QW)
print(f"  Spectrum: [{eigvals[0]:+.6e}, {eigvals[-1]:+.6e}]", flush=True)
print(f"  Positive: {np.sum(eigvals>0)}, Negative: {np.sum(eigvals<0)}", flush=True)
print(f"  epsilon_N = {eigvals[0]:+.12e}", flush=True)

# Verify smallest is even
xi = eigvecs[:, 0].copy()
es = sum(abs(xi[N+k] - xi[N-k]) for k in range(1, N+1))
os = sum(abs(xi[N+k] + xi[N-k]) for k in range(1, N+1))
print(f"  Parity: {'EVEN' if es < os else 'ODD'} (even_score={es:.2e}, odd_score={os:.2e})", flush=True)

# Normalize: sum(xi) = sqrt(L)
xs = np.sum(xi)
if abs(xs) > 1e-30:
    xi = xi * np.sqrt(L_f) / xs
print(f"  sum(xi) = {np.sum(xi):.10f} (target {np.sqrt(L_f):.10f})", flush=True)

# ====== xi_hat zeros (high precision with mpmath) ======
gammas = np.load("_zeros_500.npy")

def xi_hat_mp(z):
    """xi_hat at mpmath precision."""
    z_mp = mpf(z)
    s = mpmath.sin(z_mp * L / 2)
    total = mpf(0)
    for j in range(-N, N+1):
        dj = 2*pi*j/L
        diff = z_mp - dj
        if abs(diff) > mpf(10)**(-20):
            total += mpf(xi[j+N]) / diff
    return 2 * L**(-mpf(1)/2) * s * total

def xi_hat_np(z):
    """xi_hat at float64 (fast scan)."""
    s = np.sin(z * L_f / 2)
    if abs(s) < 1e-60:
        return 0
    t = sum(xi[j+N] / (z - 2*np.pi*j/L_f)
            for j in range(-N, N+1)
            if abs(z - 2*np.pi*j/L_f) > 1e-12)
    return 2 * L_f**(-0.5) * s * t

# Point evaluation at known zeros
print(f"\nxi_hat at zeta zeros:", flush=True)
for i in range(min(10, len(gammas))):
    val = xi_hat_mp(gammas[i])
    print(f"  gamma_{i+1:2d} = {gammas[i]:12.6f}  xi_hat = {mpmath.nstr(val, 8):>14s}", flush=True)

# Zero scan
print(f"\nScanning zeros [0.5, 80]...", flush=True)
pole_spacing = 2*np.pi/L_f
zr = np.linspace(0.5, 80, 500000)
all_roots = []
prev = xi_hat_np(zr[0])
for i in range(1, len(zr)):
    val = xi_hat_np(zr[i])
    if np.isfinite(val) and np.isfinite(prev) and prev*val < 0 and abs(val) < 1e10:
        lo, hi = zr[i-1], zr[i]
        for _ in range(80):
            mid = (lo+hi)/2
            fm = xi_hat_np(mid)
            if np.isfinite(fm) and fm*xi_hat_np(lo) < 0:
                hi = mid
            else:
                lo = mid
        r = (lo+hi)/2
        all_roots.append(r)
    prev = val

# Classify: near a pole (2*pi*j/L) or genuine zero
genuine = []
for r in all_roots:
    # Distance to nearest pole
    j_near = round(r * L_f / (2*np.pi))
    pole = 2*np.pi*j_near/L_f
    if abs(r - pole) > pole_spacing * 0.2:  # more than 20% of spacing from pole
        genuine.append(r)

print(f"  {len(all_roots)} total roots, {len(genuine)} genuine (away from poles)", flush=True)
print(f"\n  {'#':>3s}  {'xi_hat zero':>14s}  {'zeta zero':>14s}  {'diff':>14s}  {'rel%':>8s}", flush=True)
for i in range(min(25, len(genuine), len(gammas))):
    r = genuine[i]
    g = gammas[i]
    print(f"  {i+1:3d}  {r:14.6f}  {g:14.6f}  {r-g:+14.6e}  {abs(r-g)/g*100:8.4f}%", flush=True)

# High-precision evaluation at genuine zeros
if DPS >= 100 and len(genuine) > 0:
    print(f"\nHigh-precision accuracy (mpmath at {DPS}dp):", flush=True)
    for i in range(min(10, len(genuine), len(gammas))):
        g = gammas[i]
        val = xi_hat_mp(g)
        print(f"  gamma_{i+1:2d}: |xi_hat| = {mpmath.nstr(abs(val), 6)}", flush=True)

# ====== Rayleigh quotient with prolate approximation ======
print(f"\n{'='*60}", flush=True)
print("RAYLEIGH QUOTIENT TEST", flush=True)
print(f"{'='*60}", flush=True)

# The prolate PSWF concentrated on [-L/2, L/2] with bandwidth W
# Bandwidth: the highest frequency in our basis is 2*pi*N/L
# PSWF_0(t) is well-approximated by a Gaussian for small c = W*L/2
# For simplicity, use the sinc kernel eigenvector as prolate approximation

# Build the sinc kernel matrix S_{jk} = sinc(j-k) on the V_n basis
# Actually, the prolate operator in this basis is the projection onto
# band-limited functions. For the V_n basis with |n| <= N, this is just
# the identity (all V_n are band-limited). So the prolate function IS
# delta_N up to normalization.

# Better approach: use the actual PSWF via the integral equation
# K(t,s) = sin(c(t-s))/(pi(t-s)) on [-1,1], c = pi*N (bandwidth param)
# This reduces to a matrix eigenvalue problem.

# For our attack, the key is: how does <k_lambda|QW|k_lambda> compare
# to epsilon_N? We can measure this for varying lambda.

# Simplest test: delta_N itself as a test vector
delta_N = np.ones(dim) / np.sqrt(L_f)  # delta_N in V_n basis: all coefficients 1/sqrt(L)
rayleigh_delta = delta_N @ QW @ delta_N / (delta_N @ delta_N)
print(f"\n  Rayleigh quotient with delta_N: {rayleigh_delta:+.10e}", flush=True)
print(f"  epsilon_N:                      {eigvals[0]:+.10e}", flush=True)
print(f"  Ratio (Rayleigh/epsilon):       {rayleigh_delta/eigvals[0] if eigvals[0] != 0 else float('inf'):.6f}", flush=True)

# Build PSWF approximation via sinc kernel
# Discrete prolate spheroidal sequences (DPSS/Slepian)
# The concentration problem: maximize energy in [-W, W] for sequences on [-N, N]
# For our basis, this is the Shannon number c = L * (2*pi*N/L) / (2*pi) = N
# So the PSWF is essentially the first DPSS with bandwidth parameter c

# Compute DPSS: eigenvectors of the sinc matrix B_{jk} = sin(pi*c*(j-k))/(pi*(j-k))
# where c is the fractional bandwidth
# For our problem, the relevant bandwidth is related to the spectral gap

# Actually for the Rayleigh quotient test, let's try something simpler:
# Use the ACTUAL eigenvector xi and measure how close it is to the prolate function.
# The prolate function k_lambda concentrated on [lambda^{-1}, lambda] has the form:
# k_lambda(u) = sum_{n=-inf}^{inf} V_n(lambda) V_n(u) / ||V_n||^2
# In our finite basis, this is just the vector with components V_n(lambda) = lambda^{2*pi*i*n/L}

# Reproducing kernel at lambda (= exp(L/2)):
lam = np.sqrt(LAM_SQ)
k_lambda = np.array([np.exp(2j*np.pi*n*np.log(lam)/L_f).real for n in range(-N, N+1)])
# Normalize
k_lambda = k_lambda / np.linalg.norm(k_lambda)

rayleigh_k = k_lambda @ QW @ k_lambda / (k_lambda @ k_lambda)
overlap = abs(np.dot(xi / np.linalg.norm(xi), k_lambda))

print(f"\n  Reproducing kernel k_lambda:", flush=True)
print(f"    Rayleigh quotient:   {rayleigh_k:+.10e}", flush=True)
print(f"    epsilon_N:           {eigvals[0]:+.10e}", flush=True)
print(f"    |overlap(xi, k_lam)|: {overlap:.10f}", flush=True)
print(f"    Gap: {abs(rayleigh_k - eigvals[0]):.6e}", flush=True)

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
