"""
Session 22 — Grok's four proof checks.

Verify the hypotheses needed for the contour-integral argument:
1. Zero simplicity: |F'(gamma_k)| bounded below
2. Leading coefficients: F(0), F'(0) match Xi(0), Xi'(0)
3. Zero counting: argument of F on |z|=R matches N_Xi(R)
4. Growth control: Jensen integral difference on |z|=R
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad, zeta, gamma)
import sympy, time

mp.dps = 100

def primes_up_to(n):
    return list(sympy.primerange(2, n + 1))

def build_and_solve(lam_sq, N):
    L = log(mpf(lam_sq)); eL = exp(L)
    vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq):
            vM.append((pk, lp, log(pk))); pk *= p
    dim = 2 * N + 1

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

    wr_d = {}
    for nv in range(N + 1):
        w_c = euler + log(4 * pi * (eL - 1) / (eL + 1))
        def integrand(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        integral = quad(integrand, [mpf(0), L])
        wr_d[nv] = w_c + integral; wr_d[-nv] = wr_d[nv]

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
            tau[i, j] = w02 - wr - wp; tau[j, i] = tau[i, j]

    E, ER = eig(tau, left=False, right=True)
    evals = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    eps_N = evals[0][0]; idx = evals[0][1]

    xi = [ER[j, idx].real for j in range(dim)]
    xs = sum(xi); sqL = mpmath.sqrt(L)
    if abs(xs) > mpf(10) ** (-20):
        xi = [x * sqL / xs for x in xi]
    return xi, eps_N, L, N


def F_func(z, xi, L, N):
    lam = exp(L / 2)
    s = sin(z * L / 2)
    total = mpf(0)
    for j in range(-N, N + 1):
        dj = 2 * pi * j / L
        d = z - dj
        if abs(d) > mpf(10) ** (-20):
            total += xi[j + N] / d
    xi_hat = 2 * mpmath.sqrt(1 / L) * s * total
    return -mpc(0, 1) * mpmath.power(lam, -mpc(0, 1) * z) * xi_hat


def Xi_func(z):
    s = mpc(mpf(1) / 2, z)
    return mpf(1) / 2 * s * (s - 1) * mpmath.power(pi, -s / 2) * gamma(s / 2) * zeta(s)


# ========================================
print("Building QW at lambda^2=14, N=30, 100dp...", flush=True)
t0 = time.time()
xi, eps_N, L, N = build_and_solve(14, 30)
print(f"  eps_N = {mpmath.nstr(eps_N, 10)}  ({time.time()-t0:.0f}s)", flush=True)

gammas = np.load("_zeros_500.npy")

# ========================================
# CHECK 1
# ========================================
print(f"\n{'='*70}", flush=True)
print("CHECK 1: |F'(gamma_k)| -- derivative at zeta zeros", flush=True)
print(f"{'='*70}", flush=True)

h = mpf(10) ** (-30)
for k in range(10):
    gk = mpf(float(gammas[k]))
    F_val = F_func(gk, xi, L, N)
    F_deriv = (F_func(gk + h, xi, L, N) - F_func(gk - h, xi, L, N)) / (2 * h)
    Xi_deriv = (Xi_func(gk + h) - Xi_func(gk - h)) / (2 * h)

    logF = float(mpmath.log10(abs(F_val))) if abs(F_val) > 0 else -999
    logFd = float(mpmath.log10(abs(F_deriv))) if abs(F_deriv) > 0 else -999
    logXd = float(mpmath.log10(abs(Xi_deriv))) if abs(Xi_deriv) > 0 else -999
    ratio = F_deriv / Xi_deriv if abs(Xi_deriv) > mpf(10) ** (-90) else mpc(0)

    print(f"  k={k+1:2d} g={float(gk):10.4f}  log|F|={logF:7.1f}  log|F'|={logFd:7.1f}  log|Xi'|={logXd:7.1f}  F'/Xi'={mpmath.nstr(ratio, 8)}", flush=True)

# ========================================
# CHECK 2
# ========================================
print(f"\n{'='*70}", flush=True)
print("CHECK 2: F(0) vs Xi(0)", flush=True)
print(f"{'='*70}", flush=True)

F0 = F_func(mpf(0), xi, L, N)
Xi0 = Xi_func(mpf(0))
print(f"  F(0)  = {mpmath.nstr(F0, 30)}", flush=True)
print(f"  Xi(0) = {mpmath.nstr(Xi0, 30)}", flush=True)
if abs(Xi0) > 0:
    print(f"  F(0)/Xi(0) = {mpmath.nstr(F0 / Xi0, 15)}", flush=True)

Fd0 = (F_func(h, xi, L, N) - F_func(-h, xi, L, N)) / (2 * h)
Xd0 = (Xi_func(h) - Xi_func(-h)) / (2 * h)
print(f"  F'(0)  = {mpmath.nstr(Fd0, 20)}", flush=True)
print(f"  Xi'(0) = {mpmath.nstr(Xd0, 20)}", flush=True)

# ========================================
# CHECK 3
# ========================================
print(f"\n{'='*70}", flush=True)
print("CHECK 3: Zero counting via argument principle", flush=True)
print(f"{'='*70}", flush=True)

for R in [50.0]:
    n_xi_pos = sum(1 for g in gammas if 0 < g < R)
    print(f"  R={R}: Xi has {n_xi_pos} positive zeros (x2 for pairs = {2*n_xi_pos})", flush=True)

    # Compute winding number via d/dtheta arg(F)
    n_pts = 200
    thetas = [2 * pi * k / n_pts for k in range(n_pts)]
    total_arg = mpf(0)
    F_prev = F_func(mpc(R, 0), xi, L, N)
    for i in range(1, n_pts + 1):
        theta = 2 * pi * i / n_pts
        z = R * mpmath.exp(mpc(0, theta))
        F_cur = F_func(z, xi, L, N)
        if abs(F_prev) > 0 and abs(F_cur) > 0:
            darg = mpmath.im(mpmath.log(F_cur / F_prev))
            total_arg += darg
        F_prev = F_cur
    n_F = total_arg / (2 * pi)
    print(f"  Winding number of F on |z|={R}: {mpmath.nstr(n_F, 8)}", flush=True)

# ========================================
# CHECK 4
# ========================================
print(f"\n{'='*70}", flush=True)
print("CHECK 4: Jensen integral on |z|=R", flush=True)
print(f"{'='*70}", flush=True)

for R in [30.0]:
    def jensen_F(theta):
        z = R * mpmath.exp(mpc(0, theta))
        Fz = F_func(z, xi, L, N)
        return mpmath.log(abs(Fz)) if abs(Fz) > 0 else mpf(0)

    def jensen_Xi(theta):
        z = R * mpmath.exp(mpc(0, theta))
        Xz = Xi_func(z)
        return mpmath.log(abs(Xz)) if abs(Xz) > 0 else mpf(0)

    JF = quad(jensen_F, [0, 2 * pi]) / (2 * pi)
    JXi = quad(jensen_Xi, [0, 2 * pi]) / (2 * pi)
    print(f"  R={R}: Jensen(F)  = {mpmath.nstr(JF, 15)}", flush=True)
    print(f"  R={R}: Jensen(Xi) = {mpmath.nstr(JXi, 15)}", flush=True)
    print(f"  R={R}: |diff|     = {mpmath.nstr(abs(JF - JXi), 10)}", flush=True)

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
