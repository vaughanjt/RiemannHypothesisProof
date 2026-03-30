"""
Session 24: Prolate overlap measurement.

Compute overlap of xi_lambda (Weil eigenvector) with Gaussian/Hermite
approximations to the prolate functions, at varying lambda.

If overlap -> 1 as lambda -> inf, the paper's "educated guess" holds.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad)
import sympy, time

mp.dps = 50

def primes_up_to(n):
    return list(sympy.primerange(2, n + 1))

def build_xi(lam_sq, N=30):
    L = log(mpf(lam_sq)); eL = exp(L)
    vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq):
            vM.append((pk, lp, log(pk))); pk *= p
    dim = 2 * N + 1; al = {}
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
        def ig(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        wr_d[nv] = w_c + quad(ig, [mpf(0), L]); wr_d[-nv] = wr_d[nv]
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
    eps = evals[0][0]; idx = evals[0][1]
    xi = [float(ER[j, idx].real) for j in range(dim)]
    xs = sum(xi); sqL = float(mpmath.sqrt(L))
    if abs(xs) > 1e-20:
        xi = [x * sqL / xs for x in xi]
    return np.array(xi), float(eps), float(L), N

print("PROLATE OVERLAP MEASUREMENT (Session 24)", flush=True)
print("=" * 70, flush=True)

header = f"{'lam^2':>6s} {'L':>6s} {'eps_N':>12s} {'ovlp(h0)':>10s} {'best_Gauss':>10s} {'sigma*':>7s}"
print(header, flush=True)
print("-" * 60, flush=True)

for lam_sq in [14, 25, 50, 100, 200, 500, 1000]:
    t0 = time.time()
    xi, eps, L_f, N = build_xi(lam_sq, 30)
    dim = 2 * N + 1
    xi_norm = xi / np.linalg.norm(xi)

    # Project Gaussian onto V_n basis via Fourier coefficients
    h0_coeffs = np.zeros(dim)
    for j in range(-N, N + 1):
        freq = 2 * np.pi * j / L_f
        h0_coeffs[j + N] = np.exp(-freq ** 2 / 2)
    h0_norm = h0_coeffs / np.linalg.norm(h0_coeffs)
    overlap_h0 = abs(np.dot(xi_norm, h0_norm))

    # Optimize Gaussian width sigma
    best_overlap = 0
    best_sigma = 0
    for sigma in np.linspace(0.05, 10.0, 200):
        h0s = np.zeros(dim)
        for j in range(-N, N + 1):
            freq = 2 * np.pi * j / L_f
            h0s[j + N] = np.exp(-freq ** 2 * sigma ** 2 / 2)
        nrm = np.linalg.norm(h0s)
        if nrm > 0:
            ov = abs(np.dot(xi_norm, h0s / nrm))
            if ov > best_overlap:
                best_overlap = ov
                best_sigma = sigma

    dt = time.time() - t0
    print(f"  {lam_sq:5d} {L_f:6.3f} {eps:12.2e} {overlap_h0:10.6f} {best_overlap:10.6f} {best_sigma:7.3f}  ({dt:.0f}s)", flush=True)

print("", flush=True)
print("If best_Gauss -> 1: prolate concentration holds.", flush=True)
print("If plateaus < 1: xi has non-Gaussian structure.", flush=True)
