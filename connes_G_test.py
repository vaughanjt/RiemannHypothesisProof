"""
Session 23: Grok's reduced function G_{lambda,N}(z) tests.

G(z) = [xi_hat(z) / (2*L^{-1/2}*sin(zL/2))] * exp(-a_N - i*b_N*z)
     = [sum xi_j/(z - 2*pi*j/L)] * exp(-a_N - i*b_N*z)

where a_N, b_N chosen so G(0) = Xi(0), G'(0) = Xi'(0).

Tests:
1. Compute |G(gamma_k)| — should decay with lambda (tracking eps_N)
2. Jensen integral of log|G| on |z|=30 — diff with Xi should be < 0.01
3. Phase: argument change of G on |z|=50 should be +20*pi
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
        wr_d[nv] = w_c + quad(integrand, [mpf(0), L]); wr_d[-nv] = wr_d[nv]
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


def partial_frac_sum(z, xi, L, N):
    """The partial fraction sum: sum xi_j / (z - 2*pi*j/L).
    This is xi_hat(z) / [2*L^{-1/2}*sin(zL/2)] — no sin factor."""
    total = mpf(0)
    for j in range(-N, N + 1):
        dj = 2 * pi * j / L
        d = z - dj
        if abs(d) > mpf(10) ** (-20):
            total += xi[j + N] / d
    return total


def Xi_func(z):
    """Riemann Xi(z) = xi(1/2+iz)."""
    s = mpf(1) / 2 + mpc(0, 1) * z
    return mpf(1) / 2 * s * (s - 1) * mpmath.power(pi, -s / 2) * gamma(s / 2) * zeta(s)


def compute_G(lam_sq, N=30):
    """Build QW, solve, compute reduced G with normalization constants."""
    t0 = time.time()
    xi, eps_N, L, N_val = build_and_solve(lam_sq, N)

    # Raw partial fraction at z=0 and nearby for derivative
    S0 = partial_frac_sum(mpf(0), xi, L, N_val)
    h = mpf(10) ** (-30)
    S0_plus = partial_frac_sum(h, xi, L, N_val)
    S0_minus = partial_frac_sum(-h, xi, L, N_val)
    S0_deriv = (S0_plus - S0_minus) / (2 * h)

    # Xi at 0
    Xi0 = Xi_func(mpf(0))
    Xi0_plus = Xi_func(h)
    Xi0_minus = Xi_func(-h)
    Xi0_deriv = (Xi0_plus - Xi0_minus) / (2 * h)

    # G(z) = S(z) * exp(-a - i*b*z) where S(z) = partial_frac_sum
    # G(0) = S(0) * exp(-a) = Xi(0) => exp(-a) = Xi(0)/S(0)
    # G'(0) = [S'(0) - i*b*S(0)] * exp(-a) = Xi'(0)
    # => S'(0)*exp(-a) - i*b*S(0)*exp(-a) = Xi'(0)
    # => S'(0)*Xi(0)/S(0) - i*b*Xi(0) = Xi'(0)
    # => i*b = [S'(0)*Xi(0)/S(0) - Xi'(0)] / Xi(0)
    # => i*b = S'(0)/S(0) - Xi'(0)/Xi(0)

    if abs(S0) > mpf(10) ** (-20) and abs(Xi0) > mpf(10) ** (-20):
        exp_neg_a = Xi0 / S0
        a_N = -mpmath.log(exp_neg_a)
        ib_N = S0_deriv / S0 - Xi0_deriv / Xi0
        b_N = ib_N / mpc(0, 1)  # b = ib/(i)
    else:
        a_N = mpf(0); b_N = mpf(0)

    def G(z):
        S = partial_frac_sum(z, xi, L, N_val)
        return S * exp(-a_N - mpc(0, 1) * b_N * z)

    dt = time.time() - t0
    return G, eps_N, L, a_N, b_N, xi, N_val, dt


gammas = np.load("_zeros_500.npy")

# ==========================================================
print("=" * 70, flush=True)
print("TEST 1: |G(gamma_k)| — should decay with lambda", flush=True)
print("=" * 70, flush=True)

print(f"\n{'lam^2':>6s} {'eps_N':>12s} {'a_N':>14s} {'b_N':>14s} {'log|G(g1)|':>12s} {'log|G(g2)|':>12s} {'log|G(g3)|':>12s}", flush=True)
print("-" * 85, flush=True)

for lam_sq in [11, 14, 20, 25, 50]:
    G, eps, L, a_N, b_N, xi, N_val, dt = compute_G(lam_sq, N=30)
    logs = []
    for k in range(3):
        gk = mpf(float(gammas[k]))
        Gval = G(gk)
        lv = float(mpmath.log10(abs(Gval))) if abs(Gval) > 0 else -999
        logs.append(lv)

    print(f"{lam_sq:6d} {mpmath.nstr(eps, 4):>12s} {mpmath.nstr(a_N.real, 6):>14s} {mpmath.nstr(b_N.real, 6):>14s} {logs[0]:12.1f} {logs[1]:12.1f} {logs[2]:12.1f}  ({dt:.0f}s)", flush=True)


# ==========================================================
print(f"\n{'=' * 70}", flush=True)
print("TEST 2: Jensen integral of log|G| vs log|Xi| on |z|=30", flush=True)
print("=" * 70, flush=True)

for lam_sq in [14, 50]:
    G, eps, L, a_N, b_N, xi, N_val, dt = compute_G(lam_sq, N=30)
    R = 30.0

    def jensen_G(theta):
        z = R * mpmath.exp(mpc(0, theta))
        Gz = G(z)
        return mpmath.log(abs(Gz)) if abs(Gz) > 0 else mpf(0)

    def jensen_Xi(theta):
        z = R * mpmath.exp(mpc(0, theta))
        s = mpf(1) / 2 + mpc(0, 1) * z
        Xz = mpf(1) / 2 * s * (s - 1) * mpmath.power(pi, -s / 2) * gamma(s / 2) * zeta(s)
        return mpmath.log(abs(Xz)) if abs(Xz) > 0 else mpf(0)

    JG = quad(jensen_G, [0, 2 * pi]) / (2 * pi)
    JXi = quad(jensen_Xi, [0, 2 * pi]) / (2 * pi)
    diff = abs(JG - JXi)
    print(f"  lam^2={lam_sq}: Jensen(G)={mpmath.nstr(JG, 10)}, Jensen(Xi)={mpmath.nstr(JXi, 10)}, |diff|={mpmath.nstr(diff, 6)}", flush=True)


# ==========================================================
print(f"\n{'=' * 70}", flush=True)
print("TEST 3: Winding number of G on |z|=50", flush=True)
print("=" * 70, flush=True)

G14, _, _, _, _, _, _, _ = compute_G(14, N=30)
R = 50.0
n_pts = 400
total_arg = mpf(0)
G_prev = G14(mpc(R, 0))
for i in range(1, n_pts + 1):
    theta = 2 * pi * i / n_pts
    z = R * mpmath.exp(mpc(0, theta))
    G_cur = G14(z)
    if abs(G_prev) > 0 and abs(G_cur) > 0:
        darg = mpmath.im(mpmath.log(G_cur / G_prev))
        total_arg += darg
    G_prev = G_cur
winding = total_arg / (2 * pi)
print(f"  lam^2=14, R=50: winding number = {mpmath.nstr(winding, 8)}", flush=True)
print(f"  Expected: +10 (positive zeros of Xi inside |z|=50)", flush=True)

print(f"\n{'=' * 70}", flush=True)
print("DONE", flush=True)
