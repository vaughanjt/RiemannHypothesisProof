"""
Session 25b: Fourier bump test — the Grok-identified proof path.

THESIS (Grok): The prolate educated guess is NOT needed for a proof.
The measured rates (eps_N ~ 10^{-2.3L}, eigenvector freezing at N~40)
+ the variational equation + Fourier bumps close the proof directly.

KEY TEST: Compute |xi_hat(gamma_k)| two ways:
  1. F_T[xi](gamma_k) — T-inner product (NO u^{-1/2} factor)
     = L * sum_n xi_n * sinc(n - gamma_k * L / (2*pi))
  2. M[xi](1/2+i*gamma_k) — full Mellin transform (HAS u^{-1/2})
     = sum_n xi_n * 2*sinh(alpha_n * L/2) / alpha_n
     where alpha_n = 2*pi*i*n/L - 1/2 + i*gamma_k

If |F_T| / eps_N ~ O(1) but |M| / eps_N ~ 10^{41}:
  -> The 41-order gap comes ENTIRELY from u^{-1/2}
  -> The T-inner product version controls xi_hat at zeta zeros
  -> The Rouche argument goes through using F_T, not M
  -> Only remaining piece: prove eps_N and freezing rates analytically

Also: the variational equation tau*xi = eps*xi gives
  f^T * tau * xi = eps * (f^T * xi)  for any bump f
We verify this as a consistency check.
"""

import numpy as np
import mpmath
import sympy
import time
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad)

mp.dps = 50

# First 10 non-trivial zeta zeros (imaginary parts, high precision)
ZETA_ZEROS = [
    14.134725141734693, 21.022039638771555, 25.010857580145688,
    30.424876125859513, 32.935061587739189, 37.586178158825671,
    40.918719012147495, 43.327073280914999, 48.005150881167160,
    49.773832477672302
]


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def build_xi_and_tau(lam_sq, N=30):
    """Build xi AND return the tau matrix (needed for variational check)."""
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
        def ig(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        wr_d[nv] = w_c + quad(ig, [mpf(0), L])
        wr_d[-nv] = wr_d[nv]
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
            tau[i, j] = w02 - wr - wp
            tau[j, i] = tau[i, j]
    E, ER = eig(tau, left=False, right=True)
    evals = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    eps = evals[0][0]; idx = evals[0][1]
    xi = [float(ER[j, idx].real) for j in range(dim)]
    xs = sum(xi); sqL = float(mpmath.sqrt(L))
    if abs(xs) > 1e-20:
        xi = [x * sqL / xs for x in xi]

    # Convert tau to numpy for variational check
    tau_np = np.array([[float(tau[i, j]) for j in range(dim)] for i in range(dim)])

    return np.array(xi), float(eps), float(L), N, tau_np


def fourier_bump_test(xi, eps, L, N, tau_np, gamma_k):
    """Test the Fourier bump at zeta zero gamma_k.

    Returns:
      F_T: T-inner product evaluation (no u^{-1/2})
      M_val: Mellin transform evaluation (with u^{-1/2})
      var_check: variational consistency |f^T tau xi - eps * f^T xi|
    """
    dim = 2 * N + 1

    # Build Fourier bump vector: f_n = sinc(n - gamma_k * L / (2*pi))
    center = gamma_k * L / (2 * np.pi)
    f_bump = np.zeros(dim)
    for j in range(dim):
        n = j - N
        x = n - center
        if abs(x) < 1e-12:
            f_bump[j] = 1.0
        else:
            f_bump[j] = np.sin(np.pi * x) / (np.pi * x)

    # F_T[xi](gamma_k) = L * sum_n xi_n * sinc(n - center)
    #                   = L * (xi . f_bump)
    FT = L * np.dot(xi, f_bump)

    # Mellin: M[xi](1/2+i*gamma_k)
    # = sum_n xi_n * 2*sinh(alpha_n * L/2) / alpha_n
    # where alpha_n = 2*pi*i*n/L - 1/2 + i*gamma_k
    M_val = 0.0 + 0.0j
    for j in range(dim):
        n = j - N
        alpha_n = complex(-0.5, 2 * np.pi * n / L + gamma_k)
        sinh_val = np.sinh(alpha_n * L / 2)
        M_val += xi[j] * 2 * sinh_val / alpha_n

    # Variational check: f^T tau xi should equal eps * f^T xi
    ftau_xi = f_bump @ tau_np @ xi
    f_xi = np.dot(f_bump, xi)
    var_check = abs(ftau_xi - eps * f_xi)

    return FT, M_val, var_check, f_xi


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("FOURIER BUMP TEST (Session 25b)")
    print("Comparing T-inner product (no u^{-1/2}) vs Mellin (with u^{-1/2})")
    print("=" * 90)
    print()

    for lam_sq in [100, 1000, 5000, 10000]:
        t0 = time.time()
        lam = np.sqrt(lam_sq)
        print(f"lam^2 = {lam_sq}, building xi + tau...", end="", flush=True)
        xi, eps, L, N, tau_np = build_xi_and_tau(lam_sq, N=30)
        print(f" ({time.time()-t0:.0f}s)", flush=True)
        print(f"  eps_N = {eps:.6e}  (log10 = {np.log10(abs(eps)):.1f})")
        print(f"  L = {L:.4f}  N = {N}  dim = {2*N+1}")
        print()

        # Determine which zeros are "in range" (gamma_k * L/(2pi) < N)
        max_freq = gamma_k_max = 2 * np.pi * N / L
        print(f"  Max V_n frequency: 2*pi*{N}/L = {max_freq:.1f}")
        print(f"  Zeros in range: gamma_k < {max_freq:.1f}")
        print()

        print(f"  {'gamma_k':>10} {'|F_T|':>12} {'|M|':>12} {'F_T/eps':>12} {'M/eps':>12} {'gap':>8} {'var_err':>10}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

        for gamma_k in ZETA_ZEROS:
            if gamma_k * L / (2 * np.pi) > N - 1:
                # Zero is beyond V_n bandwidth — results unreliable
                note = " (out of band)"
            else:
                note = ""

            FT, M_val, var_err, f_xi = fourier_bump_test(xi, eps, L, N, tau_np, gamma_k)

            abs_FT = abs(FT)
            abs_M = abs(M_val)
            abs_eps = abs(eps)

            if abs_eps > 0:
                FT_ratio = abs_FT / abs_eps
                M_ratio = abs_M / abs_eps
            else:
                FT_ratio = float('inf')
                M_ratio = float('inf')

            if abs_FT > 0:
                gap_orders = np.log10(abs_M / abs_FT) if abs_FT > 0 else float('inf')
            else:
                gap_orders = float('inf')

            print(f"  {gamma_k:10.4f} {abs_FT:12.4e} {abs_M:12.4e} "
                  f"{FT_ratio:12.4e} {M_ratio:12.4e} {gap_orders:8.1f} {var_err:10.2e}{note}")

        print()
        print(f"  INTERPRETATION:")
        print(f"  If F_T/eps ~ O(1): T-inner product controls xi at zeta zeros -> proof path open")
        print(f"  If M/eps >> F_T/eps: gap is from u^{{-1/2}} factor, not intrinsic")
        print(f"  gap column = log10(|M|/|F_T|) = orders of magnitude from u^{{-1/2}}")
        print()
        print(f"  Time: {time.time()-t0:.0f}s")
        print()
        print("=" * 90)
        print()

    # Final summary
    print("CONCLUSION:")
    print("If F_T/eps is bounded (not growing with lam^2) -> Rouche argument works")
    print("If F_T/eps grows with lam^2 -> need additional control")
