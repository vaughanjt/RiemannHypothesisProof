"""
Session 25c: HIGH-PRECISION Fourier bump test.

The float64 test (25b) couldn't resolve F_T[xi](gamma_k) ~ eps_N ~ 10^{-49}.
Here we use mpmath at dps=80 (giving ~30 reliable digits after cancellation).

Precision budget:
  dps=80 -> eigenvector accurate to ~78 digits (gap eps_1-eps_0 ~ O(1))
  61-term sum -> ~76 digits after accumulation
  Cancellation to eps_N ~ 10^{-49} -> 76-49 = ~27 reliable digits

KEY QUESTION: Is |F_T[xi](gamma_k)| / eps_N bounded as lam^2 grows?
If yes -> Rouche argument proceeds, proof path is open.
"""

import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, sinh, hyp2f1, digamma, eig, quad, sqrt, nstr, fabs)
import sympy
import time

# First 10 zeta zeros at high precision
ZETA_ZEROS = [
    "14.134725141734693790457251983562",
    "21.022039638771554992628479593897",
    "25.010857580145688763213790992563",
    "30.424876125859513210311897530584",
    "32.935061587739189690662368964075",
    "37.586178158825671257217763480021",
    "40.918719012147495187398126914633",
    "43.327073280914999519496122165406",
    "48.005150881167159727942472749427",
    "49.773832477672302181916784678564",
]


def primes_up_to(n):
    return list(sympy.primerange(2, int(n) + 1))


def build_xi_mp(lam_sq, N=30):
    """Build xi as mpmath vector at full dps precision.

    Returns (xi_mp, eps, eps_1, L, N) where:
      xi_mp = list of mpf values (eigenvector components)
      eps = smallest eigenvalue (mpf)
      eps_1 = second smallest eigenvalue (mpf, for precision estimate)
      L = log(lam_sq) (mpf)
    """
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
    evals_sorted = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    eps = evals_sorted[0][0]; idx = evals_sorted[0][1]
    eps_1 = evals_sorted[1][0]

    # Extract eigenvector at FULL mpmath precision (not float64!)
    xi_mp = [ER[j, idx].real for j in range(dim)]
    xs = sum(xi_mp)
    sqL = sqrt(L)
    if fabs(xs) > mpf(10) ** (-20):
        xi_mp = [x * sqL / xs for x in xi_mp]

    return xi_mp, eps, eps_1, L, N


def fourier_bump_mp(xi_mp, L, N, gamma_k_str):
    """Compute F_T and M at full mpmath precision.

    F_T = L * sum_n xi_n * sinc(n - gamma_k * L / (2*pi))
        [T-inner product, no u^{-1/2}]

    M = sum_n xi_n * 2*sinh(alpha_n * L/2) / alpha_n
        where alpha_n = -1/2 + i*(2*pi*n/L + gamma_k)
        [Full Mellin transform at s=1/2+i*gamma_k]
    """
    dim = 2 * N + 1
    gamma_k = mpf(gamma_k_str)
    center = gamma_k * L / (2 * pi)

    FT = mpf(0)
    M_val = mpc(0, 0)

    for j in range(dim):
        n = j - N

        # sinc(n - center) = sin(pi*x)/(pi*x) where x = n - center
        x = mpf(n) - center
        if fabs(x) < mpf(10) ** (-(mp.dps - 10)):
            s = mpf(1)
        else:
            s = sin(pi * x) / (pi * x)
        FT += xi_mp[j] * s

        # Mellin kernel: 2*sinh(alpha*L/2)/alpha
        alpha = mpc(-mpf(1) / 2, 2 * pi * n / L + gamma_k)
        M_val += xi_mp[j] * 2 * sinh(alpha * L / 2) / alpha

    FT *= L
    return FT, M_val


# ======================================================================
if __name__ == "__main__":
    TARGET_DPS = 80

    print(f"HIGH-PRECISION FOURIER BUMP TEST (dps={TARGET_DPS})")
    print("=" * 90)
    print()

    for lam_sq in [14, 100, 1000]:
        mp.dps = TARGET_DPS
        t0 = time.time()
        print(f"lam^2 = {lam_sq}, dps = {TARGET_DPS}, building xi...", end="", flush=True)
        xi_mp, eps, eps_1, L, N = build_xi_mp(lam_sq, N=30)
        print(f" ({time.time()-t0:.0f}s)", flush=True)

        dim = 2 * N + 1
        max_freq = float(2 * pi * N / L)

        print(f"  eps_0 = {nstr(eps, 15)}  (log10 = {float(mpmath.log10(fabs(eps))):.1f})")
        print(f"  eps_1 = {nstr(eps_1, 15)}  (log10 = {float(mpmath.log10(fabs(eps_1))):.1f})")
        print(f"  gap   = {nstr(eps_1 - eps, 15)}")
        print(f"  eigenvec precision: ~{TARGET_DPS - 2:.0f} digits")
        print(f"  expected F_T precision: ~{TARGET_DPS - 2 - int(float(mpmath.log10(fabs(eps)))):.0f} digits")
        print(f"  L = {nstr(L, 8)}, max V_n freq = {max_freq:.1f}")
        print()

        # Verify eigenvector: check ||tau*xi - eps*xi|| / ||xi||
        # (Too expensive for large matrices; skip for now)

        print(f"  {'gamma_k':>10} {'in_band':>8} {'|F_T|':>20} {'|M|':>20} {'F_T/eps':>15} {'M/eps':>15}")
        print(f"  {'-'*10} {'-'*8} {'-'*20} {'-'*20} {'-'*15} {'-'*15}")

        for gamma_k_str in ZETA_ZEROS[:7]:
            gamma_k_f = float(gamma_k_str)
            in_band = "YES" if gamma_k_f * float(L) / (2 * 3.14159) < N - 1 else "no"

            FT, M_val = fourier_bump_mp(xi_mp, L, N, gamma_k_str)

            abs_FT = fabs(FT)
            abs_M = fabs(M_val)
            abs_eps = fabs(eps)

            ratio_FT = abs_FT / abs_eps
            ratio_M = abs_M / abs_eps

            print(f"  {gamma_k_f:10.4f} {in_band:>8} {nstr(abs_FT, 12):>20} {nstr(abs_M, 12):>20} "
                  f"{nstr(ratio_FT, 8):>15} {nstr(ratio_M, 8):>15}")

        print()

        # Check: what is the actual magnitude of xi_hat at gamma_1?
        FT_1, M_1 = fourier_bump_mp(xi_mp, L, N, ZETA_ZEROS[0])
        print(f"  DETAILED for gamma_1 = {ZETA_ZEROS[0][:10]}:")
        print(f"    F_T (raw value)  = {nstr(FT_1, 30)}")
        print(f"    |F_T|            = {nstr(fabs(FT_1), 20)}")
        print(f"    |eps_0|          = {nstr(fabs(eps), 20)}")
        print(f"    |F_T|/|eps_0|    = {nstr(fabs(FT_1)/fabs(eps), 15)}")
        print(f"    log10(|F_T|)     = {float(mpmath.log10(fabs(FT_1))):.2f}")
        print(f"    log10(|eps_0|)   = {float(mpmath.log10(fabs(eps))):.2f}")
        print()
        dt = time.time() - t0
        print(f"  Time: {dt:.0f}s")
        print()
        print("=" * 90)
        print()

    print("INTERPRETATION:")
    print("  F_T/eps ~ O(1):    T-inner product controls xi at zeros -> proof path OPEN")
    print("  F_T/eps ~ O(L^k):  polynomial growth -> may still work with sharper bounds")
    print("  F_T/eps ~ O(e^cL): exponential growth -> gap is intrinsic, different approach needed")
