"""
Session 30 iteration 15: Exact Weil formula with CORRECT normalizations.

From arXiv:2511.22755:
  V_n(u) = L^{-1/2} exp(2*pi*i*n*log(lambda*u)/L) on [lambda^{-1}, lambda]

  Mellin: f_tilde(s) = int f(x) x^{s-1} dx
  Fourier on R+*: F_hat(s) = int F(u) u^{-is} d*u/u
  Relation: F(x) = x^{1/2} f(x)

  The explicit formula (eq 3.2):
  sum_rho f_tilde(rho) = int f + int f^# - sum_v W_v(f)

  Weil quadratic form (eq 3.10):
  Q_W(f,g) = W_{0,2}(f*g) - W_R(f*g) - sum_p W_p(f*g)

  W_p(F) = log(p) * sum_m p^{-m/2} (F(p^m) + F(p^{-m}))

Now: compute V_n_tilde(rho) with the CORRECT conventions and
rebuild the zero sum.

V_n_tilde(s) = int V_n(u) * u^{s-1} du  (NOT d*u!)
Wait — need to be careful. V_n is defined on [lambda^{-1}, lambda].

Actually, let me compute:
V_n_hat(s) = int_{lambda^{-1}}^{lambda} V_n(u) u^{-is} du/u  (Fourier on R+*)

V_n(u) = L^{-1/2} exp(2*pi*i*n*log(lambda*u)/L)
       = L^{-1/2} (lambda*u)^{2*pi*i*n/L}
       = L^{-1/2} lambda^{2*pi*i*n/L} u^{2*pi*i*n/L}

So V_n_hat(s) = L^{-1/2} lambda^{2*pi*i*n/L} int_{lambda^{-1}}^{lambda} u^{2*pi*i*n/L - is - 1} du

Let alpha = 2*pi*n/L - s. Then:
= L^{-1/2} lambda^{2*pi*i*n/L} * [u^{i*alpha}/(i*alpha)]_{lambda^{-1}}^{lambda}
= L^{-1/2} lambda^{2*pi*i*n/L} * (lambda^{i*alpha} - lambda^{-i*alpha})/(i*alpha)
= L^{-1/2} lambda^{2*pi*i*n/L} * 2*sin(alpha*log(lambda))/(alpha)
= L^{-1/2} lambda^{2*pi*i*n/L} * 2*sin((2*pi*n/L - s)*L/2) / (2*pi*n/L - s)

Hmm wait, alpha = 2*pi*i*n/L - is = i*(2*pi*n/L - s). So:
u^{i*alpha} = u^{i*i*(2*pi*n/L-s)} = u^{-(2*pi*n/L-s)}... this is getting confusing.

Let me just compute numerically for verification.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, sinh, sin, nstr
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 30


def V_n_fourier_hat(n, s, lam, L_f):
    """V_n_hat(s) = int_{lam^{-1}}^{lam} V_n(u) u^{-is} du/u

    V_n(u) = L^{-1/2} (lam*u)^{2*pi*i*n/L}
    """
    L_mp = mpf(L_f)
    lam_mp = mpf(lam)
    s_mp = mpc(s)
    n_freq = 2*pi*n/L_mp

    # Exponent in the integral: u^{i*n_freq - is - 1} du
    # = u^{i*(n_freq - s) - 1} du
    # Integral from lam^{-1} to lam:
    alpha = mpc(0, 1) * (n_freq - s_mp)  # i*(n_freq - s)

    if abs(alpha) < mpf(10)**(-14):
        # Limit: integral du/u = log(lam) - log(1/lam) = 2*log(lam) = L
        result = L_mp
    else:
        # [u^alpha / alpha]_{1/lam}^{lam} = (lam^alpha - lam^{-alpha})/alpha
        result = (mpmath.power(lam_mp, alpha) - mpmath.power(lam_mp, -alpha)) / alpha

    # Multiply by L^{-1/2} * lam^{i*n_freq}
    return complex(result * L_mp**(-mpf(1)/2) * mpmath.power(lam_mp, mpc(0, 1)*n_freq))


def V_n_mellin_tilde(n, s, lam, L_f):
    """V_n_tilde(s) = int_{lam^{-1}}^{lam} V_n(u) u^{s-1} du

    V_n(u) = L^{-1/2} (lam*u)^{2*pi*i*n/L}

    Note: u^{s-1} du, NOT u^{-is} du/u!
    Relation: f_tilde(s) = F_hat(s) if F(u) = u^{1/2} f(u) and s = 1/2+it
    """
    L_mp = mpf(L_f)
    lam_mp = mpf(lam)
    s_mp = mpc(s)
    n_freq = 2*pi*n/L_mp

    # V_n(u) * u^{s-1} = L^{-1/2} (lam*u)^{i*n_freq} * u^{s-1}
    # = L^{-1/2} lam^{i*n_freq} * u^{i*n_freq + s - 1}
    alpha = mpc(0, 1)*n_freq + s_mp  # total exponent + 1 - 1 = i*n_freq + s - 1 + 1

    # Wait: integral u^{i*n_freq + s - 1} du = u^{i*n_freq + s}/(i*n_freq + s)
    beta = mpc(0, 1)*n_freq + s_mp  # = i*n_freq + s

    if abs(beta) < mpf(10)**(-14):
        result = L_mp  # limit
    else:
        result = (mpmath.power(lam_mp, beta) - mpmath.power(lam_mp, -beta)) / beta

    return complex(result * L_mp**(-mpf(1)/2) * mpmath.power(lam_mp, mpc(0, 1)*n_freq))


if __name__ == "__main__":
    print("EXACT WEIL FORMULA — CORRECT NORMALIZATIONS")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    for lam_sq in [14]:
        lam = np.sqrt(lam_sq)
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N + 1

        print(f"\nlam^2={lam_sq}, lam={lam:.4f}, L={L_f:.4f}, N={N}, dim={dim}")

        # Build Q_W standard way
        W02, M, QW = build_all(lam_sq, N, n_quad=10000)

        # Build zero sum with FOURIER hat convention
        print(f"\n  Zero sum with V_hat (Fourier on R+* convention):")
        for n_z in [10, 50, 200]:
            if n_z > len(gammas): break
            Z_fourier = np.zeros((dim, dim))
            for k in range(n_z):
                rho_s = 0.5 + 1j * gammas[k]
                # Use Fourier hat: V_n_hat(s) at s corresponding to rho
                # In the explicit formula: sum_rho f_tilde(rho)
                # With f_tilde(s) convention...
                # Let me try both conventions
                v = np.array([V_n_fourier_hat(n, gammas[k], lam, L_f)
                              for n in range(-N, N+1)])
                Z_fourier += np.real(np.outer(v, np.conj(v)))

            for name, target in [("QW", QW), ("QW+2WR", QW + 2*(M - np.zeros_like(M)))]:
                # Get WR by subtracting Wp from M
                pass
            err = np.linalg.norm(Z_fourier - QW) / np.linalg.norm(QW)
            print(f"    {n_z} zeros (Fourier): ||Z-QW||/||QW|| = {err:.4f}")

        # Build zero sum with MELLIN tilde convention
        print(f"\n  Zero sum with V_tilde (Mellin convention):")
        for n_z in [10, 50, 200]:
            if n_z > len(gammas): break
            Z_mellin = np.zeros((dim, dim))
            for k in range(n_z):
                rho = 0.5 + 1j * gammas[k]
                v = np.array([V_n_mellin_tilde(n, rho, lam, L_f)
                              for n in range(-N, N+1)])
                Z_mellin += np.real(np.outer(v, np.conj(v)))

            err = np.linalg.norm(Z_mellin - QW) / np.linalg.norm(QW)
            print(f"    {n_z} zeros (Mellin): ||Z-QW||/||QW|| = {err:.4f}")

        # Try with different sign/scaling
        print(f"\n  Trying -Z, Z/2, Z*L, etc.:")
        Z = Z_mellin  # use 200 zeros
        for scale_name, scale in [("-Z", -1), ("Z/L", 1/L_f), ("-Z/L", -1/L_f),
                                   ("Z/(2pi)", 1/(2*np.pi)), ("-Z/(2pi)", -1/(2*np.pi)),
                                   ("Z*L/(2pi)", L_f/(2*np.pi))]:
            target = scale * Z
            err = np.linalg.norm(target - QW) / np.linalg.norm(QW)
            if err < 1:
                print(f"    {scale_name}: err = {err:.6f} ***")
            else:
                print(f"    {scale_name}: err = {err:.4f}")

    print(f"\n{'='*70}")
