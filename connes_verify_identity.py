"""
Session 30 iteration 14: Verify Z = Q_W + 2*WR (the key identity).

If this holds, then M = W02 + 2*WR - Z, and the rank bound follows
from Slepian cancellation between 2*WR and Z.

Previous iteration 10: Z != Q_W (150% error).
But: Z = Q_W + 2*WR, so Z - 2*WR should = Q_W.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, sinh
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 30


def V_hat(n, s, L_f):
    """Mellin transform of V_n on [lambda^{-1}, lambda]."""
    freq = 2j * np.pi * n / L_f
    arg = freq - complex(s)
    if abs(arg) < 1e-14:
        return complex(L_f)
    return complex(2 * np.sinh(arg * L_f / 2) / arg)


def build_zero_sum(N_val, L_f, gammas, n_zeros):
    """Z_{nm} = sum_rho V_n_hat(rho) * conj(V_m_hat(rho))"""
    dim = 2 * N_val + 1
    Z = np.zeros((dim, dim))
    for k in range(n_zeros):
        rho = 0.5 + 1j * gammas[k]
        v = np.array([V_hat(n, rho, L_f) for n in range(-N_val, N_val + 1)])
        Z += np.real(np.outer(v, np.conj(v)))
    return Z


if __name__ == "__main__":
    print("VERIFY IDENTITY: Z = Q_W + 2*WR")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    for lam_sq in [14, 50]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2 * N + 1

        print(f"\nlam^2={lam_sq}, N={N}, dim={dim}")
        print("-" * 50)

        # Build components
        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N, n_quad=10000)
        WR = M - (M - np.zeros_like(M))  # need WR separately

        # Actually, build WR explicitly by subtracting Wp from M
        # Rebuild Wp
        L = log(mpf(lam_sq))
        L_f_mp = float(L)
        vM = []
        limit = min(lam_sq, 10000)
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5)+2):
            if i <= limit and sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        for p in range(2, limit+1):
            if sieve[p] and p <= lam_sq:
                pk = p
                while pk <= lam_sq:
                    vM.append((pk, np.log(p), np.log(pk)))
                    pk *= p

        def q_func(n, m, y):
            if n != m:
                return (np.sin(2*np.pi*m*y/L_f_mp) - np.sin(2*np.pi*n*y/L_f_mp)) / (np.pi*(n-m))
            else:
                return 2*(L_f_mp - y)/L_f_mp * np.cos(2*np.pi*n*y/L_f_mp)

        Wp = np.zeros((dim, dim))
        for i in range(dim):
            n = i - N
            for j in range(dim):
                m = j - N
                Wp[i,j] = sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
        Wp = (Wp + Wp.T)/2

        WR = M - Wp
        dt = time.time() - t0
        print(f"  Built components ({dt:.0f}s)")

        # Compute Z with increasing zeros
        for n_z in [10, 50, 100, 200, 500]:
            if n_z > len(gammas): break
            Z = build_zero_sum(N, L_f_mp, gammas, n_z)

            # Test 1: Z vs Q_W (should NOT match — large error)
            err_QW = np.linalg.norm(Z - QW) / np.linalg.norm(QW)

            # Test 2: Z vs Q_W + 2*WR (the KEY identity)
            target = QW + 2 * WR
            err_identity = np.linalg.norm(Z - target) / np.linalg.norm(target)

            # Test 3: Z vs W02 + WR - Wp (another way to write it)
            target2 = W02 + WR - Wp
            err_t2 = np.linalg.norm(Z - target2) / np.linalg.norm(target2)

            # Test 4: Z vs W02 - Wp + WR (same as test 3, sanity check)
            # Note: Q_W + 2*WR = (W02 - WR - Wp) + 2*WR = W02 + WR - Wp
            # So tests 2 and 3 should give the same result

            print(f"  Z({n_z:>3} zeros): "
                  f"||Z-QW||/||QW||={err_QW:.4f}, "
                  f"||Z-(QW+2WR)||/||QW+2WR||={err_identity:.4f}, "
                  f"||Z-(W02+WR-Wp)||/||..||={err_t2:.4f}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If ||Z - (Q_W + 2*WR)|| -> 0: the identity is confirmed.
  => M = W02 + 2*WR - Z
  => rank(M) = rank(2*WR - Z + W02)
  => 2*WR and Z cancel beyond bandwidth (Slepian)
  => rank(M) ~ bandwidth

If ||Z - (Q_W + 2*WR)|| stays large: identity is WRONG.
  => The Weil formula has different structure
  => Need to find the correct identity
""")
