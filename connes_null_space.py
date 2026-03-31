"""
Session 30f: WHY does M = WR+Wp have a large near-null space?

From complement probe: ~80% of M's eigenvalues are near 0.
xi_0 puts ~50% of its weight there. This drives eps_0 -> 0.

HYPOTHESES:
A) alpha_L(n) -> 1/4 means the Cauchy-like WR matrix becomes "flat"
   (all entries approach 0), so WR -> 0 in some sense. Combined with
   Wp (finite rank from finitely many primes), M has low effective rank.

B) M = WR + Wp where WR is Cauchy-like and Wp is a finite sum.
   The number of primes <= lam^2 is pi(lam^2) ~ lam^2/log(lam^2).
   But Wp has rank at most 2*|primes| (real + imaginary parts).
   So Wp has rank O(lam^2/log(lam^2)).
   With dim = 2N+1 ~ 16*log(lam^2), if rank(Wp) >> dim, Wp could be full rank.
   But at lam^2=50: rank(Wp) <= 2*15 = 30, dim = 63. So Wp has rank ~ dim/2.
   The NULL SPACE of Wp has dimension ~ dim/2 = N.

C) WR is Cauchy-like with entries (alpha(m)-alpha(n))/(n-m).
   Since alpha(n) -> 1/4, the entries -> 0 for large n,m.
   So WR is "approximately low rank" (its numerical rank is small).

PLAN: Compute the eigenvalue distribution of WR, Wp, and M separately.
Measure their effective ranks. See if the null space grows with N.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time

mp.dps = 50


def build_components_fast(lam_sq, N_val, n_quad=10000):
    """Build W02, WR, Wp separately with moderate precision."""
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2*N_val + 1

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
            return (np.sin(2*np.pi*m*y/L_f) - np.sin(2*np.pi*n*y/L_f)) / (np.pi*(n-m))
        else:
            return 2*(L_f - y)/L_f * np.cos(2*np.pi*n*y/L_f)

    L2_f = L_f**2; p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    W02 = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(dim):
            m = j - N_val
            W02[i,j] = pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))

    alpha = {}
    for n in range(-N_val, N_val+1):
        if n == 0: alpha[n] = 0.0
        else:
            z = exp(-2*L)
            a = pi*mpc(0,abs(n))/L + mpf(1)/4
            h = hyp2f1(1,a,a+1,z)
            f1 = exp(-L/2) * (2*L/(L+4*pi*mpc(0,abs(n)))*h).imag
            d = digamma(a).imag/2
            val = float((f1+d)/pi)
            alpha[n] = val if n>0 else -val

    wr_diag = {}
    omega_0 = mpf(2)
    for nv in range(N_val+1):
        def omega(x, nv=nv):
            return 2*(1-x/L)*cos(2*pi*nv*x/L)
        w_const = (omega_0/2)*(euler+log(4*pi*(eL-1)/(eL+1)))
        dx = L/n_quad; integral = mpf(0)
        for k in range(n_quad):
            x = dx*(k+mpf(1)/2)
            numer = exp(x/2)*omega(x)-omega_0
            denom = exp(x)-exp(-x)
            if abs(denom) > mpf(10)**(-40): integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const+integral)
        wr_diag[-nv] = wr_diag[nv]

    WR = np.zeros((dim, dim))
    Wp = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        WR[i,i] = wr_diag[n]
        for j in range(dim):
            m = j - N_val
            if n != m: WR[i,j] = (alpha[m]-alpha[n])/(n-m)
            Wp[i,j] = sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
    WR = (WR + WR.T)/2
    Wp = (Wp + Wp.T)/2
    M = WR + Wp
    return W02, WR, Wp, M


if __name__ == "__main__":
    print("M's NEAR-NULL SPACE: THE PROOF FRONTIER")
    print("=" * 70)

    # ================================================================
    # PART 1: Eigenvalue distribution of WR, Wp, M
    # ================================================================
    print("\nPART 1: EIGENVALUE DISTRIBUTIONS")
    print("-" * 70)

    for lam_sq in [50, 200, 1000]:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        dim = 2*N + 1
        n_primes = sum(1 for p in range(2, lam_sq+1)
                       if all(p % d != 0 for d in range(2, int(p**0.5)+1)) and p > 1)

        t0 = time.time()
        W02, WR, Wp, M = build_components_fast(lam_sq, N)
        dt = time.time() - t0

        evals_wr = np.linalg.eigvalsh(WR)
        evals_wp = np.linalg.eigvalsh(Wp)
        evals_m = np.linalg.eigvalsh(M)

        # Effective rank: number of eigenvalues > threshold * max
        for name, evals in [("WR", evals_wr), ("Wp", evals_wp), ("M", evals_m)]:
            abs_evals = np.abs(evals)
            max_ev = np.max(abs_evals)
            for thresh in [1e-2, 1e-4, 1e-6, 1e-8]:
                eff_rank = np.sum(abs_evals > thresh * max_ev)
                if thresh == 1e-4:
                    print(f"  lam^2={lam_sq:>5}, {name:>2}: dim={dim}, "
                          f"eff_rank(1e-4)={eff_rank:>3}, "
                          f"null_frac={1-eff_rank/dim:.1%}, "
                          f"range=[{evals[0]:.2e},{evals[-1]:.2e}] ({dt:.0f}s)")

        # Detailed eigenvalue distribution of M
        abs_m = np.sort(np.abs(evals_m))
        print(f"  M eigenvalue magnitudes (lam^2={lam_sq}):")
        print(f"    Bottom 10: {', '.join(f'{v:.2e}' for v in abs_m[:10])}")
        print(f"    Top 5: {', '.join(f'{v:.2e}' for v in abs_m[-5:])}")
        print()

    # ================================================================
    # PART 2: How does the near-null space dimension scale with N?
    # ================================================================
    print(f"{'='*70}")
    print("PART 2: NEAR-NULL SPACE SCALING WITH N")
    print("-" * 70)

    lam_sq = 200  # Fixed lambda, vary N
    print(f"\nlam^2={lam_sq}, varying N:")
    print(f"{'N':>4} {'dim':>5} {'null(1e-4)':>10} {'null(1e-6)':>10} "
          f"{'null(1e-8)':>10} {'null_frac':>10}")
    print("-" * 55)

    for N in [15, 20, 25, 30, 35, 40]:
        _, _, _, M = build_components_fast(lam_sq, N, n_quad=8000)
        evals_m = np.linalg.eigvalsh(M)
        abs_m = np.abs(evals_m)
        max_m = np.max(abs_m)
        dim = 2*N+1

        null_4 = np.sum(abs_m < 1e-4 * max_m)
        null_6 = np.sum(abs_m < 1e-6 * max_m)
        null_8 = np.sum(abs_m < 1e-8 * max_m)
        frac = null_4 / dim

        print(f"{N:>4} {dim:>5} {null_4:>10} {null_6:>10} {null_8:>10} {frac:>10.1%}")

    # ================================================================
    # PART 3: Wp rank analysis
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: Wp RANK STRUCTURE")
    print("-" * 70)

    # Wp_{nm} = sum_k Lambda(k) k^{-1/2} q(n,m,log(k))
    # Each prime power k contributes a rank-2 matrix (from sin and cos terms).
    # So rank(Wp) <= 2 * (number of prime powers <= lam^2).

    for lam_sq in [14, 50, 200, 1000]:
        L_f = np.log(lam_sq)
        N = max(15, round(8 * L_f))
        dim = 2*N+1

        _, _, Wp, _ = build_components_fast(lam_sq, N, n_quad=5000)
        _, sv_wp, _ = np.linalg.svd(Wp)

        # Count prime powers
        n_pp = 0
        for p in range(2, lam_sq+1):
            if all(p % d != 0 for d in range(2, int(p**0.5)+1)) and p > 1:
                pk = p
                while pk <= lam_sq:
                    n_pp += 1
                    pk *= p

        eff_rank = np.sum(sv_wp > sv_wp[0] * 1e-10)

        print(f"  lam^2={lam_sq:>5}: dim={dim:>3}, prime_powers={n_pp:>3}, "
              f"2*pp={2*n_pp:>3}, rank(Wp)={eff_rank:>3}, "
              f"null_dim={dim-eff_rank:>3}")

    # ================================================================
    # PART 4: WR rank analysis — the Cauchy-like structure
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: WR RANK STRUCTURE")
    print("-" * 70)

    # WR has displacement rank 2 (from D*WR - WR*D structure).
    # But its NUMERICAL rank could be much smaller if the generators vanish.
    # alpha_L(n) -> 1/4 means b_n -> 1/4, so b_n - b_m -> 0 for large n,m.
    # This makes WR_{nm} = (alpha(m)-alpha(n))/(n-m) -> 0.

    for lam_sq in [50, 200, 1000]:
        L_f = np.log(lam_sq)
        N = max(15, round(8 * L_f))
        dim = 2*N+1

        _, WR, _, _ = build_components_fast(lam_sq, N, n_quad=5000)
        _, sv_wr, _ = np.linalg.svd(WR)

        for thresh in [1e-2, 1e-4, 1e-6]:
            eff_rank = np.sum(sv_wr > sv_wr[0] * thresh)
            if thresh == 1e-4:
                print(f"  lam^2={lam_sq:>5}: dim={dim:>3}, rank(WR, 1e-4)={eff_rank:>3}, "
                      f"sv_1={sv_wr[0]:.2e}, sv_5={sv_wr[4]:.2e}, "
                      f"sv_10={sv_wr[9]:.2e}")

        # Singular value decay
        print(f"    SV decay: {', '.join(f'{sv_wr[k]/sv_wr[0]:.2e}' for k in [0,1,2,5,10,20])}")

    # ================================================================
    # PART 5: The key — M = WR + Wp null space dimension formula
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: M NULL SPACE AS FUNCTION OF LAMBDA")
    print("-" * 70)

    print(f"\n{'lam^2':>6} {'N':>4} {'dim':>5} {'rank_M':>7} {'null_M':>7} "
          f"{'rank_Wp':>8} {'rank_WR':>8} {'null_frac':>10}")
    print("-" * 65)

    for lam_sq in [14, 30, 50, 100, 200, 500, 1000]:
        L_f = np.log(lam_sq)
        N = max(15, round(8 * L_f))
        dim = 2*N+1

        _, WR, Wp, M = build_components_fast(lam_sq, N, n_quad=5000)

        _, sv_m, _ = np.linalg.svd(M)
        _, sv_wr, _ = np.linalg.svd(WR)
        _, sv_wp, _ = np.linalg.svd(Wp)

        rank_m = np.sum(sv_m > sv_m[0] * 1e-4)
        rank_wr = np.sum(sv_wr > sv_wr[0] * 1e-4)
        rank_wp = np.sum(sv_wp > sv_wp[0] * 1e-4)
        null_m = dim - rank_m
        null_frac = null_m / dim

        print(f"{lam_sq:>6} {N:>4} {dim:>5} {rank_m:>7} {null_m:>7} "
              f"{rank_wp:>8} {rank_wr:>8} {null_frac:>10.1%}")

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print("=" * 70)
