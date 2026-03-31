"""
Session 30 autonomous: Verify rank(M) at larger lambda + identify the 26 directions.

HYPOTHESIS: rank(M) ~ 2*(zeta zeros in bandwidth) + constant.
With N = 8*L, bandwidth = pi*N/L = 8*pi ~ 25.1 (CONSTANT).
Number of zeros with |gamma| < 25: about 7-8 (gamma_1=14.13, ..., gamma_8=30.42).
So rank(M) ~ 2*8 + 10 = 26. If bandwidth fixed, rank fixed.

TEST 1: Verify rank(M) at lam^2=2000,5000 (larger lambda, same bandwidth ~25).
TEST 2: Vary bandwidth by changing C in N=C*L. Does rank(M) change?
TEST 3: Project M's non-null eigenvectors onto zeta-zero bump functions.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time

mp.dps = 50


def build_M_fast(lam_sq, N_val, n_quad=8000):
    """Build M = WR+Wp quickly."""
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

    M = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        M[i,i] = wr_diag[n]
        for j in range(dim):
            m = j - N_val
            if n != m: M[i,j] += (alpha[m]-alpha[n])/(n-m)
            M[i,j] += sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
    M = (M + M.T)/2
    return M, L_f


if __name__ == "__main__":
    print("RANK(M) VERIFICATION AND IDENTIFICATION")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # ================================================================
    # TEST 1: rank(M) at larger lambda with N=8*L (bandwidth ~25)
    # ================================================================
    print("\nTEST 1: rank(M) at larger lambda (fixed bandwidth ~25)")
    print("-" * 70)

    print(f"{'lam^2':>7} {'N':>4} {'dim':>5} {'BW':>6} {'rank_M':>7} "
          f"{'null_M':>7} {'null%':>6} {'zeros<BW':>8}")
    print("-" * 60)

    for lam_sq in [50, 200, 500, 1000, 2000, 5000]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N+1
        bw = np.pi * N / L_f

        t0 = time.time()
        M, _ = build_M_fast(lam_sq, N)
        _, sv_m, _ = np.linalg.svd(M)
        max_sv = sv_m[0]
        rank_m = np.sum(sv_m > max_sv * 1e-4)
        null_m = dim - rank_m

        # Count zeta zeros in bandwidth
        n_zeros = np.sum(gammas < bw)

        print(f"{lam_sq:>7} {N:>4} {dim:>5} {bw:>6.1f} {rank_m:>7} "
              f"{null_m:>7} {100*null_m/dim:>5.0f}% {n_zeros:>8}  ({time.time()-t0:.0f}s)")

    # ================================================================
    # TEST 2: Vary bandwidth at fixed lambda — does rank change?
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 2: VARY BANDWIDTH (different C in N=C*L) at lam^2=200")
    print("-" * 70)

    lam_sq = 200
    L_f = np.log(lam_sq)

    print(f"{'C':>4} {'N':>4} {'dim':>5} {'BW':>6} {'rank_M':>7} "
          f"{'null_M':>7} {'zeros<BW':>8}")
    print("-" * 50)

    for C_val in [4, 6, 8, 10, 12, 15]:
        N = round(C_val * L_f)
        dim = 2*N+1
        bw = np.pi * N / L_f

        M, _ = build_M_fast(lam_sq, N)
        _, sv_m, _ = np.linalg.svd(M)
        rank_m = np.sum(sv_m > sv_m[0] * 1e-4)
        null_m = dim - rank_m
        n_zeros = np.sum(gammas < bw)

        print(f"{C_val:>4} {N:>4} {dim:>5} {bw:>6.1f} {rank_m:>7} "
              f"{null_m:>7} {n_zeros:>8}")

    # ================================================================
    # TEST 3: Does rank(M) = 2*(zeros in BW) + constant?
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 3: rank(M) vs 2*(zeros in bandwidth)")
    print("-" * 70)

    print(f"{'lam^2':>7} {'C':>3} {'BW':>6} {'zeros':>6} {'2*z':>5} "
          f"{'rank_M':>7} {'rank-2z':>8}")
    print("-" * 50)

    for lam_sq, C_val in [(200,4), (200,6), (200,8), (200,10), (200,12),
                           (50,8), (500,8), (1000,8), (2000,8)]:
        L_f = np.log(lam_sq)
        N = round(C_val * L_f)
        dim = 2*N+1
        bw = np.pi * N / L_f

        M, _ = build_M_fast(lam_sq, N)
        _, sv_m, _ = np.linalg.svd(M)
        rank_m = np.sum(sv_m > sv_m[0] * 1e-4)
        n_zeros = np.sum(gammas < bw)
        diff = rank_m - 2*n_zeros

        print(f"{lam_sq:>7} {C_val:>3} {bw:>6.1f} {n_zeros:>6} {2*n_zeros:>5} "
              f"{rank_m:>7} {diff:>8}")

    # ================================================================
    # TEST 4: Project non-null M eigenvectors onto zeta zero bumps
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 4: NON-NULL M EIGENVECTORS vs ZETA ZEROS")
    print("-" * 70)

    lam_sq = 200
    L_f = np.log(lam_sq)
    N = round(8 * L_f)
    dim = 2*N+1
    bw = np.pi * N / L_f

    M, _ = build_M_fast(lam_sq, N)
    evals_m, evecs_m = np.linalg.eigh(M)

    # The non-null eigenvectors (|eigenvalue| > 1e-4 * max)
    abs_evals = np.abs(evals_m)
    threshold = np.max(abs_evals) * 1e-4
    nonnull_idx = np.where(abs_evals > threshold)[0]

    # For each nonnull eigenvector, compute its "frequency content"
    # phi_k(n) = evecs_m[n+N, k]. Its Fourier transform peaks at some frequency.
    # Check if the peak frequency corresponds to a zeta zero.

    print(f"\nlam^2={lam_sq}, N={N}, {len(nonnull_idx)} non-null eigenvectors")
    print(f"Zeta zeros in BW: {gammas[gammas < bw]}")
    print(f"\n{'k':>4} {'mu_k':>10} {'peak_freq':>10} {'nearest_zero':>12} {'dist':>8} {'parity':>7}")
    print("-" * 55)

    for k in nonnull_idx:
        ev = evecs_m[:, k]
        mu_k = evals_m[k]

        # DFT to find peak frequency
        center = N
        # The "frequency" of mode n is 2*pi*n/L
        # Compute power spectrum
        fft_ev = np.abs(np.fft.fft(ev))
        peak_idx = np.argmax(fft_ev[1:dim//2]) + 1  # skip DC
        peak_freq = 2*np.pi*peak_idx / dim * N  # approximate frequency mapping

        # Actually, better: compute overlap with cos/sin at each frequency
        freqs = 2*np.pi*np.arange(0, N+1) / L_f
        max_overlap = 0
        best_freq = 0
        for f in freqs:
            cos_vec = np.array([np.cos(f * (n)) for n in range(-N, N+1)])
            sin_vec = np.array([np.sin(f * (n)) for n in range(-N, N+1)])
            cos_vec /= np.linalg.norm(cos_vec) + 1e-30
            sin_vec /= np.linalg.norm(sin_vec) + 1e-30
            overlap = np.dot(ev, cos_vec)**2 + np.dot(ev, sin_vec)**2
            if overlap > max_overlap:
                max_overlap = overlap
                best_freq = f

        # Nearest zeta zero
        if len(gammas) > 0:
            nearest_idx = np.argmin(np.abs(gammas - best_freq))
            nearest_zero = gammas[nearest_idx]
            dist = abs(best_freq - nearest_zero)
        else:
            nearest_zero = 0
            dist = 999

        # Parity
        even_s = sum(abs(ev[center+j] - ev[center-j]) for j in range(1, N+1))
        odd_s = sum(abs(ev[center+j] + ev[center-j]) for j in range(1, N+1))
        parity = "EVEN" if even_s < odd_s else "ODD"

        print(f"{k:>4} {mu_k:>10.4f} {best_freq:>10.4f} {nearest_zero:>12.4f} "
              f"{dist:>8.4f} {parity:>7}")

    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print("=" * 70)
