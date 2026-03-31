"""
Session 30 autonomous: Near-null eigenvalue decay rate of M.

The "null space" eigenvalues of M are NOT exactly zero — they're ~10^{-7}.
The DECAY RATE of these near-null eigenvalues determines whether the proof works.

If the (dim-BW) small eigenvalues decay as O(1/N^alpha) for alpha > 0:
  => Q_W on the noise subspace ~ W02 + O(1/N^alpha)
  => W02 has rank 2, so min eigenvalue ~ O(1/N^alpha)
  => eps_0 -> 0 as N -> inf
  => RH

COMPUTE: the near-null eigenvalue spectrum of M at different N and lambda.
Determine the decay rate.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh)
import time

mp.dps = 50


def build_M(lam_sq, N_val, n_quad=10000):
    """Build M = WR+Wp."""
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

    M_mat = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        M_mat[i,i] = wr_diag[n]
        for j in range(dim):
            m = j - N_val
            if n != m: M_mat[i,j] += (alpha[m]-alpha[n])/(n-m)
            M_mat[i,j] += sum(lk*k**(-0.5)*q_func(n,m,logk) for k,lk,logk in vM)
    M_mat = (M_mat + M_mat.T)/2
    return M_mat


if __name__ == "__main__":
    print("NEAR-NULL EIGENVALUE DECAY RATE")
    print("=" * 70)

    # ================================================================
    # PART 1: Near-null eigenvalues at different N (fixed lambda)
    # ================================================================
    print("\nPART 1: NEAR-NULL EIGENVALUES vs N (lam^2=200)")
    print("-" * 70)

    lam_sq = 200

    for N in [20, 25, 30, 35, 40, 45]:
        dim = 2*N+1
        t0 = time.time()
        M_mat = build_M(lam_sq, N)
        evals = np.sort(np.abs(np.linalg.eigvalsh(M_mat)))
        dt = time.time() - t0

        # The near-null eigenvalues (sorted ascending by absolute value)
        print(f"\n  N={N}, dim={dim} ({dt:.0f}s):")
        print(f"    Smallest 5 |eig|: {', '.join(f'{e:.4e}' for e in evals[:5])}")
        print(f"    5th-10th: {', '.join(f'{e:.4e}' for e in evals[4:10])}")
        print(f"    min/max ratio: {evals[0]/evals[-1]:.4e}")

    # ================================================================
    # PART 2: Does the smallest M eigenvalue decrease with N?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: SMALLEST |M eigenvalue| vs N")
    print("-" * 70)

    print(f"{'N':>4} {'dim':>5} {'min|eig|':>12} {'5th|eig|':>12} {'10th|eig|':>12} "
          f"{'N^2*min':>12}")
    print("-" * 60)

    for N in [15, 20, 25, 30, 35, 40, 50]:
        M_mat = build_M(lam_sq, N, n_quad=8000)
        evals = np.sort(np.abs(np.linalg.eigvalsh(M_mat)))
        min_e = evals[0]
        fifth = evals[4]
        tenth = evals[9] if len(evals) > 9 else 0

        print(f"{N:>4} {2*N+1:>5} {min_e:>12.4e} {fifth:>12.4e} {tenth:>12.4e} "
              f"{N**2*min_e:>12.4e}")

    # ================================================================
    # PART 3: Near-null eigenvalues at different lambda (fixed N scaling)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: NEAR-NULL EIGENVALUES vs LAMBDA (N=8*L)")
    print("-" * 70)

    print(f"{'lam^2':>6} {'N':>4} {'min|eig|':>12} {'5th|eig|':>12} {'N*min':>12} {'N^2*min':>12}")
    print("-" * 65)

    for lam_sq in [50, 100, 200, 500, 1000, 2000]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        M_mat = build_M(lam_sq, N, n_quad=8000)
        evals = np.sort(np.abs(np.linalg.eigvalsh(M_mat)))

        print(f"{lam_sq:>6} {N:>4} {evals[0]:>12.4e} {evals[4]:>12.4e} "
              f"{N*evals[0]:>12.4e} {N**2*evals[0]:>12.4e}")

    # ================================================================
    # PART 4: The near-null spectrum shape — is it universal?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: NEAR-NULL SPECTRUM SHAPE")
    print("-" * 70)

    # Do the near-null eigenvalues follow a specific pattern?
    # Like mu_k ~ k^alpha / N^beta for k = 1, 2, 3, ...?

    for lam_sq in [200, 1000]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N+1
        M_mat = build_M(lam_sq, N)
        evals_abs = np.sort(np.abs(np.linalg.eigvalsh(M_mat)))

        # The near-null eigenvalues (bottom dim-26)
        null_evals = evals_abs[:dim-26]
        ks = np.arange(1, len(null_evals)+1)

        print(f"\n  lam^2={lam_sq}, N={N}, {len(null_evals)} near-null eigenvalues:")
        print(f"    k=1: {null_evals[0]:.4e}")
        print(f"    k=5: {null_evals[4]:.4e}")
        print(f"    k=10: {null_evals[9]:.4e}")
        print(f"    k=20: {null_evals[19]:.4e}")
        if len(null_evals) > 39:
            print(f"    k=40: {null_evals[39]:.4e}")

        # Fit: mu_k ~ C * k^alpha
        log_k = np.log(ks[2:30])
        log_mu = np.log(null_evals[2:30])
        alpha_fit, log_C = np.polyfit(log_k, log_mu, 1)
        print(f"    Fit mu_k ~ {np.exp(log_C):.4e} * k^{alpha_fit:.4f}")
        print(f"    So near-null eigenvalues grow as k^{alpha_fit:.2f}")

        # Normalized: mu_k / (k/N)
        print(f"    mu_k * N / k (should be constant if mu ~ k/N):")
        for k_idx in [0, 4, 9, 19]:
            if k_idx < len(null_evals):
                print(f"      k={k_idx+1}: mu*N/k = {null_evals[k_idx]*N/(k_idx+1):.4e}")

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print("=" * 70)
    print("""
KEY QUESTION: Do the near-null eigenvalues of M decrease with N?

If min|mu| ~ 1/N^alpha (alpha > 0):
  The "noise floor" decreases, giving the optimization more room.
  eps_0 can decrease as 1/N^alpha or faster.

If min|mu| is CONSTANT (independent of N):
  The noise floor is fixed. The optimization improves only because
  there are MORE near-null directions, not because each one gets smaller.
  eps_0 decreases because sum of many small contributions.

Either way: the growing null space dimension (dim - 26) provides
the optimization room that drives eps_0 -> 0.
""")
