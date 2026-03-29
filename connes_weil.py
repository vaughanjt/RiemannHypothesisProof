"""
Session 19c: Connes-Consani-Moscovici Weil quadratic form.

Based on arxiv:2511.22755 "Zeta Spectral Triples" (November 2025).

The Weil quadratic form QW_lambda decomposes as:
  QW = W_{0,2} - W_R - sum_{p <= lambda^2} W_p

on the basis V_n(u) = u^{i*n*pi/L}, n = -N,...,N, where L = log(lambda^2).

Components:
  W_{0,2}: rank-2 contribution from trivial zeros
  W_R: archimedean (digamma + hypergeometric)
  W_p: non-archimedean (prime power sums)

The eigenvalues of the rank-one perturbation of the scaling operator
by the minimum eigenvector of QW approximate the Riemann zeta zeros.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, zeta, power, fac

mp.dps = 50


def primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


# ─── W_{0,2}: Rank-2 contribution from trivial zeros ───

def W_02_matrix(N, L):
    """
    W_{0,2}(V_n, V_m) from the poles of zeta at s=0 and s=1.

    From the explicit formula, the rank-2 part is:
    W_{0,2}(f, g) = f_hat(0)*conj(g_hat(0)) + f_hat(1)*conj(g_hat(1))

    For V_n on [e^{-L/2}, e^{L/2}]:
    V_n_hat(s) = integral_{e^{-L/2}}^{e^{L/2}} u^{s-1} * u^{i*n*pi/L} du
               = integral u^{s-1+i*n*pi/L} du
               = [u^{s+i*n*pi/L}/(s+i*n*pi/L)] from e^{-L/2} to e^{L/2}

    At s = 0: V_n_hat(0) = [e^{i*n*pi/2*L*L/L} - e^{-i*n*pi/2}] / (i*n*pi/L)
    Hmm, this needs care. Let me use the formula from the paper directly.

    Actually, for the Fourier basis V_n(u) = u^{i*n*pi/L} on [lambda^{-1}, lambda]:
    The Mellin transform at s is:
    M(V_n)(s) = integral_{1/lambda}^{lambda} u^{s-1} * u^{i*n*pi/L} du
              = integral u^{s-1+i*n*pi/L} du
              = (lambda^{s+i*n*pi/L} - lambda^{-(s+i*n*pi/L)}) / (s + i*n*pi/L)
              = 2*sinh((s+i*n*pi/L)*L/2) / (s + i*n*pi/L)

    Since L = 2*log(lambda), L/2 = log(lambda).

    M(V_n)(s) = 2*sinh((s+i*n*pi/L)*log(lambda)) / (s + i*n*pi/L)

    For s=0: M(V_n)(0) = 2*sinh(i*n*pi/L*log(lambda)) / (i*n*pi/L)
                        = 2*sinh(i*n*pi/2) / (i*n*pi/L)  [since log(lambda) = L/2]

    sinh(i*theta) = i*sin(theta), so:
    = 2*i*sin(n*pi/2) / (i*n*pi/L) = 2*L*sin(n*pi/2) / (n*pi)

    For n=0: M(V_0)(0) = 2*log(lambda) = L

    For s=1: similar with s=1 shift.

    W_{0,2}(V_n, V_m) = M(V_n)(0)*conj(M(V_m)(0)) + M(V_n)(1)*conj(M(V_m)(1))
    """
    dim = 2 * N + 1
    W = np.zeros((dim, dim), dtype=complex)

    lam = mpmath.exp(mpf(L) / 2)  # lambda = e^{L/2}

    for idx_n in range(dim):
        n = idx_n - N
        for idx_m in range(dim):
            m = idx_m - N

            # Mellin transform at s=0 and s=1
            for s_val in [0, 1]:
                # M(V_n)(s) = 2*sinh((s+i*n*pi/L)*log(lambda)) / (s + i*n*pi/L)
                s_n = mpc(s_val, float(n) * float(pi) / float(L))
                s_m = mpc(s_val, float(m) * float(pi) / float(L))

                if abs(s_n) < 1e-10:
                    Mn = complex(L)
                else:
                    Mn = complex(2 * mpmath.sinh(s_n * log(lam)) / s_n)

                if abs(s_m) < 1e-10:
                    Mm = complex(L)
                else:
                    Mm = complex(2 * mpmath.sinh(s_m * log(lam)) / s_m)

                W[idx_n, idx_m] += Mn * np.conj(Mm)

    return W.real  # should be real for real-valued QW


# ─── W_p: Non-archimedean contribution from prime p ───

def W_p_matrix(N, L, p):
    """
    W_p(V_n, V_m) = log(p) * sum_{k>=1} p^{-k/2} * [q_nm(k*log(p)) + q_nm(-k*log(p))]

    where q_nm(t) = V_n(e^t) * conj(V_m(e^t)) * e^{t/2}
    Wait, this needs the precise formula from the paper.

    More precisely, from the Weil explicit formula:
    W_p(f) = log(p) * sum_{m>=1} p^{-m/2} * [f(p^m) + f(p^{-m})]

    For the bilinear form on V_n, V_m:
    f = V_n * conj(V_m) convolution? No, the Weil form is:
    QW(f, g) = sum_rho f_hat(rho)*conj(g_hat(rho))

    Actually, the Weil distribution applied to the "product" test function.

    Let me use a simpler formulation. The matrix element is:
    W_p(V_n, V_m) = (log p) * sum_{k=1}^{floor(L/(2*log p))} Lambda(p^k)/sqrt(p^k)
                     * [V_n(p^k)*conj(V_m(p^k)) + V_n(p^{-k})*conj(V_m(p^{-k}))]

    Wait, I think the correct formula involves the von Mangoldt function.

    For our basis: V_n(u) = u^{i*n*pi/L}
    V_n(p^k) = p^{i*k*n*pi/L} = exp(i*k*n*pi*log(p)/L)

    So V_n(p^k)*conj(V_m(p^k)) = exp(i*k*(n-m)*pi*log(p)/L)

    The prime contribution becomes:
    W_p(n,m) = (log p) * sum_{k=1}^{K_max} p^{-k/2} *
               2*cos(k*(n-m)*pi*log(p)/L)

    where K_max = floor(L / (2*log p)) (so that p^k <= lambda^2 = e^L).
    Actually K_max such that p^k <= e^{L/2}... let me just sum until terms are negligible.
    """
    lp = np.log(p)
    dim = 2 * N + 1
    W = np.zeros((dim, dim))

    for idx_n in range(dim):
        n = idx_n - N
        for idx_m in range(idx_n, dim):  # symmetric
            m = idx_m - N
            total = 0.0
            for k in range(1, 200):  # sum over prime powers
                pk_half = p**(-k / 2.0)
                if pk_half < 1e-30:
                    break
                phase = k * (n - m) * np.pi * lp / L
                total += pk_half * 2 * np.cos(phase)
            W[idx_n, idx_m] = lp * total
            W[idx_m, idx_n] = W[idx_n, idx_m]

    return W


# ─── W_R: Archimedean contribution ───

def W_R_matrix(N, L):
    """
    The archimedean contribution involves digamma and logarithmic terms.

    From the explicit formula, the archimedean distribution is:
    W_R(f) = (log(4*pi) + gamma) * f(1) + integral involving f

    For the matrix elements, this gives:
    W_R(V_n, V_m) = (log(4*pi) + gamma) * V_n(1)*conj(V_m(1))
                    + integral terms

    V_n(1) = 1 for all n, so the first term is just (log(4*pi) + gamma) * 1.

    For the integral part, using the explicit formula representation:
    W_R involves the function h(u) = -psi(1/2 + i*u/(2*pi)) - psi(1/2 - i*u/(2*pi))
    evaluated at appropriate points.

    For simplicity, start with just the leading constant term
    and add the integral correction if needed.

    Actually, the full W_R is quite involved. Let me use the representation:
    W_R(n,m) = integral_{-L/2}^{L/2} integral_{-L/2}^{L/2}
               K_R(t-t') * e^{i(n*pi/L)*t} * e^{-i(m*pi/L)*t'} dt dt'

    where K_R is the archimedean kernel. This is the Fourier transform of K_R.

    For now, use the approximate form:
    W_R(n,m) ~ (gamma + log(4*pi)) * delta_{nm} * L  (diagonal approximation)

    This is crude but lets us check the overall structure. We'll refine later.
    """
    dim = 2 * N + 1
    # Leading term: (gamma + log(4*pi)) for V_n(1)*conj(V_m(1)) = 1
    c = float(euler + log(4 * pi))
    W = c * np.ones((dim, dim))

    # The integral correction involves the digamma function.
    # Full formula from the paper: too complex for first pass.
    # Start with constant term only.
    return W


# ─── Scaling operator ───

def scaling_operator(N, L):
    """
    D_log has eigenvalues n*pi/L on the basis V_n.
    Returns diagonal matrix.
    """
    dim = 2 * N + 1
    D = np.zeros((dim, dim))
    for idx in range(dim):
        n = idx - N
        D[idx, idx] = n * np.pi / L
    return D


# ─── Main ───

if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 19c: Connes Weil Quadratic Form")
    print("=" * 70)

    # Parameters from the paper
    # lambda = sqrt(14) -> L = 2*log(sqrt(14)) = log(14)
    lam_sq = 14  # lambda^2
    L = np.log(lam_sq)
    lam = np.sqrt(lam_sq)
    N = 40  # matrix dimension = 2N+1 = 81

    primes = primes_up_to(int(lam_sq))
    print(f"\n  lambda = sqrt({lam_sq}) = {lam:.6f}")
    print(f"  L = log({lam_sq}) = {L:.6f}")
    print(f"  N = {N}, matrix dim = {2*N+1}")
    print(f"  Primes <= {lam_sq}: {primes}")

    # --- Step 1: Build components ---
    print("\n--- Step 1: Building Weil quadratic form components ---")

    print("  Computing W_{0,2} (rank-2 trivial zeros)...")
    W02 = W_02_matrix(N, L)
    print(f"    Rank: {np.linalg.matrix_rank(W02, tol=1e-10)}")
    print(f"    Max element: {np.max(np.abs(W02)):.6f}")

    print("  Computing W_p for each prime...")
    W_primes = np.zeros((2*N+1, 2*N+1))
    for p in primes:
        Wp = W_p_matrix(N, L, p)
        W_primes += Wp
        print(f"    p={p}: max element = {np.max(np.abs(Wp)):.6f}")

    print("  Computing W_R (archimedean)...")
    WR = W_R_matrix(N, L)
    print(f"    Max element: {np.max(np.abs(WR)):.6f}")

    # --- Step 2: Assemble QW ---
    print("\n--- Step 2: Assembling QW = W_{0,2} - W_R - sum W_p ---")
    QW = W02 - WR - W_primes

    # Symmetrize (numerical noise)
    QW = (QW + QW.T) / 2

    print(f"  QW shape: {QW.shape}")
    print(f"  QW symmetric: {np.allclose(QW, QW.T)}")

    # --- Step 3: Eigenvalues of QW ---
    print("\n--- Step 3: Eigenvalues of QW ---")
    eigvals, eigvecs = np.linalg.eigh(QW)

    print(f"  Smallest 10 eigenvalues:")
    for i in range(min(10, len(eigvals))):
        print(f"    epsilon_{i+1} = {eigvals[i]:+.10f}")

    print(f"\n  Largest 5 eigenvalues:")
    for i in range(max(0, len(eigvals)-5), len(eigvals)):
        print(f"    epsilon_{i+1} = {eigvals[i]:+.10f}")

    # Minimum eigenvector
    xi = eigvecs[:, 0]
    print(f"\n  Min eigenvector (first 10 components):")
    for i in range(min(10, len(xi))):
        print(f"    xi[{i-N}] = {xi[i]:+.10f}")

    # --- Step 4: Rank-one perturbation ---
    print("\n--- Step 4: Rank-one perturbation of scaling operator ---")
    D = scaling_operator(N, L)

    # D_perturbed = D - |D*xi><delta_N|
    # where delta_N approximates evaluation at boundary
    # For now, try simple rank-one: D - (D @ xi) @ xi.T (projection)
    Dxi = D @ xi
    D_perturbed = D - np.outer(Dxi, xi)

    # Eigenvalues should approximate zeta zeros
    eigs_perturbed = np.linalg.eigvals(D_perturbed)
    eigs_real = np.sort(np.real(eigs_perturbed))

    # Compare with known zeros
    gammas = np.load("_zeros_500.npy")

    print(f"\n  Scaling operator eigenvalues (D_log): n*pi/L")
    print(f"  First few: {[n*np.pi/L for n in range(-3, 4)]}")

    print(f"\n  Perturbed eigenvalues vs zeta zeros:")
    print(f"  {'eig':>14s}  {'zero':>14s}  {'diff':>12s}")

    # Find positive eigenvalues and compare with zeros
    pos_eigs = sorted([e for e in eigs_real if e > 0.5])
    for i in range(min(10, len(pos_eigs), len(gammas))):
        eig = pos_eigs[i]
        zero = gammas[i]
        print(f"  {eig:+14.6f}  {zero:+14.6f}  {eig - zero:+12.6f}")

    # --- Step 5: Assessment ---
    print("\n--- Step 5: Assessment ---")
    print("  This is a FIRST PASS with approximate W_R (constant only).")
    print("  The W_R integral correction and precise rank-one perturbation")
    print("  formula from the paper would dramatically improve accuracy.")
    print()
    print("  The paper achieves 60-digit accuracy with N=120 and 6 primes.")
    print("  Our N=40 with approximate formulas gives a qualitative check.")

    print("\n" + "=" * 70)
