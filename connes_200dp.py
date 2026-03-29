"""
Session 20e: Connes spectral operator — 200-digit precision.

Following the paper's recipe exactly:
  1. Compute b_n from exact Prop 4.2 (2F1+digamma) — off-diagonal
  2. Compute a_n from eq (4.4) numerical integral at 200dp — diagonal
  3. Build tau matrix: tau_{i,j} = (b_i-b_j)/(i-j), tau_{i,i} = a_i
  4. Find smallest eigenvalue/eigenvector via mpmath.eig
  5. Compute xi_hat zeros -> should match zeta zeros
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    sinh, exp, cos, sin, hyp2f1, digamma, eig)

mp.dps = 200  # 200-digit precision as in the paper


def compute_b_n(n, L):
    """
    b_n for the FULL QW = W_{0,2} - W_R - sum W_p.

    Off-diagonal: tau_{n,m} = (b_n - b_m)/(n-m)

    b_n^{QW} = b_n^{W02} - b_n^{WR} - b_n^{Wp}

    From the structure: b_n = value such that tau_{n,0} = (b_n - b_0)/n = QW(V_n, V_0)/1
    For n != 0: b_n = n * tau_{n,0} + b_0

    But it's easier to compute b_n from each component's alpha:
    b_n^{WR} = -alpha_L^{WR}(n)  [from Prop 4.3]
    b_n^{Wp} = -alpha_L^{Wp}(n)
    b_n^{W02} = -alpha_L^{W02}(n)

    where alpha_L^{X}(n) = (1/pi) * integral sin(2*pi*n*x/L) * kernel_X(x) dx
    """
    # W_R contribution to b_n: b_n^{WR} = -alpha_L(n) from Prop 4.2 eq (4.5)
    if n == 0:
        alpha_WR = mpf(0)
    else:
        z = exp(-2 * L)
        a_arg = pi * mpc(0, n) / L + mpf(1) / 4
        h = hyp2f1(1, a_arg, a_arg + 1, z)
        f1 = exp(-L / 2) * (2 * L / (L + 4 * pi * mpc(0, n)) * h).imag
        d = digamma(a_arg).imag / 2
        alpha_WR = (f1 + d) / pi

    # W_p contribution to b_n
    # alpha_L^{Wp}(n) = (1/pi) * sum_{k} Lambda(k) * k^{-1/2} * sin(2*pi*n*log(k)/L)
    # = -(1/pi) * sum contribution (since b = -alpha)
    alpha_Wp = mpf(0)
    for p in [2, 3, 5, 7, 11, 13]:
        lp = log(mpf(p))
        pk = mpf(p)
        while pk <= exp(L):
            alpha_Wp += lp * pk**(-mpf(1) / 2) * sin(2 * pi * n * log(pk) / L)
            pk *= p
    alpha_Wp /= pi

    # W_{0,2} contribution to b_n
    # From (4.2): W02(n,m) = 32*L*sinh^2(L/4)*(L^2-16*pi^2*m*n)/[denom]
    # For n != 0, m = 0: W02(n,0) = 32*L*sinh^2(L/4)*L^2/[L^2*(L^2+16*pi^2*n^2)]
    # = 32*sinh^2(L/4)/(L^2+16*pi^2*n^2)
    # tau_{n,0}^{W02} = (b_n^{W02} - b_0^{W02})/n
    # b_0^{W02} = 0 (by antisymmetry b_{-j} = -b_j, so b_0 = 0)
    # So b_n^{W02} = n * W02(n, 0)

    # Actually, b_n is defined from the general structure. Let me use a different approach.
    # From the off-diagonal: tau_{n,m} = (b_n-b_m)/(n-m)
    # For m=0: tau_{n,0} = b_n/n (since b_0=0)
    # So b_n = n * tau_{n,0} = n * QW(V_n, V_0)
    # = n * [W02(n,0) - WR(n,0) - Wp(n,0)]

    # W02(n,0): from (4.2)
    L2 = L * L
    p2 = 16 * pi * pi
    pf = 32 * L * sinh(L / 4)**2
    W02_n0 = pf * L2 / (L2 * (L2 + p2 * n * n))  # simplified for m=0

    # WR(n,0): from Prop 4.3 off-diagonal
    # = (alpha_L^{WR}(0) - alpha_L^{WR}(n)) / (n - 0) = -alpha_WR / n
    WR_n0 = -alpha_WR / n if n != 0 else mpf(0)

    # Wp(n,0): sum Lambda(k)*k^{-1/2}*cos(2*pi*n*log(k)/L)
    Wp_n0 = mpf(0)
    for p in [2, 3, 5, 7, 11, 13]:
        lp = log(mpf(p))
        pk = mpf(p)
        while pk <= exp(L):
            Wp_n0 += lp * pk**(-mpf(1) / 2) * cos(2 * pi * n * log(pk) / L)
            pk *= p

    if n == 0:
        return mpf(0)
    return n * (W02_n0 - WR_n0 - Wp_n0)


def compute_a_n(n, L):
    """
    a_n = tau_{n,n} = QW(V_n, V_n) = W02(n,n) - WR(n,n) - Wp(n,n)

    WR diagonal from eq (4.4) at 200dp (confirmed correct by Grok).
    """
    L2 = L * L
    p2 = 16 * pi * pi
    pf = 32 * L * sinh(L / 4)**2

    # W02(n,n)
    W02 = pf * (L2 - p2 * n * n) / ((L2 + p2 * n * n)**2)

    # WR(n,n) from eq (4.4)
    omega_0 = mpf(2)
    eL = exp(L)
    w_const = (omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1)))

    n_quad = 20000  # high quadrature for 200dp
    dx = L / n_quad
    integral = mpf(0)
    for k in range(n_quad):
        x = dx * (k + mpf(1) / 2)
        omega_x = 2 * (1 - x / L) * cos(2 * pi * n * x / L)
        numer = exp(x / 2) * omega_x - omega_0
        denom = exp(x) - exp(-x)
        if abs(denom) > mpf(10)**(-180):
            integral += numer / denom
    integral *= dx
    WR = w_const + integral

    # Wp(n,n)
    Wp = mpf(0)
    for p in [2, 3, 5, 7, 11, 13]:
        lp = log(mpf(p))
        pk = mpf(p)
        while pk <= eL:
            Wp += lp * pk**(-mpf(1) / 2)  # cos(0) = 1 for diagonal
            pk *= p

    return W02 - WR - Wp


if __name__ == "__main__":
    print("=" * 70)
    print("Connes 200dp: Building tau matrix")
    print("=" * 70)

    L = log(mpf(14))
    N = 40  # start small for testing (dim = 81)
    dim = 2 * N + 1

    print(f"  L = {float(L):.6f}, N = {N}, dim = {dim}")
    print(f"  Working precision: {mp.dps} digits")

    # Compute b_n
    print("\nComputing b_n...")
    b = {}
    for n in range(-N, N + 1):
        b[n] = compute_b_n(n, L)
        if abs(n) % 10 == 0:
            print(f"  b[{n:+3d}] = {float(b[n]):+.15f}")

    # Compute a_n (only need n >= 0 since a_{-n} = a_n)
    print("\nComputing a_n (this is slow — 200dp quadrature)...")
    a = {}
    for n in range(N + 1):
        a[n] = compute_a_n(n, L)
        a[-n] = a[n]
        if n % 10 == 0:
            print(f"  a[{n:3d}] = {float(a[n]):+.15f}")

    # Build tau matrix
    print("\nBuilding tau matrix...")
    tau = mpmatrix(dim, dim)
    for i in range(dim):
        ni = i - N
        for j in range(dim):
            nj = j - N
            if ni == nj:
                tau[i, j] = a[ni]
            else:
                tau[i, j] = (b[ni] - b[nj]) / (ni - nj)

    # Eigenvalue computation
    print("Computing eigenvalues (this may take a while at 200dp)...")
    try:
        E, ER, EL = eig(tau, left=True, right=True)
        eigenvalues = sorted([float(e.real) for e in E])
        print(f"  Min eigenvalue: {eigenvalues[0]:+.15f}")
        print(f"  Max eigenvalue: {eigenvalues[-1]:+.15f}")
        print(f"  Positive: {sum(1 for e in eigenvalues if e > 0)}")
        print(f"  Negative: {sum(1 for e in eigenvalues if e < 0)}")

        # Find the eigenvector for the smallest eigenvalue
        min_idx = 0
        min_val = float('inf')
        for i, e in enumerate(E):
            if float(e.real) < min_val:
                min_val = float(e.real)
                min_idx = i

        xi = [float(ER[j, min_idx].real) for j in range(dim)]
        xi = np.array(xi)

        # Check evenness
        even_score = sum(abs(xi[N + k] - xi[N - k]) for k in range(1, N + 1))
        odd_score = sum(abs(xi[N + k] + xi[N - k]) for k in range(1, N + 1))
        print(f"  Min eigenvec: even_score={even_score:.6f}, odd_score={odd_score:.6f}")
        print(f"  -> {'EVEN' if even_score < odd_score else 'ODD'}")

        # Normalize: sum(xi) = sqrt(L)
        L_val = float(L)
        xs = np.sum(xi)
        if abs(xs) > 1e-30:
            xi = xi * np.sqrt(L_val) / xs
        print(f"  xi sum = {np.sum(xi):.6f}")

        # xi_hat zeros
        gammas = np.load("_zeros_500.npy")

        def xi_hat(z):
            s = np.sin(z * L_val / 2)
            if abs(s) < 1e-60: return 0
            t = sum(xi[j + N] / (z - 2 * np.pi * j / L_val)
                    for j in range(-N, N + 1)
                    if abs(z - 2 * np.pi * j / L_val) > 1e-12)
            return 2 * L_val**(-0.5) * s * t

        zr = np.linspace(0.5, 60, 200000)
        roots = []
        prev = xi_hat(zr[0])
        for i in range(1, len(zr)):
            val = xi_hat(zr[i])
            if np.isfinite(val) and np.isfinite(prev) and prev * val < 0 and abs(val) < 1e10:
                lo, hi = zr[i-1], zr[i]
                for _ in range(100):
                    mid = (lo + hi) / 2
                    fm = xi_hat(mid)
                    if np.isfinite(fm) and fm * xi_hat(lo) < 0: hi = mid
                    else: lo = mid
                root = (lo + hi) / 2
                if not any(abs(root - 2*np.pi*j/L_val) < 0.03 for j in range(-N, N+1)):
                    roots.append(root)
            prev = val

        print(f"\n  {len(roots)} zeros of xi_hat:")
        for i in range(min(15, len(roots), len(gammas))):
            r = roots[i]; g = gammas[i]
            print(f"    {i+1:2d}  {r:12.6f}  {g:12.6f}  {r-g:+10.4f}  {abs(r-g)/g*100:6.2f}%")

    except Exception as ex:
        print(f"  Eigenvalue computation failed: {ex}")
        print("  Trying numpy fallback...")
        tau_np = np.array([[float(tau[i, j]) for j in range(dim)] for i in range(dim)])
        eigvals, eigvecs = np.linalg.eigh(tau_np)
        print(f"  Min eigenvalue: {eigvals[0]:+.15f}")
        print(f"  Max eigenvalue: {eigvals[-1]:+.15f}")

    print("\n" + "=" * 70)
