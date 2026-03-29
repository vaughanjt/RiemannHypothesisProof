"""
Session 21: Prolate spheroidal wave function overlap with Connes eigenvector.

Route 4 from Grok: compute <xi_lambda | k_lambda> for increasing lambda
and check if ||xi - c*k|| = O(lambda^{-2}).

The prolate wave operator: PW_lambda = -d/dx[(lambda^2-x^2)d/dx] + (2*pi*lambda*x)^2
Its even eigenfunctions h_{0,lambda}, h_{4,lambda} on [-lambda, lambda].

k_lambda = E(h_lambda) where h_lambda is the linear combination of h_0 and h_4
with vanishing integral.

For large lambda: h_{n,lambda} -> h_n (Hermite functions).
h_0(x) = pi^{-1/4} exp(-x^2/2)  (Gaussian)
h_4(x) = pi^{-1/4} (4x^4-12x^2+3)/(2*sqrt(6)) * exp(-x^2/2)

Strategy:
1. Compute the QW eigenvector xi at 200dp for several lambda values
2. Compute the prolate functions h_0, h_4 (via their limiting Hermite forms)
3. Form k_lambda and compute the overlap
4. Check convergence rate
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    sinh, exp, cos, sin, hyp2f1, digamma, sqrt, eig)


def hermite_h0(x):
    """h_0(x) = pi^{-1/4} exp(-x^2/2) — zeroth even Hermite function."""
    return float(pi)**(-0.25) * np.exp(-x**2 / 2)


def hermite_h4(x):
    """h_4(x) — 4th Hermite function (even, normalized)."""
    # H_4(x) = 16x^4 - 48x^2 + 12
    # h_4(x) = H_4(x) / sqrt(2^4 * 4!) * pi^{-1/4} * exp(-x^2/2)
    # = (16x^4-48x^2+12) / sqrt(16*24) * pi^{-1/4} * exp(-x^2/2)
    # = (16x^4-48x^2+12) / (4*sqrt(24)) * pi^{-1/4} * exp(-x^2/2)
    norm = np.sqrt(2**4 * 24) # sqrt(384), since 4! = 24
    H4 = 16 * x**4 - 48 * x**2 + 12
    return H4 / norm * float(pi)**(-0.25) * np.exp(-x**2 / 2)


def h_lambda_combination(x, lam):
    """
    h_lambda(x) = linear combination of h_0 and h_4 with vanishing integral.

    integral h_0 dx = sqrt(2*pi) * pi^{-1/4} (Gaussian integral)
    integral h_4 dx = integral of polynomial*Gaussian (computable)

    We need: c_0 * integral(h_0) + c_4 * integral(h_4) = 0
    with c_0^2 + c_4^2 = 1 (normalization).

    For the LIMITING (lambda -> inf) Hermite functions on all of R:
    integral_{-inf}^{inf} h_0(x) dx = (2*pi)^{1/2} * pi^{-1/4}
    integral_{-inf}^{inf} h_4(x) dx = 0 (h_4 is orthogonal to constants in L^2)

    Wait, h_4 has integral: integral H_4(x)*exp(-x^2/2) dx / norm
    H_4(x) = 16x^4 - 48x^2 + 12
    integral (16x^4-48x^2+12)*exp(-x^2/2) dx = 16*3*sqrt(2pi) - 48*sqrt(2pi) + 12*sqrt(2pi)
    = sqrt(2pi)*(48-48+12) = 12*sqrt(2pi)

    So integral h_4 = 12*sqrt(2pi) / sqrt(384) * pi^{-1/4}
    = 12*sqrt(2pi) / (8*sqrt(6)) * pi^{-1/4}

    For the vanishing-integral combination:
    c_0 * I_0 + c_4 * I_4 = 0
    c_0/c_4 = -I_4/I_0 = -12/(sqrt(384)) = -12/(8*sqrt(6)) = -3/(2*sqrt(6))

    Normalize: c_4 = 2*sqrt(6)/sqrt(4*6+9) = 2*sqrt(6)/sqrt(33)
    c_0 = -3/sqrt(33)

    Actually let me just compute: c_0 = -I_4, c_4 = I_0 (unnormalized), then normalize.
    """
    # Integrals of Hermite functions on [-lambda, lambda]
    # For large lambda, these approach the full-line integrals
    n_pts = 10000
    x_grid = np.linspace(-lam, lam, n_pts)
    dx = x_grid[1] - x_grid[0]

    h0_vals = hermite_h0(x_grid)
    h4_vals = hermite_h4(x_grid)

    I0 = np.sum(h0_vals) * dx
    I4 = np.sum(h4_vals) * dx

    # Vanishing integral: c0*I0 + c4*I4 = 0 => c0 = -c4*I4/I0
    # Choose c4 = I0, c0 = -I4 (unnormalized)
    c0 = -I4
    c4 = I0
    norm = np.sqrt(c0**2 + c4**2)
    c0 /= norm
    c4 /= norm

    return c0 * hermite_h0(x) + c4 * hermite_h4(x)


def compute_QW_eigenvector(N, L_val, dps=50):
    """Compute the smallest eigenvalue eigenvector of QW at given precision."""
    mp.dps = dps
    L = mpf(L_val)
    dim = 2 * N + 1

    # b_n from exact formulas
    b = {}
    L2 = L * L; p2 = 16 * pi * pi; pf = 32 * L * sinh(L / 4)**2
    eL = exp(L)
    primes_list = [p for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43] if p <= float(eL)]

    for n in range(-N, N + 1):
        if n == 0:
            b[0] = mpf(0)
            continue

        # WR alpha
        z = exp(-2*L); a_arg = pi*mpc(0,n)/L + mpf(1)/4
        h = hyp2f1(1, a_arg, a_arg+1, z)
        f1 = exp(-L/2) * (2*L/(L+4*pi*mpc(0,n)) * h).imag
        d = digamma(a_arg).imag / 2
        alpha_WR = (f1 + d) / pi

        # W02(n,0)
        W02_n0 = pf * L2 / (L2 * (L2 + p2*n*n))
        WR_n0 = -alpha_WR / n

        # Wp(n,0)
        Wp_n0 = mpf(0)
        for p_ in primes_list:
            lp = log(mpf(p_)); pk = mpf(p_)
            while pk <= eL:
                Wp_n0 += lp * pk**(-mpf(1)/2) * cos(2*pi*n*log(pk)/L)
                pk *= p_

        b[n] = n * (W02_n0 - WR_n0 - Wp_n0)

    # a_n from eq (4.4)
    a = {}
    for n_val in range(N + 1):
        omega_0 = mpf(2)
        w_const = (omega_0/2) * (euler + log(4*pi*(eL-1)/(eL+1)))
        n_quad = 5000; dx = L/n_quad; integral = mpf(0)
        for k in range(n_quad):
            x = dx*(k+mpf(1)/2)
            omega_x = 2*(1-x/L)*cos(2*pi*n_val*x/L)
            numer = exp(x/2)*omega_x - omega_0
            denom = exp(x)-exp(-x)
            if abs(denom) > mpf(10)**(-dps+10): integral += numer/denom
        integral *= dx
        WR_nn = w_const + integral
        W02_nn = pf*(L2-p2*n_val*n_val)/((L2+p2*n_val*n_val)**2)
        Wp_nn = mpf(0)
        for p_ in primes_list:
            lp = log(mpf(p_)); pk = mpf(p_)
            while pk <= eL: Wp_nn += lp*pk**(-mpf(1)/2); pk *= p_
        a[n_val] = W02_nn - WR_nn - Wp_nn
        a[-n_val] = a[n_val]

    # Build tau matrix
    tau = mpmatrix(dim, dim)
    for i in range(dim):
        ni = i - N
        for j in range(dim):
            nj = j - N
            if ni == nj: tau[i,j] = a[ni]
            else: tau[i,j] = (b[ni]-b[nj])/(ni-nj)

    # Eigenvalues
    E, ER, EL_mat = eig(tau, left=True, right=True)

    # Find smallest even eigenvector
    best_even = None
    best_eval = float('inf')
    for i in range(dim):
        ev = float(E[i].real)
        xi = np.array([float(ER[j,i].real) for j in range(dim)])
        es = sum(abs(xi[N+k]-xi[N-k]) for k in range(1,N+1))
        os = sum(abs(xi[N+k]+xi[N-k]) for k in range(1,N+1))
        if es < os and ev < best_eval:
            best_eval = ev
            best_even = xi

    if best_even is None:
        # Fallback: use smallest eigenvector regardless of parity
        min_idx = min(range(dim), key=lambda i: float(E[i].real))
        best_even = np.array([float(ER[j,min_idx].real) for j in range(dim)])
        best_eval = float(E[min_idx].real)

    return best_even, best_eval, float(L)


def eigenvector_to_function(xi, N, L, x_grid):
    """
    Convert eigenvector xi (coefficients of V_n) to a function on [lambda^{-1}, lambda].

    xi(u) = sum_{n=-N}^{N} xi_n * V_n(u)
    V_n(u) = L^{-1/2} * exp(2*pi*i*n*log(lambda*u)/L)

    For real xi with even symmetry, this is a real function.
    """
    lam = np.exp(L / 2)
    result = np.zeros_like(x_grid)
    for idx in range(2 * N + 1):
        n = idx - N
        # V_n(u) = L^{-1/2} * exp(2*pi*i*n*log(lambda*u)/L)
        phase = 2 * np.pi * n * np.log(lam * x_grid) / L
        result += xi[idx] * np.cos(phase) / np.sqrt(L)  # real part for real functions
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Session 21: Prolate Overlap Computation")
    print("=" * 70)

    # Test with lambda^2 = 14 (lambda = sqrt(14)), N = 30 (small for speed)
    lam_sq_values = [14]
    N = 30

    for lam_sq in lam_sq_values:
        lam = np.sqrt(lam_sq)
        L_val = np.log(lam_sq)
        print(f"\n--- lambda = sqrt({lam_sq}) = {lam:.4f}, L = {L_val:.4f}, N = {N} ---")

        # Compute QW eigenvector at 200dp
        print("  Computing QW eigenvector at 200dp...")
        mp.dps = 200
        xi, eval_min, L_computed = compute_QW_eigenvector(N, L_val, dps=200)
        print(f"  Min eigenvalue: {eval_min:+.6f}")

        # Evaluate xi as a function on [1/lambda, lambda]
        n_pts = 1000
        u_grid = np.linspace(1/lam + 1e-6, lam - 1e-6, n_pts)
        xi_func = eigenvector_to_function(xi, N, L_val, u_grid)

        # Compute prolate combination k_lambda
        # Map u -> x = log(u) for the prolate functions (which live in log coordinates)
        x_grid = np.log(u_grid)  # x in [-L/2, L/2]
        k_func = h_lambda_combination(x_grid, lam)

        # Normalize both functions
        du = u_grid[1] - u_grid[0]
        xi_norm = np.sqrt(np.sum(xi_func**2 * du / u_grid))  # L^2 with d*u measure
        k_norm = np.sqrt(np.sum(k_func**2 * du / u_grid))

        if xi_norm > 1e-10: xi_func_n = xi_func / xi_norm
        else: xi_func_n = xi_func
        if k_norm > 1e-10: k_func_n = k_func / k_norm
        else: k_func_n = k_func

        # Overlap
        overlap = np.sum(xi_func_n * k_func_n * du / u_grid)
        print(f"  Overlap <xi|k> = {overlap:+.8f}")
        print(f"  |overlap| = {abs(overlap):.8f}")

        # L^2 distance
        diff = xi_func_n - overlap * k_func_n  # project out the k component
        residual = np.sqrt(np.sum(diff**2 * du / u_grid))
        print(f"  ||xi - <xi|k>*k|| = {residual:.8f}")

        # Compare with just h_0 (Gaussian)
        h0_func = hermite_h0(x_grid)
        h0_norm = np.sqrt(np.sum(h0_func**2 * du / u_grid))
        if h0_norm > 1e-10: h0_func_n = h0_func / h0_norm
        else: h0_func_n = h0_func
        overlap_h0 = np.sum(xi_func_n * h0_func_n * du / u_grid)
        print(f"  Overlap <xi|h_0> = {overlap_h0:+.8f}")

    print("\n" + "=" * 70)
